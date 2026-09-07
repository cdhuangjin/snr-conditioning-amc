"""Deterministic, persisted, and self-validating dataset splits for AMC V2."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import unicodedata
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from .provenance import file_sha256


SCHEMA_VERSION = 1
RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}
ROUNDING = "floor_train_floor_test_validation_remainder"
ALGORITHM = "numpy-randomstate-cell-choice-train-test-remainder"
ALGORITHM_VERSION = "1"
_NPZ_FIELDS = {
    "train_idx", "val_idx", "test_idx", "cell_starts", "cell_counts",
    "cell_classes", "cell_snrs",
}


class SplitValidationError(ValueError):
    """Raised when persisted split evidence is malformed or has drifted."""


@dataclass(frozen=True, slots=True)
class Cell:
    class_label: str
    snr: int | float
    count: int
    start: int


@dataclass(slots=True)
class FixedSplit:
    dataset_id: str
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    cells: tuple[Cell, ...]
    ratios: dict[str, float] = field(default_factory=lambda: dict(RATIOS))
    rounding: str = ROUNDING
    split_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return int(sum(cell.count for cell in self.cells))

    @property
    def sample_ids(self) -> "SampleIdView":
        return SampleIdView(self.dataset_id, self.cells)


class SampleIdView(Sequence[str]):
    """Lazy canonical IDs backed only by compact cell descriptors."""

    def __init__(self, dataset_id: str, cells: Sequence[Cell]) -> None:
        self.dataset_id = dataset_id
        self.cells = tuple(cells)
        self._starts = tuple(cell.start for cell in self.cells)
        self._length = sum(cell.count for cell in self.cells)

    def __len__(self) -> int:
        return self._length

    def _one(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("sample ID index must be an integer")
        index = int(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        cell_index = bisect_right(self._starts, index) - 1
        cell = self.cells[cell_index]
        return sample_id(self.dataset_id, cell.class_label, cell.snr, index - cell.start)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return np.asarray([self._one(i) for i in range(*index.indices(self._length))])
        if isinstance(index, (list, tuple, np.ndarray)):
            values = np.asarray(index)
            if values.dtype.kind not in "iu" or values.ndim != 1:
                raise TypeError("sample ID indices must be a one-dimensional integer array")
            return np.asarray([self._one(int(i)) for i in values])
        return self._one(index)

    def __iter__(self) -> Iterator[str]:
        for cell in self.cells:
            for within in range(cell.count):
                yield sample_id(self.dataset_id, cell.class_label, cell.snr, within)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        array = np.asarray(list(self), dtype=dtype)
        return array.copy() if copy else array

    def tolist(self) -> list[str]:
        return list(self)


def _canonical_label(value: object) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("class labels must be ASCII bytes or strings") from exc
    if not isinstance(value, str) or not value or "|" in value:
        raise ValueError("class labels must be non-empty strings without '|'")
    return unicodedata.normalize("NFC", value)


def _canonical_snr(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("SNR must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("SNR must be finite")
    return int(number) if number.is_integer() else number


def format_snr(value: int | float) -> str:
    """Format SNR with an explicit sign and without insignificant zeros."""
    return f"{float(value):+.17g}"


def sample_id(dataset_id: str, class_label: str, snr: int | float, within_cell_index: int) -> str:
    if not dataset_id or "|" in dataset_id:
        raise ValueError("dataset_id must be non-empty and must not contain '|'")
    if (
        isinstance(within_cell_index, bool)
        or not isinstance(within_cell_index, (int, np.integer))
        or int(within_cell_index) < 0
    ):
        raise ValueError("within_cell_index must be a nonnegative integer")
    return f"{dataset_id}|{_canonical_label(class_label)}|{format_snr(_canonical_snr(snr))}|{within_cell_index}"


def _normalize_cells(
    cells: Iterable[Sequence[object]], samples_per_cell: int | None
) -> tuple[Cell, ...]:
    normalized: list[Cell] = []
    start = 0
    seen: set[tuple[str, int | float]] = set()
    for raw in cells:
        if len(raw) == 2:
            if samples_per_cell is None:
                raise ValueError("samples_per_cell is required for two-field cells")
            label, snr = raw
            count = samples_per_cell
        elif len(raw) == 3:
            label, snr, count = raw
        else:
            raise ValueError("each cell must be (class, SNR) or (class, SNR, count)")
        canonical_label = _canonical_label(label)
        canonical_snr = _canonical_snr(snr)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) <= 0:
            raise ValueError("cell counts must be positive integers")
        key = (canonical_label, canonical_snr)
        if key in seen:
            raise ValueError(f"duplicate cell {key!r}")
        seen.add(key)
        normalized.append(Cell(canonical_label, canonical_snr, int(count), start))
        start += int(count)
    if not normalized:
        raise ValueError("at least one cell is required")
    return tuple(normalized)


def make_stratified_split(
    cells: Iterable[Sequence[object]],
    *,
    samples_per_cell: int | None = None,
    seed: int = 2022,
    dataset_id: str = "RML2016.10a",
) -> FixedSplit:
    """Create one 60/20/20 split using historical NumPy RandomState semantics.

    For uneven cells, train and test counts are independently floored and the
    validation split receives the remainder. The random stream chooses train
    first and test second from the legacy integer-set remainder. This exactly
    matches the historical protocol on the recorded Python/NumPy runtime.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    normalized = _normalize_cells(cells, samples_per_cell)
    rng = np.random.RandomState(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cell in normalized:
        population = np.arange(cell.start, cell.start + cell.count, dtype=np.int64)
        train_count = int(math.floor(cell.count * RATIOS["train"]))
        test_count = int(math.floor(cell.count * RATIOS["test"]))
        train = np.asarray(rng.choice(population, size=train_count, replace=False), dtype=np.int64)
        remainder = np.asarray(
            list(set(population.tolist()) - set(train.tolist())), dtype=np.int64
        )
        test = np.asarray(rng.choice(remainder, size=test_count, replace=False), dtype=np.int64)
        validation = np.asarray(
            list(set(remainder.tolist()) - set(test.tolist())), dtype=np.int64
        )
        train_parts.append(train)
        val_parts.append(validation)
        test_parts.append(test)
    return FixedSplit(
        dataset_id=dataset_id,
        seed=int(seed),
        train_idx=np.concatenate(train_parts),
        val_idx=np.concatenate(val_parts),
        test_idx=np.concatenate(test_parts),
        cells=normalized,
    )


def _per_cell_counts(split: FixedSplit) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cell in split.cells:
        start, stop = cell.start, cell.start + cell.count
        count = lambda values: int(np.count_nonzero((values >= start) & (values < stop)))
        records.append({
            "class": cell.class_label,
            "snr": cell.snr,
            "start": start,
            "all": cell.count,
            "train": count(split.train_idx),
            "validation": count(split.val_idx),
            "test": count(split.test_idx),
        })
    return records


def _total_counts(split: FixedSplit) -> dict[str, int]:
    return {
        "all": split.total_count,
        "train": int(len(split.train_idx)),
        "validation": int(len(split.val_idx)),
        "test": int(len(split.test_idx)),
    }


def _semantic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    storage_fields = {"npz_sha256", "npz_path", "storage_generation", "split_hash"}
    return {key: value for key, value in metadata.items() if key not in storage_fields}


def _compute_split_hash(split: FixedSplit, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    canonical = json.dumps(
        _semantic_metadata(metadata), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    digest.update(len(canonical).to_bytes(8, "little"))
    digest.update(canonical)
    for name, values in (
        ("train_idx", split.train_idx), ("val_idx", split.val_idx),
        ("test_idx", split.test_idx),
    ):
        digest.update(name.encode("ascii"))
        array = np.ascontiguousarray(values, dtype="<i8")
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.tobytes())
    for cell in split.cells:
        for within in range(cell.count):
            encoded = sample_id(split.dataset_id, cell.class_label, cell.snr, within).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"dataset path {path} is outside repository root {root}") from exc
    return relative.as_posix()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        deadline = time.monotonic() + 2.0
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Concurrent Windows publishers can briefly hold the metadata
                # directory entry while replacing the same commit pointer.
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a byte-stable, pickle-free NPZ for content-addressed publication."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def save_split(
    split: FixedSplit,
    npz_path: str | Path,
    metadata_path: str | Path,
    *,
    data_path: str | Path,
    repository_root: str | Path,
    _after_npz_publish: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Publish an immutable NPZ blob, then atomically advance public metadata."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    data_path = Path(data_path)
    root = Path(repository_root)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    dataset_path = _relative_posix(data_path, root)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": split.dataset_id,
        "dataset_path": dataset_path,
        "data_sha256": file_sha256(data_path),
        "seed": split.seed,
        "ratios": dict(split.ratios),
        "rounding": split.rounding,
        "per_cell_counts": _per_cell_counts(split),
        "total_counts": _total_counts(split),
        "generation": {
            "algorithm": ALGORITHM,
            "version": ALGORITHM_VERSION,
            "random_api": "numpy.random.RandomState",
            "historical_compatibility": "exact legacy integer-set remainder order; validated on the recorded runtime",
        },
    }
    split.split_hash = _compute_split_hash(split, metadata)
    metadata["split_hash"] = split.split_hash
    arrays = {
        "train_idx": np.asarray(split.train_idx, dtype=np.int64),
        "val_idx": np.asarray(split.val_idx, dtype=np.int64),
        "test_idx": np.asarray(split.test_idx, dtype=np.int64),
        "cell_starts": np.asarray([cell.start for cell in split.cells], dtype=np.int64),
        "cell_counts": np.asarray([cell.count for cell in split.cells], dtype=np.int64),
        "cell_classes": np.asarray([cell.class_label for cell in split.cells]),
        "cell_snrs": np.asarray([cell.snr for cell in split.cells], dtype=np.float64),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    published: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=npz_path.parent, prefix=f".{npz_path.name}.", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        _write_deterministic_npz(temporary, arrays)
        storage_hash = file_sha256(temporary)
        published = npz_path.with_name(f"{npz_path.stem}.{storage_hash}.npz")
        if published.exists():
            if file_sha256(published) != storage_hash:
                raise SplitValidationError(f"content-addressed NPZ collision at {published}")
        else:
            try:
                os.link(temporary, published)
            except FileExistsError:
                if file_sha256(published) != storage_hash:
                    raise SplitValidationError(f"content-addressed NPZ collision at {published}")
        metadata["npz_sha256"] = storage_hash
        metadata["npz_path"] = _relative_posix(published, root)
        metadata["storage_generation"] = {
            "format": "deterministic-npz-content-addressed",
            "version": 1,
            "sha256": storage_hash,
        }
        if _after_npz_publish is not None:
            _after_npz_publish(published, metadata)
        _write_json_atomic(metadata_path, metadata)
    except Exception:
        # A published content-addressed blob is immutable evidence.  Metadata is
        # the commit pointer, so an interrupted publication may leave an orphan
        # blob but must never remove a blob another publisher can reference.
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    split.metadata = dict(metadata)
    return metadata


def _metadata_error(field: str, message: str) -> SplitValidationError:
    return SplitValidationError(f"metadata {field}: {message}")


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise _metadata_error("$", "must be an object")
    required = {
        "schema_version", "dataset_id", "dataset_path", "data_sha256", "seed",
        "ratios", "rounding", "per_cell_counts", "total_counts", "generation",
        "npz_sha256", "split_hash",
    }
    missing = required - metadata.keys()
    if missing:
        field = sorted(missing)[0]
        raise _metadata_error(field, "missing required field")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise _metadata_error("schema_version", f"expected {SCHEMA_VERSION}")
    if not isinstance(metadata["dataset_id"], str) or not metadata["dataset_id"]:
        raise _metadata_error("dataset_id", "must be a non-empty string")
    dataset_path = metadata["dataset_path"]
    if not isinstance(dataset_path, str) or not dataset_path or "\\" in dataset_path:
        raise _metadata_error("dataset_path", "must be a repository-relative POSIX path")
    if PurePosixPath(dataset_path).is_absolute() or PureWindowsPath(dataset_path).is_absolute() or ".." in PurePosixPath(dataset_path).parts:
        raise _metadata_error("dataset_path", "must be a repository-relative POSIX path")
    for field_name in ("data_sha256", "npz_sha256", "split_hash"):
        value = metadata[field_name]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise _metadata_error(field_name, "must be a lowercase SHA-256 digest")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        raise _metadata_error("seed", "must be an integer")
    if metadata["ratios"] != RATIOS:
        raise _metadata_error("ratios", f"expected {RATIOS}")
    if metadata["rounding"] != ROUNDING:
        raise _metadata_error("rounding", f"expected {ROUNDING}")
    if not isinstance(metadata["per_cell_counts"], list) or not metadata["per_cell_counts"]:
        raise _metadata_error("per_cell_counts", "must be a non-empty list")
    if not isinstance(metadata["total_counts"], dict):
        raise _metadata_error("total_counts", "must be an object")
    if not isinstance(metadata["generation"], dict):
        raise _metadata_error("generation", "must be an object")
    if "npz_path" in metadata:
        npz_path = metadata["npz_path"]
        if not isinstance(npz_path, str) or not npz_path or "\\" in npz_path:
            raise _metadata_error("npz_path", "must be a repository-relative POSIX path")
        pure_npz = PurePosixPath(npz_path)
        if pure_npz.is_absolute() or PureWindowsPath(npz_path).is_absolute() or ".." in pure_npz.parts:
            raise _metadata_error("npz_path", "must be a repository-relative POSIX path")
    if "storage_generation" in metadata:
        storage = metadata["storage_generation"]
        if not isinstance(storage, dict) or storage.get("sha256") != metadata["npz_sha256"]:
            raise _metadata_error("storage_generation", "must bind the NPZ SHA-256")
    return metadata


def _resolve_data_path(metadata_path: Path, stored_path: str) -> Path:
    for ancestor in (metadata_path.parent, *metadata_path.parents):
        candidate = ancestor / PurePosixPath(stored_path)
        if candidate.is_file():
            return candidate
    raise SplitValidationError(f"dataset file for portable dataset_path {stored_path!r} was not found")


def _load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_FIELDS:
                raise SplitValidationError(f"NPZ fields mismatch: expected {sorted(_NPZ_FIELDS)}")
            return {name: archive[name] for name in archive.files}
    except SplitValidationError:
        raise
    except Exception as exc:
        raise SplitValidationError(f"cannot load NPZ {npz_path}: {exc}") from exc


def _validate_indices(split: FixedSplit, metadata: dict[str, Any]) -> None:
    total = split.total_count
    arrays = (split.train_idx, split.val_idx, split.test_idx)
    for name, values in zip(("train_idx", "val_idx", "test_idx"), arrays):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise SplitValidationError(f"{name} must be a one-dimensional integer array")
        if len(values) and (int(values.min()) < 0 or int(values.max()) >= total):
            raise SplitValidationError(f"{name} contains an out-of-range index")
    if sum(len(values) for values in arrays) != total:
        raise SplitValidationError("split indices do not provide full coverage")
    coverage = np.zeros(total, dtype=np.uint8)
    for values in arrays:
        np.add.at(coverage, values.astype(np.intp, copy=False), 1)
    if np.any(coverage > 1):
        raise SplitValidationError("split indices are not disjoint or contain duplicates")
    if not np.all(coverage == 1):
        raise SplitValidationError("split indices do not provide exact full coverage")
    expected_cells = _per_cell_counts(split)
    if metadata["per_cell_counts"] != expected_cells:
        raise SplitValidationError("per-cell ratios/counts do not match persisted indices")
    if metadata["total_counts"] != _total_counts(split):
        raise SplitValidationError("total_counts do not match persisted indices")
    for record in expected_cells:
        count = int(record["all"])
        expected = (
            int(math.floor(count * RATIOS["train"])),
            count - int(math.floor(count * RATIOS["train"])) - int(math.floor(count * RATIOS["test"])),
            int(math.floor(count * RATIOS["test"])),
        )
        actual = (record["train"], record["validation"], record["test"])
        if actual != expected:
            raise SplitValidationError(f"per-cell ratios are invalid for {record['class']}@{record['snr']}")


def load_split(
    metadata_path: str | Path,
    *,
    npz_path: str | Path | None = None,
    data_path: str | Path | None = None,
) -> FixedSplit:
    """Load a persisted split and reject any provenance, data, or index drift."""
    metadata_path = Path(metadata_path)
    try:
        metadata = _validate_metadata(json.loads(metadata_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise SplitValidationError(f"metadata malformed JSON at line {exc.lineno}, column {exc.colno}") from exc
    if "npz_path" in metadata:
        stored_npz = _resolve_data_path(metadata_path, metadata["npz_path"])
        if npz_path is not None and Path(npz_path).resolve() != stored_npz.resolve():
            raise SplitValidationError("explicit NPZ path does not match metadata npz_path")
        npz_path = stored_npz
    else:
        npz_path = Path(npz_path) if npz_path is not None else metadata_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise SplitValidationError(f"NPZ file not found: {npz_path}")
    actual_npz_hash = file_sha256(npz_path)
    if actual_npz_hash != metadata["npz_sha256"]:
        raise SplitValidationError(f"NPZ SHA-256 mismatch: expected {metadata['npz_sha256']}, got {actual_npz_hash}")
    resolved_data = Path(data_path) if data_path is not None else _resolve_data_path(metadata_path, metadata["dataset_path"])
    if not resolved_data.is_file():
        raise SplitValidationError(f"dataset file not found: {resolved_data}")
    actual_data_hash = file_sha256(resolved_data)
    if actual_data_hash != metadata["data_sha256"]:
        raise SplitValidationError(f"data SHA-256 mismatch: expected {metadata['data_sha256']}, got {actual_data_hash}")
    arrays = _load_arrays(npz_path)
    lengths = {len(arrays[name]) for name in ("cell_starts", "cell_counts", "cell_classes", "cell_snrs")}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise SplitValidationError("compact sample identity arrays have inconsistent lengths")
    cells: list[Cell] = []
    for label, snr, count, start in zip(
        arrays["cell_classes"].tolist(), arrays["cell_snrs"].tolist(),
        arrays["cell_counts"].tolist(), arrays["cell_starts"].tolist(),
    ):
        cells.append(Cell(_canonical_label(label), _canonical_snr(snr), int(count), int(start)))
    expected_start = 0
    for cell in cells:
        if cell.count <= 0 or cell.start != expected_start:
            raise SplitValidationError("compact sample identity cell offsets/counts are invalid")
        expected_start += cell.count
    split = FixedSplit(
        dataset_id=metadata["dataset_id"], seed=metadata["seed"],
        train_idx=np.asarray(arrays["train_idx"]), val_idx=np.asarray(arrays["val_idx"]),
        test_idx=np.asarray(arrays["test_idx"]), cells=tuple(cells),
        ratios=dict(metadata["ratios"]), rounding=metadata["rounding"],
        split_hash=metadata["split_hash"], metadata=dict(metadata),
    )
    identities = {(cell.class_label, cell.snr) for cell in split.cells}
    if len(identities) != len(split.cells):
        raise SplitValidationError("sample identity cell descriptors are not unique")
    _validate_indices(split, metadata)
    computed = _compute_split_hash(split, metadata)
    if computed != metadata["split_hash"]:
        raise SplitValidationError(f"split hash mismatch: expected {metadata['split_hash']}, got {computed}")
    return split


def cells_from_rml2016_pickle(data_path: str | Path) -> list[tuple[str, int | float, int]]:
    """Read actual RML2016 cell sizes in the legacy loader's sorted class/SNR order."""
    import pickle

    with Path(data_path).open("rb") as handle:
        dataset = pickle.load(handle, encoding="bytes")
    if not isinstance(dataset, dict) or not dataset:
        raise SplitValidationError("RML2016 pickle must contain a non-empty cell mapping")
    classes = sorted({key[0] for key in dataset})
    snrs = sorted({key[1] for key in dataset})
    cells: list[tuple[str, int | float, int]] = []
    for class_label in classes:
        for snr in snrs:
            key = (class_label, snr)
            if key not in dataset:
                raise SplitValidationError(f"dataset is missing cell {key!r}")
            values = dataset[key]
            if not hasattr(values, "shape") or not values.shape:
                raise SplitValidationError(f"dataset cell {key!r} has no sample dimension")
            cells.append((_canonical_label(class_label), _canonical_snr(snr), int(values.shape[0])))
    return cells

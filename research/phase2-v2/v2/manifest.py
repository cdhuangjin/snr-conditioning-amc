from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, get_args

from .contracts import ArtifactRecord, ArtifactStatus


_KEY_FIELDS = ("experiment", "dataset", "split_hash", "model", "conditioner", "seed")
_STRING_FIELDS = (
    "experiment",
    "dataset",
    "split_hash",
    "model",
    "conditioner",
    "checkpoint",
    "result_json",
    "notes",
)
_REQUIRED_FIELDS = frozenset((*_STRING_FIELDS, "seed", "figure_paths", "status"))
_SUPPORTED_STATUSES = frozenset(get_args(ArtifactStatus))


class ManifestValidationError(ValueError):
    """Raised when a manifest does not conform to schema version 1."""


@contextmanager
def _cross_process_lock(target: Path) -> Iterator[None]:
    """Serialize updates to a target path across Windows and POSIX processes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0)

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validation_error(path: Path, location: str, message: str) -> None:
    raise ManifestValidationError(f"{path}: {location}: {message}")


def _validate_manifest(manifest: Any, path: Path) -> None:
    if not isinstance(manifest, dict):
        _validation_error(path, "$", "manifest must be an object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        _validation_error(path, "schema_version", "unsupported schema; expected 1")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        _validation_error(path, "experiments", "must be a list")

    seen: dict[tuple[object, ...], int] = {}
    for index, record in enumerate(experiments):
        location = f"experiments[{index}]"
        if not isinstance(record, dict):
            _validation_error(path, location, "must be an object")

        missing = _REQUIRED_FIELDS - record.keys()
        if missing:
            field = sorted(missing)[0]
            _validation_error(path, f"{location}.{field}", "missing required field")
        extra = record.keys() - _REQUIRED_FIELDS
        if extra:
            field = sorted(extra)[0]
            _validation_error(path, f"{location}.{field}", "unexpected field")

        for field in _STRING_FIELDS:
            if not isinstance(record[field], str):
                _validation_error(path, f"{location}.{field}", "must be a string")
        if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
            _validation_error(path, f"{location}.seed", "must be an integer")
        if not isinstance(record["figure_paths"], list):
            _validation_error(path, f"{location}.figure_paths", "must be a list")
        for figure_index, figure_path in enumerate(record["figure_paths"]):
            if not isinstance(figure_path, str):
                _validation_error(
                    path,
                    f"{location}.figure_paths[{figure_index}]",
                    "must be a string",
                )
        if (
            not isinstance(record["status"], str)
            or record["status"] not in _SUPPORTED_STATUSES
        ):
            _validation_error(
                path,
                f"{location}.status",
                f"must be one of {sorted(_SUPPORTED_STATUSES)}",
            )

        key = tuple(record[field] for field in _KEY_FIELDS)
        if key in seen:
            _validation_error(
                path,
                location,
                f"duplicates experiments[{seen[key]}] unique key",
            )
        seen[key] = index


class ManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "experiments": []}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"{self.path}: malformed JSON at line {exc.lineno}, column {exc.colno}"
            ) from exc
        _validate_manifest(manifest, self.path)
        return manifest

    def upsert(self, record: ArtifactRecord) -> None:
        with _cross_process_lock(self.path):
            manifest = self.load()
            serialized = record.to_dict()
            key = tuple(serialized[field] for field in _KEY_FIELDS)
            experiments = manifest["experiments"]

            for index, existing in enumerate(experiments):
                existing_key = tuple(existing[field] for field in _KEY_FIELDS)
                if existing_key == key:
                    experiments[index] = serialized
                    break
            else:
                experiments.append(serialized)

            _validate_manifest(manifest, self.path)
            self._write_atomic(manifest)

    def _write_atomic(self, manifest: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

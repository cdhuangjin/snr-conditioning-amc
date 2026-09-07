#!/usr/bin/env python3
"""Run the Phase 1 fixed-split reproduction gate for AWN and AWNSNR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from v2.contracts import ArtifactRecord
from v2.manifest import ManifestStore
from v2.progress import update_phase_progress
from v2.provenance import deterministic_json_hash, environment_fingerprint, file_sha256
from v2.splits import FixedSplit, SplitValidationError, format_snr, load_split


RUNNER_VERSION = "phase1-reproduce/4"
REQUIRED_RUNS = ("plain", "conditioned")
LEGACY_PHASE1_CODE_HASH = "a4f0c41213d247121be901b2990bbd5efc7b070ecd2e827c3087cadb47aee21b"
LEGACY_PHASE1_RUNNER_SHA256 = "6c313ccf85fd36f88afb9f8505df0590a11b316a4cb708565df2e7a96d5cd9c6"
REQUIRED_CONDITIONED_MODEL_SHA256 = "802079e1d6a594a73e358d7c8c5e2800a0a428446eab9efe79eaea543ac96b2a"
BASELINE_SPEC = {
    "schema_version": 1,
    "protocol_version": "phase1-rml2016.10a/4",
    "dataset": {
        "id": "RML2016.10a",
        "path": "data/RML2016.10a_dict.pkl",
        "sha256": "b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c",
    },
    "historical_targets": {"plain": 0.621, "conditioned": 0.671},
    "tolerance_pp": 0.5,
    "seed": 2022,
    "device": "cpu",
    "training": {
        "optimizer": "Adam", "lr": 0.001, "batch_size": 128,
        "patience": 10, "max_epochs": 100,
        "milestone_step": 3, "gamma": 0.5,
    },
    "evaluation_batch_size": 64,
    "architecture": {
        "num_classes": 11, "num_levels": 1, "in_channels": 64,
        "kernel_size": 3, "latent_dim": 320,
        "regu_details": 0.01, "regu_approx": 0.01,
        "num_snr_bins": 20, "snr_embedding_dim": 8,
    },
    "models": {
        "plain": {"name": "AWN", "conditioner": "none", "source": "models/model.py"},
        "conditioned": {
            "name": "AWNSNR", "conditioner": "true_snr_embedding",
            "source": "models/model_snr.py",
        },
    },
    "memory": {
        "max_dataset_and_indices_bytes": 300_000_000,
        "measurement": "steady-state signals+labels+snrs+split-index bytes; excludes transient pickle and allocator overhead",
        "out_of_scope": "RML2018 requires the Phase11 subset/memmap path; it is not supported by Phase1.",
    },
    "training_protocol": {
        "checkpoint": "validation_accuracy_greater_or_equal_latest_tie",
        "early_stopping": "validation_loss_strict_regression_equality_resets_delta_zero",
        "lr_decay": "after_validation_when_loss_counter_nonzero_multiple_of_milestone",
        "batch_statistics": "unweighted_mean_of_batch_metrics",
        "data_order": "global_torch_rng_after_fix_seed_and_model_initialization_with_train_and_validation_shuffle",
    },
    "paths": {
        "config": "configs/v2/reproduction.yaml",
        "split_metadata": "splits/v2/RML2016.10a_seed2022.json",
        "output_root": "results/v2/reproduction",
        "gate_state": "results/v2/reproduction/gate_state.json",
        "manifest": "manifest.json",
        "progress": "reports/v2_progress.md",
        "failure_report": "reports/reproduction_failure.md",
    },
}
BASELINE_SPEC_DIGEST = deterministic_json_hash(BASELINE_SPEC)
SOURCE_DEPENDENCIES = (
    "scripts/v2/phase1_reproduce.py",
    "v2/splits.py", "v2/contracts.py", "v2/manifest.py", "v2/progress.py", "v2/provenance.py",
    "models/model.py", "models/model_snr.py", "models/lifting.py",
    "util/utils.py",
)
REFERENCE_SOURCE_DEPENDENCIES = (
    "data_loader/data_loader.py", "util/training.py", "util/early_stop.py", "util/logger.py",
)


class ReproductionValidationError(ValueError):
    """Raised when the reproduction configuration is unsafe or incomplete."""


class PhaseExecutionLock:
    """OS-released advisory lock for one Phase 1 publisher at a time."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self.owner = f"pid={os.getpid()};started={_utc_now()}"
        self.acquired = False
        self._handle: Any | None = None

    @staticmethod
    def _try_lock(handle: Any) -> bool:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    @staticmethod
    def _unlock(handle: Any) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while not self._try_lock(handle):
                if time.monotonic() >= deadline:
                    raise ReproductionValidationError(
                        f"Phase 1 execution lock timeout after {self.timeout_seconds:g}s: {self.path}"
                    )
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            handle.seek(0)
            handle.truncate()
            handle.write((self.owner + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
            self.acquired = True
        except Exception:
            handle.close()
            raise

    def release(self) -> None:
        if not self.acquired:
            return
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                self._unlock(handle)
        finally:
            if handle is not None:
                handle.close()
            self.acquired = False

    def __enter__(self) -> "PhaseExecutionLock":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json_new(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a new JSON path without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReproductionValidationError(f"immutable gate state already exists: {path}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish_gate_state(
    output_root: str | Path, current_path: str | Path, attempt: dict[str, Any],
    *, repository_root: str | Path | None = None,
) -> Path:
    """Publish immutable per-attempt evidence, then atomically advance current state."""
    output_root = Path(output_root)
    current_path = Path(current_path)
    attempt_id = attempt.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in attempt_id
    ):
        raise ReproductionValidationError("attempt_id is unsafe for immutable gate publication")
    immutable_path = output_root / "gate_states" / f"{attempt_id}.json"
    if immutable_path.exists():
        try:
            existing = json.loads(immutable_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproductionValidationError(f"immutable gate state cannot be verified: {exc}") from exc
        if existing != attempt:
            raise ReproductionValidationError(f"immutable gate state already exists with different evidence: {immutable_path}")
    else:
        _atomic_json_new(immutable_path, attempt)
    root = Path(repository_root) if repository_root is not None else output_root.parents[2]
    pointer = dict(attempt)
    pointer["attempt_state"] = _relative(root, immutable_path)
    pointer["attempt_state_sha256"] = file_sha256(immutable_path)
    _atomic_json(current_path, pointer)
    return immutable_path


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _canonical_payload_hash(value: dict[str, Any], *, omit: set[str] | None = None) -> str:
    excluded = omit or set()
    return deterministic_json_hash(
        {key: item for key, item in value.items() if key not in excluded}
    )


def _repo_path(root: Path, value: str, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReproductionValidationError(f"{field} must be a repository-relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in pure.parts:
        raise ReproductionValidationError(f"{field} must be a repository-relative POSIX path")
    lexical = root / pure
    _reject_reparse_components(root, lexical, field)
    path = lexical.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ReproductionValidationError(f"{field} escapes repository root") from exc
    return path


def _reject_reparse_components(root: Path, path: Path, field: str) -> None:
    """Reject symlinks, junctions, and Windows reparse points before resolution."""
    root = root.resolve()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise ReproductionValidationError(f"{field} escapes repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        is_junction = bool(getattr(current, "is_junction", lambda: False)())
        try:
            attributes = getattr(os.lstat(current), "st_file_attributes", 0)
        except OSError as exc:
            raise ReproductionValidationError(f"{field} cannot be safely inspected: {exc}") from exc
        if current.is_symlink() or is_junction or attributes & 0x400:
            raise ReproductionValidationError(f"{field} traverses a symlink, junction, or reparse point")


def _expect_exact_path(config_value: Any, expected: str, field: str) -> None:
    if config_value != expected:
        raise ReproductionValidationError(f"{field} must be exactly {expected}")


def _validate_finite_numbers(value: Any, location: str = "configuration") -> None:
    if isinstance(value, bool):
        raise ReproductionValidationError(f"{location} numeric values must not be booleans")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ReproductionValidationError(f"{location} numeric values must be finite")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{location}[{index}]")


def _validate_locked_mapping(actual: Any, expected: dict[str, Any], location: str) -> None:
    if not isinstance(actual, dict):
        raise ReproductionValidationError(f"{location} must be an object")
    if set(actual) != set(expected):
        raise ReproductionValidationError(f"{location} fields must be exactly {sorted(expected)}")
    for field, expected_value in expected.items():
        value = actual[field]
        field_name = f"{location}.{field}"
        if isinstance(expected_value, dict):
            _validate_locked_mapping(value, expected_value, field_name)
            continue
        if type(value) is not type(expected_value):
            raise ReproductionValidationError(
                f"{field_name} must have exact type {type(expected_value).__name__}"
            )
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ReproductionValidationError(f"{field_name} must be finite")
        if value != expected_value:
            raise ReproductionValidationError(f"{field_name} must be exactly {expected_value!r}")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReproductionValidationError(f"artifact path {path} is outside repository root") from exc


def _validate_artifact_path(root: Path, output_root: Path, path: Path, field: str) -> Path:
    _reject_reparse_components(root, path, field)
    resolved = path.resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ReproductionValidationError(f"{field} must remain under results/v2/reproduction") from exc
    return resolved


def validate_output_root(config: dict[str, Any], repository_root: str | Path) -> Path:
    root = Path(repository_root)
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ReproductionValidationError("outputs must be an object")
    value = outputs.get("root")
    if not isinstance(value, str) or not value.startswith("results/v2/reproduction"):
        raise ReproductionValidationError("outputs.root must be results/v2/reproduction or one of its subdirectories")
    path = _repo_path(root, value, "outputs.root")
    required = (root / "results" / "v2" / "reproduction").resolve()
    try:
        path.relative_to(required)
    except ValueError as exc:
        raise ReproductionValidationError("outputs.root must be under results/v2/reproduction") from exc
    return path


def load_reproduction_config(path: str | Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReproductionValidationError(f"malformed YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise ReproductionValidationError("configuration must be an object")
    return document


def _validated_config_path(
    repository_root: str | Path, config_path: str | Path, baseline_spec: dict[str, Any],
) -> Path:
    root = Path(repository_root)
    expected_value = baseline_spec["paths"]["config"]
    expected = _repo_path(root, expected_value, "config path")
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    _reject_reparse_components(root, supplied, "config path")
    if supplied.resolve() != expected:
        raise ReproductionValidationError(f"config path must be exactly {expected_value}")
    return expected


def validate_config(
    config: dict[str, Any], repository_root: str | Path,
    *, baseline_spec: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root = Path(repository_root)
    baseline = BASELINE_SPEC if baseline_spec is None else baseline_spec
    baseline_digest = deterministic_json_hash(baseline)
    _validate_finite_numbers(config)
    if config.get("schema_version") != 1:
        raise ReproductionValidationError("schema_version must be 1")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ReproductionValidationError("protocol must be an object")
    if protocol.get("version") != baseline["protocol_version"]:
        raise ReproductionValidationError(f"protocol.version must be exactly {baseline['protocol_version']}")
    if protocol.get("baseline_spec_sha256") != baseline_digest:
        raise ReproductionValidationError(
            f"protocol.baseline_spec_sha256 must be exactly {baseline_digest}"
        )
    if "verified before unpickling" not in str(protocol.get("trusted_pickle_boundary", "")):
        raise ReproductionValidationError("protocol.trusted_pickle_boundary must document pre-unpickle verification")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ReproductionValidationError("dataset must be an object")
    for field in ("id", "path", "sha256"):
        expected = baseline["dataset"][field]
        if dataset.get(field) != expected:
            raise ReproductionValidationError(f"dataset.{field} must be exactly {expected}")
    data_path = _repo_path(root, dataset.get("path", ""), "dataset.path")
    if not data_path.is_file():
        raise ReproductionValidationError(f"dataset file does not exist: {data_path}")
    actual_data_sha256 = file_sha256(data_path)
    if actual_data_sha256 != baseline["dataset"]["sha256"]:
        raise ReproductionValidationError(
            f"dataset SHA-256 mismatch: expected {baseline['dataset']['sha256']}, "
            f"got {actual_data_sha256}"
        )
    split = config.get("split")
    if not isinstance(split, dict):
        raise ReproductionValidationError("split must be an object")
    _expect_exact_path(
        split.get("metadata"), baseline["paths"]["split_metadata"], "split.metadata"
    )
    npz_value = split.get("npz")
    if not isinstance(npz_value, str):
        raise ReproductionValidationError("split.npz must be a repository-relative POSIX path")
    npz_pure = PurePosixPath(npz_value)
    if npz_pure.parts[:2] != ("splits", "v2") or npz_pure.suffix != ".npz":
        raise ReproductionValidationError("split.npz must be an NPZ under splits/v2")
    metadata_path = _repo_path(root, split.get("metadata", ""), "split.metadata")
    npz_path = _repo_path(root, npz_value, "split.npz")
    if type(config.get("seed")) is not type(baseline["seed"]) or config.get("seed") != baseline["seed"]:
        raise ReproductionValidationError(f"seed must be exactly {baseline['seed']!r}")
    if type(config.get("device")) is not type(baseline["device"]) or config.get("device") != baseline["device"]:
        raise ReproductionValidationError(f"device must be exactly {baseline['device']!r}")
    targets = config.get("historical_targets")
    if not isinstance(targets, dict) or set(targets) != set(REQUIRED_RUNS):
        raise ReproductionValidationError("historical_targets must contain exactly plain and conditioned")
    for name, expected in baseline["historical_targets"].items():
        target = targets.get(name)
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(float(target)):
            raise ReproductionValidationError(f"historical_targets.{name} must be a finite non-boolean number")
        if float(target) != float(expected):
            raise ReproductionValidationError(f"historical_targets.{name} must be exactly {expected}")
    tolerance = config.get("tolerance_pp")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)):
        raise ReproductionValidationError("tolerance_pp must be a finite non-boolean number")
    if float(tolerance) != float(baseline["tolerance_pp"]):
        raise ReproductionValidationError(f"tolerance_pp must be exactly {baseline['tolerance_pp']}")
    memory = config.get("memory")
    if not isinstance(memory, dict):
        raise ReproductionValidationError("memory must be an object")
    if memory.get("max_dataset_and_indices_bytes") != baseline["memory"]["max_dataset_and_indices_bytes"]:
        raise ReproductionValidationError("memory.max_dataset_and_indices_bytes does not match baseline")
    if memory.get("measurement") != baseline["memory"]["measurement"]:
        raise ReproductionValidationError("memory.measurement does not match steady-state baseline wording")
    if "RML2018" not in str(memory.get("scope", "")) or "memmap" not in str(memory.get("scope", "")):
        raise ReproductionValidationError("memory.scope must document the RML2018 Phase11 memmap boundary")
    _validate_locked_mapping(config.get("training"), baseline["training"], "training")
    evaluation_batch_size = config.get("evaluation_batch_size")
    expected_evaluation_batch_size = baseline["evaluation_batch_size"]
    if type(evaluation_batch_size) is not type(expected_evaluation_batch_size) or evaluation_batch_size != expected_evaluation_batch_size:
        raise ReproductionValidationError(
            f"evaluation_batch_size must be exactly {expected_evaluation_batch_size!r} with type int"
        )
    _validate_locked_mapping(config.get("architecture"), baseline["architecture"], "architecture")
    models = config.get("models")
    _validate_locked_mapping(models, baseline["models"], "models")
    outputs = config["outputs"]
    for field in ("root", "gate_state", "manifest", "progress", "failure_report"):
        baseline_field = "output_root" if field == "root" else field
        _expect_exact_path(outputs.get(field), baseline["paths"][baseline_field], f"outputs.{field}")
    output_root = validate_output_root(config, root)
    required_output_fields = ("gate_state", "manifest", "progress", "failure_report")
    paths = {
        "data": data_path, "split_metadata": metadata_path, "split_npz": npz_path,
        "output_root": output_root,
    }
    for field in required_output_fields:
        paths[field] = _repo_path(root, outputs.get(field, ""), f"outputs.{field}")
    return paths


def _canonical_source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_conditioned_dependency(repository_root: str | Path) -> str:
    root = Path(repository_root)
    relative = "models/model_snr.py"
    path = root / relative
    if not path.is_file():
        raise ReproductionValidationError(f"required conditioned model dependency is missing: {relative}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=root,
        capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise ReproductionValidationError(f"required conditioned model dependency is not tracked: {relative}")
    digest = _canonical_source_sha256(path)
    if digest != REQUIRED_CONDITIONED_MODEL_SHA256:
        raise ReproductionValidationError(
            f"conditioned model dependency SHA-256 mismatch: expected "
            f"{REQUIRED_CONDITIONED_MODEL_SHA256}, got {digest}"
        )
    return digest


def check_gate(actual: float, target: float, tolerance_pp: float) -> tuple[bool, float]:
    deviation = abs(float(actual) - float(target)) * 100.0
    return deviation <= float(tolerance_pp) + 1e-12, deviation


def _macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(np.concatenate([labels, predictions]))
    scores: list[float] = []
    for class_id in classes:
        tp = int(np.count_nonzero((labels == class_id) & (predictions == class_id)))
        fp = int(np.count_nonzero((labels != class_id) & (predictions == class_id)))
        fn = int(np.count_nonzero((labels == class_id) & (predictions != class_id)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    recalls: list[float] = []
    for class_id in np.unique(labels):
        selected = labels == class_id
        recalls.append(float(np.mean(predictions[selected] == labels[selected])))
    return float(np.mean(recalls)) if recalls else 0.0


def _validated_class_indices(values: Any, name: str, num_classes: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ReproductionValidationError(f"{name} must be a one-dimensional integer array")
    if len(array) and (int(array.min()) < 0 or int(array.max()) >= int(num_classes)):
        raise ReproductionValidationError(
            f"{name} class indices are outside the valid range [0,{int(num_classes)})"
        )
    return array.astype(np.int64, copy=False)


def evaluate_predictions(
    predictions: Any, labels: Any, snrs: Any, *, num_classes: int | None = None,
) -> dict[str, Any]:
    raw_labels = np.asarray(labels)
    if num_classes is None:
        if raw_labels.ndim != 1 or raw_labels.dtype.kind not in "iu" or len(raw_labels) == 0:
            raise ReproductionValidationError("labels must be a nonempty one-dimensional integer array")
        num_classes = int(raw_labels.max()) + 1
    predictions = _validated_class_indices(predictions, "predictions", num_classes)
    labels = _validated_class_indices(raw_labels, "labels", num_classes)
    snrs = np.asarray(snrs)
    if predictions.ndim != 1 or labels.ndim != 1 or snrs.ndim != 1:
        raise ReproductionValidationError("predictions, labels, and SNRs must be one-dimensional")
    if not (len(predictions) == len(labels) == len(snrs)) or len(labels) == 0:
        raise ReproductionValidationError("predictions, labels, and SNRs must have one equal nonzero length")
    per_snr: dict[str, float] = {}
    for snr in sorted(np.unique(snrs).tolist()):
        selected = snrs == snr
        per_snr[format_snr(float(snr))] = float(np.mean(predictions[selected] == labels[selected]))
    segments: dict[str, float | None] = {}
    for name, selected in {
        "low": snrs <= -8,
        "mid": (snrs >= -6) & (snrs <= -2),
        "high": snrs >= 0,
    }.items():
        segments[name] = float(np.mean(predictions[selected] == labels[selected])) if np.any(selected) else None
    return {
        "overall_accuracy": float(np.mean(predictions == labels)),
        "per_snr_accuracy": per_snr,
        "segments": segments,
        "macro_f1": _macro_f1(predictions, labels),
        "balanced_accuracy": _balanced_accuracy(predictions, labels),
    }


def _git_identifier(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {"commit": commit, "working_tree_dirty": dirty}


def _code_identifier(root: Path) -> dict[str, Any]:
    paths = [root / PurePosixPath(relative) for relative in SOURCE_DEPENDENCIES]
    files = {
        _relative(root, path): _canonical_source_sha256(path)
        for path in paths
    }
    return {"hash": deterministic_json_hash(files), "files": files, "git": _git_identifier(root)}


def _reference_source_hashes(root: Path) -> dict[str, Any]:
    files = {
        relative: file_sha256(root / PurePosixPath(relative))
        for relative in REFERENCE_SOURCE_DEPENDENCIES
    }
    return {"role": "non_executed_reference_only", "files": files}


def _attempt_id(config_hash: str, split_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{config_hash[:6]}-{split_hash[:6]}"


def _load_rml_dataset(
    data_path: Path, dataset_id: str, *,
    repository_root: str | Path = REPOSITORY_ROOT,
    approved_sha256: str = BASELINE_SPEC["dataset"]["sha256"],
    pickle_loader: Callable[..., Any] = pickle.load,
) -> dict[str, Any]:
    if dataset_id != "RML2016.10a":
        raise ReproductionValidationError(f"default loader only supports RML2016.10a, got {dataset_id}")
    root = Path(repository_root)
    approved_path = (root / PurePosixPath(BASELINE_SPEC["dataset"]["path"])).resolve()
    if Path(data_path).resolve() != approved_path:
        raise ReproductionValidationError(f"dataset is not the approved dataset path: {approved_path}")
    _reject_reparse_components(root, Path(data_path), "dataset.path")
    if not Path(data_path).is_file():
        raise ReproductionValidationError(f"approved dataset path does not exist: {data_path}")
    actual_sha256 = file_sha256(Path(data_path))
    if actual_sha256 != approved_sha256:
        raise ReproductionValidationError(
            f"dataset SHA-256 mismatch before pickle load: expected {approved_sha256}, got {actual_sha256}"
        )
    with data_path.open("rb") as handle:
        cells = pickle_loader(handle, encoding="bytes")
    class_to_index = {
        b"QAM16": 0, b"QAM64": 1, b"8PSK": 2, b"WBFM": 3,
        b"BPSK": 4, b"CPFSK": 5, b"AM-DSB": 6, b"GFSK": 7,
        b"PAM4": 8, b"QPSK": 9, b"AM-SSB": 10,
    }
    classes = sorted({key[0] for key in cells})
    snr_levels = sorted({key[1] for key in cells})
    signal_parts: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[float] = []
    for class_label in classes:
        if class_label not in class_to_index:
            raise ReproductionValidationError(f"unknown class label {class_label!r}")
        for snr in snr_levels:
            values = np.asarray(cells[(class_label, snr)], dtype=np.float32)
            signal_parts.append(values)
            labels.extend([class_to_index[class_label]] * len(values))
            snrs.extend([float(snr)] * len(values))
    return {
        "signals": np.vstack(signal_parts),
        "labels": np.asarray(labels, dtype=np.int64),
        "snrs": np.asarray(snrs, dtype=np.float32),
    }


def _snr_bins(snrs: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(snrs) + 20) // 2, 0, 19).astype(np.int64)


class HistoricalEpochControl:
    """Exact loss-patience/LR and accuracy-checkpoint semantics from legacy Trainer."""

    def __init__(self, *, patience: int, milestone_step: int, gamma: float, learning_rate: float) -> None:
        self.patience = int(patience)
        self.milestone_step = int(milestone_step)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.best_loss: float | None = None
        self.best_accuracy = 0.0
        self.counter = 0

    def step(self, validation_loss: float, validation_accuracy: float) -> dict[str, Any]:
        checkpoint = float(validation_accuracy) >= self.best_accuracy
        if checkpoint:
            self.best_accuracy = float(validation_accuracy)
        if self.best_loss is None or float(validation_loss) <= self.best_loss:
            self.best_loss = float(validation_loss)
            self.counter = 0
        else:
            self.counter += 1
        lr_before = self.learning_rate
        if self.counter and self.counter % self.milestone_step == 0:
            self.learning_rate *= self.gamma
        return {
            "checkpoint": checkpoint,
            "best_accuracy": self.best_accuracy,
            "counter": self.counter,
            "lr_before": lr_before,
            "lr_after": self.learning_rate,
            "stop": self.counter >= self.patience,
        }


class IndexBackedDataset:
    """A torch Dataset view that never copies the base signal or label tensors."""

    def __init__(self, signals: Any, labels: Any, snrs: Any, indices: Any) -> None:
        import torch

        self.signals = signals
        self.labels = labels
        self.snrs = snrs
        self.indices = torch.as_tensor(indices, dtype=torch.int64)
        snr_values = snrs.detach().cpu().numpy() if hasattr(snrs, "detach") else np.asarray(snrs)
        self.snr_bins = torch.from_numpy(_snr_bins(snr_values))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> tuple[Any, Any, Any]:
        index = self.indices[position]
        return self.signals[index], self.labels[index], self.snr_bins[index]


def make_index_dataset(signals: Any, labels: Any, snrs: Any, indices: Any) -> IndexBackedDataset:
    return IndexBackedDataset(signals, labels, snrs, indices)


def make_historical_loader(dataset: Any, *, batch_size: int, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0,
    )


def make_historical_train_val_loaders(
    train_dataset: Any, validation_dataset: Any, *,
    train_batch_size: int, validation_batch_size: int,
) -> tuple[Any, Any]:
    """Match legacy Create_Data_Loader: both loaders shuffle on global Torch RNG."""
    return (
        make_historical_loader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
        ),
        make_historical_loader(
            validation_dataset, batch_size=validation_batch_size, shuffle=True,
        ),
    )


def estimate_phase1_dataset_bytes(dataset: dict[str, Any], indices: Any) -> int:
    def nbytes(value: Any) -> int:
        if hasattr(value, "nbytes"):
            return int(value.nbytes)
        if hasattr(value, "element_size") and hasattr(value, "numel"):
            return int(value.element_size() * value.numel())
        return int(np.asarray(value).nbytes)

    index_bytes = sum(nbytes(item) for item in indices) if isinstance(indices, (list, tuple)) else nbytes(indices)
    return sum(nbytes(dataset[name]) for name in ("signals", "labels", "snrs")) + index_bytes


def _model_construction_identity(run: str, config: dict[str, Any]) -> dict[str, Any]:
    if run not in REQUIRED_RUNS:
        raise ReproductionValidationError(f"unknown reproduction run {run!r}")
    model = config["models"][run]
    architecture = config["architecture"]
    common_fields = (
        "num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim",
        "regu_details", "regu_approx",
    )
    arguments = {field: architecture[field] for field in common_fields if field in architecture}
    if run == "conditioned":
        if "num_snr_bins" in architecture:
            arguments["num_snr_bins"] = architecture["num_snr_bins"]
        if "snr_embedding_dim" in architecture:
            arguments["snr_emb_dim"] = architecture["snr_embedding_dim"]
    module = PurePosixPath(model["source"]).with_suffix("").as_posix().replace("/", ".")
    return {
        "class_path": f"{module}.{model['name']}",
        "source": model["source"], "name": model["name"],
        "conditioner": model["conditioner"],
        "constructor_arguments": arguments,
    }


def _default_trainer(
    run: str,
    dataset: dict[str, Any],
    fixed_split: FixedSplit,
    destination: Path,
    logger: Callable[[str], None],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    import random
    import torch
    from torch import nn, optim

    from models.model import AWN
    from models.model_snr import AWNSNR
    from util.utils import fix_seed

    seed = int(config["seed"])
    fix_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    architecture = config["architecture"]
    model_construction = _model_construction_identity(run, config)
    if run == "plain":
        if model_construction["class_path"] != "models.model.AWN":
            raise ReproductionValidationError("plain declared model does not match AWN construction")
        model = AWN(**model_construction["constructor_arguments"]).to(device)
        conditioned = False
    else:
        if model_construction["class_path"] != "models.model_snr.AWNSNR":
            raise ReproductionValidationError("conditioned declared model does not match AWNSNR construction")
        model = AWNSNR(**model_construction["constructor_arguments"]).to(device)
        conditioned = True
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    signals = torch.from_numpy(np.asarray(dataset["signals"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(dataset["labels"], dtype=np.int64))
    snrs = torch.from_numpy(np.asarray(dataset["snrs"], dtype=np.float32))
    train_dataset = make_index_dataset(signals, labels, snrs, fixed_split.train_idx)
    val_dataset = make_index_dataset(signals, labels, snrs, fixed_split.val_idx)
    training = config["training"]
    train_loader, val_loader = make_historical_train_val_loaders(
        train_dataset, val_dataset,
        train_batch_size=int(training["batch_size"]),
        validation_batch_size=int(training["batch_size"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training["lr"]))
    checkpoint = destination / "checkpoint.pt"
    best_epoch = -1
    control = HistoricalEpochControl(
        patience=int(training["patience"]), milestone_step=int(training["milestone_step"]),
        gamma=float(training["gamma"]), learning_rate=float(training["lr"]),
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    def forward(batch_signals, batch_bins):
        return model(batch_signals, batch_bins) if conditioned else model(batch_signals)

    for epoch in range(int(training["max_epochs"])):
        model.train()
        train_loss = train_accuracy = train_batches = 0.0
        for batch_signals, batch_labels, batch_bins in train_loader:
            batch_signals = batch_signals.to(device)
            batch_labels = batch_labels.to(device)
            batch_bins = batch_bins.to(device)
            logits, regularizers = forward(batch_signals, batch_bins)
            loss = criterion(logits, batch_labels) + sum(regularizers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_accuracy += float((torch.argmax(logits, dim=1) == batch_labels).double().mean().item())
            train_batches += 1
        model.eval()
        val_loss = val_accuracy = val_batches = 0.0
        with torch.no_grad():
            for batch_signals, batch_labels, batch_bins in val_loader:
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)
                batch_bins = batch_bins.to(device)
                logits, regularizers = forward(batch_signals, batch_bins)
                loss = criterion(logits, batch_labels) + sum(regularizers)
                val_loss += float(loss.item())
                val_accuracy += float((torch.argmax(logits, dim=1) == batch_labels).double().mean().item())
                val_batches += 1
        train_loss /= train_batches
        train_accuracy /= train_batches
        val_loss /= val_batches
        val_accuracy /= val_batches
        decision = control.step(val_loss, val_accuracy)
        history.append({
            "epoch": epoch,
            "learning_rate": decision["lr_before"],
            "learning_rate_after_validation": decision["lr_after"],
            "early_stopping_counter": decision["counter"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })
        logger(
            f"{run} epoch={epoch:03d} train_acc={train_accuracy:.4f} "
            f"val_acc={val_accuracy:.4f} val_loss={val_loss:.6f} "
            f"lr={decision['lr_before']:.6g} patience={decision['counter']}"
        )
        if decision["checkpoint"]:
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        for group in optimizer.param_groups:
            group["lr"] = decision["lr_after"]
        if decision["stop"]:
            logger(f"{run} early_stop epoch={epoch:03d}")
            break
    if best_epoch < 0 or not checkpoint.is_file():
        raise RuntimeError(f"{run} did not produce a best checkpoint")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    test_idx = fixed_split.test_idx.astype(np.int64, copy=False)
    test_labels = np.asarray(dataset["labels"])[test_idx].astype(np.int64)
    test_snrs = np.asarray(dataset["snrs"])[test_idx]
    test_dataset = make_index_dataset(signals, labels, snrs, test_idx)
    test_loader = make_historical_loader(
        test_dataset, batch_size=int(config.get("evaluation_batch_size", 64)), shuffle=False,
    )
    prediction_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_signals, _batch_labels, batch_bins in test_loader:
            batch_signals = batch_signals.to(device)
            batch_bins = batch_bins.to(device)
            logits, _ = forward(batch_signals, batch_bins)
            prediction_parts.append(torch.argmax(logits, dim=1).cpu().numpy())
    predictions = np.concatenate(prediction_parts).astype(np.int64)
    duration = time.perf_counter() - started
    return {
        "predictions": predictions, "labels": test_labels, "snrs": test_snrs,
        "checkpoint": checkpoint, "best_epoch": best_epoch,
        "epochs_completed": len(history), "best_val_accuracy": control.best_accuracy,
        "duration_seconds": duration, "parameter_count": parameter_count,
        "history": history,
        "model_construction": model_construction,
    }


def _write_predictions(path: Path, outcome: dict[str, Any], sample_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".npz", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(
            temporary,
            predictions=np.asarray(outcome["predictions"], dtype=np.int64),
            labels=np.asarray(outcome["labels"], dtype=np.int64),
            snrs=np.asarray(outcome["snrs"]),
            sample_ids=np.asarray(sample_ids),
        )
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    required = {"predictions", "labels", "snrs", "sample_ids"}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ReproductionValidationError(
                    f"prediction bundle fields mismatch: expected {sorted(required)}"
                )
            return {name: archive[name] for name in required}
    except ReproductionValidationError:
        raise
    except Exception as exc:
        raise ReproductionValidationError(f"cannot load prediction bundle {path}: {exc}") from exc


def _validated_bundle_metrics(
    path: Path, fixed_split: FixedSplit, dataset: dict[str, Any], num_classes: int | None = None,
) -> dict[str, Any]:
    bundle = _load_prediction_bundle(path)
    test_idx = fixed_split.test_idx.astype(np.int64)
    expected_ids = np.asarray(fixed_split.sample_ids[test_idx])
    if not np.array_equal(bundle["sample_ids"], expected_ids):
        raise ReproductionValidationError("prediction sample IDs do not match the fixed test split")
    if len(np.unique(bundle["sample_ids"])) != len(expected_ids):
        raise ReproductionValidationError("prediction sample IDs are not unique")
    try:
        expected_labels = np.asarray(dataset["labels"])[test_idx].astype(np.int64)
        expected_snrs = np.asarray(dataset["snrs"])[test_idx]
    except (KeyError, IndexError, TypeError) as exc:
        raise ReproductionValidationError(f"current dataset cannot bind fixed test identities: {exc}") from exc
    if not np.array_equal(bundle["labels"].astype(np.int64), expected_labels):
        raise ReproductionValidationError("prediction labels do not match the current fixed test split")
    if not np.array_equal(bundle["snrs"], expected_snrs):
        raise ReproductionValidationError("prediction SNR values do not match the current fixed test split")
    if num_classes is None:
        num_classes = int(np.asarray(dataset["labels"]).max()) + 1
    return evaluate_predictions(
        bundle["predictions"], bundle["labels"], bundle["snrs"], num_classes=num_classes,
    )


def _metrics_match(reported: Any, computed: Any, location: str = "metrics") -> None:
    if isinstance(computed, dict):
        if not isinstance(reported, dict) or set(reported) != set(computed):
            raise ReproductionValidationError(f"reported metrics mismatch at {location}")
        for key in computed:
            _metrics_match(reported[key], computed[key], f"{location}.{key}")
        return
    if computed is None:
        if reported is not None:
            raise ReproductionValidationError(f"reported metrics mismatch at {location}")
        return
    if isinstance(computed, (int, float)):
        if isinstance(reported, bool) or not isinstance(reported, (int, float)):
            raise ReproductionValidationError(f"reported metrics mismatch at {location}")
        if abs(float(reported) - float(computed)) > 1e-12:
            raise ReproductionValidationError(f"reported metrics mismatch at {location}")
        return
    if reported != computed:
        raise ReproductionValidationError(f"reported metrics mismatch at {location}")


def _code_record_matches(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    if not isinstance(recorded, dict) or not isinstance(recorded.get("files"), dict):
        raise ReproductionValidationError("code identity record is malformed")
    if recorded.get("hash") != deterministic_json_hash(recorded["files"]):
        raise ReproductionValidationError("code identity canonical hash mismatch")
    if recorded["hash"] == current["hash"]:
        return
    scientific_paths = {"models/model.py", "models/model_snr.py", "v2/splits.py"}
    legacy_ok = (
        recorded["hash"] == LEGACY_PHASE1_CODE_HASH
        and recorded["files"].get("scripts/v2/phase1_reproduce.py") == LEGACY_PHASE1_RUNNER_SHA256
        and all(recorded["files"].get(path) == current["files"].get(path) for path in scientific_paths)
    )
    if not legacy_ok:
        raise ReproductionValidationError("code identity does not match current or approved legacy Phase 1 code")


def verify_reusable_result(
    result_path: str | Path,
    *,
    repository_root: str | Path,
    config: dict[str, Any],
    fixed_split: FixedSplit,
    dataset: dict[str, Any],
    current_code: dict[str, Any],
    run: str,
    baseline_spec: dict[str, Any] | None = None,
    _allow_legacy_missing_integrity: bool = False,
) -> dict[str, Any]:
    """Fully bind one reusable result to identity, provenance, and persisted predictions."""
    root = Path(repository_root)
    baseline = BASELINE_SPEC if baseline_spec is None else baseline_spec
    baseline_digest = deterministic_json_hash(baseline)
    result_path = Path(result_path)
    destination = result_path.parent
    output_root = validate_output_root(config, root)
    _validate_artifact_path(root, output_root, destination, "run artifact destination")
    _validate_artifact_path(root, output_root, result_path, "result artifact")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionValidationError(f"cannot load result JSON: {exc}") from exc
    legacy_missing = _allow_legacy_missing_integrity and (
        "result_canonical_hash" not in result
        and isinstance(result.get("artifacts"), dict)
        and "run_spec_canonical_hash" not in result["artifacts"]
        and "result_json" not in result["artifacts"]
    )
    if not legacy_missing and result.get("result_canonical_hash") != _canonical_payload_hash(
        result, omit={"result_canonical_hash"}
    ):
        raise ReproductionValidationError("result canonical hash mismatch")
    if result.get("run") != run:
        raise ReproductionValidationError("run identity mismatch in result")
    if result.get("status") != "completed":
        raise ReproductionValidationError("result is not completed")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReproductionValidationError("result artifact hash fields are missing")
    expected_paths = {
        "checkpoint": _relative(root, destination / "checkpoint.pt"),
        "predictions": _relative(root, destination / "predictions.npz"),
        "run_spec": _relative(root, destination / "run_spec.json"),
        "result_json": _relative(root, result_path),
    }
    for name, expected_path in expected_paths.items():
        if legacy_missing and name == "result_json":
            continue
        if artifacts.get(name) != expected_path:
            raise ReproductionValidationError(f"artifact path mismatch for {name}")
    checkpoint = destination / "checkpoint.pt"
    predictions = destination / "predictions.npz"
    run_spec_path = destination / "run_spec.json"
    for field, artifact_path in (
        ("checkpoint artifact", checkpoint),
        ("prediction artifact", predictions),
        ("run_spec artifact", run_spec_path),
    ):
        _validate_artifact_path(root, output_root, artifact_path, field)
    if file_sha256(checkpoint) != artifacts.get("checkpoint_sha256"):
        raise ReproductionValidationError("checkpoint hash mismatch")
    if file_sha256(predictions) != artifacts.get("predictions_sha256"):
        raise ReproductionValidationError("prediction hash mismatch")
    if file_sha256(run_spec_path) != artifacts.get("run_spec_sha256"):
        raise ReproductionValidationError("run_spec file hash mismatch")
    try:
        run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionValidationError(f"cannot load run_spec: {exc}") from exc
    if not legacy_missing and _canonical_payload_hash(run_spec) != artifacts.get("run_spec_canonical_hash"):
        raise ReproductionValidationError("run_spec canonical hash mismatch")
    if run_spec.get("run") != run or result.get("attempt_id") != run_spec.get("attempt_id"):
        raise ReproductionValidationError("run identity mismatch between result and run_spec")
    expected_model = config["models"][run]
    actual_model = run_spec.get("model")
    if not isinstance(actual_model, dict) or actual_model.get("name") != expected_model["name"]:
        raise ReproductionValidationError("model identity mismatch")
    if actual_model.get("conditioner") != expected_model["conditioner"]:
        raise ReproductionValidationError("conditioner identity mismatch")
    if actual_model != expected_model:
        raise ReproductionValidationError("model identity configuration mismatch")
    if run_spec.get("seed") != config["seed"]:
        raise ReproductionValidationError("seed identity mismatch")
    if run_spec.get("dataset") != config["dataset"]:
        raise ReproductionValidationError("dataset identity mismatch")
    if run_spec.get("device") != config["device"] or run_spec.get("training") != config["training"]:
        raise ReproductionValidationError("training protocol identity mismatch")
    if run_spec.get("architecture") != config["architecture"]:
        raise ReproductionValidationError("architecture identity mismatch")
    if run_spec.get("evaluation_batch_size") != config["evaluation_batch_size"]:
        raise ReproductionValidationError("evaluation batch-size identity mismatch")
    if run_spec.get("model_construction") != _model_construction_identity(run, config):
        raise ReproductionValidationError("model construction identity mismatch")
    required_spec_fields = {
        "schema_version", "runner_version", "attempt_id", "run", "created_at",
        "dataset", "seed", "device", "training", "model", "provenance",
        "environment", "code", "deterministic_settings", "initialization",
        "selection_policy", "baseline_spec", "baseline_spec_sha256", "memory",
        "reference_source_hashes", "architecture", "evaluation_batch_size",
        "model_construction",
    }
    if set(run_spec) != required_spec_fields:
        raise ReproductionValidationError("run_spec content fields are incomplete or unexpected")
    references = run_spec.get("reference_source_hashes")
    if not isinstance(references, dict) or references.get("role") != "non_executed_reference_only":
        raise ReproductionValidationError("reference source hashes are not labeled non-executed")
    if not isinstance(references.get("files"), dict):
        raise ReproductionValidationError("reference source hash records are malformed")
    config_hash = deterministic_json_hash(config)
    expected_provenance = {
        "config_hash": config_hash,
        "split_hash": fixed_split.split_hash,
        "code_hash": run_spec["code"].get("hash"),
        "data_sha256": fixed_split.metadata["data_sha256"],
        "baseline_spec_sha256": baseline_digest,
    }
    if run_spec.get("provenance") != expected_provenance or result.get("provenance") != expected_provenance:
        raise ReproductionValidationError("config/split/code/data provenance hash mismatch")
    if run_spec.get("baseline_spec") != baseline or run_spec.get("baseline_spec_sha256") != baseline_digest:
        raise ReproductionValidationError("immutable baseline specification mismatch")
    if config.get("protocol", {}).get("baseline_spec_sha256") != baseline_digest:
        raise ReproductionValidationError("configuration baseline digest mismatch")
    _code_record_matches(run_spec["code"], current_code)
    computed_metrics = _validated_bundle_metrics(
        predictions, fixed_split, dataset, int(config["architecture"]["num_classes"]),
    )
    _metrics_match(result.get("metrics"), computed_metrics)
    target = float(config["historical_targets"][run])
    passed, deviation = check_gate(computed_metrics["overall_accuracy"], target, config["tolerance_pp"])
    if not passed:
        raise ReproductionValidationError("completed result no longer passes its configured gate")
    if result.get("target_accuracy") != target or result.get("tolerance_pp") != float(config["tolerance_pp"]):
        raise ReproductionValidationError("gate target identity mismatch")
    if abs(float(result.get("deviation_pp", -1)) - deviation) > 1e-12:
        raise ReproductionValidationError("reported gate deviation mismatch")
    return result


def upgrade_legacy_result_integrity(
    result_path: str | Path,
    *,
    repository_root: str | Path,
    config: dict[str, Any],
    fixed_split: FixedSplit,
    dataset: dict[str, Any],
    current_code: dict[str, Any],
    run: str,
    baseline_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add derived integrity fields only after a legacy result fully validates."""
    root = Path(repository_root)
    path = Path(result_path)
    legacy = verify_reusable_result(
        path,
        repository_root=root,
        config=config,
        fixed_split=fixed_split,
        dataset=dataset,
        current_code=current_code,
        run=run,
        baseline_spec=baseline_spec,
        _allow_legacy_missing_integrity=True,
    )
    if "result_canonical_hash" in legacy:
        return legacy
    upgraded = json.loads(json.dumps(legacy))
    run_spec_path = path.parent / "run_spec.json"
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    upgraded["artifacts"]["run_spec_canonical_hash"] = _canonical_payload_hash(run_spec)
    upgraded["artifacts"]["result_json"] = _relative(root, path)
    upgraded["result_canonical_hash"] = _canonical_payload_hash(
        upgraded, omit={"result_canonical_hash"}
    )
    _atomic_json(path, upgraded)
    return verify_reusable_result(
        path,
        repository_root=root,
        config=config,
        fixed_split=fixed_split,
        dataset=dataset,
        current_code=current_code,
        run=run,
        baseline_spec=baseline_spec,
    )


def _find_reusable(
    output_root: Path, run: str, *, root: Path, config: dict[str, Any],
    fixed_split: FixedSplit, dataset: dict[str, Any], current_code: dict[str, Any],
    logger: Callable[[str], None],
    baseline_spec: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    for result_path in sorted(output_root.glob(f"{run}_seed*/result.json"), reverse=True):
        try:
            result = verify_reusable_result(
                result_path, repository_root=root, config=config,
                fixed_split=fixed_split, dataset=dataset,
                current_code=current_code, run=run, baseline_spec=baseline_spec,
            )
            return result_path.parent, result
        except (OSError, ReproductionValidationError, KeyError, TypeError, ValueError) as exc:
            logger(f"{run}: rejected reusable artifact {_relative(root, result_path)}: {exc}")
    return None


def _append_failure_report(path: Path, attempt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text(encoding="utf-8") if path.exists() else "# Phase 1 Reproduction Failures\n"
    marker = f"REPRO_ATTEMPT:{attempt['attempt_id']}"
    start_marker = f"<!-- {marker}:START -->"
    end_marker = f"<!-- {marker}:END -->"
    lines = [
        start_marker, f"## Attempt `{attempt['attempt_id']}`", "",
        f"- Time: `{attempt['finished_at']}`",
        f"- Gate status: `{attempt['status']}`",
        f"- Environment: `{json.dumps(attempt['environment'], ensure_ascii=False, sort_keys=True)}`",
    ]
    for name in REQUIRED_RUNS:
        run = attempt["runs"].get(name, {})
        if "error" in run:
            lines.append(f"- {name}: execution error: {run['error']}")
        else:
            lines.append(
                f"- {name}: actual `{run.get('actual')}`, target `{run.get('target')}`, "
                f"deviation `{run.get('deviation_pp')}` pp, status `{run.get('status')}`"
            )
    if attempt.get("primary_error"):
        primary = attempt["primary_error"]
        lines.append(f"- Primary error: `{primary.get('type')}: {primary.get('message')}`")
    for error in attempt.get("finalization_errors", []):
        lines.append(f"- Finalization error: `{error}`")
    lines.extend([
        "", "### Diagnostics", "",
        "Published artifacts are retained only where the run entry above records a destination and its result evidence exists; setup failures may occur before model artifacts are created. No tolerance tuning or protocol expansion was attempted.",
        "", "### Next action", "",
        "Audit the failed run and protocol implementation before any Phase 2 expansion; rerun only as a new preserved attempt.",
        end_marker,
    ])
    block = "\n".join(lines) + "\n"
    start = prior.find(start_marker)
    if start >= 0:
        end = prior.find(end_marker, start)
        if end < 0:
            raise ReproductionValidationError(f"failure report has incomplete attempt block: {attempt['attempt_id']}")
        end += len(end_marker)
        updated = prior[:start] + block.rstrip("\n") + prior[end:]
    else:
        updated = prior.rstrip() + "\n\n" + block
    _atomic_text(path, updated)


def _manifest_record(
    root: Path, config: dict[str, Any], split: FixedSplit | str, run: str,
    destination: Path, status: str, notes: str,
) -> ArtifactRecord:
    model = config["models"][run]
    split_hash = split.split_hash if isinstance(split, FixedSplit) else split
    return ArtifactRecord(
        experiment=f"phase1_reproduction_{run}_{destination.name}", dataset=config["dataset"]["id"],
        split_hash=split_hash, model=model["name"], conditioner=model["conditioner"],
        seed=int(config["seed"]), checkpoint=_relative(root, destination / "checkpoint.pt"),
        result_json=_relative(root, destination / "result.json"), figure_paths=[],
        status=status, notes=notes,
    )


def _default_failure_paths(root: Path) -> dict[str, Path]:
    return {
        "output_root": root / "results/v2/reproduction",
        "gate_state": root / "results/v2/reproduction/gate_state.json",
        "manifest": root / "manifest.json",
        "progress": root / "reports/v2_progress.md",
        "failure_report": root / "reports/reproduction_failure.md",
    }


def _failure_artifact(
    root: Path, config: dict[str, Any], run: str, attempt: dict[str, Any],
    destination: Path, primary: dict[str, str],
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    run_spec_path = destination / "run_spec.json"
    if not run_spec_path.exists():
        _atomic_json(run_spec_path, {
            "schema_version": 1, "runner_version": RUNNER_VERSION,
            "attempt_id": attempt["attempt_id"], "run": run,
            "created_at": _utc_now(), "status": "setup_failed",
            "dataset": config.get("dataset"), "seed": config.get("seed"),
            "model": config.get("models", {}).get(run),
            "primary_error": primary,
        })
    result_path = destination / "result.json"
    failure_result = {
        "schema_version": 1, "runner_version": RUNNER_VERSION,
        "attempt_id": attempt["attempt_id"], "run": run, "status": "failed",
        "error": f"{primary['type']}: {primary['message']}",
        "traceback": primary["traceback"],
        "artifacts": {
            "result_json": _relative(root, result_path),
            "run_spec": _relative(root, run_spec_path),
            "run_spec_sha256": file_sha256(run_spec_path),
        },
    }
    failure_result["result_canonical_hash"] = _canonical_payload_hash(
        failure_result, omit={"result_canonical_hash"}
    )
    _atomic_json(result_path, failure_result)
    return result_path


def _setup_failure_artifact(
    root: Path, output_root: Path, attempt: dict[str, Any], primary: dict[str, str],
) -> Path:
    """Publish a safe setup-only artifact without consulting unvalidated config."""
    destination = output_root / f"setup_failed_{attempt['attempt_id']}"
    _validate_artifact_path(root, output_root, destination, "setup failure destination")
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    result = {
        "schema_version": 1, "runner_version": RUNNER_VERSION,
        "attempt_id": attempt["attempt_id"], "run": "setup", "status": "failed",
        "error": f"{primary['type']}: {primary['message']}",
        "traceback": primary["traceback"],
        "artifacts": {"result_json": _relative(root, result_path)},
    }
    result["result_canonical_hash"] = _canonical_payload_hash(
        result, omit={"result_canonical_hash"},
    )
    _atomic_json(result_path, result)
    attempt["setup_failure"] = {
        "status": "failed", "destination": _relative(root, destination),
        "result_json": _relative(root, result_path),
        "result_sha256": file_sha256(result_path),
        "error": result["error"],
    }
    return destination


def _finalize_failure(context: dict[str, Any], exc: Exception) -> list[str]:
    """Best-effort durable finalization; never raises over the primary failure."""
    terminal_evidence_statuses = {"completed", "gate_failed"}
    root = Path(context["root"])
    logger = context.get("logger", print)
    paths = _default_failure_paths(root) | context.get("paths", {})
    config = context.get("config")
    split = context.get("split")
    attempt = context.get("attempt") or {
        "schema_version": 1, "runner_version": RUNNER_VERSION,
        "attempt_id": _attempt_id("setup", "setup"), "started_at": _utc_now(),
        "config_hash": deterministic_json_hash(config) if isinstance(config, dict) else "unavailable",
        "split_hash": getattr(split, "split_hash", "unavailable"),
        "data_sha256": getattr(split, "metadata", {}).get("data_sha256", "unavailable"),
        "environment": environment_fingerprint(), "code": {}, "runs": {},
    }
    primary = {
        "type": type(exc).__name__, "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    attempt["primary_error"] = primary
    attempt["status"] = "failed"
    attempt["finished_at"] = _utc_now()
    secondary: list[str] = []
    destinations: dict[str, Path] = {}
    config_validated = context.get("config_validated") is True
    if config_validated and isinstance(config, dict) and isinstance(config.get("models"), dict):
        for run in REQUIRED_RUNS:
            if run not in config["models"]:
                continue
            existing = attempt.setdefault("runs", {}).get(run, {})
            if "destination" in existing:
                destination = root / PurePosixPath(existing["destination"])
            elif context.get("current_run") == run and context.get("current_destination") is not None:
                destination = Path(context["current_destination"])
            else:
                destination = paths["output_root"] / f"{run}_seed{config.get('seed', 'unknown')}_{attempt['attempt_id']}"
            destinations[run] = destination
            if existing.get("status") not in terminal_evidence_statuses:
                attempt["runs"][run] = {
                    "status": "failed", "error": f"{primary['type']}: {primary['message']}",
                    "destination": _relative(root, destination),
                }
            try:
                if attempt["runs"][run]["status"] == "failed":
                    result_path = _failure_artifact(
                        root, config, run, attempt, destination, primary,
                    )
                    attempt["runs"][run].update({
                        "result_json": _relative(root, result_path),
                        "result_sha256": file_sha256(result_path),
                    })
            except Exception as final_exc:
                secondary.append(f"failure artifact {run}: {type(final_exc).__name__}: {final_exc}")
        try:
            store = ManifestStore(paths["manifest"])
            split_identity: FixedSplit | str = split if isinstance(split, FixedSplit) else "unavailable"
            for run, destination in destinations.items():
                evidence = attempt["runs"][run]
                status = evidence["status"]
                if status in terminal_evidence_statuses:
                    notes = (
                        f"attempt={attempt['attempt_id']}; status={status}; "
                        f"deviation_pp={evidence.get('deviation_pp')}; "
                        "terminal evidence recovered during failure finalization"
                    )
                else:
                    notes = (
                        f"attempt={attempt['attempt_id']}; "
                        f"primary_error={primary['type']}: {primary['message']}"
                    )
                store.upsert(_manifest_record(
                    root, config, split_identity, run, destination, status, notes,
                ))
        except Exception as final_exc:
            secondary.append(f"manifest finalization: {type(final_exc).__name__}: {final_exc}")
    else:
        try:
            _setup_failure_artifact(root, paths["output_root"], attempt, primary)
        except Exception as final_exc:
            secondary.append(f"setup failure artifact: {type(final_exc).__name__}: {final_exc}")
    attempt["finalization_errors"] = secondary
    try:
        _append_failure_report(paths["failure_report"], attempt)
    except Exception as final_exc:
        secondary.append(f"failure report finalization: {type(final_exc).__name__}: {final_exc}")
    try:
        failure_reasons = []
        for run, evidence in attempt.get("runs", {}).items():
            if evidence.get("status") == "gate_failed":
                failure_reasons.append(
                    f"{run} gate_failed (actual={evidence.get('actual')}, "
                    f"target={evidence.get('target')}, deviation_pp={evidence.get('deviation_pp')})"
                )
            elif evidence.get("status") == "failed":
                failure_reasons.append(f"{run} failed ({evidence.get('error', 'unknown error')})")
        failure_summary = "; ".join(failure_reasons) or f"{primary['type']}: {primary['message']}"
        update_phase_progress(
            "Phase 1",
            completed="Any completed outputs from the interrupted attempt were preserved.",
            failed=f"Phase 1 failed: {failure_summary}",
            unexpected="The primary failure and any finalization errors are retained in the durable gate state.",
            interpretation="Phase 2 expansion is blocked; no tuning was performed.",
            next_gate="STOP: audit Phase 1 failure evidence before any new attempt.",
            path=paths["progress"],
        )
    except Exception as final_exc:
        secondary.append(f"progress finalization: {type(final_exc).__name__}: {final_exc}")
    attempt["finalization_errors"] = secondary
    # The immutable attempt plus current pointer are the sole final commit point.
    # Never publish them unless every required ledger/report update succeeded.
    if not secondary:
        try:
            publish_gate_state(
                paths["output_root"], paths["gate_state"], attempt, repository_root=root,
            )
        except Exception as final_exc:
            secondary.append(f"gate-state finalization: {type(final_exc).__name__}: {final_exc}")
    logger(f"Phase 1 primary failure: {primary['type']}: {primary['message']}")
    for item in secondary:
        logger(f"Phase 1 finalization error (primary preserved): {item}")
    return secondary


def _run_phase1_impl(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
    dataset_loader: Callable[..., Any] | None = None,
    force: bool = False,
    logger: Callable[[str], None] = print,
    baseline_spec: dict[str, Any],
    _failure_context: dict[str, Any],
) -> int:
    root = Path(repository_root) if repository_root is not None else REPOSITORY_ROOT
    _failure_context.update({"root": root, "logger": logger})
    validated_config_path = _validated_config_path(root, config_path, baseline_spec)
    config = load_reproduction_config(validated_config_path)
    _failure_context["config"] = config
    paths = validate_config(config, root, baseline_spec=baseline_spec)
    _failure_context["paths"] = paths
    _failure_context["config_validated"] = True
    if trainer is None:
        validate_conditioned_dependency(root)
    split = load_split(paths["split_metadata"], npz_path=paths["split_npz"], data_path=paths["data"])
    _failure_context["split"] = split
    baseline_digest = deterministic_json_hash(baseline_spec)
    config_hash = deterministic_json_hash(config)
    source_root = REPOSITORY_ROOT if trainer is not None else root
    code = _code_identifier(source_root)
    reference_source_hashes = _reference_source_hashes(source_root)
    expected = {"config_hash": config_hash, "split_hash": split.split_hash, "code_hash": code["hash"]}
    attempt_id = _attempt_id(config_hash, split.split_hash)
    environment = environment_fingerprint()
    attempt: dict[str, Any] = {
        "schema_version": 1, "runner_version": RUNNER_VERSION,
        "attempt_id": attempt_id, "started_at": _utc_now(),
        "config_hash": config_hash, "split_hash": split.split_hash,
        "data_sha256": split.metadata["data_sha256"],
        "baseline_spec": baseline_spec, "baseline_spec_sha256": baseline_digest,
        "environment": environment,
        "code": code, "reference_source_hashes": reference_source_hashes, "runs": {},
    }
    _failure_context["attempt"] = attempt
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    if dataset_loader is None:
        dataset = _load_rml_dataset(
            paths["data"], config["dataset"]["id"], repository_root=root,
            approved_sha256=baseline_spec["dataset"]["sha256"],
        )
    else:
        dataset = dataset_loader(paths["data"], config["dataset"]["id"])
    measured_dataset_bytes = estimate_phase1_dataset_bytes(
        dataset, [split.train_idx, split.val_idx, split.test_idx],
    )
    memory_ceiling = int(baseline_spec["memory"]["max_dataset_and_indices_bytes"])
    if measured_dataset_bytes > memory_ceiling:
        raise ReproductionValidationError(
            f"Phase 1 dataset/index memory {measured_dataset_bytes} exceeds ceiling {memory_ceiling}"
        )
    default_training = trainer is None
    train = trainer or _default_trainer
    store = ManifestStore(paths["manifest"])
    any_gate_failure = False
    for run in REQUIRED_RUNS:
        reusable = None if force else _find_reusable(
            output_root, run, root=root, config=config, fixed_split=split,
            dataset=dataset, current_code=code, logger=logger,
            baseline_spec=baseline_spec,
        )
        if reusable is not None:
            destination, result = reusable
            logger(f"{run}: safely reusing verified completed run at {_relative(root, destination)}")
            metrics = result["metrics"]
            actual = float(metrics["overall_accuracy"])
            passed, deviation = check_gate(actual, config["historical_targets"][run], config["tolerance_pp"])
            status = "completed" if passed else "gate_failed"
            attempt["runs"][run] = {
                "status": status, "reused": True, "actual": actual,
                "target": float(config["historical_targets"][run]), "deviation_pp": deviation,
                "destination": _relative(root, destination),
                "result_json": _relative(root, destination / "result.json"),
                "result_sha256": file_sha256(destination / "result.json"),
                "result_canonical_hash": result["result_canonical_hash"],
            }
            any_gate_failure |= not passed
            store.upsert(_manifest_record(root, config, split, run, destination, status, f"attempt={attempt_id}; deviation_pp={deviation:.6f}; verified reuse"))
            continue
        destination = output_root / f"{run}_seed{config['seed']}_{attempt_id}"
        _validate_artifact_path(root, output_root, destination, f"{run} run destination")
        destination.mkdir(parents=False, exist_ok=False)
        _failure_context["current_run"] = run
        _failure_context["current_destination"] = destination
        run_spec = {
            "schema_version": 1, "runner_version": RUNNER_VERSION,
            "attempt_id": attempt_id, "run": run, "created_at": _utc_now(),
            "dataset": config["dataset"], "seed": config["seed"], "device": config["device"],
            "training": config["training"], "model": config["models"][run],
            "architecture": config["architecture"],
            "evaluation_batch_size": config["evaluation_batch_size"],
            "model_construction": _model_construction_identity(run, config),
            "provenance": expected | {
                "data_sha256": split.metadata["data_sha256"],
                "baseline_spec_sha256": baseline_digest,
            },
            "baseline_spec": baseline_spec,
            "baseline_spec_sha256": baseline_digest,
            "environment": environment, "code": code,
            "reference_source_hashes": reference_source_hashes,
            "deterministic_settings": {
                "python_random_seed": config["seed"], "numpy_seed": config["seed"],
                "torch_seed": config["seed"],
                "data_order": "global_torch_rng_after_model_initialization_with_train_and_validation_shuffle",
                "torch_deterministic_algorithms": True, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "num_workers": 0,
            },
            "memory": {
                "dataset_and_indices_bytes": measured_dataset_bytes,
                "ceiling_bytes": memory_ceiling,
                "measurement": baseline_spec["memory"]["measurement"],
                "index_backed_views": True,
            },
            "initialization": "fresh_random_no_historical_checkpoint",
            "selection_policy": baseline_spec["training_protocol"],
        }
        _atomic_json(destination / "run_spec.json", run_spec)
        try:
            if default_training:
                outcome = train(run, dataset, split, destination, logger, config=config)
            else:
                outcome = train(run, dataset, split, destination, logger)
            if default_training and outcome.get("model_construction") != run_spec["model_construction"]:
                raise ReproductionValidationError(
                    f"{run} actual model construction does not match run_spec identity"
                )
            checkpoint = Path(outcome["checkpoint"])
            _validate_artifact_path(root, output_root, checkpoint, f"{run} checkpoint artifact")
            if checkpoint.resolve() != (destination / "checkpoint.pt").resolve() or not checkpoint.is_file():
                raise ReproductionValidationError(f"{run} trainer did not write the isolated checkpoint destination")
            num_classes = int(config["architecture"]["num_classes"])
            predictions = _validated_class_indices(outcome["predictions"], "predictions", num_classes)
            labels = _validated_class_indices(outcome["labels"], "labels", num_classes)
            snrs = np.asarray(outcome["snrs"])
            if len(predictions) != len(split.test_idx):
                raise ReproductionValidationError(f"{run} prediction count does not match fixed test split")
            prediction_path = destination / "predictions.npz"
            _write_predictions(prediction_path, outcome, split.sample_ids[split.test_idx])
            metrics = _validated_bundle_metrics(
                prediction_path, split, dataset, num_classes,
            )
            actual = float(metrics["overall_accuracy"])
            target = float(config["historical_targets"][run])
            passed, deviation = check_gate(actual, target, config["tolerance_pp"])
            status = "completed" if passed else "gate_failed"
            any_gate_failure |= not passed
            result = {
                "schema_version": 1, "runner_version": RUNNER_VERSION,
                "attempt_id": attempt_id, "run": run, "status": status,
                "target_accuracy": target, "tolerance_pp": float(config["tolerance_pp"]),
                "deviation_pp": deviation, "metrics": metrics,
                "best_epoch": int(outcome["best_epoch"]),
                "epochs_completed": int(outcome.get("epochs_completed", int(outcome["best_epoch"]) + 1)),
                "best_validation_accuracy": float(outcome["best_val_accuracy"]),
                "duration_seconds": float(outcome["duration_seconds"]),
                "parameter_count": int(outcome["parameter_count"]),
                "history": outcome.get("history", []),
                "provenance": expected | {
                    "data_sha256": split.metadata["data_sha256"],
                    "baseline_spec_sha256": baseline_digest,
                },
                "artifacts": {
                    "checkpoint": _relative(root, checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "predictions": _relative(root, prediction_path),
                    "predictions_sha256": file_sha256(prediction_path),
                    "run_spec": _relative(root, destination / "run_spec.json"),
                    "run_spec_sha256": file_sha256(destination / "run_spec.json"),
                    "run_spec_canonical_hash": _canonical_payload_hash(run_spec),
                    "result_json": _relative(root, destination / "result.json"),
                },
            }
            result["result_canonical_hash"] = _canonical_payload_hash(
                result, omit={"result_canonical_hash"}
            )
            result_path = destination / "result.json"
            _atomic_json(result_path, result)
            attempt["runs"][run] = {
                "status": status, "reused": False, "actual": actual, "target": target,
                "deviation_pp": deviation, "duration_seconds": result["duration_seconds"],
                "destination": _relative(root, destination),
                "result_json": _relative(root, result_path),
                "result_sha256": file_sha256(result_path),
                "result_canonical_hash": result["result_canonical_hash"],
            }
            store.upsert(_manifest_record(root, config, split, run, destination, status, f"attempt={attempt_id}; deviation_pp={deviation:.6f}"))
            _failure_context.pop("current_run", None)
            _failure_context.pop("current_destination", None)
        except Exception:
            raise
    attempt["finished_at"] = _utc_now()
    if any_gate_failure:
        attempt["status"] = "gate_failed"
    else:
        attempt["status"] = "pass"
    if attempt["status"] != "pass":
        _append_failure_report(paths["failure_report"], attempt)
        failed_runs = [name for name, value in attempt["runs"].items() if value["status"] in {"failed", "gate_failed"}]
        update_phase_progress(
            "Phase 1",
            completed="Fixed split persisted and both requested run attempts were retained.",
            failed=f"Reproduction gate did not pass: {', '.join(failed_runs)}.",
            unexpected="Execution errors or deviations beyond 0.5 percentage points are recorded in reports/reproduction_failure.md.",
            interpretation="Phase 2 expansion is blocked; no tuning was performed.",
            next_gate="STOP: audit Phase 1 failure evidence before any new preserved attempt.",
            path=paths["progress"],
        )
        publish_gate_state(
            paths["output_root"], paths["gate_state"], attempt, repository_root=root,
        )
        return 1
    update_phase_progress(
        "Phase 1",
        completed="Both fresh seed-2022 reproductions passed the fixed-split 0.5 percentage-point gate.",
        failed="None.", unexpected="None.",
        interpretation="The persisted split protocol is validated for controlled Phase 2 expansion.",
        next_gate="Phase 2 controlled SNR-conditioning experiments.",
        path=paths["progress"],
    )
    publish_gate_state(
        paths["output_root"], paths["gate_state"], attempt, repository_root=root,
    )
    return 0


def run_phase1(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
    dataset_loader: Callable[..., Any] | None = None,
    force: bool = False,
    logger: Callable[[str], None] = print,
    baseline_spec: dict[str, Any] | None = None,
    lock_timeout_seconds: float = 10.0,
) -> int:
    root = Path(repository_root) if repository_root is not None else REPOSITORY_ROOT
    baseline = BASELINE_SPEC if baseline_spec is None else baseline_spec
    context: dict[str, Any] = {"root": root, "logger": logger}
    try:
        lock_root = _repo_path(root, baseline["paths"]["output_root"], "execution lock root")
    except Exception as exc:
        logger(f"Phase 1 unsafe execution-lock path: {exc}")
        return 1
    lock = PhaseExecutionLock(lock_root / ".phase1.lock", timeout_seconds=lock_timeout_seconds)
    try:
        lock.acquire()
    except ReproductionValidationError as exc:
        logger(str(exc))
        return 3
    try:
        try:
            return _run_phase1_impl(
                config_path, repository_root=root, trainer=trainer,
                dataset_loader=dataset_loader, force=force, logger=logger,
                baseline_spec=baseline, _failure_context=context,
            )
        except Exception as exc:
            _finalize_failure(context, exc)
            return 1
    finally:
        lock.release()


def validate_only(
    config_path: str | Path, repository_root: str | Path | None = None,
    *, baseline_spec: dict[str, Any] | None = None,
) -> FixedSplit:
    root = Path(repository_root) if repository_root is not None else REPOSITORY_ROOT
    baseline = BASELINE_SPEC if baseline_spec is None else baseline_spec
    validated_config_path = _validated_config_path(root, config_path, baseline)
    config = load_reproduction_config(validated_config_path)
    paths = validate_config(config, root, baseline_spec=baseline)
    if config["dataset"]["id"] == "RML2016.10a":
        validate_conditioned_dependency(root)
    split = load_split(paths["split_metadata"], npz_path=paths["split_npz"], data_path=paths["data"])
    if paths["output_root"].exists() and not paths["output_root"].is_dir():
        raise ReproductionValidationError("outputs.root exists but is not a directory")
    parent = paths["output_root"].parent
    if not parent.is_dir():
        raise ReproductionValidationError(f"checkpoint destination parent does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise ReproductionValidationError(f"checkpoint destination parent is not writable: {parent}")
    return split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2/reproduction.yaml")
    parser.add_argument("--validate-only", action="store_true", help="validate config, data, split, and isolated output destinations without training")
    parser.add_argument("--force", action="store_true", help="start a new preserved attempt instead of reusing verified completed outputs")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            split = validate_only(args.config)
            print(
                f"VALID: dataset={split.dataset_id} total={split.total_count} "
                f"train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)} "
                f"split_hash={split.split_hash}"
            )
            return 0
        return run_phase1(args.config, force=args.force)
    except (ReproductionValidationError, SplitValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

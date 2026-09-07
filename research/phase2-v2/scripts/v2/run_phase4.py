"""Strict Phase 4 collapse diagnostics over registered Phase 3 predictions.

This runner never trains a model.  Prediction diagnostics are recomputed from
hash-bound Phase 3 bundles.  Representation geometry is published only from
real feature bundles produced through the injected feature-extractor hook.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.metrics import class_separability, effective_rank, top_singular_value_share
from v2.provenance import deterministic_json_hash, environment_fingerprint, file_sha256


RUNNER_VERSION = "phase4-v2.0"
PHASE3_RUNNER_VERSION = "phase3-conditioning/1"
MODELS = tuple(f"M{index}" for index in range(8))
SEEDS = (2022, 2023, 2024, 2025, 2026)
SNRS = tuple(range(-20, 20, 2))
NUM_CLASSES = 11
PREDICTION_FIELDS = {
    "y_true", "y_pred", "logits", "snr_db", "snr_bin", "sample_ids",
    "seed", "model_id", "split_hash", "preprocessing_hash", "protocol_hash",
}
FEATURE_FIELDS = {
    "features", "replayed_logits", "sample_ids", "y_true", "snr_db", "snr_bin",
    "seed", "model_id", "split_hash", "preprocessing_hash", "protocol_hash",
    "predictions_sha256", "checkpoint_sha256", "hook_id", "binding_hash",
    "extractor_code_sha256", "model_code_sha256", "environment_sha256",
}
PHASE4_SOURCE_FILES = (
    "scripts/v2/run_phase4.py", "scripts/v2/run_phase3.py", "models/model_conditioning.py",
    "models/model.py", "models/lifting.py", "v2/metrics.py", "v2/provenance.py",
    "v2/splits.py", "v2/contracts.py", "v2/progress.py", "v2/manifest.py",
)


class UpstreamValidationError(ValueError):
    """Raised when Phase 3 is incomplete or its registered evidence is invalid."""

    def __init__(self, message: str, *, completed_runs: int = 0) -> None:
        super().__init__(message)
        self.completed_runs = completed_runs


@dataclass(frozen=True)
class RegisteredRun:
    model_id: str
    seed: int
    run_dir: Path
    result_path: Path
    predictions_path: Path
    checkpoint_path: Path
    result_sha256: str
    predictions_sha256: str
    checkpoint_sha256: str
    run_fingerprint: str
    split_hash: str
    preprocessing_hash: str
    protocol_hash: str
    overall_accuracy: float
    best_val_accuracy: float
    additional_conditioner_parameters: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    names = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _portable_path(root: Path, value: Any, field: str, prefix: str | None = None) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{field} must be a non-empty repository-relative portable path")
    normalized = Path(value).as_posix()
    if prefix is not None and not (normalized == prefix or normalized.startswith(prefix + "/")):
        raise ValueError(f"{field} must remain under {prefix}")
    resolved = (root / normalized).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{field} escapes the repository")
    return resolved


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_phase4_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_phase4_config(value)
    return value


def validate_phase4_config(config: Any) -> None:
    top = {"schema_version", "source", "expected", "metrics", "threshold_sensitivity", "features", "outputs"}
    if not isinstance(config, dict) or set(config) != top or config.get("schema_version") != 1:
        raise ValueError("Phase 4 config must match schema version 1 exactly")
    if config["source"] != {
        "phase3_root": "results/v2/phase3",
        "phase3_ledger": "results/v2/phase3/phase3_ledger.json",
        "phase3_manifest": "manifest.json",
        "phase3_config": "configs/v2/phase3.yaml",
    }:
        raise ValueError("Phase 3 source paths differ from the registered protocol")
    if config["expected"] != {"models": list(MODELS), "seeds": list(SEEDS), "snr_values_db": list(SNRS), "num_classes": NUM_CLASSES}:
        raise ValueError("Phase 4 expected model, seed, SNR, or class universe is not locked")
    expected_metrics = {
        "gap_definition": "prediction_concentration - balanced_accuracy",
        "normalized_entropy_base": NUM_CLASSES,
        "ece": {"type": "top_label_equal_width", "bins": 15, "interval": "[0,1]", "last_bin_right_inclusive": True},
        "distribution_indices": ["hhi", "gini"],
    }
    if config["metrics"] != expected_metrics:
        raise ValueError("Phase 4 metric semantics differ from the preregistration")
    thresholds = config["threshold_sensitivity"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"concentration_cutoffs", "ba_gap_cutoffs", "snr_ranges"}:
        raise ValueError("threshold sensitivity schema is invalid")
    for field in ("concentration_cutoffs", "ba_gap_cutoffs"):
        values = thresholds[field]
        if not isinstance(values, list) or len(values) < 3 or len(set(values)) != len(values) or any(type(value) not in (int, float) or not 0 < value < 1 for value in values):
            raise ValueError(f"{field} must contain at least three unique cutoffs in (0,1)")
    expected_ranges = {
        "severe_low_le_minus12": [-20, -18, -16, -14, -12],
        "low_le_minus10": [-20, -18, -16, -14, -12, -10],
        "registered_low_le_minus8": [-20, -18, -16, -14, -12, -10, -8],
    }
    if thresholds["snr_ranges"] != expected_ranges:
        raise ValueError("SNR sensitivity ranges differ from the preregistered dataset bins")
    expected_features = {
        "always_models": ["M0", "M3"],
        "stronger_candidates": ["M4", "M5", "M6", "M7"],
        "stronger_selection_metric": "best_val_accuracy",
        "hook_id": "final_conditioned_classifier_input_v1",
        "hook_semantics": "final classifier input after the declared conditioner; M0 uses pooled backbone features",
    }
    if config["features"] != expected_features:
        raise ValueError("feature extraction scope or hook semantics differ from the preregistration")
    if config["outputs"] != {"root": "results/v2/phase4_collapse_metrics"}:
        raise ValueError("Phase 4 output root is locked")


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamValidationError(f"{description} cannot be read: {exc}") from exc


def _load_repository_module(path: Path, name: str):
    if not path.is_file():
        raise UpstreamValidationError(f"strict Phase 3 dependency is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise UpstreamValidationError(f"strict Phase 3 dependency cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_phase3_contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run the Phase 3 resume verifier for every declared model×seed.

    This deliberately delegates evidence semantics to the Phase 3 runner rather
    than maintaining a weaker Phase 4 facsimile.  Its verifier strict-loads each
    checkpoint, recomputes fixed-test predictions and metrics, validates the
    mechanism bundle, code/environment identity, and Phase 2 reuse provenance.
    """

    try:
        phase3 = _load_repository_module(root / "scripts/v2/run_phase3.py", "_phase4_phase3_contract")
        phase1 = _load_repository_module(root / "scripts/v2/phase1_reproduce.py", "_phase4_phase1_dataset")
        phase3_path = _portable_path(root, config["source"]["phase3_config"], "source.phase3_config", "configs/v2")
        phase3_config = phase3.load_phase3_config(phase3_path)
        paths = phase3.validate_output_paths(root, phase3_config["outputs"])
        from v2.splits import load_split
        split = load_split(root / phase3_config["split"]["metadata"], data_path=root / phase3_config["dataset"]["path"])
        if split.split_hash != phase3.LOCKED_SPLIT_HASH:
            raise ValueError("fixed split hash differs from the Phase 3 contract")
        dataset = phase1._load_rml_dataset(root / phase3_config["dataset"]["path"], phase3_config["dataset"]["id"], repository_root=root)
        code = phase3._code_identity(root)
        environment = phase3._environment_identity()
        protocol = phase3.protocol_hash(phase3_config)
        preprocessing = phase3.preprocessing_hash(phase3_config)
        ledger = phase3._load_ledger(paths["root"] / "phase3_ledger.json")
        verified: dict[tuple[str, int], Path] = {}
        for seed in phase3_config["seeds"]:
            for model_id in phase3.MODEL_IDS:
                run_identity = {
                    "runner": phase3.RUNNER_VERSION, "protocol_hash": protocol,
                    "preprocessing_hash": preprocessing, "split_hash": split.split_hash,
                    "model_id": model_id, "seed": seed, "code": code, "environment": environment,
                }
                fingerprint = deterministic_json_hash(run_identity)
                identity = {
                    "model_id": model_id, "seed": seed, "split_hash": split.split_hash,
                    "preprocessing_hash": preprocessing, "protocol_hash": protocol,
                }
                candidate = phase3._matching_completed(
                    paths["root"], model_id, seed, fingerprint, root=root, ledger=ledger,
                    split=split, dataset=dataset, identity=identity, config=phase3_config,
                    code=code, environment=environment, completed_verifier=None,
                    phase2_source_validator=None, logger=lambda _message: None,
                )
                if candidate is None:
                    raise UpstreamValidationError(
                        f"strict Phase 3 resume verification failed for {model_id}/seed{seed}",
                        completed_runs=len(verified),
                    )
                verified[(model_id, seed)] = candidate.resolve()
        return {
            "module": phase3, "config": phase3_config, "split": split,
            "dataset": dataset, "code": code, "environment": environment,
            "verified": verified, "protocol_hash": protocol,
            "preprocessing_hash": preprocessing,
        }
    except UpstreamValidationError:
        raise
    except Exception as exc:
        raise UpstreamValidationError(f"strict Phase 3 contract validation failed: {exc}") from exc


def _phase3_protocol(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
    path = _portable_path(root, config["source"]["phase3_config"], "source.phase3_config", "configs/v2")
    phase3 = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(phase3, dict):
        raise UpstreamValidationError("Phase 3 config is not an object")
    try:
        if phase3["seeds"] != list(SEEDS) or tuple(phase3["models"]) != MODELS:
            raise UpstreamValidationError("Phase 3 model/seed plan differs from Phase 4 preregistration")
        if phase3["snr"]["values_db"] != list(SNRS) or phase3["architecture"]["num_classes"] != NUM_CLASSES:
            raise UpstreamValidationError("Phase 3 SNR/class protocol differs from Phase 4 preregistration")
        split_hash = phase3["split"]["hash"]
        protocol_hash = deterministic_json_hash({key: value for key, value in phase3.items() if key != "outputs"})
        preprocessing_hash = deterministic_json_hash(phase3["preprocessing"])
    except KeyError as exc:
        raise UpstreamValidationError(f"Phase 3 config lacks required field {exc}") from exc
    if not all(_is_sha256(value) for value in (split_hash, protocol_hash, preprocessing_hash)):
        raise UpstreamValidationError("Phase 3 protocol identities are malformed")
    return phase3, split_hash, preprocessing_hash, protocol_hash


def _scalar(bundle: dict[str, np.ndarray], field: str) -> Any:
    value = bundle[field]
    if value.ndim != 0:
        raise UpstreamValidationError(f"prediction identity {field} must be scalar")
    return value.item()


def _validate_prediction(path: Path, *, model_id: str, seed: int, split_hash: str, preprocessing_hash: str, protocol_hash: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != PREDICTION_FIELDS:
                raise UpstreamValidationError("prediction bundle field schema is invalid")
            values = {field: bundle[field] for field in bundle.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, UpstreamValidationError):
            raise
        raise UpstreamValidationError(f"prediction bundle cannot be read: {exc}") from exc
    identity = {"model_id": model_id, "seed": seed, "split_hash": split_hash, "preprocessing_hash": preprocessing_hash, "protocol_hash": protocol_hash}
    for field, expected in identity.items():
        if _scalar(values, field) != expected:
            raise UpstreamValidationError(f"prediction identity {field} does not match its registered run")
    y_true = np.asarray(values["y_true"])
    y_pred = np.asarray(values["y_pred"])
    logits = np.asarray(values["logits"])
    snr_db = np.asarray(values["snr_db"])
    snr_bin = np.asarray(values["snr_bin"])
    sample_ids = np.asarray(values["sample_ids"])
    count = len(y_true)
    if y_true.ndim != 1 or y_pred.shape != (count,) or snr_db.shape != (count,) or snr_bin.shape != (count,) or sample_ids.shape != (count,) or logits.shape != (count, NUM_CLASSES) or not count:
        raise UpstreamValidationError("prediction bundle row shapes are invalid")
    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(y_pred.dtype, np.integer) or np.any(y_true < 0) or np.any(y_true >= NUM_CLASSES):
        raise UpstreamValidationError("prediction labels are invalid")
    if not np.isfinite(logits).all() or not np.isfinite(snr_db).all() or len(np.unique(sample_ids)) != count:
        raise UpstreamValidationError("prediction logits, SNR, or sample IDs are invalid")
    if not np.array_equal(y_pred.astype(np.int64), logits.argmax(1).astype(np.int64)):
        raise UpstreamValidationError("prediction labels do not equal argmax(logits)")
    lookup = {float(value): index for index, value in enumerate(SNRS)}
    try:
        expected_bins = np.asarray([lookup[float(value)] for value in snr_db], dtype=np.int64)
    except KeyError as exc:
        raise UpstreamValidationError(f"prediction contains undeclared true SNR {exc}") from exc
    if not np.issubdtype(snr_bin.dtype, np.integer):
        raise UpstreamValidationError("prediction snr_bin must have an integer dtype")
    if not np.array_equal(snr_bin.astype(np.int64), expected_bins) or set(float(value) for value in np.unique(snr_db)) != set(SNRS):
        raise UpstreamValidationError("prediction true-SNR/bin binding is invalid or incomplete")
    return values


def audit_phase3_registry(repository_root: str | Path, config: dict[str, Any], *, _strict_context: dict[str, Any] | None = None) -> list[RegisteredRun]:
    root = Path(repository_root).resolve()
    strict = _strict_context if _strict_context is not None else _strict_phase3_contract(root, config)
    phase3, split_hash, preprocessing_hash, protocol_hash = _phase3_protocol(root, config)
    phase3_root = _portable_path(root, config["source"]["phase3_root"], "source.phase3_root", "results/v2/phase3")
    ledger_path = _portable_path(root, config["source"]["phase3_ledger"], "source.phase3_ledger", "results/v2/phase3")
    manifest_path = _portable_path(root, config["source"]["phase3_manifest"], "source.phase3_manifest")
    ledger = _load_json(ledger_path, "Phase 3 ledger")
    if not isinstance(ledger, dict):
        raise UpstreamValidationError("Phase 3 ledger schema is invalid")
    records = ledger.get("records")
    if set(ledger) != {"schema_version", "runner_version", "records"} or ledger.get("schema_version") != 1 or ledger.get("runner_version") != PHASE3_RUNNER_VERSION or not isinstance(records, list):
        raise UpstreamValidationError("Phase 3 ledger schema is invalid")
    completed = [record for record in records if isinstance(record, dict) and record.get("status") == "completed"]
    expected_pairs = {(model, seed) for model in MODELS for seed in SEEDS}
    completed_pairs = [(record.get("model_id"), record.get("seed")) for record in completed]
    if len(completed) != 40 or set(completed_pairs) != expected_pairs or len(set(completed_pairs)) != len(completed_pairs):
        raise UpstreamValidationError(f"Phase 3 requires exactly 40 unique completed model×seed records; found {len(set(completed_pairs) & expected_pairs)}", completed_runs=len(set(completed_pairs) & expected_pairs))
    manifest = _load_json(manifest_path, "Phase 3 manifest")
    if not isinstance(manifest, dict):
        raise UpstreamValidationError("Phase 3 manifest schema is invalid", completed_runs=40)
    experiments = manifest.get("experiments")
    if manifest.get("schema_version") != 1 or not isinstance(experiments, list):
        raise UpstreamValidationError("Phase 3 manifest schema is invalid", completed_runs=40)
    manifest_index: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in experiments:
        if isinstance(entry, dict) and entry.get("experiment") == "phase3" and entry.get("status") == "completed":
            key = (entry.get("model"), entry.get("seed"))
            if key in manifest_index:
                raise UpstreamValidationError("Phase 3 manifest duplicates a completed model×seed", completed_runs=40)
            manifest_index[key] = entry
    if set(manifest_index) != expected_pairs:
        raise UpstreamValidationError("Phase 3 manifest does not register all 40 completed runs", completed_runs=40)

    audited: list[RegisteredRun] = []
    canonical_rows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    ledger_fields = {"model_id", "seed", "run_fingerprint", "attempt_path", "result_sha256", "status", "updated_at"}
    for record in sorted(completed, key=lambda item: (item["model_id"], item["seed"])):
        if set(record) != ledger_fields or not _is_sha256(record["result_sha256"]) or not _is_sha256(record["run_fingerprint"]):
            raise UpstreamValidationError("Phase 3 completed ledger record schema is invalid", completed_runs=40)
        model_id, seed = record["model_id"], record["seed"]
        run_dir = _portable_path(root, record["attempt_path"], "ledger.attempt_path", phase3_root.relative_to(root).as_posix())
        if strict["verified"].get((model_id, seed)) != run_dir.resolve():
            raise UpstreamValidationError("Phase 3 ledger does not select the strictly verified attempt", completed_runs=40)
        result_path = run_dir / "result.json"
        if not result_path.is_file() or file_sha256(result_path) != record["result_sha256"]:
            raise UpstreamValidationError("Phase 3 result hash differs from its ledger registration", completed_runs=40)
        result = _load_json(result_path, "Phase 3 result")
        if result.get("status") != "completed" or result.get("model_id") != model_id or result.get("seed") != seed or result.get("runner_version") != PHASE3_RUNNER_VERSION:
            raise UpstreamValidationError("Phase 3 completed result identity is invalid", completed_runs=40)
        provenance = result.get("provenance", {})
        if provenance.get("run_fingerprint") != record["run_fingerprint"] or provenance.get("split_hash") != split_hash or provenance.get("preprocessing_hash") != preprocessing_hash or provenance.get("protocol_hash") != protocol_hash:
            raise UpstreamValidationError("Phase 3 result provenance differs from the locked protocol", completed_runs=40)
        run_spec_path = run_dir / "run_spec.json"
        spec = _load_json(run_spec_path, "Phase 3 run specification")
        if spec.get("model_id") != model_id or spec.get("seed") != seed or spec.get("run_fingerprint") != record["run_fingerprint"] or spec.get("protocol_hash") != protocol_hash or spec.get("preprocessing_hash") != preprocessing_hash or spec.get("protocol_config") != phase3 or spec.get("fixed_split") != {"metadata": phase3["split"]["metadata"], "hash": split_hash}:
            raise UpstreamValidationError("Phase 3 run specification is not canonically bound", completed_runs=40)
        artifacts = result.get("artifacts")
        required_artifacts = {"run_spec": "run_spec.json", "checkpoint": "checkpoint.pt", "predictions": "predictions.npz", "mechanism": "mechanism.npz"}
        if not isinstance(artifacts, dict) or set(artifacts) != set(required_artifacts) | {f"{name}_sha256" for name in required_artifacts}:
            raise UpstreamValidationError("Phase 3 artifact schema is invalid", completed_runs=40)
        paths = {name: run_dir / filename for name, filename in required_artifacts.items()}
        for name, filename in required_artifacts.items():
            if artifacts[name] != filename or not paths[name].is_file() or artifacts[f"{name}_sha256"] != file_sha256(paths[name]):
                raise UpstreamValidationError(f"Phase 3 {name} artifact hash is invalid", completed_runs=40)
        prediction_values = _validate_prediction(paths["predictions"], model_id=model_id, seed=seed, split_hash=split_hash, preprocessing_hash=preprocessing_hash, protocol_hash=protocol_hash)
        rows = tuple(np.asarray(prediction_values[name]) for name in ("sample_ids", "y_true", "snr_db", "snr_bin"))
        if canonical_rows is None:
            canonical_rows = rows
        elif any(not np.array_equal(left, right) for left, right in zip(canonical_rows, rows)):
            raise UpstreamValidationError("Phase 3 prediction bundles violate fixed-split row binding", completed_runs=40)
        manifest_entry = manifest_index[(model_id, seed)]
        if manifest_entry.get("dataset") != "RML2016.10a" or manifest_entry.get("split_hash") != split_hash or manifest_entry.get("result_json") != result_path.relative_to(root).as_posix() or manifest_entry.get("checkpoint") != paths["checkpoint"].relative_to(root).as_posix():
            raise UpstreamValidationError("Phase 3 manifest artifact binding is invalid", completed_runs=40)
        logits = np.asarray(prediction_values["logits"])
        labels = np.asarray(prediction_values["y_true"])
        overall_accuracy = float(np.mean(logits.argmax(1) == labels))
        stored_accuracy = result.get("metrics", {}).get("overall_accuracy")
        if type(stored_accuracy) not in (int, float) or not math.isclose(float(stored_accuracy), overall_accuracy, rel_tol=1e-12, abs_tol=1e-12):
            raise UpstreamValidationError("Phase 3 stored overall accuracy differs from predictions", completed_runs=40)
        audited.append(RegisteredRun(
            model_id=model_id, seed=seed, run_dir=run_dir, result_path=result_path,
            predictions_path=paths["predictions"], checkpoint_path=paths["checkpoint"],
            result_sha256=record["result_sha256"], predictions_sha256=artifacts["predictions_sha256"],
            checkpoint_sha256=artifacts["checkpoint_sha256"], run_fingerprint=record["run_fingerprint"],
            split_hash=split_hash, preprocessing_hash=preprocessing_hash, protocol_hash=protocol_hash,
            overall_accuracy=overall_accuracy,
            best_val_accuracy=float(result.get("training", {}).get("best_val_accuracy", float("nan"))),
            additional_conditioner_parameters=int(result.get("complexity", {}).get("additional_conditioner_parameters", -1)),
        ))
    return audited


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def compute_continuous_metrics(y_true: Any, logits: Any, *, num_classes: int, ece_bins: int) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != (len(labels), num_classes) or not len(labels) or not np.isfinite(scores).all() or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("continuous metrics require finite N-by-class logits and valid non-empty labels")
    probabilities = _softmax(scores)
    predictions = scores.argmax(1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support = matrix.sum(1)
    observed = support > 0
    recalls = np.divide(np.diag(matrix), support, out=np.zeros(num_classes, dtype=float), where=observed)
    balanced = float(np.mean(recalls[observed]))
    counts = matrix.sum(0)
    shares = counts / len(labels)
    concentration = float(shares.max())
    positive = shares[shares > 0]
    normalized_entropy = -float(np.sum(positive * np.log(positive)) / np.log(num_classes)) if num_classes > 1 else 0.0
    hhi = float(np.sum(shares**2))
    gini = float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * num_classes))
    confidence = probabilities.max(1)
    correct = predictions == labels
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, ece_bins + 1)
    for index in range(ece_bins):
        mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1] if index < ece_bins - 1 else confidence <= 1.0)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "sample_count": int(len(labels)),
        "prediction_concentration": concentration,
        "balanced_accuracy": balanced,
        "concentration_minus_balanced_accuracy": float(concentration - balanced),
        "normalized_entropy": normalized_entropy,
        "hhi": hhi,
        "gini": gini,
        "ece": float(ece),
    }


def _continuous_rows(runs: list[RegisteredRun], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ece_bins = int(config["metrics"]["ece"]["bins"])
    for run in runs:
        with np.load(run.predictions_path, allow_pickle=False) as bundle:
            labels = bundle["y_true"]
            logits = bundle["logits"]
            snr_db = bundle["snr_db"]
        for snr in SNRS:
            mask = snr_db == snr
            metric = compute_continuous_metrics(labels[mask], logits[mask], num_classes=NUM_CLASSES, ece_bins=ece_bins)
            rows.append({"model_id": run.model_id, "seed": run.seed, "snr_db": snr, **metric, "predictions_sha256": run.predictions_sha256})
    return rows


def _threshold_rows(continuous: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold = config["threshold_sensitivity"]
    rows: list[dict[str, Any]] = []
    by_model = {model: [row for row in continuous if row["model_id"] == model] for model in MODELS}
    for range_name, values in threshold["snr_ranges"].items():
        allowed = set(values)
        for concentration_cutoff in threshold["concentration_cutoffs"]:
            for gap_cutoff in threshold["ba_gap_cutoffs"]:
                rates: dict[str, float] = {}
                for model_id in MODELS:
                    selected = [row for row in by_model[model_id] if row["snr_db"] in allowed]
                    collapsed = [row["prediction_concentration"] >= concentration_cutoff and row["concentration_minus_balanced_accuracy"] >= gap_cutoff for row in selected]
                    rates[model_id] = float(np.mean(collapsed))
                plain = rates["M0"]
                for model_id in MODELS:
                    rate = rates[model_id]
                    direction = "equal" if math.isclose(rate, plain, abs_tol=1e-12) else ("lower" if rate < plain else "higher")
                    rows.append({
                        "model_id": model_id, "snr_range": range_name,
                        "concentration_cutoff": concentration_cutoff, "ba_gap_cutoff": gap_cutoff,
                        "evaluated_model_seed_snr_cells": len([row for row in by_model[model_id] if row["snr_db"] in allowed]),
                        "collapse_rate": rate, "plain_collapse_rate": plain,
                        "difference_vs_plain": rate - plain, "direction_vs_plain": direction,
                    })
    dependencies = []
    for model_id in MODELS[1:]:
        for range_name in threshold["snr_ranges"]:
            observed = {row["direction_vs_plain"] for row in rows if row["model_id"] == model_id and row["snr_range"] == range_name}
            order = {"equal": 0, "lower": 1, "higher": 2}
            directions = sorted(observed, key=order.__getitem__)
            dependencies.append({"model_id": model_id, "snr_range": range_name, "stable_direction": len(observed) <= 1, "directions": directions})
    stable = all(item["stable_direction"] for item in dependencies)
    return rows, {"direction_stable_across_grid": stable, "conclusion": "continuous conclusions are directionally stable across the preregistered grid" if stable else "binary collapse conclusions depend on threshold choice; a universal collapse threshold is unsupported", "comparisons": dependencies}


def _stronger_model(runs: list[RegisteredRun]) -> tuple[str, dict[str, Any]]:
    candidates = []
    for model in ("M4", "M5", "M6", "M7"):
        selected = [run for run in runs if run.model_id == model]
        if len(selected) != len(SEEDS):
            raise UpstreamValidationError(f"validation-only selector requires five completed seeds for {model}")
        scores = [run.best_val_accuracy for run in selected]
        parameters = [run.additional_conditioner_parameters for run in selected]
        if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores) or any(value < 0 for value in parameters):
            raise UpstreamValidationError(f"validation-only selection evidence is invalid for {model}")
        candidates.append({
            "model_id": model,
            "seed_scores": {str(run.seed): run.best_val_accuracy for run in sorted(selected, key=lambda item: item.seed)},
            "mean_best_val_accuracy": float(np.mean(scores)),
            "mean_additional_conditioner_parameters": float(np.mean(parameters)),
        })
    ranked = sorted(candidates, key=lambda row: (-row["mean_best_val_accuracy"], row["mean_additional_conditioner_parameters"], row["model_id"]))
    selected = ranked[0]["model_id"]
    evidence = {
        "selection_partition": "validation_only",
        "selection_metric": "training.best_val_accuracy",
        "candidates": sorted(candidates, key=lambda row: row["model_id"]),
        "tie_break": ["mean_best_val_accuracy_desc", "mean_additional_conditioner_parameters_asc", "model_id_asc"],
        "selected_model": selected,
    }
    return selected, evidence


def _source_identity(root: Path) -> dict[str, Any]:
    files = {name: file_sha256(root / name) for name in PHASE4_SOURCE_FILES}
    return {"files": files, "hash": deterministic_json_hash(files)}


def _feature_binding(run: RegisteredRun, config: dict[str, Any], *, source: dict[str, Any], environment: dict[str, Any]) -> str:
    return deterministic_json_hash({
        "model_id": run.model_id, "seed": run.seed, "split_hash": run.split_hash,
        "preprocessing_hash": run.preprocessing_hash, "protocol_hash": run.protocol_hash,
        "predictions_sha256": run.predictions_sha256,
        "checkpoint_sha256": run.checkpoint_sha256, "hook_id": config["features"]["hook_id"],
        "hook_semantics": config["features"]["hook_semantics"], "source": source,
        "environment": environment,
    })


def _validate_feature_bundle(
    path: Path, run: RegisteredRun, config: dict[str, Any], *, source: dict[str, Any],
    environment: dict[str, Any], expected_logits: np.ndarray | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise ValueError("registered feature bundle is missing")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != FEATURE_FIELDS:
            raise ValueError("feature bundle schema is invalid")
        values = {field: bundle[field] for field in bundle.files}
    with np.load(run.predictions_path, allow_pickle=False) as prediction:
        expected_rows = {field: prediction[field] for field in ("sample_ids", "y_true", "snr_db", "snr_bin")}
        stored_logits = np.asarray(prediction["logits"], dtype=np.float32)
    features = np.asarray(values["features"], dtype=np.float64)
    expected_dimension = 136 if run.model_id == "M3" else 128
    if features.shape != (len(expected_rows["y_true"]), expected_dimension) or not np.isfinite(features).all():
        raise ValueError("feature matrix must be finite, non-empty, two-dimensional, and row-bound")
    for field, expected in expected_rows.items():
        if not np.array_equal(values[field], expected):
            raise ValueError(f"feature bundle {field} violates prediction row binding")
    identity = {
        "seed": run.seed, "model_id": run.model_id, "split_hash": run.split_hash,
        "preprocessing_hash": run.preprocessing_hash, "protocol_hash": run.protocol_hash,
        "predictions_sha256": run.predictions_sha256,
        "checkpoint_sha256": run.checkpoint_sha256, "hook_id": config["features"]["hook_id"],
        "binding_hash": _feature_binding(run, config, source=source, environment=environment),
        "extractor_code_sha256": source["hash"],
        "model_code_sha256": source["files"]["models/model_conditioning.py"],
        "environment_sha256": deterministic_json_hash(environment),
    }
    for field, expected in identity.items():
        if values[field].ndim != 0 or values[field].item() != expected:
            raise ValueError(f"feature bundle identity {field} is invalid")
    replayed = np.asarray(values["replayed_logits"], dtype=np.float32)
    if not _matching_logits(replayed, stored_logits):
        raise ValueError("feature classifier replay logits differ from registered Phase 3 predictions")
    if expected_logits is not None and not _matching_logits(replayed, np.asarray(expected_logits, dtype=np.float32)):
        raise ValueError("feature classifier replay is not reproducible from the loaded checkpoint")
    return features


def _matching_logits(actual: np.ndarray, expected: np.ndarray) -> bool:
    # CPU kernels may round differently across evaluation batch sizes. Require
    # tight numerical agreement AND identical decisions, never just accuracy.
    return bool(actual.shape == expected.shape and np.isfinite(actual).all()
                and np.isfinite(expected).all()
                and np.allclose(actual, expected, rtol=1e-6, atol=1e-7)
                and np.array_equal(actual.argmax(1), expected.argmax(1)))


def _replay_conditioned_features(model: Any, features: Any, snr_bin: Any):
    """Replay only the classifier path downstream of the declared hook."""
    if model.conditioning == "M4":
        return model.fc(features) + model.logit_bias(snr_bin)
    if model.conditioning == "M7":
        logits = features.new_zeros((features.shape[0], model.num_classes))
        for selected_bin in snr_bin.unique(sorted=True):
            indices = (snr_bin == selected_bin).nonzero(as_tuple=False).flatten()
            logits = logits.index_copy(0, indices, model.snr_heads[int(selected_bin.item())](features.index_select(0, indices)))
        return logits
    return model.fc(features)


def extract_checkpoint_features(
    run: RegisteredRun, *, phase3_config: dict[str, Any], dataset: dict[str, Any],
    split: Any, root: Path, phase4_config: dict[str, Any], source: dict[str, Any], environment: dict[str, Any],
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Strict-load one checkpoint and replay logits from its declared feature layer."""
    import torch
    from models.model_conditioning import AWNConditioned

    architecture = phase3_config["architecture"]
    model = AWNConditioned(
        **{key: architecture[key] for key in (
            "num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim",
            "regu_details", "regu_approx", "num_snr_bins", "snr_embedding_dim",
        )},
        conditioning=run.model_id, snr_min_db=float(phase3_config["snr"]["snr_min_db"]),
        snr_max_db=float(phase3_config["snr"]["snr_max_db"]),
    ).cpu().eval()
    try:
        state = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(run.checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    signals = np.asarray(dataset["signals"])[test_idx]
    labels = np.asarray(dataset["labels"])[test_idx]
    true_snr = np.asarray(dataset["snrs"])[test_idx]
    with np.load(run.predictions_path, allow_pickle=False) as prediction:
        rows = {name: prediction[name] for name in ("sample_ids", "y_true", "snr_db", "snr_bin")}
        expected_logits = np.asarray(prediction["logits"], dtype=np.float32)
    expected_ids = np.asarray(split.sample_ids)[test_idx]
    if not np.array_equal(rows["sample_ids"], expected_ids) or not np.array_equal(rows["y_true"], labels) or not np.array_equal(rows["snr_db"], true_snr):
        raise UpstreamValidationError("feature extraction rows differ from the fixed Phase 3 test split")
    bins = np.asarray(rows["snr_bin"])
    if not np.issubdtype(bins.dtype, np.integer):
        raise UpstreamValidationError("feature extraction requires integer snr_bin values")
    size = int(batch_size or phase3_config["evaluation_batch_size"])
    if size < 1:
        raise ValueError("feature extraction batch_size must be positive")
    feature_parts = []
    replay_parts = []
    with torch.no_grad():
        for start in range(0, len(test_idx), size):
            stop = min(start + size, len(test_idx))
            x = torch.as_tensor(signals[start:stop], dtype=torch.float32)
            snr_db = torch.as_tensor(true_snr[start:stop], dtype=torch.float32)
            snr_bin = torch.as_tensor(bins[start:stop], dtype=torch.long)
            pooled, _regularizers = model.extract_pooled_features(x)
            conditioned = model.apply_feature_conditioning(pooled, snr_bin=snr_bin, snr_db=snr_db)
            replay = _replay_conditioned_features(model, conditioned, snr_bin)
            feature_parts.append(conditioned.cpu().numpy().astype(np.float32, copy=False))
            replay_parts.append(replay.cpu().numpy().astype(np.float32, copy=False))
    features = np.concatenate(feature_parts, axis=0)
    replayed = np.concatenate(replay_parts, axis=0)
    expected_dimension = int(model.out_channels + (model.snr_embedding_dim if run.model_id == "M3" else 0))
    if features.shape != (len(test_idx), expected_dimension) or expected_dimension != (136 if run.model_id == "M3" else 128):
        raise UpstreamValidationError("checkpoint model violates the declared feature dimension contract")
    if not np.isfinite(features).all() or not _matching_logits(replayed, expected_logits):
        raise UpstreamValidationError("checkpoint-derived feature replay differs from the registered prediction logits")
    binding = _feature_binding(run, phase4_config, source=source, environment=environment)
    return {
        "features": features, "replayed_logits": replayed,
        "sample_ids": rows["sample_ids"], "y_true": rows["y_true"],
        "snr_db": rows["snr_db"], "snr_bin": rows["snr_bin"],
        "seed": np.asarray(run.seed, dtype=np.int64), "model_id": np.asarray(run.model_id),
        "split_hash": np.asarray(run.split_hash), "preprocessing_hash": np.asarray(run.preprocessing_hash),
        "protocol_hash": np.asarray(run.protocol_hash), "predictions_sha256": np.asarray(run.predictions_sha256),
        "checkpoint_sha256": np.asarray(run.checkpoint_sha256),
        "hook_id": np.asarray(phase4_config["features"]["hook_id"]),
        "binding_hash": np.asarray(binding), "extractor_code_sha256": np.asarray(source["hash"]),
        "model_code_sha256": np.asarray(source["files"]["models/model_conditioning.py"]),
        "environment_sha256": np.asarray(deterministic_json_hash(environment)),
    }


def _metric_or_reason(function: Callable[..., float], *args: Any, **kwargs: Any) -> tuple[float | str, str]:
    try:
        value = float(function(*args, **kwargs))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return "", str(exc)
    return (value, "") if math.isfinite(value) else ("", "undefined_for_observed_feature_group")


def _feature_rows(
    runs: list[RegisteredRun], config: dict[str, Any], output: Path, *, root: Path,
    strict: dict[str, Any], source: dict[str, Any], environment: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strongest, selection = _stronger_model(runs)
    required_models = ["M0", "M3", strongest]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for run in runs:
        if run.model_id not in required_models:
            continue
        path = output / "features" / f"{run.model_id}_seed{run.seed}.npz"
        try:
            arrays = extract_checkpoint_features(
                run, phase3_config=strict["config"], dataset=strict["dataset"], split=strict["split"],
                root=root, phase4_config=config, source=source, environment=environment,
            )
            _atomic_npz(path, arrays)
            features = _validate_feature_bundle(
                path, run, config, source=source, environment=environment,
                expected_logits=arrays["replayed_logits"],
            )
        except Exception as exc:
            missing.append({"model_id": run.model_id, "seed": run.seed, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        with np.load(run.predictions_path, allow_pickle=False) as prediction:
            labels = prediction["y_true"]
            snr_db = prediction["snr_db"]
        bundle_hash = file_sha256(path)
        for snr in SNRS:
            mask = snr_db == snr
            erank, erank_reason = _metric_or_reason(effective_rank, features[mask])
            share, share_reason = _metric_or_reason(top_singular_value_share, features[mask])
            separation, separation_reason = _metric_or_reason(class_separability, features[mask], labels[mask], class_labels=range(NUM_CLASSES))
            rows.append({
                "model_id": run.model_id, "seed": run.seed, "snr_db": snr,
                "sample_count": int(mask.sum()), "feature_dimension": int(features.shape[1]),
                "effective_rank": erank, "effective_rank_reason": erank_reason,
                "top_singular_value_share": share, "top_singular_value_share_reason": share_reason,
                "feature_separability": separation, "feature_separability_reason": separation_reason,
                "evidence_status": "measured", "hook_id": config["features"]["hook_id"],
                "feature_bundle": path.relative_to(output).as_posix(), "feature_bundle_sha256": bundle_hash,
            })
    invalid_geometry = [
        {"model_id": row["model_id"], "seed": row["seed"], "snr_db": row["snr_db"]}
        for row in rows
        if any(row[field] == "" or not math.isfinite(float(row[field])) for field in ("effective_rank", "top_singular_value_share", "feature_separability"))
    ]
    model_means: dict[str, dict[str, float | None]] = {}
    for model_id in required_models:
        selected = [row for row in rows if row["model_id"] == model_id]
        model_means[model_id] = {}
        for field in ("effective_rank", "top_singular_value_share", "feature_separability"):
            observed = [float(row[field]) for row in selected if row[field] != ""]
            model_means[model_id][field] = float(np.mean(observed)) if observed else None
    return rows, {"required_models": required_models, "best_stronger_conditioner": strongest, "stronger_selection": selection, "required_bundle_count": 15, "measured_bundle_count": 15 - len(missing), "missing_bundle_count": len(missing), "missing": missing, "invalid_geometry_count": len(invalid_geometry), "invalid_geometry": invalid_geometry, "invented_values": False, "hook_id": config["features"]["hook_id"], "hook_semantics": config["features"]["hook_semantics"], "model_means": model_means}


CONTINUOUS_FIELDS = (
    "model_id", "seed", "snr_db", "sample_count", "prediction_concentration", "balanced_accuracy",
    "concentration_minus_balanced_accuracy", "normalized_entropy", "hhi", "gini", "ece", "predictions_sha256",
)
THRESHOLD_FIELDS = (
    "model_id", "snr_range", "concentration_cutoff", "ba_gap_cutoff", "evaluated_model_seed_snr_cells",
    "collapse_rate", "plain_collapse_rate", "difference_vs_plain", "direction_vs_plain",
)
FEATURE_FIELDS_CSV = (
    "model_id", "seed", "snr_db", "sample_count", "feature_dimension", "effective_rank", "effective_rank_reason",
    "top_singular_value_share", "top_singular_value_share_reason", "feature_separability", "feature_separability_reason",
    "evidence_status", "hook_id", "feature_bundle", "feature_bundle_sha256",
)


def _means(continuous: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {model: float(np.mean([row[field] for row in continuous if row["model_id"] == model])) for model in MODELS}


def _report(summary: dict[str, Any]) -> str:
    status = summary["status"]
    lines = [
        "# Phase 4 Collapse Metric Validation", "", f"Status: {status}", "",
        "All prediction-distribution statistics are recomputed from registered, hash-bound Phase 3 prediction bundles. No training is performed.", "",
    ]
    if status == "BLOCKED_UPSTREAM":
        lines.extend(["## Blocker", "", summary["blocker"], "", "No collapse conclusion is published from a partial 40-run matrix."])
        return "\n".join(lines) + "\n"
    threshold = summary["threshold_sensitivity"]
    feature = summary["feature_evidence"]
    means = summary["model_means_across_seed_snr_cells"]
    lines.extend([
        "## Continuous diagnostics", "",
        "Prediction concentration, balanced accuracy, concentration minus balanced accuracy, normalized entropy, HHI, Gini, and 15-bin ECE are reported for every model×seed×SNR cell.", "",
        "| Model | Concentration | BA | Concentration−BA | Normalized entropy | HHI | Gini | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model_id in MODELS:
        lines.append(
            f"| {model_id} | {means['prediction_concentration'][model_id]:.6f} | "
            f"{means['balanced_accuracy'][model_id]:.6f} | "
            f"{means['concentration_minus_balanced_accuracy'][model_id]:.6f} | "
            f"{means['normalized_entropy'][model_id]:.6f} | {means['hhi'][model_id]:.6f} | "
            f"{means['gini'][model_id]:.6f} | {means['ece'][model_id]:.6f} |"
        )
    lines.extend([
        "",
        "## Threshold sensitivity", "", threshold["conclusion"] + ".", "",
        "Binary labels are sensitivity diagnostics only. Continuous prediction-concentration and discriminability degradation remain the primary description; the report does not assert a universal collapse threshold.", "",
        "## Feature geometry", "",
        f"Required models: {', '.join(feature['required_models'])}. Measured bundles: {feature['measured_bundle_count']}/{feature['required_bundle_count']}.", "",
    ])
    if feature["missing_bundle_count"]:
        lines.append("Feature geometry is blocked by missing real hook evidence. No effective-rank, singular-share, or separability value was fabricated.")
    else:
        lines.extend([
            "Effective rank, top singular-value share, and class separability were measured from row-bound feature hook bundles.", "",
            "| Model | Effective rank | Top singular-value share | Feature separability |",
            "|---|---:|---:|---:|",
        ])
        for model_id in feature["required_models"]:
            geometry = feature["model_means"][model_id]
            rendered = [("undefined" if geometry[field] is None else f"{geometry[field]:.6f}") for field in ("effective_rank", "top_singular_value_share", "feature_separability")]
            lines.append(f"| {model_id} | {rendered[0]} | {rendered[1]} | {rendered[2]} |")
    lines.extend(["", "## Interpretation boundary", "", "These diagnostics support setting-specific continuous descriptions. They do not establish a phase transition, necessary-and-sufficient rule, or universal threshold."])
    return "\n".join(lines) + "\n"


def _dependency_hashes(root: Path, config_path: Path, config: dict[str, Any], source: dict[str, Any]) -> dict[str, str]:
    dependencies = dict(source["files"])
    candidates = [
        config_path,
        _portable_path(root, config["source"]["phase3_config"], "source.phase3_config"),
        _portable_path(root, config["source"]["phase3_ledger"], "source.phase3_ledger"),
        _portable_path(root, config["source"]["phase3_manifest"], "source.phase3_manifest"),
    ]
    phase3 = yaml.safe_load(candidates[1].read_text(encoding="utf-8"))
    candidates.extend([root / phase3["split"]["metadata"], root / phase3["dataset"]["path"]])
    for path in candidates:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Phase 4 dependency escapes the repository")
        dependencies[resolved.relative_to(root).as_posix()] = file_sha256(resolved)
    return dict(sorted(dependencies.items()))


def _write_manifest(
    output: Path, status: str, *, fingerprint: str, root: Path,
    dependencies: dict[str, str], environment: dict[str, Any], upstream: dict[str, Any],
    selection: dict[str, Any] | None,
) -> None:
    artifact_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    artifacts = {
        path.relative_to(output).as_posix(): {
            "path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }
    feature_bindings = []
    for path in sorted((output / "features").glob("*.npz")) if (output / "features").is_dir() else []:
        with np.load(path, allow_pickle=False) as bundle:
            feature_bindings.append({
                "path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                **{name: bundle[name].item() for name in (
                    "model_id", "seed", "binding_hash", "predictions_sha256", "checkpoint_sha256",
                    "extractor_code_sha256", "model_code_sha256", "environment_sha256",
                )},
            })
    _atomic_json(output / "manifest.json", {
        "schema_version": 2, "runner_version": RUNNER_VERSION, "status": status,
        "analysis_fingerprint": fingerprint, "created_at": _now(),
        "dependencies": dependencies, "environment": environment,
        "environment_sha256": deterministic_json_hash(environment),
        "upstream": upstream, "stronger_selection": selection,
        "feature_bindings": feature_bindings, "artifacts": artifacts,
    })
    validate_phase4_manifest(output / "manifest.json", output, repository_root=root)


def validate_phase4_manifest(path: str | Path, output_root: str | Path, *, repository_root: str | Path | None = None) -> dict[str, Any]:
    """Validate hash closure, dependencies, environment, and every feature bundle."""

    manifest_path = Path(path)
    output = Path(output_root).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else ROOT.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "runner_version", "status", "analysis_fingerprint", "created_at",
        "dependencies", "environment", "environment_sha256", "upstream", "stronger_selection",
        "feature_bindings", "artifacts",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != 2 or value.get("runner_version") != RUNNER_VERSION:
        raise ValueError("Phase 4 manifest schema is invalid")
    if value["status"] not in {"BLOCKED_UPSTREAM", "BLOCKED_FEATURE_EVIDENCE", "COMPLETE"} or not _is_sha256(value["analysis_fingerprint"]):
        raise ValueError("Phase 4 manifest identity is invalid")
    if deterministic_json_hash(value["environment"]) != value["environment_sha256"]:
        raise ValueError("Phase 4 manifest environment hash is invalid")
    if not isinstance(value["dependencies"], dict) or not value["dependencies"]:
        raise ValueError("Phase 4 manifest dependency closure is invalid")
    for relative, expected in value["dependencies"].items():
        dependency = _portable_path(root, relative, "manifest dependency")
        if not dependency.is_file() or not _is_sha256(expected) or file_sha256(dependency) != expected:
            raise ValueError(f"Phase 4 dependency hash is invalid: {relative}")
    actual_artifacts = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"
    }
    if set(value["artifacts"]) != actual_artifacts:
        raise ValueError("Phase 4 manifest artifact closure is polluted or incomplete")
    for name, record in value["artifacts"].items():
        if not isinstance(record, dict) or record != {"path": name, "sha256": record.get("sha256")} or not _is_sha256(record.get("sha256")):
            raise ValueError("Phase 4 manifest artifact record is invalid")
        artifact = (output / name).resolve()
        if not artifact.is_relative_to(output) or not artifact.is_file() or file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"Phase 4 artifact hash is invalid: {name}")
    required_files = {"summary.json", "mechanism_report.md", "continuous_metrics.csv", "threshold_sensitivity.csv", "feature_geometry.csv"}
    if not required_files <= actual_artifacts:
        raise ValueError("Phase 4 manifest omits a required artifact")
    feature_files = sorted(name for name in actual_artifacts if name.startswith("features/") and name.endswith(".npz"))
    if value["status"] == "COMPLETE" and (len(feature_files) != 15 or len(value["feature_bindings"]) != 15):
        raise ValueError("COMPLETE Phase 4 requires exactly 15 feature bundles")
    binding_index = {record.get("path"): record for record in value["feature_bindings"] if isinstance(record, dict)}
    if set(binding_index) != set(feature_files):
        raise ValueError("Phase 4 feature binding closure is invalid")
    for name in feature_files:
        record = binding_index[name]
        path_obj = output / name
        with np.load(path_obj, allow_pickle=False) as bundle:
            if set(bundle.files) != FEATURE_FIELDS:
                raise ValueError(f"feature bundle schema is invalid: {name}")
            for field in ("binding_hash", "predictions_sha256", "checkpoint_sha256", "extractor_code_sha256", "model_code_sha256", "environment_sha256"):
                if bundle[field].ndim != 0 or record.get(field) != bundle[field].item():
                    raise ValueError(f"feature binding identity is invalid: {name}/{field}")
            if bundle["environment_sha256"].item() != value["environment_sha256"]:
                raise ValueError(f"feature environment binding is invalid: {name}")
        if record.get("sha256") != file_sha256(path_obj):
            raise ValueError(f"feature bundle hash is invalid: {name}")
    return value


class _Phase4Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False
        self.token = uuid.uuid4().hex

    @staticmethod
    def _alive(pid: Any) -> bool:
        if type(pid) is not int or pid <= 0:
            return False
        if os.name == "nt":
            # Windows os.kill(pid, 0) is not the POSIX existence probe.
            # Query a process handle without sending a signal to its owner.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: absent PID
            try:
                return kernel.WaitForSingleObject(handle, 0) != 0
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    owner = {}
                if self._alive(owner.get("pid")):
                    raise RuntimeError(f"Phase 4 is already locked at {self.path}") from exc
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        else:
            raise RuntimeError(f"Phase 4 lock cannot be acquired at {self.path}")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _now(), "token": self.token}))
        self.acquired = True
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.acquired:
            try:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    owner = {}
                if owner.get("token") == self.token:
                    self.path.unlink()
            finally:
                self.acquired = False


def _next_blocked_attempt(output: Path, prefix: str) -> Path:
    attempts = output / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    index = 1
    while (candidate := attempts / f"{prefix}_blocked{index:03d}").exists():
        index += 1
    return candidate


def _publish_staging(output: Path, staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Phase 4 attempt already exists: {destination}")
    staging.replace(destination)


def _current_complete(output: Path) -> tuple[Path, dict[str, Any]] | None:
    pointer = output / "current.json"
    if not pointer.is_file():
        return None
    value = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "attempt", "manifest_sha256", "analysis_fingerprint"} or value["schema_version"] != 1:
        raise ValueError("Phase 4 current pointer schema is invalid")
    attempt = _portable_path(output, value["attempt"], "current.attempt", "attempts")
    manifest = validate_phase4_manifest(attempt / "manifest.json", attempt, repository_root=ROOT if output.is_relative_to(ROOT) else None)
    if manifest["status"] != "COMPLETE" or file_sha256(attempt / "manifest.json") != value["manifest_sha256"] or manifest["analysis_fingerprint"] != value["analysis_fingerprint"]:
        raise ValueError("Phase 4 current pointer does not bind a COMPLETE attempt")
    return attempt, manifest


def _update_progress(root: Path, status: str, attempt: Path, summary: dict[str, Any]) -> None:
    from v2.progress import update_phase_progress
    relative = attempt.resolve().relative_to(root.resolve()).as_posix()
    if status == "COMPLETE":
        update_phase_progress(
            "Phase 4", completed=f"COMPLETE: strict 40-run audit and 15 checkpoint-derived feature bundles published at {relative}.",
            failed="None.", unexpected="None.",
            interpretation="Continuous collapse diagnostics and finite per-SNR geometry are bound to fixed-test predictions and classifier replay.",
            next_gate="PROCEED_TO_PHASE_5", path=root / "reports/v2_progress.md",
        )
    else:
        update_phase_progress(
            "Phase 4", completed="INCOMPLETE: no final Phase 4 conclusion was published.",
            failed=summary.get("blocker", "Feature evidence is incomplete."), unexpected="None.",
            interpretation="Blocked evidence is preserved in an immutable attempt and does not supersede any prior COMPLETE pointer.",
            next_gate="BLOCKED: resolve Phase 4 evidence before Phase 5.", path=root / "reports/v2_progress.md",
        )


def run_phase4(config_path: str | Path, *, repository_root: str | Path | None = None, logger: Callable[[str], None] = print) -> int:
    root = (Path(repository_root) if repository_root is not None else ROOT).resolve()
    config_file = Path(config_path).resolve()
    config = load_phase4_config(config_file)
    output = _portable_path(root, config["outputs"]["root"], "outputs.root", "results/v2/phase4_collapse_metrics")
    output.mkdir(parents=True, exist_ok=True)
    with _Phase4Lock(output / ".phase4.lock"):
        source = _source_identity(root)
        environment = environment_fingerprint()
        try:
            strict = _strict_phase3_contract(root, config)
            runs = audit_phase3_registry(root, config, _strict_context=strict)
        except UpstreamValidationError as exc:
            summary = {"schema_version": 1, "runner_version": RUNNER_VERSION, "status": "BLOCKED_UPSTREAM", "completed_phase3_runs": exc.completed_runs, "required_phase3_runs": 40, "blocker": str(exc), "published_complete": False}
            fingerprint = deterministic_json_hash({"status": "BLOCKED_UPSTREAM", "config": file_sha256(config_file), "source": source, "blocker": str(exc)})
            staging = output / f".staging-{uuid.uuid4().hex}"
            staging.mkdir(exist_ok=False)
            try:
                _atomic_csv(staging / "continuous_metrics.csv", [], CONTINUOUS_FIELDS)
                _atomic_csv(staging / "threshold_sensitivity.csv", [], THRESHOLD_FIELDS)
                _atomic_csv(staging / "feature_geometry.csv", [], FEATURE_FIELDS_CSV)
                _atomic_json(staging / "summary.json", summary)
                _atomic_text(staging / "mechanism_report.md", _report(summary))
                # A blocked attempt still closes over all dependencies that exist.
                dependencies = dict(source["files"])
                config_relative = config_file.relative_to(root).as_posix()
                dependencies[config_relative] = file_sha256(config_file)
                destination = _next_blocked_attempt(output, fingerprint[:16])
                _write_manifest(staging, summary["status"], fingerprint=fingerprint, root=root, dependencies=dependencies, environment=environment, upstream={"completed_runs": exc.completed_runs, "required_runs": 40}, selection=None)
                _publish_staging(output, staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            if not (output / "current.json").exists():
                _update_progress(root, summary["status"], destination, summary)
            logger(f"PHASE 4 COLLAPSE METRICS\nStatus: {summary['status']}\nRemaining blocker: {exc}")
            return 3

        continuous = _continuous_rows(runs, config)
        threshold_rows, threshold_summary = _threshold_rows(continuous, config)
        upstream = {
            "completed_runs": len(runs), "required_runs": 40,
            "phase3_ledger_sha256": file_sha256(_portable_path(root, config["source"]["phase3_ledger"], "source.phase3_ledger")),
            "phase3_manifest_sha256": file_sha256(_portable_path(root, config["source"]["phase3_manifest"], "source.phase3_manifest")),
            "registered_result_set_sha256": deterministic_json_hash([{"model_id": run.model_id, "seed": run.seed, "result_sha256": run.result_sha256, "predictions_sha256": run.predictions_sha256} for run in runs]),
        }
        dependencies = _dependency_hashes(root, config_file, config, source)
        fingerprint = deterministic_json_hash({"runner": RUNNER_VERSION, "config_sha256": file_sha256(config_file), "dependencies": dependencies, "environment": environment, "upstream": upstream})
        completed_attempt = output / "attempts" / fingerprint[:16]
        if completed_attempt.exists():
            manifest = validate_phase4_manifest(completed_attempt / "manifest.json", completed_attempt, repository_root=root)
            if manifest["status"] != "COMPLETE" or manifest["analysis_fingerprint"] != fingerprint:
                raise RuntimeError("existing Phase 4 attempt conflicts with the current fingerprint")
            _atomic_json(output / "current.json", {"schema_version": 1, "attempt": completed_attempt.relative_to(output).as_posix(), "manifest_sha256": file_sha256(completed_attempt / "manifest.json"), "analysis_fingerprint": fingerprint})
            logger("PHASE 4 COLLAPSE METRICS\nStatus: COMPLETE\nPROCEED_TO_PHASE_5")
            return 0

        staging = output / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(exist_ok=False)
        try:
            feature_rows, feature_summary = _feature_rows(runs, config, staging, root=root, strict=strict, source=source, environment=environment)
            status = "COMPLETE" if feature_summary["missing_bundle_count"] == 0 and feature_summary["invalid_geometry_count"] == 0 and len(feature_rows) == 15 * len(SNRS) and len(continuous) == 800 else "BLOCKED_FEATURE_EVIDENCE"
            summary = {
                "schema_version": 1, "runner_version": RUNNER_VERSION, "status": status,
                "completed_phase3_runs": len(runs), "required_phase3_runs": 40,
                "continuous_metric_rows": len(continuous),
                "metric_definitions": config["metrics"],
                "model_means_across_seed_snr_cells": {
                    "prediction_concentration": _means(continuous, "prediction_concentration"),
                    "balanced_accuracy": _means(continuous, "balanced_accuracy"),
                    "concentration_minus_balanced_accuracy": _means(continuous, "concentration_minus_balanced_accuracy"),
                    "normalized_entropy": _means(continuous, "normalized_entropy"),
                    "hhi": _means(continuous, "hhi"), "gini": _means(continuous, "gini"), "ece": _means(continuous, "ece"),
                },
                "threshold_sensitivity": threshold_summary,
                "feature_evidence": feature_summary,
                "published_complete": status == "COMPLETE",
            }
            _atomic_csv(staging / "continuous_metrics.csv", continuous, CONTINUOUS_FIELDS)
            _atomic_csv(staging / "threshold_sensitivity.csv", threshold_rows, THRESHOLD_FIELDS)
            _atomic_csv(staging / "feature_geometry.csv", feature_rows, FEATURE_FIELDS_CSV)
            _atomic_json(staging / "summary.json", summary)
            _atomic_text(staging / "mechanism_report.md", _report(summary))
            destination = completed_attempt if status == "COMPLETE" else _next_blocked_attempt(output, fingerprint[:16])
            _write_manifest(staging, status, fingerprint=fingerprint, root=root, dependencies=dependencies, environment=environment, upstream=upstream, selection=feature_summary["stronger_selection"])
            _publish_staging(output, staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if status == "COMPLETE":
            _atomic_json(output / "current.json", {"schema_version": 1, "attempt": destination.relative_to(output).as_posix(), "manifest_sha256": file_sha256(destination / "manifest.json"), "analysis_fingerprint": fingerprint})
            _update_progress(root, status, destination, summary)
        elif not (output / "current.json").exists():
            _update_progress(root, status, destination, summary)
        logger(
            "PHASE 4 COLLAPSE METRICS\n"
            f"Status: {status}\n"
            f"Continuous concentration finding: {summary['model_means_across_seed_snr_cells']['prediction_concentration']}\n"
            f"Balanced-accuracy finding: {summary['model_means_across_seed_snr_cells']['balanced_accuracy']}\n"
            f"Entropy finding: {summary['model_means_across_seed_snr_cells']['normalized_entropy']}\n"
            f"Threshold sensitivity: {threshold_summary['conclusion']}\n"
            f"Feature geometry: {feature_summary['measured_bundle_count']}/{feature_summary['required_bundle_count']} real bundles\n"
            "Collapse claim status: continuous setting-specific description only\n"
            f"Remaining blocker: {'none' if status == 'COMPLETE' else 'missing or non-finite checkpoint-derived feature evidence'}\n"
            + ("PROCEED_TO_PHASE_5" if status == "COMPLETE" else "")
        )
        return 0 if status == "COMPLETE" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/v2/phase4.yaml"))
    args = parser.parse_args(argv)
    return run_phase4(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

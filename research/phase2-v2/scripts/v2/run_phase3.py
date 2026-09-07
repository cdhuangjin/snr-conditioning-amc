#!/usr/bin/env python3
"""Resumable and auditable Phase 3 SNR-conditioning controls."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_reproduce import PhaseExecutionLock, ReproductionValidationError
from v2.contracts import ArtifactRecord
from v2.manifest import ManifestStore
from v2.progress import update_phase_progress
from v2.provenance import deterministic_json_hash, file_sha256
from v2.splits import format_snr, load_split

RUNNER_VERSION = "phase3-conditioning/1"
LATENCY_WARMUP = 3
LATENCY_REPEATS = 10
MODEL_IDS = tuple(f"M{index}" for index in range(8))
REUSED_MODELS = frozenset({"M0", "M3"})
TRAINED_MODELS = frozenset(set(MODEL_IDS) - REUSED_MODELS)
LOCKED_DATASET = {"id": "RML2016.10a", "path": "data/RML2016.10a_dict.pkl"}
LOCKED_SPLIT = "splits/v2/RML2016.10a_seed2022.json"
LOCKED_SPLIT_HASH = "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"
LOCKED_SEEDS = [2022, 2023, 2024, 2025, 2026]
LOCKED_TRAINING = {"optimizer": "Adam", "lr": 0.001, "batch_size": 128, "max_epochs": 100}
LOCKED_PREPROCESSING = {"input_layout": "NCT (I/Q channels, time)", "signal_dtype": "float32", "transforms": []}
LOCKED_SNR_VALUES = list(range(-20, 20, 2))
LOCKED_SNR = {
    "snr_min_db": -20, "snr_max_db": 18, "values_db": LOCKED_SNR_VALUES,
    "scalar_normalization": {"expression": "z = (snr_db - (-1)) / 19", "input_range_db": [-20, 18], "output_range": [-1, 1]},
}
LOCKED_SNR_BANDS = {"low": {"max_db": -8}, "mid": {"min_db": -6, "max_db": -2}, "high": {"min_db": 0}}
LOCKED_MODELS = {
    "M0": {"conditioner": "none"},
    "M1": {"conditioner": "normalized_scalar_concat", "condition_dim": 1, "location": "attended_pooled_features"},
    "M2": {"conditioner": "one_hot_snr_concat", "condition_dim": 20, "location": "attended_pooled_features"},
    "M3": {
        "conditioner": "learned_snr_embedding_concat", "condition_dim": 8, "location": "attended_pooled_features",
        "checkpoint_compatibility": "Phase 2 AWNSNR when num_classes, num_levels, in_channels, kernel_size, latent_dim, num_snr_bins, and snr_embedding_dim match",
        "capacity_semantics": "Embedding plus immediate linear concatenation supplies z-dependent additive terms, not arbitrary per-bin feature normals.",
    },
    "M4": {"conditioner": "class_logit_bias", "feature_interaction": "none"},
    "M5": {"conditioner": "film", "snr_representation": "normalized_scalar", "function": "gamma(z) * pooled_features + beta(z)", "modulation_dimension": 128},
    "M6": {"conditioner": "feature_gate", "snr_representation": "normalized_scalar", "function": "2 * sigmoid(W z + b)", "range": "(0, 2)", "dimension": 128},
    "M7": {"conditioner": "per_snr_classifier_heads", "shared_feature_extractor": True, "head_count": 20, "head_architecture": "128 -> 320 -> 11", "role": "capacity_upper_control"},
}
PREDICTION_FIELDS = (
    "y_true", "y_pred", "logits", "snr_db", "snr_bin", "sample_ids",
    "seed", "model_id", "split_hash", "preprocessing_hash", "protocol_hash",
)
SOURCE_FILES = (
    "scripts/v2/run_phase3.py", "scripts/v2/run_phase2.py", "scripts/v2/phase1_reproduce.py",
    "models/model_conditioning.py", "models/model.py", "models/model_snr.py", "models/lifting.py",
    "util/utils.py", "v2/contracts.py", "v2/splits.py", "v2/manifest.py", "v2/progress.py", "v2/provenance.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _repo_output_path(root: Path, value: Any, field: str, prefixes: tuple[str, ...]) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{field} must be a repository-relative allowlisted path")
    normalized = Path(value).as_posix()
    if not any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes):
        raise ValueError(f"{field} is outside its allowlisted location")
    resolved = (root / normalized).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{field} escapes repository root")
    return resolved


def validate_output_paths(root: Path, outputs: Any) -> dict[str, Path]:
    if not isinstance(outputs, dict) or set(outputs) != {"root", "manifest", "progress"}:
        raise ValueError("outputs must contain exactly root, manifest, and progress")
    return {
        "root": _repo_output_path(root, outputs["root"], "outputs.root", ("results/v2/phase3",)),
        "manifest": _repo_output_path(root, outputs["manifest"], "outputs.manifest", ("manifest.json", "results/v2/phase3")),
        "progress": _repo_output_path(root, outputs["progress"], "outputs.progress", ("reports/v2_progress.md", "reports/v2_phase3")),
    }


def load_phase3_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Phase 3 configuration requires schema_version: 1")
    validate_phase3_config(config)
    return config


def validate_phase3_config(config: dict[str, Any]) -> None:
    expected_top_level = {
        "schema_version", "dataset", "split", "seeds", "device", "preprocessing", "training",
        "architecture", "evaluation_batch_size", "snr", "snr_bands", "runner_call", "models", "outputs",
    }
    if set(config) != expected_top_level:
        raise ValueError("Phase 3 configuration fields differ from the predeclared protocol")
    if config.get("dataset") != LOCKED_DATASET:
        raise ValueError("dataset must be exactly the designated RML2016.10a input")
    if config.get("split") != {"metadata": LOCKED_SPLIT, "hash": LOCKED_SPLIT_HASH}:
        raise ValueError("split must be the designated fixed split and hash")
    if config.get("seeds") != LOCKED_SEEDS:
        raise ValueError(f"seeds must be exactly {LOCKED_SEEDS}")
    if config.get("preprocessing") != LOCKED_PREPROCESSING:
        raise ValueError("preprocessing must be the locked Phase 2 preprocessing")
    if config.get("training") != LOCKED_TRAINING:
        raise ValueError("training optimizer, learning rate, batch size, and epoch budget are locked")
    if config.get("evaluation_batch_size") != 64 or config.get("device") != "cpu":
        raise ValueError("evaluation batch size and device must match the Phase 2 protocol")
    if config.get("snr") != LOCKED_SNR or config.get("snr_bands") != LOCKED_SNR_BANDS:
        raise ValueError("SNR grid, normalization, and segment definitions are locked")
    if config.get("models") != LOCKED_MODELS or tuple(config["models"]) != MODEL_IDS:
        raise ValueError("M0-M7 definitions must exactly match the predeclared conditioning controls")
    architecture = config.get("architecture", {})
    required_architecture = {
        "num_classes": 11, "num_levels": 1, "in_channels": 64, "kernel_size": 3,
        "latent_dim": 320, "regu_details": 0.01, "regu_approx": 0.01,
        "num_snr_bins": 20, "snr_embedding_dim": 8,
    }
    if architecture != required_architecture:
        raise ValueError("architecture must exactly match the locked Phase 2 backbone and conditioning dimensions")
    if config.get("runner_call") != {"api": "forward_batch", "arguments": ["signals", "snr_db", "snr_bin"]}:
        raise ValueError("runner_call must require forward_batch with true SNR and SNR bin")
    if not isinstance(config.get("outputs"), dict) or set(config["outputs"]) != {"root", "manifest", "progress"}:
        raise ValueError("outputs must contain exactly root, manifest, and progress")


def preprocessing_hash(config: dict[str, Any]) -> str:
    return deterministic_json_hash(config["preprocessing"])


def protocol_hash(config: dict[str, Any]) -> str:
    """Hash the scientific protocol while excluding only artifact destinations."""
    return deterministic_json_hash({key: value for key, value in config.items() if key != "outputs"})


def _sample_ids(split: Any) -> np.ndarray:
    return np.asarray(split.sample_ids)[np.asarray(split.test_idx, dtype=np.int64)]


def _snr_bins(snrs: Any, snr_values: Iterable[int]) -> np.ndarray:
    values = np.asarray(snrs)
    lookup = {float(value): index for index, value in enumerate(snr_values)}
    try:
        return np.asarray([lookup[float(value)] for value in values], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"SNR value is outside the declared grid: {exc.args[0]}") from exc


class ConditioningDataset:
    """Index-backed view carrying true SNR dB and its independently checked bin."""

    def __init__(self, signals: Any, labels: Any, snrs: Any, indices: Any) -> None:
        import torch
        self.signals = signals
        self.labels = labels
        self.snrs = snrs
        self.indices = torch.as_tensor(indices, dtype=torch.int64)
        values = snrs.detach().cpu().numpy() if hasattr(snrs, "detach") else np.asarray(snrs)
        bins = _snr_bins(values, LOCKED_SNR_VALUES)
        if not np.array_equal(np.asarray(LOCKED_SNR_VALUES, dtype=float)[bins], np.asarray(values, dtype=float)):
            raise ValueError("SNR-to-bin mapping does not round-trip the declared grid")
        self.snr_bins = torch.from_numpy(bins)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> tuple[Any, Any, Any, Any]:
        index = self.indices[position]
        return self.signals[index], self.labels[index], self.snrs[index], self.snr_bins[index]


def make_conditioning_dataset(signals: Any, labels: Any, snrs: Any, indices: Any) -> ConditioningDataset:
    return ConditioningDataset(signals, labels, snrs, indices)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, num_classes: int) -> dict[str, Any]:
    if not len(y_true):
        raise ValueError("metric group cannot be empty")
    matrix = _confusion(y_true, y_pred, num_classes)
    support = matrix.sum(1)
    recalls = np.divide(np.diag(matrix), support, out=np.zeros(num_classes), where=support != 0)
    precision_den = matrix.sum(0)
    precision = np.divide(np.diag(matrix), precision_den, out=np.zeros(num_classes), where=precision_den != 0)
    f1 = np.divide(2 * precision * recalls, precision + recalls, out=np.zeros(num_classes), where=(precision + recalls) != 0)
    observed = support > 0
    counts = matrix.sum(0)
    shares = counts / len(y_true)
    positive = shares[shares > 0]
    dominant = int(np.argmax(counts))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1[observed])),
        "balanced_accuracy": float(np.mean(recalls[observed])),
        "prediction_concentration": float(shares.max()),
        "normalized_prediction_entropy": -float(np.sum(positive * np.log(positive)) / np.log(num_classes)),
        "gini": float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * num_classes)),
        "hhi": float(np.sum(shares**2)),
        "dominant_predicted_class": dominant,
        "dominant_predicted_class_ratio": float(shares[dominant]),
        "count": int(len(y_true)),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_metrics(y_true: Any, logits: Any, snr_db: Any, *, num_classes: int, snr_values: Iterable[int]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    snr_db = np.asarray(snr_db)
    if y_true.ndim != 1 or logits.shape != (len(y_true), num_classes) or snr_db.shape != y_true.shape or not len(y_true):
        raise ValueError("labels, SNR, and finite N-by-class logits must describe the same non-empty rows")
    if not np.isfinite(logits).all() or np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("logits must be finite and labels must be valid class indices")
    probabilities = _softmax(logits)
    y_pred = logits.argmax(1).astype(np.int64)
    overall = _group_metrics(y_true, y_pred, probabilities, num_classes)
    maxima = logits.max(axis=1)
    centered_logsumexp = np.log(np.exp(logits - maxima[:, None]).sum(axis=1))
    overall["nll"] = float(np.mean((maxima - logits[np.arange(len(y_true)), y_true]) + centered_logsumexp))
    confidence = probabilities.max(1)
    correctness = y_pred == y_true
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1] if index < 14 else confidence <= 1.0)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    overall["ece"] = float(ece)
    per_snr = {}
    for value in snr_values:
        mask = snr_db == value
        if not mask.any():
            raise ValueError(f"fixed test split has no rows for declared SNR {value}")
        per_snr[format_snr(float(value))] = _group_metrics(y_true[mask], y_pred[mask], probabilities[mask], num_classes)
    segment_masks = {
        "low": snr_db <= -8,
        "mid": (snr_db >= -6) & (snr_db <= -2),
        "high": snr_db >= 0,
    }
    segments = {name: _group_metrics(y_true[mask], y_pred[mask], probabilities[mask], num_classes) for name, mask in segment_masks.items()}
    return _json_value({
        "overall_accuracy": overall.pop("accuracy"),
        "macro_f1": overall.pop("macro_f1"),
        "balanced_accuracy": overall.pop("balanced_accuracy"),
        **overall,
        "ece_definition": {"type": "top_label_equal_width", "bins": 15, "interval": "[0,1]", "last_bin_right_inclusive": True},
        "per_snr": per_snr,
        "segments": segments,
    })


def write_prediction_bundle(path: Path, outcome: dict[str, Any], split: Any, identity: dict[str, Any]) -> None:
    logits = np.asarray(outcome["logits"], dtype=np.float32)
    labels = np.asarray(outcome["labels"], dtype=np.int64)
    snrs = np.asarray(outcome["snrs"], dtype=np.float32)
    arrays = {
        "y_true": labels,
        "y_pred": logits.argmax(1).astype(np.int64),
        "logits": logits,
        "snr_db": snrs,
        "snr_bin": _snr_bins(snrs, LOCKED_SNR_VALUES),
        "sample_ids": _sample_ids(split),
        "seed": np.asarray(identity["seed"], dtype=np.int64),
        "model_id": np.asarray(identity["model_id"]),
        "split_hash": np.asarray(identity["split_hash"]),
        "preprocessing_hash": np.asarray(identity["preprocessing_hash"]),
        "protocol_hash": np.asarray(identity["protocol_hash"]),
    }
    _atomic_npz(path, arrays)


def validate_prediction_bundle(path: str | Path, split: Any, dataset: dict[str, Any], *, identity: dict[str, Any], num_classes: int, snr_values: Iterable[int]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(PREDICTION_FIELDS):
            raise ValueError(f"prediction bundle fields must be exactly {sorted(PREDICTION_FIELDS)}")
        values = {name: bundle[name] for name in bundle.files}
    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    if len(values["y_true"]) != len(test_idx):
        raise ValueError("prediction count does not match fixed test split")
    if not np.array_equal(values["sample_ids"], _sample_ids(split)) or len(np.unique(values["sample_ids"])) != len(test_idx):
        raise ValueError("prediction sample IDs do not match fixed test split")
    if not np.array_equal(values["y_true"], np.asarray(dataset["labels"])[test_idx]):
        raise ValueError("prediction labels do not match fixed test split")
    if not np.array_equal(values["snr_db"], np.asarray(dataset["snrs"])[test_idx]):
        raise ValueError("prediction SNR values do not match fixed test split")
    if not np.array_equal(values["snr_bin"], _snr_bins(values["snr_db"], snr_values)):
        raise ValueError("prediction SNR bins do not match true SNR values")
    for field in ("seed", "model_id", "split_hash", "preprocessing_hash", "protocol_hash"):
        if values[field].ndim != 0 or values[field].item() != identity[field]:
            raise ValueError(f"prediction identity field {field} does not match run specification")
    if values["logits"].shape != (len(test_idx), num_classes) or not np.isfinite(values["logits"]).all():
        raise ValueError("prediction logits have invalid shape or non-finite values")
    recomputed_pred = values["logits"].argmax(1).astype(np.int64)
    if not np.array_equal(values["y_pred"], recomputed_pred):
        raise ValueError("y_pred does not equal argmax(logits)")
    return compute_metrics(values["y_true"], values["logits"], values["snr_db"], num_classes=num_classes, snr_values=snr_values)


def write_mechanism_bundle(path: Path, outcome: dict[str, Any], identity: dict[str, Any]) -> None:
    tensors = outcome.get("mechanism_tensors", {})
    if not isinstance(tensors, dict):
        raise ValueError("mechanism_tensors must be a mapping")
    names = sorted(tensors)
    arrays: dict[str, Any] = {
        "model_id": np.asarray(identity["model_id"]), "seed": np.asarray(identity["seed"], dtype=np.int64),
        "split_hash": np.asarray(identity["split_hash"]), "preprocessing_hash": np.asarray(identity["preprocessing_hash"]),
        "protocol_hash": np.asarray(identity["protocol_hash"]), "checkpoint_sha256": np.asarray(identity["checkpoint_sha256"]),
        "tensor_names": np.asarray(names),
    }
    for index, name in enumerate(names):
        value = np.asarray(tensors[name])
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"mechanism tensor {name} must be a finite numeric array")
        arrays[f"tensor_{index:03d}"] = value
    _atomic_npz(path, arrays)


def _mechanism_shapes(model_id: str, architecture: dict[str, Any], pooled_feature_dim: int) -> dict[str, tuple[int, ...]]:
    bins = int(architecture["num_snr_bins"]); latent = int(architecture["latent_dim"]); classes = int(architecture["num_classes"])
    if model_id == "M0": return {}
    if model_id == "M1": return {"immediate_affine.condition_weight": (latent, 1)}
    if model_id == "M2": return {"immediate_affine.condition_weight": (latent, bins)}
    if model_id == "M3":
        embedding_dim = int(architecture["snr_embedding_dim"])
        return {
            "snr_embedding_vectors": (bins, embedding_dim),
            "immediate_affine.condition_weight": (latent, embedding_dim),
            "per_snr_additive_contribution": (bins, latent),
        }
    if model_id == "M4": return {"per_snr_logit_bias": (bins, classes)}
    if model_id == "M5": return {"per_snr_gamma": (bins, pooled_feature_dim), "per_snr_beta": (bins, pooled_feature_dim)}
    if model_id == "M6": return {"per_snr_gates": (bins, pooled_feature_dim)}
    result = {}
    for index in range(bins):
        result[f"snr_heads.{index}.0.weight"] = (latent, pooled_feature_dim)
        result[f"snr_heads.{index}.0.bias"] = (latent,)
        result[f"snr_heads.{index}.2.weight"] = (classes, latent)
        result[f"snr_heads.{index}.2.bias"] = (classes,)
    return result


def validate_mechanism_bundle(path: str | Path, *, identity: dict[str, Any], architecture: dict[str, Any], pooled_feature_dim: int, expected_tensors: dict[str, Any]) -> None:
    identity_fields = ("model_id", "seed", "split_hash", "preprocessing_hash", "protocol_hash", "checkpoint_sha256")
    with np.load(path, allow_pickle=False) as bundle:
        names = bundle["tensor_names"].tolist() if "tensor_names" in bundle.files else None
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("mechanism tensor_names must be a string vector")
        required = set(identity_fields) | {"tensor_names"} | {f"tensor_{index:03d}" for index in range(len(names))}
        if set(bundle.files) != required:
            raise ValueError("mechanism bundle fields do not match its exact schema")
        for field in identity_fields:
            if bundle[field].ndim != 0 or bundle[field].item() != identity[field]:
                raise ValueError(f"mechanism identity field {field} does not match")
        shapes = _mechanism_shapes(str(identity["model_id"]), architecture, pooled_feature_dim)
        if names != sorted(shapes) or set(expected_tensors) != set(shapes):
            raise ValueError("mechanism tensors do not match the declared model schema")
        for index, name in enumerate(names):
            value = bundle[f"tensor_{index:03d}"]
            expected = np.asarray(expected_tensors[name])
            if value.shape != shapes[name] or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"mechanism tensor {name} has invalid shape or values")
            if expected.shape != shapes[name] or not np.array_equal(value, expected):
                raise ValueError(f"mechanism tensor {name} differs from checkpoint-derived evidence")


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < minimum:
        raise ValueError(f"{field} must be a finite number >= {minimum}")
    return float(value)


def _validate_available_evidence(value: Any, field: str) -> None:
    if not isinstance(value, dict) or type(value.get("available")) is not bool:
        raise ValueError(f"{field} must declare boolean availability")
    if value["available"]:
        if set(value) != {"available", "value", "unit", "method"} or not all(isinstance(value[key], str) and value[key] for key in ("unit", "method")):
            raise ValueError(f"available {field} evidence requires value, unit, and method")
        _finite_number(value["value"], f"{field}.value")
    elif set(value) != {"available", "reason"} or not isinstance(value["reason"], str) or not value["reason"]:
        raise ValueError(f"unavailable {field} evidence requires a reason")


def validate_complexity(value: Any, *, fixed_batch: int) -> None:
    required = {"total_parameters", "additional_conditioner_parameters", "parameter_increase_percent", "flops", "latency", "memory"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("complexity evidence fields do not match the exact schema")
    total = value["total_parameters"]; additional = value["additional_conditioner_parameters"]
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0 or isinstance(additional, bool) or not isinstance(additional, int) or not 0 <= additional < total:
        raise ValueError("parameter counts must be positive integer totals and valid additions")
    percent = _finite_number(value["parameter_increase_percent"], "parameter_increase_percent")
    expected_percent = 100.0 * additional / (total - additional)
    if not math.isclose(percent, expected_percent, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("parameter increase percent does not match the parameter counts")
    _validate_available_evidence(value["flops"], "flops")
    _validate_available_evidence(value["memory"], "memory")
    latency = value["latency"]
    latency_fields = {"device", "warmup", "repeats", "timer", "bin_construction", "single_bin", "mixed_bins", "summary"}
    if not isinstance(latency, dict) or set(latency) != latency_fields:
        raise ValueError("latency evidence fields do not match the exact schema")
    if latency["device"] != "cpu" or latency["timer"] != "time.perf_counter" or type(latency["warmup"]) is not int or latency["warmup"] < 1 or type(latency["repeats"]) is not int or latency["repeats"] < 1 or latency["bin_construction"] != "single_bin uses bin 0; mixed_bins cycles deterministically over bins 0..19":
        raise ValueError("latency protocol must document CPU, timer, warmup, and repeats")
    for distribution in ("single_bin", "mixed_bins"):
        if not isinstance(latency[distribution], dict) or set(latency[distribution]) != {"batch_1", "fixed_batch"}:
            raise ValueError(f"latency.{distribution} must contain batch_1 and fixed_batch")
        for name, batch_size in (("batch_1", 1), ("fixed_batch", fixed_batch)):
            measurement = latency[distribution][name]
            if not isinstance(measurement, dict) or set(measurement) != {"batch_size", "mean_ms", "median_ms"} or measurement["batch_size"] != batch_size:
                raise ValueError(f"{distribution}.{name} latency schema or batch size is invalid")
            _finite_number(measurement["mean_ms"], f"latency.{distribution}.{name}.mean_ms")
            _finite_number(measurement["median_ms"], f"latency.{distribution}.{name}.median_ms")
    summary = latency["summary"]
    mixed = latency["mixed_bins"]["fixed_batch"]
    if not isinstance(summary, dict) or set(summary) != {"basis", "mean_ms", "median_ms"} or summary.get("basis") != "mixed_bins.fixed_batch":
        raise ValueError("latency summary must use the representative mixed_bins.fixed_batch measurement")
    if summary["mean_ms"] != mixed["mean_ms"] or summary["median_ms"] != mixed["median_ms"]:
        raise ValueError("latency summary values must equal mixed_bins.fixed_batch")


def _next_destination(output_root: Path, model_id: str, seed: int, fingerprint: str) -> Path:
    base = output_root / f"{model_id}_seed{seed}_{fingerprint[:12]}"
    if not base.exists():
        return base
    retry = 1
    while (candidate := output_root / f"{base.name}_retry{retry}").exists():
        retry += 1
    return candidate


def _code_identity(root: Path) -> dict[str, Any]:
    files = {name: file_sha256(root / name) for name in SOURCE_FILES}
    return {"files": files, "hash": deterministic_json_hash(files)}


def _environment_identity() -> dict[str, Any]:
    """Capture runtime properties that can affect inference identity and timing."""
    import torch
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_build": torch.__config__.show(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu": platform.processor() or platform.machine(),
        "device": "cpu",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "timing": {
            "timer": "time.perf_counter", "warmup": LATENCY_WARMUP, "repeats": LATENCY_REPEATS,
            "inputs": "deterministic zeros", "mixed_bin_construction": "cycle bins 0..19",
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
    }


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes repository root")
    return resolved.relative_to(root.resolve()).as_posix()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


_LEDGER_FIELDS = {"model_id", "seed", "run_fingerprint", "attempt_path", "result_sha256", "status", "updated_at"}


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "runner_version": RUNNER_VERSION, "records": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "runner_version", "records"} or value["schema_version"] != 1 or value["runner_version"] != RUNNER_VERSION or not isinstance(value["records"], list):
        raise ValueError("Phase 3 ledger has an invalid schema")
    seen = set()
    for record in value["records"]:
        if not isinstance(record, dict) or set(record) != _LEDGER_FIELDS or record["status"] not in {"completed", "failed"}:
            raise ValueError("Phase 3 ledger record has an invalid schema")
        key = record["attempt_path"]
        if not isinstance(key, str) or Path(key).is_absolute() or ".." in Path(key).parts or key in seen:
            raise ValueError("Phase 3 ledger attempt path is invalid or duplicated")
        seen.add(key)
        if not _is_sha256(record["result_sha256"]):
            raise ValueError("Phase 3 ledger result hash is invalid")
        if record["model_id"] not in MODEL_IDS or record["seed"] not in LOCKED_SEEDS or not _is_sha256(record["run_fingerprint"]) or not isinstance(record["updated_at"], str) or not record["updated_at"]:
            raise ValueError("Phase 3 ledger record identity is invalid")
    return value


def _ledger_upsert(path: Path, record: dict[str, Any]) -> None:
    ledger = _load_ledger(path)
    for index, existing in enumerate(ledger["records"]):
        if existing["attempt_path"] == record["attempt_path"]:
            ledger["records"][index] = record
            break
    else:
        ledger["records"].append(record)
    _atomic_json(path, ledger)


def _ledger_has_result(ledger: dict[str, Any], root: Path, candidate: Path, result_path: Path) -> bool:
    attempt_path = _relative(root, candidate)
    actual_hash = file_sha256(result_path)
    return any(record["attempt_path"] == attempt_path and record["status"] == "completed" and record["result_sha256"] == actual_hash for record in ledger["records"])


def _validate_result_schema(result: Any, *, model_id: str, seed: int, fingerprint: str, split_hash: str, preprocessing: str, protocol: str, code: dict[str, Any], environment: dict[str, Any], fixed_batch: int) -> None:
    required = {"schema_version", "runner_version", "status", "model_id", "seed", "completed_at", "execution", "metrics", "complexity", "training", "artifacts", "provenance"}
    if not isinstance(result, dict) or set(result) != required or result.get("schema_version") != 1 or result.get("runner_version") != RUNNER_VERSION or result.get("status") != "completed" or result.get("model_id") != model_id or result.get("seed") != seed:
        raise ValueError("completed result has an invalid top-level schema or identity")
    if not isinstance(result["completed_at"], str) or not result["completed_at"]:
        raise ValueError("completed result timestamp is invalid")
    execution = result["execution"]
    if not isinstance(execution, dict) or set(execution) != {"mode", "reuse_source", "source_artifact_hashes"} or execution["mode"] not in {"reused", "trained"}:
        raise ValueError("completed result execution schema is invalid")
    if (model_id in REUSED_MODELS and execution["mode"] != "reused") or (model_id not in REUSED_MODELS and execution["mode"] != "trained"):
        raise ValueError("completed result execution mode contradicts the model protocol")
    source_hash_fields = {"run_spec_sha256", "result_sha256", "checkpoint_sha256", "predictions_sha256"}
    if model_id in REUSED_MODELS:
        if not isinstance(execution["reuse_source"], str) or Path(execution["reuse_source"]).is_absolute() or ".." in Path(execution["reuse_source"]).parts:
            raise ValueError("completed reuse source must be repository-relative")
        hashes = execution["source_artifact_hashes"]
        if not isinstance(hashes, dict) or set(hashes) != source_hash_fields or any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError("completed reuse source hashes are invalid")
    elif execution["reuse_source"] is not None or execution["source_artifact_hashes"] != {}:
        raise ValueError("trained result must not claim Phase 2 source artifacts")
    if not isinstance(result["metrics"], dict):
        raise ValueError("completed result metrics must be an object")
    validate_complexity(result["complexity"], fixed_batch=fixed_batch)
    if not isinstance(result["training"], dict) or set(result["training"]) != {"best_epoch", "epochs_completed", "best_val_accuracy", "duration_seconds", "scheduler"}:
        raise ValueError("completed result training schema is invalid")
    training = result["training"]
    if type(training["best_epoch"]) is not int or not 0 <= training["best_epoch"] < 100 or training["epochs_completed"] != 100:
        raise ValueError("completed result epoch evidence is invalid")
    best_accuracy = _finite_number(training["best_val_accuracy"], "training.best_val_accuracy")
    if best_accuracy > 1.0:
        raise ValueError("training.best_val_accuracy must not exceed one")
    _finite_number(training["duration_seconds"], "training.duration_seconds")
    if training["scheduler"] != {"name": "none", "reason": "Phase 2 protocol used no scheduler"}:
        raise ValueError("scheduler evidence differs from the locked no-scheduler protocol")
    artifacts = result["artifacts"]
    expected_artifacts = {"run_spec": "run_spec.json", "checkpoint": "checkpoint.pt", "predictions": "predictions.npz", "mechanism": "mechanism.npz"}
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts) | {f"{name}_sha256" for name in expected_artifacts}:
        raise ValueError("completed result artifact schema is invalid")
    for name, filename in expected_artifacts.items():
        if artifacts[name] != filename or not _is_sha256(artifacts[f"{name}_sha256"]):
            raise ValueError("completed result artifact identity is invalid")
    expected_provenance = {"run_fingerprint": fingerprint, "split_hash": split_hash, "preprocessing_hash": preprocessing, "protocol_hash": protocol, "code": code, "environment": environment}
    if result["provenance"] != expected_provenance:
        raise ValueError("completed result provenance is invalid")


def _validate_run_spec_schema(spec: Any, *, model_id: str, seed: int, fingerprint: str, split_hash: str, preprocessing: str, protocol: str, config: dict[str, Any], code: dict[str, Any], environment: dict[str, Any]) -> None:
    required = {"schema_version", "runner_version", "created_at", "model_id", "seed", "run_fingerprint", "protocol_hash", "preprocessing_hash", "protocol_config", "fixed_split", "execution_mode", "provenance"}
    if not isinstance(spec, dict) or set(spec) != required or spec["schema_version"] != 1 or spec["runner_version"] != RUNNER_VERSION:
        raise ValueError("run specification has an invalid schema")
    expected = {
        "model_id": model_id, "seed": seed, "run_fingerprint": fingerprint, "protocol_hash": protocol,
        "preprocessing_hash": preprocessing, "protocol_config": config,
        "fixed_split": {"metadata": config["split"]["metadata"], "hash": split_hash},
        "execution_mode": "reused_phase2" if model_id in REUSED_MODELS else "trained_phase3", "provenance": {"code": code, "environment": environment},
    }
    if any(spec.get(key) != value for key, value in expected.items()) or not isinstance(spec["created_at"], str) or not spec["created_at"]:
        raise ValueError("run specification content differs from its canonical identity")


def _default_completed_verifier(model_id: str, dataset: dict[str, Any], split: Any, candidate: Path, logger: Callable[[str], None], *, config: dict[str, Any]) -> dict[str, Any]:
    import torch
    device = torch.device(config["device"])
    model = _model(config, model_id).to(device)
    model.load_state_dict(_load_state(candidate / "checkpoint.pt", device), strict=True)
    counts = model.parameter_counts(); base = counts["m0_equivalent"]
    return {
        "logits": _evaluate(model, dataset, split, config, device),
        "mechanism_tensors": _mechanism_tensors(model, config),
        "parameter_counts": {
            "total_parameters": counts["total"], "additional_conditioner_parameters": counts["additional_conditioner"],
            "parameter_increase_percent": float(100.0 * counts["additional_conditioner"] / base),
        },
    }


def _matching_completed(output_root: Path, model_id: str, seed: int, fingerprint: str, *, root: Path, ledger: dict[str, Any], split: Any, dataset: dict[str, Any], identity: dict[str, Any], config: dict[str, Any], code: dict[str, Any], environment: dict[str, Any], completed_verifier: Callable[..., dict[str, Any]] | None, phase2_source_validator: Callable[..., None] | None, logger: Callable[[str], None]) -> Path | None:
    for candidate in sorted(output_root.glob(f"{model_id}_seed{seed}_{fingerprint[:12]}*")):
        try:
            spec_path = candidate / "run_spec.json"
            result_path = candidate / "result.json"
            checkpoint = candidate / "checkpoint.pt"
            predictions = candidate / "predictions.npz"
            mechanism = candidate / "mechanism.npz"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = result["artifacts"]
            if not _ledger_has_result(ledger, root, candidate, result_path) or spec.get("run_fingerprint") != fingerprint:
                continue
            _validate_run_spec_schema(spec, model_id=model_id, seed=seed, fingerprint=fingerprint, split_hash=split.split_hash, preprocessing=identity["preprocessing_hash"], protocol=identity["protocol_hash"], config=config, code=code, environment=environment)
            _validate_result_schema(result, model_id=model_id, seed=seed, fingerprint=fingerprint, split_hash=split.split_hash, preprocessing=identity["preprocessing_hash"], protocol=identity["protocol_hash"], code=code, environment=environment, fixed_batch=int(config["evaluation_batch_size"]))
            if not all(path.is_file() for path in (checkpoint, predictions, mechanism)):
                continue
            hashes = {"run_spec": spec_path, "checkpoint": checkpoint, "predictions": predictions, "mechanism": mechanism}
            if any(artifacts.get(f"{name}_sha256") != file_sha256(path) for name, path in hashes.items()):
                continue
            metrics = validate_prediction_bundle(predictions, split, dataset, identity=identity, num_classes=int(config["architecture"]["num_classes"]), snr_values=config["snr"]["values_db"])
            if result.get("metrics") != metrics:
                continue
            verified = (completed_verifier or _default_completed_verifier)(model_id, dataset, split, candidate, logger, config=config)
            if not isinstance(verified, dict) or set(verified) != {"logits", "mechanism_tensors", "parameter_counts"}:
                continue
            if verified["parameter_counts"] != {key: result["complexity"][key] for key in ("total_parameters", "additional_conditioner_parameters", "parameter_increase_percent")}:
                continue
            with np.load(predictions, allow_pickle=False) as bundle:
                if not np.array_equal(np.asarray(verified["logits"], dtype=np.float32), bundle["logits"]):
                    continue
            mechanism_identity = identity | {"checkpoint_sha256": file_sha256(checkpoint)}
            validate_mechanism_bundle(mechanism, identity=mechanism_identity, architecture=config["architecture"], pooled_feature_dim=int(config["models"].get("M5", {}).get("modulation_dimension", 128)), expected_tensors=verified["mechanism_tensors"])
            if model_id in REUSED_MODELS:
                execution = result["execution"]
                validator = phase2_source_validator or _validate_phase2_reuse_provenance
                validator(model_id, seed, dataset, split, execution["reuse_source"], execution["source_artifact_hashes"], config=config | {"_repository_root": str(root.resolve())})
            return candidate
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return None


def _model(config: dict[str, Any], model_id: str):
    from models.model_conditioning import AWNConditioned
    architecture = config["architecture"]
    return AWNConditioned(
        **{key: architecture[key] for key in ("num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim", "regu_details", "regu_approx", "num_snr_bins", "snr_embedding_dim")},
        conditioning=model_id,
        snr_min_db=float(config["snr"]["snr_min_db"]),
        snr_max_db=float(config["snr"]["snr_max_db"]),
    )


def _load_state(path: Path, device: Any):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_phase2_compatible_model(model_id: str, checkpoint: Path, config: dict[str, Any]):
    """Strict-load a Phase 2 checkpoint into its declared unified Phase 3 model."""
    import torch
    if model_id not in REUSED_MODELS:
        raise ValueError("only M0 and M3 have Phase 2-compatible checkpoints")
    model = _model(config, model_id).to(torch.device("cpu"))
    model.load_state_dict(_load_state(checkpoint, torch.device("cpu")), strict=True)
    return model.eval()


def _validate_phase2_prediction_rows(path: str | Path, split: Any, dataset: dict[str, Any]) -> np.ndarray:
    """Validate the legacy Phase 2 bundle against fixed test rows without pickle."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {"predictions", "labels", "snrs", "sample_ids"}
        if set(bundle.files) != required:
            raise ValueError("Phase 2 prediction bundle fields differ from the exact legacy schema")
        values = {name: bundle[name] for name in required}
    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    if not np.array_equal(values["sample_ids"], _sample_ids(split)) or len(np.unique(values["sample_ids"])) != len(test_idx):
        raise ValueError("Phase 2 prediction sample IDs do not match the fixed split")
    if not np.array_equal(values["labels"], np.asarray(dataset["labels"])[test_idx]) or not np.array_equal(values["snrs"], np.asarray(dataset["snrs"])[test_idx]):
        raise ValueError("Phase 2 prediction labels or SNR values do not match fixed test rows")
    predictions = np.asarray(values["predictions"], dtype=np.int64)
    if predictions.shape != (len(test_idx),):
        raise ValueError("Phase 2 prediction count does not match the fixed split")
    return predictions


def _atomic_torch_save(state: Any, path: Path) -> None:
    import torch
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _latency(model: Any, *, signal_shape: tuple[int, ...], fixed_batch: int, snr_min: float, device: Any) -> dict[str, Any]:
    import torch
    warmup, repeats = LATENCY_WARMUP, LATENCY_REPEATS
    snr_grid = torch.as_tensor(LOCKED_SNR_VALUES, dtype=torch.float32, device=device)
    def measure(batch: int, *, mixed: bool) -> dict[str, float]:
        signals = torch.zeros((batch, *signal_shape), dtype=torch.float32, device=device)
        snr_bin = (torch.arange(batch, dtype=torch.long, device=device) % len(LOCKED_SNR_VALUES)) if mixed else torch.zeros(batch, dtype=torch.long, device=device)
        snr_db = snr_grid.index_select(0, snr_bin)
        with torch.no_grad():
            for _ in range(warmup):
                model.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                model.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)
                samples.append((time.perf_counter() - started) * 1000.0)
        return {"batch_size": batch, "mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples))}
    single_bin = {"batch_1": measure(1, mixed=False), "fixed_batch": measure(fixed_batch, mixed=False)}
    mixed_bins = {"batch_1": measure(1, mixed=True), "fixed_batch": measure(fixed_batch, mixed=True)}
    return {
        "device": "cpu", "warmup": warmup, "repeats": repeats, "timer": "time.perf_counter",
        "bin_construction": "single_bin uses bin 0; mixed_bins cycles deterministically over bins 0..19",
        "single_bin": single_bin, "mixed_bins": mixed_bins,
        "summary": {"basis": "mixed_bins.fixed_batch", "mean_ms": mixed_bins["fixed_batch"]["mean_ms"], "median_ms": mixed_bins["fixed_batch"]["median_ms"]},
    }


def _complexity(model: Any, config: dict[str, Any], signal_shape: tuple[int, ...], device: Any) -> dict[str, Any]:
    counts = model.parameter_counts()
    base = counts["m0_equivalent"]
    return {
        "total_parameters": counts["total"],
        "additional_conditioner_parameters": counts["additional_conditioner"],
        "parameter_increase_percent": float(100.0 * counts["additional_conditioner"] / base),
        "flops": {"available": False, "reason": "no validated FLOP counter supports the custom lifting modules"},
        "latency": _latency(model, signal_shape=signal_shape, fixed_batch=int(config["evaluation_batch_size"]), snr_min=float(config["snr"]["snr_min_db"]), device=device),
        "memory": {"available": False, "reason": "portable reliable CPU peak-memory instrumentation is unavailable"},
    }


def _evaluate(model: Any, dataset: dict[str, Any], split: Any, config: dict[str, Any], device: Any) -> np.ndarray:
    import torch
    from phase1_reproduce import make_historical_loader
    signals = torch.from_numpy(np.asarray(dataset["signals"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(dataset["labels"], dtype=np.int64))
    snrs = torch.from_numpy(np.asarray(dataset["snrs"], dtype=np.float32))
    test_set = make_conditioning_dataset(signals, labels, snrs, np.asarray(split.test_idx, dtype=np.int64))
    parts = []
    model.eval()
    with torch.no_grad():
        for x, _y, snr_db, bins in make_historical_loader(test_set, batch_size=int(config["evaluation_batch_size"]), shuffle=False):
            x, snr_db, bins = x.to(device), snr_db.to(device), bins.to(device)
            logits, _ = model.forward_batch(x, snr_db=snr_db, snr_bin=bins)
            parts.append(logits.cpu().numpy())
    return np.concatenate(parts)


def _mechanism_tensors(model: Any, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract inspectable parameters or realized controls on the fixed SNR grid."""
    import torch
    model_id = model.conditioning
    if model_id == "M0":
        return {}
    if model_id == "M3":
        embeddings = model.snr_embedding.weight.detach()
        projection = model.fc[0].weight[:, -model.snr_embedding_dim :].detach()
        return {
            "snr_embedding_vectors": embeddings.cpu().numpy(),
            "immediate_affine.condition_weight": projection.cpu().numpy(),
            "per_snr_additive_contribution": (embeddings @ projection.T).cpu().numpy(),
        }
    if model_id == "M4":
        return {"per_snr_logit_bias": model.logit_bias.weight.detach().cpu().numpy()}
    if model_id in {"M5", "M6"}:
        parameter = next(model.parameters())
        snr = torch.as_tensor(config["snr"]["values_db"], dtype=parameter.dtype, device=parameter.device)
        scalar = model.normalize_snr(snr).unsqueeze(1)
        with torch.no_grad():
            if model_id == "M5":
                gamma, beta = model.film(scalar).chunk(2, dim=1)
                return {"per_snr_gamma": gamma.cpu().numpy(), "per_snr_beta": beta.cpu().numpy()}
            gates = 2.0 * torch.sigmoid(model.gate(scalar))
            return {"per_snr_gates": gates.cpu().numpy()}
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.conditioner_parameters().items()}


def _outcome_from_model(model: Any, checkpoint: Path, dataset: dict[str, Any], split: Any, config: dict[str, Any], device: Any, **training: Any) -> dict[str, Any]:
    model.load_state_dict(_load_state(checkpoint, device), strict=True)
    logits = _evaluate(model, dataset, split, config, device)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    mechanisms = _mechanism_tensors(model, config)
    return {
        "checkpoint": checkpoint, "logits": logits, "labels": np.asarray(dataset["labels"])[test_idx],
        "snrs": np.asarray(dataset["snrs"])[test_idx], "mechanism_tensors": mechanisms,
        "complexity": _complexity(model, config, tuple(np.asarray(dataset["signals"]).shape[1:]), device), **training,
    }


def _training_loss(logits: Any, labels: Any, regularizers: Iterable[Any]):
    from torch.nn import functional as F
    return F.cross_entropy(logits, labels) + sum(regularizers)


def _should_checkpoint(validation_accuracy: float, best_accuracy: float) -> bool:
    """Preserve Phase 2 tie semantics: the latest equal-best epoch replaces prior."""
    return float(validation_accuracy) >= float(best_accuracy)


def _default_trainer(model_id: str, dataset: dict[str, Any], split: Any, destination: Path, logger: Callable[[str], None], *, config: dict[str, Any]) -> dict[str, Any]:
    if model_id in REUSED_MODELS:
        raise RuntimeError(f"{model_id} is reuse-only and must never be trained")
    import torch
    from torch import nn, optim
    from phase1_reproduce import make_historical_train_val_loaders
    from util.utils import fix_seed
    seed = int(config["_active_seed"])
    fix_seed(seed); random.seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    model = _model(config, model_id).to(device)
    signals = torch.from_numpy(np.asarray(dataset["signals"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(dataset["labels"], dtype=np.int64))
    snrs = torch.from_numpy(np.asarray(dataset["snrs"], dtype=np.float32))
    train_set, val_set = (make_conditioning_dataset(signals, labels, snrs, indices) for indices in (split.train_idx, split.val_idx))
    training = config["training"]
    train_loader, val_loader = make_historical_train_val_loaders(train_set, val_set, train_batch_size=int(training["batch_size"]), validation_batch_size=int(training["batch_size"]))
    optimizer = optim.Adam(model.parameters(), lr=float(training["lr"]))
    checkpoint = destination / "checkpoint.pt"
    best_accuracy, best_epoch = -1.0, -1
    started = time.perf_counter()
    for epoch in range(int(training["max_epochs"])):
        model.train()
        for x, y, snr_db, bins in train_loader:
            x, y, snr_db, bins = x.to(device), y.to(device), snr_db.to(device), bins.to(device)
            logits, regularizers = model.forward_batch(x, snr_db=snr_db, snr_bin=bins)
            loss = _training_loss(logits, y, regularizers)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y, snr_db, bins in val_loader:
                x, snr_db, bins = x.to(device), snr_db.to(device), bins.to(device)
                logits, _ = model.forward_batch(x, snr_db=snr_db, snr_bin=bins)
                correct += int((logits.argmax(1).cpu() == y).sum()); total += len(y)
        accuracy = correct / total
        if _should_checkpoint(accuracy, best_accuracy):
            best_accuracy, best_epoch = accuracy, epoch
            _atomic_torch_save(model.state_dict(), checkpoint)
    return _outcome_from_model(model, checkpoint, dataset, split, config, device,
        best_epoch=best_epoch, epochs_completed=int(training["max_epochs"]), best_val_accuracy=best_accuracy,
        duration_seconds=time.perf_counter() - started, scheduler={"name": "none", "reason": "Phase 2 protocol used no scheduler"})


def _load_phase2_module(root: Path):
    spec = importlib.util.spec_from_file_location("phase2_reuse_validation", root / "scripts/v2/run_phase2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 2 runner for reuse validation")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _find_phase2_source(model_id: str, seed: int, dataset: dict[str, Any], split: Any, config: dict[str, Any], root: Path) -> Path:
    if model_id not in REUSED_MODELS:
        raise ValueError("only M0 and M3 can reuse Phase 2")
    phase2 = _load_phase2_module(root)
    phase2_config = phase2.load_phase2_config(root / "configs/v2/phase2.yaml")
    treatment = "plain" if model_id == "M0" else "conditioned"
    phase2_code = phase2._code_identity(root)
    phase2_identity = {"runner": phase2.RUNNER_VERSION, "config": phase2_config, "split_hash": split.split_hash, "seed": seed, "treatment": treatment, "code": phase2_code}
    phase2_fingerprint = deterministic_json_hash(phase2_identity)
    source = phase2._matching_completed(root / "results/v2/phase2", treatment, seed, phase2_fingerprint,
        split=split, dataset=dataset, num_classes=int(config["architecture"]["num_classes"]), code=phase2_code)
    if source is None:
        raise RuntimeError(f"no strictly compatible completed Phase 2 {treatment} artifact for seed {seed}; {model_id} will not be retrained")
    source_spec = json.loads((source / "run_spec.json").read_text(encoding="utf-8"))
    source_result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if source_spec["protocol_config"] != phase2_config or source_spec["architecture"] != config["architecture"] or source_spec["training"] != config["training"]:
        raise RuntimeError("Phase 2 reuse protocol is not identical to Phase 3")
    if source_spec["fixed_split"]["hash"] != split.split_hash or source_result["provenance"]["split_hash"] != split.split_hash:
        raise RuntimeError("Phase 2 reuse split identity differs")
    return source


def _validate_phase2_reuse_provenance(model_id: str, seed: int, dataset: dict[str, Any], split: Any, reuse_source: str, source_hashes: dict[str, Any], *, config: dict[str, Any]) -> None:
    root = Path(config["_repository_root"])
    if not isinstance(reuse_source, str) or Path(reuse_source).is_absolute() or ".." in Path(reuse_source).parts:
        raise ValueError("Phase 2 reuse source must be repository-relative")
    expected_hash_fields = {"run_spec_sha256", "result_sha256", "checkpoint_sha256", "predictions_sha256"}
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_hash_fields or any(not _is_sha256(value) for value in source_hashes.values()):
        raise ValueError("Phase 2 source artifact hashes do not match the exact schema")
    source = _find_phase2_source(model_id, seed, dataset, split, config, root)
    if _relative(root, source) != Path(reuse_source).as_posix():
        raise ValueError("Phase 2 reuse source differs from the strictly validated source")
    actual = {
        "run_spec_sha256": file_sha256(source / "run_spec.json"), "result_sha256": file_sha256(source / "result.json"),
        "checkpoint_sha256": file_sha256(source / "checkpoint.pt"), "predictions_sha256": file_sha256(source / "predictions.npz"),
    }
    if source_hashes != actual:
        raise ValueError("Phase 2 reuse source hashes differ from current immutable artifacts")


def _default_phase2_reuser(model_id: str, seed: int, dataset: dict[str, Any], split: Any, destination: Path, logger: Callable[[str], None], *, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    source = _find_phase2_source(model_id, seed, dataset, split, config, root)
    source_result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    destination_checkpoint = destination / "checkpoint.pt"
    _atomic_copy(source / "checkpoint.pt", destination_checkpoint)
    import torch
    device = torch.device(config["device"])
    model = _load_phase2_compatible_model(model_id, destination_checkpoint, config).to(device)
    outcome = _outcome_from_model(model, destination_checkpoint, dataset, split, config, device,
        best_epoch=source_result["training"].get("best_epoch"), epochs_completed=source_result["training"].get("epochs_completed"),
        best_val_accuracy=source_result["training"].get("best_val_accuracy"), duration_seconds=0.0,
        scheduler={"name": "none", "reason": "Phase 2 protocol used no scheduler"})
    source_predictions = _validate_phase2_prediction_rows(source / "predictions.npz", split, dataset)
    if not np.array_equal(outcome["logits"].argmax(1), source_predictions):
        raise RuntimeError("strict checkpoint inference does not reproduce Phase 2 predictions")
    outcome["reuse_source"] = _relative(root, source)
    outcome["source_artifact_hashes"] = {
        "run_spec_sha256": file_sha256(source / "run_spec.json"), "result_sha256": file_sha256(source / "result.json"),
        "checkpoint_sha256": file_sha256(source / "checkpoint.pt"), "predictions_sha256": file_sha256(source / "predictions.npz"),
    }
    return outcome


def _manifest_record(root: Path, config: dict[str, Any], split: Any, model_id: str, seed: int, destination: Path, status: str, notes: str) -> ArtifactRecord:
    checkpoint = destination / "checkpoint.pt"
    return ArtifactRecord(
        experiment="phase3", dataset=config["dataset"]["id"], split_hash=split.split_hash,
        model=model_id, conditioner=str(config["models"][model_id]["conditioner"]), seed=seed,
        checkpoint=_relative(root, checkpoint) if checkpoint.is_file() else "",
        result_json=_relative(root, destination / "result.json"), figure_paths=[], status=status, notes=notes,
    )


def _run_phase3_unlocked(config_path: str | Path, *, repository_root: str | Path | None = None, trainer: Callable[..., dict[str, Any]] | None = None, phase2_reuser: Callable[..., dict[str, Any]] | None = None, dataset_loader: Callable[..., Any] | None = None, completed_verifier: Callable[..., dict[str, Any]] | None = None, phase2_source_validator: Callable[..., None] | None = None, logger: Callable[[str], None] = print, smoke: bool = False) -> int:
    if smoke and (trainer is None or phase2_reuser is None):
        raise ValueError("smoke mode requires injected trainer and Phase 2 reuser")
    root = Path(repository_root) if repository_root is not None else ROOT
    config = load_phase3_config(config_path)
    paths = validate_output_paths(root, config["outputs"])
    output_root = paths["root"]; output_root.mkdir(parents=True, exist_ok=True)
    split = load_split(_resolve(root, config["split"]["metadata"]), data_path=_resolve(root, config["dataset"]["path"]))
    if split.split_hash != LOCKED_SPLIT_HASH:
        raise ValueError("loaded split hash differs from designated fixed split")
    if dataset_loader is None:
        from phase1_reproduce import _load_rml_dataset
        dataset = _load_rml_dataset(_resolve(root, config["dataset"]["path"]), config["dataset"]["id"], repository_root=root)
    else:
        dataset = dataset_loader(_resolve(root, config["dataset"]["path"]), config["dataset"]["id"])
    code = _code_identity(root)
    environment = _environment_identity()
    protocol = protocol_hash(config); preprocessing = preprocessing_hash(config)
    store = ManifestStore(paths["manifest"])
    ledger_path = output_root / "phase3_ledger.json"
    ledger = _load_ledger(ledger_path)
    failures: list[str] = []
    accounted: dict[tuple[str, int], tuple[str, str]] = {}
    for seed in config["seeds"]:
        for model_id in MODEL_IDS:
            run_identity = {"runner": RUNNER_VERSION, "protocol_hash": protocol, "preprocessing_hash": preprocessing, "split_hash": split.split_hash, "model_id": model_id, "seed": seed, "code": code, "environment": environment}
            fingerprint = deterministic_json_hash(run_identity)
            bundle_identity = {"model_id": model_id, "seed": seed, "split_hash": split.split_hash, "preprocessing_hash": preprocessing, "protocol_hash": protocol}
            reusable = _matching_completed(output_root, model_id, seed, fingerprint, root=root, ledger=ledger, split=split, dataset=dataset, identity=bundle_identity, config=config, code=code, environment=environment, completed_verifier=completed_verifier, phase2_source_validator=phase2_source_validator, logger=logger)
            if reusable is not None:
                result = json.loads((reusable / "result.json").read_text(encoding="utf-8"))
                mode = result.get("execution", {}).get("mode", "trained")
                accounted[(model_id, seed)] = ("completed", mode)
                store.upsert(_manifest_record(root, config, split, model_id, seed, reusable, "completed", f"validated resume ({mode})"))
                logger(f"{model_id} seed={seed}: validated resume {reusable.name}")
                continue
            destination = _next_destination(output_root, model_id, seed, fingerprint); destination.mkdir(exist_ok=False)
            run_spec = {
                "schema_version": 1, "runner_version": RUNNER_VERSION, "created_at": _now(), "model_id": model_id,
                "seed": seed, "run_fingerprint": fingerprint, "protocol_hash": protocol, "preprocessing_hash": preprocessing,
                "protocol_config": config, "fixed_split": {"metadata": config["split"]["metadata"], "hash": split.split_hash},
                "execution_mode": "reused_phase2" if model_id in REUSED_MODELS else "trained_phase3", "provenance": {"code": code, "environment": environment},
            }
            _atomic_json(destination / "run_spec.json", run_spec)
            status = "failed"; mode = "reused" if model_id in REUSED_MODELS else "trained"
            try:
                active = dict(config); active["_active_seed"] = seed; active["_repository_root"] = str(root.resolve())
                executor = (phase2_reuser or _default_phase2_reuser) if model_id in REUSED_MODELS else (trainer or _default_trainer)
                if model_id in REUSED_MODELS:
                    outcome = executor(model_id, seed, dataset, split, destination, logger, config=active)
                else:
                    outcome = executor(model_id, dataset, split, destination, logger, config=active)
                checkpoint = Path(outcome["checkpoint"])
                if not checkpoint.is_file() or checkpoint.resolve() != (destination / "checkpoint.pt").resolve():
                    raise RuntimeError("executor must persist checkpoint.pt in the run directory")
                predictions = destination / "predictions.npz"; mechanism = destination / "mechanism.npz"
                write_prediction_bundle(predictions, outcome, split, bundle_identity)
                metrics = validate_prediction_bundle(predictions, split, dataset, identity=bundle_identity, num_classes=int(config["architecture"]["num_classes"]), snr_values=config["snr"]["values_db"])
                mechanism_identity = bundle_identity | {"checkpoint_sha256": file_sha256(checkpoint)}
                write_mechanism_bundle(mechanism, outcome, mechanism_identity)
                validate_mechanism_bundle(mechanism, identity=mechanism_identity, architecture=config["architecture"], pooled_feature_dim=int(config["models"]["M5"]["modulation_dimension"]), expected_tensors=outcome["mechanism_tensors"])
                complexity = outcome.get("complexity")
                validate_complexity(complexity, fixed_batch=int(config["evaluation_batch_size"]))
                source_hashes = outcome.get("source_artifact_hashes", {})
                if model_id in REUSED_MODELS:
                    validator = phase2_source_validator or _validate_phase2_reuse_provenance
                    validator(model_id, seed, dataset, split, outcome.get("reuse_source"), source_hashes, config=active)
                result = {
                    "schema_version": 1, "runner_version": RUNNER_VERSION, "status": "completed", "model_id": model_id,
                    "seed": seed, "completed_at": _now(), "execution": {"mode": mode, "reuse_source": outcome.get("reuse_source"), "source_artifact_hashes": outcome.get("source_artifact_hashes", {})},
                    "metrics": metrics, "complexity": complexity,
                    "training": {key: outcome.get(key) for key in ("best_epoch", "epochs_completed", "best_val_accuracy", "duration_seconds", "scheduler")},
                    "artifacts": {
                        "run_spec": "run_spec.json", "run_spec_sha256": file_sha256(destination / "run_spec.json"),
                        "checkpoint": "checkpoint.pt", "checkpoint_sha256": file_sha256(checkpoint),
                        "predictions": "predictions.npz", "predictions_sha256": file_sha256(predictions),
                        "mechanism": "mechanism.npz", "mechanism_sha256": file_sha256(mechanism),
                    },
                    "provenance": {"run_fingerprint": fingerprint, "split_hash": split.split_hash, "preprocessing_hash": preprocessing, "protocol_hash": protocol, "code": code, "environment": environment},
                }
                _validate_result_schema(result, model_id=model_id, seed=seed, fingerprint=fingerprint, split_hash=split.split_hash, preprocessing=preprocessing, protocol=protocol, code=code, environment=environment, fixed_batch=int(config["evaluation_batch_size"]))
                _atomic_json(destination / "result.json", result)
                ledger_record = {"model_id": model_id, "seed": seed, "run_fingerprint": fingerprint, "attempt_path": _relative(root, destination), "result_sha256": file_sha256(destination / "result.json"), "status": "completed", "updated_at": _now()}
                _ledger_upsert(ledger_path, ledger_record); ledger = _load_ledger(ledger_path)
                status = "completed"; notes = f"{mode}; completed outcome retained regardless of effect direction"
            except Exception as exc:
                failure = {"status": "failed", "failed_at": _now(), "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                _atomic_json(destination / "failure.json", failure)
                _atomic_json(destination / "result.json", {
                    "schema_version": 1, "runner_version": RUNNER_VERSION, "status": "failed", "model_id": model_id, "seed": seed,
                    "failure": failure, "artifacts": {"run_spec": "run_spec.json", "failure": "failure.json"},
                    "provenance": {"run_fingerprint": fingerprint, "split_hash": split.split_hash, "preprocessing_hash": preprocessing, "protocol_hash": protocol, "code": code, "environment": environment},
                })
                ledger_record = {"model_id": model_id, "seed": seed, "run_fingerprint": fingerprint, "attempt_path": _relative(root, destination), "result_sha256": file_sha256(destination / "result.json"), "status": "failed", "updated_at": _now()}
                _ledger_upsert(ledger_path, ledger_record); ledger = _load_ledger(ledger_path)
                failures.append(f"{model_id}/seed{seed}: {exc}"); notes = str(exc)
            accounted[(model_id, seed)] = (status, mode)
            store.upsert(_manifest_record(root, config, split, model_id, seed, destination, status, notes))
    planned = len(MODEL_IDS) * len(config["seeds"])
    if len(accounted) == planned:
        reused_count = sum(mode == "reused" for _status, mode in accounted.values())
        trained_count = sum(mode == "trained" for _status, mode in accounted.values())
        completed_count = sum(status == "completed" for status, _mode in accounted.values())
        failed_count = sum(status == "failed" for status, _mode in accounted.values())
        pending_count = planned - completed_count - failed_count
        all_complete = completed_count == planned
        completion_text = (f"COMPLETE: {completed_count}/{planned} completed; {reused_count} reused and {trained_count} trained." if all_complete else f"INCOMPLETE: {completed_count} completed, {failed_count} failed, {pending_count} pending out of {planned}; planned modes are {reused_count} reused and {trained_count} trained.")
        update_phase_progress(
            "Phase 3", completed=completion_text,
            failed="None." if not failures else "\n".join(failures), unexpected="None.",
            interpretation="Metrics and mechanism evidence are recomputed from machine-readable artifacts; completed negative results are retained.",
            next_gate=("Proceed after all 40 records and their hashes pass audit." if all_complete else "BLOCKED: resolve failed and pending records before Phase 3 interpretation."), path=paths["progress"],
        )
    return 1 if failures else 0


def run_phase3(config_path: str | Path, *, repository_root: str | Path | None = None, trainer: Callable[..., dict[str, Any]] | None = None, phase2_reuser: Callable[..., dict[str, Any]] | None = None, dataset_loader: Callable[..., Any] | None = None, completed_verifier: Callable[..., dict[str, Any]] | None = None, phase2_source_validator: Callable[..., None] | None = None, logger: Callable[[str], None] = print, smoke: bool = False, lock_timeout_seconds: float = 10.0) -> int:
    root = Path(repository_root) if repository_root is not None else ROOT
    try:
        config = load_phase3_config(config_path)
        output_root = validate_output_paths(root, config["outputs"])["root"]
        lock = PhaseExecutionLock(output_root / ".phase3.lock", timeout_seconds=lock_timeout_seconds)
        lock.acquire()
    except (ValueError, ReproductionValidationError) as exc:
        logger(str(exc)); return 3
    try:
        return _run_phase3_unlocked(config_path, repository_root=root, trainer=trainer, phase2_reuser=phase2_reuser, dataset_loader=dataset_loader, completed_verifier=completed_verifier, phase2_source_validator=phase2_source_validator, logger=logger, smoke=smoke)
    finally:
        lock.release()


def smoke_check(config_path: str | Path) -> dict[str, list[int]]:
    """Exercise every declared ``forward_batch`` path without data or writes."""
    import torch
    config = load_phase3_config(config_path)
    signals = torch.zeros((2, 2, 128), dtype=torch.float32)
    snr_db = torch.tensor([config["snr"]["snr_min_db"], config["snr"]["snr_max_db"]], dtype=torch.float32)
    snr_bin = torch.tensor([0, len(config["snr"]["values_db"]) - 1], dtype=torch.long)
    summary: dict[str, list[int]] = {}
    with torch.no_grad():
        for model_id in MODEL_IDS:
            model = _model(config, model_id).eval()
            logits, regularizers = model.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)
            expected = (2, int(config["architecture"]["num_classes"]))
            if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()) or not all(bool(torch.isfinite(item).all()) for item in regularizers):
                raise RuntimeError(f"{model_id} forward_batch smoke check failed")
            summary[model_id] = list(logits.shape)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2/phase3.yaml")
    parser.add_argument("--smoke", action="store_true", help="exercise all forward_batch paths without data access or artifact writes")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke_check(args.config), sort_keys=True))
        return 0
    return run_phase3(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

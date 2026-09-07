#!/usr/bin/env python3
"""Reproducible, resumable Phase 2 controlled RML2016.10a experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASE1_DIR = Path(__file__).resolve().parent
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))
from phase1_reproduce import PhaseExecutionLock, ReproductionValidationError
from v2.contracts import ArtifactRecord
from v2.manifest import ManifestStore
from v2.progress import update_phase_progress
from v2.provenance import deterministic_json_hash, file_sha256
from v2.splits import format_snr, load_split
from v2.statistics import paired_summary

RUNNER_VERSION = "phase2-controlled/1"
REQUIRED_TREATMENTS = ("plain", "conditioned", "focal", "snr_reweighted")
SOURCE_FILES = (
    "scripts/v2/run_phase2.py", "scripts/v2/phase1_reproduce.py",
    "models/model.py", "models/model_snr.py", "models/lifting.py",
    "util/utils.py", "v2/splits.py", "v2/statistics.py", "v2/metrics.py",
)
LOCKED_DATASET = {"id": "RML2016.10a", "path": "data/RML2016.10a_dict.pkl"}
LOCKED_SPLIT = "splits/v2/RML2016.10a_seed2022.json"
LOCKED_SPLIT_HASH = "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"
LOCKED_SEEDS = [2022, 2023, 2024, 2025, 2026]


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


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _repo_output_path(root: Path, value: str, field: str, prefixes: tuple[str, ...]) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{field} must be a repository-relative allowlisted path")
    normalized = Path(value).as_posix()
    if not any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes):
        raise ValueError(f"{field} is outside its allowlisted location")
    path = (root / normalized).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{field} escapes repository root")
    return path


def load_phase2_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Phase 2 configuration requires schema_version: 1")
    validate_phase2_config(config)
    return config


def validate_phase2_config(config: dict[str, Any]) -> None:
    """Reject mutations to the predeclared dataset, split, and seed protocol."""
    if config.get("dataset") != LOCKED_DATASET:
        raise ValueError("dataset must be exactly the designated RML2016.10a input")
    if config.get("split") != {"metadata": LOCKED_SPLIT, "hash": LOCKED_SPLIT_HASH}:
        raise ValueError("split must be the designated Phase 1 fixed split identity and hash")
    if config.get("seeds") != LOCKED_SEEDS:
        raise ValueError(f"seeds must be exactly {LOCKED_SEEDS}")
    if not isinstance(config.get("seeds"), list) or not config["seeds"]:
        raise ValueError("seeds must be a non-empty list")
    if len(set(config["seeds"])) != len(config["seeds"]) or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("seeds must be unique integers")
    treatments = config.get("treatments")
    if not isinstance(treatments, dict) or set(treatments) != set(REQUIRED_TREATMENTS):
        raise ValueError(f"treatments must be exactly {list(REQUIRED_TREATMENTS)}")
    expected_semantics = {
        "plain": {"conditioner": "none", "loss": "cross_entropy", "snr_reweighting": "none"},
        "conditioned": {"conditioner": "true_snr_embedding", "loss": "cross_entropy", "snr_reweighting": "none"},
        "focal": {"conditioner": "none", "loss": "focal", "gamma": 2.0, "snr_reweighting": "none"},
    }
    for name, expected in expected_semantics.items():
        if treatments.get(name) != expected:
            raise ValueError(f"{name} treatment semantics must exactly match the controlled contrast")
    weighted = treatments["snr_reweighted"]
    expected_weighted = {"conditioner": "none", "loss": "cross_entropy", "snr_reweighting": "fixed_segment_weights", "segment_weights": {"low": 2.0, "mid": 1.0, "high": 0.5}}
    if weighted != expected_weighted:
        raise ValueError("snr_reweighted must use the locked predeclared segment weights")
    if config.get("training", {}).get("optimizer") != "Adam":
        raise ValueError("training.optimizer must be exactly Adam")


def snr_segment_weights(snrs: Any, treatment_spec: dict[str, Any]) -> list[float]:
    """Return fixed predeclared loss weights; independent of split frequencies."""
    if treatment_spec.get("snr_reweighting") != "fixed_segment_weights":
        raise ValueError("only fixed_segment_weights is supported")
    weights = treatment_spec["segment_weights"]
    values = np.asarray(snrs, dtype=float)
    return [float(weights["low"] if value <= -8 else weights["mid"] if value <= -2 else weights["high"]) for value in values]


def validate_output_paths(root: Path, outputs: Any) -> dict[str, Path]:
    if not isinstance(outputs, dict) or set(outputs) != {"root", "manifest", "progress"}:
        raise ValueError("outputs must contain exactly root, manifest, and progress")
    return {"root": _repo_output_path(root, outputs["root"], "outputs.root", ("results/v2/phase2",)),
            "manifest": _repo_output_path(root, outputs["manifest"], "outputs.manifest", ("manifest.json", "results/v2/phase2")),
            "progress": _repo_output_path(root, outputs["progress"], "outputs.progress", ("reports/v2_progress.md", "reports/v2_phase2"))}


def _sample_ids(split: Any) -> np.ndarray:
    return np.asarray(split.sample_ids)[np.asarray(split.test_idx, dtype=np.int64)]


def _write_predictions(path: Path, outcome: dict[str, Any], split: Any) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, predictions=np.asarray(outcome["predictions"], dtype=np.int64),
                                labels=np.asarray(outcome["labels"], dtype=np.int64), snrs=np.asarray(outcome["snrs"]),
                                sample_ids=_sample_ids(split))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _confusion(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def _metrics(labels: Any, predictions: Any, snrs: Any, *, n_classes: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    snrs = np.asarray(snrs)
    if labels.ndim != 1 or not (len(labels) == len(predictions) == len(snrs)) or not len(labels):
        raise ValueError("predictions, labels, and snrs must be equal non-empty vectors")
    if np.any(labels < 0) or np.any(labels >= n_classes) or np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise ValueError("predictions and labels must be valid class indices")
    matrix = _confusion(labels, predictions, n_classes)
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.full(n_classes, np.nan), where=support != 0)
    precision_denominator = matrix.sum(axis=0)
    precision = np.divide(np.diag(matrix), precision_denominator, out=np.zeros(n_classes), where=precision_denominator != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes), where=(precision + recall) != 0)
    accuracy = float(np.mean(labels == predictions))
    expected = float(np.dot(matrix.sum(0), matrix.sum(1)) / (len(labels) ** 2))
    kappa = (accuracy - expected) / (1 - expected) if expected != 1 else float("nan")
    counts = matrix.sum(axis=0)
    shares = counts / len(labels)
    entropy = -float(np.sum(shares[shares > 0] * np.log(shares[shares > 0])) / np.log(n_classes)) if n_classes > 1 else float("nan")
    gini = float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * n_classes))
    per_snr = {format_snr(float(s)): float(np.mean(predictions[snrs == s] == labels[snrs == s])) for s in sorted(np.unique(snrs))}
    segments = {name: (float(np.mean(predictions[mask] == labels[mask])) if mask.any() else None) for name, mask in {
        "low": snrs <= -8, "mid": (snrs >= -6) & (snrs <= -2), "high": snrs >= 0}.items()}
    dominant = int(np.argmax(counts))
    balanced = float(np.mean(recall))
    return _json_value({"overall_accuracy": accuracy, "per_snr_accuracy": per_snr, "segments": segments,
        "macro_f1": float(np.mean(f1)), "balanced_accuracy": balanced, "kappa": float(kappa),
        "prediction_concentration": float(shares.max()), "prediction_entropy": entropy,
        "gini_prediction_distribution": gini, "hhi": float(np.sum(shares ** 2)), "concentration_minus_balanced_accuracy": float(shares.max() - balanced),
        "per_class_recall": {str(index): float(value) for index, value in enumerate(recall)},
        "dominant_predicted_class": dominant, "confusion_counts": matrix.tolist(),
        "feature_geometry": {"available": False, "reason": "model_interface_does_not_expose_features"}})


def recompute_metrics(predictions_path: str | Path, *, num_classes: int) -> dict[str, Any]:
    """Recompute persisted scalar outcomes from the pickle-free prediction bundle."""
    with np.load(predictions_path, allow_pickle=False) as bundle:
        required = {"predictions", "labels", "snrs", "sample_ids"}
        if set(bundle.files) != required:
            raise ValueError(f"prediction bundle fields must be exactly {sorted(required)}")
        return _metrics(bundle["labels"], bundle["predictions"], bundle["snrs"], n_classes=num_classes)


def validate_prediction_bundle(path: str | Path, split: Any, dataset: dict[str, Any], *, num_classes: int) -> dict[str, Any]:
    """Bind predictions to the exact fixed test indices before reporting them."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {"predictions", "labels", "snrs", "sample_ids"}
        if set(bundle.files) != required:
            raise ValueError(f"prediction bundle fields must be exactly {sorted(required)}")
        values = {name: bundle[name] for name in required}
    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    if len(values["predictions"]) != len(test_idx):
        raise ValueError("prediction count does not match fixed test split")
    if not np.array_equal(values["sample_ids"], _sample_ids(split)) or len(np.unique(values["sample_ids"])) != len(test_idx):
        raise ValueError("prediction sample IDs do not match fixed test split")
    if not np.array_equal(values["labels"], np.asarray(dataset["labels"])[test_idx]):
        raise ValueError("prediction labels do not match fixed test split")
    if not np.array_equal(values["snrs"], np.asarray(dataset["snrs"])[test_idx]):
        raise ValueError("prediction SNR values do not match fixed test split")
    return _metrics(values["labels"], values["predictions"], values["snrs"], n_classes=num_classes)


def _default_trainer(treatment: str, dataset: dict[str, Any], split: Any, destination: Path, logger: Callable[[str], None], *, config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the Phase 1 AWN mechanics, varying only the declared treatment."""
    import random
    import torch
    from torch import nn, optim
    from models.model import AWN
    from models.model_snr import AWNSNR
    from util.utils import fix_seed
    phase1_dir = Path(__file__).parent
    if str(phase1_dir) not in sys.path: sys.path.insert(0, str(phase1_dir))
    from phase1_reproduce import make_historical_loader, make_historical_train_val_loaders, make_index_dataset
    seed = int(config["_active_seed"]); fix_seed(seed); random.seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    architecture = config["architecture"]; device = torch.device(config.get("device", "cpu")); conditioned = treatment == "conditioned"
    common = {key: architecture[key] for key in ("num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim", "regu_details", "regu_approx")}
    model = (AWNSNR(**(common | {"num_snr_bins": architecture.get("num_snr_bins", 20), "snr_emb_dim": architecture.get("snr_embedding_dim", 8)})) if conditioned else AWN(**common)).to(device)
    signals = torch.from_numpy(np.asarray(dataset["signals"], dtype=np.float32)); labels = torch.from_numpy(np.asarray(dataset["labels"], dtype=np.int64)); snrs = torch.from_numpy(np.asarray(dataset["snrs"], dtype=np.float32))
    train_set, val_set = (make_index_dataset(signals, labels, snrs, idx) for idx in (split.train_idx, split.val_idx))
    training = config["training"]; train_loader, val_loader = make_historical_train_val_loaders(train_set, val_set, train_batch_size=training["batch_size"], validation_batch_size=training["batch_size"])
    optimizer = optim.Adam(model.parameters(), lr=float(training["lr"])); checkpoint = destination / "checkpoint.pt"; best = (-1.0, -1); started = time.perf_counter()
    weights = None
    if treatment == "snr_reweighted":
        # Fixed before execution: emphasizes difficult low-SNR samples even
        # though this stratified split is frequency-balanced by SNR.
        weights = np.asarray(snr_segment_weights(np.arange(20) * 2 - 20, config["treatments"][treatment]), dtype=np.float32)
    for epoch in range(int(training["max_epochs"])):
        model.train()
        for x, y, bins in train_loader:
            x, y, bins = x.to(device), y.to(device), bins.to(device); logits, regs = (model(x, bins) if conditioned else model(x)); ce = nn.functional.cross_entropy(logits, y, reduction="none")
            if treatment == "focal": ce = (1 - torch.exp(-ce)).pow(float(config["treatments"][treatment].get("gamma", 2.0))) * ce
            if weights is not None:
                bin_weights = torch.as_tensor(weights[bins.cpu().numpy()], dtype=ce.dtype, device=device); ce = ce * bin_weights
            loss = ce.mean() + sum(regs); optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y, bins in val_loader:
                logits, _ = (model(x.to(device), bins.to(device)) if conditioned else model(x.to(device))); correct += int((logits.argmax(1).cpu() == y).sum()); total += len(y)
        if correct / total >= best[0]: best = (correct / total, epoch); torch.save(model.state_dict(), checkpoint)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only keyword
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state); model.eval(); test_idx = np.asarray(split.test_idx, dtype=np.int64); test_set = make_index_dataset(signals, labels, snrs, test_idx); parts=[]
    with torch.no_grad():
        for x, _y, bins in make_historical_loader(test_set, batch_size=int(config.get("evaluation_batch_size", 64)), shuffle=False):
            logits, _ = (model(x.to(device), bins.to(device)) if conditioned else model(x.to(device))); parts.append(logits.argmax(1).cpu().numpy())
    return {"checkpoint": checkpoint, "predictions": np.concatenate(parts), "labels": np.asarray(dataset["labels"])[test_idx], "snrs": np.asarray(dataset["snrs"])[test_idx], "best_epoch": best[1], "epochs_completed": int(training["max_epochs"]), "best_val_accuracy": best[0], "duration_seconds": time.perf_counter()-started, "parameter_count": sum(p.numel() for p in model.parameters())}


def _matching_completed(output_root: Path, treatment: str, seed: int, fingerprint: str, *, split: Any, dataset: dict[str, Any], num_classes: int, code: dict[str, Any]) -> Path | None:
    for candidate in output_root.glob(f"{treatment}_seed{seed}_*"):
        try:
            spec_path, result_path, prediction_path, checkpoint_path = (candidate / name for name in ("run_spec.json", "result.json", "predictions.npz", "checkpoint.pt"))
            spec = json.loads(spec_path.read_text(encoding="utf-8")); result = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = result.get("artifacts", {})
            if (spec.get("run_fingerprint") != fingerprint or spec.get("provenance", {}).get("code") != code or result.get("status") != "completed" or not checkpoint_path.is_file() or not prediction_path.is_file()):
                continue
            if artifacts.get("run_spec_sha256") != file_sha256(spec_path) or artifacts.get("predictions_sha256") != file_sha256(prediction_path) or artifacts.get("checkpoint_sha256") != file_sha256(checkpoint_path):
                continue
            if result.get("metrics") != validate_prediction_bundle(prediction_path, split, dataset, num_classes=num_classes):
                continue
            return candidate
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return None


def _next_destination(output_root: Path, treatment: str, seed: int, fingerprint: str) -> Path:
    base = output_root / f"{treatment}_seed{seed}_{fingerprint[:12]}"
    if not base.exists():
        return base
    retry = 1
    while (candidate := output_root / f"{base.name}_retry{retry}").exists():
        retry += 1
    return candidate


def _code_identity(root: Path) -> dict[str, Any]:
    files = {name: file_sha256(root / name) for name in SOURCE_FILES}
    return {"files": files, "hash": deterministic_json_hash(files)}


def _run_phase2_unlocked(config_path: str | Path, *, repository_root: str | Path | None = None, trainer: Callable[..., dict[str, Any]] | None = None, dataset_loader: Callable[..., Any] | None = None, logger: Callable[[str], None] = print) -> int:
    root = Path(repository_root) if repository_root is not None else ROOT; config = load_phase2_config(config_path); paths = validate_output_paths(root, config["outputs"]); output_root = paths["root"]; output_root.mkdir(parents=True, exist_ok=True)
    split_path = _resolve(root, config["split"]["metadata"]); split = load_split(split_path, data_path=_resolve(root, config["dataset"]["path"]));
    if split.split_hash != LOCKED_SPLIT_HASH:
        raise ValueError("loaded split hash differs from designated Phase 1 fixed split")
    if dataset_loader is None:
        phase1_dir = Path(__file__).parent
        if str(phase1_dir) not in sys.path: sys.path.insert(0, str(phase1_dir))
        from phase1_reproduce import _load_rml_dataset
        dataset = _load_rml_dataset(_resolve(root, config["dataset"]["path"]), config["dataset"]["id"], repository_root=root)
    else: dataset = dataset_loader(_resolve(root, config["dataset"]["path"]), config["dataset"]["id"])
    code = _code_identity(root)
    store = ManifestStore(paths["manifest"]); completed: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in REQUIRED_TREATMENTS}; failures = []
    for seed in config["seeds"]:
        for treatment in REQUIRED_TREATMENTS:
            identity = {"runner": RUNNER_VERSION, "config": config, "split_hash": split.split_hash, "seed": seed, "treatment": treatment, "code": code}; fingerprint = deterministic_json_hash(identity); reusable = _matching_completed(output_root, treatment, seed, fingerprint, split=split, dataset=dataset, num_classes=int(config["architecture"]["num_classes"]), code=code)
            if reusable:
                result = json.loads((reusable / "result.json").read_text(encoding="utf-8")); completed[treatment][seed] = result; logger(f"{treatment} seed={seed}: reused {reusable.name}"); continue
            destination = _next_destination(output_root, treatment, seed, fingerprint); destination.mkdir(exist_ok=False); run_spec = {"schema_version": 1, "runner_version": RUNNER_VERSION, "created_at": _now(), "treatment": treatment, "seed": seed, "run_fingerprint": fingerprint, "protocol_config": config, "fixed_split": {"metadata": str(split_path), "hash": split.split_hash}, "dataset": config["dataset"], "training": config["training"], "architecture": config["architecture"], "treatment_spec": config["treatments"][treatment], "provenance": {"code": code}}; _atomic_json(destination / "run_spec.json", run_spec)
            try:
                active = dict(config); active["_active_seed"] = seed; outcome = (trainer or _default_trainer)(treatment, dataset, split, destination, logger, config=active); checkpoint = Path(outcome["checkpoint"])
                if not checkpoint.is_file() or checkpoint.resolve() != (destination / "checkpoint.pt").resolve(): raise RuntimeError("trainer must persist checkpoint.pt in the run directory")
                prediction_path = destination / "predictions.npz"; _write_predictions(prediction_path, outcome, split); metrics = validate_prediction_bundle(prediction_path, split, dataset, num_classes=int(config["architecture"]["num_classes"]))
                result = {"schema_version": 1, "runner_version": RUNNER_VERSION, "status": "completed", "treatment": treatment, "seed": seed, "completed_at": _now(), "metrics": metrics, "training": {key: outcome.get(key) for key in ("best_epoch", "epochs_completed", "best_val_accuracy", "duration_seconds", "parameter_count")}, "artifacts": {"run_spec": "run_spec.json", "run_spec_sha256": file_sha256(destination / "run_spec.json"), "checkpoint": "checkpoint.pt", "checkpoint_sha256": file_sha256(checkpoint), "predictions": "predictions.npz", "predictions_sha256": file_sha256(prediction_path)}, "provenance": {"run_fingerprint": fingerprint, "split_hash": split.split_hash, "code": code}}; _atomic_json(destination / "result.json", result); completed[treatment][seed] = result; status="completed"; notes="controlled result retained; negative results are completed"
            except Exception as exc:
                failure = {"status":"failed", "failed_at":_now(), "type":type(exc).__name__, "message":str(exc), "traceback":traceback.format_exc()}
                _atomic_json(destination / "failure.json", failure)
                _atomic_json(destination / "result.json", {"schema_version":1, "runner_version":RUNNER_VERSION, "status":"failed", "treatment":treatment, "seed":seed, "failure":failure, "artifacts":{"run_spec":"run_spec.json", "failure":"failure.json"}, "provenance":{"run_fingerprint":fingerprint, "split_hash":split.split_hash}})
                failures.append(f"{treatment}/seed{seed}: {exc}"); status="failed"; notes=str(exc)
            checkpoint_record = str(destination / "checkpoint.pt") if (destination / "checkpoint.pt").is_file() else ""
            store.upsert(ArtifactRecord(experiment="phase2", dataset=config["dataset"]["id"], split_hash=split.split_hash, model=treatment, conditioner=str(config["treatments"][treatment].get("conditioner", "none")), seed=seed, checkpoint=checkpoint_record, result_json=str(destination / "result.json"), figure_paths=[], status=status, notes=notes))
    comparisons = {}
    for treatment in REQUIRED_TREATMENTS[1:]:
        seeds = [seed for seed in config["seeds"] if seed in completed["plain"] and seed in completed[treatment]]
        comparisons[f"{treatment}_vs_plain"] = {"status": "complete" if len(seeds) == len(config["seeds"]) else "incomplete_invalid", "seeds": seeds, "missing_seeds": [seed for seed in config["seeds"] if seed not in seeds], "overall_accuracy": _json_value(paired_summary([completed["plain"][seed]["metrics"]["overall_accuracy"] for seed in seeds], [completed[treatment][seed]["metrics"]["overall_accuracy"] for seed in seeds]))}
    _atomic_json(output_root / "paired_summary.json", {"schema_version":1, "runner_version":RUNNER_VERSION, "comparisons":comparisons})
    planned = len(config["seeds"])
    incomplete = [name for name, comparison in comparisons.items() if len(comparison["seeds"]) != planned]
    interpretation = "Paired summaries compare each treatment to plain without hand-entered values."
    if incomplete:
        interpretation += f" INCOMPLETE/INVALID planned five-seed comparisons: {', '.join(incomplete)}."
    update_phase_progress("Phase 2", completed=f"Completed {sum(len(item) for item in completed.values())} controlled runs; retained all completed outcomes.", failed="None." if not failures else "\n".join(failures), unexpected="None." if not incomplete else f"Incomplete planned comparisons: {', '.join(incomplete)}.", interpretation=interpretation, next_gate="Review paired effect estimates and artifacts before interpretation.", path=paths["progress"])
    return 1 if failures else 0


def run_phase2(config_path: str | Path, *, repository_root: str | Path | None = None, trainer: Callable[..., dict[str, Any]] | None = None, dataset_loader: Callable[..., Any] | None = None, logger: Callable[[str], None] = print, lock_timeout_seconds: float = 10.0) -> int:
    """Serialize destination allocation through final publication for one protocol root."""
    root = Path(repository_root) if repository_root is not None else ROOT
    try:
        config = load_phase2_config(config_path)
        output_root = validate_output_paths(root, config["outputs"])["root"]
        lock = PhaseExecutionLock(output_root / ".phase2.lock", timeout_seconds=lock_timeout_seconds)
        lock.acquire()
    except (ValueError, ReproductionValidationError) as exc:
        logger(str(exc))
        return 3
    try:
        return _run_phase2_unlocked(config_path, repository_root=root, trainer=trainer, dataset_loader=dataset_loader, logger=logger)
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v2/phase2.yaml"); return run_phase2(parser.parse_args(argv).config)
if __name__ == "__main__": raise SystemExit(main())

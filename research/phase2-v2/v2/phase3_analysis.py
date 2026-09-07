"""Read-only Phase 3 conditioning analysis and atomic evidence publication."""

from __future__ import annotations

import hashlib
import csv
import importlib.util
import json
import os
import shutil
import socket
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from v2.statistics import paired_summary


MODEL_IDS = tuple(f"M{index}" for index in range(8))
SEEDS = (2022, 2023, 2024, 2025, 2026)
SNR_VALUES = tuple(range(-20, 20, 2))
LOCKED_SPLIT_HASH = "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"
EXPECTED_PAIRS = tuple((f"M{index}", "M0") for index in range(1, 8)) + (("M5", "M3"), ("M6", "M3"), ("M7", "M3"))
DEFAULT_SOURCE_MANIFEST = "manifest.json"
DEFAULT_OUTPUT = "results/v2/phase3_conditioning"
OVERALL_METRICS = ("overall_accuracy", "macro_f1", "balanced_accuracy", "nll", "ece")
PER_SNR_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "prediction_concentration", "normalized_prediction_entropy", "gini", "hhi", "dominant_predicted_class_ratio")
ANALYSIS_SCHEMA_VERSION = 2
_COMPLETED_VERIFIER_OVERRIDE = None
_PHASE2_SOURCE_VALIDATOR_OVERRIDE = None


class AnalysisNotReadyError(RuntimeError):
    """Raised when the registered Phase 3 matrix is incomplete or invalid."""


class AnalysisConflictError(RuntimeError):
    """Raised rather than overwriting an existing analysis publication."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(root: Path, value: str, *, prefix: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("artifact paths must be portable repository-relative paths")
    normalized = Path(value).as_posix()
    if normalized != prefix and not normalized.startswith(prefix + "/"):
        raise ValueError(f"artifact path is outside {prefix}")
    resolved = (root / normalized).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes repository root")
    return resolved


def _scalar(bundle: Any, field: str) -> Any:
    value = bundle[field]
    if value.ndim != 0:
        raise ValueError(f"prediction identity {field} must be scalar")
    return value.item()


def _snr_key(value: int | float) -> str:
    return f"{float(value):+.17g}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def _group_metrics(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict[str, Any]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support = matrix.sum(1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(classes, dtype=float), where=support != 0)
    predicted_support = matrix.sum(0)
    precision = np.divide(np.diag(matrix), predicted_support, out=np.zeros(classes, dtype=float), where=predicted_support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
    shares = predicted_support / len(labels)
    positive = shares[shares > 0]
    normalized_entropy = -float(np.sum(positive * np.log(positive))) / np.log(classes) if classes > 1 else 0.0
    dominant = int(np.argmax(shares))
    sorted_shares = np.sort(shares)
    gini = float((2 * np.sum(np.arange(1, classes + 1) * sorted_shares) / (classes * np.sum(sorted_shares))) - (classes + 1) / classes)
    return {
        "accuracy": float(np.mean(labels == predictions)), "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)), "prediction_concentration": float(shares.max()),
        "normalized_prediction_entropy": normalized_entropy, "gini_prediction_distribution": gini,
        "hhi": float(np.sum(shares ** 2)), "dominant_predicted_class": dominant,
        "dominant_predicted_class_ratio": float(shares[dominant]), "count": int(len(labels)),
    }


def _validate_metrics(result_metrics: dict[str, Any], labels: np.ndarray, predictions: np.ndarray, logits: np.ndarray, snrs: np.ndarray, classes: int) -> None:
    overall = _group_metrics(labels, predictions, classes)
    checks = {"overall_accuracy": overall["accuracy"], "macro_f1": overall["macro_f1"], "balanced_accuracy": overall["balanced_accuracy"]}
    for field, expected in checks.items():
        if not np.isclose(float(result_metrics[field]), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"result metric {field} is not recomputable from predictions")
    maxima = logits.max(1); centered = logits - maxima[:, None]
    nll = float(np.mean((maxima - logits[np.arange(len(labels)), labels]) + np.log(np.exp(centered).sum(1))))
    probabilities = np.exp(centered); probabilities /= probabilities.sum(1, keepdims=True)
    confidence = probabilities.max(1); correct = predictions == labels; ece = 0.0; boundaries = np.linspace(0, 1, 16)
    for index in range(15):
        mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1] if index < 14 else confidence <= 1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    for field, expected in (("nll", nll), ("ece", ece)):
        if not np.isclose(float(result_metrics[field]), expected, rtol=0.0, atol=1e-7):
            raise ValueError(f"result metric {field} is not recomputable from prediction logits")
    for snr in SNR_VALUES:
        metric = result_metrics["per_snr"][_snr_key(snr)]
        expected = _group_metrics(labels[snrs == snr], predictions[snrs == snr], classes)
        for field in ("accuracy", "macro_f1", "balanced_accuracy", "prediction_concentration", "normalized_prediction_entropy", "hhi", "dominant_predicted_class_ratio", "count"):
            if not np.isclose(float(metric[field]), float(expected[field]), rtol=0.0, atol=1e-12):
                raise ValueError(f"per-SNR metric {field} at {snr} dB is not recomputable")


def _validate_complexity(value: Any) -> None:
    fields = {"total_parameters", "additional_conditioner_parameters", "parameter_increase_percent", "flops", "latency", "memory"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("complexity schema differs")
    total = value["total_parameters"]; additional = value["additional_conditioner_parameters"]; percent = value["parameter_increase_percent"]
    if type(total) is not int or total <= 0 or type(additional) is not int or not 0 <= additional < total or not isinstance(percent, (int, float)) or not np.isfinite(percent):
        raise ValueError("complexity parameter evidence is invalid")
    if not np.isclose(float(percent), 100 * additional / (total - additional), rtol=1e-12, atol=1e-12):
        raise ValueError("complexity parameter increase is inconsistent")
    summary = value["latency"].get("summary") if isinstance(value["latency"], dict) else None
    if not isinstance(summary, dict) or summary.get("basis") != "mixed_bins.fixed_batch" or any(not isinstance(summary.get(field), (int, float)) or not np.isfinite(summary[field]) for field in ("mean_ms", "median_ms")):
        raise ValueError("complexity latency summary is invalid")
    for field in ("flops", "memory"):
        evidence = value[field]
        if not isinstance(evidence, dict) or type(evidence.get("available")) is not bool or (not evidence["available"] and not isinstance(evidence.get("reason"), str)):
            raise ValueError(f"complexity {field} availability evidence is invalid")


def _validate_mechanism_tensors(model: str, names: list[str], tensors: list[np.ndarray]) -> None:
    mapping = dict(zip(names, tensors))
    expected_simple = {
        "M0": set(), "M1": {"immediate_affine.condition_weight"}, "M2": {"immediate_affine.condition_weight"},
        "M3": {"snr_embedding_vectors", "immediate_affine.condition_weight", "per_snr_additive_contribution"},
        "M4": {"per_snr_logit_bias"}, "M5": {"per_snr_gamma", "per_snr_beta"}, "M6": {"per_snr_gates"},
    }
    if model in expected_simple and set(names) != expected_simple[model]:
        raise ValueError(f"mechanism tensors do not match {model}")
    if model == "M1" and (mapping[names[0]].ndim != 2 or mapping[names[0]].shape[1] != 1):
        raise ValueError("mechanism M1 condition projection shape differs")
    if model == "M2" and (mapping[names[0]].ndim != 2 or mapping[names[0]].shape[1] != 20):
        raise ValueError("mechanism M2 condition projection shape differs")
    if model == "M3":
        embedding = mapping["snr_embedding_vectors"]; projection = mapping["immediate_affine.condition_weight"]; contribution = mapping["per_snr_additive_contribution"]
        if embedding.ndim != 2 or embedding.shape[0] != 20 or projection.ndim != 2 or projection.shape[1] != embedding.shape[1] or contribution.shape != (20, projection.shape[0]):
            raise ValueError("mechanism M3 tensor shapes differ")
        if not np.allclose(contribution, embedding @ projection.T, rtol=1e-6, atol=1e-7):
            raise ValueError("mechanism M3 additive contribution does not equal embedding @ projection.T")
    if model == "M4" and (mapping["per_snr_logit_bias"].ndim != 2 or mapping["per_snr_logit_bias"].shape[0] != 20):
        raise ValueError("mechanism M4 bias shape differs")
    if model == "M5" and (mapping["per_snr_gamma"].ndim != 2 or mapping["per_snr_gamma"].shape[0] != 20 or mapping["per_snr_gamma"].shape != mapping["per_snr_beta"].shape):
        raise ValueError("mechanism M5 gamma/beta shapes differ")
    if model == "M6" and (mapping["per_snr_gates"].ndim != 2 or mapping["per_snr_gates"].shape[0] != 20):
        raise ValueError("mechanism M6 gate shape differs")
    if model == "M6" and not np.all((mapping["per_snr_gates"] > 0) & (mapping["per_snr_gates"] < 2)):
        raise ValueError("mechanism M6 gates must lie strictly inside (0, 2)")
    if model == "M7":
        expected = {f"snr_heads.{index}.{suffix}" for index in range(20) for suffix in ("0.weight", "0.bias", "2.weight", "2.bias")}
        if set(names) != expected:
            raise ValueError("mechanism M7 head tensor names differ")
        shapes = {suffix: mapping[f"snr_heads.0.{suffix}"].shape for suffix in ("0.weight", "0.bias", "2.weight", "2.bias")}
        if any(mapping[f"snr_heads.{index}.{suffix}"].shape != shapes[suffix] for index in range(20) for suffix in shapes):
            raise ValueError("mechanism M7 head tensor shapes differ")


def _load_phase3_contracts() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts/v2/run_phase3.py"
    module_name = "_phase3_runner_contracts_for_analysis"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 3 runner contracts")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixed_data(root: Path, config: dict[str, Any], phase3: Any) -> tuple[dict[str, Any], Any]:
    split_path = root / config["split"]["metadata"]
    data_path = root / config["dataset"]["path"]
    split = phase3.load_split(split_path, data_path=data_path)
    from phase1_reproduce import _load_rml_dataset
    dataset = _load_rml_dataset(data_path, config["dataset"]["id"], repository_root=root)
    return dataset, split


def _exact_metrics_equal(left: Any, right: Any) -> bool:
    return json.dumps(_json_safe(left), sort_keys=True, separators=(",", ":")) == json.dumps(_json_safe(right), sort_keys=True, separators=(",", ":"))


def _load_registered_runs(root: Path, manifest_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]], list[str]]:
    phase3 = _load_phase3_contracts()
    planned = {(model, seed) for model in MODEL_IDS for seed in SEEDS}
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    ledger_path = (root / "results/v2/phase3/phase3_ledger.json") if manifest_path == (root / "manifest.json").resolve() else manifest_path.parent / "phase3_ledger.json"
    if not ledger_path.is_file():
        return {}, [{"model_id": model, "seed": seed} for model, seed in sorted(planned)], [f"Phase 3 ledger is missing: {ledger_path}"]
    try:
        ledger = phase3._load_ledger(ledger_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["experiments"]
        if manifest.get("schema_version") != 1 or not isinstance(records, list):
            raise ValueError("source manifest must use schema_version 1 and an experiments list")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {}, [{"model_id": model, "seed": seed} for model, seed in sorted(planned)], [str(exc)]
    phase3_records = [record for record in records if isinstance(record, dict) and record.get("experiment") == "phase3"]
    completed_ledger = [record for record in ledger["records"] if record["status"] == "completed"]
    if len(completed_ledger) != 40:
        errors.append(f"Phase 3 ledger must contain exactly 40 completed attempts; found {len(completed_ledger)}")
    config: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    split: Any = None
    for index, record in enumerate(phase3_records):
        try:
            if not isinstance(record, dict) or record.get("dataset") != "RML2016.10a":
                raise ValueError("record is not a registered Phase 3 RML2016.10a experiment")
            model = record.get("model"); seed = record.get("seed"); key = (model, seed)
            if key not in planned:
                raise ValueError("record has an unplanned model or seed")
            if key in runs:
                raise ValueError("duplicate model/seed record")
            if record.get("status") != "completed":
                raise ValueError("planned record is not completed")
            result_path = _repo_file(root, record.get("result_json"), prefix="results/v2/phase3")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or result.get("model_id") != model or result.get("seed") != seed:
                raise ValueError("result completion identity differs from its registry record")
            provenance = result.get("provenance", {})
            if provenance.get("split_hash") != record.get("split_hash") or provenance.get("split_hash") != LOCKED_SPLIT_HASH:
                raise ValueError("split hash differs between registry and result")
            artifacts = result.get("artifacts", {})
            run_dir = result_path.parent
            if not phase3._ledger_has_result(ledger, root, run_dir, result_path):
                raise ValueError("independent Phase 3 ledger does not close over this exact result hash")
            resolved_artifacts: dict[str, Path] = {}
            for name in ("run_spec", "checkpoint", "predictions", "mechanism"):
                filename = artifacts.get(name)
                if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
                    raise ValueError(f"{name} path is not a safe run-relative filename")
                path = (run_dir / filename).resolve()
                if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
                    raise ValueError(f"{name} artifact is missing or escapes its run")
                if artifacts.get(f"{name}_sha256") != _sha256(path):
                    raise ValueError(f"{name} artifact hash differs from result")
                resolved_artifacts[name] = path
            spec_value = json.loads(resolved_artifacts["run_spec"].read_text(encoding="utf-8"))
            candidate_config = spec_value.get("protocol_config")
            phase3.validate_phase3_config(candidate_config)
            if config is None:
                dataset, split = _load_fixed_data(root, candidate_config, phase3)
                if split.split_hash != LOCKED_SPLIT_HASH:
                    raise ValueError("loaded fixed split hash differs from the locked split")
                config = candidate_config
            elif candidate_config != config:
                raise ValueError("run specification protocol config differs across registered runs")
            assert dataset is not None and split is not None and config is not None
            code = spec_value.get("provenance", {}).get("code")
            environment = spec_value.get("provenance", {}).get("environment")
            preprocessing = phase3.preprocessing_hash(config)
            protocol = phase3.protocol_hash(config)
            run_identity = {
                "runner": phase3.RUNNER_VERSION, "protocol_hash": protocol,
                "preprocessing_hash": preprocessing, "split_hash": split.split_hash,
                "model_id": model, "seed": seed, "code": code, "environment": environment,
            }
            fingerprint = phase3.deterministic_json_hash(run_identity)
            phase3._validate_run_spec_schema(
                spec_value, model_id=model, seed=seed, fingerprint=fingerprint,
                split_hash=split.split_hash, preprocessing=preprocessing, protocol=protocol,
                config=config, code=code, environment=environment,
            )
            phase3._validate_result_schema(
                result, model_id=model, seed=seed, fingerprint=fingerprint,
                split_hash=split.split_hash, preprocessing=preprocessing, protocol=protocol,
                code=code, environment=environment, fixed_batch=int(config["evaluation_batch_size"]),
            )
            registered_checkpoint = _repo_file(root, record.get("checkpoint"), prefix="results/v2/phase3")
            if registered_checkpoint != resolved_artifacts["checkpoint"]:
                raise ValueError("registered checkpoint path differs from result artifact")
            bundle_identity = {"model_id": model, "seed": seed, "split_hash": split.split_hash, "preprocessing_hash": preprocessing, "protocol_hash": protocol}
            recomputed_metrics = phase3.validate_prediction_bundle(
                resolved_artifacts["predictions"], split, dataset, identity=bundle_identity,
                num_classes=11, snr_values=SNR_VALUES,
            )
            if not _exact_metrics_equal(result["metrics"], recomputed_metrics):
                raise ValueError("result metrics are not recomputable from the strict prediction bundle")
            phase3.validate_complexity(result["complexity"], fixed_batch=int(config["evaluation_batch_size"]))
            with np.load(resolved_artifacts["predictions"], allow_pickle=False) as bundle:
                labels = np.asarray(bundle["y_true"], dtype=np.int64)
                predictions = np.asarray(bundle["y_pred"], dtype=np.int64)
                logits = np.asarray(bundle["logits"])
                snrs = np.asarray(bundle["snr_db"])
                snr_bins = np.asarray(bundle["snr_bin"])
                sample_ids = np.asarray(bundle["sample_ids"])
            with np.load(resolved_artifacts["mechanism"], allow_pickle=False) as mechanism:
                for field, expected in (("model_id", model), ("seed", seed), ("split_hash", provenance.get("split_hash")), ("preprocessing_hash", provenance.get("preprocessing_hash")), ("protocol_hash", provenance.get("protocol_hash")), ("checkpoint_sha256", artifacts["checkpoint_sha256"])):
                    if field not in mechanism.files or _scalar(mechanism, field) != expected:
                        raise ValueError(f"mechanism identity {field} differs from result")
                names = mechanism["tensor_names"].tolist() if "tensor_names" in mechanism.files else None
                if not isinstance(names, list) or len(set(names)) != len(names) or set(mechanism.files) != {"model_id", "seed", "split_hash", "preprocessing_hash", "protocol_hash", "checkpoint_sha256", "tensor_names"} | {f"tensor_{item:03d}" for item in range(len(names))}:
                    raise ValueError("mechanism tensor schema differs")
                if any(not np.issubdtype(mechanism[f"tensor_{item:03d}"].dtype, np.number) or not np.isfinite(mechanism[f"tensor_{item:03d}"]).all() for item in range(len(names))):
                    raise ValueError("mechanism tensors must be finite numeric arrays")
                mechanism_tensors = {name: np.asarray(mechanism[f"tensor_{item:03d}"]) for item, name in enumerate(names)}
                _validate_mechanism_tensors(model, names, list(mechanism_tensors.values()))
            verifier = _COMPLETED_VERIFIER_OVERRIDE or phase3._default_completed_verifier
            active_config = dict(config); active_config["_active_seed"] = seed; active_config["_repository_root"] = str(root)
            evidence = verifier(model, dataset, split, run_dir, lambda _line: None, config=active_config)
            phase3.validate_mechanism_bundle(
                resolved_artifacts["mechanism"], identity=bundle_identity | {"checkpoint_sha256": artifacts["checkpoint_sha256"]},
                architecture=config["architecture"], pooled_feature_dim=int(config["models"]["M5"]["modulation_dimension"]),
                expected_tensors=evidence["mechanism_tensors"],
            )
            rerun_logits = np.asarray(evidence["logits"])
            if rerun_logits.shape != logits.shape or not np.allclose(rerun_logits, logits, rtol=1e-6, atol=1e-7):
                raise ValueError("checkpoint re-inference logits differ from registered predictions")
            counts = evidence["parameter_counts"]
            if any(counts[name] != result["complexity"][name] for name in ("total_parameters", "additional_conditioner_parameters")) or not np.isclose(counts["parameter_increase_percent"], result["complexity"]["parameter_increase_percent"], rtol=1e-12, atol=1e-12):
                raise ValueError("checkpoint-derived parameter counts differ from complexity evidence")
            if model in phase3.REUSED_MODELS:
                source_validator = _PHASE2_SOURCE_VALIDATOR_OVERRIDE or phase3._validate_phase2_reuse_provenance
                execution = result["execution"]
                source_validator(model, seed, dataset, split, execution["reuse_source"], execution["source_artifact_hashes"], config=active_config)
            runs[key] = {
                "record": record, "result": result, "paths": resolved_artifacts,
                "result_path": result_path, "result_sha256": _sha256(result_path),
                "run_spec": spec_value, "row_identity": (labels, snrs, snr_bins, sample_ids),
                "ledger_record": next(item for item in completed_ledger if item["attempt_path"] == run_dir.relative_to(root).as_posix()),
                "ledger_path": ledger_path,
            }
        except (OSError, json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"phase3_experiments[{index}]: {exc}")
    if runs:
        provenances = {(run["result"]["provenance"].get("split_hash"), run["result"]["provenance"].get("preprocessing_hash"), run["result"]["provenance"].get("protocol_hash")) for run in runs.values()}
        if len(provenances) != 1:
            errors.append("registered runs do not share one split/preprocessing/protocol identity")
        reference = next(iter(runs.values()))["row_identity"]
        if any(not all(np.array_equal(left, right) for left, right in zip(run["row_identity"], reference)) for run in runs.values()):
            errors.append("registered prediction rows do not share identical labels, true SNR, SNR bins, and sample IDs")
    missing = [{"model_id": model, "seed": seed} for model, seed in sorted(planned - set(runs))]
    return runs, missing, errors


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    from scipy import stats
    array = np.asarray(values, dtype=float); n = len(array)
    mean = float(np.mean(array)); std = float(np.std(array, ddof=1))
    margin = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _paired_statistics(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for treatment, baseline in EXPECTED_PAIRS:
        metrics: dict[str, Any] = {}
        for metric in OVERALL_METRICS[:3]:
            baseline_values = [runs[(baseline, seed)]["result"]["metrics"][metric] for seed in SEEDS]
            treatment_values = [runs[(treatment, seed)]["result"]["metrics"][metric] for seed in SEEDS]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                summary = paired_summary(baseline_values, treatment_values)
            differences = np.asarray(summary["differences"])
            summary["seeds"] = list(SEEDS)
            summary["seed_direction"] = {"positive": int(np.sum(differences > 0)), "zero": int(np.sum(differences == 0)), "negative": int(np.sum(differences < 0))}
            metrics[metric] = summary
        comparisons[f"{treatment}_minus_{baseline}"] = metrics
    return {"schema_version": 1, "matched_seed_required": True, "confidence_interval": "two-sided 95% Student t interval on paired differences", "effect_size": "Cohen's dz", "comparisons": comparisons}


def _summary_tables(runs: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, Any] = {}; summary_rows: list[dict[str, Any]] = []; per_snr_rows: list[dict[str, Any]] = []; complexity_rows: list[dict[str, Any]] = []
    for model in MODEL_IDS:
        model_summary: dict[str, Any] = {}
        for metric in OVERALL_METRICS:
            values = [float(runs[(model, seed)]["result"]["metrics"][metric]) for seed in SEEDS]
            stats = _mean_ci(values); model_summary[metric] = stats | {"seed_values": dict(zip(map(str, SEEDS), values))}
            summary_rows.append({"model_id": model, "metric": metric, **stats, **{f"seed_{seed}": value for seed, value in zip(SEEDS, values)}})
        models[model] = model_summary
        model_summary["segments"] = {
            segment: {metric: _mean_ci([float(runs[(model, seed)]["result"]["metrics"]["segments"][segment][metric]) for seed in SEEDS])
                      for metric in ("accuracy", "balanced_accuracy", "prediction_concentration", "normalized_prediction_entropy")}
            for segment in ("low", "mid", "high")
        }
        for snr in SNR_VALUES:
            for metric in PER_SNR_METRICS:
                values = [float(runs[(model, seed)]["result"]["metrics"]["per_snr"][_snr_key(snr)][metric]) for seed in SEEDS]
                per_snr_rows.append({"model_id": model, "snr_db": snr, "metric": metric, **_mean_ci(values), **{f"seed_{seed}": value for seed, value in zip(SEEDS, values)}})
        for seed in SEEDS:
            complexity = runs[(model, seed)]["result"]["complexity"]
            latency = complexity["latency"]["summary"]
            complexity_rows.append({
                "model_id": model, "seed": seed, "total_parameters": complexity["total_parameters"],
                "additional_conditioner_parameters": complexity["additional_conditioner_parameters"],
                "parameter_increase_percent": complexity["parameter_increase_percent"],
                "latency_basis": latency["basis"], "latency_mean_ms": latency["mean_ms"], "latency_median_ms": latency["median_ms"],
                "batch_1_latency_basis": "mixed_bins.batch_1",
                "batch_1_latency_mean_ms": complexity["latency"]["mixed_bins"]["batch_1"]["mean_ms"],
                "batch_1_latency_median_ms": complexity["latency"]["mixed_bins"]["batch_1"]["median_ms"],
                "fixed_batch_size": complexity["latency"]["mixed_bins"]["fixed_batch"]["batch_size"],
                "flops_available": complexity["flops"]["available"], "flops_value": complexity["flops"].get("value"), "flops_unavailable_reason": complexity["flops"].get("reason"),
                "memory_available": complexity["memory"]["available"], "memory_value": complexity["memory"].get("value"), "memory_unavailable_reason": complexity["memory"].get("reason"),
            })
    summary = {
        "schema_version": 1, "models": models,
        "low_snr_definition": {"preregistered": True, "max_db": -8, "values_db": [value for value in SNR_VALUES if value <= -8]},
        "interpolation_diagnostic": {"status": "unsupported", "reason": "all discrete 2 dB SNR bins are observed during training; no strict unseen or intermediate condition exists in this dataset protocol", "scores": None},
    }
    return summary, summary_rows, per_snr_rows, complexity_rows


def _mechanism_tensors(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        names = bundle["tensor_names"].tolist()
        return {name: np.asarray(bundle[f"tensor_{index:03d}"]) for index, name in enumerate(names)}


def _pairwise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difference = values[:, None, :] - values[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    norms = np.linalg.norm(values, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(values @ values.T, denominator, out=np.zeros_like(distance), where=denominator > 0)
    return distance, cosine


def _geometry(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    seeds = np.asarray(SEEDS); snrs = np.asarray(SNR_VALUES)
    embeddings = []; embedding_adjacent = []; embedding_distance = []; embedding_cosine = []; embedding_norm = []; pca = []
    biases = []; bias_norm = []; bias_distance = []; bias_shift = []
    gammas = []; betas = []; gamma_adjacent = []; beta_adjacent = []
    gates = []; gate_sparsity = []; gate_entropy = []; gate_adjacent = []
    heads = []; head_cosine = []; head_distance = []; head_norm = []; head_drift = []
    for seed in SEEDS:
        tensors = _mechanism_tensors(runs[("M3", seed)]["paths"]["mechanism"]); embedding = tensors["snr_embedding_vectors"].astype(float)
        distance, cosine = _pairwise(embedding); centered = embedding - embedding.mean(0); _, _, vh = np.linalg.svd(centered, full_matrices=False); scores = centered @ vh[:2].T
        embeddings.append(embedding); embedding_adjacent.append(np.linalg.norm(np.diff(embedding, axis=0), axis=1)); embedding_distance.append(distance); embedding_cosine.append(cosine); embedding_norm.append(np.linalg.norm(embedding, axis=1)); pca.append(scores)
        bias = _mechanism_tensors(runs[("M4", seed)]["paths"]["mechanism"])["per_snr_logit_bias"].astype(float); distance, _ = _pairwise(bias)
        biases.append(bias); bias_norm.append(np.linalg.norm(bias, axis=1)); bias_distance.append(distance); bias_shift.append(bias - bias.mean(0))
        film = _mechanism_tensors(runs[("M5", seed)]["paths"]["mechanism"]); gamma = film["per_snr_gamma"].astype(float); beta = film["per_snr_beta"].astype(float)
        gammas.append(gamma); betas.append(beta); gamma_adjacent.append(np.linalg.norm(np.diff(gamma, axis=0), axis=1)); beta_adjacent.append(np.linalg.norm(np.diff(beta, axis=0), axis=1))
        gate = _mechanism_tensors(runs[("M6", seed)]["paths"]["mechanism"])["per_snr_gates"].astype(float); shares = np.abs(gate) / np.abs(gate).sum(1, keepdims=True); positive = np.where(shares > 0, shares, 1)
        gates.append(gate); gate_sparsity.append(np.mean(np.abs(gate) <= 1e-6, axis=1)); gate_entropy.append(-np.sum(np.where(shares > 0, shares * np.log(positive), 0), axis=1) / np.log(gate.shape[1])); gate_adjacent.append(np.linalg.norm(np.diff(gate, axis=0), axis=1))
        head_tensors = _mechanism_tensors(runs[("M7", seed)]["paths"]["mechanism"]); flattened = []
        for index in range(20):
            flattened.append(np.concatenate([value.ravel() for name, value in sorted(head_tensors.items()) if name.startswith(f"snr_heads.{index}.")]))
        flattened_array = np.stack(flattened); distance, cosine = _pairwise(flattened_array)
        weights = np.stack([np.concatenate([head_tensors[f"snr_heads.{index}.{layer}.weight"].ravel() for layer in (0, 2)]) for index in range(20)])
        heads.append(flattened_array); head_cosine.append(cosine); head_distance.append(distance); head_norm.append(np.linalg.norm(weights, axis=1)); head_drift.append(np.linalg.norm(np.diff(flattened_array, axis=0), axis=1))
    return {
        "embedding_geometry.npz": {"seeds": seeds, "snr_db": snrs, "embedding_vectors": np.stack(embeddings), "adjacent_euclidean_distance": np.stack(embedding_adjacent), "pairwise_euclidean_distance": np.stack(embedding_distance), "cosine_similarity": np.stack(embedding_cosine), "embedding_norm": np.stack(embedding_norm), "pca_scores": np.stack(pca)},
        "conditional_bias.npz": {"seeds": seeds, "snr_db": snrs, "bias_vectors": np.stack(biases), "bias_magnitude": np.stack(bias_norm), "pairwise_euclidean_distance": np.stack(bias_distance), "class_shift_from_snr_mean": np.stack(bias_shift)},
        "film_parameters.npz": {"seeds": seeds, "snr_db": snrs, "gamma": np.stack(gammas), "beta": np.stack(betas), "gamma_mean": np.mean(gammas, axis=2), "gamma_std": np.std(gammas, axis=2), "beta_mean": np.mean(betas, axis=2), "beta_std": np.std(betas, axis=2), "adjacent_gamma_change": np.stack(gamma_adjacent), "adjacent_beta_change": np.stack(beta_adjacent)},
        "gating_parameters.npz": {"seeds": seeds, "snr_db": snrs, "gates": np.stack(gates), "gate_sparsity": np.stack(gate_sparsity), "normalized_gate_entropy": np.stack(gate_entropy), "adjacent_gate_change": np.stack(gate_adjacent)},
        "per_bin_head_geometry.npz": {"seeds": seeds, "snr_db": snrs, "flattened_head_parameters": np.stack(heads), "cosine_similarity": np.stack(head_cosine), "frobenius_distance": np.stack(head_distance), "classifier_weight_norm": np.stack(head_norm), "boundary_parameter_drift": np.stack(head_drift)},
    }


def _mechanism_report(summary: dict[str, Any], paired: dict[str, Any], complexity: list[dict[str, Any]], runs: dict[tuple[str, int], dict[str, Any]]) -> str:
    best = max(MODEL_IDS[1:], key=lambda model: summary["models"][model]["overall_accuracy"]["mean"])
    lines = ["# Phase 3 Conditioning Mechanism Report", "",
             f"Best overall conditioner: {best} (descriptive ranking on these five seeds, not a population superiority claim).", "",
             "| Model | Accuracy mean ± SD | Low-SNR accuracy | High-SNR accuracy | Parameters | Batch-1 latency ms | Fixed-batch latency ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODEL_IDS:
        entry = summary["models"][model]; rows = [row for row in complexity if row["model_id"] == model]
        lines.append(f"| {model} | {entry['overall_accuracy']['mean']:.6f} ± {entry['overall_accuracy']['std']:.6f} | {entry['segments']['low']['accuracy']['mean']:.6f} | {entry['segments']['high']['accuracy']['mean']:.6f} | {np.mean([row['total_parameters'] for row in rows]):.0f} | {np.mean([row['batch_1_latency_mean_ms'] for row in rows]):.6f} | {np.mean([row['latency_mean_ms'] for row in rows]):.6f} |")
    lines.extend(["", "Paired comparisons (treatment minus baseline; 95% Student t CI, n=5; secondary p-values are unadjusted):", ""])
    for name, metrics in paired["comparisons"].items():
        for metric, entry in metrics.items():
            lines.append(f"- {name}, {metric}: Δ={entry['difference_mean']:.6f}, 95% CI [{entry['ci95_low']}, {entry['ci95_high']}], dz={entry['cohen_dz']}, p={entry['p_value_secondary']}, directions={entry['seed_direction']}.")
    lines.extend(["", "Low/high-SNR diagnostic (paired deltas vs M0; concentration/entropy diagnose prediction collapse, not correctness):", ""])
    for model in MODEL_IDS[1:]:
        for segment in ("low", "high"):
            for metric in ("accuracy", "balanced_accuracy", "prediction_concentration", "normalized_prediction_entropy"):
                deltas = [runs[(model, seed)]["result"]["metrics"]["segments"][segment][metric] - runs[("M0", seed)]["result"]["metrics"]["segments"][segment][metric] for seed in SEEDS]
                stats = _mean_ci(deltas)
                lines.append(f"- {model} − M0, {segment}, {metric}: Δ={stats['mean']:.6f}, 95% CI [{stats['ci95_low']:.6f}, {stats['ci95_high']:.6f}].")
    lines.extend(["", "Capacity and efficiency: M7 is a parameter-rich per-bin control; M7−M3 above measures observed benefit, not guaranteed superiority. Parameter and measured latency costs are reported separately; latency depends on the recorded hardware/batch protocol and is not a universal speed ranking.", "",
                  "Claim allowed: report observed matched-seed changes, confidence intervals, low/high-SNR trade-offs and parameter/runtime costs within this fixed dataset/split protocol.",
                  "Claim forbidden: absence of a significant difference does not establish equivalence; no population-best, causal mechanism, unseen-SNR interpolation, or universal efficiency claim follows from these five seeds.", ""])
    return "\n".join(lines) + """

All values in this report are derived from registered Phase 3 artifacts. The preregistered low-SNR region is -20 through -8 dB inclusive.

For embedding concatenation followed by a linear map:

`linear([f(x); e(z)]) = W_f f(x) + W_e e(z) + b`

The real M3 head is Linear → LeakyReLU → Linear: `logits = W2 LeakyReLU(W_f f(x) + W_e e(z) + b1) + b2`. The direct contribution is an additive hidden preactivation shift, not a pure final-logit bias. Although W_f is shared, condition-dependent activation masks can change the local feature-to-logit Jacobian. M4 alone adds a final-logit bias. FiLM and gating introduce explicit feature × condition interaction; M7 permits independent per-bin classifier parameters but does not guarantee higher accuracy.

Geometry is descriptive: PCA is computed independently per seed and only seed 2022 is plotted; unaligned coordinates are never averaged. M7 distances include all head weights and biases, whereas classifier_weight_norm includes weights only. Parameter drift is not a measured decision-boundary distance. Gate sparsity counts |gate| ≤ 1e-6; gate entropy normalizes positive gates across features and is not predictive entropy. M4 class_shift_from_snr_mean subtracts each class's across-SNR mean, not the softmax-invariant common class offset.

## Interpolation diagnostic

不具备严格 unseen/intermediate experimental support。All discrete 2 dB SNR bins are observed during training, so no holdout interpolation score is fabricated. Embedding geometry is descriptive and does not establish continuous generalization.
"""


def _figure_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })


def _save_figure(fig: Any, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _render_figures(staging: Path) -> None:
    """Render only from the staged machine-readable tables and NPZ files."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _figure_style()
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    linestyles = ("-", "--", "-.", ":", "-", "--", "-.", ":")
    figures = staging / "figures"; figures.mkdir()
    summary = json.loads((staging / "conditioning_summary.json").read_text(encoding="utf-8"))
    paired = json.loads((staging / "paired_statistics.json").read_text(encoding="utf-8"))
    with (staging / "per_snr_metrics.csv").open(encoding="utf-8", newline="") as handle:
        per_snr = list(csv.DictReader(handle))
    with (staging / "complexity.csv").open(encoding="utf-8", newline="") as handle:
        complexity = list(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    means = np.asarray([summary["models"][model]["overall_accuracy"]["mean"] for model in MODEL_IDS])
    low = means - np.asarray([summary["models"][model]["overall_accuracy"]["ci95_low"] for model in MODEL_IDS])
    high = np.asarray([summary["models"][model]["overall_accuracy"]["ci95_high"] for model in MODEL_IDS]) - means
    ax.errorbar(np.arange(8), means, yerr=np.stack([low, high]), fmt="none", color="black", capsize=3, label="95% t CI")
    for index, model in enumerate(MODEL_IDS):
        values = list(summary["models"][model]["overall_accuracy"]["seed_values"].values())
        ax.scatter(np.full(5, index), values, color=colors[index], marker=markers[index], s=22, zorder=3)
    ax.set(xticks=np.arange(8), xticklabels=MODEL_IDS, xlabel="Conditioner", ylabel="Overall accuracy", title="Conditioner comparison (n=5 matched seeds)")
    ax.legend(loc="upper left", frameon=False); _save_figure(fig, figures, "overall_conditioner_comparison")

    def snr_figure(metric: str, name: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for index, model in enumerate(MODEL_IDS):
            rows = sorted((row for row in per_snr if row["model_id"] == model and row["metric"] == metric), key=lambda row: int(row["snr_db"]))
            x = np.asarray([int(row["snr_db"]) for row in rows]); mean = np.asarray([float(row["mean"]) for row in rows]); lower = np.asarray([float(row["ci95_low"]) for row in rows]); upper = np.asarray([float(row["ci95_high"]) for row in rows])
            ax.plot(x, mean, color=colors[index], marker=markers[index], linestyle=linestyles[index], markersize=3, linewidth=1.2, label=model)
            ax.fill_between(x, lower, upper, color=colors[index], alpha=0.08, linewidth=0)
        ax.axvspan(-20, -8, color="#999999", alpha=0.08, label="Preregistered low SNR")
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=f"{ylabel} versus SNR (mean and 95% t CI; n=5)")
        ax.set_xticks(SNR_VALUES); ax.tick_params(axis="x", rotation=45); ax.legend(ncol=3, frameon=False); _save_figure(fig, figures, name)
    snr_figure("accuracy", "per_snr_accuracy", "Accuracy")
    snr_figure("balanced_accuracy", "per_snr_balanced_accuracy", "Balanced accuracy")
    snr_figure("prediction_concentration", "per_snr_concentration", "Prediction concentration")
    snr_figure("normalized_prediction_entropy", "per_snr_entropy", "Normalized prediction entropy")
    snr_figure("dominant_predicted_class_ratio", "per_snr_dominant_class_ratio", "Dominant-class ratio")

    def delta_snr_figure(metric: str, name: str, ylabel: str) -> None:
        indexed = {(row["model_id"], int(row["snr_db"]), row["metric"]): row for row in per_snr}
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for index, model in enumerate(MODEL_IDS[1:], start=1):
            means = []; lower = []; upper = []
            for snr in SNR_VALUES:
                treatment = indexed[(model, snr, metric)]; baseline = indexed[("M0", snr, metric)]
                differences = [float(treatment[f"seed_{seed}"]) - float(baseline[f"seed_{seed}"]) for seed in SEEDS]
                stats = _mean_ci(differences); means.append(stats["mean"]); lower.append(stats["ci95_low"]); upper.append(stats["ci95_high"])
            ax.plot(SNR_VALUES, means, color=colors[index], marker=markers[index], linestyle=linestyles[index], markersize=3, linewidth=1.2, label=f"{model} − M0")
            ax.fill_between(SNR_VALUES, lower, upper, color=colors[index], alpha=0.08, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.8); ax.axvspan(-20, -8, color="#999999", alpha=0.08)
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=f"Matched-seed {ylabel} (mean and 95% t CI; n=5)")
        ax.set_xticks(SNR_VALUES); ax.tick_params(axis="x", rotation=45); ax.legend(ncol=3, frameon=False); _save_figure(fig, figures, name)
    delta_snr_figure("accuracy", "delta_accuracy_vs_snr", "ΔAccuracy")
    delta_snr_figure("balanced_accuracy", "delta_balanced_accuracy_vs_snr", "ΔBalanced accuracy")
    delta_snr_figure("prediction_concentration", "delta_concentration_vs_snr", "ΔPrediction concentration")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    names = list(paired["comparisons"]); entries = [paired["comparisons"][name]["overall_accuracy"] for name in names]
    means = np.asarray([entry["difference_mean"] for entry in entries]); ci_low = np.asarray([np.nan if entry["ci95_low"] is None else entry["ci95_low"] for entry in entries]); ci_high = np.asarray([np.nan if entry["ci95_high"] is None else entry["ci95_high"] for entry in entries]); low = means - ci_low; high = ci_high - means
    ax.errorbar(means, np.arange(len(names)), xerr=np.stack([low, high]), fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8); ax.set(yticks=np.arange(len(names)), yticklabels=[name.replace("_minus_", " − ") for name in names], xlabel="Paired accuracy difference", title="Matched-seed improvements (95% t CI; n=5)")
    _save_figure(fig, figures, "paired_improvement_forest")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, model in enumerate(MODEL_IDS):
        parameter_values = [int(row["total_parameters"]) for row in complexity if row["model_id"] == model]
        ax.scatter(np.mean(parameter_values), summary["models"][model]["overall_accuracy"]["mean"], color=colors[index], marker=markers[index], s=35, label=model)
    ax.set(xlabel="Total parameters", ylabel="Overall accuracy", title="Accuracy–complexity trade-off (accuracy mean over n=5)"); ax.legend(ncol=4, frameon=False)
    _save_figure(fig, figures, "accuracy_vs_complexity")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    with np.load(staging / "embedding_geometry.npz") as data:
        scores = data["pca_scores"][0]; axes[0, 0].plot(scores[:, 0], scores[:, 1], "o-", color=colors[3], markersize=3); axes[0, 0].set(xlabel="PC1", ylabel="PC2", title="M3 PCA (seed 2022)")
    with np.load(staging / "conditional_bias.npz") as data:
        axes[0, 1].plot(data["snr_db"], data["bias_magnitude"].mean(0), "s-", color=colors[4]); axes[0, 1].set(xlabel="SNR (dB)", ylabel="Bias norm", title="M4 conditional bias")
    with np.load(staging / "film_parameters.npz") as data:
        axes[0, 2].plot(data["snr_db"], data["gamma_mean"].mean(0), "^-", label="gamma", color=colors[5]); axes[0, 2].plot(data["snr_db"], data["beta_mean"].mean(0), "v--", label="beta", color=colors[1]); axes[0, 2].set(xlabel="SNR (dB)", ylabel="Feature mean", title="M5 FiLM"); axes[0, 2].legend(frameon=False)
    with np.load(staging / "gating_parameters.npz") as data:
        axes[1, 0].plot(data["snr_db"], data["normalized_gate_entropy"].mean(0), "D-", color=colors[6]); axes[1, 0].set(xlabel="SNR (dB)", ylabel="Normalized entropy", title="M6 gates")
    with np.load(staging / "per_bin_head_geometry.npz") as data:
        midpoint = (data["snr_db"][:-1] + data["snr_db"][1:]) / 2; axes[1, 1].plot(midpoint, data["boundary_parameter_drift"].mean(0), "P-", color=colors[7]); axes[1, 1].set(xlabel="Adjacent-bin midpoint (dB)", ylabel="Parameter drift", title="M7 per-bin heads")
    axes[1, 2].axis("off")
    for label, axis in zip("ABCDE", axes.ravel()[:5]): axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")
    fig.tight_layout(); _save_figure(fig, figures, "conditioner_mechanisms")


def _source_fingerprint(manifest_path: Path, runs: dict[tuple[str, int], dict[str, Any]]) -> str:
    evidence = _source_evidence(manifest_path, runs)
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_evidence(manifest_path: Path, runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__).resolve(), code_root / "scripts/v2/analyze_phase3.py", code_root / "scripts/v2/run_phase3.py", *sorted((code_root / "v2").glob("*.py"))]
    from importlib.metadata import version
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_code": {path.relative_to(code_root).as_posix(): _sha256(path) for path in code_files},
        "runtime": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "torch")},
        "python": sys.version,
        "manifest_sha256": _sha256(manifest_path),
        "ledger_sha256": _sha256(next(iter(runs.values()))["ledger_path"]),
        "results": {f"{model}:{seed}": {"result_sha256": _sha256(run["result_path"]), **{f"{name}_sha256": _sha256(path) for name, path in run["paths"].items()}}
                    for (model, seed), run in sorted(runs.items())},
    }


def _publish_tables(staging: Path, root: Path, manifest_path: Path, runs: dict[tuple[str, int], dict[str, Any]], fingerprint: str) -> None:
    summary, summary_rows, per_snr_rows, complexity_rows = _summary_tables(runs)
    _write_json(staging / "config.json", {"schema_version": 1, "analysis_fingerprint": fingerprint, "source_manifest": manifest_path.relative_to(root).as_posix(), "models": list(MODEL_IDS), "seeds": list(SEEDS), "snr_values_db": list(SNR_VALUES), "matched_seed": True, "ci": "two-sided 95% Student t", "low_snr_max_db": -8})
    _write_json(staging / "conditioning_summary.json", summary)
    seed_fields = [f"seed_{seed}" for seed in SEEDS]
    _write_csv(staging / "conditioning_summary.csv", ["model_id", "metric", "n", "mean", "std", "ci95_low", "ci95_high", *seed_fields], summary_rows)
    paired = _paired_statistics(runs)
    _write_json(staging / "paired_statistics.json", paired)
    _write_csv(staging / "per_snr_metrics.csv", ["model_id", "snr_db", "metric", "n", "mean", "std", "ci95_low", "ci95_high", *seed_fields], per_snr_rows)
    _write_csv(staging / "complexity.csv", list(complexity_rows[0]), complexity_rows)
    dominant_rows = []
    for (model, seed), run in sorted(runs.items()):
        for snr in SNR_VALUES:
            metrics = run["result"]["metrics"]["per_snr"][_snr_key(snr)]
            dominant_rows.append({"model_id": model, "seed": seed, "snr_db": snr,
                                  **{name: metrics[name] for name in ("dominant_predicted_class", "dominant_predicted_class_ratio", "count")}})
    _write_csv(staging / "per_snr_dominant_classes.csv", list(dominant_rows[0]), dominant_rows)
    geometry_models = {"embedding_geometry.npz": "M3", "conditional_bias.npz": "M4", "film_parameters.npz": "M5", "gating_parameters.npz": "M6", "per_bin_head_geometry.npz": "M7"}
    for filename, arrays in _geometry(runs).items():
        model = geometry_models[filename]
        identity = {
            "analysis_fingerprint": np.asarray(fingerprint), "model_id": np.asarray(model),
            "source_result_sha256": np.asarray([_sha256(runs[(model, seed)]["result_path"]) for seed in SEEDS]),
        }
        np.savez_compressed(staging / filename, **identity, **arrays)
    (staging / "mechanism_report.md").write_text(_mechanism_report(summary, paired, complexity_rows, runs), encoding="utf-8")
    _render_figures(staging)
    files = {path.relative_to(staging).as_posix(): _sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
    source_artifacts = [{"model_id": model, "seed": seed, "result_json": run["result_path"].relative_to(root).as_posix(), "result_sha256": _sha256(run["result_path"])} for (model, seed), run in sorted(runs.items())]
    _write_json(staging / "manifest.json", {"schema_version": ANALYSIS_SCHEMA_VERSION, "status": "completed", "analysis_fingerprint": fingerprint, "source_evidence": _source_evidence(manifest_path, runs), "source_manifest_sha256": _sha256(manifest_path), "source_artifacts": source_artifacts, "files": files})


def _validate_existing(output_path: Path, fingerprint: str) -> None:
    try:
        manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisConflictError("existing analysis has no valid manifest") from exc
    if manifest.get("status") != "completed" or manifest.get("analysis_fingerprint") != fingerprint:
        raise AnalysisConflictError("existing analysis fingerprint conflicts with current registered inputs")
    evidence_hash = hashlib.sha256(json.dumps(manifest.get("source_evidence"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION or evidence_hash != fingerprint:
        raise AnalysisConflictError("existing source evidence fingerprint differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AnalysisConflictError("existing analysis manifest has no artifact hashes")
    actual_files = {path.relative_to(output_path).as_posix() for path in output_path.rglob("*") if path.is_file()}
    expected_files = set(files) | {"manifest.json"}
    if actual_files != expected_files:
        raise AnalysisConflictError("existing analysis contains missing or unregistered artifacts")
    for name, expected in files.items():
        try:
            path = _repo_file(output_path.parent.parent.parent, f"results/v2/phase3_conditioning/{name}", prefix="results/v2/phase3_conditioning")
        except ValueError as exc:
            raise AnalysisConflictError("existing analysis manifest contains an unsafe path") from exc
        if not path.is_file() or _sha256(path) != expected:
            raise AnalysisConflictError(f"existing analysis artifact hash mismatch: {name}")


def analyze_phase3(*, repository_root: str | Path, source_manifest: str | Path | None = None, output: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Validate registered Phase 3 inputs and, when complete, publish analysis."""
    root = Path(repository_root).resolve()
    manifest_path = Path(source_manifest).resolve() if source_manifest is not None else (root / DEFAULT_SOURCE_MANIFEST).resolve()
    allowed_registry = manifest_path == (root / "manifest.json").resolve() or manifest_path.is_relative_to((root / "results/v2/phase3").resolve())
    if not allowed_registry:
        raise ValueError("source manifest must be the configured root manifest or remain inside results/v2/phase3")
    output_path = Path(output).resolve() if output is not None else (root / DEFAULT_OUTPUT).resolve()
    if output_path != (root / DEFAULT_OUTPUT).resolve():
        raise ValueError("analysis output must be results/v2/phase3_conditioning")
    runs, missing, errors = _load_registered_runs(root, manifest_path)
    report = {"ready": not missing and not errors and len(runs) == 40, "completed_count": len(runs), "missing": missing, "errors": errors, "would_publish": DEFAULT_OUTPUT}
    if dry_run:
        return report
    if not report["ready"]:
        raise AnalysisNotReadyError(f"Phase 3 analysis requires 40/40 valid completed registered runs; found {len(runs)}/40")
    fingerprint = _source_fingerprint(manifest_path, runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise AnalysisConflictError("analysis publication lock already exists; another writer or interrupted publication requires review") from exc
    staging = output_path.with_name(f".{output_path.name}.staging.{uuid.uuid4().hex}")
    try:
        with os.fdopen(lock_descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "fingerprint": fingerprint}, handle)
        if output_path.exists():
            _validate_existing(output_path, fingerprint)
            return {"status": "reused", "analysis_fingerprint": fingerprint, "output": DEFAULT_OUTPUT}
        staging.mkdir(parents=True, exist_ok=False)
        _publish_tables(staging, root, manifest_path, runs, fingerprint)
        if _source_fingerprint(manifest_path, runs) != fingerprint:
            raise AnalysisConflictError("source/code fingerprint changed during publication")
        if output_path.exists():
            raise AnalysisConflictError("publication destination appeared while staging")
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink()
    return {"status": "published", "analysis_fingerprint": fingerprint, "output": DEFAULT_OUTPUT}

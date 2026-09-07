"""Immutable execution and numeric replay for the preregistered Phase 6 audit."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .audit import array_hash, json_hash, sha, verify_artifacts, write_json
from .experiment import checkpoint_streaming, coupling_table, load_phase3, plot_results, run_probes, transform_audit, write_csv


def _relative(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("unsafe repository-relative path")
    return path


def validate_attempt(output, root):
    output, root = Path(output).resolve(), Path(root).resolve()
    manifest = verify_artifacts(output, root)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    from v2.splits import load_split
    split = load_split(root / config["split_path"], data_path=root / config["data_path"])
    ids = np.asarray(split.sample_ids)
    partitions = {"train": split.train_idx, "validation": split.val_idx, "test": split.test_idx}
    metrics_checked = 0
    for mode in config["modes"]:
        directory = output / mode
        state = json.loads((directory / "preprocessing_state.json").read_text(encoding="utf-8"))
        if state["fit_sample_ids_sha256"] != array_hash(ids[split.train_idx]) or state["fit_indices_sha256"] != array_hash(split.train_idx):
            raise ValueError("scaler fit ID mismatch")
        with np.load(directory / "features.npz", allow_pickle=False) as bundle:
            features = bundle["features"]
            if not np.array_equal(bundle["sample_ids"], ids) or not np.isfinite(features).all():
                raise ValueError("feature ID or finiteness mismatch")
            for name, indices in partitions.items():
                if not np.array_equal(bundle[name + "_indices"], indices):
                    raise ValueError("feature partition mismatch")
        with np.load(directory / "all_partition_predictions.npz", allow_pickle=False) as bundle:
            if not np.array_equal(bundle["sample_ids"], ids):
                raise ValueError("prediction ID mismatch")
            for target in ("snr_ridge", "snr_logistic", "modulation_logistic"):
                model = json.loads((directory / f"{target}_state.json").read_text(encoding="utf-8"))
                if model["fit_sample_ids_sha256"] != array_hash(ids[split.train_idx]):
                    raise ValueError("probe fit ID mismatch")
                scaled = (features - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
                logits = scaled @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
                expected = logits if target == "snr_ridge" else np.asarray(model["classes"])[logits.argmax(axis=1)]
                if not np.allclose(expected, bundle[target], rtol=1e-10, atol=1e-10):
                    raise ValueError("saved probe is not replayable")
            from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
            with (directory / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    target = row["target"]
                    if target not in ("snr_ridge", "snr_logistic", "modulation_logistic"):
                        continue
                    index = partitions[row["partition"]]
                    truth = bundle["modulation"][index] if target.startswith("modulation") else bundle["snr_db"][index]
                    if target == "snr_logistic":
                        truth = ((truth + 20) / 2).astype(int)
                    predicted = bundle[target][index]
                    functions = {"mae_db": lambda: mean_absolute_error(truth, predicted), "rmse_db": lambda: np.sqrt(mean_squared_error(truth, predicted)), "accuracy": lambda: accuracy_score(truth, predicted), "macro_f1": lambda: f1_score(truth, predicted, average="macro", zero_division=0)}
                    if not np.isclose(float(row["value"]), functions[row["metric"]](), rtol=1e-12, atol=1e-12):
                        raise ValueError("probe metric replay mismatch")
                    metrics_checked += 1
        with np.load(directory / "transform_streaming.npz", allow_pickle=False) as bundle:
            if not np.array_equal(bundle["sample_ids"], ids[split.test_idx]):
                raise ValueError("streaming ID mismatch")
            for name in ("batch_vs_frame_max_abs", "reordered_max_abs", "changed_future_max_abs"):
                if not np.all(bundle[name] == 0):
                    raise ValueError("streaming transform invariance failed")
    for record in summary["checkpoint_streaming"]:
        with np.load(output / "checkpoint_streaming" / f"{record['model_id']}_seed{record['seed']}.npz", allow_pickle=False) as bundle:
            if not np.array_equal(bundle["sample_ids"], ids[split.test_idx]):
                raise ValueError("checkpoint ID mismatch")
            diff = np.abs(bundle["actual_logits"].astype(float) - bundle["reference_logits"].astype(float))
            passes = (diff <= config["streaming"]["atol"] + config["streaming"]["rtol"] * np.abs(bundle["reference_logits"])).all(axis=1)
            if not np.array_equal(passes, bundle["per_row_tolerance_pass"]) or not np.array_equal(diff.max(axis=1), bundle["per_row_max_abs"]):
                raise ValueError("checkpoint difference replay mismatch")
            same = bundle["actual_logits"].argmax(axis=1) == bundle["reference_logits"].argmax(axis=1)
            if not np.array_equal(same, bundle["per_row_same_argmax"]) or record["buffers_before"] != record["buffers_after"]:
                raise ValueError("checkpoint buffer or decisions mismatch")
            if manifest["status"] == "COMPLETE" and not (passes.all() and same.all()):
                raise ValueError("COMPLETE checkpoint tolerance failed")
    if manifest["status"] != summary["status"]:
        raise ValueError("manifest/summary status mismatch")
    return {"status": "VALIDATED", "files": len(manifest["files"]), "modes": len(config["modes"]), "probe_metric_rows_recomputed": metrics_checked, "checkpoint_test_frames": sum(row["n"] for row in summary["checkpoint_streaming"])}


def run(config_path, root):
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve(); config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["modes"] != ["legacy_identity", "train_global", "per_frame_rms"] or config["probes"]["fit_partition"] != "train" or config["probes"]["selection_partition"] != "validation":
        raise ValueError("Phase6 requires all three modes and train/validation-only fit/selection")
    output = _relative(root, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    phase5_pointer = root / "results/v2/phase5_toy/current.json"
    prior = json.loads(phase5_pointer.read_text(encoding="utf-8"))
    phase5 = _relative(root, prior["attempt"])
    if prior["status"] != "COMPLETE" or sha(phase5 / "manifest.json") != prior["manifest_sha256"] or prior["validation"]["status"] != "VALIDATED":
        raise ValueError("Phase5 gate has not passed")
    # Immutable Phase5 manifest closes the gate; do not change its artifacts.
    from v2.phase5.runner import validate_attempt as validate_phase5
    validate_phase5(phase5, root)
    source_paths = [config_path, root / "scripts/v2/run_phase6.py", *sorted((root / "v2/phase6").glob("*.py")), root / "scripts/v2/phase1_reproduce.py", root / "scripts/v2/run_phase3.py", root / "configs/v2/phase3.yaml", root / "models/model.py", root / "models/model_conditioning.py", root / "models/lifting.py", root / "v2/splits.py", root / config["data_path"], root / config["split_path"], phase5 / "manifest.json"]
    dependencies = {p.relative_to(root).as_posix(): sha(p) for p in source_paths}
    torch.set_num_threads(config["streaming"]["torch_threads"])
    torch.set_num_interop_threads(1)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(), "threadpool_limit": 1}
    identity = json_hash({"config": config, "dependencies": dependencies, "environment": environment})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = output / "attempts" / f"{stamp}_{identity[:8]}"
    attempt.parent.mkdir(exist_ok=True)
    staging = output / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    log_path = staging / "execution.log"
    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    try:
        write_json(staging / "config.json", config); write_json(staging / "environment.json", environment)
        phase3 = load_phase3(root)
        from phase1_reproduce import _load_rml_dataset
        from v2.splits import load_split
        split = load_split(root / config["split_path"], data_path=root / config["data_path"])
        if split.split_hash != config["split_hash"]:
            raise ValueError("fixed split mismatch")
        dataset = _load_rml_dataset(root / config["data_path"], "RML2016.10a", repository_root=root)
        partitions = {"train": np.asarray(split.train_idx), "validation": np.asarray(split.val_idx), "test": np.asarray(split.test_idx)}
        all_indices = np.concatenate(list(partitions.values()))
        if len(np.unique(all_indices)) != len(dataset["signals"]) or not np.array_equal(np.sort(all_indices), np.arange(len(dataset["signals"]))):
            raise ValueError("partitions overlap or omit samples")
        ids = np.asarray(split.sample_ids)
        np.savez_compressed(staging / "split_binding.npz", sample_ids=ids, y_true=dataset["labels"], snr_db=dataset["snrs"], **{name + "_indices": idx for name, idx in partitions.items()})
        metrics, transforms, coupling = [], {}, []
        with threadpool_limits(limits=1):
            for mode in config["modes"]:
                features, transforms[mode] = transform_audit(dataset["signals"], partitions, ids, config, staging / mode, mode, log)
                if not transforms[mode]["passed"]:
                    raise ValueError("preprocessing streaming gate failed; downstream fitting stopped")
                metrics.extend(run_probes(features, dataset["labels"], dataset["snrs"], partitions, ids, config["probes"], staging / mode, mode, log))
                coupling.extend(coupling_table(features, dataset["labels"], dataset["snrs"], partitions, mode))
            replays = checkpoint_streaming(root, phase3, dataset, split, config, staging / "checkpoint_streaming", dependencies, log)
        write_csv(staging / "metrics.csv", metrics); write_csv(staging / "modulation_snr_statistic_coupling.csv", coupling)
        plot_results(metrics, staging)
        status = "COMPLETE" if all(item["passed"] for item in replays) else "BLOCKED_STREAMING_REPLAY"
        summary = {"status": status, "transform_streaming": transforms, "checkpoint_streaming": replays, "partition_counts": {k: len(v) for k, v in partitions.items()}, "probe_feature_family": "simple_frame_statistics", "deterministic_fit_repeated": False, "scope": "all three transforms and probes; M0/M3 seed2022 full test checkpoint streaming", "unrun": ["checkpoint seeds 2023-2026", "flattened-IQ probe", "MLP probe", "AMC retraining on new transforms"], "upstream_dataset_preprocessing_provenance": "not established by repository loader", "main_protocol_restart_required": False, "restart_note": "Source audit found no downstream test/future fitted normalization. Numerical replay failures, if any, require diagnosis before downstream work."}
        write_json(staging / "summary.json", summary)
        lines = ["# Phase 6 preprocessing and streaming report", "", f"Status: {status}", "", "One fixed 132000/44000/44000 train/validation/test split; deterministic fits are not independent seed replications. All scaler/probe fitting uses train only; hyperparameters use validation only. No train+validation refit. Unselected candidates never access test.", "", "| Mode | Ridge test MAE dB | SNR-bin test accuracy | Modulation-statistic test accuracy |", "|---|---:|---:|---:|"]
        for mode in config["modes"]:
            def value(target, metric):
                return next(r["value"] for r in metrics if r["mode"] == mode and r["target"] == target and r["partition"] == "test" and r["metric"] == metric)
            lines.append(f"| {mode} | {value('snr_ridge','mae_db'):.6f} | {value('snr_logistic','accuracy'):.6f} | {value('modulation_logistic','accuracy'):.6f} |")
        lines.extend(["", "All 44000 test frames in each preprocessing mode were replayed independently, reordered, and tested with altered future frames; exact input-transform equality and unchanged scaler state are required.", "", "Checkpoint streaming (batch 1; configured rtol 1e-6, atol 1e-7, exact argmax; buffers include BatchNorm):"])
        for row in replays:
            lines.append(f"- {row['model_id']}, seed {row['seed']}: {row['n']} frames, max absolute logit error {row['max_abs']:.9g}, tolerance failures {row['tolerance_failed_rows']}, changed decisions {row['argmax_changed_rows']}, unchanged buffers {row['buffers_unchanged']}.")
        lines.extend(["", "Inference boundaries: a fixed global affine scaling does not remove all SNR information. Successful SNR prediction from energy/waveform statistics is inferability or a shortcut, not alone prohibited leakage. Per-frame RMS removes an amplitude degree of freedom; retained statistics may encode SNR and modulation. The supervised dataset-trained ridge estimator is not a physical pilot estimator. Modulation-conditioned statistic tables and per-modulation errors document coupling without proving causal leakage.", "", "No input preprocessing was present in the frozen legacy loader. Upstream distributed-pickle power-normalization provenance remains unverified. No main-protocol leakage requiring a Phase 1 restart was established by this audit. If a material prohibited path is subsequently confirmed, stop dependent work, assign a new protocol hash and revalidate from Phase 1.", "", "Scope limits: full-test checkpoint replay covers M0/M3 seed 2022 only. Seeds 2023–2026, flattened-IQ/MLP probes and AMC retraining on changed normalization are unrun. The original simple-statistic probe requirement is completed if all gates pass; changed-normalization AMC accuracy is not claimed. Figures show point estimates from one fixed test split, no artificial across-seed uncertainty. Source CSVs accompany every figure."])
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"Phase6 execution finished: {status}")
        for name, expected in dependencies.items():
            if sha(root / name) != expected:
                raise ValueError(f"dependency changed during execution: {name}")
        write_json(staging / "manifest.json", {"schema_version": 1, "status": status, "identity": identity, "dependencies": dependencies, "files": {p.relative_to(staging).as_posix(): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}})
        validation = validate_attempt(staging, root)
        staging.replace(attempt)
        if status == "COMPLETE":
            pointer_tmp = output / f".current-{uuid.uuid4().hex}.json"
            write_json(pointer_tmp, {"attempt": attempt.relative_to(root).as_posix(), "manifest_sha256": sha(attempt / "manifest.json"), "status": status, "validation": validation})
            os.replace(pointer_tmp, output / "current.json")
        print(json.dumps({"status": status, "attempt": str(attempt), "validation": validation}), flush=True)
        return 0 if status == "COMPLETE" else 3
    except Exception:
        (staging / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        staging.replace(attempt.with_name(attempt.name + "_FAILED"))
        raise

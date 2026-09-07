"""Train-only simple-statistic SNR probes and full-test streaming controls."""
from __future__ import annotations

import csv
import importlib.util
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .audit import FEATURE_NAMES, array_hash, fit_transform, frame_features, json_hash, sha, transform, write_json


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def select_probe(train_x, train_y, val_x, val_y, kind, config):
    """Selection API intentionally has no test partition argument."""
    scaler = StandardScaler().fit(train_x)
    tx, vx = scaler.transform(train_x), scaler.transform(val_x)
    candidates, fitted = [], []
    grid = config["ridge_alphas"] if kind == "ridge" else config["logistic_cs"]
    for value in grid:
        if kind == "ridge":
            model = Ridge(alpha=value, solver="svd")
        elif kind == "logistic":
            model = LogisticRegression(C=value, solver="lbfgs", max_iter=config["logistic_max_iter"], tol=config["logistic_tol"], random_state=config["deterministic_seed"])
        else:
            raise ValueError("unknown probe kind")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(tx, train_y)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in recorded)
        predicted = model.predict(vx)
        score = float(mean_absolute_error(val_y, predicted)) if kind == "ridge" else float(f1_score(val_y, predicted, average="macro", zero_division=0))
        candidates.append({"hyperparameter": float(value), "validation_score": score, "converged": converged, "iterations": int(np.max(model.n_iter_)) if kind == "logistic" else 1})
        fitted.append(model)
    viable = [i for i, row in enumerate(candidates) if row["converged"]]
    if not viable:
        raise ValueError("no converged candidate; no test score may be published")
    key = (lambda i: (candidates[i]["validation_score"], -grid[i])) if kind == "ridge" else (lambda i: (-candidates[i]["validation_score"], grid[i]))
    winner = min(viable, key=key)
    evidence = {"fit_partition": "train", "selection_partition": "validation", "metric": "mae_db" if kind == "ridge" else "macro_f1", "tie_break": "larger_alpha" if kind == "ridge" else "smaller_C", "candidates": candidates, "selected_hyperparameter": float(grid[winner]), "train_x_sha256": array_hash(train_x), "train_y_sha256": array_hash(train_y), "validation_x_sha256": array_hash(val_x), "validation_y_sha256": array_hash(val_y), "refit_train_validation": False}
    return fitted[winner], scaler, evidence


def _probe_state(model, scaler, train_ids, kind):
    return {"kind": kind, "feature_names": list(FEATURE_NAMES), "feature_mean": scaler.mean_.tolist(), "feature_scale": scaler.scale_.tolist(), "coef": model.coef_.tolist(), "intercept": np.asarray(model.intercept_).tolist(), "classes": model.classes_.tolist() if kind == "logistic" else None, "fit_partition": "train", "fit_sample_ids_sha256": array_hash(train_ids), "fit_count": len(train_ids), "estimator_scope": "supervised dataset-trained statistic estimator; not a physical pilot estimator"}


def run_probes(features, labels, snrs, partitions, sample_ids, config, output, mode, log):
    train, val = partitions["train"], partitions["validation"]
    metrics, per_group = [], []
    predictions = {"sample_ids": sample_ids, "modulation": labels, "snr_db": snrs}
    targets = {"snr_ridge": snrs, "snr_logistic": ((snrs + 20) / 2).astype(np.int64), "modulation_logistic": labels}
    for target_name, target in targets.items():
        kind = "ridge" if target_name.endswith("ridge") else "logistic"
        log(f"Phase6 {mode}: fitting {target_name} on {len(train)} training rows")
        model, scaler, selection = select_probe(features[train], target[train], features[val], target[val], kind, config)
        # Freeze selected state and validation table on disk before test evaluation.
        write_json(output / f"{target_name}_selection.json", selection)
        write_json(output / f"{target_name}_state.json", _probe_state(model, scaler, sample_ids[train], kind))
        pred = np.empty(len(features), dtype=np.float64 if kind == "ridge" else np.int64)
        for partition, indices in partitions.items():
            pred[indices] = model.predict(scaler.transform(features[indices]))
            observed, estimated = target[indices], pred[indices]
            if kind == "ridge":
                scores = {"mae_db": float(mean_absolute_error(observed, estimated)), "rmse_db": float(np.sqrt(mean_squared_error(observed, estimated)))}
            else:
                scores = {"accuracy": float(accuracy_score(observed, estimated)), "macro_f1": float(f1_score(observed, estimated, average="macro", zero_division=0))}
                np.save(output / f"{target_name}_{partition}_confusion.npy", confusion_matrix(observed, estimated, labels=np.unique(target[train])))
            metrics.extend({"mode": mode, "target": target_name, "partition": partition, "metric": name, "value": score, "n": len(indices)} for name, score in scores.items())
            if partition == "test":
                for group_name, groups in (("true_snr_db", snrs), ("modulation", labels)):
                    for group in np.unique(groups[indices]):
                        idx = indices[groups[indices] == group]
                        error = float(mean_absolute_error(target[idx], pred[idx])) if kind == "ridge" else float(accuracy_score(target[idx], pred[idx]))
                        per_group.append({"mode": mode, "target": target_name, "group_type": group_name, "group": float(group), "metric": "mae_db" if kind == "ridge" else "accuracy", "value": error, "n": len(idx)})
        predictions[target_name] = pred
    # Baselines are fitted on training targets only and evaluated in all partitions.
    baseline_db = float(np.mean(snrs[train]))
    for partition, indices in partitions.items():
        metrics.extend([
            {"mode": mode, "target": "train_mean_snr", "partition": partition, "metric": "mae_db", "value": float(np.mean(np.abs(snrs[indices] - baseline_db))), "n": len(indices)},
            {"mode": mode, "target": "train_majority_snr_bin", "partition": partition, "metric": "accuracy", "value": float(np.mean(snrs[indices] == np.unique(snrs[train], return_counts=True)[0][np.argmax(np.unique(snrs[train], return_counts=True)[1])])), "n": len(indices)},
        ])
    predictions.update({name + "_indices": indices for name, indices in partitions.items()})
    np.savez_compressed(output / "all_partition_predictions.npz", **predictions)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "per_group_metrics.csv", per_group)
    return metrics


def transform_audit(signals, partitions, sample_ids, config, output, mode, log):
    state = fit_transform(signals, partitions["train"], mode, config["epsilon"])
    state["fit_sample_ids_sha256"] = array_hash(sample_ids[partitions["train"]])
    state_hash = json_hash(state)
    output.mkdir(parents=True)
    write_json(output / "preprocessing_state.json", state)
    features = np.empty((len(signals), len(FEATURE_NAMES)), dtype=np.float64)
    # Small batches bound memory; no statistics are fitted in this loop.
    for start in range(0, len(signals), 2048):
        features[start:start + 2048] = frame_features(transform(signals[start:start + 2048], state), config["epsilon"])
    np.savez_compressed(output / "features.npz", features=features, feature_names=np.asarray(FEATURE_NAMES), sample_ids=sample_ids, **{name + "_indices": idx for name, idx in partitions.items()})
    test_idx = partitions["test"]
    errors = np.empty(len(test_idx))
    reordered_errors = np.empty(len(test_idx))
    future_errors = np.empty(len(test_idx))
    expected = transform(signals[test_idx], state)
    permutation = np.random.default_rng(config["streaming"]["order_seed"]).permutation(len(test_idx))
    for position, row in enumerate(test_idx):
        streamed = transform(signals[row:row + 1], state)[0]
        errors[position] = np.max(np.abs(streamed - expected[position]))
        p = permutation[position]
        reordered_errors[p] = np.max(np.abs(transform(signals[test_idx[p]:test_idx[p] + 1], state)[0] - expected[p]))
        future_row = test_idx[(position + 1) % len(test_idx)]
        altered = np.stack([signals[row], signals[future_row] * config["streaming"]["future_test_gain"]])
        future_errors[position] = np.max(np.abs(transform(altered, state)[0] - expected[position]))
        if position and position % 10000 == 0:
            log(f"Phase6 {mode}: streaming transform {position}/{len(test_idx)}")
    unchanged = state_hash == json_hash(state)
    np.savez_compressed(output / "transform_streaming.npz", sample_ids=sample_ids[test_idx], batch_vs_frame_max_abs=errors, reordered_max_abs=reordered_errors, changed_future_max_abs=future_errors, permutation=permutation)
    summary = {"n": len(test_idx), "batch_frame_max_abs": float(errors.max()), "reorder_max_abs": float(reordered_errors.max()), "future_max_abs": float(future_errors.max()), "state_unchanged": unchanged, "state_before_sha256": state_hash, "state_after_sha256": json_hash(state), "transformed_test_sha256": array_hash(expected), "zero_energy_input_frames": int(np.sum(np.sum(np.asarray(signals[test_idx], dtype=np.float64) ** 2, axis=(1, 2)) == 0)), "passed": bool(unchanged and np.all(errors == 0) and np.all(reordered_errors == 0) and np.all(future_errors == 0))}
    write_json(output / "transform_streaming.json", summary)
    return features, summary


def load_phase3(root):
    scripts = root / "scripts/v2"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("phase6_frozen_phase3", scripts / "run_phase3.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_streaming(root, phase3, dataset, split, config, output, dependencies, log):
    import torch
    phase3_config = phase3.load_phase3_config(root / "configs/v2/phase3.yaml")
    ledger_path = root / "results/v2/phase3/phase3_ledger.json"
    ledger = phase3._load_ledger(ledger_path)
    dependencies[ledger_path.relative_to(root).as_posix()] = sha(ledger_path)
    records = [r for r in ledger["records"] if r["status"] == "completed"]
    rows = []
    output.mkdir()
    for seed in config["streaming"]["seeds"]:
        for model_id in config["streaming"]["models"]:
            matches = [r for r in records if r["model_id"] == model_id and r["seed"] == seed]
            if len(matches) != 1:
                raise ValueError("streaming needs one hash-bound registered checkpoint")
            record = matches[0]; run = (root / record["attempt_path"]).resolve()
            if not run.is_relative_to(root) or sha(run / "result.json") != record["result_sha256"]:
                raise ValueError("checkpoint ledger mismatch")
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            for name in ("checkpoint", "predictions", "run_spec", "mechanism"):
                path = run / result["artifacts"][name]
                if not path.resolve().is_relative_to(run) or sha(path) != result["artifacts"][name + "_sha256"]:
                    raise ValueError("checkpoint upstream hash mismatch")
                dependencies[path.relative_to(root).as_posix()] = sha(path)
            dependencies[(run / "result.json").relative_to(root).as_posix()] = sha(run / "result.json")
            with np.load(run / result["artifacts"]["predictions"], allow_pickle=False) as bundle:
                expected = np.asarray(bundle["logits"])
                bins = np.asarray(bundle["snr_bin"])
                ids = np.asarray(bundle["sample_ids"])
                if not np.array_equal(ids, np.asarray(split.sample_ids)[split.test_idx]) or not np.array_equal(bundle["y_true"], dataset["labels"][split.test_idx]) or not np.array_equal(bundle["snr_db"], dataset["snrs"][split.test_idx]):
                    raise ValueError("checkpoint test row binding mismatch")
            model = phase3._model(phase3_config, model_id).cpu().eval()
            model.load_state_dict(torch.load(run / result["artifacts"]["checkpoint"], map_location="cpu", weights_only=True), strict=True)
            before = {name: array_hash(buffer.cpu().numpy()) for name, buffer in model.named_buffers()}
            actual = np.empty_like(expected)
            start_time = time.perf_counter()
            with torch.no_grad():
                for position, row in enumerate(split.test_idx):
                    x = torch.as_tensor(dataset["signals"][row:row + 1], dtype=torch.float32)
                    snr = torch.as_tensor(dataset["snrs"][row:row + 1], dtype=torch.float32)
                    b = torch.as_tensor(bins[position:position + 1], dtype=torch.long)
                    actual[position] = model.forward_batch(x, snr_db=snr, snr_bin=b)[0].numpy()[0]
                    if position and position % 10000 == 0:
                        log(f"Phase6 {model_id}/seed{seed}: checkpoint batch1 {position}/{len(expected)}")
            after = {name: array_hash(buffer.cpu().numpy()) for name, buffer in model.named_buffers()}
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            threshold = config["streaming"]["atol"] + config["streaming"]["rtol"] * np.abs(expected)
            row_pass = (difference <= threshold).all(axis=1)
            same_decisions = actual.argmax(axis=1) == expected.argmax(axis=1)
            finite = bool(np.isfinite(actual).all())
            name = f"{model_id}_seed{seed}"
            np.savez_compressed(output / f"{name}.npz", sample_ids=ids, actual_logits=actual, reference_logits=expected, per_row_max_abs=difference.max(axis=1), per_row_tolerance_pass=row_pass, per_row_same_argmax=same_decisions)
            item = {"model_id": model_id, "seed": seed, "n": len(ids), "max_abs": float(difference.max()), "tolerance_failed_rows": int((~row_pass).sum()), "argmax_changed_rows": int((~same_decisions).sum()), "buffers_unchanged": before == after, "buffers_before": before, "buffers_after": after, "finite": finite, "passed": bool(finite and row_pass.all() and same_decisions.all() and before == after), "elapsed_seconds": time.perf_counter() - start_time}
            write_json(output / f"{name}.json", item); rows.append(item)
    return rows


def coupling_table(features, labels, snrs, partitions, mode):
    rows = []
    for partition, indices in partitions.items():
        for snr in np.unique(snrs[indices]):
            for label in np.unique(labels[indices]):
                selected = indices[(snrs[indices] == snr) & (labels[indices] == label)]
                for column, name in enumerate(FEATURE_NAMES):
                    values = features[selected, column]
                    rows.append({"mode": mode, "partition": partition, "snr_db": float(snr), "modulation": int(label), "statistic": name, "n": len(selected), "mean": float(values.mean()), "std": float(values.std(ddof=1))})
    return rows


def plot_results(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    # Separate single-panel figures avoid implying comparable metric units.
    descriptions = [("snr_ridge", "mae_db", "SNR estimation MAE (dB)", "snr_mae"), ("snr_logistic", "accuracy", "SNR-bin accuracy", "snr_bin_accuracy"), ("modulation_logistic", "accuracy", "Modulation accuracy from statistics", "modulation_coupling")]
    modes = ["legacy_identity", "train_global", "per_frame_rms"]
    labels = ["Legacy identity", "Train global", "Per-frame RMS"]
    colors = ["#737373", "#4477AA", "#228833"]
    for target, metric, ylabel, name in descriptions:
        selected = [next(row for row in metrics if row["mode"] == mode and row["target"] == target and row["partition"] == "test" and row["metric"] == metric) for mode in modes]
        write_csv(output / f"{name}_source.csv", selected)
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        ax.bar(labels, [row["value"] for row in selected], color=colors, width=.58)
        ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
        ax.set_title("Fixed test split; validation-selected probes", fontsize=10)
        fig.subplots_adjust(left=.16, right=.97, bottom=.20, top=.86)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=300)
        plt.close(fig)

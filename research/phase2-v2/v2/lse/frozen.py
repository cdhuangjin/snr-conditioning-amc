"""Frozen-classifier substitution: oracle / Ridge / TinyCNN condition sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from v2.phase8.core import grouped_metrics, head_logits
from .shared import (
    load_frozen_model,
    load_project,
    load_ridge_estimator,
    pooled_features,
    SEEDS,
)


def _row(model_id, seed, condition, metrics):
    overall = metrics["overall"]
    bands = metrics.get("bands", {})

    def band(name, key):
        if name in bands:
            return bands[name].get(key)
        return None

    return {
        "classifier_seed": seed,
        "model": model_id,
        "condition_source": condition,
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "macro_f1": overall["macro_f1"],
        "concentration": overall["concentration"],
        "prediction_entropy": overall["prediction_entropy"],
        "low_snr_accuracy": band("low", "accuracy"),
        "mid_snr_accuracy": band("mid", "accuracy"),
        "high_snr_accuracy": band("high", "accuracy"),
        "low_snr_concentration": band("low", "concentration"),
        "low_snr_entropy": band("low", "prediction_entropy"),
        "count": overall["count"],
    }


def _classify_without_condition(model, features, device, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            chunk = torch.as_tensor(features[start : start + batch_size], device=torch.device(device))
            out.append(model.classify_pooled(chunk).cpu().numpy())
    return np.concatenate(out)


def run_frozen(root, out_root, *, raw=None, device="cpu", batch_size=64):
    """Run frozen M3/M6 substitution (oracle/ridge/tinycnn) and the M0 baseline."""
    out_root = Path(out_root)
    root = Path(root).resolve()
    _phase3, p3config, runs, dataset, split = load_project(root)
    signals = dataset["signals"]
    y = dataset["labels"]
    truth = dataset["snrs"].astype(np.float32)
    test = np.asarray(split.test_idx, dtype=np.int64)
    test_signals = signals[test]
    y_test = y[test]
    truth_test = truth[test]

    if raw is None:
        raw, _ridge_evidence = load_ridge_estimator(root)
    ridge_test = np.asarray(raw)[test].astype(np.float32)
    estimator_predictions = {
        seed: np.load(
            out_root / "raw" / f"estimator_predictions_seed{seed}.npz", allow_pickle=False
        )["predicted_snr_db"][test].astype(np.float32)
        for seed in SEEDS
    }

    rows = []
    m0_rows = []
    cache = {}
    for model_id in ("M3", "M6"):
        for seed in SEEDS:
            key = (model_id, seed)
            if key not in cache:
                model = load_frozen_model(root, p3config, runs, model_id, seed, device)
                cache[key] = (model, pooled_features(model, test_signals, device, batch_size))
            model, features = cache[key]

            for condition, source in (
                ("oracle", truth_test),
                ("ridge", ridge_test),
                ("tinycnn", estimator_predictions[seed]),
            ):
                logits = head_logits(model, features, source, batch_size)
                metrics = grouped_metrics(y_test, logits, truth_test)
                rows.append(_row(model_id, seed, condition, metrics))
            print(f"[frozen] {model_id} seed {seed} done", flush=True)

    # M0 baseline (no condition), five seeds.
    for seed in SEEDS:
        model = load_frozen_model(root, p3config, runs, "M0", seed, device)
        features = pooled_features(model, test_signals, device, batch_size)
        logits = _classify_without_condition(model, features, device, batch_size)
        metrics = grouped_metrics(y_test, logits, truth_test)
        m0_rows.append(_row("M0", seed, "none", metrics))
    print("[frozen] M0 baseline done", flush=True)

    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "frozen_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (raw_dir / "frozen_m0_rows.json").write_text(
        json.dumps(m0_rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return rows, m0_rows

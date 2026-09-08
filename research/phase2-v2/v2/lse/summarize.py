"""Build the LSE tables (1-5) and figures (1-2) from persisted raw evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .shared import (
    estimator_loss,
    load_project,
    load_ridge_estimator,
    pearson_r,
    SEEDS,
)


CLIP_MIN_DB = -20.0
CLIP_MAX_DB = 18.0
SNR_VALUES_DB = list(range(-20, 20, 2))


def _write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _est_predictions(out_root, seed, test_idx):
    with np.load(out_root / "raw" / f"estimator_predictions_seed{seed}.npz", allow_pickle=False) as b:
        return b["predicted_snr_db"][test_idx].astype(np.float64)


def _table1(root, out_root, raw, test_idx, truth_test):
    import json

    tables = out_root / "tables"
    rows = []
    ridge_pred = np.asarray(raw)[test_idx].astype(np.float64)
    m = estimator_loss(ridge_pred, truth_test)
    rows.append([
        "ridge", "ridge", f"{m['mae_db']:.4f}", f"{m['rmse_db']:.4f}",
        f"{pearson_r(ridge_pred, truth_test):.4f}", f"{m['bias_db']:.4f}",
        f"{np.mean(ridge_pred < CLIP_MIN_DB):.4f}", f"{np.mean(ridge_pred > CLIP_MAX_DB):.4f}",
    ])
    for seed in SEEDS:
        pred = _est_predictions(out_root, seed, test_idx)
        m = estimator_loss(pred, truth_test)
        rows.append([
            seed, "tinycnn", f"{m['mae_db']:.4f}", f"{m['rmse_db']:.4f}",
            f"{pearson_r(pred, truth_test):.4f}", f"{m['bias_db']:.4f}",
            f"{np.mean(pred < CLIP_MIN_DB):.4f}", f"{np.mean(pred > CLIP_MAX_DB):.4f}",
        ])
    _write_csv(
        tables / "Table_LSE_1_estimator_metrics.csv",
        ["seed", "estimator", "MAE", "RMSE", "Pearson_r", "Bias", "ClipLowRate", "ClipHighRate"],
        rows,
    )
    return rows


def _table2(root, out_root, raw, test_idx, truth_test):
    tables = out_root / "tables"
    ridge_pred = np.asarray(raw)[test_idx].astype(np.float64)
    per_seed = {s: _est_predictions(out_root, s, test_idx) for s in SEEDS}
    rows = []
    for snr in SNR_VALUES_DB:
        mask = truth_test == snr
        if not mask.any():
            continue
        r_mae = float(np.mean(np.abs(ridge_pred[mask] - snr)))
        r_bias = float(np.mean(ridge_pred[mask] - snr))
        cnn = np.stack([per_seed[s][mask] for s in SEEDS])
        c_mae = float(np.mean(np.mean(np.abs(cnn - snr), axis=1)))
        c_bias = float(np.mean(np.mean(cnn - snr, axis=1)))
        c_mae_sd = float(np.std([np.mean(np.abs(cnn[i] - snr)) for i in range(len(SEEDS))], ddof=1))
        rows.append([snr, f"{r_mae:.4f}", f"{c_mae:.4f}", f"{c_mae_sd:.4f}", f"{r_bias:.4f}", f"{c_bias:.4f}"])
    _write_csv(
        tables / "Table_LSE_2_mae_by_snr.csv",
        ["true_snr", "ridge_mae", "cnn_mae", "cnn_mae_sd", "ridge_bias", "cnn_bias"],
        rows,
    )
    return rows


def _table3(out_root, rows):
    tables = out_root / "tables"
    header = [
        "classifier_seed", "model", "condition_source", "accuracy", "balanced_accuracy",
        "macro_f1", "concentration", "prediction_entropy", "low_snr_accuracy",
        "mid_snr_accuracy", "high_snr_accuracy", "low_snr_concentration",
        "low_snr_entropy", "count",
    ]
    data = [[r[k] for k in header] for r in rows]
    _write_csv(tables / "Table_LSE_3_frozen_classification.csv", header, data)
    return data


def _table4(out_root, rows, m0_rows):
    tables = out_root / "tables"
    m0_by_seed = {r["classifier_seed"]: r["accuracy"] for r in m0_rows}
    model_conds = [("M3", c) for c in ("oracle", "ridge", "tinycnn")] + [("M6", c) for c in ("oracle", "ridge", "tinycnn")]
    header = ["model", "condition", "accuracy_mean", "accuracy_sd", "gain_vs_m0_mean", "gain_vs_m0_sd", "drop_vs_oracle_mean", "drop_vs_oracle_sd"]
    out = []
    for model, cond in model_conds:
        accs = np.array([r["accuracy"] for r in rows if r["model"] == model and r["condition_source"] == cond])
        m0 = np.array([m0_by_seed[s] for s in SEEDS])
        oracle = np.array([r["accuracy"] for r in rows if r["model"] == model and r["condition_source"] == "oracle"])
        gain = accs - m0
        drop = accs - oracle
        out.append([model, cond, f"{accs.mean():.4f}", f"{accs.std(ddof=1):.4f}",
                    f"{gain.mean():.4f}", f"{gain.std(ddof=1):.4f}",
                    f"{drop.mean():.4f}", f"{drop.std(ddof=1):.4f}"])
    for seed in SEEDS:
        out.append(["M0", "none", f"{m0_by_seed[seed]:.4f}", "0.0000", "0.0000", "0.0000", "0.0000", "0.0000"])
    _write_csv(tables / "Table_LSE_4_summary.csv", header, out)
    return out


def _paired(out_root, table_rows, rows):
    acc = {
        (r["model"], r["classifier_seed"], r["condition_source"]): r["accuracy"]
        for r in rows
    }
    m0 = {r["classifier_seed"]: r["accuracy"] for r in table_rows}
    comparisons = [
        ("tinycnn-vs-ridge", "M3", "tinycnn", "ridge"),
        ("tinycnn-vs-ridge", "M6", "tinycnn", "ridge"),
        ("tinycnn-vs-oracle", "M3", "tinycnn", "oracle"),
        ("tinycnn-vs-oracle", "M6", "tinycnn", "oracle"),
    ]
    out = []
    for comparison, model, left, right in comparisons:
        diffs = np.array([acc[(model, s, left)] - acc[(model, s, right)] for s in SEEDS]) * 100.0
        half = t.ppf(0.975, 4) * diffs.std(ddof=1) / np.sqrt(5)
        out.append([comparison, model, f"{diffs.mean():.4f}", f"{diffs.std(ddof=1):.4f}",
                    f"{diffs.mean() - half:.4f}", f"{diffs.mean() + half:.4f}"])
    # TinyCNN vs M0
    for model in ("M3", "M6"):
        diffs = np.array([acc[(model, s, "tinycnn")] - m0[s] for s in SEEDS]) * 100.0
        half = t.ppf(0.975, 4) * diffs.std(ddof=1) / np.sqrt(5)
        out.append(["tinycnn-vs-m0", model, f"{diffs.mean():.4f}", f"{diffs.std(ddof=1):.4f}",
                    f"{diffs.mean() - half:.4f}", f"{diffs.mean() + half:.4f}"])
    _write_csv(
        out_root / "tables" / "Table_LSE_5_paired_CI.csv",
        ["comparison", "model", "mean_difference_pp", "sd_difference", "ci95_lower", "ci95_upper"],
        out,
    )
    return out


def _fig1(out_root, raw, test_idx, truth_test):
    figures = out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ridge_pred = np.asarray(raw)[test_idx].astype(np.float64)
    per_seed = {s: _est_predictions(out_root, s, test_idx) for s in SEEDS}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs, ridge_ys, cnn_ys, cnn_err = [], [], [], []
    for snr in SNR_VALUES_DB:
        mask = truth_test == snr
        if not mask.any():
            continue
        xs.append(snr)
        ridge_ys.append(np.mean(np.abs(ridge_pred[mask] - snr)))
        cnn_mae = [np.mean(np.abs(per_seed[s][mask] - snr)) for s in SEEDS]
        cnn_ys.append(float(np.mean(cnn_mae)))
        cnn_err.append(float(np.std(cnn_mae, ddof=1)))
    ax.errorbar(xs, cnn_ys, yerr=cnn_err, fmt="o-", color="#0072B2", label="TinyCNN", capsize=3)
    ax.plot(xs, ridge_ys, "s--", color="#D55E00", label="Ridge")
    ax.set_xlabel("True SNR (dB)")
    ax.set_ylabel("SNR estimation MAE (dB)")
    ax.set_title("Estimator MAE by SNR")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "Fig_LSE_1_estimator_mae_by_snr.png", dpi=300)
    fig.savefig(figures / "Fig_LSE_1_estimator_mae_by_snr.pdf")
    plt.close(fig)


def _fig2(out_root, rows, raw, test_idx, truth_test):
    figures = out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ridge_pred = np.asarray(raw)[test_idx].astype(np.float64)
    ridge_mae = float(np.mean(np.abs(ridge_pred - truth_test)))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model, color, marker in (("M3", "#0072B2", "o"), ("M6", "#D55E00", "s")):
        xs = []
        ys = []
        for seed in SEEDS:
            pred = _est_predictions(out_root, seed, test_idx)
            mae = float(np.mean(np.abs(pred - truth_test)))
            acc = next(
                r["accuracy"]
                for r in rows
                if r["model"] == model and r["classifier_seed"] == seed and r["condition_source"] == "tinycnn"
            )
            xs.append(mae)
            ys.append(acc * 100.0)
        ridge_acc = np.mean([r["accuracy"] for r in rows if r["model"] == model and r["condition_source"] == "ridge"]) * 100.0
        ax.scatter(xs, ys, color=color, marker=marker, label=f"{model} (TinyCNN)")
        ax.scatter([ridge_mae], [ridge_acc], color=color, marker="^", facecolors="none", edgecolors=color, s=70, label=f"{model} (Ridge)")
    ax.set_xlabel("Estimator test MAE (dB)")
    ax.set_ylabel("Downstream classification accuracy (%)")
    ax.set_title("Estimator error vs frozen-condition accuracy")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "Fig_LSE_2_estimator_error_vs_classification.png", dpi=300)
    fig.savefig(figures / "Fig_LSE_2_estimator_error_vs_classification.pdf")
    plt.close(fig)


def summarize(root, out_root, *, raw=None):
    root = Path(root).resolve()
    out_root = Path(out_root)
    _phase3, _p3config, _runs, dataset, split = load_project(root)
    test = np.asarray(split.test_idx, dtype=np.int64)
    truth_test = np.asarray(dataset["snrs"])[test].astype(np.float64)
    if raw is None:
        raw, _evidence = load_ridge_estimator(root)

    rows = _read_json(out_root / "raw" / "frozen_rows.json")
    m0_rows = _read_json(out_root / "raw" / "frozen_m0_rows.json")
    _table1(root, out_root, raw, test, truth_test)
    _table2(root, out_root, raw, test, truth_test)
    _table3(out_root, rows)
    _table4(out_root, rows, m0_rows)
    _paired(out_root, m0_rows, rows)
    _fig1(out_root, raw, test, truth_test)
    _fig2(out_root, rows, raw, test, truth_test)
    return {
        "tables": sorted(p.name for p in (out_root / "tables").glob("*.csv")),
        "figures": sorted(p.name for p in (out_root / "figures").glob("*.*")),
    }


def estimator_metrics(root, out_root, *, raw=None):
    """Build only the estimator tables (Table_LSE_1 and Table_LSE_2)."""
    root = Path(root).resolve()
    out_root = Path(out_root)
    _phase3, _p3config, _runs, dataset, split = load_project(root)
    test = np.asarray(split.test_idx, dtype=np.int64)
    truth_test = np.asarray(dataset["snrs"])[test].astype(np.float64)
    if raw is None:
        raw, _evidence = load_ridge_estimator(root)
    table1 = _table1(root, out_root, raw, test, truth_test)
    table2 = _table2(root, out_root, raw, test, truth_test)
    return {"estimator_table_rows": len(table1), "mae_by_snr_rows": len(table2)}


def _read_json(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))

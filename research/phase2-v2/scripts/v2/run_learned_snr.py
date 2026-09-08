"""TinyCNN-SNR supplement: train, frozen-substitute, and summarize.

Modes
-----
train   -- train the estimator for the requested seeds (writes checkpoints/predictions).
frozen  -- run frozen M3/M6 substitution for oracle/ridge/tinycnn + M0 baseline.
summary -- build Table_LSE_1..5 and Fig_LSE_1..2 from persisted raw evidence.
all     -- train + frozen + summary.
smoke   -- train seed 2022 for 2 epochs on a subsample (pipeline check only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.lse.shared import load_project, load_ridge_estimator, SEEDS  # noqa: E402
from v2.lse.train import train_seed  # noqa: E402
from v2.lse.frozen import run_frozen  # noqa: E402
from v2.lse.summarize import estimator_metrics, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["train", "eval", "frozen", "summary", "all", "smoke"])
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--out", default=str(ROOT / "results" / "learned_snr_estimator"))
    parser.add_argument("--seeds", default="2022,2023,2024,2025,2026")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--frozen-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    seeds = [int(s) for s in args.seeds.split(",") if s]

    if args.mode in ("train", "all"):
        project = load_project(root)
        results = {}
        for seed in seeds:
            results[seed] = train_seed(
                root,
                out,
                seed,
                project=project,
                device=args.device,
                batch_size=args.train_batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
        print("TRAIN_DONE", json.dumps({str(k): v["best_val_mae"] for k, v in results.items()}), flush=True)

    if args.mode == "smoke":
        result = train_seed(root, out, 2022, device=args.device, batch_size=256, max_epochs=50, patience=5, smoke=True)
        print("SMOKE_DONE", json.dumps(result["best_val_mae"]), flush=True)
        return 0

    if args.mode in ("frozen", "all"):
        raw, _evidence = load_ridge_estimator(root)
        rows, m0_rows = run_frozen(root, out, raw=raw, device=args.device, batch_size=args.frozen_batch_size)
        print(f"FROZEN_DONE cond_rows={len(rows)} m0_rows={len(m0_rows)}", flush=True)

    if args.mode == "eval":
        raw, _evidence = load_ridge_estimator(root)
        built = estimator_metrics(root, out, raw=raw)
        print("EVAL_DONE", json.dumps(built), flush=True)

    if args.mode in ("summary", "all"):
        raw, _evidence = load_ridge_estimator(root)
        built = summarize(root, out, raw=raw)
        print("SUMMARY_DONE", json.dumps(built), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

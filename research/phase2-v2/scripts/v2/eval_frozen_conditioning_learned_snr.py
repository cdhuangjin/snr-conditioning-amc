"""Frozen M0/M3/M6 substitution for oracle / ridge / tinycnn conditions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.lse.shared import load_ridge_estimator  # noqa: E402
from v2.lse.frozen import run_frozen  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--out", default=str(ROOT / "results/learned_snr_estimator"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    root = Path(args.repo_root).resolve()
    raw, _ = load_ridge_estimator(root)
    rows, m0 = run_frozen(root, Path(args.out).resolve(), raw=raw, device=args.device, batch_size=args.batch_size)
    print(json.dumps({"condition_rows": len(rows), "m0_rows": len(m0)}))


if __name__ == "__main__":
    raise SystemExit(main())

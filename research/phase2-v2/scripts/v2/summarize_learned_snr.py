"""Build all LSE tables and figures from persisted raw evidence."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.lse.shared import load_ridge_estimator  # noqa: E402
from v2.lse.summarize import summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--out", default=str(ROOT / "results/learned_snr_estimator"))
    args = p.parse_args()
    root = Path(args.repo_root).resolve()
    raw, _ = load_ridge_estimator(root)
    print(json.dumps(summarize(root, Path(args.out).resolve(), raw=raw)))


if __name__ == "__main__":
    raise SystemExit(main())

"""Thin wrapper to train the TinyCNN-SNR estimator."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.lse.shared import SEEDS, load_project  # noqa: E402
from v2.lse.train import train_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--out", default=str(ROOT / "results/learned_snr_estimator"))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    project = load_project(Path(args.repo_root).resolve())
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        print(train_seed(
            Path(args.repo_root).resolve(), Path(args.out).resolve(), seed,
            project=project, device=args.device, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience,
        ))


if __name__ == "__main__":
    raise SystemExit(main())

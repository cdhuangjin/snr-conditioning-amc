"""Execute or independently replay Phase 6 preprocessing evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.phase6.runner import run, validate_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/v2/phase6.yaml"))
    parser.add_argument("--validate")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_attempt(Path(args.validate), ROOT)))
        return 0
    return run(args.config, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

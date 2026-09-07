"""Publish or validate the reviewed bounded Phase 6 composite receipt."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.phase6.composite import validate_composite, publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        data = validate_composite(ROOT, ROOT / "results/v2/phase6_preprocessing/current.json")
        print(json.dumps({"status": "VALIDATED_BOUNDED_COMPOSITE", "data_attempt": data.relative_to(ROOT).as_posix(), "original_fp32_tight_replay": "FAILED_UNCHANGED"}))
    else:
        print(json.dumps(publish(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

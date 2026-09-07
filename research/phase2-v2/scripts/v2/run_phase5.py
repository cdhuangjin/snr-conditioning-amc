"""Run or independently replay the preregistered Phase 5 toy matrix."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase5.runner import run, validate_attempt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/v2/phase5.yaml'))
    parser.add_argument('--repository-root',default=str(ROOT))
    parser.add_argument('--validate')
    args=parser.parse_args()
    if args.validate:
        print(json.dumps(validate_attempt(Path(args.validate),Path(args.repository_root))))
        return 0
    return run(args.config,args.repository_root)


if __name__=='__main__': raise SystemExit(main())

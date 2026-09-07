"""Prepare smoke, execute gated reliability-floor evaluation, or verify artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase10.runner import execute, validate, smoke


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/v2/phase10.yaml'))
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--smoke',action='store_true')
    group.add_argument('--validate')
    group.add_argument('--resume')
    args=parser.parse_args()
    result=smoke(ROOT) if args.smoke else validate(Path(args.validate),ROOT) if args.validate else execute(args.config,ROOT,args.resume)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

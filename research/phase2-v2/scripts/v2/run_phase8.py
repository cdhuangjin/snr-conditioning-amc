"""Execute gated Phase 8 controls or independently validate an existing attempt."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase8.runner import execute,validate,validate_config
from v2.phase8.evidence import check_gates
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/v2/phase8.yaml'));parser.add_argument('--repository-root',default=str(ROOT));parser.add_argument('--validate');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();root=Path(args.repository_root)
    if args.preflight:
        config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));validate_config(config)
        check_gates(root,config)
        print(json.dumps({'ready':True}));return 0
    result=validate(args.validate,root) if args.validate else execute(args.config,root)
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())

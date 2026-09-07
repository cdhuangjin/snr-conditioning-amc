"""Phase 9 is gated by actual Phase 8 COMPLETE; --preflight never trains."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase9.runner import execute,validate,phase8_gate,validate_config
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/v2/phase9.yaml'));parser.add_argument('--repository-root',default=str(ROOT));parser.add_argument('--preflight',action='store_true');parser.add_argument('--validate')
    args=parser.parse_args();root=Path(args.repository_root)
    if args.preflight:
        config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));validate_config(config);phase8_gate(root,config);print(json.dumps({'ready':True}));return 0
    print(json.dumps(validate(args.validate,root) if args.validate else execute(args.config,root)));return 0


if __name__=='__main__':raise SystemExit(main())

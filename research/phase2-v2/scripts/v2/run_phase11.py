import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase11.runner import execute,validate,check_gates,validate_config
import yaml

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/v2/phase11.yaml'));parser.add_argument('--preflight',action='store_true');parser.add_argument('--validate');args=parser.parse_args()
    if args.validate:result=validate(args.validate,ROOT)
    elif args.preflight:
        config=yaml.safe_load(Path(args.config).read_text());validate_config(config);check_gates(ROOT,config);result={'status':'GATES_READY'}
    else:result=execute(args.config,ROOT)
    print(json.dumps(result,indent=2))

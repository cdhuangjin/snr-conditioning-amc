"""Phase12 full execution is gated; --smoke runs only a tiny separate protocol."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase12.runner import execute,validate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/v2/phase12_execution.yaml'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--validate')
    args=parser.parse_args()
    if args.validate:print(validate(Path(args.validate),scientific=not args.smoke))
    else:execute(args.config,smoke=args.smoke,resume=args.resume)

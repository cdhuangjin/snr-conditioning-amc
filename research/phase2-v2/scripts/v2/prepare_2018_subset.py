"""Create or independently replay the prospective fixed 2018 data subset."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.phase11.subset import build_subset, validate_subset, sha

SOURCE = ROOT.parents[1]/'data/GOLD_XYZ_OSC.0001_1024.hdf5'
OUTPUT = ROOT/'results/v2/phase11_preflight/2018_subset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    inventory = ROOT/'results/v2/phase11_preflight/2018_inventory'
    manifest = json.loads((inventory/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((inventory/'summary.json').read_text(encoding='utf-8'))
    if sha(inventory/'summary.json') != manifest['files']['summary.json'] or Path(summary['source']).resolve() != SOURCE.resolve():
        raise ValueError('inventory binding differs')
    if summary['source_sha256'] != 'e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38' or summary['rows'] != 2555904:
        raise ValueError('unexpected approved source')
    if not args.validate_only:
        dependencies = {str(path.resolve()): sha(path) for path in [inventory/'manifest.json', inventory/'summary.json', Path(__file__)]}
        print('BUILDING_FIXED_SUBSET: 624 cells x 1024; no model fitting', flush=True)
        result = build_subset(SOURCE, OUTPUT, grid=list(range(-20, 31, 2)), classes=24, per_cell=1024,
                              seed=20260906, expected_sha=summary['source_sha256'], source_dependencies=dependencies)
        print(json.dumps(result), flush=True)
    print(json.dumps(validate_subset(OUTPUT, SOURCE, ROOT, replay_source=True)), flush=True)


if __name__ == '__main__':
    main()

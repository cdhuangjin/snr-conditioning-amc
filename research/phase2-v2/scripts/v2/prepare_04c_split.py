"""Bind audited 04C dedup rows to fixed per-cell train/validation/test IDs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.phase11.subset import sha, dump, cell_ids, split_rows

SOURCE = ROOT.parents[1]/'data/RML2016.04c.dat'
EXPECTED = '3794a88b8d4bc8d20ac2a9e4516f5443283459ff9efd8c992055fe2745311200'
AUDIT = ROOT/'results/v2/phase11_preflight/04c_source_dedup'
OUTPUT = ROOT/'results/v2/phase11_preflight/04c_split'


def audit_input():
    m = json.loads((AUDIT/'manifest.json').read_text(encoding='utf-8'))
    if m['status'] != 'SOURCE_AND_DEDUP_AUDITED_NOT_TRAINED' or m['dependencies'].get(str(SOURCE)) != EXPECTED:
        raise ValueError('unexpected dedup audit/source binding')
    for name, value in m['files'].items():
        path = (AUDIT/name).resolve()
        if path.parent != AUDIT.resolve() or sha(path) != value:
            raise ValueError('dedup artifact hash differs')
    for name, value in m['dependencies'].items():
        path = Path(name).resolve()
        if (path != SOURCE.resolve() and not path.is_relative_to(ROOT)) or sha(path) != value:
            raise ValueError('dedup source outside allowlist or changed')
    with np.load(AUDIT/'dedup_map.npz', allow_pickle=False) as b:
        data = {key: b[key] for key in b.files}
    grid = list(range(-20, 19, 2))
    labels, snr = data['original_labels'], data['original_snrs']
    cells = cell_ids(labels, snr, grid, 11)
    retained = data['retained_indices']
    if len(labels) != 162060 or len(data['class_names']) != 11 or np.any(np.diff(retained) <= 0):
        raise ValueError('dedup scope/retained ordering differs')
    accounted = np.concatenate([retained, data['duplicate_indices'], data['conflicting_indices']])
    if not np.array_equal(np.sort(accounted), np.arange(len(labels))):
        raise ValueError('dedup accounting does not close')
    parts = split_rows(retained, cells, grid, 11, 20260906)
    identity = dict(source_rows=np.arange(len(labels)), labels=labels, snr_db=snr,
                    snr_bin=np.searchsorted(grid, snr), class_names=data['class_names'],
                    original_within_cell=data['original_within_cell'], retained_indices=retained,
                    duplicate_indices=data['duplicate_indices'], conflicting_indices=data['conflicting_indices'],
                    **{key+'_indices': value for key, value in parts.items()})
    return identity, parts


def validate():
    manifest = json.loads((OUTPUT/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'DATA_PREPARED_NOT_TRAINED':
        raise ValueError('split is not data-only preparation')
    if {p.name for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json'} != set(manifest['files']):
        raise ValueError('split closure differs')
    for name, value in manifest['files'].items():
        if (OUTPUT/name).resolve().parent != OUTPUT.resolve() or sha(OUTPUT/name) != value:
            raise ValueError('split artifact changed')
    for name, value in manifest['dependencies'].items():
        path = Path(name).resolve()
        if (path != SOURCE.resolve() and not path.is_relative_to(ROOT)) or sha(path) != value:
            raise ValueError('split dependency changed')
    expected, parts = audit_input()
    with np.load(OUTPUT/'identity.npz', allow_pickle=False) as actual:
        if set(actual.files) != set(expected):
            raise ValueError('split identity schema differs')
        for key, values in expected.items():
            np.testing.assert_array_equal(actual[key], values)
    summary = json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    if summary['partition_counts'] != {key: len(value) for key, value in parts.items()}:
        raise ValueError('split counts differ')
    return dict(status='VALIDATED_FIXED_SPLIT_NOT_TRAINED', retained_rows=len(expected['retained_indices']), partition_counts=summary['partition_counts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.validate_only:
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        identity, parts = audit_input()
        OUTPUT.mkdir(parents=True)
        np.savez_compressed(OUTPUT/'identity.npz', **identity)
        dump(OUTPUT/'config.json', dict(seed=20260906, training_seeds_separate=[2022,2023,2024,2025,2026],
             fractions={'train': .6, 'validation': .2, 'test': 'remainder'}, grid=list(range(-20,19,2)),
             rng='PCG64(SeedSequence([seed,class_index,snr_grid_index,1]))', numpy_version=np.__version__,
             layout='NCT', frame_samples=128, preprocessing='source float32 values unchanged',
             source_row_order='sorted pickle class then SNR then within-cell row', source_sha256=EXPECTED))
        dump(OUTPUT/'summary.json', dict(status='DATA_PREPARED_NOT_TRAINED', source=str(SOURCE), source_sha256=EXPECTED,
             selected_rows=len(identity['source_rows']), retained_rows=len(identity['retained_indices']),
             class_names=identity['class_names'].tolist(), partition_counts={key: len(value) for key,value in parts.items()},
             limitation='Exact dedup only; unequal cells require weighted and equal-cell reporting. No fitting performed.'))
        paths = [SOURCE, AUDIT/'manifest.json', AUDIT/'summary.json', AUDIT/'dedup_map.npz', Path(__file__).resolve(), ROOT/'v2/phase11/subset.py']
        dump(OUTPUT/'manifest.json', dict(status='DATA_PREPARED_NOT_TRAINED', files={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()},
             dependencies={str(p.resolve()):sha(p) for p in paths}))
    print(json.dumps(validate()), flush=True)


if __name__ == '__main__':
    main()

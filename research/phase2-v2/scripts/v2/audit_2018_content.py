"""Read-only full 2018 HDF5 inventory; no subset or training selection."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parents[1] / 'data/GOLD_XYZ_OSC.0001_1024.hdf5'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = ROOT / 'results/v2/phase11_preflight/2018_inventory'
    if output.exists():
        raise FileExistsError(output)
    before = SOURCE.stat()
    cells = {}
    rows = 0
    with h5py.File(SOURCE, 'r') as source:
        x, y, z = (source[k] for k in ('X', 'Y', 'Z'))
        if x.shape[1:] != (1024, 2) or y.shape != (len(x), 24) or z.shape != (len(x), 1):
            raise ValueError('Unexpected HDF5 dimensions')
        header = {k: {'shape': list(source[k].shape), 'dtype': str(source[k].dtype)} for k in ('X', 'Y', 'Z')}
        for start in range(0, len(x), 4096):
            signals, onehot, snr = x[start:start+4096], y[start:start+4096], z[start:start+4096, 0]
            if not np.isfinite(signals).all() or not np.isfinite(snr).all():
                raise ValueError('Nonfinite source values')
            if not np.isin(onehot, [0, 1]).all() or not (onehot.sum(1) == 1).all():
                raise ValueError('Labels are not one-hot')
            labels = onehot.argmax(1)
            pairs, counts = np.unique(np.stack([labels, snr], axis=1), axis=0, return_counts=True)
            for pair, count in zip(pairs, counts):
                key = tuple(map(int, pair))
                cells[key] = cells.get(key, 0) + int(count)
            rows += len(signals)
        attributes = list(source.attrs)
    digest = sha(SOURCE)
    after = SOURCE.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during audit')
    value = {'status': 'FULL_CONTENT_INVENTORIED_NOT_TRAINED', 'source': str(SOURCE),
             'source_sha256': digest, 'bytes': before.st_size, 'rows': rows,
             'header': header, 'root_attribute_names': attributes,
             'class_indices': sorted({key[0] for key in cells}),
             'snr_db': sorted({key[1] for key in cells}),
             'cells': [{'class_index': key[0], 'snr_db': key[1], 'count': count} for key, count in sorted(cells.items())],
             'finite_signal_rows_checked': rows, 'one_hot_rows_checked': rows,
             'limitations': ['Class names are not inferred from unlabeled numeric columns.',
                              'No historical download provenance or row independence inferred.',
                              'This inventory selects no training subset and fits no model.']}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    manifest = {'status': value['status'], 'files': {'summary.json': sha(output/'summary.json')},
                'dependencies': {str(SOURCE): digest, str(Path(__file__).resolve()): sha(__file__)}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': value['status'], 'rows': rows, 'cells': len(cells), 'sha256': digest}), flush=True)


if __name__ == '__main__':
    main()

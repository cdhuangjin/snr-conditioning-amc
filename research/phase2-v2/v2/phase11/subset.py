"""Fixed per-cell subset selection, canonical deduplication and splits."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
APPROVED_SOURCE = ROOT.parents[1]/'data/GOLD_XYZ_OSC.0001_1024.hdf5'
APPROVED_SHA = 'e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38'


def check_formal_contract(source, expected_sha, grid, classes, per_cell, seed):
    if Path(source).resolve() != APPROVED_SOURCE.resolve() or expected_sha != APPROVED_SHA:
        raise ValueError('formal source outside approved path/hash allowlist')
    if list(grid) != list(range(-20, 31, 2)) or classes != 24 or per_cell != 1024 or seed != 20260906:
        raise ValueError('formal subset construction contract differs')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def cell_ids(labels, snr, grid, classes):
    labels, snr, grid = np.asarray(labels), np.asarray(snr), np.asarray(grid)
    if labels.ndim != 1 or snr.shape != labels.shape or labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError('invalid class labels')
    if not np.array_equal(grid, np.unique(grid)) or not np.isfinite(snr).all() or not np.isin(snr, grid).all():
        raise ValueError('invalid or unsupported SNR')
    return labels.astype(np.int64)*len(grid)+np.searchsorted(grid, snr)


def cell_rng(seed, cell, grid_size, purpose):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, cell//grid_size, cell%grid_size, purpose])))


def choose_rows(labels, snr, grid, classes, per_cell, seed):
    cells = cell_ids(labels, snr, grid, classes)
    order = np.argsort(cells, kind='stable')
    counts = np.bincount(cells, minlength=classes*len(grid))
    if per_cell <= 0 or np.any(counts < per_cell):
        raise ValueError('insufficient source cell count')
    edges = np.r_[0, np.cumsum(counts)]
    selected = []
    for cell in range(len(counts)):
        source_rows = order[edges[cell]:edges[cell+1]]
        selected.append(cell_rng(seed, cell, len(grid), 0).permutation(source_rows)[:per_cell])
    return np.sort(np.concatenate(selected))


def deduplicate(hashes, cells):
    hashes, cells = np.asarray(hashes), np.asarray(cells)
    if hashes.dtype != np.dtype('V32') or hashes.shape != cells.shape or hashes.ndim != 1:
        raise ValueError('invalid canonical hash array')
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    low = np.full(len(first), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full(len(first), -1, dtype=np.int64)
    np.minimum.at(low, inverse, cells); np.maximum.at(high, inverse, cells)
    conflict = low != high
    # 0 retained, 1 same-cell duplicate, 2 conflicting class/SNR group.
    reasons = np.ones(len(cells), dtype=np.uint8)
    reasons[first[~conflict]] = 0
    reasons[conflict[inverse]] = 2
    return np.flatnonzero(reasons == 0), reasons


def split_rows(retained, cells, grid, classes, seed):
    retained, cells = np.asarray(retained), np.asarray(cells)
    result = {key: [] for key in ('train', 'validation', 'test')}
    for cell in range(classes*len(grid)):
        indices = retained[cells[retained] == cell]
        indices = cell_rng(seed, cell, len(grid), 1).permutation(indices)
        ntrain, nval = int(.6*len(indices)), int(.2*len(indices))
        if min(ntrain, nval, len(indices)-ntrain-nval) == 0:
            raise ValueError('empty retained cell partition')
        for name, values in zip(result, (indices[:ntrain], indices[ntrain:ntrain+nval], indices[ntrain+nval:])):
            result[name].append(values)
    return {name: np.sort(np.concatenate(values)) for name, values in result.items()}


def build_subset(source, output, *, grid, classes, per_cell, seed, expected_sha, source_dependencies, scientific=True):
    """Copy only bounded selected-row chunks to NCT mmap; never load all X."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if scientific:
        check_formal_contract(source, expected_sha, grid, classes, per_cell, seed)
    if set(source_dependencies) & {str(source), str(Path(__file__).resolve())}:
        raise ValueError('caller cannot override authoritative dependency hashes')
    if output.exists():
        raise FileExistsError(output)
    before = source.stat()
    if sha(source) != expected_sha:
        raise ValueError('source hash changed')
    output.mkdir(parents=True)
    config = dict(grid=list(grid), classes=classes, per_cell=per_cell, seed=seed,
                  rng='PCG64(SeedSequence([seed,class_index,snr_grid_index,purpose]))',
                  purpose={'subset': 0, 'split': 1}, source_row_order='ascending',
                  duplicate_representative='minimum original source row',
                  numpy_version=np.__version__, layout='NCT', dtype='float32', frame_samples=1024)
    dump(output/'config.json', config)
    with h5py.File(source, 'r') as handle:
        x, y, z = (handle[key] for key in ('X', 'Y', 'Z'))
        if x.shape[1:] != (1024, 2) or y.shape != (len(x), classes) or z.shape != (len(x), 1):
            raise ValueError('source dimensions changed')
        labels = np.empty(len(x), dtype=np.int16)
        snr = np.empty(len(x), dtype=np.int16)
        for start in range(0, len(x), 4096):
            values, db = y[start:start+4096], z[start:start+4096, 0]
            if not np.isin(values, [0, 1]).all() or not np.all(values.sum(1) == 1) or not np.isin(db, grid).all():
                raise ValueError('invalid one-hot or SNR metadata')
            labels[start:start+len(values)] = values.argmax(1)
            snr[start:start+len(values)] = db
        rows = choose_rows(labels, snr, grid, classes, per_cell, seed)
        selected_labels, selected_snr = labels[rows], snr[rows]
        cells = cell_ids(selected_labels, selected_snr, grid, classes)
        cache = np.lib.format.open_memmap(output/'signals.npy', mode='w+', dtype='<f4', shape=(len(rows), 2, 1024))
        hashes = np.empty(len(rows), dtype='V32')
        # Every source fancy-index read contains at most 128 rows (~1 MiB).
        for start in range(0, len(rows), 128):
            chunk = np.asarray(x[rows[start:start+128]], dtype='<f4').transpose(0, 2, 1).copy()
            if not np.isfinite(chunk).all():
                raise ValueError('nonfinite selected signal')
            cache[start:start+len(chunk)] = chunk
            chunk[chunk == 0] = 0.0  # Hash canonicalization only, not preprocessing.
            for offset, frame in enumerate(chunk):
                hashes[start+offset] = np.void(hashlib.sha256(frame.tobytes()).digest())
        cache.flush(); del cache
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('source changed while copying')
    retained, reasons = deduplicate(hashes, cells)
    parts = split_rows(retained, cells, grid, classes, seed)
    np.savez(output/'identity.npz', source_rows=rows, labels=selected_labels, snr_db=selected_snr,
             snr_bin=np.searchsorted(grid, selected_snr), canonical_sha256=hashes,
             retained_indices=retained, exclusion_reason=reasons, **{k+'_indices': v for k, v in parts.items()})
    summary = dict(status='DATA_PREPARED_NOT_TRAINED' if scientific else 'SMOKE_ONLY_NON_SCIENTIFIC',
                   source=str(source), source_sha256=expected_sha, source_rows_total=len(labels),
                   selected_rows=len(rows), retained_rows=len(retained), same_cell_duplicates=int(np.sum(reasons == 1)),
                   cross_cell_conflict_rows=int(np.sum(reasons == 2)), partition_counts={k: len(v) for k, v in parts.items()},
                   signal_shape=[len(rows), 2, 1024], preprocessing='layout/dtype only; cached amplitudes unchanged',
                   class_names='numeric indices only; mapping unverified')
    dump(output/'summary.json', summary)
    dependencies = {str(source): expected_sha, str(Path(__file__).resolve()): sha(__file__), **source_dependencies}
    dump(output/'manifest.json', dict(status=summary['status'], files={p.name: sha(p) for p in output.iterdir() if p.is_file()}, dependencies=dependencies))
    return summary


def validate_subset(output, source, repository_root, *, replay_source=True):
    """Verify closed files, allowlisted source, identities and actual cached rows."""
    output, source, root = Path(output).resolve(), Path(source).resolve(), Path(repository_root).resolve()
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    actual = {p.name for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest['files']) or any(p.is_dir() for p in output.iterdir()):
        raise ValueError('subset artifact closure differs')
    for name, expected in manifest['files'].items():
        path = (output/name).resolve()
        if path.parent != output or sha(path) != expected:
            raise ValueError('subset artifact hash differs')
    for name, expected in manifest['dependencies'].items():
        path = Path(name).resolve()
        if (path != source and not path.is_relative_to(root)) or sha(path) != expected:
            raise ValueError('subset dependency outside allowlist or hash differs')
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if manifest['status'] not in ('DATA_PREPARED_NOT_TRAINED', 'SMOKE_ONLY_NON_SCIENTIFIC'):
        raise ValueError('unrecognized data-preparation status')
    if manifest['status'] == 'DATA_PREPARED_NOT_TRAINED':
        check_formal_contract(source, summary['source_sha256'], config['grid'], config['classes'], config['per_cell'], config['seed'])
    if summary['status'] != manifest['status'] or summary['source'] != str(source) or manifest['dependencies'].get(str(source)) != summary['source_sha256']:
        raise ValueError('subset source/status binding differs')
    with np.load(output/'identity.npz', allow_pickle=False) as identity:
        b = {key: identity[key] for key in identity.files}
    rows = b['source_rows']
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or np.any(np.diff(rows) <= 0) or len(rows) != config['classes']*len(config['grid'])*config['per_cell']:
        raise ValueError('invalid selected source rows')
    cache = np.load(output/'signals.npy', mmap_mode='r', allow_pickle=False)
    if cache.shape != (len(rows), 2, 1024) or cache.dtype != np.dtype('<f4'):
        raise ValueError('cached signal layout or length differs')
    cells = cell_ids(b['labels'], b['snr_db'], config['grid'], config['classes'])
    if not np.array_equal(b['snr_bin'], np.searchsorted(config['grid'], b['snr_db'])):
        raise ValueError('dynamic SNR bins differ')
    retained, reasons = deduplicate(b['canonical_sha256'], cells)
    if not np.array_equal(retained, b['retained_indices']) or not np.array_equal(reasons, b['exclusion_reason']):
        raise ValueError('dedup decisions differ')
    if summary['selected_rows'] != len(rows) or summary['retained_rows'] != len(retained) or summary['same_cell_duplicates'] != int(np.sum(reasons == 1)) or summary['cross_cell_conflict_rows'] != int(np.sum(reasons == 2)):
        raise ValueError('dedup summary counts differ')
    parts = split_rows(retained, cells, config['grid'], config['classes'], config['seed'])
    for name, values in parts.items():
        if not np.array_equal(values, b[name+'_indices']) or summary['partition_counts'][name] != len(values):
            raise ValueError('split identities/counts differ')
    if replay_source:
        with h5py.File(source, 'r') as handle:
            labels = np.empty(len(handle['X']), dtype=np.int16)
            snr = np.empty(len(labels), dtype=np.int16)
            for start in range(0, len(labels), 4096):
                onehot = handle['Y'][start:start+4096]
                if not np.isin(onehot, [0, 1]).all() or not np.all(onehot.sum(1) == 1):
                    raise ValueError('source one-hot changed')
                labels[start:start+len(onehot)] = onehot.argmax(1)
                snr[start:start+len(onehot)] = handle['Z'][start:start+4096, 0]
            expected_rows = choose_rows(labels, snr, config['grid'], config['classes'], config['per_cell'], config['seed'])
            if not np.array_equal(rows, expected_rows) or not np.array_equal(b['labels'], labels[rows]) or not np.array_equal(b['snr_db'], snr[rows]):
                raise ValueError('selected source identity replay differs')
            for start in range(0, len(rows), 128):
                chunk = np.asarray(cache[start:start+128]).copy()
                source_chunk = np.asarray(handle['X'][rows[start:start+128]], dtype='<f4').transpose(0, 2, 1)
                if not np.array_equal(chunk, source_chunk) or not np.isfinite(chunk).all():
                    raise ValueError('cached waveform source replay differs')
                chunk[chunk == 0] = 0.0
                for offset, frame in enumerate(chunk):
                    if np.void(hashlib.sha256(frame.tobytes()).digest()) != b['canonical_sha256'][start+offset]:
                        raise ValueError('canonical waveform hash differs')
    return dict(status='VALIDATED_DATA_PREPARATION', rows=len(rows), retained=len(retained),
                partitions={key: len(value) for key, value in parts.items()}, source_replayed=replay_source,
                scientific_results=False)

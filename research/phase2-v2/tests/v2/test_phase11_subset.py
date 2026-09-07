import numpy as np
import pytest

from v2.phase11.subset import choose_rows, deduplicate, split_rows, build_subset, validate_subset, sha


def test_selection_is_fixed_per_cell_and_does_not_use_training_seed():
    labels = np.repeat(np.arange(2), 16)
    snr = np.tile(np.repeat([-2, 0], 8), 2)
    chosen = choose_rows(labels, snr, [-2, 0], 2, 4, 20260906)
    np.testing.assert_array_equal(chosen, choose_rows(labels, snr, [-2, 0], 2, 4, 20260906))
    assert len(chosen) == 16 and np.all(np.diff(chosen) > 0)
    for c in range(2):
        for db in (-2, 0):
            assert np.sum((labels[chosen] == c) & (snr[chosen] == db)) == 4
    with pytest.raises(ValueError, match='cell'):
        choose_rows(labels, snr, [-2, 0], 2, 9, 20260906)


def test_dedup_retains_minimum_source_row_and_drops_cross_cell_conflicts():
    hashes = np.array([b'a'*32, b'a'*32, b'b'*32, b'b'*32, b'c'*32], dtype='V32')
    # The canonical cache is already sorted by source row.
    retained, reasons = deduplicate(hashes, np.array([0, 0, 0, 1, 1]))
    np.testing.assert_array_equal(retained, [0, 4])
    np.testing.assert_array_equal(reasons, [0, 1, 2, 2, 0])


def test_split_uses_retained_rows_and_unequal_cell_counts_without_overlap():
    cells = np.repeat([0, 1], [11, 13])
    retained = np.delete(np.arange(24), [2, 12])
    parts = split_rows(retained, cells, [-2, 0], 1, 20260906)
    joined = np.concatenate(list(parts.values()))
    np.testing.assert_array_equal(np.sort(joined), retained)
    assert len(np.unique(joined)) == len(joined)
    assert [np.sum(cells[parts[p]] == 0) for p in ('train', 'validation', 'test')] == [6, 2, 2]
    assert [np.sum(cells[parts[p]] == 1) for p in ('train', 'validation', 'test')] == [7, 2, 3]


def test_extra_snr_cannot_silently_be_clipped_to_last_bin():
    with pytest.raises(ValueError, match='SNR'):
        choose_rows(np.zeros(3, dtype=int), np.array([-2, 0, 2]), [-2, 0], 1, 1, 1)


def test_full_hdf5_to_mmap_smoke_keeps_1024_samples_and_source_rows(tmp_path):
    import h5py
    source = tmp_path/'source.h5'
    values = np.random.default_rng(10).normal(size=(48, 1024, 2)).astype('float32')
    labels = np.repeat([0, 1], 24)
    snr = np.tile(np.repeat([-2, 20], 12), 2)
    selected = choose_rows(labels, snr, [-2, 20], 2, 10, 20260906)
    a, b = selected[:2]
    values[b] = values[a]
    values[a, 0, 0] = 0.0
    values[b, 0, 0] = -0.0
    with h5py.File(source, 'w') as handle:
        handle['X'] = values
        handle['Y'] = np.eye(2, dtype='int64')[labels]
        handle['Z'] = snr[:, None]
    output = tmp_path/'prepared'
    with pytest.raises(ValueError, match='source hash'):
        build_subset(source, output, grid=[-2, 20], classes=2, per_cell=10, seed=20260906,
                     expected_sha='0'*64, source_dependencies={}, scientific=False)
    assert not output.exists()
    build_subset(source, output, grid=[-2, 20], classes=2, per_cell=10, seed=20260906,
                 expected_sha=sha(source), source_dependencies={}, scientific=False)
    from pathlib import Path
    verified = validate_subset(output, source, Path(__file__).resolve().parents[2])
    assert verified['rows'] == 40 and verified['retained'] == 39 and verified['source_replayed'] and not verified['scientific_results']
    cached = np.load(output/'signals.npy', mmap_mode='r')
    assert cached.shape == (40, 2, 1024) and isinstance(cached, np.memmap)
    with (output/'signals.npy').open('r+b') as stream:
        stream.seek(-4, 2); stream.write(b'xxxx')
    with pytest.raises(ValueError, match='artifact hash'):
        validate_subset(output, source, Path(__file__).resolve().parents[2])


def test_all_26_bins_are_preserved_without_18_db_saturation():
    from v2.phase11.subset import cell_ids
    grid = np.arange(-20, 31, 2)
    labels = np.zeros(52, dtype=int)
    snr = np.repeat(grid, 2)
    selected = choose_rows(labels, snr, grid, 1, 1, 20260906)
    np.testing.assert_array_equal(cell_ids(labels[selected], snr[selected], grid, 1), np.arange(26))

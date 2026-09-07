import numpy as np
import pytest

from v2.phase10.core import floor_conditions, metrics_with_strata, evaluate_floors, paired_rows


def test_floor_after_range_clip_preserves_raw_errors_and_nearest_ties():
    raw = np.array([-100., -11., -9., 2., 40.])
    truth = np.array([-20., -10., -8., 2., 18.])
    result = floor_conditions(raw, truth, -10)
    np.testing.assert_array_equal(result['applied_condition_db'], [-10, -10, -9, 2, 18])
    np.testing.assert_array_equal(result['condition_bin'], [5, 5, 5, 11, 19])
    np.testing.assert_array_equal(result['raw_error_db'], raw-truth)
    np.testing.assert_array_equal(result['floor_active_mask'], [True, True, False, False, False])
    np.testing.assert_array_equal(result['range_clipped_mask'], [True, False, False, False, True])


def test_no_floor_matches_phase8_mapping_exactly():
    from v2.phase8.core import conditions
    raw = np.array([-25., -19., -9.001, -1., 20.])
    evidence = floor_conditions(raw, np.zeros(5), None)
    db, bins = conditions(raw)
    np.testing.assert_array_equal(evidence['applied_condition_db'], db)
    np.testing.assert_array_equal(evidence['condition_bin'], bins)
    assert not evidence['floor_active_mask'].any()


def test_empty_error_strata_are_null_not_zero_accuracy():
    y = np.arange(4); logits = np.eye(11)[y]
    result = metrics_with_strata(y, logits, np.array([-20., -8., -2., 10.]), np.zeros(4))
    assert result['error_strata']['4_to_8']['count'] == 0
    assert result['error_strata']['4_to_8']['accuracy'] is None
    assert result['error_strata']['0_to_2']['count'] == 4


@pytest.mark.parametrize('model_id', ['M3', 'M6'])
def test_actual_small_head_no_floor_and_above_floor_unchanged(model_id):
    import torch
    from models.model_conditioning import AWNConditioned
    from v2.phase8.core import head_logits
    torch.manual_seed(23)
    model = AWNConditioned(num_classes=11,num_levels=1,in_channels=64,kernel_size=3,latent_dim=320,regu_details=.01,regu_approx=.01,num_snr_bins=20,snr_embedding_dim=8,conditioning=model_id).eval()
    features = np.random.default_rng(1).normal(size=(16,128)).astype(np.float32)
    raw = np.linspace(-20,18,16)
    baseline = head_logits(model, features, raw, 8)
    results = evaluate_floors(model, features, raw, raw, [None,-10,-8,-6], baseline, batch_size=8)
    np.testing.assert_array_equal(results[0]['logits'], baseline)
    for row in results[1:]:
        unchanged = ~row['floor_active_mask']
        np.testing.assert_array_equal(row['logits'][unchanged], baseline[unchanged])


def test_floor_pairing_uses_five_matched_seeds_and_null_empty_cells():
    rows = []
    for seed in range(2022,2027):
        for floor in [None,-10,-8,-6]:
            value = .2 + ((seed-2021)*.01 if floor is not None else 0)
            rows.append({'model':'M3','seed':seed,'condition':'oracle','floor_db':floor,'metrics':{'overall':{'accuracy':value},'bands':{},'snr':{},'error_strata':{'empty':{'accuracy':None}}}})
    result = paired_rows(rows, ['M3'], ['oracle'])
    selected = next(r for r in result if r['floor_db']==-10 and r['group']=='overall' and r['metric']=='accuracy')
    assert selected['n_pairs']==5
    np.testing.assert_allclose(selected['differences'], [.01,.02,.03,.04,.05])
    empty = next(r for r in result if r['group']=='error_strata/empty')
    assert empty['n_pairs']==0 and empty['mean'] is None


def test_formal_gate_requires_real_phase8_and_phase9(tmp_path):
    from v2.phase10.runner import check_gates
    with pytest.raises(ValueError, match='upstream'):
        check_gates(tmp_path, {'phase8_current':'missing8.json','phase9_current':'missing9.json'})


def test_closure_accepts_exact_real_root_registry_as_hash_bound_leaf(tmp_path):
    from pathlib import Path
    from v2.phase10.runner import manifest_closure
    from v2.phase8.evidence import dump,sha
    registry=Path(__file__).resolve().parents[2]/'manifest.json'
    (tmp_path/'manifest.json').write_bytes(registry.read_bytes())
    attempt=tmp_path/'attempt';attempt.mkdir();(attempt/'data.bin').write_bytes(b'actual artifact bytes')
    dump(attempt/'manifest.json',dict(files={'data.bin':sha(attempt/'data.bin')},dependencies={'manifest.json':sha(tmp_path/'manifest.json')}))
    assert 'manifest.json' in manifest_closure(tmp_path,attempt/'manifest.json')
    (tmp_path/'manifest.json').write_bytes(registry.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='hash mismatch'):manifest_closure(tmp_path,attempt/'manifest.json')


def test_closure_does_not_treat_other_malformed_manifest_as_registry(tmp_path):
    from v2.phase10.runner import manifest_closure
    from v2.phase8.evidence import dump,sha
    other=tmp_path/'other';other.mkdir();dump(other/'manifest.json',dict(schema_version=1,experiments=[]))
    attempt=tmp_path/'attempt';attempt.mkdir();(attempt/'data').write_bytes(b'data')
    dump(attempt/'manifest.json',dict(files={'data':sha(attempt/'data')},dependencies={'other/manifest.json':sha(other/'manifest.json')}))
    with pytest.raises(ValueError,match='lacks hash closure'):manifest_closure(tmp_path,attempt/'manifest.json')


def test_complete_publication_failure_preserves_closure_and_can_retry(tmp_path,monkeypatch):
    from pathlib import Path
    from v2.phase10.runner import _publish_current,_record_failure,manifest_closure
    from v2.phase8.evidence import dump,sha,load
    output=tmp_path/'output';attempt=output/'attempts'/'publication_unit';attempt.mkdir(parents=True)
    (attempt/'artifact').write_bytes(b'publication transport test only')
    dump(attempt/'manifest.json',dict(status='COMPLETE',files={'artifact':sha(attempt/'artifact')},dependencies={}))
    before=manifest_closure(tmp_path,attempt/'manifest.json');replace=Path.replace
    def fail(self,target):raise OSError('simulated publication failure')
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError,match='simulated'):_publish_current(attempt,output,tmp_path,{'status':'UNIT_TEST_ONLY'})
    _record_failure(attempt,output,'simulated publication failure')
    assert before==manifest_closure(tmp_path,attempt/'manifest.json')
    assert list((output/'publication_failures').glob('*.json'))
    monkeypatch.setattr(Path,'replace',replace)
    _publish_current(attempt,output,tmp_path,{'status':'UNIT_TEST_ONLY'})
    assert load(output/'current.json')['manifest_sha256']==sha(attempt/'manifest.json')

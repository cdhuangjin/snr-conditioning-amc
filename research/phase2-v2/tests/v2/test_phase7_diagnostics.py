import json
import numpy as np
import pytest
from v2.phase7.diagnostics import signal_statistics, prediction_summary, feature_summary, analyze_bundles, verify_manifest


def test_tone_and_white_noise_statistics():
    n = 128
    z = np.exp(2j*np.pi*8*np.arange(n)/n)
    x = np.stack([z.real, z.imag])[None]
    s = signal_statistics(x)
    assert s['energy'][0] == pytest.approx(n)
    assert s['power'][0] == pytest.approx(1)
    assert s['spectral_flatness'][0] < 1e-8
    assert s['bandwidth_90'][0] == pytest.approx(1/n)
    assert abs(s['autocorrelation'][0, 1]) == pytest.approx(1)
    noise = np.random.default_rng(7).normal(size=(100,2,n))
    assert np.mean(signal_statistics(noise)['spectral_flatness']) > .4


def test_prediction_denominators_and_undefined_values():
    r = prediction_summary(np.array([0,0,1,1]), np.array([1,0,0,1]), 2, 1)
    assert r['error_to_wbfm_share'] == .5
    assert r['wbfm_recall'] == .5
    assert r['true_frequency'] == [.5,.5]
    r = prediction_summary(np.array([0]), np.array([0]), 2, 1)
    assert r['error_to_wbfm_share'] is None
    assert r['wbfm_recall'] is None


def test_feature_separation_is_real_and_translation_invariant():
    f = np.array([[0.,0.],[0.,1.],[10.,0.],[10.,1.]])
    y = np.array([0,0,1,1])
    a = feature_summary(f,y,2,1)
    b = feature_summary(f+100,y,2,1)
    assert a['between_within_ratio'] == pytest.approx(100)
    assert a['wbfm_centroid_distances'] == b['wbfm_centroid_distances']


def fixture(tmp_path):
    ids=np.array(['a','b','c','d'])
    y=np.array([0,0,1,1]); snr=np.array([-20,-20,-20,-20])
    raw=tmp_path/'raw.npz'
    np.savez(raw, signals=np.random.default_rng(2).normal(size=(4,2,16)), sample_ids=ids, y_true=y, snr_db=snr)
    runs=[]
    for seed in range(2022,2027):
        p=tmp_path/f'p{seed}.npz'; f=tmp_path/f'f{seed}.npz'
        np.savez(p,sample_ids=ids,y_true=y,snr_db=snr,y_pred=np.array([1,0,1,1]),seed=seed,model_id='M0',split_hash='fixed')
        np.savez(f,sample_ids=ids,y_true=y,snr_db=snr,features=np.arange(12).reshape(4,3))
        runs.append(dict(seed=seed,model_id='M0',predictions=str(p),features=str(f)))
    return raw,runs


def test_bundle_smoke_hash_closure_and_recomputation(tmp_path):
    raw,runs=fixture(tmp_path)
    out=tmp_path/'out'
    analyze_bundles(raw,runs,out,class_names=['A','WBFM'],split_hash='fixed')
    m=verify_manifest(out)
    assert m['status']=='DIAGNOSTICS_COMPLETE_CONTROLS_PENDING'
    rows=json.loads((out/'summary.json').read_text())['predictions']
    assert len(rows)==5
    assert rows[0]['error_to_wbfm_share']==1
    with np.load(out/'raw_statistics.npz') as a:
        assert len(a['sample_ids'])==4
    (out/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        verify_manifest(out)


def test_reject_misaligned_features_missing_seed_and_reuse(tmp_path):
    raw,runs=fixture(tmp_path)
    with pytest.raises(ValueError,match='five'):
        analyze_bundles(raw,runs[:-1],tmp_path/'missing',class_names=['A','WBFM'],split_hash='fixed')
    p=runs[0]['features']
    with np.load(p) as a: d=dict(a)
    d['sample_ids']=d['sample_ids'][::-1]; np.savez(p,**d)
    with pytest.raises(ValueError,match='sample_ids'):
        analyze_bundles(raw,runs,tmp_path/'bad',class_names=['A','WBFM'],split_hash='fixed')


def test_zeros_are_finite_and_bad_signals_rejected():
    s=signal_statistics(np.zeros((2,2,16)))
    assert all(np.isfinite(v).all() for v in s.values())
    with pytest.raises(ValueError,match='finite'):
        signal_statistics(np.full((1,2,16),np.nan))


def test_reject_wrong_label_range_and_split(tmp_path):
    raw,runs=fixture(tmp_path)
    with pytest.raises(ValueError,match='provenance'):
        analyze_bundles(raw,runs,tmp_path/'wrong',class_names=['A','WBFM'],split_hash='other')
    with pytest.raises(ValueError,match='class labels'):
        prediction_summary(np.array([-1]),np.array([0]),2,1)


def test_cli_gate_rejects_stop_rule(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'scripts/v2/analyze_phase7.py'
    spec=importlib.util.spec_from_file_location('phase7_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps({'status':'COMPLETE','stop_rule_triggered':True}))
    with pytest.raises(ValueError,match='Phase 6'):
        module.validate_gate(gate)


def test_gate_manifest_requires_matching_hash(tmp_path):
    import importlib.util
    from pathlib import Path
    from v2.phase7.diagnostics import sha256
    path=Path(__file__).resolve().parents[2]/'scripts/v2/analyze_phase7.py'
    spec=importlib.util.spec_from_file_location('phase7_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'status':'COMPLETE'}))
    gate=tmp_path/'gate.json'
    data=dict(status='COMPLETE',stop_rule_triggered=False,manifest_path='manifest.json',manifest_sha256=sha256(manifest))
    gate.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='composite'):
        module.validate_gate(gate,tmp_path)
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='composite'):
        module.validate_gate(gate,tmp_path)


def test_immutable_output_and_source_tampering(tmp_path):
    raw,runs=fixture(tmp_path);out=tmp_path/'out'
    analyze_bundles(raw,runs,out,class_names=['A','WBFM'],split_hash='fixed')
    with pytest.raises(ValueError,match='immutable'):
        analyze_bundles(raw,runs,out,class_names=['A','WBFM'],split_hash='fixed')
    raw.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='source hash'):
        verify_manifest(out)

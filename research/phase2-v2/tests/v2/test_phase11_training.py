import numpy as np
import pytest
from v2.phase11.data import dynamic_conditions,LazyFrames,bounded_features
from v2.phase11.estimator import fit_ridge,predict_ridge


def test_dynamic_grid_preserves_2018_high_snr_and_lower_midpoint():
    db,bins=dynamic_conditions(np.array([-30.,-19.,18.,29.,40.]),np.arange(-20,32,2))
    np.testing.assert_array_equal(db,[-20,-19,18,29,30]);np.testing.assert_array_equal(bins,[0,0,19,24,25])


def test_memmap_loading_preserves_length_and_bounded_feature_calls(tmp_path):
    path=tmp_path/'signals.npy';x=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(18,2,1024));x[:]=1;x.flush();del x
    signals=np.load(path,mmap_mode='r');dataset=LazyFrames(signals,np.arange(18)%3,np.arange(18)%3*2-20,np.array([4,1]),np.arange(-20,32,2))
    assert isinstance(dataset.signals,np.memmap) and dataset[0][0].shape==(2,1024)
    calls=[]
    def feature(chunk):calls.append(len(chunk));return np.zeros((len(chunk),7))
    assert bounded_features(signals,chunk_size=4,feature_fn=feature).shape==(18,7)
    assert max(calls)<=4 and sum(calls)==18


def test_ridge_fit_and_scaler_exclude_validation_and_test():
    rng=np.random.default_rng(5);features=rng.normal(size=(30,7));truth=features[:,0]+.1*rng.normal(size=30);ids=np.array([f'row{i}' for i in range(30)])
    train=np.arange(18);val=np.arange(18,24)
    state,selection=fit_ridge(features,truth,ids,train,val,[.1])
    changed=features.copy();changed[18:]+=1000
    other,_=fit_ridge(changed,truth,ids,train,val,[.1])
    assert state==other and state['fit_count']==18
    assert not np.array_equal(predict_ridge(features,state)[18:],predict_ridge(changed,other)[18:])
    with pytest.raises(ValueError):fit_ridge(features,truth,ids,train,train[:2],[.1])


def test_missing_phase10_gate_rejects_before_training(tmp_path):
    from v2.phase11.runner import check_gates
    with pytest.raises(ValueError,match='Phase10'):check_gates(tmp_path,{'phase10_current':'missing.json'})


def test_unequal_cells_weighted_and_equal_cell_metrics_differ():
    from v2.phase11.analysis import metrics
    y=np.array([0,0,0,1]);snr=np.zeros(4);logits=np.array([[5.,0.]]*4)
    result=metrics(y,logits,snr,2)['overall']
    assert result['sample_weighted_accuracy']==.75 and result['equal_cell_accuracy']==.5
    assert result['balanced_accuracy']==.5


def test_model_factory_uses_dynamic_bounds_and_real_adapter():
    import yaml
    from pathlib import Path
    from v2.phase11.runner import factory
    c=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v2/phase11.yaml').read_text())
    model=factory(c,{'classes':24,'grid':list(range(-20,32,2))},'M6')
    assert model.num_classes==24 and model.num_snr_bins==26 and model.snr_max_db==30
    assert 'reflection_pad1d_slice_flip_cat_v1' in str(__import__('v2.phase7.deterministic_padding',fromlist=['ADAPTER_ID']).ADAPTER_ID)


def test_external_dependency_allowlist_and_hash_fail_closed(tmp_path):
    from v2.phase11.runner import verify_hashes
    from v2.phase8.evidence import dump,sha
    root=tmp_path/'repo'/'worktrees'/'phase';root.mkdir(parents=True);attempt=root/'attempt';attempt.mkdir();(attempt/'data').write_bytes(b'output')
    outside=tmp_path/'not_allowed';outside.write_bytes(b'source')
    dump(attempt/'manifest.json',dict(files={'data':sha(attempt/'data')},dependencies={str(outside):sha(outside)}))
    with pytest.raises(ValueError,match='unallowlisted'):verify_hashes(root,attempt/'manifest.json')
    dump(attempt/'manifest.json',dict(files={'data':sha(attempt/'data')},dependencies={str(attempt/'data'):'0'*64}))
    with pytest.raises(ValueError,match='changed'):verify_hashes(root,attempt/'manifest.json')

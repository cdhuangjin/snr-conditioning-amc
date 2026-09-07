import numpy as np
import pytest
import torch

from v2.phase8.core import donor_indices, conditions, grouped_metrics, fit_e4, e4_logits, cache_and_replay
from v2.phase8.evidence import check_gates


def test_shuffles_are_reproducible_bijections_with_declared_strata():
    snr=np.repeat([-20.,0.],12); labels=np.tile(np.repeat(np.arange(3),4),2)
    for mode in ('global','within_snr','across_class'):
        idx=donor_indices(snr,labels,mode,2022)
        np.testing.assert_array_equal(np.sort(idx),np.arange(24))
        np.testing.assert_array_equal(idx,donor_indices(snr,labels,mode,2022))
        if mode!='global': np.testing.assert_array_equal(snr[idx],snr)
        if mode=='across_class': assert np.all(labels[idx]!=labels)
    with pytest.raises(ValueError,match='majority'):
        donor_indices(np.zeros(5),np.array([0,0,0,0,1]),'across_class',2022)


def test_deployed_clipping_and_midpoint_ties_choose_lower_bin():
    db,bins=conditions(np.array([-100.,-19.,-18.999,19.]))
    np.testing.assert_allclose(db,[-20,-19,-18.999,18])
    np.testing.assert_array_equal(bins,[0,0,1,19])


def test_metrics_handle_class_conditioned_support_without_silent_zero_recall():
    y=np.array([1,1]); logits=np.array([[0.,9.,0.],[9.,0.,0.]])
    result=grouped_metrics(y,logits,np.array([-20.,0.]),classes=3)
    assert result['overall']['balanced_accuracy']==.5
    assert result['overall']['recall']==[None,.5,None]
    assert result['overall']['confusion']==[[0,0,0],[1,1,0],[0,0,0]]


@pytest.mark.parametrize('model_id',['M3','M6'])
def test_raw_cache_replays_full_forward_and_preserves_unconditioned_dimension(model_id):
    from models.model_conditioning import AWNConditioned
    torch.manual_seed(21)
    model=AWNConditioned(num_classes=3,num_levels=1,in_channels=8,latent_dim=12,conditioning=model_id).eval()
    x=np.random.default_rng(2).normal(size=(6,2,32)).astype(np.float32)
    snr=np.array([-20,-18,-16,-14,-12,-10],dtype=np.float32)
    with torch.no_grad(): reference=model.forward_batch(torch.from_numpy(x),snr_db=torch.from_numpy(snr),snr_bin=torch.arange(6))[0].numpy()
    cache,score,evidence=cache_and_replay(model,x,snr,reference,batch_size=2)
    assert cache.shape==(6,model.out_channels)
    np.testing.assert_allclose(score,reference,rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError,match='registered'):
        cache_and_replay(model,x,snr,reference+1,batch_size=2)


def test_e4_uses_train_statistics_and_is_one_shared_fit():
    x=np.linspace(-3,3,90); y=(x>0).astype(int)
    options=dict(representations=['scalar'],cs=[.01,1.],max_iter=2000)
    model,evidence=fit_e4(x[:60],y[:60],x[60:],y[60:],options)
    assert evidence['fit_partition']=='train' and evidence['n_independent_fits']==1
    np.testing.assert_allclose(model['mean'],[x[:60].mean()])
    assert e4_logits(x[:3],model).shape==(3,2)


def test_missing_dependency_gate_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='gate'):
        check_gates(tmp_path,{'phase6_current':'missing.json','phase7_gate':'missing_gate.json'})


def test_checkpoint_loader_uses_real_phase3_contract_and_strict_state(tmp_path):
    from pathlib import Path
    from v2.phase3_analysis import _load_phase3_contracts
    from v2.phase8.runner import get_model
    phase3=_load_phase3_contracts()
    config=phase3.load_phase3_config(Path(__file__).resolve().parents[2]/'configs/v2/phase3.yaml')
    original=phase3._model(config,'M3')
    path=tmp_path/'checkpoint.pt';torch.save(original.state_dict(),path)
    loaded=get_model(phase3,config,{'paths':{'checkpoint':path}},'M3','cpu')
    for name,value in original.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[name])


def test_gate_manifest_checks_complete_closure_and_source_hashes(tmp_path):
    from v2.phase8.evidence import dump,sha,verify_manifest
    output=tmp_path/'attempt';output.mkdir()
    dump(output/'summary.json',{'status':'COMPLETE'})
    dump(output/'manifest.json',{'status':'COMPLETE','files':{'summary.json':sha(output/'summary.json')},'dependencies':{}})
    verify_manifest(tmp_path,output/'manifest.json',sha(output/'manifest.json'))
    (output/'unregistered.txt').write_text('pollution')
    with pytest.raises(ValueError,match='polluted'):
        verify_manifest(tmp_path,output/'manifest.json',sha(output/'manifest.json'))


@pytest.mark.parametrize('change',[{'batch_size':128},{'torch_num_threads':1},{'device':'cuda'}])
def test_registered_execution_config_rejects_batch_thread_or_device_drift(change):
    from pathlib import Path
    import yaml
    from v2.phase8.runner import validate_config
    config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v2/phase8.yaml').read_text(encoding='utf-8'))
    validate_config(config)
    with pytest.raises(ValueError,match='CPU batch64'):
        validate_config(config|change)


def test_plain_complete_phase6_pointer_cannot_relabel_original_failure(tmp_path):
    from v2.phase8.evidence import dump
    dump(tmp_path/'phase6_current.json',{'status':'COMPLETE','attempt':'original_blocked_attempt','manifest_sha256':'0'*64})
    with pytest.raises(ValueError,match='bounded scientific composite'):
        check_gates(tmp_path,{'phase6_current':'phase6_current.json','phase7_gate':'unused.json'})


def _negative_composite_graph(tmp_path,decision):
    from v2.phase8.evidence import dump,sha
    from v2.phase6.composite import BASE,KIND
    source=tmp_path/'frozen_source.py';source.write_text('frozen implementation')
    original=tmp_path/'original';original.mkdir()
    dump(original/'summary.json',{'status':'BLOCKED_STREAMING_REPLAY'})
    dump(original/'manifest.json',{'files':{'summary.json':sha(original/'summary.json')},'dependencies':{'frozen_source.py':sha(source)}})
    receipt=tmp_path/BASE/'receipts'/'negative_unit';receipt.mkdir(parents=True)
    dump(receipt/'decision.json',decision)
    dump(receipt/'manifest.json',{'status':'COMPLETE','kind':KIND,'files':{'decision.json':sha(receipt/'decision.json')},'dependencies':{'original/manifest.json':sha(original/'manifest.json')}})
    current={'schema_version':1,'kind':KIND,'status':'COMPLETE','attempt':receipt.relative_to(tmp_path).as_posix(),'data_attempt':'original','manifest_sha256':sha(receipt/'manifest.json')}
    current['gate']={**current,'manifest_path':current['attempt']+'/manifest.json','stop_rule_triggered':False}
    dump(tmp_path/'phase6_current.json',current)
    # Negative-only graph: it cannot pass the full scientific evidence checks.
    return source,{'phase6_current':'phase6_current.json','phase7_gate':'unused.json'}


def test_phase8_dependency_contract_preserves_failed_fp32_decision(tmp_path):
    from v2.phase6.composite import decision_template
    decision=decision_template();decision['original_fp32_tight_replay']='PASS'
    _source,config=_negative_composite_graph(tmp_path,decision)
    with pytest.raises(ValueError,match='preserve failed FP32'):
        check_gates(tmp_path,config)


def test_phase8_dependency_contract_rejects_nested_source_tampering(tmp_path):
    from v2.phase6.composite import decision_template
    source,config=_negative_composite_graph(tmp_path,decision_template())
    source.write_text('tampered implementation')
    with pytest.raises(ValueError,match='hash'):
        check_gates(tmp_path,config)

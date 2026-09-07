from pathlib import Path
import numpy as np
import pytest
from v2.phase12.assembly import assemble, validate_dataset, load_dataset
from v2.phase12.execution import build_model, check_gates, runtime_config, evaluation_conditions


def test_assembly_pairing_replay_and_tamper(tmp_path):
    out=tmp_path/'data'
    assemble(out, counts={'train':2,'validation':1,'test':1}, snr_indices=[0,10,19], scientific=False)
    result=validate_dataset(out,replay=True)
    assert result['observations']==90 and result['latent_frames']==72
    data=load_dataset(out)
    ids=data['identity']
    assert len(np.unique(ids['sample_ids']))==90
    assert len(ids['train_indices'])==36 and len(ids['validation_indices'])==18
    a,b=ids['test_A_indices'],ids['test_B_indices']
    np.testing.assert_array_equal(ids['waveform_ids'][a],ids['waveform_ids'][b])
    assert not set(ids['waveform_ids'][a]) & set(ids['waveform_ids'][ids['train_indices']])
    with (out/'pilot.npy').open('r+b') as f:
        f.seek(-1,2);byte=f.read(1);f.seek(-1,2);f.write(bytes([byte[0]^1]))
    with pytest.raises(ValueError,match='hash'):
        validate_dataset(out)


def test_missing_gate_and_formal_assembly_fail_closed(tmp_path):
    cfg=runtime_config()
    with pytest.raises(ValueError):check_gates(tmp_path,cfg)
    with pytest.raises(ValueError,match='gate'):
        assemble(tmp_path/'formal',scientific=True)
    assert not (tmp_path/'formal').exists()


def test_cldnn_ignores_conditions_and_pilot_k0(tmp_path):
    import torch
    cfg=runtime_config()
    model=build_model('CLDNN',cfg).eval()
    x=torch.randn(3,2,128)
    with torch.no_grad():
        a,_=model.forward_batch(x,snr_db=torch.zeros(3),snr_bin=torch.zeros(3,dtype=torch.long))
        b,_=model.forward_batch(x,snr_db=torch.ones(3)*18,snr_bin=torch.ones(3,dtype=torch.long)*19)
    assert torch.equal(a,b) and a.shape==(3,6)
    out=tmp_path/'data';assemble(out,counts={'train':1,'validation':1,'test':1},snr_indices=[10],scientific=False)
    data=load_dataset(out);n=len(data['identity']['labels'])
    choices=evaluation_conditions(data,np.full(n,-3.),np.full(n,7.))
    assert 'K0_frame_estimator' in choices and 'K0_pilot' not in choices
    assert set(choices)=={'oracle','K0_frame_estimator','source10a_frozen','K8_pilot','K16_pilot','K32_pilot','K64_pilot'}
    assert np.array_equal(choices['K0_frame_estimator']['raw_db'],np.full(n,-3.))

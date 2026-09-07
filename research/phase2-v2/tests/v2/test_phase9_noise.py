from pathlib import Path
import numpy as np
import pytest
import torch

from v2.phase9.noise import make_folds,crossfit,ConditionNoise
from v2.phase9.trainer import train_model
from v2.phase9.runner import validate_config,phase8_gate


def tiny_data():
    rng=np.random.default_rng(2);snr=np.tile(np.repeat([-4.,0.,4.],10),2);labels=np.repeat([0,1],30)
    x=rng.normal(size=(60,3));x[:,0]+=snr
    return x,snr,labels,np.array([f'train:{i}' for i in range(60)])


def test_crossfit_excludes_own_rows_from_scaler_and_ridge_fit():
    x,y,labels,ids=tiny_data();folds=make_folds(y,labels,5,2022)
    original=crossfit(x,y,ids,folds,.1)
    changed=x.copy();changed[folds==0]+=1000
    other=crossfit(changed,y,ids,folds,.1)
    np.testing.assert_array_equal(original['coefficients'][0],other['coefficients'][0])
    np.testing.assert_array_equal(original['means'][0],other['means'][0])
    assert not np.allclose(original['oof_estimates'][folds==0],other['oof_estimates'][folds==0])
    for fold in range(5):
        assert set(original[f'fit_ids_{fold}']).isdisjoint(original[f'heldout_ids_{fold}'])
    np.testing.assert_array_equal(folds,make_folds(y,labels,5,2022))


def test_empirical_pool_uses_only_snr_and_restores_its_own_rng():
    truth=np.repeat([-4.,4.],5);errors=np.array([-2,-1,0,1,2,10,11,12,13,14.])
    noise=ConditionNoise('empirical_snr',2022,pool_snr=truth,pool_errors=errors)
    global_before=np.random.get_state();state=noise.state()
    raw,donors=noise.sample(np.array([-4.,4.]*20))
    assert np.all(truth[donors]==np.array([-4.,4.]*20))
    np.testing.assert_array_equal(raw-np.array([-4.,4.]*20),errors[donors])
    noise.restore(state);again,donors2=noise.sample(np.array([-4.,4.]*20))
    np.testing.assert_array_equal(raw,again);np.testing.assert_array_equal(donors,donors2)
    np.testing.assert_array_equal(global_before[1],np.random.get_state()[1])


class TinyConditioned(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(4,3);self.dropout=torch.nn.Dropout(.2)
    def forward_batch(self,x,*,snr_db,snr_bin):
        return self.fc(self.dropout(torch.cat([x,snr_db[:,None]/20],1))),[]


def loaders():
    generator=torch.Generator().manual_seed(8)
    x=torch.randn(24,3,generator=generator);y=torch.arange(24)%3;snr=(torch.arange(24)%3*4-4).float();ids=torch.arange(24)
    train=torch.utils.data.TensorDataset(x[:18],y[:18],snr[:18],ids[:18])
    val=torch.utils.data.TensorDataset(x[18:],y[18:],snr[18:]+.5,ids[18:])
    return torch.utils.data.DataLoader(train,batch_size=6,shuffle=True,num_workers=0),torch.utils.data.DataLoader(val,batch_size=6,num_workers=0)


def test_epoch_resume_is_exact_for_model_optimizer_and_condition_rng(tmp_path):
    training=dict(lr=.001,max_epochs=3)
    loss=lambda logits,y,reg:torch.nn.functional.cross_entropy(logits,y)
    def run(folder,logger=None):
        return train_model(TinyConditioned,loaders,folder,seed=2022,training=training,identity='test',loss_fn=loss,condition_noise=ConditionNoise('generic',2022),threads=1,logger=logger,device='cpu')
    run(tmp_path/'whole')
    def interrupt(row):
        if row['epoch']==0:raise RuntimeError('simulated interruption after atomic snapshot')
    with pytest.raises(RuntimeError,match='simulated'):run(tmp_path/'resumed',interrupt)
    run(tmp_path/'resumed')
    a=torch.load(tmp_path/'whole'/'resume.pt',weights_only=True);b=torch.load(tmp_path/'resumed'/'resume.pt',weights_only=True)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['condition_rng']==b['condition_rng']
    torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for name,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
    for x,y in zip(a['history'],b['history']):
        assert {k:v for k,v in x.items() if k!='seconds'}=={k:v for k,v in y.items() if k!='seconds'}


def test_formal_config_requires_fifteen_runs_and_phase8_gate(tmp_path):
    import yaml
    config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v2/phase9.yaml').read_text())
    validate_config(config)
    with pytest.raises(ValueError):validate_config(config|{'arms':['generic','empirical_snr']})
    with pytest.raises(ValueError,match='Phase 8'):phase8_gate(tmp_path,config)


def test_cliff_matrix_contains_both_origins_and_each_boundary_side():
    import yaml
    from v2.phase9.analysis import cliff_specs
    config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v2/phase9.yaml').read_text())
    rows=cliff_specs(config)
    assert len(rows)==545 and len({r['id'] for r in rows})==545
    boundaries=[r for r in rows if r['kind']=='boundary']
    assert len(boundaries)==285 and {r['offset_db'] for r in boundaries}=={-.001,0.,.001}
    assert {r['origin'] for r in rows if r['kind']=='additive'}=={'oracle','estimated'}


def test_clean_and_noisy_arms_match_initialization_and_data_order(tmp_path):
    import json
    outputs=[]
    for arm in ['clean','generic']:
        outputs.append(train_model(TinyConditioned,loaders,tmp_path/arm,seed=2022,training={'lr':.001,'max_epochs':2},identity=arm,loss_fn=lambda l,y,r:torch.nn.functional.cross_entropy(l,y),condition_noise=ConditionNoise(arm,2022),threads=1))
    assert outputs[0]['initial_model_sha256']==outputs[1]['initial_model_sha256']
    histories=[json.loads((tmp_path/arm/'epochs.json').read_text()) for arm in ['clean','generic']]
    assert [r['data_order_sha256'] for r in histories[0]]==[r['data_order_sha256'] for r in histories[1]]

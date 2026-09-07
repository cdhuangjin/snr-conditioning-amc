from copy import deepcopy
import torch
import pytest
from v2.phase12.execution import runtime_config
from v2.phase12.assembly import assemble,load_dataset
from v2.phase12.runner import train_one


@pytest.mark.parametrize('model',['M0','M6','CLDNN'])
def test_epoch_resume_matches_uninterrupted(tmp_path,model):
    c=deepcopy(runtime_config());c.update(device='cpu',torch_num_threads=1)
    c['training'].update(max_epochs=2,batch_size=12)
    assemble(tmp_path/'data',counts={'train':1,'validation':1,'test':1},snr_indices=[0,10],scientific=False)
    data=load_dataset(tmp_path/'data')
    train_one(tmp_path/'continuous',data,model,2022,c,'fixed-smoke-identity')
    def interrupt(row):raise RuntimeError('intentional post-snapshot interruption')
    with pytest.raises(RuntimeError,match='intentional'):
        train_one(tmp_path/'resumed',data,model,2022,c,'fixed-smoke-identity',interrupt)
    train_one(tmp_path/'resumed',data,model,2022,c,'fixed-smoke-identity')
    a=torch.load(tmp_path/'continuous/resume.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/resume.pt',weights_only=True)
    for key in ('model','best_state'):
        assert a[key].keys()==b[key].keys()
        assert all(torch.equal(a[key][name],b[key][name]) for name in a[key])
    assert a['best_epoch']==b['best_epoch'] and a['best']==b['best']
    for key in a['optimizer']['state']:
        assert all(torch.equal(a['optimizer']['state'][key][name],b['optimizer']['state'][key][name]) for name in a['optimizer']['state'][key])

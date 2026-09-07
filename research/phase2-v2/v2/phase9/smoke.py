"""Small real AWN training/resume exercise; never a formal scientific result."""
import json
from pathlib import Path
import numpy as np
import torch
from models.model_conditioning import AWNConditioned
from v2.phase7.deterministic_padding import install_adapter
from v2.phase8.evidence import dump,sha
from .noise import ConditionNoise
from .trainer import train_model


def run(destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('smoke destination must be new')
    def model():
        result=AWNConditioned(num_classes=3,num_levels=1,in_channels=8,latent_dim=12,conditioning='M6')
        assert install_adapter(result)
        return result
    def loaders():
        x=torch.randn(24,2,32,generator=torch.Generator().manual_seed(17))
        y=torch.arange(24)%3;snr=(torch.arange(24)%3*4-4).float();ids=torch.arange(24)
        return tuple(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[s],y[s],snr[s]+shift,ids[s]),batch_size=6,shuffle=True,num_workers=0) for s,shift in [(slice(0,18),0),(slice(18,24),.5)])
    def execute(arm,folder,logger=None):
        return train_model(model,loaders,destination/folder,seed=2022,training=dict(lr=.001,max_epochs=2),identity='real_awn_smoke_not_formal',loss_fn=lambda l,y,r:torch.nn.functional.cross_entropy(l,y)+sum(r),condition_noise=ConditionNoise(arm,2022,pool_snr=np.repeat([-4.,0.,4.],3),pool_errors=np.tile([-1.,0.,1.],3)),threads=1,logger=logger)
    outcomes={arm:execute(arm,arm) for arm in ['clean','generic','empirical_snr']}
    def interrupt(row):
        if row['epoch']==0:raise RuntimeError('intentional interruption')
    try:execute('empirical_snr','resumed',interrupt)
    except RuntimeError as error:
        if str(error)!='intentional interruption':raise
    execute('empirical_snr','resumed')
    a=torch.load(destination/'empirical_snr/resume.pt',weights_only=True)
    b=torch.load(destination/'resumed/resume.pt',weights_only=True)
    def equal(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    for key in ['model','optimizer','best_state','condition_rng','torch_rng','python_rng','numpy_rng']:equal(a[key],b[key])
    for x,y in zip(a['history'],b['history']):equal({k:v for k,v in x.items() if k!='seconds'},{k:v for k,v in y.items() if k!='seconds'})
    assert len({o['initial_model_sha256'] for o in outcomes.values()})==1
    orders={tuple(r['data_order_sha256'] for r in json.loads((destination/arm/'epochs.json').read_text())) for arm in outcomes}
    assert len(orders)==1
    verdict=dict(status='SMOKE_NOT_FORMAL',synthetic_inputs=True,model='real AWNConditioned M6 reduced dimensions with frozen deterministic adapter',arms=list(outcomes),epochs=2,seed=2022,exact_resume=True,matched_initialization_and_order=True)
    dump(destination/'verdict.json',verdict)
    dump(destination/'manifest.json',dict(status='SMOKE_NOT_FORMAL',files={p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}))
    return verdict


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination')
    print(json.dumps(run(parser.parse_args().destination),indent=2))

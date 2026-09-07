"""Tiny real-data CPU smoke; never publishes a scientific phase current."""
from pathlib import Path
import numpy as np
import torch
import yaml
from v2.phase7.controls import train_model
from v2.phase8.evidence import dump,sha
from .runner import factory,loss_fn,predict
from .data import load_dataset,LazyFrames,bounded_features
from .estimator import fit_ridge,predict_ridge


def run(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('new smoke directory required')
    destination.mkdir(parents=True);c=yaml.safe_load((root/'configs/v2/phase11.yaml').read_text());rows=[]
    for name in ['04C','2018']:
        data=load_dataset(root,name);i=data['identity'];truth=i['snr_db']
        train=i['train_indices'][:12];val=i['validation_indices'][:6];test=i['test_indices'][:6]
        # Small deterministic source rows: evidence only, not a sample for claims.
        subset=np.concatenate([train,val,test]);features=bounded_features(data['signals'][subset],chunk_size=4)
        state,selection=fit_ridge(features,truth[subset],data['ids'][subset],np.arange(12),np.arange(12,18),[.01,.1,1,10,100])
        estimated=predict_ridge(features,state);folder=destination/name;folder.mkdir()
        np.savez_compressed(folder/'source_rows.npz',source_rows=i['source_rows'][subset],sample_ids=data['ids'][subset],estimated_db=estimated)
        def loaders():return tuple(torch.utils.data.DataLoader(LazyFrames(data['signals'],i['labels'],truth,idx,data['grid']),batch_size=3,shuffle=True,num_workers=0) for idx in [train,val])
        def execute(model,subfolder,logger=None):return train_model(lambda:factory(c,data,model),loaders,folder/subfolder,seed=2022,training={'lr':.001,'max_epochs':2},identity='real_data_smoke_not_formal',loss_fn=loss_fn,threads=1,device='cpu',logger=logger)
        for model in ['M0','M6']:
            execute(model,model);net=factory(c,data,model);net.load_state_dict(torch.load(folder/model/'checkpoint.pt',weights_only=True));score=predict(net,data,test,truth,3)
            assert score.shape==(6,data['classes']) and np.isfinite(score).all()
        def interrupt(row):
            if row['epoch']==0:raise RuntimeError('intentional checkpoint interruption')
        try:execute('M6','resumed',interrupt)
        except RuntimeError as exc:
            if str(exc)!='intentional checkpoint interruption':raise
        execute('M6','resumed')
        a=torch.load(folder/'M6/resume.pt',weights_only=True);b=torch.load(folder/'resumed/resume.pt',weights_only=True)
        def same(x,y):
            if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for k in x:same(x[k],y[k])
            elif isinstance(x,(tuple,list)):
                assert len(x)==len(y)
                for u,v in zip(x,y):same(u,v)
            else:assert x==y
        for key in ['model','optimizer','best_state','torch_rng','python_rng','numpy_rng']:same(a[key],b[key])
        for x,y in zip(a['history'],b['history']):same({k:v for k,v in x.items() if k!='seconds'},{k:v for k,v in y.items() if k!='seconds'})
        rows.append(dict(dataset=name,source_sha256=sha(data['source']),frame_samples=data['signals'].shape[-1],classes=data['classes'],bins=len(data['grid']),train_rows=12,validation_rows=6,test_rows=6,epochs=2,models=['M0','M6'],exact_M6_resume=True))
        dump(destination/'progress.json',dict(status='SMOKE_NOT_FORMAL',datasets=rows))
    result=dict(status='SMOKE_NOT_FORMAL',real_data=True,seed=2022,not_representative_no_performance_claim=True,datasets=rows)
    dump(destination/'verdict.json',result);dump(destination/'manifest.json',dict(status='SMOKE_NOT_FORMAL',files={p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}));return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args();print(run(Path(__file__).resolve().parents[2],args.destination))

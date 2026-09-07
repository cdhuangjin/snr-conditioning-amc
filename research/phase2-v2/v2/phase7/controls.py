"""Independent Phase 7 sensitivity protocols; frozen upstream files remain read-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SEEDS=tuple(range(2022,2027))
CONTROLS=('remove_wbfm','frame_rms')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def prepare_control(data,split,control):
    if control not in CONTROLS:raise ValueError('unknown control')
    labels=np.asarray(data['labels']);signals=np.asarray(data['signals']);ids=np.asarray(split.sample_ids)
    if labels.ndim!=1 or labels.dtype.kind not in 'iu' or np.any(labels<0) or np.any(labels>10) or len(ids)!=len(labels) or len(np.unique(ids))!=len(ids):raise ValueError('invalid dataset rows')
    if signals.ndim!=3 or signals.shape[0]!=len(labels) or signals.shape[1]!=2 or not np.isfinite(signals).all():raise ValueError('invalid signals')
    groups=[np.asarray(getattr(split,name),dtype=np.int64) for name in ('train_idx','val_idx','test_idx')]
    flat=np.concatenate(groups)
    if np.any(flat<0) or np.any(flat>=len(labels)) or len(np.unique(flat))!=len(flat):raise ValueError('partition overlap or out-of-range')
    result=dict(data);mapping=np.arange(11,dtype=np.int64)
    if control=='remove_wbfm':
        mapping[3]=-1;mapping[4:]-=1
        result['labels']=mapping[labels]
        groups=[g[labels[g]!=3] for g in groups]
    else:
        power=np.mean(np.sum(signals.astype(np.float64)**2,axis=1),axis=1)
        scale=np.sqrt(power);scale[scale==0]=1
        result['signals']=(signals/scale[:,None,None]).astype(np.float32)
    for group in groups:
        if not len(group):raise ValueError('empty control partition')
    meta=dict(control=control,parent_split_hash=split.split_hash,class_mapping=mapping.tolist(),
        preprocessing='identity' if control=='remove_wbfm' else 'per-frame complex RMS; zero frames unchanged',
        sample_ids_hash=digest(ids.tolist()),
        train_idx=groups[0].tolist(),val_idx=groups[1].tolist(),test_idx=groups[2].tolist(),
        retained_labels_hash=digest(result['labels'][flat].tolist()))
    new=SimpleNamespace(train_idx=groups[0],val_idx=groups[1],test_idx=groups[2],sample_ids=ids,split_hash=digest(meta))
    return result,new,meta


def balanced_control(labels,num_classes=11):
    y=np.asarray(labels)
    if y.ndim!=1 or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=num_classes):raise ValueError('invalid labels')
    counts=np.bincount(y,minlength=num_classes)
    if np.any(counts==0) or not np.all(counts==counts[0]):raise ValueError('unequal or empty class counts: no equivalence claim')
    return dict(counts=counts.tolist(),sample_count=len(y),inverse_frequency_weight=float(1/counts[0]),
        normalized_per_sample_probability=float(1/len(y)),equivalent_uniform_permutation=True,
        proof='All sample weights equal 1/n_c. At draw k without replacement each remaining sample has probability 1/(N-k), hence every permutation probability is 1/N!.',
        limitation='Equivalent probability law only; not identical RNG streams or exactly balanced minibatches. No new training run.')


def remap_control(logits,labels,permutation):
    x=np.asarray(logits);y=np.asarray(labels);p=np.asarray(permutation)
    if x.ndim!=2 or not np.isfinite(x).all() or sorted(p.tolist())!=list(range(x.shape[1])) or y.shape!=(len(x),):raise ValueError('invalid permutation or logits')
    if y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=len(p)):raise ValueError('invalid labels')
    mapped=np.empty_like(x);mapped[:,p]=x
    old=x.argmax(1);new=mapped.argmax(1);inverse=np.argsort(p)[new]
    return dict(logits=mapped,y_true=p[y],y_pred=new,mapped_original_predictions=p[old],
        inverse_mapped_predictions=inverse,tie_rows=(x==x.max(1,keepdims=True)).sum(1)>1,
        identity_changed=inverse!=old,permutation=p)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def atomic_torch(path,value):
    import torch
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def train_model(model_factory,loader_factory,destination,*,seed,training,identity,loss_fn,threads=24,logger=None,device="cpu"):
    """Device-bound deterministic epoch resume; factories must use global torch RNG.

    model.forward_batch(x,snr_db=...,snr_bin=...) returns logits, regularizers.
    loader_factory returns (train_loader,validation_loader) with four tensor fields.
    Snapshots intentionally use only weights-only-loadable primitives/tensors.
    """
    import random
    import time
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    selected=torch.device(device)
    if selected.type not in ("cpu","cuda"):raise ValueError("unsupported training device")
    if selected.type=="cuda":torch.cuda.manual_seed_all(seed)
    model=model_factory().to(selected);optimizer=torch.optim.Adam(model.parameters(),lr=float(training['lr']))
    train_loader,val_loader=loader_factory()
    target=int(training['max_epochs'])
    snapshot=destination/'resume.pt'
    fingerprint=digest(dict(identity=identity,seed=int(seed),training=training,threads=threads,device=str(selected)))
    start=0;best=-1.;best_epoch=-1;best_state=None;history=[]
    if snapshot.is_file():
        state=torch.load(snapshot,map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('resume identity mismatch')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start=state['next_epoch'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];history=state['history']
        if not 0<=start<=target or len(history)!=start:raise ValueError('invalid resume epoch state')
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if selected.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        nr=state['numpy_rng'];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
    for epoch in range(start,target):
        started=time.perf_counter();model.train();loss_sum=0.;seen=0
        for x,y,snr,bins in train_loader:
            x,y,snr,bins=(v.to(selected) for v in (x,y,snr,bins))
            logits,reg=model.forward_batch(x,snr_db=snr,snr_bin=bins)
            loss=loss_fn(logits,y,reg)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(y);seen+=len(y)
        model.eval();correct=total=0
        with torch.no_grad():
            for x,y,snr,bins in val_loader:
                x,y,snr,bins=(v.to(selected) for v in (x,y,snr,bins))
                logits,_=model.forward_batch(x,snr_db=snr,snr_bin=bins)
                if not torch.isfinite(logits).all():raise ValueError('nonfinite validation logits')
                correct+=int((logits.argmax(1)==y).sum());total+=len(y)
        if seen==0 or total==0:raise ValueError('empty training or validation loader')
        accuracy=correct/total
        if accuracy>=best:
            best=accuracy;best_epoch=epoch;best_state={key:val.detach().cpu().clone() for key,val in model.state_dict().items()}
        row=dict(epoch=epoch,train_loss=loss_sum/seen,validation_accuracy=accuracy,best_epoch=best_epoch,seconds=time.perf_counter()-started)
        history.append(row);nr=np.random.get_state()
        state=dict(fingerprint=fingerprint,next_epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,
            history=history,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),numpy_rng=(nr[0],nr[1].tolist(),nr[2],nr[3],nr[4]))
        if selected.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
        atomic_torch(snapshot,state)
        if logger is not None:logger(row)
    if len(history)!=target or best_state is None:raise ValueError('training incomplete')
    checkpoint=destination/'checkpoint.pt';atomic_torch(checkpoint,best_state)
    atomic_json(destination/'epochs.json',history)
    return dict(checkpoint=str(checkpoint),epochs_completed=target,best_epoch=best_epoch,best_val_accuracy=best,duration_seconds=sum(r['seconds'] for r in history),
        complexity={'flops':'not measured','latency':'not measured','note':'class-removal changes output-head parameter count; record actual count separately'})


def write_manifest(output,identity,sources,*,expected_sources=None):
    output=Path(output)
    if (output/'manifest.json').exists():raise ValueError('immutable completed manifest exists')
    actual_sources={str(Path(p).resolve()):file_hash(p) for p in sources}
    if expected_sources is not None and actual_sources!=expected_sources:raise ValueError('source changed during execution')
    manifest=dict(schema_version=1,status='COMPLETE',**identity,
        sources=actual_sources,
        artifacts={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    atomic_json(output/'manifest.json',manifest)
    return manifest


def verify_control_manifest(output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if m.get('status')!='COMPLETE' or m.get('epochs_completed')!=100:raise ValueError('control has not completed 100 epochs')
    names={p.name for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    if names!=set(m['artifacts']):raise ValueError('artifact hash closure differs')
    for name,h in m['artifacts'].items():
        if file_hash(output/name)!=h:raise ValueError(f'artifact hash mismatch: {name}')
    for name,h in m['sources'].items():
        if not Path(name).is_file() or file_hash(name)!=h:raise ValueError(f'source hash mismatch: {name}')
    return m


def summarize_matrix(directories):
    rows=[]
    for path in directories:
        m=verify_control_manifest(path)
        rows.append(dict(control=m['control'],seed=m['seed'],path=str(Path(path).resolve()),manifest_sha256=file_hash(Path(path)/'manifest.json'),
            metrics=json.loads((Path(path)/'metrics.json').read_text(encoding='utf-8'))))
    pairs=[(r['control'],r['seed']) for r in rows]
    if len(rows)!=10 or set(pairs)!={(c,s) for c in CONTROLS for s in SEEDS}:raise ValueError('exactly ten unique completed control/seed runs required')
    return dict(status='RETRAINING_CONTROLS_COMPLETE',runs=sorted(rows,key=lambda r:(r['control'],r['seed'])),
        limitation='This is not full Phase 7 completion; other dataset/channel and causal evidence remain outside scope.')

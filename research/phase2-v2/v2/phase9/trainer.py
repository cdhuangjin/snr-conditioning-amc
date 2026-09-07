"""Independent epoch-resumable trainer with dedicated condition-noise state.

The frozen Phase 7 trainer is deliberately not modified: it has no sampler
state hook. Atomic I/O remains shared; the owning Phase 9 protocol hashes it.
"""
import hashlib
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from v2.phase7.controls import atomic_json,atomic_torch,digest
from v2.phase8.core import conditions


def train_model(model_factory,loader_factory,destination,*,seed,training,identity,loss_fn,condition_noise,threads=24,logger=None,device='cpu'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    selected=torch.device(device)
    if selected.type not in ('cpu','cuda'):raise ValueError('unsupported device')
    if selected.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=model_factory().to(selected);optimizer=torch.optim.Adam(model.parameters(),lr=float(training['lr']))
    initial_digest=hashlib.sha256()
    for name,value in model.state_dict().items():initial_digest.update(name.encode());initial_digest.update(value.detach().cpu().numpy().tobytes())
    initial_model_sha256=initial_digest.hexdigest()
    train_loader,val_loader=loader_factory();target=int(training['max_epochs'])
    fingerprint=digest(dict(identity=identity,seed=seed,training=training,device=str(selected),threads=threads,arm=condition_noise.arm,std_db=condition_noise.std_db))
    snapshot=destination/'resume.pt';start=0;history=[];best=-1.;best_epoch=-1;best_state=None
    if snapshot.exists():
        state=torch.load(snapshot,map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('resume identity mismatch')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start=state['next_epoch'];history=state['history'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state']
        if len(history)!=start or not 0<=start<=target:raise ValueError('invalid epoch snapshot')
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if selected.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        nr=state['numpy_rng'];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
        condition_noise.restore(state['condition_rng'])
    for epoch in range(start,target):
        began=time.perf_counter();model.train();loss_sum=0.;seen=0;clipped=0;trace=hashlib.sha256();order_trace=hashlib.sha256();rng_before=condition_noise.state()
        for x,y,true_snr,row_ids in train_loader:
            raw,donors=condition_noise.sample(true_snr.numpy());db,bins=conditions(raw)
            trace.update(np.asarray(row_ids,dtype=np.int64).tobytes());trace.update(raw.tobytes());trace.update(donors.tobytes())
            order_trace.update(np.asarray(row_ids,dtype=np.int64).tobytes())
            clipped+=int(np.sum((raw<-20)|(raw>18)))
            logits,regularizers=model.forward_batch(x.to(selected),snr_db=torch.as_tensor(db,device=selected),snr_bin=torch.as_tensor(bins,device=selected))
            loss=loss_fn(logits,y.to(selected),regularizers)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
        model.eval();correct=total=0
        with torch.no_grad():
            for x,y,estimated_snr,_row_ids in val_loader:
                db,bins=conditions(estimated_snr.numpy())
                logits,_=model.forward_batch(x.to(selected),snr_db=torch.as_tensor(db,device=selected),snr_bin=torch.as_tensor(bins,device=selected))
                if not torch.isfinite(logits).all():raise ValueError('nonfinite validation logits')
                correct+=int((logits.argmax(1)==y.to(selected)).sum());total+=len(y)
        if not seen or not total:raise ValueError('empty train/validation data')
        accuracy=correct/total
        if accuracy>=best:
            best=accuracy;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row=dict(epoch=epoch,train_loss=loss_sum/seen,validation_accuracy=accuracy,best_epoch=best_epoch,data_order_sha256=order_trace.hexdigest(),condition_trace_sha256=trace.hexdigest(),condition_rng_before=rng_before,condition_rng_after=condition_noise.state(),clipped_training_fraction=clipped/seen,seconds=time.perf_counter()-began)
        history.append(row);nr=np.random.get_state()
        state=dict(fingerprint=fingerprint,next_epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,history=history,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),numpy_rng=(nr[0],nr[1].tolist(),nr[2],nr[3],nr[4]),condition_rng=condition_noise.state())
        if selected.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
        atomic_torch(snapshot,state)
        if logger:logger(row)
    if len(history)!=target or best_state is None:raise ValueError('training incomplete')
    atomic_torch(destination/'checkpoint.pt',best_state);atomic_json(destination/'epochs.json',history)
    return dict(checkpoint=str(destination/'checkpoint.pt'),epochs_completed=target,best_epoch=best_epoch,best_val_accuracy=best,initial_model_sha256=initial_model_sha256,validation_condition='frozen_full_training_estimate',condition_rng_separate=True)

"""Post-hoc target, prospective five-seed intervention; separate from original controls."""
from types import SimpleNamespace
import numpy as np
from .controls import digest


def prepare_amssb(data,split):
    y=np.asarray(data['labels']);ids=np.asarray(split.sample_ids)
    if y.ndim!=1 or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>10) or len(ids)!=len(y) or len(np.unique(ids))!=len(ids):raise ValueError('invalid original rows')
    groups=[np.asarray(getattr(split,k)) for k in ('train_idx','val_idx','test_idx')]
    flat=np.concatenate(groups)
    if np.any(flat<0) or np.any(flat>=len(y)) or len(np.unique(flat))!=len(flat):raise ValueError('invalid partitions')
    groups=[g[y[g]!=10] for g in groups]
    if any(len(g)==0 for g in groups):raise ValueError('empty retained partition')
    mapping=np.r_[np.arange(10),-1];result=dict(data);result['labels']=mapping[y]
    meta=dict(control='remove_amssb',target_original_class=10,target_selection='posthoc_from_existing_test_predictions',
        parent_split_hash=split.split_hash,class_mapping=mapping.tolist(),preprocessing='identity',sample_ids_hash=digest(ids.tolist()),
        train_idx=groups[0].tolist(),val_idx=groups[1].tolist(),test_idx=groups[2].tolist(),retained_labels_hash=digest(result['labels'][np.concatenate(groups)].tolist()))
    return result,SimpleNamespace(train_idx=groups[0],val_idx=groups[1],test_idx=groups[2],sample_ids=ids,split_hash=digest(meta)),meta


def align_predictions(source_ids,predictions,target_ids):
    source_ids=np.asarray(source_ids);target_ids=np.asarray(target_ids)
    if len(np.unique(source_ids))!=len(source_ids) or len(predictions)!=len(source_ids) or len(np.unique(target_ids))!=len(target_ids):raise ValueError('ambiguous sample IDs')
    positions={v:i for i,v in enumerate(source_ids.tolist())}
    if any(v not in positions for v in target_ids.tolist()):raise ValueError('missing matched sample ID')
    return np.asarray(predictions)[[positions[v] for v in target_ids.tolist()]]


def distribution_metrics(y,pred,num_outputs):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.shape!=pred.shape or y.ndim!=1 or not len(y) or np.any(y<0) or np.any(y>=num_outputs) or np.any(pred<0) or np.any(pred>=num_outputs):raise ValueError('invalid distribution inputs')
    confusion=np.bincount(y*num_outputs+pred,minlength=num_outputs**2).reshape(num_outputs,num_outputs)
    totals=confusion.sum(1);recall=np.divide(confusion.diagonal(),totals,out=np.zeros(num_outputs,dtype=float),where=totals>0)
    counts=confusion.sum(0);p=counts/len(y);ba=float(recall[totals>0].mean());maximum=float(p.max())
    return dict(n=len(y),num_outputs=num_outputs,accuracy=float((y==pred).mean()),balanced_accuracy=ba,
        concentration=maximum,dominant_class=int(counts.argmax()),concentration_minus_ba=maximum-ba,
        normalized_entropy=float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(num_outputs)),hhi=float(p@p),
        counts=counts.tolist(),confusion=confusion.tolist(),recalls=[float(v) if t else None for v,t in zip(recall,totals)])


def band_metrics(y,pred,snr,num_outputs):
    snr=np.asarray(snr);masks={'all':np.ones(len(snr),dtype=bool),'low':snr<=-8,'mid':(snr>=-6)&(snr<=-2),'high':snr>=0}
    masks.update({f'snr_{int(s)}':snr==s for s in np.unique(snr)})
    return {name:distribution_metrics(np.asarray(y)[mask],np.asarray(pred)[mask],num_outputs) for name,mask in masks.items() if mask.any()}

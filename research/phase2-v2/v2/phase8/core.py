import warnings
import numpy as np
import torch
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def donor_indices(snr, labels, mode, seed):
    snr,labels=np.asarray(snr),np.asarray(labels)
    if snr.shape!=labels.shape or snr.ndim!=1: raise ValueError('one-dimensional aligned strata required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,8]))
    if mode=='global': return rng.permutation(len(snr))
    if mode not in ('within_snr','across_class'): raise ValueError('invalid donor mode')
    donors=np.arange(len(snr))
    for level in np.unique(snr):
        indices=np.flatnonzero(snr==level)
        if mode=='within_snr': donors[indices]=rng.permutation(indices);continue
        classes=rng.permutation(np.unique(labels[indices]))
        groups=[rng.permutation(indices[labels[indices]==c]) for c in classes]
        largest=max(map(len,groups))
        if largest>len(indices)/2: raise ValueError('majority class prevents an across-class donor bijection')
        ordered=np.concatenate(groups)
        donors[ordered]=np.roll(ordered,largest)
        if np.any(labels[donors[indices]]==labels[indices]): raise ValueError('across-class donor construction failed')
    return donors


def conditions(estimates):
    raw=np.asarray(estimates)
    if raw.ndim!=1 or not np.isfinite(raw).all(): raise ValueError('finite one-dimensional SNR estimates required')
    db=np.clip(raw,-20.,18.).astype(np.float32)
    bins=np.abs(db[:,None]-np.arange(-20,20,2)[None,:]).argmin(1).astype(np.int64)
    return db,bins


def one_metrics(y,logits,classes):
    pred=logits.argmax(1);matrix=np.zeros((classes,classes),dtype=np.int64)
    np.add.at(matrix,(y,pred),1);support=matrix.sum(1)
    recall=np.divide(matrix.diagonal(),support,out=np.zeros(classes),where=support>0)
    precision=np.divide(matrix.diagonal(),matrix.sum(0),out=np.zeros(classes),where=matrix.sum(0)>0)
    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(classes),where=recall+precision>0)
    shares=matrix.sum(0)/len(y);nonzero=shares[shares>0]
    return dict(count=len(y),accuracy=float(np.mean(y==pred)),balanced_accuracy=float(recall[support>0].mean()),
                macro_f1=float(f1.mean()),recall=[float(r) if n else None for r,n in zip(recall,support)],
                concentration=float(shares.max()),prediction_entropy=float(-np.sum(nonzero*np.log(nonzero))/np.log(classes)),
                dominant_class=int(shares.argmax()),confusion=matrix.tolist())


def grouped_metrics(y,logits,snr,classes=11):
    y=np.asarray(y,dtype=np.int64);logits=np.asarray(logits);snr=np.asarray(snr)
    if logits.shape!=(len(y),classes) or not np.isfinite(logits).all(): raise ValueError('invalid logits')
    result={'overall':one_metrics(y,logits,classes)}
    for name,groups in [('snr',snr),('class',y)]:
        result[name]={str(float(v) if name=='snr' else int(v)):one_metrics(y[groups==v],logits[groups==v],classes) for v in np.unique(groups)}
    result['bands']={name:one_metrics(y[mask],logits[mask],classes) for name,mask in {'low':snr<=-8,'mid':(snr>=-6)&(snr<=-2),'high':snr>=0}.items() if mask.any()}
    return result


def estimate_metrics(truth,estimated,labels):
    def score(mask):
        error=estimated[mask]-truth[mask]
        return dict(n=int(mask.sum()),mae_db=float(np.abs(error).mean()),rmse_db=float(np.sqrt(np.mean(error**2))))
    return dict(overall=score(np.ones(len(truth),dtype=bool)),snr={str(float(v)):score(truth==v) for v in np.unique(truth)},classes={str(int(v)):score(labels==v) for v in np.unique(labels)})


def representation(x,name):
    x=np.asarray(x,dtype=float).reshape(-1,1)
    if name=='scalar': return x
    if name=='cubic': return np.column_stack([x,x**2,x**3])
    if name in ('rbf2','rbf6'):
        return np.exp(-.5*((x-np.arange(-20,20,2)[None,:])/float(name[3:]))**2)
    raise ValueError('unknown E4 representation')


def e4_logits(x,state):
    z=(representation(x,state['representation'])-np.asarray(state['mean']))/np.asarray(state['scale'])
    scores=z@np.asarray(state['coef']).T+np.asarray(state['intercept'])
    return np.column_stack([np.zeros(len(scores)),scores[:,0]]) if scores.shape[1]==1 else scores


def fit_e4(train_x,train_y,val_x,val_y,options):
    candidates=[];states=[]
    for name in options['representations']:
        scaler=StandardScaler().fit(representation(train_x,name))
        for c in options['cs']:
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model=LogisticRegression(C=c,max_iter=options['max_iter'],solver='lbfgs',random_state=2022).fit(scaler.transform(representation(train_x,name)),train_y)
            state=dict(representation=name,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=model.intercept_.tolist(),classes=model.classes_.tolist(),C=c)
            pred=np.asarray(state['classes'])[e4_logits(val_x,state).argmax(1)]
            candidates.append(dict(representation=name,C=c,validation_macro_f1=float(f1_score(val_y,pred,labels=model.classes_,average='macro',zero_division=0))))
            states.append(state)
    winner=min(range(len(candidates)),key=lambda i:(-candidates[i]['validation_macro_f1'],options['representations'].index(candidates[i]['representation']),candidates[i]['C']))
    return states[winner],dict(fit_partition='train',selection_partition='validation',refit_train_validation=False,n_independent_fits=1,candidates=candidates,winner_index=winner,tie_break=['validation_macro_f1_desc','representation_order_asc','C_asc'])


def head_logits(model,features,snr,batch_size=64):
    device=next(model.parameters()).device;db,bins=conditions(snr);output=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            end=start+batch_size
            output.append(model.classify_pooled(torch.as_tensor(features[start:end],device=device),snr_db=torch.as_tensor(db[start:end],device=device),snr_bin=torch.as_tensor(bins[start:end],device=device)).cpu().numpy())
    return np.concatenate(output)


def cache_and_replay(model,signals,snr,registered,batch_size=64):
    device=next(model.parameters()).device;db,bins=conditions(snr);features=[];full=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(signals),batch_size):
            end=start+batch_size;x=torch.as_tensor(signals[start:end],device=device)
            z=torch.as_tensor(db[start:end],device=device);b=torch.as_tensor(bins[start:end],device=device)
            full.append(model.forward_batch(x,snr_db=z,snr_bin=b)[0].cpu().numpy())
            features.append(model.extract_pooled_features(x)[0].cpu().numpy())
    full=np.concatenate(full);features=np.concatenate(features)
    if not np.allclose(full,registered,rtol=1e-6,atol=1e-7) or not np.array_equal(full.argmax(1),registered.argmax(1)):
        raise ValueError('E1 full forward differs from registered predictions')
    cached=head_logits(model,features,snr,batch_size)
    if not np.allclose(cached,full,rtol=1e-6,atol=1e-7) or not np.array_equal(cached.argmax(1),full.argmax(1)):
        raise ValueError('raw cache head replay differs from full forward')
    return features,cached,dict(hook='extract_pooled_features: attended before conditioning',raw_dimension=int(features.shape[1]),full_vs_registered_max_abs=float(np.abs(full-registered).max()),cache_vs_full_max_abs=float(np.abs(cached-full).max()),rtol=1e-6,atol=1e-7)

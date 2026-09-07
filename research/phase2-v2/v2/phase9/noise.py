"""Training-only conditional-on-frozen-alpha OOF residuals and isolated noise RNG."""
from copy import deepcopy
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def make_folds(snr,labels,n_folds,seed):
    snr,labels=np.asarray(snr),np.asarray(labels)
    if snr.shape!=labels.shape or n_folds!=5:raise ValueError('aligned training strata and five folds required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,90]));folds=np.full(len(snr),-1,dtype=np.int64)
    for s in np.unique(snr):
        for c in np.unique(labels):
            indices=np.flatnonzero((snr==s)&(labels==c))
            if len(indices)<n_folds:raise ValueError('each training class/SNR cell must contain at least five rows')
            order=rng.permutation(indices);folds[order]=np.arange(len(order))%n_folds
    if np.any(folds<0):raise ValueError('incomplete fold assignment')
    return folds


def crossfit(features,snr,sample_ids,folds,alpha):
    features=np.asarray(features,dtype=np.float64);snr=np.asarray(snr,dtype=np.float64);ids=np.asarray(sample_ids);folds=np.asarray(folds)
    if len(np.unique(ids))!=len(ids) or len(features)!=len(ids) or snr.shape!=folds.shape or len(snr)!=len(ids) or set(np.unique(folds))!=set(range(5)):
        raise ValueError('OOF row/fold identity mismatch')
    if not np.isfinite(features).all() or not np.isfinite(snr).all():raise ValueError('nonfinite OOF source')
    out={'sample_ids':ids,'folds':folds,'true_snr_db':snr,'oof_estimates':np.empty(len(ids)),'alpha':np.asarray(alpha)}
    means=[];scales=[];coefficients=[];intercepts=[]
    for fold in range(5):
        fit=folds!=fold;held=~fit;scaler=StandardScaler().fit(features[fit]);model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(features[fit]),snr[fit])
        out['oof_estimates'][held]=model.predict(scaler.transform(features[held]))
        out[f'fit_ids_{fold}']=ids[fit];out[f'heldout_ids_{fold}']=ids[held]
        means.append(scaler.mean_);scales.append(scaler.scale_);coefficients.append(model.coef_);intercepts.append(model.intercept_)
    out.update(means=np.asarray(means),scales=np.asarray(scales),coefficients=np.asarray(coefficients),intercepts=np.asarray(intercepts))
    out['residuals']=out['oof_estimates']-snr
    return out


class ConditionNoise:
    def __init__(self,arm,seed,*,pool_snr=None,pool_errors=None,std_db=2.):
        if arm not in ('clean','generic','empirical_snr'):raise ValueError('unknown noise arm')
        self.arm=arm;self.std_db=float(std_db);self.rng=np.random.default_rng(np.random.SeedSequence([seed,9]))
        self.pool_snr=None if pool_snr is None else np.asarray(pool_snr,dtype=float)
        self.pool_errors=None if pool_errors is None else np.asarray(pool_errors,dtype=float)
        if arm=='empirical_snr' and (self.pool_snr is None or self.pool_errors is None or self.pool_snr.shape!=self.pool_errors.shape or not np.isfinite(self.pool_errors).all()):raise ValueError('finite aligned training OOF pool required')
        self.groups={} if self.pool_snr is None else {s:np.flatnonzero(self.pool_snr==s) for s in np.unique(self.pool_snr)}

    def state(self):return deepcopy(self.rng.bit_generator.state)
    def restore(self,state):self.rng.bit_generator.state=deepcopy(state)

    def sample(self,true_snr):
        truth=np.asarray(true_snr,dtype=float)
        if truth.ndim!=1 or not np.isfinite(truth).all():raise ValueError('finite 1D true training SNR required')
        donors=np.full(len(truth),-1,dtype=np.int64)
        if self.arm=='clean':return truth.copy(),donors
        if self.arm=='generic':return truth+self.rng.normal(0,self.std_db,len(truth)),donors
        for s in np.unique(truth):
            if s not in self.groups:raise ValueError('training SNR absent from OOF pool')
            target=np.flatnonzero(truth==s);donors[target]=self.rng.choice(self.groups[s],len(target),replace=True)
        return truth+self.pool_errors[donors],donors

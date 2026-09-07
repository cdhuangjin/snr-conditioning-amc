"""Training-only scaler/Ridge, validation-only alpha; no train+val refit."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from v2.phase6.audit import FEATURE_NAMES,array_hash


def predict_ridge(features,state):
    values=np.asarray(features,dtype=np.float64)
    return ((values-np.asarray(state['feature_mean']))/np.asarray(state['feature_scale']))@np.asarray(state['coef'])+state['intercept']


def fit_ridge(features,truth,ids,train,validation,alphas):
    features=np.asarray(features,dtype=np.float64);truth=np.asarray(truth);ids=np.asarray(ids);train=np.asarray(train);validation=np.asarray(validation)
    if features.shape!=(len(truth),7) or len(ids)!=len(truth) or len(np.unique(ids))!=len(ids) or not np.isfinite(features).all() or not np.isfinite(truth).all():raise ValueError('invalid estimator input')
    if not len(train) or not len(validation) or len(np.unique(train))!=len(train) or len(np.unique(validation))!=len(validation) or np.intersect1d(train,validation).size or np.any(train<0) or np.any(validation<0) or max(train.max(),validation.max())>=len(truth):raise ValueError('invalid fit/selection partitions')
    if not alphas or any(not np.isfinite(a) or a<=0 for a in alphas):raise ValueError('invalid ridge grid')
    scaler=StandardScaler().fit(features[train]);tx=scaler.transform(features[train]);vx=scaler.transform(features[validation]);candidates=[];models=[]
    for alpha in alphas:
        model=Ridge(alpha=alpha,solver='svd').fit(tx,truth[train]);models.append(model);candidates.append(dict(alpha=float(alpha),validation_mae=float(np.mean(np.abs(model.predict(vx)-truth[validation])))))
    chosen=min(range(len(candidates)),key=lambda i:(candidates[i]['validation_mae'],-candidates[i]['alpha']));model=models[chosen]
    state=dict(feature_names=list(FEATURE_NAMES),feature_mean=scaler.mean_.tolist(),feature_scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_),fit_partition='train',fit_count=len(train),fit_sample_ids_sha256=array_hash(ids[train]),alpha=float(alphas[chosen]))
    selection=dict(partition='validation',candidates=candidates,selected_alpha=float(alphas[chosen]),validation_sample_ids_sha256=array_hash(ids[validation]),train_features_sha256=array_hash(features[train]),train_targets_sha256=array_hash(truth[train]),validation_features_sha256=array_hash(features[validation]),validation_targets_sha256=array_hash(truth[validation]),refit_train_plus_validation=False)
    return state,selection

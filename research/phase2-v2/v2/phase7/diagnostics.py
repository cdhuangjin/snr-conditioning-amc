"""Descriptive WBFM diagnostics. No causal or Phase 7 completion claim."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = tuple(range(2022, 2027))
PENDING = ['remove-WBFM retraining', 'label permutation retraining', 'class-balanced sampling retraining', 'equalized-statistics control', 'different dataset/channel dominance']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def signal_statistics(signals):
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] < 4 or not np.isfinite(x).all():
        raise ValueError('signals must be finite N,2,T with T >= 4')
    z = x[:,0] + 1j*x[:,1]; n = z.shape[1]
    p = np.abs(z)**2; energy = p.sum(axis=1); power = p.mean(axis=1)
    eps = np.finfo(float).tiny
    psd = np.abs(np.fft.fftshift(np.fft.fft(z,axis=1),axes=1))**2/n
    shape = psd/np.maximum(psd.sum(axis=1,keepdims=True),eps)
    cdf = np.cumsum(shape,axis=1)
    # Contiguous central 90% spectral interval; normalized cycles/sample.
    lo = np.argmax(cdf >= .05,axis=1); hi = np.argmax(cdf >= .95,axis=1)
    bw = (hi-lo+1)/n
    zero = energy == 0
    bw[zero] = 0
    flat = np.exp(np.log(np.maximum(psd,eps)).mean(axis=1))/np.maximum(psd.mean(axis=1),eps)
    flat[zero] = 0
    autocorr = np.stack([(z[:,lag:]*z[:,:n-lag].conj()).mean(axis=1)/np.maximum(power,eps) for lag in range(min(17,n))],axis=1)
    amp = np.abs(z); centered = x-x.mean(axis=2,keepdims=True)
    variance = (centered**2).mean(axis=2)
    kurtosis = (centered**4).mean(axis=2)/np.maximum(variance**2,eps)-3
    kurtosis[variance == 0] = 0
    return dict(energy=energy,power=power,spectral_flatness=flat,psd=psd,psd_shape=shape,
        bandwidth_90=bw,autocorrelation=autocorr,iq_mean=x.mean(axis=2),iq_variance=variance,
        iq_excess_kurtosis=kurtosis,amplitude_mean=amp.mean(axis=1),amplitude_std=amp.std(axis=1),
        peak_to_average=p.max(axis=1)/np.maximum(power,eps),
        temporal_difference_power=np.abs(np.diff(z,axis=1)) .__pow__(2).mean(axis=1),
        lag1_phase_increment_mean=np.angle(z[:,1:]*z[:,:-1].conj()).mean(axis=1),zero_energy=zero)


def prediction_summary(y_true,y_pred,num_classes,wbfm):
    y=np.asarray(y_true); p=np.asarray(y_pred)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or y.dtype.kind not in 'iu' or p.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=num_classes) or np.any(p<0) or np.any(p>=num_classes):
        raise ValueError('invalid class labels')
    errors=p!=y; target=y==wbfm
    tc=np.bincount(y,minlength=num_classes); pc=np.bincount(p,minlength=num_classes)
    return dict(n=len(y),true_count=tc.tolist(),predicted_count=pc.tolist(),true_frequency=(tc/len(y)).tolist(),predicted_frequency=(pc/len(y)).tolist(),
        errors=int(errors.sum()),errors_to_wbfm=int((errors & (p==wbfm)).sum()),
        error_to_wbfm_share=float((errors & (p==wbfm)).sum()/errors.sum()) if errors.any() else None,
        wbfm_recall=float((p[target]==wbfm).mean()) if target.any() else None)


def feature_summary(features,labels,num_classes,wbfm):
    f=np.asarray(features,dtype=float); y=np.asarray(labels)
    if f.ndim != 2 or f.shape[0]!=len(y) or not np.isfinite(f).all(): raise ValueError('invalid features')
    centers=[f[y==c].mean(axis=0) if np.any(y==c) else None for c in range(num_classes)]
    mean=f.mean(axis=0)
    between=sum(np.sum(y==c)*np.sum((center-mean)**2) for c,center in enumerate(centers) if center is not None)
    within=sum(np.sum((f[y==c]-center)**2) for c,center in enumerate(centers) if center is not None)
    distances=[float(np.linalg.norm(center-centers[wbfm])) if center is not None and centers[wbfm] is not None else None for center in centers]
    return dict(between_scatter=float(between),within_scatter=float(within),between_within_ratio=float(between/within) if within>0 else None,wbfm_centroid_distances=distances,
                interpretation='Euclidean distances are comparable only within this model/seed/hook.')


def _load(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k] for k in a.files}


def _aligned(a,b):
    for k in ('sample_ids','y_true','snr_db'):
        if k not in b or not np.array_equal(a[k],b[k]): raise ValueError(f'{k} row alignment differs')


def _json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def analyze_bundles(raw_path,runs,output,*,class_names,split_hash,source_paths=()):
    """Inputs must already be vetted upstream; runner supplies verified Phase 4 sources."""
    output=Path(output)
    if output.exists(): raise ValueError('immutable output already exists')
    if len(class_names)!=len(set(class_names)) or class_names.count('WBFM')!=1: raise ValueError('unique WBFM class required')
    for model in set(r['model_id'] for r in runs):
        seeds=[r['seed'] for r in runs if r['model_id']==model]
        if sorted(seeds)!=list(SEEDS): raise ValueError('exactly five unique seeds per model required')
    if not runs: raise ValueError('five seed runs required')
    raw=_load(raw_path); ids=raw['sample_ids']; y=raw['y_true']; snr=raw['snr_db']
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or len(ids)!=len(raw['signals']) or y.shape!=ids.shape or snr.shape!=ids.shape or not np.isfinite(snr).all(): raise ValueError('raw row schema invalid')
    stats=signal_statistics(raw['signals']); w=class_names.index('WBFM'); k=len(class_names)
    # Power-matched proper complex iid Gaussian noise; not a recovered dataset noise component.
    rng=np.random.default_rng(2022)
    noise=rng.normal(size=raw['signals'].shape)
    noise*=np.sqrt(stats['power']/np.maximum((noise**2).sum(axis=1).mean(axis=1),np.finfo(float).tiny))[:,None,None]
    ns=signal_statistics(noise)
    scalar_keys=[name for name,val in stats.items() if val.ndim==1 and val.dtype.kind!='b']
    pred_rows=[]; feature_rows=[]; signal_rows=[]; noise_rows=[]
    sources={str(Path(p).resolve()):sha256(p) for p in [raw_path,*source_paths]}
    bundles=[]
    for r in runs:
        p=_load(r['predictions']); _aligned(raw,p)
        if int(p['seed'])!=r['seed'] or str(p['model_id'].item())!=r['model_id'] or p['split_hash'].item()!=split_hash: raise ValueError('prediction provenance differs')
        f=None
        if r.get('features'):
            f=_load(r['features']); _aligned(raw,f)
            if 'predictions_sha256' in f and f['predictions_sha256'].item()!=sha256(r['predictions']): raise ValueError('feature prediction hash differs')
        for path in [r['predictions']]+([r['features']] if f is not None else []): sources[str(Path(path).resolve())]=sha256(path)
        bundles.append((r,p,f))
    for db in np.unique(snr):
        m=snr==db
        for c in range(k):
            cm=m&(y==c)
            if not cm.any(): continue
            signal_rows.append(dict(snr_db=float(db),class_id=c,n=int(cm.sum()),**{name:float(val[cm].mean()) for name,val in stats.items() if name in scalar_keys}))
            # Hellinger PSD-shape distance and lag 1..16 autocorrelation distance.
            noise_rows.append(dict(snr_db=float(db),class_id=c,n=int(cm.sum()),
                psd_hellinger_mean=float(np.sqrt(.5*np.sum((np.sqrt(stats['psd_shape'][cm])-np.sqrt(ns['psd_shape'][cm]))**2,axis=1)).mean()),
                autocorrelation_l2_mean=float(np.linalg.norm(stats['autocorrelation'][cm,1:]-ns['autocorrelation'][cm,1:],axis=1).mean())))
        for r,p,f in bundles:
            identity=dict(model_id=r['model_id'],seed=r['seed'],snr_db=float(db))
            pred_rows.append(dict(**identity,**prediction_summary(y[m],p['y_pred'][m],k,w)))
            if f is not None: feature_rows.append(dict(**identity,**feature_summary(f['features'][m],y[m],k,w)))
    output.mkdir(parents=True)
    np.savez_compressed(output/'raw_statistics.npz',sample_ids=ids,y_true=y,snr_db=snr,**stats)
    np.savez_compressed(output/'noise_reference.npz',sample_ids=ids,signals=noise,**ns)
    summary=dict(schema_version=1,split_hash=split_hash,class_names=class_names,predictions=pred_rows,features=feature_rows,signals=signal_rows,noise_distances=noise_rows,
        noise_reference={'kind':'generated proper complex iid Gaussian, per-frame power matched','seed':2022,'scope':'raw signal statistic comparison only; no noise features extracted','not_dataset_noise':True},pending_controls=PENDING)
    _json(output/'summary.json',summary)
    for name,rows in [('predictions',pred_rows),('features',feature_rows),('signals',signal_rows),('noise_distances',noise_rows)]:
        if rows:
            with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows({key:json.dumps(val) if isinstance(val,list) else val for key,val in row.items()} for row in rows)
    sources[str(Path(__file__).resolve())]=sha256(__file__)
    manifest=dict(schema_version=1,status='DIAGNOSTICS_COMPLETE_CONTROLS_PENDING',split_hash=split_hash,sources=sources,artifacts={p.name:sha256(p) for p in output.iterdir() if p.is_file()},pending_controls=PENDING)
    _json(output/'manifest.json',manifest)
    return manifest


def verify_manifest(output):
    output=Path(output); m=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    files={p.name for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    if files!=set(m['artifacts']): raise ValueError('artifact hash closure mismatch')
    for name,h in m['artifacts'].items():
        if sha256(output/name)!=h: raise ValueError(f'artifact hash mismatch: {name}')
    for name,h in m['sources'].items():
        if not Path(name).is_file() or sha256(name)!=h: raise ValueError(f'source hash mismatch: {name}')
    return m

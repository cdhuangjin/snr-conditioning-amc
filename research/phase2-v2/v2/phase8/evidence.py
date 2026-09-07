"""Fail-closed dependency and estimator alignment checks."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def safe(root,relative):
    path=(Path(root)/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):raise ValueError('unsafe evidence path')
    return path


def verify_manifest(root,path,expected_hash,allowed_status=('COMPLETE',)):
    if sha(path)!=expected_hash:raise ValueError('gate manifest hash mismatch')
    manifest=load(path)
    if manifest.get('status') not in allowed_status:raise ValueError('gate manifest is not complete')
    files=manifest.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('gate manifest lacks file closure')
    for name,digest in files.items():
        if sha(safe(path.parent,name))!=digest:raise ValueError('gate artifact hash mismatch')
    actual={p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file() and p!=path}
    if actual!=set(files):raise ValueError('gate artifact closure is incomplete or polluted')
    for name,digest in manifest.get('dependencies',manifest.get('source_files',{})).items():
        if sha(safe(root,name))!=digest:raise ValueError('gate dependency hash mismatch')
    return manifest


def check_gates(root,config):
    root=Path(root)
    try:
        current=load(safe(root,config['phase6_current']))
        # This protocol has a preserved failed FP32 attempt plus a separate
        # bounded scientific receipt. A plain rewritten COMPLETE pointer is
        # not an alternative acceptance path.
        if current.get('kind')!='phase6_bounded_scientific_composite' or current.get('status')!='COMPLETE':
            raise ValueError('Phase 6 gate requires a bounded scientific composite receipt')
        from v2.phase6.composite import validate_composite
        phase6=validate_composite(root,current)
        if phase6.resolve()!=safe(root,current['data_attempt']):
            raise ValueError('Phase 6 validated data attempt differs from current binding')
        gate=load(safe(root,config['phase7_gate']))
        if gate.get('status')!='COMPLETE' or gate.get('allow_phase8') is not True:
            raise ValueError('dependency gate does not permit Phase 8')
        from v2.phase7.primary_receipt import validate_primary_gate
        validate_primary_gate(root,gate)
        return phase6,{'phase6_current':current,'phase6_data_attempt':phase6.relative_to(root.resolve()).as_posix(),'phase7_gate':gate,'gate_file_hashes':{config['phase6_current']:sha(root/config['phase6_current']),config['phase7_gate']:sha(root/config['phase7_gate'])}}
    except (OSError,KeyError,TypeError) as exc:
        raise ValueError(f'dependency gate missing or malformed: {exc}') from exc


def load_estimator(phase6,dataset,split):
    from v2.phase6.audit import array_hash,frame_features
    directory=Path(phase6)/'legacy_identity'
    with np.load(directory/'features.npz',allow_pickle=False) as bundle:features={k:bundle[k] for k in bundle.files}
    with np.load(directory/'all_partition_predictions.npz',allow_pickle=False) as bundle:pred={k:bundle[k] for k in bundle.files}
    state=load(directory/'snr_ridge_state.json');selection=load(directory/'snr_ridge_selection.json')
    partitions={'train':split.train_idx,'validation':split.val_idx,'test':split.test_idx}
    for name,indices in partitions.items():
        np.testing.assert_array_equal(pred[name+'_indices'],indices)
        np.testing.assert_array_equal(features[name+'_indices'],indices)
    for ids in (features['sample_ids'],pred['sample_ids']):np.testing.assert_array_equal(ids,split.sample_ids)
    np.testing.assert_array_equal(pred['modulation'],dataset['labels']);np.testing.assert_array_equal(pred['snr_db'],dataset['snrs'])
    if state['fit_partition']!='train' or state['fit_sample_ids_sha256']!=array_hash(split.sample_ids[split.train_idx]) or state['fit_count']!=len(split.train_idx):
        raise ValueError('estimator fit provenance is not train-only')
    if selection['fit_partition']!='train' or selection['selection_partition']!='validation' or selection['refit_train_validation'] is not False:
        raise ValueError('estimator selection provenance differs')
    recomputed=frame_features(dataset['signals'])
    np.testing.assert_allclose(features['features'],recomputed,rtol=1e-12,atol=1e-12)
    train=recomputed[split.train_idx]
    np.testing.assert_allclose(state['feature_mean'],train.mean(0),rtol=1e-10,atol=1e-12)
    # StandardScaler population variance; constant-feature guard follows sklearn.
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(train)
    np.testing.assert_allclose(state['feature_scale'],scaler.scale_,rtol=1e-10,atol=1e-12)
    for name,indices in [('train',split.train_idx),('validation',split.val_idx)]:
        if selection[name+'_x_sha256']!=array_hash(recomputed[indices]) or selection[name+'_y_sha256']!=array_hash(dataset['snrs'][indices]):
            raise ValueError('estimator selection partition hash mismatch')
    from sklearn.linear_model import Ridge
    refit=Ridge(alpha=selection['selected_hyperparameter'],solver='svd').fit(scaler.transform(train),dataset['snrs'][split.train_idx])
    np.testing.assert_allclose(state['coef'],refit.coef_,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(state['intercept'],refit.intercept_,rtol=1e-10,atol=1e-10)
    estimates=((recomputed-np.asarray(state['feature_mean']))/np.asarray(state['feature_scale']))@np.asarray(state['coef'])+state['intercept']
    np.testing.assert_allclose(estimates,pred['snr_ridge'],rtol=1e-10,atol=1e-10)
    if not np.isfinite(estimates).all():raise ValueError('nonfinite deployment estimates')
    return estimates,dict(state=state,selection=selection,source_files={p.relative_to(phase6).as_posix():sha(p) for p in [directory/'features.npz',directory/'all_partition_predictions.npz',directory/'snr_ridge_state.json',directory/'snr_ridge_selection.json']})

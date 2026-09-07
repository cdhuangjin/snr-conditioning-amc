"""Gated serial cross-dataset training, bounded inference and immutable replay."""
import importlib.metadata
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from filelock import FileLock
from models.model_conditioning import AWNConditioned
from v2.phase7.controls import train_model,digest,atomic_json
from v2.phase7.deterministic_padding import install_adapter,ADAPTER_ID,DeterministicReflectionPad1d
from v2.phase8.evidence import load,dump,sha,safe
from v2.phase10.runner import manifest_closure,check_gates as upstream_gates
from v2.phase6.composite import validate_composite
from v2.phase9.analysis import error_strata
from .data import load_dataset,LazyFrames,bounded_features,dynamic_conditions
from .estimator import fit_ridge,predict_ridge
from .analysis import metrics,paired,figures


def validate_config(c):
    if c['datasets'] not in (['04C','2018'], ['04C']) or c['models']!=['M0','M6'] or c['seeds']!=list(range(2022,2027)):raise ValueError('fixed training runs required')
    if c['training']!=dict(optimizer='Adam',lr=.001,batch_size=128,max_epochs=100) or c['ridge_alphas']!=[.01,.1,1.,10.,100.]:raise ValueError('training/estimator protocol drift')
    if c['architecture']!=dict(num_levels=1,in_channels=64,kernel_size=3,latent_dim=320,regu_details=.01,regu_approx=.01,snr_embedding_dim=8):raise ValueError('architecture drift')
    if c['torch_num_threads']!=24 or c['evaluation_batch_size']!=64 or c['feature_chunk_size']!=128 or c['device'] not in ['cpu','cuda'] or c['distribution_shift_status']!='PENDING_PHASE12_CHANNEL':raise ValueError('runtime/scope drift')


def check_gates(root,c):
    root=Path(root).resolve()
    try:
        path=safe(root,c['phase10_current']);current=load(path);attempt=safe(root,current['attempt'])
        if current['status']!='COMPLETE' or sha(attempt/'manifest.json')!=current['manifest_sha256'] or load(attempt/'manifest.json')['status']!='COMPLETE':raise ValueError('Phase10 not complete/hash bound')
        closure=manifest_closure(root,attempt/'manifest.json');closure[c['phase10_current']]=sha(path)
        _paths,gates,_deps=upstream_gates(root,load(attempt/'config.json'))
        if load(attempt/'gates.json')!=gates or len(load(attempt/'rows.json'))!=80 or current.get('validation',{}).get('status')!='VALIDATED':raise ValueError('Phase10 scientific matrix/gate incomplete')
        return current,closure
    except (OSError,KeyError,TypeError) as exc:raise ValueError(f'Phase10 gate missing/malformed: {exc}') from exc


def factory(c,data,model):
    result=AWNConditioned(num_classes=data['classes'],conditioning=model,num_snr_bins=len(data['grid']),snr_min_db=min(data['grid']),snr_max_db=max(data['grid']),**c['architecture'])
    paths=install_adapter(result)
    if not paths:raise ValueError('adapter installed no paths')
    return result


def loss_fn(logits,target,regularizers):
    return torch.nn.functional.cross_entropy(logits,target)+sum(regularizers)


def predict(model,data,indices,condition,batch=64):
    loader=torch.utils.data.DataLoader(LazyFrames(data['signals'],data['identity']['labels'],condition,indices,data['grid']),batch_size=batch,shuffle=False,num_workers=0)
    device=next(model.parameters()).device;model.eval();scores=[]
    with torch.no_grad():
        for x,y,db,bins in loader:
            logits=model.forward_batch(x.to(device),snr_db=db.to(device),snr_bin=bins.to(device))[0]
            if not torch.isfinite(logits).all():raise ValueError('nonfinite held-out logits')
            scores.append(logits.cpu().numpy())
    return np.concatenate(scores)


def _save_arrays(path,arrays,verify=False):
    if path.exists():
        with np.load(path,allow_pickle=False) as b:
            if set(b.files)!=set(arrays):raise ValueError('array evidence schema drift')
            for key,value in arrays.items():np.testing.assert_array_equal(b[key],value)
    elif verify:raise ValueError('missing array evidence')
    else:np.savez_compressed(path,**arrays)


def estimator_evidence(folder,data,c,transfer_state,verify=False):
    if not verify:folder.mkdir(exist_ok=True)
    identity=data['identity'];truth=identity['snr_db'];features=bounded_features(data['signals'],c['feature_chunk_size'])
    state,selection=fit_ridge(features,truth,data['ids'],identity['train_indices'],identity['validation_indices'],c['ridge_alphas'])
    estimates=predict_ridge(features,state);transfer=predict_ridge(features,transfer_state)
    arrays=dict(features=features,sample_ids=data['ids'],source_rows=identity['source_rows'],labels=identity['labels'],snr_db=truth,estimated_snr_db=estimates,transfer_10a_snr_db=transfer,**{k:identity[k] for k in ['train_indices','validation_indices','test_indices']})
    _save_arrays(folder/'estimates.npz',arrays,verify)
    errors={partition:{kind:error_strata(truth[idx],values[idx],identity['labels'][idx],dataset=data['name'],channel='upstream unspecified') for kind,values in [('dataset_trainfit',estimates),('10a_transfer_no_refit',transfer)]} for partition,idx in [(p,identity[p+'_indices']) for p in ['train','validation','test']]}
    for name,value in [('state.json',state),('selection.json',selection),('errors.json',errors)]:
        if (folder/name).exists():
            if load(folder/name)!=value:raise ValueError('estimator scientific replay differs')
        elif verify:raise ValueError('missing estimator evidence')
        else:dump(folder/name,value)
    return estimates


def run_one(folder,data,model_id,seed,c,identity,estimates,verify=False):
    i=data['identity'];truth=i['snr_db'];test=i['test_indices']
    if not verify:
        folder.mkdir(parents=True,exist_ok=True)
        def loaders():
            return tuple(torch.utils.data.DataLoader(LazyFrames(data['signals'],i['labels'],truth,i[p+'_indices'],data['grid']),batch_size=128,shuffle=True,num_workers=0) for p in ['train','validation'])
        if not (folder/'training.json').exists():
            outcome=train_model(lambda:factory(c,data,model_id),loaders,folder,seed=seed,training=c['training'],identity=identity,loss_fn=loss_fn,threads=c['torch_num_threads'],device=c['device'],logger=lambda row:print(f"{data['name']}/{model_id}/{seed} epoch {row['epoch']} val {row['validation_accuracy']:.6f}",flush=True))
            # Frozen trainer's class-removal note does not apply to this phase.
            inspection=factory(c,data,model_id)
            outcome['complexity']=dict(parameter_counts=inspection.parameter_counts(),flops='not measured',latency='not measured')
            outcome.update(model=model_id,seed=seed,dataset=data['name'],architecture=c['architecture'],num_classes=data['classes'],num_snr_bins=len(data['grid']),grid=data['grid'],runtime_adapter=ADAPTER_ID,adapter_module_paths=[name for name,module in inspection.named_modules() if isinstance(module,DeterministicReflectionPad1d)],actual_factory='v2.phase11.runner.factory',validation_condition='oracle',protocol_identity=identity)
            dump(folder/'training.json',outcome)
    training=load(folder/'training.json')
    if training['epochs_completed']!=100 or training['model']!=model_id or training['seed']!=seed or training['grid']!=data['grid'] or training['protocol_identity']!=identity or training['architecture']!=c['architecture'] or training['runtime_adapter']!=ADAPTER_ID or training['num_classes']!=data['classes']:raise ValueError('incomplete/wrong training run')
    history=load(folder/'epochs.json')
    if len(history)!=100 or [r['epoch'] for r in history]!=list(range(100)):raise ValueError('incomplete history')
    best=max(range(100),key=lambda epoch:(history[epoch]['validation_accuracy'],epoch))
    snapshot=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    expected_fingerprint=digest(dict(identity=identity,seed=int(seed),training=c['training'],threads=c['torch_num_threads'],device=c['device']))
    if snapshot['next_epoch']!=100 or snapshot['fingerprint']!=expected_fingerprint or snapshot['history']!=history or snapshot['best_epoch']!=best or training['best_epoch']!=best:raise ValueError('snapshot/best-validation binding differs')
    selected_state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    if selected_state.keys()!=snapshot['best_state'].keys():raise ValueError('selected checkpoint schema differs')
    for key,value in selected_state.items():torch.testing.assert_close(value,snapshot['best_state'][key],rtol=0,atol=0)
    model=factory(c,data,model_id).to(c['device']);model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=c['device'],weights_only=True),strict=True)
    if training['adapter_module_paths']!=[name for name,module in model.named_modules() if isinstance(module,DeterministicReflectionPad1d)]:raise ValueError('adapter paths changed')
    rows=[]
    for condition in (['agnostic'] if model_id=='M0' else ['oracle','estimated']):
        raw=estimates if condition=='estimated' else truth;score=predict(model,data,test,raw,c['evaluation_batch_size']);db,bins=dynamic_conditions(raw[test],data['grid'])
        arrays=dict(sample_ids=data['ids'][test],source_rows=i['source_rows'][test],y_true=i['labels'][test],snr_db=truth[test],raw_condition_db=raw[test],applied_condition_db=db,condition_bin=bins,raw_estimate_db=estimates[test],logits=score,y_pred=score.argmax(1))
        _save_arrays(folder/f'{condition}.npz',arrays,verify)
        row=dict(dataset=data['name'],model=model_id,seed=seed,condition=condition,checkpoint_sha256=sha(folder/'checkpoint.pt'),bundle=f"{data['name']}/{model_id}_{seed}/{condition}.npz",bundle_sha256=sha(folder/f'{condition}.npz'),metrics=metrics(i['labels'][test],score,truth[test],data['classes']))
        if (folder/f'{condition}.json').exists():
            if load(folder/f'{condition}.json')!=row:raise ValueError('prediction metric replay differs')
        elif verify:raise ValueError('missing metrics')
        else:dump(folder/f'{condition}.json',row)
        rows.append(row)
    return rows


def dependencies(root,config_path,gate_closure):
    paths=[config_path,root/'scripts/v2/run_phase11.py',root/'docs/superpowers/plans/2026-09-06-phase11-cross-dataset.md',*sorted((root/'v2/phase11').glob('*.py')),root/'scripts/v2/prepare_04c_split.py',root/'v2/phase7/controls.py',root/'v2/phase7/deterministic_padding.py',root/'v2/phase6/audit.py',root/'models/model.py',root/'models/model_conditioning.py',root/'models/lifting.py']
    result={str((root/name).resolve()):h for name,h in gate_closure.items()};result.update({str(p.resolve()):sha(p) for p in paths})
    config=yaml.safe_load(Path(config_path).read_text())
    for name in config['datasets']:
        folder=root/'results/v2/phase11_preflight'/('04c_split' if name=='04C' else '2018_subset');manifest=load(folder/'manifest.json')
        result[str(folder/'manifest.json')]=sha(folder/'manifest.json');result.update(manifest['dependencies']);result.update({str(folder/p):h for p,h in manifest['files'].items()})
    return result


def verify_hashes(root,manifest_path):
    m=load(manifest_path);folder=manifest_path.parent
    if {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p!=manifest_path}!=set(m['files']):raise ValueError('Phase11 closure polluted/incomplete')
    for name,h in m['files'].items():
        if sha(safe(folder,name))!=h:raise ValueError('Phase11 artifact hash mismatch')
    external={(root.parents[1]/'data'/n).resolve() for n in ['RML2016.04c.dat','GOLD_XYZ_OSC.0001_1024.hdf5']}
    for name,h in m['dependencies'].items():
        path=Path(name).resolve()
        if (not path.is_relative_to(root) and path not in external) or sha(path)!=h:raise ValueError('unallowlisted/changed external or local source')
    return m


def validate(attempt,root,allow_candidate=False):
    root=Path(root).resolve();attempt=Path(attempt).resolve();c=load(attempt/'config.json');validate_config(c);torch.set_num_threads(c['torch_num_threads'])
    m=verify_hashes(root,attempt/'manifest.json')
    if m['status'] not in (['COMPLETE','CANDIDATE'] if allow_candidate else ['COMPLETE']):raise ValueError('Phase11 incomplete')
    gate,_=check_gates(root,c)
    if gate!=load(attempt/'gate.json'):raise ValueError('Phase10 pointer changed')
    transfer=load(attempt/'transfer_10a_state.json');rows=[]
    identity=digest(dict(config=c,dependencies=m['dependencies'],environment=load(attempt/'environment.json')))
    p6=validate_composite(root,load(root/c['phase6_current']))
    if transfer!=load(p6/'legacy_identity/snr_ridge_state.json'):raise ValueError('transfer estimator changed')
    for name in c['datasets']:
        data=load_dataset(root,name);estimates=estimator_evidence(attempt/name,data,c,transfer,True)
        for model in c['models']:
            for seed in c['seeds']:rows+=run_one(attempt/name/f'{model}_{seed}',data,model,seed,c,identity,estimates,True)
    expected_rows=15*len(c['datasets']);expected_runs=10*len(c['datasets'])
    if len(rows)!=expected_rows or load(attempt/'rows.json')!=rows or load(attempt/'paired.json')!=paired(rows):raise ValueError('cross-dataset matrix/paired replay differs')
    return dict(status='VALIDATED',training_runs=expected_runs,evaluation_rows=expected_rows,distribution_shift_status='PENDING_PHASE12_CHANNEL')


def execute(config_path,root):
    root=Path(root).resolve();config_path=Path(config_path).resolve();c=yaml.safe_load(config_path.read_text());validate_config(c);torch.set_num_threads(c['torch_num_threads'])
    gate,closure=check_gates(root,c) # Must precede dataset fitting or attempt creation.
    output=safe(root,c['output']);output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.phase11.lock'),timeout=0):
        deps=dependencies(root,config_path,closure);environment=dict(device=c['device'],threads=24,python=sys.version,os=platform.platform(),gpu_name=torch.cuda.get_device_name(0) if c['device']=='cuda' else None,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn']})
        identity=digest(dict(config=c,dependencies=deps,environment=environment));attempt=output/'attempts'/identity[:20];attempt.mkdir(parents=True,exist_ok=True)
        if (attempt/'manifest.json').exists():
            verdict=validate(attempt,root,load(attempt/'manifest.json')['status']=='CANDIDATE');m=load(attempt/'manifest.json')
        else:
            dump(attempt/'config.json',c);dump(attempt/'environment.json',environment);dump(attempt/'gate.json',gate)
            p6=validate_composite(root,load(root/c['phase6_current']));transfer=load(p6/'legacy_identity/snr_ridge_state.json');dump(attempt/'transfer_10a_state.json',transfer);rows=[]
            for name in c['datasets']:
                data=load_dataset(root,name);estimates=estimator_evidence(attempt/name,data,c,transfer)
                for model in c['models']:
                    for seed in c['seeds']:rows+=run_one(attempt/name/f'{model}_{seed}',data,model,seed,c,identity,estimates)
            dump(attempt/'rows.json',rows);dump(attempt/'paired.json',paired(rows));figures(attempt,rows)
            if '2018' in c['datasets']:
                report='# Phase11 cross-dataset evaluation\n\n04C uses the audited unequal-cell fixed split; 2018 uses the prospectively fixed 1024-per-cell subset, never the full source training set. Class labels for 2018 are numeric indices with unverified name mapping. Signals retain original per-frame amplitudes and 1024 samples for 2018; upstream generator/preprocessing provenance remains limited.\n'
            else:
                report='# Phase11 cross-dataset evaluation\n\n04C uses the audited unequal-cell fixed split; signals retain original per-frame amplitudes and 128 samples; class labels use the audited dedup mapping.\n'
            report+=('\nM0/M6 each use five training seeds, true-SNR training and latest-tie oracle validation selection. M6 was selected on earlier 10a validation only. Dataset-trained Ridge uses training-only scaling/fitting and validation-only alpha selection; the 10a coefficient transfer is reported without refit or calibrated-transfer claims. Raw errors are preserved before classifier range clipping.\n\nOverall, every SNR, low <= -8, mid -6 through -2, high >= 0, unequal-cell accuracy, class-balanced accuracy, concentration, dominance, recall and calibration are retained. ')
            report+=('The 2018 high band extends to 30 dB, whereas 04C ends at 18 dB. ' if '2018' in c['datasets'] else '')
            report+='Five-seed intervals quantify network-seed variability on one fixed partition, not independent-dataset uncertainty. Independent channel shift remains pending Phase12. Negative results must not be removed or trigger subset enlargement.\n'
            (attempt/'report.md').write_text(report,encoding='utf-8')
            m=dict(schema_version=1,status='CANDIDATE',distribution_shift_status='PENDING_PHASE12_CHANNEL',dependencies=deps,files={p.relative_to(attempt).as_posix():sha(p) for p in attempt.rglob('*') if p.is_file()})
            dump(attempt/'manifest.json',m);verdict=validate(attempt,root,True)
        m['status']='COMPLETE';dump(attempt/'manifest.json',m)
        atomic_json(output/'current.json',dict(status='COMPLETE',attempt=attempt.relative_to(root).as_posix(),manifest_sha256=sha(attempt/'manifest.json'),validation=verdict))
        return verdict

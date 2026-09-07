"""Prepare/replay OOF noise and run the gated 15-training-run Phase 9 matrix."""
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from filelock import FileLock

from v2.phase8.evidence import dump,load,sha,safe,check_gates,load_estimator,verify_manifest
from v2.phase8.core import conditions,grouped_metrics
from v2.phase8.runner import load_sources,select_stronger
from v2.phase7.controls import digest
from v2.phase7.deterministic_padding import install_adapter,ADAPTER_ID
from .noise import make_folds,crossfit,ConditionNoise
from .trainer import train_model
from .analysis import error_strata,matched_statistics,run_cliffs,figures,cliff_specs


def validate_config(config):
    if config.get('seeds')!=list(range(2022,2027)) or config.get('arms')!=['clean','generic','empirical_snr'] or config.get('model')!='M6':raise ValueError('exactly fifteen new matched training runs required')
    expected=dict(optimizer='Adam',lr=.001,batch_size=128,max_epochs=100,validation_condition='frozen_phase6_estimate',validation_metric='accuracy',checkpoint_tie='latest_epoch')
    if config['training']!=expected or config['runtime_adapter']!=ADAPTER_ID:raise ValueError('locked matched training protocol changed')
    if config['noise']!=dict(generic_std_db=2.,empirical_stratum='true_snr_only',folds=5,fold_seed=2022,alpha_source='frozen_phase6_validation_selected',feature_mode='legacy_identity',condition_rng_stream=9):raise ValueError('locked OOF/noise protocol changed')
    if config['cliff']!=dict(origins=['oracle','estimated'],deltas_db=[-12,-8,-4,-2,-1,-.1,0,.1,1,2,4,8,12],boundary_epsilon_db=.001,models=['M3','M6']):raise ValueError('locked cliff protocol changed')
    if config.get('distribution_shift_status')!='PENDING_PHASE11_12' or config.get('evaluation_batch_size')!=64 or config.get('cpu_threads')!=24:raise ValueError('distribution/runtime scope changed')
    if config['device'] not in ('cpu','cuda'):raise ValueError('unsupported device')


def phase8_gate(root,config):
    root=Path(root).resolve()
    try:
        pointer=safe(root,config['phase8_current']);current=load(pointer)
        if current.get('status')!='COMPLETE':raise ValueError('Phase 8 is not COMPLETE')
        attempt=safe(root,current['attempt']);manifest=attempt/'manifest.json'
        verify_manifest(root,manifest,current['manifest_sha256'])
        # Do not flatten away the bounded Phase 6 scientific receipt or Phase 7 gate.
        p8config=load(attempt/'config.json');check_gates(root,p8config)
        rows=load(attempt/'rows.json')
        if len(rows)!=50 or {(r['model'],r['seed'],r['condition']) for r in rows}!={(m,s,c) for m in ['M3','M6'] for s in range(2022,2027) for c in ['E1','E2','E3_global','E3_within_snr','E3_across_class']}:
            raise ValueError('Phase 8 matrix incomplete')
        return attempt,dict(pointer=config['phase8_current'],current=current,pointer_sha256=sha(pointer))
    except (OSError,KeyError,TypeError) as exc:raise ValueError(f'Phase 8 gate missing or malformed: {exc}') from exc


class FrameRows(torch.utils.data.Dataset):
    def __init__(self,data,condition,indices):self.data=data;self.condition=condition;self.indices=np.asarray(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        row=int(self.indices[index])
        return torch.from_numpy(self.data['signals'][row]),int(self.data['labels'][row]),np.float32(self.condition[row]),row


def build_model(phase3,p3config):
    model=phase3._model(p3config,'M6');paths=install_adapter(model)
    if not paths:raise ValueError('declared deterministic padding adapter replaced no modules')
    return model


def predict(model,data,indices,condition,batch_size):
    selected=next(model.parameters()).device;outputs=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            rows=indices[start:start+batch_size];db,bins=conditions(condition[rows])
            score=model.forward_batch(torch.as_tensor(data['signals'][rows],device=selected),snr_db=torch.as_tensor(db,device=selected),snr_bin=torch.as_tensor(bins,device=selected))[0]
            if not torch.isfinite(score).all():raise ValueError('nonfinite deployment logits')
            outputs.append(score.cpu().numpy())
    return np.concatenate(outputs)


def prepare_oof(attempt,data,split,estimate,alpha,config):
    from v2.phase6.audit import frame_features
    idx=split.train_idx;features=frame_features(data['signals'][idx]);truth=data['snrs'][idx];labels=data['labels'][idx]
    folds=make_folds(truth,labels,5,config['noise']['fold_seed']);arrays=crossfit(features,truth,split.sample_ids[idx],folds,alpha)
    arrays.update(train_indices=idx,features=features,modulation=labels,full_fit_estimates=estimate[idx],full_fit_residuals=estimate[idx]-truth)
    path=attempt/'oof.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            for name,value in arrays.items():np.testing.assert_array_equal(old[name],value)
    else:np.savez_compressed(path,**arrays)
    info=dict(fit_partition='train',alpha_source='frozen_validation_selected_conditional_crossfit',n_rows=len(idx),fold_fit_counts=[len(arrays[f'fit_ids_{f}']) for f in range(5)],pool_stratum='true_snr_only_no_current_label',oof_error=error_strata(truth,arrays['oof_estimates'],labels),full_fit_training_error_diagnostic_only=error_strata(truth,estimate[idx],labels),limitations=['80%-training residuals approximate full-training deployment error.','SNR-only pooling deliberately discards class/feature dependence.','No validation/test residual is fitted, reweighted or pooled.'])
    dump(attempt/'oof_summary.json',info);return arrays


def write_manifest(path,dependencies,metadata):
    path=Path(path)
    if (path/'manifest.json').exists():raise ValueError('immutable manifest already exists')
    manifest=dict(schema_version=1,status='COMPLETE',dependencies=dependencies,files={p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob('*')) if p.is_file()},**metadata)
    dump(path/'manifest.json',manifest)


def verify_run(directory,root,phase3,p3config,data,split,estimate,config,arm,seed):
    m=verify_manifest(root,directory/'manifest.json',sha(directory/'manifest.json'))
    if m.get('arm')!=arm or m.get('seed')!=seed or m.get('epochs_completed')!=100:raise ValueError('training-run matrix mismatch')
    model=build_model(phase3,p3config).to(config['device']);model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location=config['device'],weights_only=True),strict=True)
    history=load(directory/'epochs.json')
    if len(history)!=100 or [r['epoch'] for r in history]!=list(range(100)):raise ValueError('incomplete training history')
    for deployment in (['true','estimated'] if arm=='clean' else ['estimated']):
        with np.load(directory/f'{deployment}.npz',allow_pickle=False) as b:
            np.testing.assert_array_equal(b['sample_ids'],split.sample_ids[split.test_idx]);np.testing.assert_array_equal(b['y_true'],data['labels'][split.test_idx]);np.testing.assert_array_equal(b['snr_db'],data['snrs'][split.test_idx])
            source=data['snrs'] if deployment=='true' else estimate
            score=predict(model,data,split.test_idx,source,64)
            np.testing.assert_array_equal(b['logits'],score);np.testing.assert_array_equal(b['y_pred'],score.argmax(1));np.testing.assert_array_equal(b['raw_condition_db'],source[split.test_idx]);np.testing.assert_array_equal(b['deployed_condition_db'],conditions(source[split.test_idx])[0])
            metrics=grouped_metrics(b['y_true'],score,b['snr_db'])
            if metrics!=load(directory/f'{deployment}_metrics.json'):raise ValueError('saved deployment metrics differ')
    return m


def train_one(root,attempt,arm,seed,phase3,p3config,data,split,estimate,oof,config,dependencies):
    directory=attempt/'runs'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():verify_run(directory,root,phase3,p3config,data,split,estimate,config,arm,seed);return directory
    identity=digest(dict(config=config,sources=dependencies,arm=arm,seed=seed,oof_sha256=sha(attempt/'oof.npz')))
    spec=dict(identity=identity,arm=arm,seed=seed,validation_condition='frozen_phase6_estimate',runtime_adapter=ADAPTER_ID)
    if (directory/'run_spec.json').exists() and load(directory/'run_spec.json')!=spec:raise ValueError('partial training protocol changed')
    dump(directory/'run_spec.json',spec)
    noise=ConditionNoise(arm,seed,pool_snr=oof['true_snr_db'],pool_errors=oof['residuals'],std_db=2.)
    def loaders():
        from phase1_reproduce import make_historical_train_val_loaders
        return make_historical_train_val_loaders(FrameRows(data,data['snrs'],split.train_idx),FrameRows(data,estimate,split.val_idx),train_batch_size=128,validation_batch_size=128)
    def log(row):
        print(f"{arm} seed{seed} epoch{row['epoch']} loss={row['train_loss']:.6f} estimated_val={row['validation_accuracy']:.6f}",flush=True)
    outcome=train_model(lambda:build_model(phase3,p3config),loaders,directory,seed=seed,training=config['training'],identity=identity,loss_fn=phase3._training_loss,condition_noise=noise,threads=config['cpu_threads'],logger=log,device=config['device'])
    dump(directory/'training.json',outcome)
    model=build_model(phase3,p3config).to(config['device']);model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location=config['device'],weights_only=True),strict=True)
    for deployment in (['true','estimated'] if arm=='clean' else ['estimated']):
        source=data['snrs'] if deployment=='true' else estimate;score=predict(model,data,split.test_idx,source,64);idx=split.test_idx
        np.savez_compressed(directory/f'{deployment}.npz',sample_ids=split.sample_ids[idx],y_true=data['labels'][idx],snr_db=data['snrs'][idx],estimated_snr_db=estimate[idx],raw_condition_db=source[idx],deployed_condition_db=conditions(source[idx])[0],logits=score,y_pred=score.argmax(1),seed=np.asarray(seed),arm=np.asarray(arm),deployment=np.asarray(deployment))
        dump(directory/f'{deployment}_metrics.json',grouped_metrics(data['labels'][idx],score,data['snrs'][idx]))
    write_manifest(directory,dependencies,dict(arm=arm,seed=seed,epochs_completed=100,protocol_identity=identity,distribution_shift_status='PENDING_PHASE11_12'))
    return directory


def source_files(root,config_path,p8attempt):
    paths=[config_path,root/'scripts/v2/run_phase9.py',root/'docs/superpowers/plans/2026-09-06-phase9-deployment-noise.md',*sorted((root/'v2/phase9').glob('*.py')),*sorted((root/'v2/phase8').glob('*.py')),root/'v2/phase7/controls.py',root/'v2/phase7/deterministic_padding.py',root/'models/model.py',root/'models/model_conditioning.py',root/'models/lifting.py',root/'scripts/v2/run_phase3.py',root/'scripts/v2/phase1_reproduce.py',root/'configs/v2/phase3.yaml',p8attempt/'manifest.json']
    return {p.relative_to(root).as_posix():sha(p) for p in paths}


def collect_rows(attempt,p8attempt,config):
    rows=[]
    for arm in config['arms']:
        for seed in config['seeds']:
            directory=attempt/'runs'/f'{arm}_seed{seed}'
            for deployment in (['true','estimated'] if arm=='clean' else ['estimated']):
                rows.append(dict(role='matched_primary',arm=arm,seed=seed,deployment=deployment,bundle=(directory/f'{deployment}.npz').relative_to(attempt).as_posix(),metrics=load(directory/f'{deployment}_metrics.json')))
    for item in load(p8attempt/'rows.json'):
        if item['model']=='M6' and item['condition'] in ('E1','E2'):
            rows.append(dict(role='historical_reference_not_matched_training',arm='historical_true',seed=item['seed'],deployment='true' if item['condition']=='E1' else 'estimated',source_bundle=(p8attempt/item['bundle']).as_posix(),source_sha256=sha(p8attempt/item['bundle']),metrics=item['metrics']))
    if len(rows)!=30:raise ValueError('required 20 matched + 10 historical evaluations incomplete')
    return rows


def validate(attempt,root,allow_candidate=False):
    root=Path(root).resolve();attempt=Path(attempt).resolve();config=load(attempt/'config.json');validate_config(config)
    torch.set_num_threads(config['cpu_threads'])
    manifest=verify_manifest(root,attempt/'manifest.json',sha(attempt/'manifest.json'),allowed_status=('CANDIDATE','COMPLETE') if allow_candidate else ('COMPLETE',))
    if manifest.get('distribution_shift_status')!='PENDING_PHASE11_12':raise ValueError('unseen-distribution scope improperly closed')
    p8attempt,gate=phase8_gate(root,config)
    if load(attempt/'upstream.json')!=gate:raise ValueError('Phase 8 pointer changed')
    phase3,p3config,runs,data,split=load_sources(root);p6,_=check_gates(root,load(p8attempt/'config.json'));estimate,estimator=load_estimator(p6,data,split)
    if load(attempt/'model_selection.json')!=select_stronger(runs):raise ValueError('validation-only M6 selection changed')
    if load(attempt/'estimator.json')!=estimator:raise ValueError('deployment estimator changed')
    from v2.phase6.audit import frame_features
    with np.load(attempt/'oof.npz',allow_pickle=False) as saved:
        expected=crossfit(frame_features(data['signals'][split.train_idx]),data['snrs'][split.train_idx],split.sample_ids[split.train_idx],make_folds(data['snrs'][split.train_idx],data['labels'][split.train_idx],5,2022),estimator['selection']['selected_hyperparameter'])
        expected.update(train_indices=split.train_idx,features=frame_features(data['signals'][split.train_idx]),modulation=data['labels'][split.train_idx],full_fit_estimates=estimate[split.train_idx],full_fit_residuals=estimate[split.train_idx]-data['snrs'][split.train_idx])
        if set(saved.files)!=set(expected):raise ValueError('OOF evidence schema mismatch')
        for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
    expected_errors={name:error_strata(data['snrs'][idx],estimate[idx],data['labels'][idx]) for name,idx in [('train',split.train_idx),('validation',split.val_idx),('test',split.test_idx)]}
    if load(attempt/'estimator_error_strata.json')!=expected_errors:raise ValueError('estimator diagnostic replay mismatch')
    for arm in config['arms']:
        for seed in config['seeds']:verify_run(attempt/'runs'/f'{arm}_seed{seed}',root,phase3,p3config,data,split,estimate,config,arm,seed)
    for seed in config['seeds']:
        initial={load(attempt/'runs'/f'{arm}_seed{seed}'/'training.json')['initial_model_sha256'] for arm in config['arms']}
        orders={tuple(r['data_order_sha256'] for r in load(attempt/'runs'/f'{arm}_seed{seed}'/'epochs.json')) for arm in config['arms']}
        if len(initial)!=1 or len(orders)!=1:raise ValueError('matched initialization or minibatch order differs across arms')
    rows=collect_rows(attempt,p8attempt,config)
    if load(attempt/'rows.json')!=rows or load(attempt/'paired_statistics.json')!=matched_statistics(rows):raise ValueError('result/paired replay mismatch')
    cliffs=load(attempt/'cliff_rows.json')
    if len(cliffs)!=545 or {r['id'] for r in cliffs}!={s['id'] for s in cliff_specs(config)}:raise ValueError('cliff matrix incomplete')
    for row in cliffs:
        path=safe(attempt,row['bundle'])
        if sha(path)!=row['bundle_sha256']:raise ValueError('cliff bundle binding mismatch')
        with np.load(path,allow_pickle=False) as b:
            if not np.isfinite(b['logits']).all() or grouped_metrics(b['y_true'],b['logits'],b['snr_db'])!=row['metrics']:raise ValueError('cliff metric replay mismatch')
            np.testing.assert_array_equal(b['sample_ids'],split.sample_ids[split.test_idx]);np.testing.assert_array_equal(b['condition_bin'],conditions(b['raw_condition_db'])[1])
    run_cliffs(attempt,p8attempt,phase3,p3config,runs,data,split,estimate,config,verify_only=True)
    return dict(status='VALIDATED',new_training_runs=15,matched_evaluations=20,historical_references=10,cliff_interventions=545,distribution_shift_status='PENDING_PHASE11_12')


def execute(config_path,root):
    root=Path(root).resolve();config_path=Path(config_path).resolve();config=yaml.safe_load(config_path.read_text(encoding='utf-8'));validate_config(config)
    torch.set_num_threads(config['cpu_threads'])
    p8attempt,gate=phase8_gate(root,config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    output=safe(root,config['output']);output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.phase9.lock'),timeout=0):
        phase3,p3config,runs,data,split=load_sources(root);select_stronger(runs)
        p6,_=check_gates(root,load(p8attempt/'config.json'));estimate,estimator=load_estimator(p6,data,split)
        deps=source_files(root,config_path,p8attempt)
        environment=dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','torch','filelock']},device=config['device'],gpu=torch.cuda.get_device_name(0) if config['device']=='cuda' else None)
        fingerprint=digest(dict(config=config,dependencies=deps,environment=environment,gate=gate))
        attempt=output/'attempts'/fingerprint[:20]
        if (attempt/'manifest.json').exists():
            # Recover a crash between successful candidate validation and publication.
            existing=load(attempt/'manifest.json')
            verdict=validate(attempt,root,allow_candidate=existing.get('status')=='CANDIDATE')
            if existing['status']=='CANDIDATE':
                existing['status']='COMPLETE';dump(attempt/'manifest.json',existing)
            dump(output/'current.tmp.json',dict(status='COMPLETE',attempt=attempt.relative_to(root).as_posix(),manifest_sha256=sha(attempt/'manifest.json'),validation=verdict));(output/'current.tmp.json').replace(output/'current.json')
            return verdict
        attempt.mkdir(parents=True,exist_ok=True)
        dump(attempt/'config.json',config);dump(attempt/'upstream.json',gate);dump(attempt/'estimator.json',estimator);dump(attempt/'environment.json',environment);dump(attempt/'model_selection.json',select_stronger(runs))
        oof=prepare_oof(attempt,data,split,estimate,estimator['selection']['selected_hyperparameter'],config)
        deps[(attempt/'oof.npz').relative_to(root).as_posix()]=sha(attempt/'oof.npz')
        for arm in config['arms']:
            for seed in config['seeds']:train_one(root,attempt,arm,seed,phase3,p3config,data,split,estimate,oof,config,deps)
        torch.set_num_threads(24)
        rows=collect_rows(attempt,p8attempt,config);dump(attempt/'rows.json',rows);dump(attempt/'paired_statistics.json',matched_statistics(rows))
        dump(attempt/'estimator_error_strata.json',dict(train=error_strata(data['snrs'][split.train_idx],estimate[split.train_idx],data['labels'][split.train_idx]),validation=error_strata(data['snrs'][split.val_idx],estimate[split.val_idx],data['labels'][split.val_idx]),test=error_strata(data['snrs'][split.test_idx],estimate[split.test_idx],data['labels'][split.test_idx])))
        cliffs=run_cliffs(attempt,p8attempt,phase3,p3config,runs,data,split,estimate,config);figures(attempt,rows,cliffs)
        if any(sha(root/name)!=h for name,h in deps.items()):raise ValueError('Phase 9 source changed during execution')
        manifest=dict(schema_version=1,status='CANDIDATE',distribution_shift_status='PENDING_PHASE11_12',dependencies=deps,files={p.relative_to(attempt).as_posix():sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()})
        dump(attempt/'manifest.json',manifest);verdict=validate(attempt,root,allow_candidate=True);manifest['status']='COMPLETE';dump(attempt/'manifest.json',manifest)
        dump(output/'current.tmp.json',dict(status='COMPLETE',attempt=attempt.relative_to(root).as_posix(),manifest_sha256=sha(attempt/'manifest.json'),validation=verdict));(output/'current.tmp.json').replace(output/'current.json')
        return verdict

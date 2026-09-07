"""Gated, immutable E1-E4 execution and evidence replay."""
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import sys
import traceback
import uuid

import numpy as np
import torch
import yaml
from scipy.stats import t

from .core import cache_and_replay,conditions,donor_indices,e4_logits,estimate_metrics,fit_e4,grouped_metrics,head_logits
from .evidence import check_gates,dump,load,load_estimator,safe,sha,verify_manifest


def validate_config(config):
    if config['seeds']!=list(range(2022,2027)) or config['models']!=['M3','M6'] or config['conditions']!=['E1','E2','E3_global','E3_within_snr','E3_across_class']:
        raise ValueError('unregistered Phase 8 matrix')
    if config.get('batch_size')!=64 or config.get('torch_num_threads')!=24 or config.get('device')!='cpu':
        raise ValueError('registered FP32 replay requires CPU batch64 and torch_num_threads24')


def load_sources(root):
    from v2.phase3_analysis import _load_registered_runs,_load_fixed_data,_load_phase3_contracts
    phase3=_load_phase3_contracts();config=phase3.load_phase3_config(root/'configs/v2/phase3.yaml')
    if config['evaluation_batch_size']!=64:raise ValueError('frozen Phase 3 evaluation batch size changed')
    runs,missing,errors=_load_registered_runs(root,(root/'manifest.json').resolve())
    if missing or errors:raise ValueError(f'Phase 3 source invalid: {missing} {errors}')
    dataset,split=_load_fixed_data(root,config,phase3)
    return phase3,config,runs,dataset,split


def select_stronger(runs):
    candidates=[]
    for model in ('M4','M5','M6','M7'):
        results=[runs[(model,seed)]['result'] for seed in range(2022,2027)]
        scores=[r['training']['best_val_accuracy'] for r in results]
        if not np.isfinite(scores).all():raise ValueError('invalid validation-only model selection scores')
        candidates.append(dict(model_id=model,seeds=list(range(2022,2027)),scores=scores,mean=float(np.mean(scores)),additional_parameters=float(np.mean([r['complexity']['additional_conditioner_parameters'] for r in results]))))
    winner=min(candidates,key=lambda c:(-c['mean'],c['additional_parameters'],c['model_id']))['model_id']
    if winner!='M6':raise ValueError('preregistered M6 is not validation-selected; protocol review required')
    return dict(selected=winner,candidates=candidates,selection_partition='validation',metric='training.best_val_accuracy')


def get_model(phase3,config,run,model_id,device):
    model=phase3._model(config,model_id).to(device)
    model.load_state_dict(phase3._load_state(run['paths']['checkpoint'],device),strict=True)
    return model.eval()


def paired(rows,config):
    lookup={(r['model'],r['seed'],r['condition']):r for r in rows};output=[]
    for model in config['models']:
        for condition in config['conditions'][1:]:
            for baseline in (['E1','E2'] if condition.startswith('E3') else ['E1']):
                for metric in ('accuracy','balanced_accuracy','concentration','prediction_entropy'):
                    differences=np.array([lookup[(model,s,condition)]['metrics']['overall'][metric]-lookup[(model,s,baseline)]['metrics']['overall'][metric] for s in config['seeds']])
                    half=float(t.ppf(.975,4)*differences.std(ddof=1)/np.sqrt(5))
                    output.append(dict(model=model,treatment=condition,baseline=baseline,metric=metric,seeds=config['seeds'],differences=differences.tolist(),mean=float(differences.mean()),ci95=[float(differences.mean()-half),float(differences.mean()+half)],n_pairs=5))
    return output


def e4_diagnostic(y,logits,train_y,margin):
    n=len(y);accuracy=float(np.mean(logits.argmax(1)==y));z=1.959963984540054
    center=(accuracy+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    majority=int(np.bincount(train_y,minlength=11).argmax())
    baseline=float(np.mean(y==majority));threshold=max(1/11,baseline)+margin
    return dict(status='SIDE_CHANNEL_DETECTED' if center-half>threshold else 'NO_CLEAR_SIDE_CHANNEL_BY_PREREGISTERED_RULE',n_independent_fits=1,accuracy=accuracy,accuracy_wilson95=[float(center-half),float(center+half)],chance=1/11,train_majority_class=majority,train_majority_test_accuracy=baseline,threshold=threshold,requires_phase12_unseen_channel=bool(center-half>threshold),limitation='Wilson interval treats examples as independent; waveform dependence may reduce effective sample size.')


def figures(attempt,rows,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for axis,metric in zip(axes,['accuracy','balanced_accuracy','concentration']):
        for model,color,offset in [('M3','#0072B2',-.1),('M6','#D55E00',.1)]:
            values=np.array([[next(r for r in rows if (r['model'],r['seed'],r['condition'])==(model,s,c))['metrics']['overall'][metric] for s in config['seeds']] for c in config['conditions']])
            axis.errorbar(np.arange(5)+offset,values.mean(1),yerr=values.std(1,ddof=1),fmt='o-',color=color,label=model,capsize=3)
        axis.set_xticks(np.arange(5),config['conditions'],rotation=35,ha='right');axis.set_ylabel(metric);axis.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Side-channel controls: five matched checkpoint seeds; mean +/- SD');fig.tight_layout()
    fig.savefig(attempt/'sidechannel_controls.png',dpi=300);fig.savefig(attempt/'sidechannel_controls.pdf');plt.close(fig)
    dump(attempt/'figure_sources.json',{'rows.json':sha(attempt/'rows.json'),'plots':['sidechannel_controls.png','sidechannel_controls.pdf']})


def validate(attempt,root,allow_candidate=False):
    root=Path(root).resolve();attempt=Path(attempt).resolve();manifest=load(attempt/'manifest.json')
    verify_manifest(root,attempt/'manifest.json',sha(attempt/'manifest.json'),allowed_status=('CANDIDATE','COMPLETE') if allow_candidate else ('COMPLETE',))
    actual={p.relative_to(attempt).as_posix() for p in attempt.rglob('*') if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}:raise ValueError('unregistered attempt pollution')
    config=load(attempt/'config.json');validate_config(config);torch.set_num_threads(config['torch_num_threads'])
    phase6,gates=check_gates(root,config)
    if load(attempt/'gates.json')!=gates:raise ValueError('upstream gate changed')
    rows=load(attempt/'rows.json');planned={(m,s,c) for m in config['models'] for s in config['seeds'] for c in config['conditions']}
    keys=[(r['model'],r['seed'],r['condition']) for r in rows]
    if len(keys)!=50 or set(keys)!=planned:raise ValueError('incomplete checkpoint-condition matrix')
    phase3,p3config,runs,dataset,split=load_sources(root)
    if load(attempt/'model_selection.json')!=select_stronger(runs):raise ValueError('model-selection replay mismatch')
    raw,estimator_evidence=load_estimator(phase6,dataset,split)
    if load(attempt/'estimator.json')!=estimator_evidence:raise ValueError('estimator source mismatch')
    with np.load(attempt/'all_estimates.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sample_ids'],split.sample_ids)
        np.testing.assert_allclose(saved['estimated_snr_db'],raw,rtol=1e-10,atol=1e-10)
        for name,indices in [('train',split.train_idx),('validation',split.val_idx),('test',split.test_idx)]:np.testing.assert_array_equal(saved[name+'_indices'],indices)
    lookup=dict(zip(keys,rows));test=split.test_idx;y=dataset['labels'][test];truth=dataset['snrs'][test]
    for model_id in config['models']:
        for seed in config['seeds']:
            run=runs[(model_id,seed)];model=get_model(phase3,p3config,run,model_id,config['device'])
            with np.load(attempt/'caches'/f'{model_id}_{seed}.npz',allow_pickle=False) as b:cache={k:b[k] for k in b.files}
            np.testing.assert_array_equal(cache['sample_ids'],split.sample_ids[test])
            if cache['checkpoint_sha256'].item()!=sha(run['paths']['checkpoint']):raise ValueError('cache checkpoint binding mismatch')
            with np.load(run['paths']['predictions'],allow_pickle=False) as b:registered=b['logits']
            # Recompute the raw pooled representation, not merely saved head logits.
            fresh,_score,_evidence=cache_and_replay(model,dataset['signals'][test],truth,registered,config['batch_size'])
            np.testing.assert_allclose(cache['raw_pooled_features'],fresh,rtol=1e-6,atol=1e-7)
            for condition in config['conditions']:
                row=lookup[(model_id,seed,condition)]
                with np.load(safe(attempt,row['bundle']),allow_pickle=False) as b:a={k:b[k] for k in b.files}
                if any(not np.isfinite(v).all() for v in a.values() if np.issubdtype(v.dtype,np.number)):raise ValueError('nonfinite condition evidence')
                for name,expected in [('seed',seed),('model_id',model_id),('condition',condition),('checkpoint_sha256',sha(run['paths']['checkpoint']))]:
                    if a[name].item()!=expected:raise ValueError('condition identity binding mismatch')
                expected_donors=np.arange(len(test)) if condition in ('E1','E2') else donor_indices(truth,y,condition.removeprefix('E3_'),seed)
                np.testing.assert_array_equal(a['donor_indices'],expected_donors);np.testing.assert_array_equal(a['sample_ids'],split.sample_ids[test])
                np.testing.assert_array_equal(a['donor_sample_ids'],split.sample_ids[test][expected_donors])
                np.testing.assert_array_equal(a['y_true'],y);np.testing.assert_array_equal(a['snr_db'],truth)
                source=truth if condition=='E1' else raw[test][expected_donors]
                np.testing.assert_array_equal(a['raw_condition_db'],source)
                db,bins=conditions(source);np.testing.assert_array_equal(a['deployed_condition_db'],db);np.testing.assert_array_equal(a['condition_bin'],bins)
                score=head_logits(model,fresh,source,config['batch_size'])
                np.testing.assert_allclose(a['logits'],score,rtol=1e-6,atol=1e-7);np.testing.assert_array_equal(a['y_pred'],score.argmax(1))
                if row['metrics']!=grouped_metrics(y,a['logits'],truth):raise ValueError('condition metric replay mismatch')
                if row['raw_error']!=estimate_metrics(truth,source,y) or row['deployed_error']!=estimate_metrics(truth,db,y):raise ValueError('condition error replay mismatch')
                if row['clipped_fraction']!=float(np.mean((source<-20)|(source>18))):raise ValueError('clipping rate replay mismatch')
    if load(attempt/'paired_statistics.json')!=paired(rows,config):raise ValueError('paired statistics replay mismatch')
    e4=load(attempt/'e4_model.json');selected,selection=fit_e4(raw[split.train_idx],dataset['labels'][split.train_idx],raw[split.val_idx],dataset['labels'][split.val_idx],config['e4'])
    if e4!=selected or load(attempt/'e4_selection.json')!=selection:raise ValueError('E4 train-only replay mismatch')
    with np.load(attempt/'e4_predictions.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(b['sample_ids'],split.sample_ids[test]);np.testing.assert_array_equal(b['train_sample_ids'],split.sample_ids[split.train_idx]);np.testing.assert_array_equal(b['validation_sample_ids'],split.sample_ids[split.val_idx])
        score=e4_logits(raw[test],e4);np.testing.assert_allclose(b['logits'],score,rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(b['y_true'],y);np.testing.assert_array_equal(b['snr_db'],truth);np.testing.assert_array_equal(b['y_pred'],score.argmax(1));np.testing.assert_allclose(b['estimated_snr_db'],raw[test],rtol=1e-10,atol=1e-10)
        if load(attempt/'e4_metrics.json')!=grouped_metrics(y,score,truth):raise ValueError('E4 metric replay mismatch')
        if load(attempt/'e4_diagnostic.json')!=e4_diagnostic(y,score,dataset['labels'][split.train_idx],config['e4']['sidechannel_margin']):raise ValueError('E4 flag replay mismatch')
    if load(attempt/'figure_sources.json')['rows.json']!=sha(attempt/'rows.json'):raise ValueError('figure source mismatch')
    return dict(status='VALIDATED',checkpoint_conditions=50,raw_caches=10,e4_independent_fits=1,files=len(manifest['files']))


def execute(config_path,root):
    root=Path(root).resolve();config_path=Path(config_path).resolve();config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    validate_config(config);torch.set_num_threads(config['torch_num_threads'])
    phase6,gates=check_gates(root,config)
    phase3,p3config,runs,dataset,split=load_sources(root);selection=select_stronger(runs)
    raw,estimator=load_estimator(phase6,dataset,split)
    output=root/'results/v2/phase8_sidechannel';attempt=output/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);attempt.mkdir(parents=True)
    sources=[config_path,root/'docs/phase8_design.md',root/'scripts/v2/run_phase8.py',*sorted((root/'v2/phase8').glob('*.py')),root/'models/model_conditioning.py',root/'models/model.py',root/'models/lifting.py',root/'v2/phase3_analysis.py',root/'scripts/v2/run_phase3.py',root/'v2/phase6/audit.py',root/'v2/phase6/composite.py',root/'manifest.json',root/'results/v2/phase3/phase3_ledger.json']
    dependencies={p.relative_to(root).as_posix():sha(p) for p in sources}
    for model_id in config['models']:
        for seed in config['seeds']:
            run=runs[(model_id,seed)]
            for p in [run['result_path'],*run['paths'].values()]:dependencies[p.relative_to(root).as_posix()]=sha(p)
    rows=[]
    try:
        dump(attempt/'config.json',config);dump(attempt/'gates.json',gates);dump(attempt/'model_selection.json',selection);dump(attempt/'estimator.json',estimator)
        dump(attempt/'environment.json',dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','torch','matplotlib']}))
        np.savez_compressed(attempt/'all_estimates.npz',sample_ids=split.sample_ids,estimated_snr_db=raw,train_indices=split.train_idx,validation_indices=split.val_idx,test_indices=split.test_idx)
        e4,e4selection=fit_e4(raw[split.train_idx],dataset['labels'][split.train_idx],raw[split.val_idx],dataset['labels'][split.val_idx],config['e4'])
        if e4['classes']!=list(range(11)):raise ValueError('E4 training lacks planned modulation classes')
        dump(attempt/'e4_model.json',e4);dump(attempt/'e4_selection.json',e4selection)
        test=split.test_idx;y=dataset['labels'][test];truth=dataset['snrs'][test];ids=split.sample_ids[test];score=e4_logits(raw[test],e4)
        np.savez_compressed(attempt/'e4_predictions.npz',sample_ids=ids,train_sample_ids=split.sample_ids[split.train_idx],validation_sample_ids=split.sample_ids[split.val_idx],y_true=y,snr_db=truth,estimated_snr_db=raw[test],logits=score,y_pred=score.argmax(1))
        dump(attempt/'e4_metrics.json',grouped_metrics(y,score,truth));dump(attempt/'e4_diagnostic.json',e4_diagnostic(y,score,dataset['labels'][split.train_idx],config['e4']['sidechannel_margin']))
        (attempt/'caches').mkdir();(attempt/'conditions').mkdir()
        for model_id in config['models']:
            for seed in config['seeds']:
                run=runs[(model_id,seed)];model=get_model(phase3,p3config,run,model_id,config['device'])
                with np.load(run['paths']['predictions'],allow_pickle=False) as b:registered=b['logits'];np.testing.assert_array_equal(b['sample_ids'],ids)
                cache,e1,replay=cache_and_replay(model,dataset['signals'][test],truth,registered,config['batch_size'])
                np.savez_compressed(attempt/'caches'/f'{model_id}_{seed}.npz',raw_pooled_features=cache,sample_ids=ids,checkpoint_sha256=np.asarray(sha(run['paths']['checkpoint'])))
                for condition in config['conditions']:
                    donors=np.arange(len(test)) if condition in ('E1','E2') else donor_indices(truth,y,condition.removeprefix('E3_'),seed)
                    source=truth if condition=='E1' else raw[test][donors];db,bins=conditions(source)
                    logits=e1 if condition=='E1' else head_logits(model,cache,source,config['batch_size'])
                    path=attempt/'conditions'/f'{model_id}_{seed}_{condition}.npz'
                    np.savez_compressed(path,sample_ids=ids,donor_sample_ids=ids[donors],donor_indices=donors,y_true=y,y_pred=logits.argmax(1),snr_db=truth,raw_condition_db=source,deployed_condition_db=db,condition_bin=bins,logits=logits,seed=np.asarray(seed),model_id=np.asarray(model_id),condition=np.asarray(condition),checkpoint_sha256=np.asarray(sha(run['paths']['checkpoint'])))
                    rows.append(dict(model=model_id,seed=seed,condition=condition,bundle=path.relative_to(attempt).as_posix(),metrics=grouped_metrics(y,logits,truth),raw_error=estimate_metrics(truth,source,y),deployed_error=estimate_metrics(truth,db,y),clipped_fraction=float(np.mean((source<-20)|(source>18))),replay=replay,deployable=condition=='E2',role='oracle' if condition=='E1' else 'diagnostic' if condition.startswith('E3') else 'dataset_trained_estimator'))
                print(f'completed {model_id} seed {seed}: {len(rows)}/50 conditions',flush=True)
        dump(attempt/'rows.json',rows);dump(attempt/'paired_statistics.json',paired(rows,config));figures(attempt,rows,config)
        dump(attempt/'summary.json',dict(status='COMPUTED',checkpoint_conditions=50,e4_independent_fits=1,phase12_unseen_channel_required=load(attempt/'e4_diagnostic.json')['requires_phase12_unseen_channel'],limits=['E3 uses true strata and is diagnostic, not deployable.','Across-class donors exclude same-class examples; this is not proof of independence.','E4 is one shared fixed-split fit, not five independent fits.','In-dataset estimates do not establish unseen-channel performance.']))
        if any(sha(root/name)!=digest for name,digest in dependencies.items()):raise ValueError('source changed during Phase 8')
        manifest=dict(status='CANDIDATE',schema_version=1,dependencies=dependencies,files={p.relative_to(attempt).as_posix():sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()})
        dump(attempt/'manifest.json',manifest)
        validation=validate(attempt,root,allow_candidate=True)
        manifest['status']='COMPLETE';dump(attempt/'manifest.json',manifest)
        dump(output/'current.tmp.json',dict(status='COMPLETE',attempt=attempt.relative_to(root).as_posix(),manifest_sha256=sha(attempt/'manifest.json'),validation=validation));(output/'current.tmp.json').replace(output/'current.json')
        print(str(attempt),flush=True)
        return validation
    except Exception:
        dump(attempt/'failure.json',dict(status='FAILED',completed_conditions=len(rows),traceback=traceback.format_exc()))
        raise

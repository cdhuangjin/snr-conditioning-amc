"""Immutable, fully replayable Phase 5 experiment generations."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback
import uuid

import numpy as np
from scipy.special import softmax
from scipy.stats import t
import yaml

from .core import generate, bayes_scores, decide, fit_erm, erm_scores, metrics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''): digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def close(actual, expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)


def summarize(rows, settings):
    names = ['accuracy','balanced_accuracy','concentration','prediction_entropy','mean_posterior_entropy','nll']
    output = []
    for setting in settings:
        for classifier in ('bayes','erm'):
            selected = sorted([r for r in rows if r['setting_id']==setting['id'] and r['classifier']==classifier],key=lambda r:r['seed'])
            values = {name:np.array([r['metrics'][name] for r in selected]) for name in names}
            output.append(dict(setting_id=setting['id'],family=setting['family'],classifier=classifier,
                               seeds=[r['seed'] for r in selected],metrics={k:dict(values=v.tolist(),mean=float(v.mean()),std=float(v.std(ddof=1))) for k,v in values.items()}))
    return output


def contrasts(rows):
    pairs = [('sym_noise16','sym_noise0.25'),('asym4_noise4','asym1_noise4'),
             ('prior_rotation0','sym_noise16'),('global_gain4','global_gain1'),
             ('signal_gain4','signal_gain1'),('separation4','separation1'),
             ('temperature4','temperature1'),('zero_tie_random','zero_tie_first')]
    index={(r['setting_id'],r['classifier'],r['seed']):r['metrics'] for r in rows}
    output=[]
    for treatment,baseline in pairs:
        for classifier in ('bayes','erm'):
            for metric in ('concentration','balanced_accuracy','prediction_entropy','mean_posterior_entropy'):
                diff=np.array([index[(treatment,classifier,s)][metric]-index[(baseline,classifier,s)][metric] for s in range(2022,2027)])
                half=float(t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5))
                output.append(dict(treatment=treatment,baseline=baseline,classifier=classifier,metric=metric,seeds=list(range(2022,2027)),differences=diff.tolist(),mean=float(diff.mean()),ci95=[float(diff.mean()-half),float(diff.mean()+half)],n_pairs=5))
    return output


def figures(attempt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=load(attempt/'summary.json')
    settings=load(attempt/'config.json')['settings']
    ids=[s['id'] for s in settings]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    for ax,metric in zip(axes,['concentration','balanced_accuracy','prediction_entropy']):
        for classifier,color,offset in [('bayes','#0072B2',-.13),('erm','#D55E00',.13)]:
            selected=[next(r for r in summary['by_setting'] if r['setting_id']==sid and r['classifier']==classifier) for sid in ids]
            x=np.arange(len(ids))+offset
            ax.errorbar(x,[r['metrics'][metric]['mean'] for r in selected],yerr=[r['metrics'][metric]['std'] for r in selected],fmt='o',ms=3,capsize=2,color=color,label=classifier)
            for i,r in enumerate(selected): ax.scatter(np.full(5,x[i]),r['metrics'][metric]['values'],s=5,alpha=.3,color=color)
        ax.set(ylabel=metric,ylim=(-.03,1.04));ax.grid(axis='y',alpha=.2)
    axes[0].legend();axes[0].set_title('Phase 5 controlled slices: all five seeds, mean +/- sample SD')
    axes[-1].set_xticks(np.arange(len(ids)),ids,rotation=65,ha='right')
    fig.tight_layout()
    fig.savefig(attempt/'all_controls.png',dpi=300);fig.savefig(attempt/'all_controls.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for classifier,color in [('bayes','#0072B2'),('erm','#D55E00')]:
        for ax,metric in zip(axes,['concentration','mean_posterior_entropy']):
            selected=[next(r for r in summary['by_setting'] if r['setting_id']==f'temperature{v}' and r['classifier']==classifier) for v in [.25,1,4]]
            ax.plot([.25,1,4],[r['metrics'][metric]['mean'] for r in selected],'o-',color=color,label=classifier)
            ax.set(xscale='log',xlabel='Positive softmax temperature',ylabel=metric,ylim=(0,1.05))
    axes[0].legend();fig.tight_layout();fig.savefig(attempt/'temperature_control.png',dpi=300);fig.savefig(attempt/'temperature_control.pdf');plt.close(fig)
    dump(attempt/'figure_sources.json',dict(numeric_source='summary.json',sha256=sha(attempt/'summary.json'),figures=['all_controls.png','all_controls.pdf','temperature_control.png','temperature_control.pdf'],script='v2/phase5/runner.py'))


def validate_attempt(attempt, repository_root=None):
    attempt=Path(attempt);manifest=load(attempt/'manifest.json')
    for name,digest in manifest['files'].items():
        path=(attempt/name).resolve()
        if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha(path)!=digest:
            raise ValueError(f'artifact hash mismatch: {name}')
    actual={p.relative_to(attempt).as_posix() for p in attempt.rglob('*') if p.is_file()}
    if actual != set(manifest['files'])|{'manifest.json'}: raise ValueError('unregistered artifact pollution')
    if repository_root is not None:
        for name,digest in manifest['source_files'].items():
            if sha(Path(repository_root)/name)!=digest: raise ValueError(f'source hash mismatch: {name}')
    config=load(attempt/'config.json');rows=load(attempt/'rows.json')
    if config['seeds']!=list(range(2022,2027)) or len(config['settings'])!=35: raise ValueError('preregistered matrix changed')
    planned={(s['id'],seed,c) for s in config['settings'] for seed in config['seeds'] for c in ('bayes','erm')}
    keys=[(r['setting_id'],r['seed'],r['classifier']) for r in rows]
    if len(keys)!=len(planned) or set(keys)!=planned: raise ValueError('incomplete or duplicate matrix')
    index={key:r for key,r in zip(keys,rows)}
    for setting in config['settings']:
        for seed in config['seeds']:
            bundle_path=attempt/'runs'/f"{setting['id']}_seed{seed}.npz"
            with np.load(bundle_path,allow_pickle=False) as bundle: arrays={k:bundle[k] for k in bundle.files}
            if any(not np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype,np.number)):
                raise ValueError('nonfinite numeric evidence')
            generated=generate(setting,seed,config['n_train'],config['n_test'])
            for key,value in generated.items(): np.testing.assert_array_equal(arrays[key],value)
            if set(arrays['train_ids']) & set(arrays['test_ids']): raise ValueError('train/test ID overlap')
            fitted=fit_erm(arrays['train_x'],arrays['train_y'],config['erm'])
            if not fitted['converged']: raise ValueError('ERM did not converge on replay')
            for key in ('weights','mean','scale'): close(arrays['erm_'+key],fitted[key])
            for classifier,score in [('bayes',bayes_scores(arrays['test_x'],generated)),('erm',erm_scores(arrays['test_x'],fitted))]:
                close(arrays[classifier+'_scores'],score)
                probabilities=softmax(score/setting['temperature'],axis=1)
                predictions=decide(score,setting['tie_rule'],seed)
                close(arrays[classifier+'_probabilities'],probabilities)
                np.testing.assert_array_equal(arrays[classifier+'_predictions'],predictions)
                row=index[(setting['id'],seed,classifier)]
                if row['metrics']!=metrics(arrays['test_y'],predictions,probabilities): raise ValueError('metric replay mismatch')
                if row['bundle_sha256']!=sha(bundle_path): raise ValueError('row bundle binding mismatch')
    summary=load(attempt/'summary.json')
    if summary['by_setting']!=summarize(rows,config['settings']): raise ValueError('summary replay mismatch')
    if load(attempt/'paired_contrasts.json')!=contrasts(rows): raise ValueError('contrast replay mismatch')
    source=load(attempt/'figure_sources.json')
    if source['sha256']!=sha(attempt/source['numeric_source']): raise ValueError('figure source mismatch')
    return dict(status='VALIDATED',datasets=len(planned)//2,evaluation_rows=len(rows),files=len(manifest['files']))


def run(config_path, root):
    root=Path(root).resolve();config_path=Path(config_path).resolve()
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output=root/'results/v2/phase5_toy';output.mkdir(parents=True,exist_ok=True)
    attempt=output/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True);(attempt/'runs').mkdir()
    files=['configs/v2/phase5.yaml','scripts/v2/run_phase5.py','docs/phase5_design.md',*sorted(p.relative_to(root).as_posix() for p in (root/'v2/phase5').glob('*.py'))]
    identity={name:sha(root/name) for name in files}
    dump(attempt/'config.json',config)
    dump(attempt/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','PyYAML')}))
    rows=[]
    try:
        for setting in config['settings']:
            for seed in config['seeds']:
                data=generate(setting,seed,config['n_train'],config['n_test'])
                model=fit_erm(data['train_x'],data['train_y'],config['erm'])
                if not model['converged']: raise RuntimeError(f"ERM nonconvergence: {setting['id']} {seed}: {model['message']}")
                arrays=data | {'erm_'+key:model[key] for key in ('weights','mean','scale')}
                run_rows=[]
                for classifier,score in [('bayes',bayes_scores(data['test_x'],data)),('erm',erm_scores(data['test_x'],model))]:
                    p=softmax(score/setting['temperature'],axis=1);pred=decide(score,setting['tie_rule'],seed)
                    arrays.update({classifier+'_scores':score,classifier+'_probabilities':p,classifier+'_predictions':pred})
                    run_rows.append(dict(setting_id=setting['id'],family=setting['family'],seed=seed,classifier=classifier,metrics=metrics(data['test_y'],pred,p),exact_tie_fraction=float(np.mean(np.sum(score==score.max(1,keepdims=True),axis=1)>1)),optimizer={k:v for k,v in model.items() if k not in ('weights','mean','scale')} if classifier=='erm' else None))
                path=attempt/'runs'/f"{setting['id']}_seed{seed}.npz";np.savez_compressed(path,**arrays)
                for row in run_rows: row['bundle']=path.relative_to(attempt).as_posix();row['bundle_sha256']=sha(path)
                rows.extend(run_rows)
            print(f"completed {setting['id']}: {len(rows)}/350 evaluation rows",flush=True)
        dump(attempt/'rows.json',rows)
        dump(attempt/'summary.json',dict(status='COMPLETE',dataset_count=175,evaluation_rows=350,by_setting=summarize(rows,config['settings']),limits=['Controlled toy slices are not universal inevitability evidence.','Linear ERM is misspecified for unequal class variances.','Temperature is confidence sharpening, not increased separability.','Five seeds quantify finite-sample variation, not generality over distributions.']))
        dump(attempt/'paired_contrasts.json',contrasts(rows));figures(attempt)
        (attempt/'report.md').write_text('# Phase 5 toy counterexamples\n\nAll 35 preregistered settings and five seeds are retained: 175 datasets, 350 Bayes/ERM evaluation rows. Metrics and fitted parameters are numerically replayable. See summary.json and paired_contrasts.json for complete seed-level results and intervals.\n\nSymmetry alone does not imply hard prediction concentration. At zero information with uniform priors, deterministic label tie-breaking and random tie-breaking can yield different hard prediction concentrations under the same posterior. Positive temperature changes posterior confidence without changing score ordering. Unequal priors or feature variances are distribution assumptions, not a universal noise law. Linear ERM and Bayes are separate baselines; these toys do not establish the mechanism of any radio dataset or neural checkpoint.\n',encoding='utf-8')
        if any(sha(root/name)!=digest for name,digest in identity.items()): raise RuntimeError('source changed during run')
        dump(attempt/'manifest.json',dict(schema_version=1,status='COMPLETE',source_files=identity,files={p.relative_to(attempt).as_posix():sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()}))
        validation=validate_attempt(attempt,root)
        dump(output/'current.tmp.json',dict(status='COMPLETE',attempt=attempt.relative_to(root).as_posix(),manifest_sha256=sha(attempt/'manifest.json'),validation=validation))
        (output/'current.tmp.json').replace(output/'current.json')
        print(json.dumps(dict(attempt=str(attempt),**validation)),flush=True)
        return 0
    except Exception:
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        dump(attempt/'failed.json',dict(status='FAILED',completed_rows=len(rows)))
        raise

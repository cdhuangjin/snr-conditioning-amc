"""Phase 7 M0 sensitivity controls; run only after independent review and Phase 6 gate."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import traceback
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,Path(__file__).resolve().parent):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from analyze_phase7 import validate_gate,gate_source_paths,load_phase4
from v2.phase7.controls import (CONTROLS,SEEDS,prepare_control,balanced_control,remap_control,train_model,
    digest,file_hash,atomic_json,write_manifest,verify_control_manifest,summarize_matrix)


def source_context(gate_path,config_path):
    import torch
    gate_manifest=validate_gate(gate_path)
    p4=load_phase4();cfg4=p4.load_phase4_config(ROOT/'configs/v2/phase4.yaml')
    p4root=ROOT/cfg4['outputs']['root'];pointer=p4root/'current.json'
    point=json.loads(pointer.read_text(encoding='utf-8'));attempt=(p4root/point['attempt']).resolve()
    if not attempt.is_relative_to(p4root.resolve()):raise ValueError('Phase 4 path escape')
    manifest=attempt/'manifest.json'
    if file_hash(manifest)!=point['manifest_sha256']:raise ValueError('Phase 4 pointer mismatch')
    m=p4.validate_phase4_manifest(manifest,attempt,repository_root=ROOT)
    if m['status']!='COMPLETE':raise ValueError('Phase 4 incomplete')
    context=p4._strict_phase3_contract(ROOT,cfg4)
    runs=p4.audit_phase3_registry(ROOT,cfg4,_strict_context=context)
    sources=[Path(__file__),ROOT/'v2/phase7/controls.py',ROOT/'v2/phase7/deterministic_padding.py',ROOT/'scripts/v2/analyze_phase7.py',ROOT/'v2/phase7/diagnostics.py',
        config_path,gate_path,gate_manifest,*gate_source_paths(gate_manifest),pointer,manifest,*[ROOT/name for name in m['dependencies']]]
    for r in runs:
        if r.model_id=='M0':sources.extend([r.predictions_path,r.checkpoint_path,r.result_path])
    return p4,context,[r for r in runs if r.model_id=='M0'],list(dict.fromkeys(sources))


def checked_config(path,device_override=None):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('seeds')!=list(SEEDS) or cfg.get('controls')!=list(CONTROLS):raise ValueError('locked matrix changed')
    if cfg.get('training')!={'optimizer':'Adam','lr':.001,'batch_size':128,'max_epochs':100}:raise ValueError('locked training changed')
    if cfg.get('cpu_threads')!=24:raise ValueError('requires 24 CPU threads')
    if device_override is not None:cfg['device']=device_override
    if cfg.get('device') not in ('cpu','cuda'):raise ValueError('device must be cpu or cuda')
    from v2.phase7.deterministic_padding import ADAPTER_ID
    if cfg.get('runtime_adapter') not in ('none',ADAPTER_ID):raise ValueError('unregistered runtime adapter')
    return cfg



def build_control_model(p3,config):
    from v2.phase7.deterministic_padding import ADAPTER_ID,install_adapter
    record=config['runtime_adapter']
    if record['source_sha256']!=file_hash(ROOT/'v2/phase7/deterministic_padding.py'):
        raise ValueError('runtime adapter source hash differs')
    model=p3._model(config,'M0')
    if record['id']==ADAPTER_ID:
        paths=install_adapter(model)
        if not paths:raise ValueError('registered adapter found no reflection modules')
    elif record['id']=='none':paths=[]
    else:raise ValueError('unregistered runtime adapter identity')
    if 'module_paths' in record and record['module_paths']!=paths:
        raise ValueError('runtime adapter module paths differ')
    return model

def replay_run(directory,p3,dataset,split,config,identity,fingerprint):
    import torch
    m=verify_control_manifest(directory)
    if m['fingerprint']!=fingerprint:raise ValueError('completed identity differs')
    model=build_control_model(p3,config).to(torch.device(config['device']))
    model.load_state_dict(p3._load_state(directory/'checkpoint.pt',torch.device(config['device'])),strict=True)
    actual=p3._evaluate(model,dataset,split,config,torch.device(config['device']))
    with np.load(directory/'predictions.npz',allow_pickle=False) as a:
        if not np.array_equal(actual,a['logits']):raise ValueError('checkpoint does not replay saved logits exactly')
    metrics=p3.validate_prediction_bundle(directory/'predictions.npz',split,dataset,identity=identity,num_classes=config['architecture']['num_classes'],snr_values=config['snr']['values_db'])
    if metrics!=json.loads((directory/'metrics.json').read_text(encoding='utf-8')):raise ValueError('metric recomputation differs')
    return m


def run_one(base,control,seed,context,config,sources,source_hashes):
    import torch
    from phase1_reproduce import make_historical_train_val_loaders
    p3=context['module'];dataset,split,mapping=prepare_control(context['dataset'],context['split'],control)
    active=deepcopy(context['config']);active['device']=config['device'];active['training']=config['training'];active['architecture']['num_classes']=10 if control=='remove_wbfm' else 11
    from v2.phase7.deterministic_padding import DeterministicReflectionPad1d
    active['runtime_adapter']={'id':config['runtime_adapter'],'source_sha256':file_hash(ROOT/'v2/phase7/deterministic_padding.py')}
    probe=build_control_model(p3,active)
    active['runtime_adapter']['module_paths']=[name for name,m in probe.named_modules() if isinstance(m,DeterministicReflectionPad1d)]
    del probe
    environment=p3._environment_identity()
    if config['device']=='cuda':environment=environment|{'phase7_gpu_name':torch.cuda.get_device_name(0),'phase7_cuda_version':torch.version.cuda}
    protocol=dict(config=config,active_model_config=active,control=control,split=mapping,environment=environment,sources=source_hashes)
    protocol_hash=digest(protocol);fingerprint=digest(dict(protocol_hash=protocol_hash,seed=seed))
    directory=base/'runs'/f'{control}_seed{seed}_{fingerprint[:16]}'
    identity=dict(seed=seed,model_id='M0',split_hash=split.split_hash,preprocessing_hash=digest(mapping['preprocessing']),protocol_hash=protocol_hash)
    if (directory/'manifest.json').is_file():
        replay_run(directory,p3,dataset,split,active,identity,fingerprint);return directory
    directory.mkdir(parents=True,exist_ok=True)
    spec=dict(fingerprint=fingerprint,protocol_hash=protocol_hash,seed=seed,control=control,protocol=protocol)
    if (directory/'run_spec.json').exists():
        if json.loads((directory/'run_spec.json').read_text(encoding='utf-8'))!=spec:raise ValueError('partial run specification differs')
    else:
        atomic_json(directory/'run_spec.json',spec)
        np.savez_compressed(directory/'split.npz',train_idx=split.train_idx,val_idx=split.val_idx,test_idx=split.test_idx,sample_ids=split.sample_ids,
            original_labels=context['dataset']['labels'],labels=dataset['labels'],class_mapping=np.asarray(mapping['class_mapping']),parent_split_hash=context['split'].split_hash,split_hash=split.split_hash)
    def factories():
        signals=torch.from_numpy(np.asarray(dataset['signals'],dtype=np.float32));labels=torch.from_numpy(np.asarray(dataset['labels'],dtype=np.int64));snrs=torch.from_numpy(np.asarray(dataset['snrs'],dtype=np.float32))
        tr,va=(p3.make_conditioning_dataset(signals,labels,snrs,idx) for idx in (split.train_idx,split.val_idx))
        return make_historical_train_val_loaders(tr,va,train_batch_size=128,validation_batch_size=128)
    def log(row):
        text=json.dumps(row,allow_nan=False)
        with (directory/'training.log').open('a',encoding='utf-8') as stream:stream.write(text+'\n')
        print(f'{control} seed={seed}: {text}',flush=True)
    try:
        outcome=train_model(lambda:build_control_model(p3,active),factories,directory,seed=seed,training=active['training'],identity=fingerprint,
            loss_fn=p3._training_loss,threads=config['cpu_threads'],logger=log,device=config['device'])
        model=build_control_model(p3,active).to(torch.device(config['device']));model.load_state_dict(p3._load_state(Path(outcome['checkpoint']),torch.device(config['device'])),strict=True)
        logits=p3._evaluate(model,dataset,split,active,torch.device(config['device']))
        p3.write_prediction_bundle(directory/'predictions.npz',dict(logits=logits,labels=dataset['labels'][split.test_idx],snrs=dataset['snrs'][split.test_idx]),split,identity)
        metrics=p3.validate_prediction_bundle(directory/'predictions.npz',split,dataset,identity=identity,num_classes=active['architecture']['num_classes'],snr_values=active['snr']['values_db'])
        atomic_json(directory/'metrics.json',metrics)
        outcome['parameter_count']=sum(p.numel() for p in model.parameters());outcome['device']=config['device']
        atomic_json(directory/'training.json',outcome)
        if not (directory/'training.log').exists(): (directory/'training.log').write_text('Recovered all epochs from verified resume state.\n',encoding='utf-8')
        write_manifest(directory,dict(control=control,seed=seed,fingerprint=fingerprint,protocol_hash=protocol_hash,split_hash=split.split_hash,epochs_completed=100),sources,expected_sources=source_hashes)
        replay_run(directory,p3,dataset,split,active,identity,fingerprint)
        return directory
    except Exception:
        name=datetime.now(timezone.utc).strftime('failure_%Y%m%dT%H%M%S%fZ.log')
        (directory/name).write_text(traceback.format_exc(),encoding='utf-8');raise


def algebraic_controls(base,context,baselines,sources,source_hashes,config):
    p3=context['module'];dataset=context['dataset'];split=context['split']
    fingerprint=digest(dict(sources=source_hashes,config=config,kind='balanced_and_remap_v1'))
    directory=base/'algebraic'/fingerprint[:16]
    if (directory/'manifest.json').exists():
        m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for name,h in m['artifacts'].items():
            if file_hash(directory/name)!=h:raise ValueError('algebraic artifact hash differs')
        for name,h in m['sources'].items():
            if file_hash(name)!=h:raise ValueError('algebraic source hash differs')
        return directory
    directory.mkdir(parents=True,exist_ok=True)
    balance=balanced_control(dataset['labels'][split.train_idx]);atomic_json(directory/'balanced_counts.json',balance)
    rows=[]
    for r in sorted(baselines,key=lambda r:r.seed):
        with np.load(r.predictions_path,allow_pickle=False) as a:values={k:a[k] for k in a.files}
        remap=remap_control(values['logits'],values['y_true'],np.arange(10,-1,-1))
        np.savez_compressed(directory/f'remap_seed{r.seed}.npz',**remap,sample_ids=values['sample_ids'],snr_db=values['snr_db'],seed=r.seed,source_predictions_sha256=r.predictions_sha256)
        calc=lambda y,l,s:p3.compute_metrics(y,l,s,num_classes=11,snr_values=context['config']['snr']['values_db'])
        retained=values['y_true']!=3
        rows.append(dict(seed=r.seed,baseline_predictions=str(r.predictions_path),baseline_sha256=r.predictions_sha256,
            baseline=calc(values['y_true'],values['logits'],values['snr_db']),
            baseline_retained_test=calc(values['y_true'][retained],values['logits'][retained],values['snr_db'][retained]),
            remapped=calc(remap['y_true'],remap['logits'],values['snr_db']),tie_rows=int(remap['tie_rows'].sum()),identity_changes=int(remap['identity_changed'].sum())))
    if [r['seed'] for r in rows]!=list(SEEDS):raise ValueError('five verified baselines required')
    atomic_json(directory/'summary.json',dict(baselines_and_remaps=rows,balanced_sampling=balance,limitation='Remapping measures index/tie effects only; removal baseline retains 11 output classes on the reduced test set.'))
    write_manifest(directory,dict(fingerprint=fingerprint,kind='algebraic_controls'),sources,expected_sources=source_hashes)
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase6-gate',required=True);parser.add_argument('--config',default='configs/v2/phase7_controls.yaml')
    parser.add_argument('--control',choices=['all',*CONTROLS],default='all');parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--device',choices=['cpu','cuda'],help='explicit fallback creates a new protocol fingerprint')
    parser.add_argument('--prepare-only',action='store_true',help='verify sources and save algebraic controls; no training')
    args=parser.parse_args();config_path=(ROOT/args.config).resolve();cfg=checked_config(config_path,args.device)
    import torch
    torch.set_num_threads(cfg['cpu_threads'])
    if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; rerun --device cpu for a distinct protocol')
    base=ROOT/'results/v2/phase7_controls';base.mkdir(parents=True,exist_ok=True)
    print('UPSTREAM_VERIFICATION_STARTED: recursive Phase 6 composite and strict Phase 3/4 contracts',flush=True)
    p4,context,baselines,sources=source_context(Path(args.phase6_gate).resolve(),config_path)
    print('UPSTREAM_VERIFICATION_PASSED: strict replay complete; preparing controls',flush=True)
    source_hashes={str(Path(p).resolve()):file_hash(p) for p in sources}
    with p4._Phase4Lock(base/'.controls.lock'):
        algebra=algebraic_controls(base,context,baselines,sources,source_hashes,cfg)
        if args.prepare_only:print(f'Algebraic controls prepared at {algebra}; retraining pending.');return
        selected=CONTROLS if args.control=='all' else (args.control,)
        seeds=SEEDS if args.seed is None else (args.seed,)
        directories=[]
        for control in selected:
            for seed in seeds:directories.append(run_one(base,control,seed,context,cfg,sources,source_hashes))
        if len(directories)==10:
            summary=summarize_matrix(directories);summary['algebraic_path']=str(algebra);summary['algebraic_manifest_sha256']=file_hash(algebra/'manifest.json')
            from v2.statistics import paired_summary
            baseline_rows=json.loads((algebra/'summary.json').read_text(encoding='utf-8'))['baselines_and_remaps']
            comparisons={}
            for control in CONTROLS:
                field='baseline_retained_test' if control=='remove_wbfm' else 'baseline'
                a=[r[field]['overall_accuracy'] for r in baseline_rows]
                b=[r['metrics']['overall_accuracy'] for r in summary['runs'] if r['control']==control]
                comparisons[control]=context['module']._json_value(paired_summary(a,b))
            summary['paired_accuracy']=comparisons
            target=base/'summaries'/digest(summary)[:16];target.mkdir(parents=True,exist_ok=True)
            if not (target/'summary.json').exists():atomic_json(target/'summary.json',summary)
            print(f'Retraining matrix summary: {target}',flush=True)
        else:print('Selected runs verified. Full ten-run aggregation remains pending.',flush=True)


if __name__=='__main__':main()

"""Independent post-hoc AM-SSB intervention; never starts before primary gate."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,Path(__file__).resolve().parent):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from run_phase7_controls import source_context,build_control_model,replay_run
from v2.phase7.controls import SEEDS,digest,file_hash,atomic_json,train_model,write_manifest,verify_control_manifest
from v2.phase7.amssb_followup import prepare_amssb,align_predictions,band_metrics
from v2.phase7.primary_receipt import validate_primary_gate,load
from v2.phase6.composite import manifest_closure


def checked_followup_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('control')!='remove_amssb' or cfg.get('target_selection')!='posthoc_from_existing_test_predictions' or cfg.get('seeds')!=list(SEEDS):raise ValueError('followup target or seeds changed')
    if cfg.get('training')!={'optimizer':'Adam','lr':.001,'batch_size':128,'max_epochs':100} or cfg.get('cpu_threads')!=24:raise ValueError('followup hyperparameters changed')
    from v2.phase7.deterministic_padding import ADAPTER_ID
    if cfg.get('device')!='cuda' or cfg.get('runtime_adapter')!=ADAPTER_ID:raise ValueError('followup requires registered CUDA adapter protocol')
    return cfg

def run_followup(base,control,seed,context,config,sources,source_hashes):
    import torch
    from phase1_reproduce import make_historical_train_val_loaders
    p3=context['module'];dataset,split,mapping=prepare_amssb(context['dataset'],context['split'])
    active=deepcopy(context['config']);active['device']=config['device'];active['training']=config['training'];active['architecture']['num_classes']=10
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


def summarize_followup(directories,baselines,primary_summary,output,sources,source_hashes):
    if len(directories)!=5:raise ValueError('five followup seeds required')
    records=[];bundles={}
    for directory in directories:
        m=verify_control_manifest(directory)
        if m['control']!='remove_amssb':raise ValueError('wrong followup control')
        seed=m['seed']
        with np.load(directory/'predictions.npz',allow_pickle=False) as p:values={k:p[k] for k in p.files}
        baseline=next(r for r in baselines if r.seed==seed)
        with np.load(baseline.predictions_path,allow_pickle=False) as p:old={k:p[k] for k in p.files}
        ids=values['sample_ids'];y=values['y_true'];snr=values['snr_db'];pred=values['logits'].argmax(1)
        old_pred=align_predictions(old['sample_ids'],old['logits'].argmax(1),ids)
        np.testing.assert_array_equal(align_predictions(old['sample_ids'],old['y_true'],ids),y)
        common=y!=3;common_ids=ids[common]
        removal=next(r for r in primary_summary['runs'] if r['control']=='remove_wbfm' and r['seed']==seed)
        with np.load(Path(removal['path'])/'predictions.npz',allow_pickle=False) as p:
            inverse=np.array([0,1,2,4,5,6,7,8,9,10])
            wbfm_pred=align_predictions(p['sample_ids'],inverse[p['logits'].argmax(1)],common_ids)
            np.testing.assert_array_equal(align_predictions(p['sample_ids'],inverse[p['y_true']],common_ids),y[common])
        record=dict(seed=seed,run_path=str(directory),manifest_sha256=file_hash(directory/'manifest.json'),
            retained_baseline=band_metrics(y,old_pred,snr,11),retained_followup=band_metrics(y,pred,snr,10),
            common_wbfm_removed=band_metrics(y[common],wbfm_pred,snr[common],11),common_amssb_removed=band_metrics(y[common],pred[common],snr[common],11))
        # Common-row distributions use original 11-class identity coordinates;
        # both removals have 10 allowed outputs, so normalize entropy by log(10).
        for key in ('common_wbfm_removed','common_amssb_removed'):
            for metrics in record[key].values():
                metrics['normalized_entropy']*=float(np.log(11)/np.log(10));metrics['num_outputs']=10;metrics['identity_coordinates']=11
        records.append(record)
        bundles.update({f'seed{seed}_{k}':v for k,v in dict(sample_ids=ids,y_true=y,snr_db=snr,baseline_prediction=old_pred,followup_prediction=pred,common_ids=common_ids,common_wbfm_prediction=wbfm_pred).items()})
    if sorted(r['seed'] for r in records)!=list(SEEDS):raise ValueError('duplicate or missing followup seeds')
    from v2.statistics import paired_summary
    comparisons={}
    for endpoint in ('concentration','balanced_accuracy','accuracy','normalized_entropy','hhi','concentration_minus_ba'):
        for label,a,b in [('retained','retained_baseline','retained_followup'),('common','common_wbfm_removed','common_amssb_removed')]:
            import run_phase3 as p3
            comparisons[label+'_'+endpoint]=p3._json_value(paired_summary([r[a]['low'][endpoint] for r in records],[r[b]['low'][endpoint] for r in records]))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'matched_predictions.npz',**bundles)
    atomic_json(output/'summary.json',dict(status='FIVE_SEED_POSTHOC_FOLLOWUP_COMPLETE',target_selection='posthoc_from_existing_test_predictions',full_phase7_complete=False,pending=['cross_dataset_channel_phase11'],runs=records,paired_low_snr=comparisons))
    run_sources=[p for d in directories for p in d.iterdir() if p.is_file()]
    all_sources=list(sources)+run_sources
    expected=dict(source_hashes);expected.update({str(p.resolve()):file_hash(p) for p in run_sources})
    write_manifest(output,dict(kind='amssb_exploratory_followup_summary'),all_sources,expected_sources=expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/v2/phase7_amssb_followup.yaml')
    parser.add_argument('--execute',action='store_true',help='train only after parent scheduling and validated primary receipt')
    args=parser.parse_args();config_path=ROOT/args.config;cfg=checked_followup_config(config_path)
    gate_path=ROOT/'results/v2/phase7_controls/gate.json';gate=load(gate_path);receipt=validate_primary_gate(ROOT,gate)
    primary_summary=load(ROOT/load(receipt/'decision.json')['summary_path'])
    import torch
    torch.set_num_threads(24)
    p4,context,baselines,sources=source_context(ROOT/'results/v2/phase6_preprocessing/current.json',ROOT/'configs/v2/phase7_controls.yaml')
    sources += [Path(__file__),ROOT/'v2/phase7/amssb_followup.py',config_path,gate_path,ROOT/'docs/phase7_remove_amssb_prospective.md',ROOT/'docs/phase7_amssb_implementation.md']
    sources += [ROOT/n for n in manifest_closure(ROOT,ROOT/gate['manifest_path'])]
    sources += [Path(r['path'])/'manifest.json' for r in primary_summary['runs']]
    sources=list(dict.fromkeys(p.resolve() for p in sources));hashes={str(p):file_hash(p) for p in sources}
    data,split,mapping=prepare_amssb(context['dataset'],context['split'])
    counts=[len(split.train_idx),len(split.val_idx),len(split.test_idx)]
    if counts!=[120000,40000,40000] or int((data['snrs'][split.test_idx]<=-8).sum())!=14000:raise ValueError('registered removal population changed')
    print('PREFLIGHT_PASSED',counts,split.split_hash,flush=True)
    if not args.execute:return
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<1536*1024**2:raise RuntimeError('CUDA safety margin unavailable; no training started')
    base=ROOT/'results/v2/phase7_amssb_followup';base.mkdir(parents=True,exist_ok=True)
    with p4._Phase4Lock(ROOT/'results/v2/phase7_controls/.controls.lock'):
        directories=[run_followup(base,'remove_amssb',s,context,cfg,sources,hashes) for s in SEEDS]
        summary_path=base/'summaries'/digest({'sources':hashes,'runs':[str(d) for d in directories]})[:16]
        if summary_path.exists():
            from v2.phase7.primary_receipt import raw_sources
            raw_sources(summary_path,'COMPLETE')
        else:summarize_followup(directories,baselines,primary_summary,summary_path,sources,hashes)
        print(summary_path)


if __name__=='__main__':main()


"""Phase8-cache matched floor interventions; formal execution requires Phase8/9."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import traceback
import uuid

import numpy as np
import torch
import yaml

from v2.phase6.composite import validate_composite
from v2.phase8.evidence import dump, load, safe, sha, check_gates as phase8_gates
from v2.phase8.runner import get_model
from .core import FLOORS, evaluate_floors, metrics_with_strata, paired_rows, floor_conditions


def manifest_closure(root, manifest_path):
    """Artifact closure with exactly one legacy registry leaf exception."""
    root=Path(root).resolve();checked={};visited=set()
    def check(path,expected=None):
        path=Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('missing or escaped dependency')
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('dependency hash mismatch')
        checked[path.relative_to(root).as_posix()]=actual
        return path
    def visit(path):
        path=check(path)
        if path in visited:return
        visited.add(path);manifest=load(path)
        if path==root/'manifest.json':
            if manifest.get('schema_version')!=1 or not isinstance(manifest.get('experiments'),list):raise ValueError('invalid legacy root registry')
            return
        files=manifest.get('files');dependencies=manifest.get('dependencies',manifest.get('source_files'))
        if not isinstance(files,dict) or not files or not isinstance(dependencies,dict):raise ValueError('artifact manifest lacks hash closure')
        actual={p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file() and p!=path}
        if actual!=set(files):raise ValueError('artifact closure polluted or incomplete')
        for name,digest in files.items():check(safe(path.parent,name),digest)
        for name,digest in dependencies.items():
            dependency=check(safe(root,name),digest)
            if dependency.name=='manifest.json':visit(dependency)
    try:visit(manifest_path)
    except (OSError,KeyError,TypeError) as exc:raise ValueError(f'invalid hash closure: {exc}') from exc
    return dict(sorted(checked.items()))


def validate_config(config):
    expected = {'schema_version':1,'runner_version':'phase10-tested-floors/1','models':['M3','M6'],'seeds':list(range(2022,2027)),'conditions':['oracle','estimate'],'floors_db':list(FLOORS),'batch_size':64,'torch_num_threads':24,'device':'cpu','error_strata_abs_db':[0,2,4,8],'midpoint_rule':'lower_bin_on_equal_distance','floor_order':'range_clip_then_float32_then_floor_then_phase8_nearest_bin','unchanged_condition_logits':'exact','raw_error_semantics':'source_condition_minus_true_snr_before_range_clip_and_floor','oracle_error_semantics':'zero_by_construction_not_pooled_with_estimates','phase9_required_distribution_shift_status':'PENDING_PHASE11_12'}
    if any(config.get(k)!=v for k,v in expected.items()) or config['no_floor_parity']!={'rtol':1e-6,'atol':1e-7,'exact_argmax':True}:
        raise ValueError('Phase10 prospective floor matrix/numeric contract changed')


def check_gates(root, config):
    root = Path(root).resolve()
    try:
        pointers, paths, closure = {}, {}, {}
        for phase, key in [('phase8','phase8_current'),('phase9','phase9_current')]:
            pointer_path = safe(root,config[key]); pointer = load(pointer_path)
            if pointer.get('status')!='COMPLETE':
                raise ValueError('upstream formal gate is not COMPLETE')
            attempt = safe(root,pointer['attempt']); manifest_path=attempt/'manifest.json'
            if sha(manifest_path)!=pointer['manifest_sha256'] or load(manifest_path).get('status')!='COMPLETE':
                raise ValueError('upstream gate hash/status differs')
            closure.update(manifest_closure(root,manifest_path));closure[pointer_path.relative_to(root).as_posix()]=sha(pointer_path)
            pointers[phase]=pointer;paths[phase]=attempt
        phase9_manifest=load(paths['phase9']/'manifest.json')
        if phase9_manifest.get('distribution_shift_status')!=config['phase9_required_distribution_shift_status']:
            raise ValueError('Phase9 primary-scope gate must retain pending Phase11/12 distribution shift')
        p8config=load(paths['phase8']/'config.json')
        if p8config['models']!=['M3','M6'] or p8config['seeds']!=list(range(2022,2027)) or p8config['batch_size']!=64 or p8config['torch_num_threads']!=24:
            raise ValueError('Phase8 matched source scope/runtime differs')
        data, inherited=phase8_gates(root,p8config)
        if inherited!=load(paths['phase8']/'gates.json'):
            raise ValueError('Phase8 inherited gates changed')
        current6=load(root/config['phase6_current']);validate_composite(root,current6)
        closure.update(manifest_closure(root,root/current6['gate']['manifest_path']))
        closure[config['phase6_current']]=sha(root/config['phase6_current'])
        gate7=load(root/config['phase7_gate'])
        for name in ('manifest_path','diagnostics_manifest_path'):
            closure.update(manifest_closure(root,root/gate7[name]))
        closure[config['phase7_gate']]=sha(root/config['phase7_gate'])
        selection=load(paths['phase8']/'model_selection.json')
        if selection['selected']!='M6' or selection['selection_partition']!='validation':
            raise ValueError('M6 is not validation-selected in Phase8')
        return paths, {'phase8':pointers['phase8'],'phase9':pointers['phase9'],'phase9_distribution_shift_status':phase9_manifest['distribution_shift_status'],'phase6_data_attempt':data.relative_to(root).as_posix(),'classification_only_no_phase6_numeric_exemption':True}, closure
    except (OSError,KeyError,TypeError) as exc:
        raise ValueError(f'upstream gate missing or malformed: {exc}') from exc


def _model_sources(root, config, phase8):
    from v2.phase3_analysis import _load_phase3_contracts
    phase3=_load_phase3_contracts();p3config=phase3.load_phase3_config(root/'configs/v2/phase3.yaml')
    ledger=phase3._load_ledger(root/'results/v2/phase3/phase3_ledger.json')
    manifest8=load(phase8/'manifest.json'); runs={}
    for model in config['models']:
        for seed in config['seeds']:
            matches=[r for r in ledger['records'] if r['status']=='completed' and r['model_id']==model and r['seed']==seed]
            if len(matches)!=1:raise ValueError('ambiguous registered source checkpoint')
            entry=matches[0];directory=safe(root,entry['attempt_path']);result_path=directory/'result.json';result=load(result_path)
            if sha(result_path)!=entry['result_sha256'] or manifest8['dependencies'].get(result_path.relative_to(root).as_posix())!=entry['result_sha256']:
                raise ValueError('Phase8/ledger result hash differs')
            checkpoint=directory/result['artifacts']['checkpoint']
            if sha(checkpoint)!=result['artifacts']['checkpoint_sha256'] or manifest8['dependencies'].get(checkpoint.relative_to(root).as_posix())!=sha(checkpoint):
                raise ValueError('Phase8 checkpoint identity differs')
            runs[(model,seed)]={'paths':{'checkpoint':checkpoint},'result':result}
    return phase3,p3config,runs


def _source_bundle(phase8, model, seed, condition, raw_estimates, test_ids):
    label='E1' if condition=='oracle' else 'E2'
    path=phase8/'conditions'/f'{model}_{seed}_{label}.npz'
    with np.load(path,allow_pickle=False) as b:a={k:b[k] for k in b.files}
    if a['model_id'].item()!=model or a['seed'].item()!=seed or a['condition'].item()!=label or not np.array_equal(a['sample_ids'],test_ids):
        raise ValueError('Phase8 baseline row identity differs')
    expected=a['snr_db'] if condition=='oracle' else raw_estimates
    if not np.array_equal(a['raw_condition_db'],expected):raise ValueError('Phase8 no-floor raw condition differs')
    return a,path


def _evaluation_arrays(evaluation, baseline, raw, model, seed, condition, checkpoint_hash, cache_hash, reference_hash):
    return {**{k:evaluation[k] for k in ('source_condition_db','range_clipped_condition_db','applied_condition_db','condition_bin','raw_error_db','range_clipped_mask','floor_active_mask','logits')},'sample_ids':baseline['sample_ids'],'y_true':baseline['y_true'],'snr_db':baseline['snr_db'],'raw_estimate_db':raw,'raw_estimator_error_db':raw.astype(float)-baseline['snr_db'].astype(float),'model_id':np.asarray(model),'seed':np.asarray(seed),'condition':np.asarray(condition),'floor_enabled':np.asarray(evaluation['floor_db'] is not None),'floor_db':np.asarray(0. if evaluation['floor_db'] is None else evaluation['floor_db']),'checkpoint_sha256':np.asarray(checkpoint_hash),'cache_sha256':np.asarray(cache_hash),'phase8_baseline_sha256':np.asarray(reference_hash)}


def _persist_bundle(path, arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('resume refuses to overwrite differing floor artifacts')
        return
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def _evaluate_matrix(root, phase8, config, attempt, validate_only=False):
    phase3,p3config,runs=_model_sources(root,config,phase8)
    with np.load(phase8/'all_estimates.npz',allow_pickle=False) as b:
        raw=b['estimated_snr_db'][b['test_indices']];ids=b['sample_ids'][b['test_indices']]
    if len(ids)!=44000 or not np.isfinite(raw).all():raise ValueError('complete fixed test estimator scope required')
    rows=[]
    for model_id in config['models']:
        for seed in config['seeds']:
            run=runs[(model_id,seed)];model=get_model(phase3,p3config,run,model_id,'cpu')
            cache_path=phase8/'caches'/f'{model_id}_{seed}.npz'
            with np.load(cache_path,allow_pickle=False) as b:
                features=b['raw_pooled_features']
                if features.shape!=(44000,128) or not np.isfinite(features).all() or not np.array_equal(b['sample_ids'],ids) or b['checkpoint_sha256'].item()!=sha(run['paths']['checkpoint']):
                    raise ValueError('raw pooled cache scope/checkpoint binding differs')
            for condition in config['conditions']:
                source,path=_source_bundle(phase8,model_id,seed,condition,raw,ids)
                evaluations=evaluate_floors(model,features,source['raw_condition_db'],source['snr_db'],config['floors_db'],source['logits'],config['batch_size'])
                for evaluation in evaluations:
                    floor=evaluation['floor_db'];key=f'{model_id}_{seed}_{condition}_'+('none' if floor is None else f'floor{floor}')
                    bundle=attempt/'evaluations'/f'{key}.npz';row_path=attempt/'evaluations'/f'{key}.json'
                    arrays=_evaluation_arrays(evaluation,source,raw,model_id,seed,condition,sha(run['paths']['checkpoint']),sha(cache_path),sha(path))
                    row={'model':model_id,'seed':seed,'condition':condition,'floor_db':floor,'bundle':bundle.relative_to(attempt).as_posix(),'metrics':metrics_with_strata(source['y_true'],evaluation['logits'],source['snr_db'],evaluation['raw_error_db']),'raw_error_definition':'oracle_zero' if condition=='oracle' else 'unclipped_estimate_minus_true_snr','floor_active_fraction':float(evaluation['floor_active_mask'].mean()),'range_clipped_fraction':float(evaluation['range_clipped_mask'].mean()),'unchanged_condition_rows':evaluation['unchanged_rows'],'unchanged_logits_exact':evaluation['unchanged_logits_exact'],'no_floor_phase8_max_abs':evaluation['no_floor_phase8_max_abs']}
                    if validate_only:
                        if not bundle.exists() or not row_path.exists():raise ValueError('missing planned floor evaluation')
                        with np.load(bundle,allow_pickle=False) as saved:
                            if set(saved.files)!=set(arrays):raise ValueError('floor bundle schema differs')
                            for name,value in arrays.items():
                                if not np.array_equal(saved[name],value):raise ValueError(f'floor source/logit replay differs: {key}/{name}')
                        if load(row_path)!=row:raise ValueError('floor metric/error/strata replay differs')
                    else:
                        _persist_bundle(bundle,arrays)
                        if row_path.exists():
                            if load(row_path)!=row:raise ValueError('resume metric record differs')
                        else:dump(row_path,row)
                    rows.append(row)
            if not validate_only:
                dump(attempt/'resume_ledger.json',{'status':'RUNNING','completed_evaluations':len(rows),'completed_bundle_hashes':{r['bundle']:sha(attempt/r['bundle']) for r in rows}})
                print(f'Phase10 {model_id}/seed{seed}: {len(rows)}/80 evaluations',flush=True)
    return rows


def _figures(attempt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=load(attempt/'rows.json')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    for metric in ('accuracy','balanced_accuracy','concentration'):
        fig,ax=plt.subplots(figsize=(6.2,4.0))
        for model,color in [('M3','#4477AA'),('M6','#228833')]:
            for condition,style in [('oracle','-'),('estimate','--')]:
                values=np.array([[next(r for r in rows if (r['model'],r['seed'],r['condition'],r['floor_db'])==(model,seed,condition,floor))['metrics']['overall'][metric] for seed in range(2022,2027)] for floor in FLOORS])
                ax.errorbar(range(4),values.mean(1),yerr=values.std(1,ddof=1),fmt='o'+style,color=color,capsize=3,label=model+' '+condition)
        ax.set_xticks(range(4),['No floor','-10 dB','-8 dB','-6 dB']);ax.set_ylabel(metric.replace('_',' '));ax.set_xlabel('Tested reliability floor');ax.legend(fontsize=8)
        ax.set_title('Five matched training seeds; mean ± SD',fontsize=10);fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(attempt/f'floor_{metric}.{ext}',dpi=300)
        plt.close(fig)
    dump(attempt/'figure_sources.json',{'rows.json':sha(attempt/'rows.json'),'paired_statistics.json':sha(attempt/'paired_statistics.json')})


def validate(attempt, root, allow_candidate=False):
    root=Path(root).resolve();attempt=Path(attempt).resolve();manifest=load(attempt/'manifest.json')
    if manifest['status'] not in (['CANDIDATE','COMPLETE'] if allow_candidate else ['COMPLETE']):raise ValueError('formal Phase10 manifest incomplete')
    manifest_closure(root,attempt/'manifest.json')
    config=load(attempt/'config.json');validate_config(config);torch.set_num_threads(config['torch_num_threads'])
    paths,gates,_=check_gates(root,config)
    if load(attempt/'gates.json')!=gates:raise ValueError('Phase10 inherited gates changed')
    expected=_evaluate_matrix(root,paths['phase8'],config,attempt,validate_only=True)
    if len(expected)!=80 or load(attempt/'rows.json')!=expected or load(attempt/'paired_statistics.json')!=paired_rows(expected):raise ValueError('80-row matrix or matched CI replay differs')
    source=load(attempt/'figure_sources.json')
    if source!={'rows.json':sha(attempt/'rows.json'),'paired_statistics.json':sha(attempt/'paired_statistics.json')}:raise ValueError('figure source binding differs')
    return {'status':'VALIDATED','evaluations':80,'checkpoints':10,'phase9_distribution_shift_status':'PENDING_PHASE11_12','files':len(manifest['files'])}


def _publish_current(attempt,output,root,verification):
    temporary=output/('current.'+uuid.uuid4().hex+'.tmp.json')
    dump(temporary,{'status':'COMPLETE','attempt':attempt.relative_to(root).as_posix(),'manifest_sha256':sha(attempt/'manifest.json'),'validation':verification})
    temporary.replace(output/'current.json')


def _record_failure(attempt,output,error):
    complete=(attempt/'manifest.json').exists() and load(attempt/'manifest.json').get('status')=='COMPLETE'
    destination=output/'publication_failures' if complete else attempt
    dump(destination/('failure_'+uuid.uuid4().hex[:8]+'.json'),{'status':'FAILED','attempt':str(attempt),'traceback':error})


def execute(config_path, root, resume=None):
    root=Path(root).resolve();config_path=Path(config_path).resolve();config=yaml.safe_load(config_path.read_text(encoding='utf-8'));validate_config(config)
    torch.set_num_threads(config['torch_num_threads']);paths,gates,dependencies=check_gates(root,config)
    source_paths=[config_path,root/'scripts/v2/run_phase10.py',*sorted((root/'v2/phase10').glob('*.py')),*sorted((root/'v2/phase8').glob('*.py')),root/'v2/phase6/composite.py',root/'models/model.py',root/'models/model_conditioning.py',root/'models/lifting.py',root/'scripts/v2/run_phase3.py',root/'manifest.json',root/'results/v2/phase3/phase3_ledger.json']
    dependencies.update({p.relative_to(root).as_posix():sha(p) for p in source_paths})
    output=safe(root,config['output']);output.mkdir(parents=True,exist_ok=True)
    if resume:
        attempt=safe(root,resume)
        if not attempt.is_relative_to(output/'attempts'):raise ValueError('resume must be a Phase10 attempt')
        if (attempt/'manifest.json').exists() and load(attempt/'manifest.json')['status']=='COMPLETE':
            verification=validate(attempt,root)
            _publish_current(attempt,output,root,verification)
            return verification
        if load(attempt/'config.json')!=config or load(attempt/'dependencies.json')!=dependencies or load(attempt/'gates.json')!=gates:raise ValueError('resume source/config/gates changed')
        expected_environment={'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','matplotlib')},'device':'cpu','torch_num_threads':torch.get_num_threads()}
        if load(attempt/'environment.json')!=expected_environment:raise ValueError('resume runtime environment changed')
    else:
        attempt=output/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);(attempt/'evaluations').mkdir(parents=True)
        dump(attempt/'config.json',config);dump(attempt/'dependencies.json',dependencies);dump(attempt/'gates.json',gates)
        dump(attempt/'environment.json',{'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','matplotlib')},'device':'cpu','torch_num_threads':torch.get_num_threads()})
    try:
        rows=_evaluate_matrix(root,paths['phase8'],config,attempt)
        dump(attempt/'rows.json',rows);dump(attempt/'paired_statistics.json',paired_rows(rows));_figures(attempt)
        dump(attempt/'summary.json',{'status':'COMPUTED','evaluations':80,'checkpoints':10,'five_seed_one_partition':True,'phase9_distribution_shift_status':'PENDING_PHASE11_12','test_selected_floor':False,'no_floor_role':'matched reproduction of Phase8 E1/E2','estimator_scope':'supervised dataset-trained Phase6 ridge, not a physical pilot estimator','terms':'tested reliability floor; no universal threshold claim'})
        (attempt/'report.md').write_text('# Phase 10 tested reliability floors\n\nAll 80 matched evaluations use M3/M6, five checkpoint seeds, oracle or frozen supervised SNR estimate and no floor/-10/-8/-6 dB. No floor must reproduce Phase8 E1/E2 before floor results are accepted. Raw estimator errors precede range clipping and flooring; oracle errors are zero by construction and reported separately.\n\nOverall, low (≤-8), mid (-6 through -2), high (≥0), every registered SNR and absolute-error strata [0,2), [2,4), [4,8), [8,∞) are retained. Empty strata have count zero and null metrics. BA averages represented true classes; macro F1 retains all 11 classes. Error-to-dominant share divides wrong predictions sent to the group dominant predicted class by all wrong predictions; it is null if there are no errors.\n\nPaired tables report each floor minus no floor for five matched training seeds on one fixed test set. CI is a two-sided Student-t 95% interval, not independent-dataset uncertainty. No apparently best test floor is selected or called universal/optimal/a phase transition. Figures read saved row/paired JSON. Phase9 cross-domain evaluation remains pending Phase11/12; this run consumes only its completed primary-dataset gate.\n',encoding='utf-8')
        dump(attempt/'resume_ledger.json',{'status':'COMPUTED','completed_evaluations':80,'completed_bundle_hashes':{r['bundle']:sha(attempt/r['bundle']) for r in rows}})
        for name,digest in dependencies.items():
            if sha(root/name)!=digest:raise ValueError('dependency changed during Phase10')
        manifest={'schema_version':1,'status':'CANDIDATE','dependencies':dependencies,'files':{p.relative_to(attempt).as_posix():sha(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
        dump(attempt/'manifest.json',manifest);verification=validate(attempt,root,allow_candidate=True)
        manifest['status']='COMPLETE';dump(attempt/'manifest.json',manifest)
        _publish_current(attempt,output,root,verification)
        return verification
    except Exception:
        _record_failure(attempt,output,traceback.format_exc())
        raise


def smoke(root):
    """Small untrained-head test; no formal gate, no scientific current pointer."""
    from models.model_conditioning import AWNConditioned
    root=Path(root).resolve();torch.set_num_threads(1);torch.manual_seed(41)
    output=root/'results/v2/phase10_reliability_floors/smoke'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);output.mkdir(parents=True)
    features=np.random.default_rng(41).normal(size=(88,128)).astype(np.float32);truth=np.tile(np.arange(-20,20,2),5)[:88].astype(np.float32);raw=truth+np.linspace(-12,12,88);y=np.arange(88)%11
    rows=[]
    from v2.phase8.core import head_logits
    for model_id in ('M3','M6'):
        model=AWNConditioned(num_classes=11,num_levels=1,in_channels=64,kernel_size=3,latent_dim=320,regu_details=.01,regu_approx=.01,num_snr_bins=20,snr_embedding_dim=8,conditioning=model_id).eval()
        for condition,source in [('oracle',truth),('estimate',raw)]:
            baseline=head_logits(model,features,source,16)
            for item in evaluate_floors(model,features,source,truth,list(FLOORS),baseline,16):
                metrics=metrics_with_strata(y,item['logits'],truth,item['raw_error_db'])
                rows.append({'model':model_id,'condition':condition,'floor_db':item['floor_db'],'count':metrics['overall']['count'],'unchanged_logits_exact':item['unchanged_logits_exact']})
    result={'status':'SMOKE_ONLY_NON_SCIENTIFIC','untrained_random_heads':True,'real_dataset_used':False,'formal_phase8_phase9_gates_bypassed':False,'formal_matrix_run':False,'evaluations':len(rows),'rows':rows}
    dump(output/'smoke.json',result)
    return {'status':result['status'],'artifact':(output/'smoke.json').relative_to(root).as_posix(),'evaluations':len(rows)}

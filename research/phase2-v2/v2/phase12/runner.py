"""Gated Phase12 runner and small end-to-end smoke, with immutable resumable output."""
from copy import deepcopy
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from filelock import FileLock
from v2.phase7.controls import train_model,digest
from v2.phase8.evidence import sha,dump,load,safe
from v2.phase8.core import conditions,head_logits
from .assembly import ROOT,assemble,load_dataset,validate_dataset,seal,verify
from .execution import runtime_config,check_gates,build_model,loaders,fit_estimator,predict_estimator,evaluation_conditions
from .reporting import summarize,paired,figures


def source_files(config_path,formal=True):
    paths=[*sorted((ROOT/'v2/phase12').glob('*.py')),Path(config_path),ROOT/'configs/v2/phase12.yaml',
        ROOT/'models/model_conditioning.py',ROOT/'models/model.py',ROOT/'models/lifting.py',
        ROOT/'v2/phase7/controls.py',ROOT/'v2/phase7/deterministic_padding.py',ROOT/'v2/phase8/core.py',ROOT/'v2/phase8/evidence.py',ROOT/'v2/phase6/audit.py',ROOT/'scripts/v2/run_phase12.py',
        ROOT.parents[1]/'experiments/train_baselines.py']
    if formal:paths.extend([ROOT/'v2/phase6/composite.py',ROOT/'v2/phase11/runner.py'])
    return {str(p.resolve()):sha(p) for p in paths}


def equal_bundle(path,values):
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as b:
            if set(b.files)!=set(values):raise ValueError('saved prediction keys differ')
            for name,value in values.items():
                value=np.asarray(value)
                if not np.array_equal(b[name],value,equal_nan=value.dtype.kind in 'fc'):raise ValueError('saved prediction replay differs: '+name)
    else:np.savez_compressed(path,**values)


def eval_model(model,data,indices,choices,batch):
    device=next(model.parameters()).device;full=[];features=[];model.eval()
    cache=hasattr(model,'extract_pooled_features')
    before={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch];db,bins=conditions(data['identity']['snr_db'][ix]);x=torch.from_numpy(np.array(data['signals'][ix],copy=True)).to(device)
            full.append(model.forward_batch(x,snr_db=torch.as_tensor(db,device=device),snr_bin=torch.as_tensor(bins,device=device))[0].cpu().numpy())
            if cache:features.append(model.extract_pooled_features(x)[0].cpu().numpy())
    full=np.concatenate(full);z=np.concatenate(features) if cache else None
    if not np.isfinite(full).all():raise ValueError('nonfinite evaluation logits')
    predictions={}
    for name,choice in choices.items():
        predictions[name]=head_logits(model,z,choice['clipped_db'][indices],batch) if cache else full.copy()
    if not np.allclose(predictions['oracle'],full,rtol=1e-6,atol=1e-7) or not np.array_equal(predictions['oracle'].argmax(1),full.argmax(1)):raise ValueError('pooled cache differs from actual oracle forward')
    if any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.named_buffers()):raise ValueError('evaluation updated model buffers')
    return predictions,z


def loss(logits,y,reg):return torch.nn.functional.cross_entropy(logits,y)+sum(reg)


def train_one(folder,data,model_name,seed,c,identity,logger=None):
    return train_model(lambda:build_model(model_name,c),lambda:loaders(data,c['training']['batch_size']),folder,
         seed=seed,training=c['training'],identity=identity,loss_fn=loss,threads=c['torch_num_threads'],device=c['device'],logger=logger)


def execute(config_path=None,*,smoke=False,resume=None):
    config_path=Path(config_path or ROOT/'configs/v2/phase12_execution.yaml').resolve();c=runtime_config(config_path)
    gates=None if smoke else check_gates(ROOT,c)
    if smoke:
        c=deepcopy(c);c.update(device='cpu',torch_num_threads=1,seeds=[2022]);c['training'].update(max_epochs=2,batch_size=16)
    base=ROOT/c['output'];base.mkdir(parents=True,exist_ok=True)
    with FileLock(str(base/'.execution.lock')):
        folder=Path(resume).resolve() if resume else base/('smoke' if smoke else 'attempts')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        if not folder.is_relative_to(base):raise ValueError('resume escapes registered output')
        sources=source_files(config_path,formal=not smoke)
        if not smoke:
            source_path=Path(gates['phase6_data_attempt'])/'legacy_identity/snr_ridge_state.json'
            sources[str(source_path)]=sha(source_path)
            sources.update(gates['source_files'])
        env={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','scikit-learn')}
        env.update(device=c['device'],cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if c['device']=='cuda' else None)
        protocol=dict(config=c,smoke=smoke,sources=sources,environment=env,gates=gates)
        if (folder/'manifest.json').exists():
            if load(folder/'protocol.json')!=protocol:raise ValueError('completed execution identity differs')
            validate(folder,scientific=not smoke);return folder
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'protocol.json').exists():
            if load(folder/'protocol.json')!=protocol:raise ValueError('resume protocol/source/environment differs')
        else:dump(folder/'protocol.json',protocol)
        data_path=folder/'data'
        if not data_path.exists():assemble(data_path,counts={'train':2,'validation':1,'test':1} if smoke else None,snr_indices=[0,7,10,19] if smoke else None,scientific=not smoke,execution_config=None if smoke else c)
        validate_dataset(data_path,replay=True);data=load_dataset(data_path);i=data['identity']
        estimator=folder/'frame_estimator'
        if estimator.exists():
            with np.load(estimator/'features_predictions.npz',allow_pickle=False) as b:x=b['features'];frame=b['predictions']
            state=load(estimator/'state.json')
            np.testing.assert_array_equal(frame,predict_estimator(x,state))
        else:x,frame,state=fit_estimator(data,c['ridge_alphas'],estimator)
        source=None
        if not smoke:
            source=predict_estimator(x,load(source_path))
            dump(folder/'frozen_source10a_state.json',load(source_path))
        choices=evaluation_conditions(data,frame,source)
        equal_bundle(folder/'conditions.npz',{f'{name}__{key}':np.asarray(value) for name,choice in choices.items() for key,value in choice.items() if value is not None})
        rows=[];pairing=[]
        for model_name in c['models']:
            for seed in c['seeds']:
                run=folder/'runs'/f'{model_name}_seed{seed}';identity=digest(dict(protocol=protocol,dataset_manifest=sha(data_path/'manifest.json'),model=model_name,seed=seed))
                if not (run/'manifest.json').exists():
                    def log(row):
                        with (run/'training.log').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                        print(f'{model_name} seed {seed}: {json.dumps(row)}',flush=True)
                    outcome=train_one(run,data,model_name,seed,c,identity,log);dump(run/'training.json',outcome)
                else:verify(run)
                model=build_model(model_name,c).to(c['device']);model.load_state_dict(torch.load(run/'checkpoint.pt',map_location=c['device'],weights_only=True),strict=True)
                predictions={}
                for channel in ('A','B'):
                    ix=i[f'test_{channel}_indices'];pred,z=eval_model(model,data,ix,choices,c['evaluation_batch_size']);predictions[channel]=pred
                    values=dict(sample_ids=i['sample_ids'][ix],waveform_ids=i['waveform_ids'][ix],labels=i['labels'][ix],snr_db=i['snr_db'][ix],**{name+'_logits':v for name,v in pred.items()})
                    if z is not None:values['pooled_features']=z
                    equal_bundle(run/f'test_{channel}.npz',values)
                    for name,logits in pred.items():
                        local={key:(value[ix] if isinstance(value,np.ndarray) else value) for key,value in choices[name].items()}
                        rows.append(dict(model=model_name,seed=seed,channel=channel,condition=name,metrics=summarize(i['labels'][ix],logits,i['snr_db'][ix],local)))
                for name in choices:
                    a,b=(predictions[ch][name].argmax(1) for ch in ('A','B'))
                    pairing.append(dict(model=model_name,seed=seed,condition=name,count=len(a),prediction_changed=int(np.sum(a!=b)),paired_waveform_ids_sha256=digest(i['waveform_ids'][i['test_A_indices']].tolist())))
                if not (run/'manifest.json').exists():seal(run,'NON_SCIENTIFIC_SMOKE' if smoke else 'TRAINED_EVALUATED',sources,identity=identity)
        dump(folder/'rows.json',rows);dump(folder/'paired.json',paired(rows,c['seeds']));dump(folder/'paired_payloads.json',pairing)
        figures(folder,rows,not smoke)
        dump(folder/'summary.json',dict(status='NON_SCIENTIFIC_SMOKE' if smoke else 'COMPLETE',training_runs=len(c['models'])*len(c['seeds']),evaluation_rows=len(rows),observations=len(i['labels']),latent_frames=len(np.unique(i['waveform_ids'])),source10a_status='NOT_EXERCISED_IN_SMOKE' if smoke else 'FROZEN_NO_REFIT',distribution_shift_scope='independent_defined_waveforms_A_vs_B_only_no_general_fading_claim'))
        seal(folder,'NON_SCIENTIFIC_SMOKE' if smoke else 'COMPLETE',sources)
        validate(folder,scientific=not smoke,replay_models=False)
        if not smoke:dump(base/'current.json',dict(status='COMPLETE',attempt=folder.relative_to(ROOT).as_posix(),manifest_sha256=sha(folder/'manifest.json'),validation=dict(status='VALIDATED')))
        print(folder,flush=True);return folder


def validate(folder,*,scientific=True,replay_models=True):
    folder=Path(folder);m=verify(folder);p=load(folder/'protocol.json');c=p['config'];summary=load(folder/'summary.json')
    if m['status']!=('COMPLETE' if scientific else 'NON_SCIENTIFIC_SMOKE') or p['smoke']==scientific:raise ValueError('scientific/smoke scope mismatch')
    if scientific:
        gates=check_gates(ROOT,c)
        if gates!=p['gates'] or summary['training_runs']!=15 or summary['evaluation_rows']!=210:raise ValueError('formal gate/matrix differs')
    validate_dataset(folder/'data',replay=False)
    expected=len(c['models'])*len(c['seeds'])*2*(7 if scientific else 6)
    rows=load(folder/'rows.json')
    if len(rows)!=expected or summary['evaluation_rows']!=expected:raise ValueError('evaluation matrix incomplete')
    if load(folder/'paired.json')!=paired(rows,c['seeds']):raise ValueError('paired statistics differ')
    data=load_dataset(folder/'data');i=data['identity']
    with np.load(folder/'frame_estimator/features_predictions.npz',allow_pickle=False) as b:features=b['features'];estimated=b['predictions'];saved_ids=b['sample_ids'];fit_ids=b['fit_sample_ids'];val_ids=b['validation_sample_ids']
    np.testing.assert_array_equal(saved_ids,i['sample_ids']);np.testing.assert_array_equal(fit_ids,i['sample_ids'][i['train_indices']]);np.testing.assert_array_equal(val_ids,i['sample_ids'][i['validation_indices']])
    from .execution import frame_features
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    state=load(folder/'frame_estimator/state.json');selection=load(folder/'frame_estimator/selection.json')
    for start in range(0,len(features),128):np.testing.assert_array_equal(features[start:start+128],frame_features(data['signals'][start:start+128]))
    tr,va=i['train_indices'],i['validation_indices'];scaler=StandardScaler().fit(features[tr]);y=i['snr_db'];candidates=[];fitted=[]
    for alpha in c['ridge_alphas']:
        ridge=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(features[tr]),y[tr]);fitted.append(ridge)
        candidates.append(dict(alpha=float(alpha),validation_mae=float(abs(ridge.predict(scaler.transform(features[va]))-y[va]).mean())))
    winner=min(range(len(candidates)),key=lambda n:(candidates[n]['validation_mae'],-c['ridge_alphas'][n]))
    if candidates!=selection['candidates'] or selection['selected_alpha']!=c['ridge_alphas'][winner] or selection['refit_train_validation'] is not False:raise ValueError('train-only estimator selection differs')
    for name,expected_value in [('feature_mean',scaler.mean_),('feature_scale',scaler.scale_),('coef',fitted[winner].coef_),('intercept',fitted[winner].intercept_)]:np.testing.assert_array_equal(state[name],expected_value)
    np.testing.assert_array_equal(estimated,predict_estimator(features,state))
    source=predict_estimator(features,load(folder/'frozen_source10a_state.json')) if scientific else None
    choices=evaluation_conditions(data,estimated,source)
    equal_bundle(folder/'conditions.npz',{f'{name}__{key}':np.asarray(value) for name,choice in choices.items() for key,value in choice.items() if value is not None})
    replayed=[]
    for model in c['models']:
        for seed in c['seeds']:
            run=folder/'runs'/f'{model}_seed{seed}';verify(run)
            if load(run/'training.json')['epochs_completed']!=c['training']['max_epochs']:raise ValueError('training incomplete')
            network=None
            if replay_models:
                network=build_model(model,c).to(c['device']);network.load_state_dict(torch.load(run/'checkpoint.pt',map_location=c['device'],weights_only=True),strict=True)
            for channel in ('A','B'):
                ix=i[f'test_{channel}_indices']
                with np.load(run/f'test_{channel}.npz',allow_pickle=False) as b:values={k:b[k] for k in b.files}
                for key in ('sample_ids','waveform_ids','labels','snr_db'):np.testing.assert_array_equal(values[key],i[key][ix])
                predictions,z=eval_model(network,data,ix,choices,c['evaluation_batch_size']) if replay_models else (None,None)
                for name,choice in choices.items():
                    logits=values[name+'_logits']
                    if replay_models:np.testing.assert_array_equal(logits,predictions[name])
                    local={key:(value[ix] if isinstance(value,np.ndarray) else value) for key,value in choice.items()}
                    replayed.append(dict(model=model,seed=seed,channel=channel,condition=name,metrics=summarize(i['labels'][ix],logits,i['snr_db'][ix],local)))
    if replayed!=rows:raise ValueError('raw-array metric replay differs')
    return dict(status='VALIDATED',training_runs=summary['training_runs'],evaluation_rows=expected,scientific=scientific)

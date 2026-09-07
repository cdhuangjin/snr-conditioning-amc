"""Phase12 gated execution primitives; frozen core has no runtime overrides."""
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from v2.phase8.evidence import sha,load,safe,dump
from v2.phase8.core import conditions
from v2.phase6.audit import frame_features, array_hash
from v2.phase7.deterministic_padding import install_adapter,ADAPTER_ID
from .core import pilot_estimate,pilot_sequence
from .assembly import ROOT


def runtime_config(path=None):
    c=yaml.safe_load(Path(path or ROOT/'configs/v2/phase12_execution.yaml').read_text(encoding='utf-8'))
    if c['models']!=['M0','M6','CLDNN'] or c['seeds']!=list(range(2022,2027)) or c['pilot_counts']!=[0,8,16,32,64]:raise ValueError('registered model/seed/pilot matrix differs')
    if c['training']!={'optimizer':'Adam','lr':.001,'batch_size':128,'max_epochs':100} or c['architecture']['num_classes']!=6 or c['architecture']['num_snr_bins']!=20:raise ValueError('registered training differs')
    if c['runtime_adapter']!=ADAPTER_ID or c['formal_observations']!=144000 or c['formal_latent_frames']!=120000 or c['formal_training_runs']!=15:raise ValueError('registered data/runtime differs')
    if c['architecture']!={'num_classes':6,'num_levels':1,'in_channels':64,'kernel_size':3,'latent_dim':320,'regu_details':.01,'regu_approx':.01,'num_snr_bins':20,'snr_embedding_dim':8}:raise ValueError('registered architecture differs')
    if c['device'] not in ('cpu','cuda') or c['torch_num_threads']!=24 or c['evaluation_batch_size']!=64 or c['ridge_alphas']!=[.01,.1,1.,10.,100.]:raise ValueError('registered runtime/estimator differs')
    return c


def check_gates(root,c):
    root=Path(root).resolve()
    try:
        path=safe(root,c['phase11_current']);current=load(path);attempt=safe(root,current['attempt'])
        m=load(attempt/'manifest.json')
        if current['status']!='COMPLETE' or sha(attempt/'manifest.json')!=current['manifest_sha256'] or m['status']!='COMPLETE':raise ValueError('Phase11 incomplete or unbound')
        if m['distribution_shift_status']!=c['required_phase11_shift_status']:raise ValueError('Phase11 scientific scope differs')
        from v2.phase11.runner import validate
        c11=load(attempt/'config.json');expected_runs=10*len(c11['datasets']);expected_rows=15*len(c11['datasets'])
        result=validate(attempt,root)
        if result['status']!='VALIDATED' or result['training_runs']!=expected_runs or result['evaluation_rows']!=expected_rows:raise ValueError('Phase11 matrix is not validated')
        from v2.phase6.composite import validate_composite
        phase6=validate_composite(root,load(safe(root,c['phase6_current'])))
        return dict(phase11_attempt=str(attempt),phase11_current=current,phase11_validation=result,
                    phase6_data_attempt=str(phase6),source_files={str(path):sha(path),str(attempt/'manifest.json'):sha(attempt/'manifest.json'),str(safe(root,c['phase6_current'])):sha(safe(root,c['phase6_current']))})
    except (OSError,KeyError,TypeError,ImportError) as exc:raise ValueError(f'Phase12 gate missing/malformed: {exc}') from exc


class AgnosticCLDNN(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def forward_batch(self,x,*,snr_db,snr_bin):
        # Deliberately discard both condition tensors; they never enter classifier.
        return self.backbone(x)


def build_model(name,c):
    if name=='CLDNN':
        source=ROOT.parents[1]/'experiments/train_baselines.py'
        if sha(source)!=c['baseline_source_sha256']:raise ValueError('reviewed CLDNN source hash differs')
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='CLDNN')
        namespace={'nn':nn,'torch':torch}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        return AgnosticCLDNN(namespace['CLDNN'](num_classes=6))
    if name not in ('M0','M6'):raise ValueError('unknown registered model')
    from models.model_conditioning import AWNConditioned
    model=AWNConditioned(conditioning=name,**c['architecture'],snr_min_db=-20.,snr_max_db=18.)
    if not install_adapter(model):raise ValueError('required deterministic adapter found no modules')
    return model


class Frames(torch.utils.data.Dataset):
    def __init__(self,data,indices):self.data=data;self.indices=np.asarray(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        row=self.indices[index];i=self.data['identity'];db=i['snr_db'][row]
        return torch.from_numpy(np.array(self.data['signals'][row],copy=True)),int(i['labels'][row]),np.float32(db),np.int64((db+20)/2)


def loaders(data,batch):
    return tuple(torch.utils.data.DataLoader(Frames(data,data['identity'][name+'_indices']),batch_size=batch,shuffle=True,num_workers=0,drop_last=False) for name in ('train','validation'))


def fit_estimator(data,alphas,output):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    x=np.empty((len(data['signals']),7),np.float64)
    for start in range(0,len(x),128):x[start:start+128]=frame_features(data['signals'][start:start+128])
    i=data['identity'];tr,va=i['train_indices'],i['validation_indices'];y=i['snr_db']
    if np.any(i['channels'][tr]!='A') or np.any(i['channels'][va]!='A') or np.intersect1d(tr,va).size:raise ValueError('estimator partition leakage')
    scaler=StandardScaler().fit(x[tr]);tx,vx=scaler.transform(x[tr]),scaler.transform(x[va]);candidates=[];models=[]
    for alpha in alphas:
        model=Ridge(alpha=alpha,solver='svd').fit(tx,y[tr]);models.append(model)
        candidates.append(dict(alpha=float(alpha),validation_mae=float(np.mean(abs(model.predict(vx)-y[va])))))
    winner=min(range(len(alphas)),key=lambda n:(candidates[n]['validation_mae'],-alphas[n]));model=models[winner]
    state=dict(feature_mean=scaler.mean_.tolist(),feature_scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_),fit_count=len(tr),fit_partition='train_A',fit_ids_sha256=array_hash(i['sample_ids'][tr]),estimator_scope='supervised_dataset_trained_statistics_not_physical_pilot')
    selection=dict(candidates=candidates,selected_alpha=float(alphas[winner]),fit_partition='train_A',selection_partition='validation_A',refit_train_validation=False,train_features_sha256=array_hash(x[tr]),train_targets_sha256=array_hash(y[tr]),validation_features_sha256=array_hash(x[va]),validation_targets_sha256=array_hash(y[va]))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # Freeze fitted state/selection before computing deployment predictions.
    dump(output/'state.json',state);dump(output/'selection.json',selection)
    pred=predict_estimator(x,state)
    np.savez_compressed(output/'features_predictions.npz',features=x,predictions=pred,sample_ids=i['sample_ids'],snr_db=y,fit_sample_ids=i['sample_ids'][tr],validation_sample_ids=i['sample_ids'][va])
    return x,pred,state


def predict_estimator(features,state):
    return ((features-np.asarray(state['feature_mean']))/np.asarray(state['feature_scale']))@np.asarray(state['coef'])+state['intercept']


def evaluation_conditions(data,frame_predictions,source_predictions=None):
    truth=data['identity']['snr_db'];n=len(truth)
    if np.shape(frame_predictions)!=(n,):raise ValueError('unaligned frame estimator')
    choices={name:dict(raw_db=np.asarray(values),clipped_db=conditions(values)[0],K=k,estimator=name) for name,values,k in [('oracle',truth,None),('K0_frame_estimator',frame_predictions,0)]}
    if source_predictions is not None:choices['source10a_frozen']=dict(raw_db=np.asarray(source_predictions),clipped_db=conditions(source_predictions)[0],K=None,estimator='frozen_source10a_uncalibrated_shift')
    for k in (8,16,32,64):
        chunks=[]
        for start in range(0,n,128):chunks.append(pilot_estimate(data['pilot'][start:start+128,-k:],pilot_sequence(k)))
        values={name:np.concatenate([np.atleast_1d(chunk[name]) for chunk in chunks]) for name in chunks[0] if name not in ('pilot_count','pilot_energy')}
        values['raw_pilot_db']=values.pop('raw_db');values['raw_db']=values['guarded_db'].copy()
        values.update(K=k,estimator='observed_pilot_constant_complex_LS',pilot_energy=float(k))
        choices[f'K{k}_pilot']=values
    return choices

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from v2.phase7.amssb_followup import prepare_amssb,distribution_metrics,align_predictions


def test_fixed_partition_filter_and_identity_mapping():
    y=np.tile(np.arange(11),3);x=np.arange(66*8,dtype=np.float32).reshape(33,2,8)
    split=SimpleNamespace(train_idx=np.arange(11),val_idx=np.arange(11,22),test_idx=np.arange(22,33),sample_ids=np.arange(33)+100,split_hash='parent')
    data,new,meta=prepare_amssb({'labels':y,'signals':x,'snrs':np.zeros(33)},split)
    assert data['signals'] is x
    np.testing.assert_array_equal(new.test_idx,np.arange(22,32))
    assert meta['class_mapping']==list(range(10))+[-1]
    assert data['labels'][32]==-1 and np.all(data['labels'][new.test_idx]>=0)


def test_original_output_errors_and_id_alignment():
    ids=np.array([3,1,2]);pred=np.array([10,0,1])
    aligned=align_predictions(ids,pred,np.array([1,2,3]))
    np.testing.assert_array_equal(aligned,[0,1,10])
    m=distribution_metrics(np.array([0,1,2]),aligned,11)
    assert m['accuracy']==pytest.approx(2/3)
    assert m['counts'][10]==1 and m['balanced_accuracy']==pytest.approx(2/3)
    with pytest.raises(ValueError):align_predictions(ids,pred,np.array([4]))


def test_real_awn_filtered_cpu_resume(tmp_path):
    import sys,torch
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts/v2'))
    import run_phase3 as p3
    from run_phase7_controls import build_control_model
    from phase1_reproduce import make_historical_train_val_loaders
    from v2.phase7.controls import train_model,file_hash
    from v2.phase7.deterministic_padding import ADAPTER_ID
    cfg=p3.load_phase3_config(root/'configs/v2/phase3.yaml');cfg['architecture']['num_classes']=10
    cfg['runtime_adapter']={'id':ADAPTER_ID,'source_sha256':file_hash(root/'v2/phase7/deterministic_padding.py')}
    y=np.tile(np.arange(11),39);x=np.random.default_rng(4).normal(size=(429,2,128)).astype('float32')
    original=SimpleNamespace(train_idx=np.arange(143),val_idx=np.arange(143,286),test_idx=np.arange(286,429),sample_ids=np.arange(429),split_hash='smoke')
    d,s,_=prepare_amssb(dict(signals=x,labels=y,snrs=np.zeros(429,dtype='float32')),original)
    def loaders():
        tensors=[torch.from_numpy(d[k]) for k in ('signals','labels','snrs')]
        tr,va=[p3.make_conditioning_dataset(*tensors,idx) for idx in (s.train_idx,s.val_idx)]
        return make_historical_train_val_loaders(tr,va,train_batch_size=128,validation_batch_size=128)
    args=dict(model_factory=lambda:build_control_model(p3,cfg),loader_factory=loaders,seed=2022,training=dict(lr=.001,batch_size=128,max_epochs=2),identity='non_scientific_amssb_smoke',loss_fn=p3._training_loss,threads=2,device='cpu')
    train_model(destination=tmp_path/'full',**args)
    def interrupt(row):raise RuntimeError('smoke interruption after saved epoch')
    with pytest.raises(RuntimeError,match='smoke interruption'):train_model(destination=tmp_path/'resume',logger=interrupt,**args)
    train_model(destination=tmp_path/'resume',**args)
    a=torch.load(tmp_path/'full/resume.pt',weights_only=True);b=torch.load(tmp_path/'resume/resume.pt',weights_only=True)
    assert a['next_epoch']==b['next_epoch']==2
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    assert torch.equal(a['torch_rng'],b['torch_rng'])


def test_summary_zero_variance_and_common_identity(tmp_path):
    import sys,json
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts/v2'))
    from run_phase7_amssb_followup import summarize_followup
    from v2.phase7.controls import SEEDS,atomic_json,write_manifest
    dirs=[];baselines=[];rows=[]
    for seed in SEEDS:
        d=tmp_path/f'run{seed}';d.mkdir();dirs.append(d)
        ids=np.array([100,101,102]);y=np.array([0,3,1]);snr=np.array([-20,-20,0]);logits=np.zeros((3,10));logits[:,0]=1
        np.savez(d/'predictions.npz',sample_ids=ids,y_true=y,snr_db=snr,logits=logits)
        write_manifest(d,dict(control='remove_amssb',seed=seed,epochs_completed=100),[])
        original=tmp_path/f'baseline{seed}.npz';np.savez(original,sample_ids=ids,y_true=y,snr_db=snr,logits=np.c_[logits,np.zeros(3)])
        baselines.append(SimpleNamespace(seed=seed,predictions_path=original))
        w=tmp_path/f'wbfm{seed}';w.mkdir();np.savez(w/'predictions.npz',sample_ids=ids[[0,2]],y_true=y[[0,2]],snr_db=snr[[0,2]],logits=logits[[0,2]])
        rows.append(dict(seed=seed,control='remove_wbfm',path=str(w)))
    summarize_followup(dirs,baselines,{'runs':rows},tmp_path/'summary',[],{})
    result=json.loads((tmp_path/'summary/summary.json').read_text())
    assert result['paired_low_snr']['retained_concentration']['ci95_low'] is None
    assert result['paired_low_snr']['retained_concentration']['undefined']['paired_inference']=='zero_difference_variance'
    assert result['runs'][0]['common_wbfm_removed']['all']['num_outputs']==10

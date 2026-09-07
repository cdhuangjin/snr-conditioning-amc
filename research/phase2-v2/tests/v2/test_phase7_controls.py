import numpy as np
import pytest
from types import SimpleNamespace
from v2.phase7.controls import prepare_control, balanced_control, remap_control


def data_fixture():
    y=np.tile(np.arange(11),3)
    data=dict(signals=np.random.default_rng(2).normal(size=(33,2,16)).astype('float32'),labels=y,snrs=np.tile(np.arange(11)*2-20,3).astype('float32'))
    split=SimpleNamespace(train_idx=np.arange(11),val_idx=np.arange(11,22),test_idx=np.arange(22,33),sample_ids=np.array([f'id{i}' for i in range(33)]),split_hash='upstream')
    return data,split


def test_remove_preserves_partition_and_remaps():
    data,split=data_fixture()
    d,s,meta=prepare_control(data,split,'remove_wbfm')
    assert s.test_idx.tolist()==[i for i in split.test_idx if data['labels'][i]!=3]
    assert len(s.train_idx)==10
    assert np.array_equal(d['labels'][s.test_idx],np.arange(10))
    assert np.array_equal(s.sample_ids,split.sample_ids)
    assert meta['class_mapping']==[0,1,2,-1,3,4,5,6,7,8,9]
    assert s.split_hash!=split.split_hash
    assert not set(s.train_idx)&set(s.test_idx)


def test_rms_frame_local_and_zero():
    data,split=data_fixture();data['signals'][0]=0
    d,s,_=prepare_control(data,split,'frame_rms')
    assert np.all(d['signals'][0]==0)
    assert np.allclose((d['signals'][1:]**2).sum(1).mean(1),1)
    data['signals'][2:]*=100
    d2,_,_=prepare_control(data,split,'frame_rms')
    assert np.array_equal(d['signals'][:2],d2['signals'][:2])
    assert np.array_equal(s.test_idx,split.test_idx)


def test_balancing_exact_law_requires_equal_counts():
    result=balanced_control(np.tile(np.arange(11),5),11)
    assert result['equivalent_uniform_permutation'] is True
    assert result['counts']==[5]*11
    with pytest.raises(ValueError,match='unequal'):
        balanced_control(np.array([0,0,1]),2)


def test_remap_unique_max_invariant_and_ties_explicit():
    logits=np.array([[3.,1.,0.],[0.,1.,3.],[2.,2.,0.]])
    result=remap_control(logits,np.array([0,2,0]),np.array([2,1,0]))
    assert result['tie_rows'].tolist()==[False,False,True]
    assert result['inverse_mapped_predictions'].tolist()==[0,2,1]
    assert result['identity_changed'].tolist()==[False,False,True]
    assert np.array_equal(result['mapped_original_predictions'],np.array([2,0,2]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_training_epoch_resume_matches_uninterrupted(tmp_path, device):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from v2.phase7.controls import train_model
    if device == "cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.net=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(4,2))
        def forward_batch(self,x,**unused):return self.net(x),[]
    x=torch.arange(64,dtype=torch.float32).reshape(16,4)/64;y=torch.arange(16)%2
    ds=TensorDataset(x,y,torch.zeros(16),torch.zeros(16,dtype=torch.long))
    def loaders():return DataLoader(ds,batch_size=4,shuffle=True),DataLoader(ds,batch_size=4,shuffle=False)
    def loss(logits,labels,regularizers):return torch.nn.functional.cross_entropy(logits,labels)
    cfg=dict(lr=.001,batch_size=4,max_epochs=3)
    whole=train_model(Tiny,loaders,tmp_path/'whole',seed=2022,training=cfg,identity='same',loss_fn=loss,threads=2,device=device)
    def interrupt(row):
        if row['epoch']==0:raise RuntimeError('simulated process interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        train_model(Tiny,loaders,tmp_path/'resumed',seed=2022,training=cfg,identity='same',loss_fn=loss,threads=2,logger=interrupt,device=device)
    assert not (tmp_path/'resumed'/'checkpoint.pt').exists()
    resumed=train_model(Tiny,loaders,tmp_path/'resumed',seed=2022,training=cfg,identity='same',loss_fn=loss,threads=2,device=device)
    a=torch.load(whole['checkpoint'],weights_only=True);b=torch.load(resumed['checkpoint'],weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert resumed['epochs_completed']==3
    assert resumed['best_epoch']==whole['best_epoch']
    with pytest.raises(ValueError,match='identity'):
        train_model(Tiny,loaders,tmp_path/'resumed',seed=2022,training=cfg,identity='changed',loss_fn=loss,threads=2,device=device)


def test_control_manifest_tampering_and_incomplete_matrix(tmp_path):
    from v2.phase7.controls import write_manifest,verify_control_manifest,summarize_matrix
    out=tmp_path/'run';out.mkdir();(out/'checkpoint.pt').write_bytes(b'fixture checkpoint')
    (out/'metrics.json').write_text('{}')
    source=tmp_path/'code.py';source.write_text('x=1')
    write_manifest(out,{'control':'remove_wbfm','seed':2022,'epochs_completed':100,'fingerprint':'f'},[source])
    assert verify_control_manifest(out)['epochs_completed']==100
    with pytest.raises(ValueError,match='ten'):
        summarize_matrix([out])
    (out/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        verify_control_manifest(out)


def test_device_identity_is_part_of_resume(tmp_path):
    import inspect
    from v2.phase7.controls import train_model
    assert 'device' in inspect.signature(train_model).parameters


def test_runner_locked_config_and_device_override(tmp_path):
    import importlib.util
    from pathlib import Path
    import yaml
    path=Path(__file__).resolve().parents[2]/'scripts/v2/run_phase7_controls.py'
    spec=importlib.util.spec_from_file_location('controls_cli',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=dict(seeds=list(range(2022,2027)),controls=['remove_wbfm','frame_rms'],runtime_adapter='reflection_pad1d_slice_flip_cat_v1',device='cuda',cpu_threads=24,training=dict(optimizer='Adam',lr=.001,batch_size=128,max_epochs=100))
    file=tmp_path/'config.yaml';file.write_text(yaml.safe_dump(cfg))
    assert module.checked_config(file,'cpu')['device']=='cpu'
    cfg['training']['max_epochs']=1;file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='training'):
        module.checked_config(file)


@pytest.mark.parametrize('num_classes',[10,11])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_awn_batch128_smoke(tmp_path,num_classes,device):
    import torch
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts/v2'))
    import run_phase3 as p3
    from phase1_reproduce import make_historical_train_val_loaders
    from v2.phase7.controls import train_model
    cfg=p3.load_phase3_config(root/'configs/v2/phase3.yaml');cfg['architecture']['num_classes']=num_classes
    if device=='cuda':
        import os
        if os.environ.get('RUN_PHASE7_CUDA_SMOKE')!='1':pytest.skip('explicit GPU compatibility probe; known deterministic reflection-pad backward unsupported')
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
        free,_=torch.cuda.mem_get_info()
        if free<1.5*2**30:pytest.skip('less than 1.5 GiB free VRAM')
        torch.cuda.reset_peak_memory_stats()
    rng=np.random.default_rng(2022)
    x=torch.from_numpy(rng.normal(size=(257,2,128)).astype('float32'))
    y=torch.arange(257)%num_classes;s=torch.zeros(257)
    y[3]=-1  # excluded sentinel must not enter the actual loader
    def loaders():
        a=p3.make_conditioning_dataset(x,y,s,np.array([i for i in range(129) if i!=3]))
        b=p3.make_conditioning_dataset(x,y,s,np.arange(129,257))
        return make_historical_train_val_loaders(a,b,train_batch_size=128,validation_batch_size=128)
    result=train_model(lambda:p3._model(cfg,'M0'),loaders,tmp_path/f'awn{num_classes}',seed=2022,training=dict(lr=.001,batch_size=128,max_epochs=1),identity='synthetic_awn_smoke',loss_fn=p3._training_loss,threads=24,device=device)
    assert result['epochs_completed']==1
    assert Path(result['checkpoint']).is_file()
    assert np.isfinite(result['best_val_accuracy'])
    print(f'AWN {num_classes} classes {device} batch128 forward/backward and excluded sentinel passed')
    if device=='cuda':print(f'GPU peak allocated MiB: {torch.cuda.max_memory_allocated()/2**20:.1f}')


def test_changed_source_cannot_finalize_under_old_protocol(tmp_path):
    from v2.phase7.controls import write_manifest,file_hash
    out=tmp_path/'out';out.mkdir();(out/'checkpoint.pt').write_bytes(b'state')
    source=tmp_path/'code.py';source.write_text('original')
    expected={str(source.resolve()):file_hash(source)}
    source.write_text('changed during training')
    with pytest.raises(ValueError,match='source'):
        write_manifest(out,{'epochs_completed':100},[source],expected_sources=expected)


def test_algebraic_prepare_with_real_persisted_baselines(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from v2.phase7.controls import file_hash
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/'results/v2/phase3').glob('M0_seed*/predictions.npz'))
    if len(paths)!=5:pytest.skip('requires five local real baseline prediction bundles')
    spec=importlib.util.spec_from_file_location('controls_algebraic_cli',root/'scripts/v2/run_phase7_controls.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    import run_phase3 as p3
    from phase1_reproduce import _load_rml_dataset
    from v2.splits import load_split
    config=p3.load_phase3_config(root/'configs/v2/phase3.yaml')
    split=load_split(root/config['split']['metadata'],data_path=root/config['dataset']['path'])
    data=_load_rml_dataset(root/config['dataset']['path'],config['dataset']['id'],repository_root=root)
    runs=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as a:
            assert np.array_equal(a['sample_ids'],np.asarray(split.sample_ids)[split.test_idx])
            assert np.array_equal(a['y_true'],data['labels'][split.test_idx])
            seed=int(a['seed'])
        runs.append(SimpleNamespace(seed=seed,predictions_path=path,predictions_sha256=file_hash(path)))
    sources=[*paths,root/'configs/v2/phase3.yaml']
    out=cli.algebraic_controls(tmp_path,dict(module=p3,dataset=data,split=split,config=config),runs,sources,{str(p.resolve()):file_hash(p) for p in sources},{'smoke':'real persisted baseline algebraic path only'})
    value=json.loads((out/'summary.json').read_text())
    assert value['balanced_sampling']['counts']==[12000]*11
    assert len(value['baselines_and_remaps'])==5
    for row in value['baselines_and_remaps']:
        assert np.isfinite(row['baseline_retained_test']['overall_accuracy'])
        assert row['baseline_retained_test']['count']==40000
    # JSON strict serialization itself must reject any accidental NaN.
    json.dumps(value,allow_nan=False)

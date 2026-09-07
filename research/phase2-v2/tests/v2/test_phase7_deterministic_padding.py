import copy
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from v2.phase7.deterministic_padding import DeterministicReflectionPad1d,install_adapter


@pytest.mark.parametrize('length,padding',[(2,0),(2,1),(5,(0,3)),(5,(3,0)),(5,(4,4)),(64,2)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.float16])
def test_forward_exact_and_cpu_random_cotangent_gradient(length,padding,dtype):
    torch.manual_seed(23)
    x=torch.randn(2,3,length,dtype=dtype,requires_grad=True)
    reference=torch.nn.ReflectionPad1d(padding)(x)
    actual=DeterministicReflectionPad1d(padding)(x)
    assert torch.equal(reference,actual)
    cotangent=torch.randn_like(reference)
    a=torch.autograd.grad(reference,x,cotangent,retain_graph=True)[0]
    b=torch.autograd.grad(actual,x,cotangent)[0]
    torch.testing.assert_close(a,b,atol=8*torch.finfo(dtype).eps,rtol=8*torch.finfo(dtype).eps)


def require_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    free,_=torch.cuda.mem_get_info()
    if free<1.5*2**30:pytest.skip('less than 1.5 GiB free VRAM')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def test_cuda_gradient_deterministic_and_matches_cpu():
    require_cuda()
    x=torch.randn(2,3,64,dtype=torch.float64)
    cot=torch.randn(2,3,68,dtype=torch.float64)
    cpu=x.clone().requires_grad_();baseline=torch.nn.ReflectionPad1d(2)(cpu)
    expected=torch.autograd.grad(baseline,cpu,cot)[0]
    results=[]
    for _ in range(2):
        gpu=x.cuda().requires_grad_();out=DeterministicReflectionPad1d(2)(gpu)
        results.append(torch.autograd.grad(out,gpu,cot.cuda())[0].cpu())
    assert torch.equal(results[0],results[1])
    torch.testing.assert_close(results[0],expected,atol=1e-14,rtol=1e-14)


def phase3():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts/v2'))
    import run_phase3
    return run_phase3,run_phase3.load_phase3_config(root/'configs/v2/phase3.yaml')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_awn_state_keys_and_eval_output_unchanged(device):
    p3,cfg=phase3();torch.set_num_threads(24);torch.manual_seed(2022)
    if device=='cuda':require_cuda()
    original=p3._model(cfg,'M0').eval().to(device);adapted=copy.deepcopy(original)
    names=install_adapter(adapted)
    assert len(names)==2
    assert list(original.state_dict())==list(adapted.state_dict())
    for k in original.state_dict():assert torch.equal(original.state_dict()[k],adapted.state_dict()[k])
    x=torch.randn(4,2,128,device=device);snr=torch.zeros(4,device=device);bins=torch.full((4,),10,dtype=torch.long,device=device)
    with torch.no_grad():
        a,ar=original.forward_batch(x,snr_db=snr,snr_bin=bins)
        b,br=adapted.forward_batch(x,snr_db=snr,snr_bin=bins)
    assert torch.equal(a,b)
    assert all(torch.equal(u,v) for u,v in zip(ar,br))
    assert install_adapter(adapted)==[]


def test_full_awn_cuda_batch128_and_epoch_resume(tmp_path):
    require_cuda();p3,cfg=phase3()
    from phase1_reproduce import make_historical_train_val_loaders
    from v2.phase7.controls import train_model
    rng=np.random.default_rng(2022)
    x=torch.from_numpy(rng.normal(size=(256,2,128)).astype('float32'))
    y=torch.arange(256)%11;snr=torch.zeros(256)
    def model():
        m=p3._model(cfg,'M0');install_adapter(m);return m
    def loaders():
        a,b=(p3.make_conditioning_dataset(x,y,snr,np.arange(start,start+128)) for start in (0,128))
        return make_historical_train_val_loaders(a,b,train_batch_size=128,validation_batch_size=128)
    training=dict(lr=.001,batch_size=128,max_epochs=2)
    torch.cuda.reset_peak_memory_stats()
    full=train_model(model,loaders,tmp_path/'full',seed=2022,training=training,identity='adapter_smoke',loss_fn=p3._training_loss,threads=24,device='cuda')
    def interrupt(row):
        if row['epoch']==0:raise RuntimeError('intentional epoch interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        train_model(model,loaders,tmp_path/'resumed',seed=2022,training=training,identity='adapter_smoke',loss_fn=p3._training_loss,threads=24,device='cuda',logger=interrupt)
    resumed=train_model(model,loaders,tmp_path/'resumed',seed=2022,training=training,identity='adapter_smoke',loss_fn=p3._training_loss,threads=24,device='cuda')
    a=torch.load(full['checkpoint'],weights_only=True);b=torch.load(resumed['checkpoint'],weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    final_a=torch.load(tmp_path/'full'/'resume.pt',weights_only=True,map_location='cpu')
    final_b=torch.load(tmp_path/'resumed'/'resume.pt',weights_only=True,map_location='cpu')
    assert all(torch.equal(final_a['model'][k],final_b['model'][k]) for k in final_a['model'])
    assert torch.equal(final_a['torch_rng'],final_b['torch_rng'])
    assert all(torch.equal(u,v) for u,v in zip(final_a['cuda_rng'],final_b['cuda_rng']))
    print(f'Deterministic adapted AWN batch128 CUDA peak allocated MiB: {torch.cuda.max_memory_allocated()/2**20:.1f}')
    assert torch.are_deterministic_algorithms_enabled()


def test_control_factory_binds_adapter_identity_and_module_paths():
    import importlib.util
    from v2.phase7.controls import file_hash
    from v2.phase7.deterministic_padding import ADAPTER_ID
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('padding_controls_cli',root/'scripts/v2/run_phase7_controls.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    p3,cfg=phase3()
    cfg['runtime_adapter']={'id':ADAPTER_ID,'source_sha256':file_hash(root/'v2/phase7/deterministic_padding.py')}
    model=cli.build_control_model(p3,cfg)
    paths=[name for name,m in model.named_modules() if isinstance(m,DeterministicReflectionPad1d)]
    assert len(paths)==2
    cfg['runtime_adapter']['module_paths']=['incorrect.path']
    with pytest.raises(ValueError,match='paths'):
        cli.build_control_model(p3,cfg)

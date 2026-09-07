"""Read-only primary-source/checkpoint audit; no fitting or test-set scoring."""
import ast
import inspect
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import FrameID, generate_frame
from .smoke import ROOT, sha, dump


def run():
    primary = ROOT.parents[1]
    source = primary/'experiments/train_baselines.py'
    checkpoint = primary/'experiments/2016.10a_cldnn/models/2016.10a_CLDNN.pkl'
    log = primary/'experiments/2016.10a_cldnn/log/log.txt'
    legacy_result = primary/'experiments/snr_cldnn.json'
    out = ROOT/'results/v2/phase12_independent_waveforms/baseline_audit'
    out.mkdir(parents=True, exist_ok=False)
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name == 'CLDNN')
    namespace = {'nn':nn,'torch':torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    cls = namespace['CLDNN']
    assert list(inspect.signature(cls.forward).parameters) == ['self','x']
    torch.set_num_threads(1)
    torch.manual_seed(20260906)
    x = torch.from_numpy(np.stack([generate_frame(FrameID('train',c,10,i))['nct'] for c in range(6) for i in range(2)]))
    old = cls(num_classes=11).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    old.load_state_dict(state, strict=True)
    fresh = cls(num_classes=6).eval()
    before = {k:v.clone() for k,v in fresh.named_buffers()}
    started = time.perf_counter()
    with torch.inference_mode():
        old_logits, _ = old(x)
        fresh_logits, _ = fresh(x)
        singles = torch.cat([fresh(row[None])[0] for row in x])
    elapsed = time.perf_counter()-started
    assert old_logits.shape == (12,11) and fresh_logits.shape == (12,6)
    assert torch.isfinite(old_logits).all() and torch.isfinite(fresh_logits).all()
    assert all(torch.equal(v,before[k]) for k,v in fresh.named_buffers())
    # Shape/backprop viability only. No optimizer step, no architecture selection from labels.
    fresh.zero_grad(set_to_none=True)
    fresh(x)[0].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fresh.parameters())
    np.savez_compressed(out/'forward_smoke.npz', inputs=x.numpy(), legacy_logits=old_logits.numpy(),
                        fresh_six_class_logits=fresh_logits.numpy(), fresh_single_logits=singles.numpy())
    result = dict(status='NON_SCIENTIFIC_IMPLEMENTATION_AUDIT', recommendation='CLDNN from primary experiments/train_baselines.py',
        reason='existing SNR-agnostic class with strict-compatible historical checkpoint and training log; selected without new dataset accuracy',
        checkpoint_strict_load='PASS', checkpoint_format='state_dict_only_no_embedded_source_or_split_hash',
        prior_result_status='HISTORICAL_UNREPLAYED_NOT_CURRENT_SCIENTIFIC_EVIDENCE',
        forward_arguments=['self','x'], legacy_shape=[12,11], fresh_shape=[12,6],
        finite_backward='PASS_NO_OPTIMIZER_STEP', evaluation_buffers_unchanged=True,
        batch_single_max_absolute=float((fresh_logits-singles).abs().max()),
        cpu_threads=1, forward_seconds=elapsed, parameters=sum(p.numel() for p in fresh.parameters()),
        future_use='reinitialize six-class model; do not transfer historical weights; freeze extracted class source hash in formal run',
        scope='implementation viability, not independently verified historical accuracy or assertion of state-of-the-art strength')
    dump(out/'review.json', result)
    dependencies = [source, checkpoint, log, legacy_result, Path(__file__), ROOT/'v2/phase12/core.py', ROOT/'v2/phase12/smoke.py']
    dump(out/'manifest.json', dict(status=result['status'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},dependencies={str(p):sha(p) for p in dependencies}))
    print(out)


if __name__ == '__main__':
    run()

"""Small artifact-producing smoke only: python -m v2.phase12.smoke.

No formal generator/trainer and no current.json publication are implemented.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import scipy
import torch
from torch import nn

from .core import FrameID, FIR, generate_frame, pilot_estimate, pilot_sequence, overhead

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run():
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(20260906)
    out = ROOT/'results/v2/phase12_independent_waveforms/smoke'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    records = [generate_frame(FrameID(part, label, snr, 0), channel)
               for part in ('train','validation','test') for label in range(6)
               for snr in (0,19) for channel in (('A','B') if part == 'test' else ('A',))]
    array_keys = ('clean','payload','pilot','payload_noise','pilot_noise','nct')
    np.savez_compressed(out/'waveforms.npz', **{k:np.stack([r[k] for r in records]) for k in array_keys},
                        waveform_ids=np.array([r['waveform_id'] for r in records]),
                        channels=np.array([r['channel'] for r in records]), fir_coefficients=FIR)
    metadata = []
    for record in records:
        item = {k:v for k,v in record.items() if k not in array_keys and k != 'channel_coefficient'}
        item['channel_coefficient_real_imag'] = [record['channel_coefficient'].real, record['channel_coefficient'].imag]
        metadata.append(item)
    dump(out/'metadata.json', metadata)
    estimates = {}
    for k in (8,16,32,64):
        result = pilot_estimate(np.stack([r['pilot'][-k:] for r in records]), pilot_sequence(k))
        for name, values in result.items():
            estimates[f'K{k}_{name}'] = np.asarray(values)
    np.savez_compressed(out/'pilot_estimates.npz', **estimates)
    # Independently recompute LS from persisted observations (not latent truth).
    with np.load(out/'waveforms.npz', allow_pickle=False) as saved:
        for k in (8,16,32,64):
            y, p = saved['pilot'][:,-k:], pilot_sequence(k)
            h = (p.conj()*y).sum(-1)/k
            var = (abs(y-h[:,None]*p)**2).sum(-1)/(k-1)
            if not np.array_equal(h, estimates[f'K{k}_coefficient']) or not np.array_equal(var, estimates[f'K{k}_noise_variance']):
                raise ValueError('saved observation LS replay differs')
        x = torch.from_numpy(saved['nct'].copy())
    from models.model import AWN
    model = AWN(num_classes=6).eval()
    with torch.inference_mode():
        logits, _ = model(x)
    if logits.shape != (len(records),6) or not torch.isfinite(logits).all():
        raise ValueError('AWN six-class forward smoke failed')
    np.save(out/'untrained_awn_logits.npy', logits.numpy())
    # Existing primary classes only, extracted via AST to avoid legacy module import side effects.
    primary_source = ROOT.parents[1]/'experiments/train_baselines.py'
    tree = ast.parse(primary_source.read_text(encoding='utf-8'))
    extracted = ast.Module(body=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('CLDNN','LSTMNet')], type_ignores=[])
    namespace = {'nn':nn, 'torch':torch}
    exec(compile(extracted, str(primary_source), 'exec'), namespace)
    baseline = {}
    for name in ('CLDNN','LSTMNet'):
        candidate = namespace[name](num_classes=6).eval()
        with torch.inference_mode():
            pred, _ = candidate(x[:4])
        if pred.shape != (4,6) or not torch.isfinite(pred).all():
            raise ValueError('existing baseline shape/finite smoke failed')
        baseline[name] = dict(status='EXISTING_IMPLEMENTATION_FORWARD_SMOKE_ONLY',
                              parameters=sum(p.numel() for p in candidate.parameters()),
                              source=str(primary_source), source_sha256=sha(primary_source))
    summary = dict(status='NON_SCIENTIFIC_SMOKE', frames=len(records), paired_test_frames=12,
                   pilot_observation_ls_replay='PASS', untrained_awn_shape=list(logits.shape),
                   candidate_baselines=baseline, formal_results=False, training_executed=False,
                   overhead_examples={str(k):overhead(k, .5) for k in (0,8,16,32,64)},
                   elapsed_seconds=time.perf_counter()-started,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__))
    dump(out/'summary.json', summary)
    dependencies = [Path(__file__), ROOT/'v2/phase12/core.py', ROOT/'configs/v2/phase12.yaml',
                    ROOT/'models/model.py', ROOT/'models/lifting.py', primary_source]
    dump(out/'manifest.json', dict(status='NON_SCIENTIFIC_SMOKE',
         files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
         dependencies={str(p):sha(p) for p in dependencies}))
    print(json.dumps({'status':summary['status'],'output':str(out),'frames':len(records)}, ensure_ascii=False))
    return out


if __name__ == '__main__':
    run()

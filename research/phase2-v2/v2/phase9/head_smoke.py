"""Real M3/M6 heads, synthetic caches; exercises all 545 intervention paths."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from models.model_conditioning import AWNConditioned
from v2.phase8.evidence import dump,sha
from .analysis import run_cliffs,figures


def run(destination,config_path):
    destination=Path(destination)
    if destination.exists():raise ValueError('new smoke directory required')
    destination.mkdir(parents=True);cache=destination/'synthetic_phase8';(cache/'caches').mkdir(parents=True)
    torch.set_num_threads(1);config=yaml.safe_load(Path(config_path).read_text())
    def model(_config,name):return AWNConditioned(num_classes=11,num_levels=1,in_channels=64,latent_dim=320,conditioning=name).eval()
    phase3=SimpleNamespace(_model=model,_load_state=lambda path,device:torch.load(path,map_location=device,weights_only=True))
    ids=np.array([f'synthetic:{i}' for i in range(22)]);features=np.random.default_rng(7).normal(size=(22,128)).astype(np.float32)
    data={'labels':np.arange(22)%11,'snrs':np.resize(np.arange(-20,20,2),22).astype(np.float32)};split=SimpleNamespace(test_idx=np.arange(22),sample_ids=ids);runs={}
    for name in ['M3','M6']:
        for seed in config['seeds']:
            torch.manual_seed(seed);path=cache/f'{name}_{seed}.pt';torch.save(model({},name).state_dict(),path);runs[(name,seed)]={'paths':{'checkpoint':path}}
            np.savez_compressed(cache/'caches'/f'{name}_{seed}.npz',raw_pooled_features=features,sample_ids=ids)
    estimates=data['snrs']+np.linspace(-3,3,22)
    rows=run_cliffs(destination,cache,phase3,{},runs,data,split,estimates,config)
    run_cliffs(destination,cache,phase3,{},runs,data,split,estimates,config,verify_only=True)
    for name in ['M3','M6']:
        zero=[r for r in rows if r['model']==name and r['kind']=='additive' and r['value']==0]
        assert all(r['prediction_disagreement']==0 and r['parameter_distance_mean']==0 for r in zero)
    for seed in config['seeds']:
        for boundary in range(-19,18,2):
            selected=[r for r in rows if r['kind']=='boundary' and r['seed']==seed and r['boundary_db']==boundary]
            left,mid,right=sorted(selected,key=lambda r:r['offset_db'])
            with np.load(destination/left['bundle']) as a,np.load(destination/mid['bundle']) as b,np.load(destination/right['bundle']) as c:
                np.testing.assert_array_equal(a['logits'],b['logits']);np.testing.assert_array_equal(c['condition_bin'],b['condition_bin']+1)
    # Plot source uses clearly synthetic comparison rows solely to exercise I/O.
    comparison=[{'role':'matched_primary','deployment':'estimated','arm':arm,'seed':seed,'metrics':rows[0]['metrics']} for arm in ['clean','generic','empirical_snr'] for seed in config['seeds']]
    dump(destination/'rows.json',comparison);figures(destination,comparison,rows)
    verdict=dict(status='SMOKE_NOT_FORMAL',synthetic_inputs=True,random_untrained_heads=True,interventions=len(rows),exact_replay=True,midpoint_lower_bin=True,zero_delta_identity=True)
    dump(destination/'verdict.json',verdict);dump(destination/'manifest.json',dict(status='SMOKE_NOT_FORMAL',files={p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}))
    return verdict


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination');parser.add_argument('--config',default='configs/v2/phase9.yaml');args=parser.parse_args();print(run(args.destination,args.config))

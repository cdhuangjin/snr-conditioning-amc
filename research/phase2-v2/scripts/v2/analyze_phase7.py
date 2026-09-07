"""Read-only upstream verification followed by an immutable diagnostic generation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v2.phase7.diagnostics import analyze_bundles,sha256,verify_manifest


def validate_gate(path,repository_root=ROOT):
    """Require the recursively verified bounded scientific composite, not flat status."""
    root=Path(repository_root).resolve();path=Path(path).resolve()
    current=json.loads(path.read_text(encoding='utf-8'))
    if current.get('kind')!='phase6_bounded_scientific_composite':
        raise ValueError('Phase 6 composite gate required; flat completion status is insufficient')
    from v2.phase6.composite import validate_composite
    validate_composite(root,path)
    # The composite verifier establishes exact identity with attempt/manifest.json.
    return (root/current['gate']['manifest_path']).resolve()


def gate_source_paths(manifest,repository_root=ROOT):
    from v2.phase6.composite import manifest_closure
    root=Path(repository_root).resolve()
    closure=manifest_closure(root,Path(manifest))
    return [root/name for name in closure]+[root/'v2/phase6/composite.py']


def load_phase4():
    path=ROOT/'scripts/v2/run_phase4.py'
    spec=importlib.util.spec_from_file_location('_phase7_phase4',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase6-gate',required=True)
    parser.add_argument('--config',default='configs/v2/phase7.yaml')
    parser.add_argument('--output',required=True,help='new immutable output directory')
    args=parser.parse_args()
    gate=Path(args.phase6_gate).resolve();gate_manifest=validate_gate(gate)
    config_path=(ROOT/args.config).resolve();cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if cfg['seeds']!=list(range(2022,2027)): raise ValueError('five locked seeds required')
    output=Path(args.output).resolve()
    if output.exists():
        verify_manifest(output)
        print('Existing diagnostic generation verified; controls remain pending.');return
    p4=load_phase4();p4cfg=p4.load_phase4_config(ROOT/'configs/v2/phase4.yaml')
    p4root=ROOT/p4cfg['outputs']['root'];pointer=p4root/'current.json'
    current=json.loads(pointer.read_text(encoding='utf-8'))
    attempt=(p4root/current['attempt']).resolve()
    if not attempt.is_relative_to(p4root.resolve()): raise ValueError('Phase 4 path escape')
    manifest=attempt/'manifest.json'
    if sha256(manifest)!=current['manifest_sha256']: raise ValueError('Phase 4 pointer hash differs')
    m=p4.validate_phase4_manifest(manifest,attempt,repository_root=ROOT)
    if m['status']!='COMPLETE': raise ValueError('Phase 4 incomplete')
    context=p4._strict_phase3_contract(ROOT,p4cfg)
    registered=p4.audit_phase3_registry(ROOT,p4cfg,_strict_context=context)
    split=context['split'];data=context['dataset'];idx=split.test_idx
    output.parent.mkdir(parents=True,exist_ok=True)
    # Raw input artifact remains alongside generation so its source hash is auditable.
    raw_path=output.parent/(output.name+'_fixed_raw.npz')
    if raw_path.exists(): raise ValueError('raw input path exists; choose a new generation name')
    np.savez_compressed(raw_path,signals=data['signals'][idx],sample_ids=np.asarray(split.sample_ids)[idx],y_true=data['labels'][idx],snr_db=data['snrs'][idx])
    runs=[]
    for r in registered:
        feature=attempt/'features'/f'{r.model_id}_seed{r.seed}.npz'
        runs.append(dict(model_id=r.model_id,seed=r.seed,predictions=str(r.predictions_path),features=str(feature) if feature.is_file() else None))
    sources=[gate,gate_manifest,*gate_source_paths(gate_manifest),config_path,Path(__file__),pointer,manifest,*[ROOT/p for p in m['dependencies']]]
    result=analyze_bundles(raw_path,runs,output,class_names=cfg['class_names'],split_hash=split.split_hash,source_paths=sources)
    verify_manifest(output)
    print(json.dumps({'status':result['status'],'output':str(output)}))


if __name__=='__main__': main()

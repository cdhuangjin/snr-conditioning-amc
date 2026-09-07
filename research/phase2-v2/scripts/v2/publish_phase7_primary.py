"""Publish the scoped primary receipt only after ten frozen controls replay."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v2.phase7.primary_receipt import load,require_matrix,publish_primary,validate_primary_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary')
    parser.add_argument('--diagnostics',default='results/v2/phase7_diagnostics/20260906T0110')
    parser.add_argument('--dominance-review',default='results/v2/phase7_dominance_review/20260906T0130')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        print(validate_primary_gate(ROOT,ROOT/'results/v2/phase7_controls/gate.json'));return
    if not args.summary:parser.error('--summary is required for publication')
    summary_path=ROOT/args.summary;require_matrix(ROOT,load(summary_path))
    from run_phase7_controls import source_context,checked_config,replay_run,file_hash,prepare_control,digest
    config_path=ROOT/'configs/v2/phase7_controls.yaml';config=checked_config(config_path)
    import torch
    torch.set_num_threads(config['cpu_threads'])
    p4,context,baselines,sources=source_context(ROOT/'results/v2/phase6_preprocessing/current.json',config_path)
    hashes={str(Path(p).resolve()):file_hash(p) for p in sources}
    base=ROOT/'results/v2/phase7_controls'
    with p4._Phase4Lock(base/'.controls.lock'):
        def replay(row):
            path=Path(row['path']);spec=load(path/'run_spec.json');protocol=spec['protocol']
            dataset,split,mapping=prepare_control(context['dataset'],context['split'],row['control'])
            if protocol['sources']!=hashes or protocol['config']!=config or protocol['split']!=mapping:raise ValueError('replay protocol changed')
            protocol_hash=digest(protocol);fingerprint=digest(dict(protocol_hash=protocol_hash,seed=row['seed']))
            if spec['protocol_hash']!=protocol_hash or spec['fingerprint']!=fingerprint:raise ValueError('invalid protocol fingerprint')
            identity=dict(seed=row['seed'],model_id='M0',split_hash=split.split_hash,preprocessing_hash=digest(mapping['preprocessing']),protocol_hash=protocol_hash)
            replay_run(path,context['module'],dataset,split,protocol['active_model_config'],identity,fingerprint)
        print(publish_primary(ROOT,summary_path,ROOT/args.diagnostics,ROOT/args.dominance_review,replay=replay,
            extra_sources=[Path(__file__),ROOT/'docs/phase7_primary_receipt_design.md']))


if __name__=='__main__':main()

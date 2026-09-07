import json
from pathlib import Path
import pytest
from v2.phase7.primary_receipt import snapshot_sources, validate_snapshots, require_matrix, validate_primary_gate


def test_snapshots_preserve_original_locator(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'manifest.json').write_text('{"old_schema":true}')
    leaf=raw/'values.csv';leaf.write_text('1,2')
    dest=tmp_path/'receipt';dest.mkdir()
    from v2.phase7.controls import file_hash
    dependencies=snapshot_sources(tmp_path,dest,{str(p):file_hash(p) for p in raw.iterdir()})
    assert 'raw/values.csv' in dependencies
    assert 'raw/manifest.json' not in dependencies
    validate_snapshots(tmp_path,dest)
    (raw/'manifest.json').write_text('{"old_schema":false}')
    with pytest.raises(ValueError,match='original'):validate_snapshots(tmp_path,dest)


def test_incomplete_matrix_and_boolean_gate_rejected(tmp_path):
    with pytest.raises(ValueError,match='ten'):require_matrix(tmp_path,{'runs':[]})
    with pytest.raises(ValueError):validate_primary_gate(tmp_path,{'status':'COMPLETE','allow_phase8':True})


def test_complete_scoped_receipt_and_original_drift(tmp_path):
    from v2.phase7.controls import atomic_json,file_hash,CONTROLS,SEEDS
    from v2.phase7.primary_receipt import publish_primary
    def raw(directory,identity,artifacts):
        directory.mkdir(parents=True)
        for name,value in artifacts.items():atomic_json(directory/name,value)
        atomic_json(directory/'manifest.json',dict(status='COMPLETE',sources={},artifacts={p.name:file_hash(p) for p in directory.iterdir()},**identity))
        return directory
    rows=[]
    for c in CONTROLS:
        for s in SEEDS:
            p=raw(tmp_path/f'{c}{s}',dict(control=c,seed=s,epochs_completed=100,protocol_hash=c),{'metrics.json':{'overall_accuracy':.5},'epochs.json':[{'epoch':i} for i in range(100)]})
            rows.append(dict(control=c,seed=s,path=str(p),manifest_sha256=file_hash(p/'manifest.json'),metrics={'overall_accuracy':.5}))
    a=raw(tmp_path/'algebra',{}, {'summary.json':{'baselines_and_remaps':[{'seed':s} for s in SEEDS]}})
    d=raw(tmp_path/'diagnostics',{}, {'summary.json':{}})
    dm=json.loads((d/'manifest.json').read_text());dm['status']='DIAGNOSTICS_COMPLETE_CONTROLS_PENDING';atomic_json(d/'manifest.json',dm)
    review=raw(tmp_path/'review',{}, {'summary.json':{}})
    rm=json.loads((review/'manifest.json').read_text());rm['status']='INDEPENDENT_DESCRIPTIVE_REVIEW_COMPLETE';atomic_json(review/'manifest.json',rm)
    summary=tmp_path/'summary.json';atomic_json(summary,dict(runs=rows,algebraic_path=str(a),algebraic_manifest_sha256=file_hash(a/'manifest.json')))
    # Source binding of implementation is required to stay within fixture root.
    import v2.phase7.primary_receipt as module
    from unittest.mock import patch
    implementation=tmp_path/'primary_receipt.py';implementation.write_text('fixture')
    calls=[]
    with patch.object(module,'__file__',str(implementation)):
        gate=publish_primary(tmp_path,summary,d,review,replay=lambda row:calls.append(row['seed']))
    assert len(calls)==10 and gate['full_phase7_complete'] is False
    validate_primary_gate(tmp_path,gate)
    manifest=Path(rows[0]['path'])/'manifest.json'
    manifest.write_text(manifest.read_text()+' ')
    with pytest.raises(ValueError,match='original'):validate_primary_gate(tmp_path,gate)

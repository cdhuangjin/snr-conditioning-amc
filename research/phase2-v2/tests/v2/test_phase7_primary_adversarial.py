"""Independent receipt-transport counterexamples, not scientific gate fixtures."""
from pathlib import Path
import pytest
from v2.phase7.controls import atomic_json,file_hash
from v2.phase7.primary_receipt import (write_component,raw_sources,load,
    validate_receipt_sources,validate_diagnostics_binding,validate_algebraic)


def test_deleted_leaf_is_rejected_even_after_receipt_hashes_are_self_consistent(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'values.bin').write_bytes(b'necessary original diagnostic values')
    atomic_json(raw/'manifest.json',dict(status='COMPLETE',sources={},artifacts={'values.bin':file_hash(raw/'values.bin')}))
    sources=raw_sources(raw,'COMPLETE');receipt=tmp_path/'receipt'
    write_component(tmp_path,receipt,sources,{'unit_transport_only':True})
    validate_receipt_sources(tmp_path,receipt,sources)
    manifest=load(receipt/'manifest.json');del manifest['dependencies']['raw/values.bin'];atomic_json(receipt/'manifest.json',manifest)
    # The artifact manifest remains internally hash-consistent; original closure must win.
    with pytest.raises(ValueError):validate_receipt_sources(tmp_path,receipt,sources)


def test_deleted_snapshot_is_rejected_after_index_and_file_hashes_are_reclosed(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'value').write_bytes(b'value')
    atomic_json(raw/'manifest.json',dict(status='COMPLETE',sources={},artifacts={'value':file_hash(raw/'value')}))
    sources=raw_sources(raw,'COMPLETE');receipt=tmp_path/'receipt';write_component(tmp_path,receipt,sources,{'unit_transport_only':True})
    manifest=load(receipt/'manifest.json');atomic_json(receipt/'snapshot_index.json',[])
    manifest['files']['snapshot_index.json']=file_hash(receipt/'snapshot_index.json');atomic_json(receipt/'manifest.json',manifest)
    with pytest.raises(ValueError):validate_receipt_sources(tmp_path,receipt,sources)


def test_gate_cannot_substitute_another_internally_valid_diagnostic_receipt(tmp_path):
    for name in ['diag_a','diag_b']:
        folder=tmp_path/name;folder.mkdir();atomic_json(folder/'decision.json',{'unit_transport_only':True})
        atomic_json(folder/'manifest.json',dict(status='COMPLETE',files={'decision.json':file_hash(folder/'decision.json')},dependencies={}))
    primary=tmp_path/'primary';primary.mkdir();atomic_json(primary/'decision.json',{'diagnostics_manifest_path':'diag_a/manifest.json'})
    atomic_json(primary/'manifest.json',dict(status='COMPLETE',files={'decision.json':file_hash(primary/'decision.json')},dependencies={'diag_a/manifest.json':file_hash(tmp_path/'diag_a/manifest.json')}))
    gate={'diagnostics_manifest_path':'diag_b/manifest.json','diagnostics_manifest_sha256':file_hash(tmp_path/'diag_b/manifest.json')}
    with pytest.raises(ValueError):validate_diagnostics_binding(tmp_path,gate,primary/'manifest.json')


def test_five_seed_names_without_algebraic_evidence_cannot_pass(tmp_path):
    raw=tmp_path/'algebra';raw.mkdir();atomic_json(raw/'summary.json',{'baselines_and_remaps':[{'seed':s} for s in range(2022,2027)]})
    atomic_json(raw/'manifest.json',dict(status='COMPLETE',sources={},artifacts={'summary.json':file_hash(raw/'summary.json')}))
    with pytest.raises(ValueError):validate_algebraic(tmp_path,raw)

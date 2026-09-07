import copy
import json
from pathlib import Path

import pytest

from v2.phase6.audit import sha
from v2.phase6.composite import decision_template, validate_decision, manifest_closure, validate_composite


@pytest.mark.parametrize("field,value", [("original_fp32_tight_replay", "PASS"), ("upstream_pickle_preprocessing", "PASS"), ("stop_rule_triggered", True)])
def test_composite_cannot_erase_failures_or_upstream_unknown(field, value):
    decision = decision_template()
    decision[field] = value
    with pytest.raises(ValueError):
        validate_decision(decision)


def test_composite_scope_cannot_extend_to_untested_model_or_seed():
    decision = decision_template()
    decision["scope"]["models"].append("M6")
    with pytest.raises(ValueError, match="scope"):
        validate_decision(decision)


def test_recursive_closure_detects_nested_tampering(tmp_path):
    source = tmp_path / "raw.txt"; source.write_text("source")
    upstream = tmp_path / "upstream"; upstream.mkdir()
    (upstream / "a.txt").write_text("evidence")
    (upstream / "manifest.json").write_text(json.dumps({"files": {"a.txt": sha(upstream / "a.txt")}, "dependencies": {"raw.txt": sha(source)}}))
    receipt = tmp_path / "receipt"; receipt.mkdir()
    (receipt / "decision.json").write_text("{}")
    (receipt / "manifest.json").write_text(json.dumps({"files": {"decision.json": sha(receipt / "decision.json")}, "dependencies": {"upstream/manifest.json": sha(upstream / "manifest.json")}}))
    closure = manifest_closure(tmp_path, receipt / "manifest.json")
    assert set(closure) == {"raw.txt", "upstream/a.txt", "upstream/manifest.json", "receipt/decision.json", "receipt/manifest.json"}
    source.write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        manifest_closure(tmp_path, receipt / "manifest.json")


def test_current_cannot_point_straight_to_original_blocked_attempt(tmp_path):
    with pytest.raises(ValueError, match="current"):
        validate_composite(tmp_path, {"status": "COMPLETE", "attempt": "original", "manifest_sha256": "0" * 64})

"""Bounded scientific receipt preserving the failed original FP32 tolerance gate."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

from .audit import json_hash, sha, write_json

KIND = "phase6_bounded_scientific_composite"
BASE = "results/v2/phase6_preprocessing"
ORIGINAL = BASE + "/attempts/20260906T004123Z_a6acf299"
SUPPLEMENT = BASE + "/numerical_supplement_20260906"
REVIEW = BASE + "/review_evidence"
SCOPE = {"models": ["M0", "M3"], "seeds": [2022], "test_rows_each": 44000, "preprocessing_modes": ["legacy_identity", "train_global", "per_frame_rms"], "output": "classification_logits_only", "not_certified": ["auxiliary_regularizers", "other_checkpoints", "other_environments", "upstream_dataset_generation"]}


def decision_template():
    return {"schema_version": 1, "kind": KIND, "status": "COMPLETE", "completion_meaning": "bounded_composite_scientific_acceptance_only", "data_isolation": "PASS", "original_fp32_tight_replay": "FAILED_UNCHANGED", "supplemental_numerical_noninterference": "SUPPORTED_POST_HOC", "upstream_pickle_preprocessing": "UNKNOWN", "stop_rule_triggered": False, "scope": json.loads(json.dumps(SCOPE)), "review_basis": "parent acceptance of independent source and full-test numerical review; original FP32 criterion remains failed"}


def validate_decision(value):
    expected = decision_template()
    if not isinstance(value, dict) or value.get("scope") != SCOPE:
        raise ValueError("composite scope differs from reviewed evidence")
    if value != expected:
        raise ValueError("composite decision must preserve failed FP32, post-hoc scope and unknown upstream provenance")


def _path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("unsafe relative path in composite evidence")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("composite evidence escapes repository")
    return resolved


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid composite evidence JSON: {path}") from exc


def manifest_closure(root, manifest_path):
    """Verify all nested files/dependencies and return repository-relative hashes."""
    root = Path(root).resolve(); manifest_path = Path(manifest_path).resolve()
    checked, visited = {}, set()
    def check(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or escaped hash dependency: {path}")
        name = path.relative_to(root).as_posix()
        actual = checked.get(name)
        if actual is None:
            actual = sha(path); checked[name] = actual
        if expected is not None and (not isinstance(expected, str) or actual != expected):
            raise ValueError(f"hash mismatch: {name}")
        return actual
    def visit(path):
        check(path)
        if path in visited:
            return
        visited.add(path)
        manifest = _load(path)
        files = manifest.get("files")
        dependencies = manifest.get("dependencies", manifest.get("source_files"))
        if not isinstance(files, dict) or not files or not isinstance(dependencies, dict):
            raise ValueError("nested manifest has no hash closure")
        folder = path.parent
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p != path}
        if actual != set(files):
            raise ValueError("nested artifact hash closure polluted or incomplete")
        for name, expected in files.items():
            item = _path(folder, name)
            check(item, expected)
        for name, expected in dependencies.items():
            dependency = _path(root, name)
            check(dependency, expected)
            if dependency.name == "manifest.json":
                visit(dependency)
    try:
        visit(manifest_path)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid recursive hash closure: {exc}") from exc
    return dict(sorted(checked.items()))


def _validate_science(root, data, supplement):
    original = _load(data / "summary.json")
    original_manifest = _load(data / "manifest.json")
    config = _load(data / "config.json")
    if original_manifest.get("status") != "BLOCKED_STREAMING_REPLAY" or original.get("status") != "BLOCKED_STREAMING_REPLAY":
        raise ValueError("original failed FP32 state was changed")
    if config["modes"] != SCOPE["preprocessing_modes"] or config["streaming"]["models"] != SCOPE["models"] or config["streaming"]["seeds"] != [2022] or config["streaming"]["rtol"] != 1e-6 or config["streaming"]["atol"] != 1e-7:
        raise ValueError("original scope or tight tolerance changed")
    if original["partition_counts"] != {"train": 132000, "validation": 44000, "test": 44000} or set(original["transform_streaming"]) != set(SCOPE["preprocessing_modes"]):
        raise ValueError("data-isolation evidence scope is incomplete")
    with np.load(data / "split_binding.npz", allow_pickle=False) as b:
        all_ids = b["sample_ids"]; test_indices = b["test_indices"]; ids = all_ids[test_indices]
        train_indices, val_indices = b["train_indices"], b["validation_indices"]
    combined = np.concatenate([train_indices, val_indices, test_indices])
    if len(ids) != 44000 or len(np.unique(combined)) != 220000 or not np.array_equal(np.sort(combined), np.arange(220000)):
        raise ValueError("split IDs overlap or scope differs")
    for mode, item in original["transform_streaming"].items():
        if item.get("passed") is not True or item["n"] != 44000 or item["state_unchanged"] is not True or item["state_before_sha256"] != item["state_after_sha256"] or any(item[name] != 0 for name in ("batch_frame_max_abs", "reorder_max_abs", "future_max_abs")):
            raise ValueError("non-negotiable data-isolation gate failed")
        with np.load(data / mode / "transform_streaming.npz", allow_pickle=False) as b:
            if not np.array_equal(b["sample_ids"], ids) or any(not np.all(b[name] == 0) for name in ("batch_vs_frame_max_abs", "reordered_max_abs", "changed_future_max_abs")):
                raise ValueError("frame-isolation numerical evidence failed")
    supplemental = _load(supplement / "summary.json")
    supplemental_manifest = _load(supplement / "manifest.json")
    contract = _load(supplement / "contract.json")
    if supplemental.get("status") != "NUMERICAL_NONINTERFERENCE_SUPPORTED" or supplemental_manifest.get("status") != "NUMERICAL_NONINTERFERENCE_SUPPORTED" or supplemental.get("original_tight_float32_gate") != "FAILED_UNCHANGED":
        raise ValueError("post-hoc supplement unsupported or original failure erased")
    if contract["models"] != SCOPE["models"] or contract["seed"] != 2022 or contract["full_test_n_each_model"] != 44000 or contract["float64_compare_batches"] != [1, 64] or contract["float64_diagnostic_rtol"] != 1e-12 or contract["float64_diagnostic_atol"] != 1e-12 or contract["original_gate_preserved"] is not True or contract["diagnostic_only"] is not True:
        raise ValueError("post-hoc diagnostic scope changed")
    original_records = original["checkpoint_streaming"]
    supplemental_records = supplemental["results"]
    if len(original_records) != 2 or len(supplemental_records) != 2 or {r["model_id"] for r in original_records} != {"M0", "M3"} or {r["model_id"] for r in supplemental_records} != {"M0", "M3"}:
        raise ValueError("checkpoint matrix scope differs")
    for model in SCOPE["models"]:
        old = next(r for r in original_records if r["model_id"] == model)
        new = next(r for r in supplemental_records if r["model_id"] == model)
        if old["seed"] != 2022 or old["n"] != 44000 or old["passed"] is not False or old["tolerance_failed_rows"] <= 0 or old["argmax_changed_rows"] != 0 or old["buffers_before"] != old["buffers_after"]:
            raise ValueError("original failure/buffer evidence changed")
        if new["n"] != 44000 or any(new[field] is not True for field in ("float32_reversed_order_exact", "float32_buffers_unchanged", "float64_buffers_unchanged", "noninterference_precision_diagnosis_passed")):
            raise ValueError("supplement scope or state invariance failed")
        with np.load(data / "checkpoint_streaming" / f"{model}_seed2022.npz", allow_pickle=False) as b:
            if not np.array_equal(b["sample_ids"], ids):
                raise ValueError("original checkpoint row IDs changed")
            fp32, reference = b["actual_logits"], b["reference_logits"]
        with np.load(supplement / f"{model}_full_test_precision.npz", allow_pickle=False) as b:
            if not np.array_equal(b["sample_ids"], ids) or not np.array_equal(b["reversed_float32_batch1"], fp32):
                raise ValueError("full-test FP32 order noninterference failed")
            double1, double64 = b["float64_batch1"], b["float64_batch64"]
        if any(x.shape != (44000, 11) or not np.isfinite(x).all() for x in (fp32, reference, double1, double64)):
            raise ValueError("full-test finite logit scope violated")
        error = np.abs(fp32.astype(float) - reference.astype(float))
        failures = np.any(error > 1e-7 + 1e-6 * np.abs(reference), axis=1)
        if int(failures.sum()) != old["tolerance_failed_rows"] or float(error.max()) != old["max_abs"]:
            raise ValueError("original failed tight threshold was not preserved")
        if not np.allclose(double1, double64, rtol=1e-12, atol=1e-12):
            raise ValueError("full-test post-hoc double precision bound failed")
        decisions = reference.argmax(1)
        if any(not np.array_equal(x.argmax(1), decisions) for x in (fp32, double1, double64)):
            raise ValueError("classification semantics changed")


def validate_composite(root, current):
    """Return the original data attempt only after recursively validating receipt."""
    root = Path(root).resolve()
    value = _load(current) if isinstance(current, (str, Path)) else current
    fields = {"schema_version", "kind", "status", "attempt", "data_attempt", "manifest_sha256", "gate"}
    gate_fields = (fields - {"gate"}) | {"manifest_path", "stop_rule_triggered"}
    if isinstance(value, dict) and set(value) == gate_fields:
        value = {**{k: value[k] for k in fields - {"gate"}}, "gate": value}
    if not isinstance(value, dict) or set(value) != fields or value["schema_version"] != 1 or value["kind"] != KIND or value["status"] != "COMPLETE":
        raise ValueError("current is not a bounded Phase6 composite receipt")
    try:
        receipt = _path(root, value["attempt"])
        if not receipt.is_relative_to(root / BASE / "receipts"):
            raise ValueError("current must point to an independent receipt")
        manifest_path = receipt / "manifest.json"
        expected_gate = {**{k: value[k] for k in fields - {"gate"}}, "manifest_path": manifest_path.relative_to(root).as_posix(), "stop_rule_triggered": False}
        if value["gate"] != expected_gate or sha(manifest_path) != value["manifest_sha256"]:
            raise ValueError("current receipt gate/hash mismatch")
        manifest_closure(root, manifest_path)
        manifest = _load(manifest_path)
        if manifest.get("status") != "COMPLETE" or manifest.get("kind") != KIND:
            raise ValueError("receipt COMPLETE must mean bounded scientific acceptance")
        decision = _load(receipt / "decision.json"); validate_decision(decision)
        evidence = manifest["evidence"]
        required = {"original_manifest", "supplement_manifest", "review_evidence_manifest", "impact", "independent_review"}
        if set(evidence) != required:
            raise ValueError("receipt evidence set incomplete")
        for key, name in evidence.items():
            if name not in manifest["dependencies"]:
                raise ValueError("evidence is not hash-bound")
        data = _path(root, evidence["original_manifest"]).parent
        supplement = _path(root, evidence["supplement_manifest"]).parent
        if _path(root, value["data_attempt"]) != data:
            raise ValueError("current data_attempt differs from bound original evidence")
        _validate_science(root, data, supplement)
        if _load(_path(root, evidence["review_evidence_manifest"])).get("status") != "AUDIT_EVIDENCE":
            raise ValueError("independent source audit evidence missing")
        impact = _load(_path(root, evidence["impact"]))
        if impact.get("status") != "POST_HOC_NUMERICAL_IMPACT" or impact.get("changes_do_not_override_original_tolerance_failure") is not True:
            raise ValueError("numerical impact provenance changed")
        if not _path(root, evidence["independent_review"]).read_text(encoding="utf-8").strip():
            raise ValueError("independent review is empty")
        return data
    except (OSError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"invalid composite evidence: {exc}") from exc


def publish(root):
    root = Path(root).resolve()
    evidence = {"original_manifest": ORIGINAL + "/manifest.json", "supplement_manifest": SUPPLEMENT + "/manifest.json", "review_evidence_manifest": REVIEW + "/manifest.json", "impact": BASE + "/supplement_impact.json", "independent_review": "docs/phase6_numerical_gate_review.md"}
    for key in ("original_manifest", "supplement_manifest", "review_evidence_manifest"):
        manifest_closure(root, root / evidence[key])
    _validate_science(root, root / ORIGINAL, root / SUPPLEMENT)
    decision = decision_template()
    names = [*evidence.values(), "v2/phase6/composite.py", "scripts/v2/publish_phase6_composite.py"]
    dependencies = {name: sha(root / name) for name in names}
    fingerprint = json_hash({"decision": decision, "dependencies": dependencies})
    receipt = root / BASE / "receipts" / fingerprint[:16]
    if not receipt.exists():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        stage = receipt.parent / (".staging-" + uuid.uuid4().hex); stage.mkdir()
        write_json(stage / "decision.json", decision)
        (stage / "report.md").write_text("# Phase 6 bounded composite acceptance\n\nCOMPLETE refers only to this reviewed scientific receipt. Data isolation PASS; original tight FP32 replay FAILED_UNCHANGED; full-test precision/noninterference SUPPORTED_POST_HOC. No original result, tolerance, model, input, fit or protocol was changed.\n\nScope: three preprocessing modes, M0/M3 seed 2022, 44000 test frames each, classification logits only. Upstream pickle normalization provenance remains UNKNOWN. This is not a numerical exemption for other models, seeds, environments or auxiliary regularizers. No material data leakage triggering the Phase 1 restart rule was found in this audited scope.\n\nThe original blocked attempt, numerical supplement, independent source/figure QA evidence, calibration impact and independent review are recursively bound by the manifest. Downstream consumers must call validate_composite and use its returned original data attempt; the receipt contains no copied datasets.\n", encoding="utf-8")
        write_json(stage / "manifest.json", {"schema_version": 1, "kind": KIND, "status": "COMPLETE", "completion_meaning": "bounded_composite_scientific_acceptance_only", "fingerprint": fingerprint, "evidence": evidence, "dependencies": dependencies, "files": {p.name: sha(p) for p in stage.iterdir() if p.is_file()}})
        stage.replace(receipt)
    current = {"schema_version": 1, "kind": KIND, "status": "COMPLETE", "attempt": receipt.relative_to(root).as_posix(), "data_attempt": ORIGINAL, "manifest_sha256": sha(receipt / "manifest.json")}
    current["gate"] = {**current, "manifest_path": current["attempt"] + "/manifest.json", "stop_rule_triggered": False}
    validate_composite(root, current)
    temporary = root / BASE / (".current-" + uuid.uuid4().hex + ".json")
    write_json(temporary, current)
    os.replace(temporary, root / BASE / "current.json")
    return current

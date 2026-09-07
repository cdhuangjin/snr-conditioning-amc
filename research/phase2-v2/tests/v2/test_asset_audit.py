from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write(path: Path, payload: bytes = b"evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _synthetic_repository(tmp_path: Path) -> Path:
    root = tmp_path / "AWN"
    _write(root / "checkpoint" / "best.pt")
    _write(root / "training" / "train_model.py")
    _write(root / "experiments" / "multiseed" / "base_2022.json")
    _write(root / "experiments" / "v2" / "legacy_result.json")
    _write(root / "experiments" / "result" / "old_figure.png")
    _write(root / "experiments" / "run.log")
    _write(root / "data" / "RML2016.10a_dict.pkl")
    _write(root / "downloads" / "RML2016.04C" / "WBFM -8.txt")
    _write(root / "models" / "model.py")
    _write(root / "data_loader" / "data_loader.py")
    _write(root / "configs" / "v2" / "base.yaml")
    _write(root / "config" / "2016.10a.yml")
    _write(root / "splits" / "v2" / "fixed_split.npz")
    _write(root / "run_trial.py")
    _write(root / "plot_results.py")
    _write(tmp_path / "triple" / "results" / "A_asymmetry.json")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("path", "asset_type"),
    [
        ("experiments/multiseed/base_2022.json", "historical_result"),
        (
            r"experiments\generic\resnet\seed2022\models\resnet.pkl",
            "checkpoint",
        ),
        ("data/RML2016.10a_dict.pkl", "dataset"),
        (r"experiments\train_snrc.py", "experiment_script"),
        ("splits/v2/RML2016.10a_seed2022.npz", "split_file"),
        ("experiments/result/acc.svg", "figure"),
        ("data_loader/data_loader.py", "preprocessing_pipeline"),
        ("v2/preprocessing.py", "preprocessing_pipeline"),
        ("configs/v2/base.yaml", "config"),
        ("experiments/run.log", "log"),
        ("README.md", "unknown"),
    ],
)
def test_classify_path_normalizes_separators(path, asset_type):
    from scripts.v2.phase0_audit import classify_path

    assert classify_path(path) == asset_type


def test_normalization_preserves_parent_prefix_and_split_matching_is_protocol_specific():
    from scripts.v2.phase0_audit import _normalized, classify_path

    assert _normalized("../triple/results/value.json") == "../triple/results/value.json"
    assert _normalized("./experiments/value.json") == "experiments/value.json"
    assert classify_path("experiments/notsplitter.json") == "historical_result"
    assert classify_path("experiments/split_notes.txt") == "unknown"
    assert classify_path("splits/v2/fixed.json") == "split_file"


def test_scan_assets_is_deterministic_and_covers_declared_roots(tmp_path):
    from scripts.v2.phase0_audit import scan_assets

    root = _synthetic_repository(tmp_path)
    first = scan_assets(root, max_hash_bytes=1024)
    second = scan_assets(root, max_hash_bytes=1024)

    first_paths = [item["path"] for item in first]
    assert first == second
    assert first_paths == sorted(first_paths, key=lambda value: (value.casefold(), value))
    assert "../triple/results/A_asymmetry.json" in first_paths
    assert {
        "checkpoint/best.pt",
        "training/train_model.py",
        "experiments/multiseed/base_2022.json",
        "data/RML2016.10a_dict.pkl",
        "downloads/RML2016.04C/WBFM -8.txt",
        "models/model.py",
        "data_loader/data_loader.py",
        "configs/v2/base.yaml",
        "config/2016.10a.yml",
        "run_trial.py",
        "plot_results.py",
    } <= set(first_paths)
    assert all(
        set(item)
        == {
            "path",
            "asset_type",
            "size_bytes",
            "mtime",
            "sha256",
            "evidence_state",
            "reuse",
            "notes",
        }
        for item in first
    )


def test_deterministic_sort_key_breaks_casefold_ties():
    from scripts.v2.phase0_audit import _sort_key

    values = ["a", "A", "b"]
    assert sorted(values, key=_sort_key) == ["A", "a", "b"]


def test_oversized_assets_use_size_limit_marker(tmp_path):
    from scripts.v2.phase0_audit import scan_assets

    root = _synthetic_repository(tmp_path)
    large = root / "checkpoint" / "large.pt"
    large.write_bytes(b"x" * 32)

    record = next(
        item for item in scan_assets(root, max_hash_bytes=16) if item["path"] == "checkpoint/large.pt"
    )
    assert record["sha256"] == "not_computed:size_limit"


def test_scan_fails_closed_on_out_of_scope_symlinks(tmp_path):
    from scripts.v2.phase0_audit import AuditSecurityError, scan_repository

    root = _synthetic_repository(tmp_path)
    outside = _write(tmp_path / "outside" / "secret.json", b"outside")
    link = root / "experiments" / "outside_link"
    try:
        link.symlink_to(outside.parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this Windows host: {exc}")

    with pytest.raises(AuditSecurityError, match="reparse"):
        scan_repository(root, max_hash_bytes=1024)


def test_reports_symlink_is_rejected_without_touching_outside(tmp_path):
    from scripts.v2.phase0_audit import AuditSecurityError, run_audit

    root = _synthetic_repository(tmp_path)
    outside = tmp_path / "outside_reports"
    outside.mkdir()
    sentinel = _write(outside / "asset_inventory.md", b"outside sentinel")
    try:
        (root / "reports").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this Windows host: {exc}")

    with pytest.raises(AuditSecurityError, match="reports"):
        run_audit(root, max_hash_bytes=1024)
    assert sentinel.read_bytes() == b"outside sentinel"
    assert not (outside / "reviewer_issue_matrix.md").exists()


def test_report_destination_symlink_is_rejected_without_overwrite(tmp_path):
    from scripts.v2.phase0_audit import AuditSecurityError, run_audit

    root = _synthetic_repository(tmp_path)
    reports = root / "reports"
    reports.mkdir()
    outside = _write(tmp_path / "outside.md", b"outside sentinel")
    try:
        (reports / "asset_inventory.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this Windows host: {exc}")

    with pytest.raises(AuditSecurityError, match="asset_inventory"):
        run_audit(root, max_hash_bytes=1024)
    assert outside.read_bytes() == b"outside sentinel"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_reports_junction_is_rejected_without_touching_target(tmp_path):
    from scripts.v2.phase0_audit import AuditSecurityError, run_audit

    root = _synthetic_repository(tmp_path)
    outside = tmp_path / "junction_target"
    outside.mkdir()
    sentinel = _write(outside / "sentinel.txt", b"outside sentinel")
    reports = root / "reports"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(reports), str(outside)],
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout!r}")
    try:
        with pytest.raises(AuditSecurityError, match="reparse"):
            run_audit(root, max_hash_bytes=1024)
        assert sentinel.read_bytes() == b"outside sentinel"
        assert not (outside / "asset_inventory.md").exists()
    finally:
        os.rmdir(reports)


def test_growth_past_hash_limit_is_stopped_and_reported(tmp_path):
    from scripts.v2.phase0_audit import scan_repository

    root = _synthetic_repository(tmp_path)
    target = _write(root / "checkpoint" / "growing.pt", b"12345678")
    changed = False

    def grow(event, path, _total):
        nonlocal changed
        if event == "after_chunk" and path == target and not changed:
            with path.open("ab") as handle:
                handle.write(b"abcdefgh")
            changed = True

    result = scan_repository(root, max_hash_bytes=10, event_hook=grow, chunk_size=8)
    record = next(item for item in result.assets if item["path"] == "checkpoint/growing.pt")
    assert record["sha256"] == "not_computed:size_limit"
    assert any(issue["error_kind"] == "unstable_during_scan" for issue in result.issues)
    assert not result.complete


def test_same_size_mutation_is_not_hashed_as_stable(tmp_path):
    from scripts.v2.phase0_audit import scan_repository

    root = _synthetic_repository(tmp_path)
    target = _write(root / "checkpoint" / "mutating.pt", b"12345678")
    changed = False

    def mutate(event, path, _total):
        nonlocal changed
        if event == "after_chunk" and path == target and not changed:
            path.write_bytes(b"ABCDEFGH")
            changed = True

    result = scan_repository(root, max_hash_bytes=32, event_hook=mutate, chunk_size=8)
    record = next(item for item in result.assets if item["path"] == "checkpoint/mutating.pt")
    assert record["sha256"] == "not_computed:unstable_during_scan"
    assert any(issue["error_kind"] == "unstable_during_scan" for issue in result.issues)


def test_vanished_file_is_retained_as_structured_issue(tmp_path):
    from scripts.v2.phase0_audit import scan_repository

    root = _synthetic_repository(tmp_path)
    target = _write(root / "checkpoint" / "vanishing.pt", b"12345678")

    def vanish(event, path, _total):
        if event == "before_post_lstat" and path == target and path.exists():
            path.unlink()

    result = scan_repository(root, max_hash_bytes=32, event_hook=vanish)
    record = next(item for item in result.assets if item["path"] == "checkpoint/vanishing.pt")
    assert record["sha256"] == "not_computed:vanished"
    assert any(issue["error_kind"] == "vanished" for issue in result.issues)


def test_unreadable_and_hash_errors_are_structured_and_incomplete(tmp_path):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)
    unreadable = _write(root / "checkpoint" / "unreadable.pt", b"secret")
    broken = _write(root / "checkpoint" / "hash_error.pt", b"secret")

    def open_file(path):
        if path == unreadable:
            raise PermissionError("injected unreadable")
        return audit._open_binary_no_follow(path)

    def read_chunk(handle, size):
        if Path(handle.name) == broken:
            raise OSError("injected hash error")
        return handle.read(size)

    result = audit.scan_repository(
        root,
        max_hash_bytes=32,
        open_file=open_file,
        read_chunk=read_chunk,
    )
    by_path = {item["path"]: item for item in result.assets}
    assert by_path["checkpoint/unreadable.pt"]["sha256"] == "not_computed:unreadable"
    assert by_path["checkpoint/hash_error.pt"]["sha256"] == "not_computed:hash_error"
    assert {issue["error_kind"] for issue in result.issues} >= {"unreadable", "hash_error"}
    assert not result.complete


def test_broken_reparse_point_fails_closed_when_supported(tmp_path):
    from scripts.v2.phase0_audit import AuditSecurityError, scan_repository

    root = _synthetic_repository(tmp_path)
    target = _write(tmp_path / "temporary_target.json", b"secret")
    link = root / "experiments" / "broken.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this Windows host: {exc}")
    target.unlink()

    with pytest.raises(AuditSecurityError, match="reparse"):
        scan_repository(root, max_hash_bytes=32)


def test_nested_directory_disappearance_is_reported_without_timing_race(
    tmp_path, monkeypatch
):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)
    vanishing = root / "experiments" / "vanishing_directory"
    _write(vanishing / "result.json", b"{}")
    original_lstat = audit._lstat
    injected = False

    def disappearing_lstat(path):
        nonlocal injected
        if Path(path) == vanishing and not injected:
            injected = True
            raise FileNotFoundError("injected directory disappearance")
        return original_lstat(path)

    monkeypatch.setattr(audit, "_lstat", disappearing_lstat)
    result = audit.run_audit(root, max_hash_bytes=1024)

    assert result["complete"] is False
    assert any(issue["error_kind"] == "walk_vanished" for issue in result["issues"])
    placeholder = next(
        item
        for item in result["assets"]
        if item["path"] == "experiments/vanishing_directory/"
    )
    assert placeholder["sha256"] == "not_computed:walk_vanished"
    inventory = (root / "reports" / "asset_inventory.md").read_text(encoding="utf-8")
    assert "walk_vanished" in inventory
    assert "Completeness: `incomplete`" in inventory


def test_fdopen_failure_closes_transferred_descriptor_exactly_once(tmp_path):
    from scripts.v2.phase0_audit import _fdopen_owned_descriptor

    closed = []

    def failing_fdopen(*_args, **_kwargs):
        raise RuntimeError("injected fdopen failure")

    with pytest.raises(RuntimeError, match="injected fdopen failure"):
        _fdopen_owned_descriptor(
            12345,
            tmp_path / "asset.bin",
            fdopen=failing_fdopen,
            close_descriptor=closed.append,
        )
    assert closed == [12345]


def test_run_audit_writes_required_reports_and_preserves_sources(tmp_path):
    from scripts.v2.phase0_audit import run_audit

    root = _synthetic_repository(tmp_path)
    sources = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {path: (_sha256(path), path.stat().st_mtime_ns) for path in sources}

    result = run_audit(root, max_hash_bytes=1024)

    assert result["asset_count"] == len(result["assets"])
    assert result["complete"] is True
    assert result["issue_count"] == 0
    assert result["audit_id"]
    assert result["inventory_digest"]
    assert result["root_identity"]
    assert set(result["reports"]) == {
        "asset_inventory",
        "reviewer_issue_matrix",
        "v2_reuse_map",
        "v2_progress",
    }
    for path in result["reports"].values():
        assert Path(path).is_file()
    assert before == {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in sources
    }

    inventory = (root / "reports" / "asset_inventory.md").read_text(encoding="utf-8")
    assert "# V2 Phase 0 Asset Inventory" in inventory
    assert "## Evidence summary" in inventory
    assert "Phase 0 performed no training" in inventory
    assert "## historical_result" in inventory
    assert (
        "| path | asset type | evidence state | bytes | mtime | sha256 | reuse decision | risk/notes |"
        in inventory
    )

    matrix = (root / "reports" / "reviewer_issue_matrix.md").read_text(encoding="utf-8")
    assert "| issue | existing evidence | missing experiment | required code change | phase | gate |" in matrix
    required_topics = [
        "SNR-agnostic classifiers inevitably collapse",
        "loss reweighting is always ineffective",
        "external SNR is universally necessary",
        "embedding implements arbitrary per-SNR affine boundaries",
        "discrete embedding is inherently continuous",
        "-8 dB is a universal phase transition",
        "fixed persisted splits and five seeds",
        "empirical feature geometry and diagnostics",
        "symmetric non-collapse counterexample",
        "preprocessing leakage and SNR inferability",
        "WBFM dominance analysis",
        "estimator side-channel controls E1-E4",
        "deployment-matched estimator noise",
        "Reliability-floor conditioning",
        "RML2016.04C cross-dataset protocol",
        "RML2018.01a cross-dataset protocol",
        "independent synthetic benchmark",
        "persisted JSON/NPZ-only figures and tables",
        "negative result retention",
        "paired statistics",
        "manifest, progress, and final deliverables",
    ]
    assert all(topic.casefold() in matrix.casefold() for topic in required_topics)
    assert "not found in audited assets" in matrix

    reuse = (root / "reports" / "v2_reuse_map.md").read_text(encoding="utf-8")
    assert "reuse_after_protocol_hash_check" in reuse
    assert "reference_only" in reuse
    assert "do_not_reuse" in reuse
    assert "experiments/v2 is legacy" in reuse
    assert "results/v2" in reuse
    assert "Preserve all historical result candidates" in reuse

    common_fields = [
        f"Audit ID: `{result['audit_id']}`",
        f"Inventory digest: `{result['inventory_digest']}`",
        f"Root identity: `{result['root_identity']}`",
        "Scanner version:",
        "Completeness: `complete`",
        "Scan issues: `0`",
    ]
    for report_path in result["reports"].values():
        text = Path(report_path).read_text(encoding="utf-8")
        assert all(field in text for field in common_fields)


def test_historical_result_validation_does_not_trust_filename(tmp_path):
    from scripts.v2.phase0_audit import scan_assets

    root = _synthetic_repository(tmp_path)
    expected_hash = "a" * 64
    _write(
        root / "experiments" / "validated.json",
        json.dumps(
            {
                "status": "completed",
                "metrics": {"accuracy": 0.5},
                "protocol_hash": expected_hash.upper(),
            }
        ).encode(),
    )
    _write(
        root / "experiments" / "needs_protocol.json",
        json.dumps({"status": "completed", "metrics": {"accuracy": 0.5}}).encode(),
    )
    _write(root / "experiments" / "success_result.json", b"{}")
    _write(root / "experiments" / "malformed.json", b"{not-json")
    _write(
        root / "experiments" / "failed.json",
        json.dumps({"status": "failed", "metrics": {"accuracy": 0.9}}).encode(),
    )

    records = {item["path"]: item for item in scan_assets(root, max_hash_bytes=1024)}
    assert (
        records["experiments/validated.json"]["evidence_state"]
        == "validated historical result candidate"
    )
    assert records["experiments/validated.json"]["reuse"] == "reuse_after_protocol_hash_check"
    assert records["experiments/needs_protocol.json"]["evidence_state"] == (
        "unverified historical result"
    )
    assert records["experiments/needs_protocol.json"]["reuse"] == "reuse_after_protocol_hash_check"
    for name in ["success_result.json", "malformed.json", "failed.json"]:
        assert records[f"experiments/{name}"]["evidence_state"] == "unverified historical result"
        assert records[f"experiments/{name}"]["reuse"] != "reuse_as_is"

    trusted = {
        item["path"]: item
        for item in scan_assets(
            root,
            max_hash_bytes=1024,
            expected_protocol_hash=expected_hash,
        )
    }
    assert (
        trusted["experiments/validated.json"]["evidence_state"]
        == "validated machine-readable result"
    )
    assert trusted["experiments/validated.json"]["reuse"] == "reuse_as_is"


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("null.json", {"status": "completed", "metrics": {"accuracy": None}}),
        ("bool.json", {"status": "completed", "metrics": {"accuracy": True}}),
        ("string.json", {"status": "completed", "metrics": {"accuracy": "0.9"}}),
        ("nan.json", {"status": "completed", "metrics": {"accuracy": float("nan")}}),
        ("inf.json", {"status": "completed", "metrics": {"accuracy": float("inf")}}),
        ("wrong_status.json", {"status": "success", "metrics": {"accuracy": 0.9}}),
    ],
)
def test_historical_validation_rejects_nonfinite_nonnumeric_or_wrong_status(
    tmp_path, name, payload
):
    from scripts.v2.phase0_audit import scan_assets

    root = _synthetic_repository(tmp_path)
    payload["protocol_hash"] = "a" * 64
    _write(root / "experiments" / name, json.dumps(payload).encode())

    record = next(
        item
        for item in scan_assets(
            root,
            max_hash_bytes=1024,
            expected_protocol_hash="a" * 64,
        )
        if item["path"] == f"experiments/{name}"
    )
    assert record["evidence_state"] == "unverified historical result"
    assert record["reuse"] != "reuse_as_is"


@pytest.mark.parametrize(
    ("protocol_hash", "expected_hash", "expected_state"),
    [
        ("abc123", "a" * 64, "unverified historical result"),
        ("b" * 64, "a" * 64, "validated historical result candidate"),
        ("g" * 64, "a" * 64, "unverified historical result"),
    ],
)
def test_valid_result_structure_without_matching_sha256_remains_candidate(
    tmp_path, protocol_hash, expected_hash, expected_state
):
    from scripts.v2.phase0_audit import scan_assets

    root = _synthetic_repository(tmp_path)
    _write(
        root / "experiments" / "candidate.json",
        json.dumps(
            {
                "status": "completed",
                "metrics": {"balanced_accuracy": 0.75},
                "protocol_hash": protocol_hash,
            }
        ).encode(),
    )

    record = next(
        item
        for item in scan_assets(
            root,
            max_hash_bytes=1024,
            expected_protocol_hash=expected_hash,
        )
        if item["path"] == "experiments/candidate.json"
    )
    assert record["evidence_state"] == expected_state
    assert record["reuse"] == "reuse_after_protocol_hash_check"
    assert "historical-result/v1" in record["notes"]


def test_reviewer_evidence_labels_candidate_and_validation_strength(tmp_path):
    from scripts.v2.phase0_audit import _reviewer_report, scan_assets

    root = _synthetic_repository(tmp_path)
    _write(
        root / "experiments" / "base_validated.json",
        json.dumps(
            {
                "status": "completed",
                "metrics": {"accuracy": 0.5},
                "protocol_hash": "a" * 64,
            }
        ).encode(),
    )
    report = _reviewer_report(scan_assets(root, max_hash_bytes=1024), metadata="")
    assert "validated historical result candidate" in report
    assert "unverified historical result" in report
    assert "candidate/reference" in report


def test_incomplete_scan_is_published_with_issue_details(tmp_path):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)
    unreadable = root / "checkpoint" / "best.pt"

    def open_file(path):
        if path == unreadable:
            raise PermissionError("injected unreadable")
        return audit._open_binary_no_follow(path)

    result = audit.run_audit(root, max_hash_bytes=1024, open_file=open_file)
    assert result["complete"] is False
    assert result["issue_count"] == 1
    for report_path in result["reports"].values():
        report = Path(report_path).read_text(encoding="utf-8")
        assert "Completeness: `incomplete`" in report
        assert "Scan issues: `1`" in report
    inventory = (root / "reports" / "asset_inventory.md").read_text(encoding="utf-8")
    assert "unreadable" in inventory
    assert "injected unreadable" in inventory
    progress = (root / "reports" / "v2_progress.md").read_text(encoding="utf-8")
    assert "Diagnostic reports are always generated" in progress
    assert "--allow-incomplete permits exit code 0 despite incompleteness" in progress


def test_cli_requires_allow_incomplete_for_zero_exit(monkeypatch, tmp_path, capsys):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)

    def unreadable(_path):
        raise PermissionError("injected CLI unreadable")

    monkeypatch.setattr(audit, "_open_binary_no_follow", unreadable)
    assert audit.main(["--root", str(root)]) == 2
    inventory = root / "reports" / "asset_inventory.md"
    assert inventory.is_file()
    assert "Completeness: `incomplete`" in inventory.read_text(encoding="utf-8")

    inventory.unlink()
    assert audit.main(["--root", str(root), "--allow-incomplete"]) == 0
    assert inventory.is_file()
    assert "Completeness: `incomplete`" in inventory.read_text(encoding="utf-8")
    assert "incomplete" in capsys.readouterr().out.casefold()


def test_allow_incomplete_help_describes_exit_status_not_report_generation(capsys):
    from scripts.v2.phase0_audit import main

    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out
    assert "Diagnostic reports are always generated" in help_text
    assert "permits exit code 0 despite incompleteness" in help_text


@pytest.mark.parametrize("failure_boundary", [1, 2, 3, 4])
def test_report_set_replace_failure_rolls_back_all_reports(tmp_path, failure_boundary):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)
    reports = root / "reports"
    reports.mkdir()
    destinations = [
        reports / "asset_inventory.md",
        reports / "reviewer_issue_matrix.md",
        reports / "v2_reuse_map.md",
        reports / "v2_progress.md",
    ]
    for index, path in enumerate(destinations):
        path.write_text(f"old-{index}\n", encoding="utf-8")
    calls = 0

    def failing_replace(source, destination):
        nonlocal calls
        if ".stage." in source.name:
            calls += 1
            if calls == failure_boundary:
                raise OSError(f"injected replace {failure_boundary}")
        return source.replace(destination)

    with pytest.raises(OSError, match=f"injected replace {failure_boundary}"):
        audit.run_audit(root, max_hash_bytes=1024, replace_file=failing_replace)

    assert [path.read_text(encoding="utf-8") for path in destinations] == [
        "old-0\n",
        "old-1\n",
        "old-2\n",
        "old-3\n",
    ]
    assert list(reports.glob(".*.stage.*")) == []
    assert list(reports.glob(".*.backup.*")) == []


def test_primary_exception_survives_progress_render_failure(tmp_path, monkeypatch):
    import scripts.v2.phase0_audit as audit

    root = _synthetic_repository(tmp_path)

    def primary_failure(*args, **kwargs):
        raise ValueError("primary scan failure")

    def progress_failure(*args, **kwargs):
        raise RuntimeError("secondary progress failure")

    monkeypatch.setattr(audit, "scan_repository", primary_failure)
    monkeypatch.setattr(audit, "update_phase_progress", progress_failure)
    with pytest.raises(ValueError, match="primary scan failure") as caught:
        audit.run_audit(root, max_hash_bytes=1024)
    assert any("secondary progress failure" in note for note in getattr(caught.value, "__notes__", []))


def test_markdown_escaping_handles_pipes_newlines_and_backticks():
    from scripts.v2.phase0_audit import _code_path, _markdown

    assert _markdown("a|b\n`c`") == "a\\|b<br>&#96;c&#96;"
    assert _code_path("a|b\n`c`") == "<code>a\\|b&lt;br&gt;&#96;c&#96;</code>"


def test_lock_sidecars_have_exact_root_ignore_rules():
    root = Path(__file__).resolve().parents[2]
    rules = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.manifest.json.lock" in rules
    assert "/reports/.v2_progress.md.lock" in rules


def test_progress_rerun_replaces_single_phase_zero_section(tmp_path):
    from scripts.v2.phase0_audit import run_audit

    root = _synthetic_repository(tmp_path)
    run_audit(root, max_hash_bytes=1024)
    run_audit(root, max_hash_bytes=1024)

    report = (root / "reports" / "v2_progress.md").read_text(encoding="utf-8")
    assert report.count("<!-- V2_PHASE:Phase 0:START -->") == 1
    assert report.count("<!-- V2_PHASE:Phase 0:END -->") == 1
    for heading in ["Completed", "Failed", "Unexpected", "Interpretation", "Next gate"]:
        assert f"**{heading}**" in report
    assert "No training or inference was performed" in report


def test_run_audit_does_not_import_torch(tmp_path, monkeypatch):
    import sys

    from scripts.v2.phase0_audit import run_audit

    root = _synthetic_repository(tmp_path)
    sys.modules.pop("torch", None)
    run_audit(root, max_hash_bytes=1024)
    assert "torch" not in sys.modules

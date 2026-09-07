from __future__ import annotations

import json
import multiprocessing
import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest


def _record(**overrides):
    from v2.contracts import ArtifactRecord

    values = {
        "experiment": "repro_awn_plain",
        "dataset": "RML2016.10a",
        "split_hash": "abc",
        "model": "AWN",
        "conditioner": "none",
        "seed": 2022,
        "checkpoint": "results/v2/reproduction/awn_plain_2022/checkpoint.pt",
        "result_json": "results/v2/reproduction/awn_plain_2022/result.json",
        "figure_paths": [],
        "status": "completed",
        "notes": "negative results retained",
    }
    values.update(overrides)
    return ArtifactRecord(**values)


def _manifest_writer(path, seed, start, ready, outcomes):
    try:
        from v2.manifest import ManifestStore

        store = ManifestStore(path)
        original_load = store.load

        def slow_load():
            manifest = original_load()
            time.sleep(0.5)
            return manifest

        store.load = slow_load
        ready.put(seed)
        if not start.wait(5):
            raise TimeoutError("manifest writer start timed out")
        store.upsert(_record(seed=seed))
        outcomes.put((seed, None))
    except BaseException:
        outcomes.put((seed, traceback.format_exc()))


def _progress_writer(path, phase, start, ready, outcomes):
    try:
        from v2.progress import update_phase_progress

        original_read_text = Path.read_text

        def slow_read_text(self, *args, **kwargs):
            report = original_read_text(self, *args, **kwargs)
            time.sleep(0.5)
            return report

        Path.read_text = slow_read_text
        ready.put(phase)
        if not start.wait(5):
            raise TimeoutError("progress writer start timed out")
        update_phase_progress(
            phase,
            completed=f"{phase} complete",
            failed=f"{phase} negative result",
            unexpected="none",
            interpretation="retain evidence",
            next_gate="review",
            path=path,
        )
        outcomes.put((phase, None))
    except BaseException:
        outcomes.put((phase, traceback.format_exc()))


def _run_spawned_writers(target, worker, identities):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = context.Queue()
    outcomes = context.Queue()
    processes = [
        context.Process(target=worker, args=(target, identity, start, ready, outcomes))
        for identity in identities
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            ready.get(timeout=10)
        start.set()
        for process in processes:
            process.join(timeout=15)
        assert all(not process.is_alive() for process in processes)
        results = [outcomes.get(timeout=5) for _ in processes]
        assert [error for _, error in results if error is not None] == []
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        ready.close()
        outcomes.close()
        ready.join_thread()
        outcomes.join_thread()


def test_artifact_record_has_exact_manifest_fields_and_serializes():
    from v2.contracts import ArtifactRecord

    rec = _record()

    assert [field.name for field in fields(ArtifactRecord)] == [
        "experiment",
        "dataset",
        "split_hash",
        "model",
        "conditioner",
        "seed",
        "checkpoint",
        "result_json",
        "figure_paths",
        "status",
        "notes",
    ]
    assert json.loads(json.dumps(rec.to_dict()))["seed"] == 2022


def test_artifact_status_supports_all_gate_states(tmp_path):
    from v2.contracts import ArtifactStatus
    from v2.manifest import ManifestStore

    statuses = {
        "pending",
        "running",
        "completed",
        "failed",
        "gate_failed",
        "skipped",
    }
    assert set(get_args(ArtifactStatus)) == statuses

    store = ManifestStore(tmp_path / "manifest.json")
    for seed, status in enumerate(sorted(statuses), start=2022):
        store.upsert(_record(seed=seed, status=status))
    assert {item["status"] for item in store.load()["experiments"]} == statuses


def test_missing_manifest_loads_as_empty_schema(tmp_path):
    from v2.manifest import ManifestStore

    store = ManifestStore(tmp_path / "manifest.json")

    assert store.load() == {"schema_version": 1, "experiments": []}


def test_manifest_upsert_is_idempotent(tmp_path):
    from v2.contracts import ArtifactRecord
    from v2.manifest import ManifestStore

    store = ManifestStore(tmp_path / "manifest.json")
    rec = ArtifactRecord(
        experiment="repro_awn_plain",
        dataset="RML2016.10a",
        split_hash="abc",
        model="AWN",
        conditioner="none",
        seed=2022,
        checkpoint="results/v2/reproduction/awn_plain_2022/checkpoint.pt",
        result_json="results/v2/reproduction/awn_plain_2022/result.json",
        figure_paths=[],
        status="completed",
        notes="negative results retained",
    )
    store.upsert(rec)
    store.upsert(rec)

    assert len(store.load()["experiments"]) == 1


def test_manifest_upsert_updates_status_and_notes_for_same_key(tmp_path):
    from v2.manifest import ManifestStore

    store = ManifestStore(tmp_path / "manifest.json")
    store.upsert(_record(status="running", notes="initial"))
    store.upsert(_record(status="failed", notes="accuracy regressed at low SNR"))

    experiments = store.load()["experiments"]
    assert len(experiments) == 1
    assert experiments[0]["status"] == "failed"
    assert experiments[0]["notes"] == "accuracy regressed at low SNR"


def test_manifest_upsert_appends_a_second_unique_key(tmp_path):
    from v2.manifest import ManifestStore

    store = ManifestStore(tmp_path / "manifest.json")
    store.upsert(_record(seed=2022))
    store.upsert(_record(seed=2023))

    assert [item["seed"] for item in store.load()["experiments"]] == [2022, 2023]


def test_manifest_write_uses_same_directory_path_replace(tmp_path, monkeypatch):
    from v2.manifest import ManifestStore

    target = tmp_path / "nested" / "manifest.json"
    observed = []
    original_replace = Path.replace

    def recording_replace(source, destination):
        observed.append((Path(source), Path(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", recording_replace)

    ManifestStore(target).upsert(_record())

    assert len(observed) == 1
    temporary, destination = observed[0]
    assert temporary.parent == target.parent
    assert destination == target


def test_manifest_concurrent_process_upserts_retain_distinct_records(tmp_path):
    target = tmp_path / "manifest.json"
    seeds = [2022, 2023, 2024, 2025]

    _run_spawned_writers(target, _manifest_writer, seeds)

    from v2.manifest import ManifestStore

    assert {item["seed"] for item in ManifestStore(target).load()["experiments"]} == set(
        seeds
    )


def test_manifest_malformed_json_raises_path_aware_validation_error(tmp_path):
    from v2.manifest import ManifestStore, ManifestValidationError

    target = tmp_path / "manifest.json"
    target.write_text('{"schema_version": 1,', encoding="utf-8")

    with pytest.raises(ManifestValidationError) as caught:
        ManifestStore(target).load()

    assert str(target) in str(caught.value)
    assert "malformed JSON" in str(caught.value)


@pytest.mark.parametrize(
    ("manifest", "error_fragment"),
    [
        ([], "$"),
        ({"schema_version": 2, "experiments": []}, "schema_version"),
        ({"schema_version": 1.0, "experiments": []}, "schema_version"),
        ({"schema_version": 1, "experiments": {}}, "experiments"),
        ({"schema_version": 1, "experiments": ["not an object"]}, "experiments[0]"),
        (
            {
                "schema_version": 1,
                "experiments": [_record().to_dict() | {"seed": "2022"}],
            },
            "experiments[0].seed",
        ),
        (
            {"schema_version": 1, "experiments": [_record(model=None).to_dict()]},
            "experiments[0].model",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [
                    {
                        key: value
                        for key, value in _record().to_dict().items()
                        if key != "notes"
                    }
                ],
            },
            "experiments[0].notes",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [_record(figure_paths=["figure.pdf", 7]).to_dict()],
            },
            "experiments[0].figure_paths[1]",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [_record(status="unknown").to_dict()],
            },
            "experiments[0].status",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [_record(status=["failed"]).to_dict()],
            },
            "experiments[0].status",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [_record().to_dict() | {"unexpected": "field"}],
            },
            "experiments[0].unexpected",
        ),
        (
            {
                "schema_version": 1,
                "experiments": [_record().to_dict(), _record(notes="duplicate").to_dict()],
            },
            "duplicates experiments[0]",
        ),
    ],
)
def test_manifest_schema_validation_is_path_and_index_aware(
    tmp_path, manifest, error_fragment
):
    from v2.manifest import ManifestStore, ManifestValidationError

    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestValidationError) as caught:
        ManifestStore(target).load()

    assert str(target) in str(caught.value)
    assert error_fragment in str(caught.value)


def test_upsert_rejects_unsupported_runtime_status_without_changing_manifest(tmp_path):
    from v2.manifest import ManifestStore, ManifestValidationError

    target = tmp_path / "manifest.json"
    store = ManifestStore(target)
    store.upsert(_record())
    original = target.read_bytes()

    with pytest.raises(ManifestValidationError, match=r"experiments\[1\]\.status"):
        store.upsert(_record(seed=2023, status="unknown"))

    assert target.read_bytes() == original


@pytest.mark.parametrize("failure_point", ["serialize", "replace"])
def test_manifest_atomic_failure_preserves_original_and_cleans_temp(
    tmp_path, monkeypatch, failure_point
):
    import v2.manifest as manifest_module
    from v2.manifest import ManifestStore

    target = tmp_path / "manifest.json"
    target.write_text('{"schema_version": 1, "experiments": []}\n', encoding="utf-8")
    original = target.read_bytes()

    def injected_failure(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "serialize":
        monkeypatch.setattr(manifest_module.json, "dump", injected_failure)
    else:
        monkeypatch.setattr(Path, "replace", injected_failure)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        ManifestStore(target).upsert(_record())

    assert target.read_bytes() == original
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_sha256_file_hashes_file_content(tmp_path):
    from v2.provenance import sha256_file

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")

    assert sha256_file(payload) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("chunk_size", [True, False, 1.0, "1", None, 0, -1])
def test_file_sha256_rejects_invalid_chunk_size(tmp_path, chunk_size):
    from v2.provenance import file_sha256

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        file_sha256(payload, chunk_size=chunk_size)


def test_deterministic_json_hash_ignores_mapping_insertion_order():
    from v2.provenance import deterministic_json_hash

    left = {"dataset": "RML2016.10a", "seeds": [2022, 2023]}
    right = {"seeds": [2022, 2023], "dataset": "RML2016.10a"}

    assert deterministic_json_hash(left) == deterministic_json_hash(right)
    assert deterministic_json_hash(left) != deterministic_json_hash(
        {"dataset": "RML2016.10a", "seeds": [2022, 2024]}
    )


def test_environment_fingerprint_contains_runtime_metadata():
    from v2.provenance import environment_fingerprint

    fingerprint = environment_fingerprint()

    assert fingerprint["python"]
    assert fingerprint["platform"]
    assert isinstance(fingerprint["packages"], dict)


def test_update_phase_progress_replaces_phase_and_preserves_other_phases(tmp_path):
    from v2.progress import update_phase_progress

    path = tmp_path / "reports" / "v2_progress.md"
    update_phase_progress(
        "phase-0",
        completed="split frozen",
        failed="baseline missed tolerance",
        unexpected="low-SNR regression",
        interpretation="negative result retained",
        next_gate="audit split",
        path=path,
    )
    update_phase_progress(
        "phase-1",
        completed="runner added",
        failed="none",
        unexpected="none",
        interpretation="ready",
        next_gate="launch",
        path=path,
    )
    update_phase_progress(
        "phase-0",
        completed="split rechecked",
        failed="baseline still missed tolerance\nraw negative result retained verbatim",
        unexpected="low-SNR regression",
        interpretation="do not suppress failure",
        next_gate="stop gate",
        path=path,
    )

    report = path.read_text(encoding="utf-8")
    assert report.count("<!-- V2_PHASE:phase-0:START -->") == 1
    assert report.count("<!-- V2_PHASE:phase-1:START -->") == 1
    assert "split frozen" not in report
    assert "runner added" in report
    assert "baseline still missed tolerance\nraw negative result retained verbatim" in report


def test_progress_concurrent_process_updates_retain_distinct_phases(tmp_path):
    target = tmp_path / "v2_progress.md"
    target.write_text("# V2 Progress\n", encoding="utf-8")
    phases = ["phase-0", "phase-1", "phase-2", "phase-3"]

    _run_spawned_writers(target, _progress_writer, phases)

    report = target.read_text(encoding="utf-8")
    for phase in phases:
        assert report.count(f"<!-- V2_PHASE:{phase}:START -->") == 1
        assert f"{phase} negative result" in report


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_progress_atomic_failure_preserves_original_and_cleans_temp(
    tmp_path, monkeypatch, failure_point
):
    from v2.progress import update_phase_progress

    target = tmp_path / "v2_progress.md"
    target.write_text("# Existing report\n\nnegative result must remain\n", encoding="utf-8")
    original = target.read_bytes()

    def injected_failure(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "write":
        monkeypatch.setattr(Path, "write_text", injected_failure)
    else:
        monkeypatch.setattr(Path, "replace", injected_failure)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        update_phase_progress(
            "phase-0",
            completed="none",
            failed="new failure",
            unexpected="none",
            interpretation="stop",
            next_gate="blocked",
            path=target,
        )

    assert target.read_bytes() == original
    assert list(tmp_path.glob(".v2_progress.md.*.tmp")) == []

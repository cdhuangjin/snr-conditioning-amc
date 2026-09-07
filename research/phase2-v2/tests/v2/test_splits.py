from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml


def _cells(count: int = 10):
    return [("BPSK", -2, count), ("QPSK", 0, count)]


def _save(tmp_path: Path, *, cells=None, seed=2022):
    from v2.splits import make_stratified_split, save_split

    data = tmp_path / "data" / "tiny.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"immutable tiny dataset")
    split = make_stratified_split(cells or _cells(), seed=seed, dataset_id="Tiny")
    npz = tmp_path / "splits" / "v2" / "tiny.npz"
    meta = npz.with_suffix(".json")
    metadata = save_split(split, npz, meta, data_path=data, repository_root=tmp_path)
    return split, tmp_path / metadata["npz_path"], meta, data


def _load_script():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "v2" / "phase1_reproduce.py"
    spec = importlib.util.spec_from_file_location("phase1_reproduce", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_same_seed_is_stable_and_different_seed_changes_membership():
    from v2.splits import make_stratified_split

    first = make_stratified_split(_cells(100), seed=2022, dataset_id="Tiny")
    again = make_stratified_split(_cells(100), seed=2022, dataset_id="Tiny")
    other = make_stratified_split(_cells(100), seed=2023, dataset_id="Tiny")

    assert np.array_equal(first.train_idx, again.train_idx)
    assert np.array_equal(first.val_idx, again.val_idx)
    assert not np.array_equal(first.train_idx, other.train_idx)


def test_split_matches_historical_randomstate_and_set_remainder_protocol():
    from v2.splits import make_stratified_split

    split = make_stratified_split(_cells(100), seed=2022, dataset_id="Tiny")
    np.random.seed(2022)
    historical_train, historical_val, historical_test = [], [], []
    for start in (0, 100):
        population = range(start, start + 100)
        train = np.random.choice(population, size=60, replace=False)
        remainder = list(set(population) - set(train))
        test = np.random.choice(remainder, size=20, replace=False)
        validation = list(set(remainder) - set(test))
        historical_train.extend(train)
        historical_val.extend(validation)
        historical_test.extend(test)
    assert np.array_equal(split.train_idx, historical_train)
    assert np.array_equal(split.val_idx, historical_val)
    assert np.array_equal(split.test_idx, historical_test)


def test_split_is_disjoint_complete_and_exact_per_cell():
    from v2.splits import make_stratified_split

    split = make_stratified_split(_cells(1000), seed=2022, dataset_id="Tiny")
    all_indices = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert len(np.unique(all_indices)) == 2000
    assert set(all_indices) == set(range(2000))
    for start in (0, 1000):
        stop = start + 1000
        assert np.count_nonzero((split.train_idx >= start) & (split.train_idx < stop)) == 600
        assert np.count_nonzero((split.val_idx >= start) & (split.val_idx < stop)) == 200
        assert np.count_nonzero((split.test_idx >= start) & (split.test_idx < stop)) == 200


def test_uneven_cells_use_floor_train_and_test_with_remainder_to_validation():
    from v2.splits import make_stratified_split

    split = make_stratified_split([("BPSK", -2, 7)], seed=2022, dataset_id="Tiny")
    assert (len(split.train_idx), len(split.val_idx), len(split.test_idx)) == (4, 2, 1)
    assert split.rounding == "floor_train_floor_test_validation_remainder"


def test_sample_ids_are_canonical_unique_and_zero_based():
    from v2.splits import make_stratified_split

    split = make_stratified_split(_cells(3), seed=2022, dataset_id="Tiny")
    assert split.sample_ids.tolist() == [
        "Tiny|BPSK|-2|0", "Tiny|BPSK|-2|1", "Tiny|BPSK|-2|2",
        "Tiny|QPSK|+0|0", "Tiny|QPSK|+0|1", "Tiny|QPSK|+0|2",
    ]
    assert len(set(split.sample_ids.tolist())) == 6


def test_sample_identity_is_lazy_unicode_normalized_and_uses_round_trip_snr():
    from v2.splits import make_stratified_split, sample_id

    split = make_stratified_split([("e\u0301", 0.1, 3)], seed=2022, dataset_id="Tiny")
    assert not isinstance(split.sample_ids, np.ndarray)
    assert split.sample_ids[0] == "Tiny|é|+0.10000000000000001|0"
    assert split.sample_ids[np.array([0, 2])].tolist() == [
        "Tiny|é|+0.10000000000000001|0", "Tiny|é|+0.10000000000000001|2",
    ]
    with pytest.raises(ValueError, match="duplicate cell"):
        make_stratified_split([("é", 0.1, 1), ("e\u0301", 0.1, 1)])
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="within_cell_index"):
            sample_id("Tiny", "é", 0.1, invalid)


def test_json_npz_round_trip_and_json_key_order_independence(tmp_path):
    from v2.splits import load_split

    expected, npz, meta, data = _save(tmp_path)
    loaded = load_split(meta, npz_path=npz, data_path=data)
    assert loaded.split_hash == expected.split_hash
    assert np.array_equal(loaded.sample_ids, expected.sample_ids)

    document = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps(dict(reversed(list(document.items()))), indent=2), encoding="utf-8")
    reordered = load_split(meta, npz_path=npz, data_path=data)
    assert reordered.split_hash == loaded.split_hash


def test_metadata_contains_required_provenance_and_portable_path(tmp_path):
    _, npz, meta, data = _save(tmp_path)
    document = json.loads(meta.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["dataset_id"] == "Tiny"
    assert document["dataset_path"] == "data/tiny.bin"
    assert document["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert document["seed"] == 2022
    assert document["ratios"] == {"train": 0.6, "validation": 0.2, "test": 0.2}
    assert document["total_counts"] == {"all": 20, "train": 12, "validation": 4, "test": 4}
    assert document["npz_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert len(document["split_hash"]) == 64


def test_tampered_npz_and_data_drift_are_rejected(tmp_path):
    from v2.splits import SplitValidationError, load_split

    _, npz, meta, data = _save(tmp_path)
    original = npz.read_bytes()
    npz.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(SplitValidationError, match="NPZ SHA-256"):
        load_split(meta, npz_path=npz, data_path=data)

    npz.write_bytes(original)
    data.write_bytes(b"drift")
    with pytest.raises(SplitValidationError, match="data SHA-256"):
        load_split(meta, npz_path=npz, data_path=data)


def test_tampered_indices_with_rehashed_npz_are_rejected_by_split_hash(tmp_path):
    from v2.splits import SplitValidationError, load_split

    _, npz, meta, data = _save(tmp_path)
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["train_idx"] = arrays["train_idx"].copy()
    arrays["train_idx"][0], arrays["train_idx"][1] = arrays["train_idx"][1], arrays["train_idx"][0]
    np.savez(npz, **arrays)
    document = json.loads(meta.read_text(encoding="utf-8"))
    document["npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    document["storage_generation"]["sha256"] = document["npz_sha256"]
    meta.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SplitValidationError, match="split hash"):
        load_split(meta, npz_path=npz, data_path=data)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda d: d.pop("seed"), "seed"),
        (lambda d: d.__setitem__("schema_version", 99), "schema_version"),
        (lambda d: d.__setitem__("per_cell_counts", []), "per_cell_counts"),
        (lambda d: d.__setitem__("dataset_path", "C:\\absolute\\data.pkl"), "dataset_path"),
    ],
)
def test_malformed_metadata_is_rejected(tmp_path, mutation, error):
    from v2.splits import SplitValidationError, load_split

    _, npz, meta, data = _save(tmp_path)
    document = json.loads(meta.read_text(encoding="utf-8"))
    mutation(document)
    meta.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SplitValidationError, match=error):
        load_split(meta, npz_path=npz, data_path=data)


def test_metadata_and_npz_are_portable_with_repository_tree(tmp_path):
    from v2.splits import load_split

    _, npz, meta, _ = _save(tmp_path)
    loaded = load_split(meta, npz_path=npz)
    assert loaded.metadata["dataset_path"] == "data/tiny.bin"


def test_content_addressed_split_publication_preserves_previous_pair_on_failure(tmp_path):
    from v2.splits import load_split, make_stratified_split, save_split

    data = tmp_path / "data" / "tiny.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"stable data")
    metadata_path = tmp_path / "splits" / "v2" / "tiny.json"
    public_npz = tmp_path / "splits" / "v2" / "tiny.npz"
    first = make_stratified_split(_cells(10), seed=2022, dataset_id="Tiny")
    first_meta = save_split(first, public_npz, metadata_path, data_path=data, repository_root=tmp_path)
    metadata_bytes = metadata_path.read_bytes()
    first_blob = tmp_path / first_meta["npz_path"]
    first_blob_hash = hashlib.sha256(first_blob.read_bytes()).hexdigest()
    blobs_before = set((tmp_path / "splits" / "v2").glob("*.npz"))

    second = make_stratified_split(_cells(10), seed=2023, dataset_id="Tiny")
    with pytest.raises(RuntimeError, match="injected publication failure"):
        save_split(
            second, public_npz, metadata_path, data_path=data, repository_root=tmp_path,
            _after_npz_publish=lambda *_: (_ for _ in ()).throw(
                RuntimeError("injected publication failure")
            ),
        )
    assert metadata_path.read_bytes() == metadata_bytes
    assert hashlib.sha256(first_blob.read_bytes()).hexdigest() == first_blob_hash
    blobs_after = set((tmp_path / "splits" / "v2").glob("*.npz"))
    assert blobs_before < blobs_after
    orphan = next(iter(blobs_after - blobs_before))
    assert hashlib.sha256(orphan.read_bytes()).hexdigest() in orphan.name
    assert load_split(metadata_path, data_path=data).split_hash == first.split_hash


def test_content_addressed_split_two_process_interleaving_keeps_both_blobs_and_valid_pointer(tmp_path):
    from v2.splits import load_split, make_stratified_split

    data = tmp_path / "data" / "tiny.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"stable data")
    split_dir = tmp_path / "splits" / "v2"
    split_dir.mkdir(parents=True)
    metadata = split_dir / "tiny.json"
    go = tmp_path / "go"
    children = []
    for seed in (2022, 2023):
        ready = tmp_path / f"ready-{seed}"
        code = f"""
from pathlib import Path
import time
from v2.splits import make_stratified_split, save_split
root=Path({str(tmp_path)!r})
split=make_stratified_split([('BPSK',-2,10),('QPSK',0,10)],seed={seed},dataset_id='Tiny')
def hook(*_):
    (root/'ready-{seed}').write_text('ready',encoding='utf-8')
    deadline=time.monotonic()+5
    while not (root/'go').exists():
        if time.monotonic()>deadline: raise RuntimeError('barrier timeout')
        time.sleep(0.01)
meta=save_split(split,root/'splits/v2/tiny.npz',root/'splits/v2/tiny.json',data_path=root/'data/tiny.bin',repository_root=root,_after_npz_publish=hook)
(root/'hash-{seed}').write_text(meta['split_hash'],encoding='utf-8')
"""
        children.append(subprocess.Popen(
            [__import__("sys").executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))
    deadline = __import__("time").monotonic() + 5
    while not all((tmp_path / f"ready-{seed}").exists() for seed in (2022, 2023)):
        if __import__("time").monotonic() > deadline:
            raise AssertionError("children did not reach publication barrier")
        __import__("time").sleep(0.01)
    go.write_text("go", encoding="utf-8")
    for child in children:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stdout + stderr
    expected = {(tmp_path / f"hash-{seed}").read_text(encoding="utf-8") for seed in (2022, 2023)}
    assert len(list(split_dir.glob("tiny.*.npz"))) == 2
    assert load_split(metadata, data_path=data).split_hash in expected
    assert not list(split_dir.glob(".*.tmp"))


def test_content_addressed_split_keeps_old_blob_and_semantic_hash_is_stable(tmp_path):
    from v2.splits import load_split, make_stratified_split, save_split

    data = tmp_path / "data.bin"
    data.write_bytes(b"stable data")
    metadata_path = tmp_path / "splits" / "v2" / "tiny.json"
    public_npz = tmp_path / "splits" / "v2" / "tiny.npz"
    first = make_stratified_split(_cells(10), seed=2022, dataset_id="Tiny")
    first_meta = save_split(first, public_npz, metadata_path, data_path=data, repository_root=tmp_path)
    first_blob = tmp_path / first_meta["npz_path"]
    same = make_stratified_split(_cells(10), seed=2022, dataset_id="Tiny")
    same_meta = save_split(same, public_npz, metadata_path, data_path=data, repository_root=tmp_path)
    assert first_meta["split_hash"] == same_meta["split_hash"]
    assert first_meta["npz_path"] == same_meta["npz_path"]
    second = make_stratified_split(_cells(10), seed=2023, dataset_id="Tiny")
    second_meta = save_split(second, public_npz, metadata_path, data_path=data, repository_root=tmp_path)
    assert second_meta["npz_path"] != first_meta["npz_path"]
    assert first_blob.is_file()
    assert load_split(metadata_path, data_path=data).split_hash == second.split_hash


def test_split_validation_uses_bounded_coverage_bitmap_without_full_concatenation(tmp_path, monkeypatch):
    import v2.splits as splits_module

    expected, _npz, metadata, data = _save(tmp_path)
    monkeypatch.setattr(
        splits_module.np, "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full split concatenate forbidden")),
    )
    loaded = splits_module.load_split(metadata, data_path=data)
    assert loaded.split_hash == expected.split_hash


def test_gate_tolerance_includes_exactly_half_percentage_point():
    script = _load_script()
    passed, deviation = script.check_gate(0.616, 0.621, 0.5)
    assert passed is True
    assert deviation == pytest.approx(0.5)
    assert script.check_gate(0.61599, 0.621, 0.5)[0] is False


def test_evaluator_reports_overall_per_snr_segments_macro_f1_and_balanced_accuracy():
    script = _load_script()
    labels = np.array([0, 0, 1, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 0, 0])
    snrs = np.array([-20, -8, -6, -2, 0, 18])
    metrics = script.evaluate_predictions(predictions, labels, snrs)
    assert metrics["overall_accuracy"] == pytest.approx(4 / 6)
    assert set(metrics["per_snr_accuracy"]) == {"-20", "-8", "-6", "-2", "+0", "+18"}
    assert set(metrics["segments"]) == {"low", "mid", "high"}
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1


def _tiny_phase1(tmp_path, *, targets=None, count=500):
    script = _load_script()
    root = tmp_path
    (root / "data").mkdir(parents=True)
    data = root / "data" / "tiny.bin"
    data.write_bytes(b"tiny immutable dataset")
    from v2.splits import make_stratified_split, save_split

    split = make_stratified_split(_cells(count), seed=2022, dataset_id="Tiny")
    npz = root / "splits" / "v2" / "tiny.npz"
    meta = npz.with_suffix(".json")
    split_metadata = save_split(split, npz, meta, data_path=data, repository_root=root)
    data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    config = {
        "schema_version": 1,
        "protocol": {
            "version": "phase1-rml2016.10a/4", "baseline_spec_sha256": "pending",
            "trusted_pickle_boundary": "Exact test path and digest are verified before unpickling.",
        },
        "dataset": {"id": "Tiny", "path": "data/tiny.bin", "sha256": data_sha256},
        "split": {"metadata": "splits/v2/tiny.json", "npz": split_metadata["npz_path"]},
        "seed": 2022, "device": "cpu",
        "historical_targets": targets or {"plain": 0.6, "conditioned": 0.7},
        "tolerance_pp": 0.5, "training": {"optimizer": "Adam", "lr": 0.001, "batch_size": 2, "patience": 2, "max_epochs": 2, "milestone_step": 3, "gamma": 0.5},
        "evaluation_batch_size": 2,
        "memory": {
            "max_dataset_and_indices_bytes": 300_000_000,
            "measurement": "steady-state signals+labels+snrs+split-index bytes; excludes transient pickle and allocator overhead",
            "scope": "RML2016.10a test only; RML2018 requires Phase11 memmap.",
        },
        "architecture": {"num_classes": 2},
        "outputs": {"root": "results/v2/reproduction", "gate_state": "results/v2/reproduction/gate_state.json", "manifest": "manifest.json", "progress": "reports/v2_progress.md", "failure_report": "reports/reproduction_failure.md"},
        "models": {
            "plain": {"name": "TinyPlain", "conditioner": "none", "source": "models/model.py"},
            "conditioned": {"name": "TinyCond", "conditioner": "true_snr_embedding", "source": "models/model_snr.py"},
        },
    }
    config["protocol"]["baseline_spec_sha256"] = script.deterministic_json_hash(
        _tiny_baseline(script, config)
    )
    cfg_path = root / "configs" / "v2" / "reproduction.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dataset = {
        "signals": np.zeros((count * 2, 1), dtype=np.float32),
        "labels": np.concatenate([
            np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)
        ]),
        "snrs": np.concatenate([
            np.full(count, -2, dtype=np.float32), np.zeros(count, dtype=np.float32)
        ]),
    }
    return script, root, config, cfg_path, split, dataset


def _tiny_baseline(script, config):
    baseline = copy.deepcopy(script.BASELINE_SPEC)
    baseline["dataset"] = copy.deepcopy(config["dataset"])
    baseline["historical_targets"] = copy.deepcopy(config["historical_targets"])
    baseline["seed"] = config["seed"] if type(config["seed"]) is int else 2022
    baseline["device"] = config["device"]
    baseline["training"] = copy.deepcopy(config["training"])
    baseline["evaluation_batch_size"] = config["evaluation_batch_size"]
    baseline["architecture"] = copy.deepcopy(config["architecture"])
    baseline["models"] = copy.deepcopy(config["models"])
    baseline["paths"]["split_metadata"] = config["split"]["metadata"]
    baseline["memory"]["max_dataset_and_indices_bytes"] = config["memory"]["max_dataset_and_indices_bytes"]
    baseline["memory"]["measurement"] = config["memory"]["measurement"]
    return baseline


def _accuracy_trainer(accuracies, *, exception_run=None, inject_overall=None):
    def trainer(run, dataset, fixed_split, destination, logger):
        if run == exception_run:
            raise RuntimeError(f"injected {run} failure")
        checkpoint = destination / "checkpoint.pt"
        checkpoint.write_bytes(f"{run}-checkpoint".encode())
        indices = fixed_split.test_idx
        labels = np.asarray(dataset["labels"])[indices].copy()
        snrs = np.asarray(dataset["snrs"])[indices].copy()
        predictions = labels.copy()
        error_count = int(round((1.0 - accuracies[run]) * len(labels)))
        predictions[:error_count] = 1 - predictions[:error_count]
        outcome = {
            "labels": labels, "predictions": predictions, "snrs": snrs,
            "checkpoint": checkpoint, "best_epoch": 1,
            "best_val_accuracy": accuracies[run], "duration_seconds": 0.01,
            "parameter_count": 3,
        }
        if inject_overall is not None:
            outcome["overall_accuracy"] = inject_overall[run]
        return outcome
    return trainer


def _run_tiny(setup, trainer, **kwargs):
    script, root, config, cfg_path, _split, dataset = setup
    return script.run_phase1(
        cfg_path, repository_root=root, trainer=trainer,
        dataset_loader=lambda *_: dataset,
        baseline_spec=_tiny_baseline(script, config), **kwargs,
    )


def _assert_no_legacy_writes(root):
    assert not (root / "training").exists()
    assert not (root / "checkpoint").exists()
    assert not (root / "experiments").exists()


def test_prediction_only_gate_passes_and_exact_half_pp_is_inclusive(tmp_path):
    setup = _tiny_phase1(
        tmp_path, targets={"plain": 0.605, "conditioned": 0.705}
    )
    script, root, config, _cfg_path, _split, _dataset = setup
    code = _run_tiny(setup, _accuracy_trainer({"plain": 0.6, "conditioned": 0.7}))
    assert code == 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["runs"]["plain"]["deviation_pp"] == pytest.approx(0.5)
    assert gate["runs"]["conditioned"]["deviation_pp"] == pytest.approx(0.5)
    for run in ("plain", "conditioned"):
        result_path = root / gate["runs"][run]["result_json"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run_spec = json.loads((result_path.parent / "run_spec.json").read_text(encoding="utf-8"))
        assert gate["runs"][run]["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
        assert gate["runs"][run]["result_canonical_hash"] == result["result_canonical_hash"]
        assert gate["baseline_spec_sha256"] == config["protocol"]["baseline_spec_sha256"]
        assert run_spec["baseline_spec_sha256"] == gate["baseline_spec_sha256"]
        assert run_spec["baseline_spec"] == _tiny_baseline(script, config)
        assert run_spec["memory"]["index_backed_views"] is True
        assert run_spec["reference_source_hashes"]["role"] == "non_executed_reference_only"
    _assert_no_legacy_writes(root)


def test_outcome_accuracy_cannot_override_persisted_prediction_gate(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    trainer = _accuracy_trainer(
        {"plain": 0.4, "conditioned": 0.7},
        inject_overall={"plain": 0.6, "conditioned": 0.7},
    )
    assert _run_tiny(setup, trainer) != 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["status"] == "gate_failed"
    assert gate["runs"]["plain"]["actual"] == pytest.approx(0.4)


def test_gate_miss_records_stop_state_manifest_progress_and_preserves_outputs(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    assert _run_tiny(setup, _accuracy_trainer({"plain": 0.5, "conditioned": 0.7})) != 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["status"] == "gate_failed"
    assert gate["runs"]["plain"]["deviation_pp"] > 0.5
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiments"]
    assert {item["model"]: item["status"] for item in manifest} == {
        "TinyPlain": "gate_failed", "TinyCond": "completed"
    }
    failure = (root / "reports/reproduction_failure.md").read_text(encoding="utf-8")
    assert gate["attempt_id"] in failure
    progress = (root / "reports/v2_progress.md").read_text(encoding="utf-8")
    assert "STOP" in progress and "gate did not pass" in progress
    for run in ("plain", "conditioned"):
        destination = root / gate["runs"][run]["destination"]
        assert (destination / "checkpoint.pt").is_file()
        assert (destination / "predictions.npz").is_file()
        assert (destination / "result.json").is_file()
    _assert_no_legacy_writes(root)


def test_run_exception_uses_common_finalizer_and_preserves_prior_failure(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    failure_path = root / config["outputs"]["failure_report"]
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text("# Failures\n\n## Attempt `older`\nold evidence\n", encoding="utf-8")
    assert _run_tiny(
        setup,
        _accuracy_trainer({"plain": 0.6, "conditioned": 0.7}, exception_run="conditioned"),
    ) != 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["status"] == "failed"
    assert gate["runs"]["conditioned"]["status"] == "failed"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiments"]
    assert {item["model"]: item["status"] for item in manifest}["TinyCond"] == "failed"
    failure = failure_path.read_text(encoding="utf-8")
    assert "older" in failure and "old evidence" in failure
    assert "injected conditioned failure" in failure
    progress = (root / "reports/v2_progress.md").read_text(encoding="utf-8")
    assert "STOP" in progress and "injected conditioned failure" in progress
    _assert_no_legacy_writes(root)


def test_conditioned_exception_preserves_plain_gate_miss_as_terminal_evidence(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, cfg_path, _split, dataset = setup
    plain_trainer = _accuracy_trainer({"plain": 0.5, "conditioned": 0.7})
    captured = {}

    def gate_miss_then_exception(run, current_dataset, fixed_split, destination, logger):
        if run == "conditioned":
            plain_result = next(destination.parent.glob("plain_seed*/result.json"))
            captured["path"] = plain_result
            captured["bytes"] = plain_result.read_bytes()
            captured["sha256"] = hashlib.sha256(captured["bytes"]).hexdigest()
            raise RuntimeError("conditioned execution exploded")
        return plain_trainer(run, current_dataset, fixed_split, destination, logger)

    assert script.run_phase1(
        cfg_path, repository_root=root, trainer=gate_miss_then_exception,
        dataset_loader=lambda *_: dataset, baseline_spec=_tiny_baseline(script, config),
    ) != 0
    assert captured["path"].read_bytes() == captured["bytes"]
    assert hashlib.sha256(captured["path"].read_bytes()).hexdigest() == captured["sha256"]
    plain = json.loads(captured["path"].read_text(encoding="utf-8"))
    assert plain["status"] == "gate_failed"
    assert plain["metrics"]["overall_accuracy"] == pytest.approx(0.5)
    assert plain["deviation_pp"] > 0.5

    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["status"] == "failed"
    assert gate["runs"]["plain"]["status"] == "gate_failed"
    assert gate["runs"]["plain"]["actual"] == pytest.approx(0.5)
    assert gate["runs"]["conditioned"]["status"] == "failed"
    assert "conditioned execution exploded" in gate["runs"]["conditioned"]["error"]
    conditioned_result = root / gate["runs"]["conditioned"]["destination"] / "result.json"
    assert json.loads(conditioned_result.read_text(encoding="utf-8"))["status"] == "failed"

    report = (root / config["outputs"]["failure_report"]).read_text(encoding="utf-8")
    assert "plain" in report and "gate_failed" in report
    assert "conditioned execution exploded" in report
    progress = (root / config["outputs"]["progress"]).read_text(encoding="utf-8")
    assert "plain gate_failed" in progress
    assert "conditioned" in progress and "conditioned execution exploded" in progress
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiments"]
    statuses = {item["conditioner"]: item["status"] for item in manifest}
    assert statuses == {"none": "gate_failed", "true_snr_embedding": "failed"}
    _assert_no_legacy_writes(root)


def _valid_tiny_artifact(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    assert _run_tiny(setup, _accuracy_trainer({"plain": 0.6, "conditioned": 0.7})) == 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    destination = root / gate["runs"]["plain"]["destination"]
    code = script._code_identifier(Path(__file__).resolve().parents[2])
    return setup, destination, code


def _canonical_hash(value, *, omit=()):
    from v2.provenance import deterministic_json_hash
    return deterministic_json_hash({key: item for key, item in value.items() if key not in set(omit)})


def _refresh_result_integrity(destination):
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_spec = json.loads((destination / "run_spec.json").read_text(encoding="utf-8"))
    result["artifacts"]["run_spec_sha256"] = hashlib.sha256(
        (destination / "run_spec.json").read_bytes()
    ).hexdigest()
    result["artifacts"]["run_spec_canonical_hash"] = _canonical_hash(run_spec)
    result["artifacts"]["predictions_sha256"] = hashlib.sha256(
        (destination / "predictions.npz").read_bytes()
    ).hexdigest()
    result["result_canonical_hash"] = _canonical_hash(
        result, omit={"result_canonical_hash"}
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _rewrite_npz(path, mutation):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mutation(arrays)
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("wrong_model", "model identity"),
        ("wrong_conditioner", "conditioner identity"),
        ("wrong_seed", "seed identity"),
        ("wrong_path", "artifact path"),
        ("tampered_run_spec", "run_spec file hash"),
        ("reordered_ids", "sample IDs"),
        ("wrong_labels", "labels"),
        ("wrong_snr", "SNR"),
        ("stale_result", "reported metrics"),
        ("hash_mismatch", "checkpoint hash"),
    ],
)
def test_hardened_reuse_rejects_identity_content_and_hash_drift(tmp_path, case, error):
    setup, destination, code = _valid_tiny_artifact(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_spec_path = destination / "run_spec.json"
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    predictions_path = destination / "predictions.npz"
    if case == "wrong_model":
        run_spec["model"]["name"] = "WrongModel"
        run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")
        _refresh_result_integrity(destination)
    elif case == "wrong_conditioner":
        run_spec["model"]["conditioner"] = "wrong"
        run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")
        _refresh_result_integrity(destination)
    elif case == "wrong_seed":
        run_spec["seed"] = 2023
        run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")
        _refresh_result_integrity(destination)
    elif case == "wrong_path":
        result["artifacts"]["checkpoint"] = "results/v2/reproduction/elsewhere/checkpoint.pt"
        result["result_canonical_hash"] = _canonical_hash(result, omit={"result_canonical_hash"})
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif case == "tampered_run_spec":
        run_spec_path.write_text(run_spec_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif case == "reordered_ids":
        _rewrite_npz(predictions_path, lambda a: a.__setitem__("sample_ids", a["sample_ids"][::-1]))
        _refresh_result_integrity(destination)
    elif case == "wrong_labels":
        _rewrite_npz(predictions_path, lambda a: a["labels"].__setitem__(0, 1 - a["labels"][0]))
        _refresh_result_integrity(destination)
    elif case == "wrong_snr":
        _rewrite_npz(predictions_path, lambda a: a["snrs"].__setitem__(0, 18))
        _refresh_result_integrity(destination)
    elif case == "stale_result":
        result["metrics"]["overall_accuracy"] = 0.0
        result["result_canonical_hash"] = _canonical_hash(result, omit={"result_canonical_hash"})
        result_path.write_text(json.dumps(result), encoding="utf-8")
    else:
        (destination / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(script.ReproductionValidationError, match=error):
        script.verify_reusable_result(
            result_path, repository_root=root, config=config, fixed_split=split,
            dataset=dataset, current_code=code, run="plain",
            baseline_spec=_tiny_baseline(script, config),
        )


def test_hardened_reuse_accepts_complete_matching_artifact(tmp_path):
    setup, destination, code = _valid_tiny_artifact(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    result = script.verify_reusable_result(
        destination / "result.json", repository_root=root, config=config,
        fixed_split=split, dataset=dataset, current_code=code, run="plain",
        baseline_spec=_tiny_baseline(script, config),
    )
    assert result["metrics"]["overall_accuracy"] == pytest.approx(0.6)


def test_hardened_reuse_rejects_reparse_artifact_component(tmp_path, monkeypatch):
    setup, destination, code = _valid_tiny_artifact(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    original_is_junction = Path.is_junction
    monkeypatch.setattr(
        Path, "is_junction",
        lambda self: self.name == "checkpoint.pt" or original_is_junction(self),
    )
    with pytest.raises(script.ReproductionValidationError, match="symlink|junction|reparse"):
        script.verify_reusable_result(
            destination / "result.json", repository_root=root, config=config,
            fixed_split=split, dataset=dataset, current_code=code, run="plain",
            baseline_spec=_tiny_baseline(script, config),
        )


def test_legacy_result_integrity_fields_are_added_only_after_full_validation(tmp_path):
    setup, destination, code = _valid_tiny_artifact(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("result_canonical_hash")
    result["artifacts"].pop("run_spec_canonical_hash")
    result["artifacts"].pop("result_json")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    upgraded = script.upgrade_legacy_result_integrity(
        result_path, repository_root=root, config=config, fixed_split=split,
        dataset=dataset, current_code=code, run="plain",
        baseline_spec=_tiny_baseline(script, config),
    )
    assert upgraded["result_canonical_hash"]
    assert upgraded["artifacts"]["run_spec_canonical_hash"]
    script.verify_reusable_result(
        result_path, repository_root=root, config=config, fixed_split=split,
        dataset=dataset, current_code=code, run="plain",
        baseline_spec=_tiny_baseline(script, config),
    )


def test_legacy_result_upgrade_rejects_reordered_sample_ids(tmp_path):
    setup, destination, code = _valid_tiny_artifact(tmp_path)
    script, root, config, _cfg_path, split, dataset = setup
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("result_canonical_hash")
    result["artifacts"].pop("run_spec_canonical_hash")
    result["artifacts"].pop("result_json")
    predictions = destination / "predictions.npz"
    _rewrite_npz(predictions, lambda a: a.__setitem__("sample_ids", a["sample_ids"][::-1]))
    result["artifacts"]["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(script.ReproductionValidationError, match="sample IDs"):
        script.upgrade_legacy_result_integrity(
            result_path, repository_root=root, config=config, fixed_split=split,
            dataset=dataset, current_code=code, run="plain",
            baseline_spec=_tiny_baseline(script, config),
        )


@pytest.mark.parametrize("failure_stage", ["loader", "unpickle", "split", "manifest"])
def test_setup_failures_are_durable_and_do_not_touch_legacy_dirs(tmp_path, monkeypatch, failure_stage):
    setup = _tiny_phase1(tmp_path)
    script, root, config, cfg_path, _split, dataset = setup
    loader = lambda *_: dataset
    if failure_stage == "loader":
        loader = lambda *_: (_ for _ in ()).throw(RuntimeError("loader exploded"))
    elif failure_stage == "unpickle":
        loader = lambda *_: (_ for _ in ()).throw(pickle.UnpicklingError("bad pickle"))
    elif failure_stage == "split":
        monkeypatch.setattr(script, "load_split", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("split invalid")))
    else:
        class BrokenManifest:
            def __init__(self, path): self.path = path
            def upsert(self, record): raise RuntimeError("manifest locked")
        monkeypatch.setattr(script, "ManifestStore", BrokenManifest)
    code = script.run_phase1(
        cfg_path, repository_root=root,
        trainer=_accuracy_trainer({"plain": 0.6, "conditioned": 0.7}),
        dataset_loader=loader,
        baseline_spec=_tiny_baseline(script, config),
    )
    assert code != 0
    gate_path = root / config["outputs"]["gate_state"]
    if failure_stage == "manifest":
        assert not gate_path.exists()
        assert "manifest locked" in (root / config["outputs"]["failure_report"]).read_text(encoding="utf-8")
        assert "STOP" in (root / config["outputs"]["progress"]).read_text(encoding="utf-8")
        _assert_no_legacy_writes(root)
        return
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "failed"
    expected_error = "pickle" if failure_stage == "unpickle" else failure_stage
    assert expected_error in (gate["primary_error"]["message"] + gate["primary_error"]["type"]).lower()
    assert (root / config["outputs"]["failure_report"]).is_file()
    assert "STOP" in (root / config["outputs"]["progress"]).read_text(encoding="utf-8")
    statuses = {item["status"] for item in json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiments"]}
    assert statuses == {"failed"}
    _assert_no_legacy_writes(root)


def test_failure_finalizer_reports_secondary_error_without_masking_primary(tmp_path, monkeypatch):
    setup = _tiny_phase1(tmp_path)
    script, root, config, cfg_path, _split, _dataset = setup
    monkeypatch.setattr(script, "_append_failure_report", lambda *_: (_ for _ in ()).throw(OSError("report disk full")))
    logs = []
    code = script.run_phase1(
        cfg_path, repository_root=root,
        trainer=_accuracy_trainer({"plain": 0.6, "conditioned": 0.7}),
        dataset_loader=lambda *_: (_ for _ in ()).throw(RuntimeError("primary loader failure")),
        baseline_spec=_tiny_baseline(script, config),
        logger=logs.append,
    )
    assert code != 0
    assert not (root / config["outputs"]["gate_state"]).exists()
    assert any("primary loader failure" in line for line in logs)
    assert any("report disk full" in line for line in logs)


@pytest.mark.parametrize("stage", ["manifest", "progress", "failure_report"])
def test_terminal_prerequisite_one_shot_failure_publishes_exactly_one_failed_commit(
    tmp_path, monkeypatch, stage,
):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    output_root = root / config["outputs"]["root"]
    current = root / config["outputs"]["gate_state"]
    script.publish_gate_state(output_root, current, {"attempt_id": "previous", "status": "pass", "runs": {}})
    publish_calls = []
    real_publish = script.publish_gate_state

    def counted_publish(*args, **kwargs):
        publish_calls.append(args[2]["attempt_id"])
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(script, "publish_gate_state", counted_publish)
    if stage == "manifest":
        real_store = script.ManifestStore
        calls = []

        class OneShotManifest:
            def __init__(self, path): self.inner = real_store(path)
            def upsert(self, record):
                calls.append(record)
                if len(calls) == 1:
                    raise OSError("injected manifest failure")
                return self.inner.upsert(record)

        monkeypatch.setattr(script, "ManifestStore", OneShotManifest)
    elif stage == "progress":
        real_progress = script.update_phase_progress
        calls = []

        def one_shot_progress(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1:
                raise OSError("injected progress failure")
            return real_progress(*args, **kwargs)

        monkeypatch.setattr(script, "update_phase_progress", one_shot_progress)
    else:
        real_report = script._append_failure_report
        calls = []

        def one_shot_report(*args, **kwargs):
            calls.append(args[1]["attempt_id"])
            if len(calls) == 1:
                raise OSError("injected failure-report failure")
            return real_report(*args, **kwargs)

        monkeypatch.setattr(script, "_append_failure_report", one_shot_report)

    trainer = _accuracy_trainer(
        {"plain": 0.5 if stage == "failure_report" else 0.6, "conditioned": 0.7}
    )
    assert _run_tiny(setup, trainer, force=True) != 0
    pointer = json.loads(current.read_text(encoding="utf-8"))
    assert pointer["status"] == "failed"
    assert pointer["attempt_id"] != "previous"
    assert publish_calls == [pointer["attempt_id"]]
    immutable = sorted((output_root / "gate_states").glob("*.json"))
    assert len(immutable) == 2
    assert json.loads((root / pointer["attempt_state"]).read_text(encoding="utf-8"))["status"] == "failed"
    report = (root / config["outputs"]["failure_report"]).read_text(encoding="utf-8")
    assert report.count(f"## Attempt `{pointer['attempt_id']}`") == 1
    if stage == "manifest":
        manifest_path = root / config["outputs"]["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["experiments"]
        assert len(records) == 2
        by_result = {record["result_json"]: record for record in records}
        assert set(by_result) == {
            pointer["runs"][run]["result_json"] for run in ("plain", "conditioned")
        }
        for run in ("plain", "conditioned"):
            evidence = pointer["runs"][run]
            assert by_result[evidence["result_json"]]["status"] == evidence["status"]
            result = json.loads((root / evidence["result_json"]).read_text(encoding="utf-8"))
            assert result["status"] == evidence["status"]
        assert pointer["runs"]["plain"]["status"] == "completed"
        assert pointer["runs"]["conditioned"]["status"] == "failed"
        assert [record.experiment for record in calls].count(calls[0].experiment) == 2
        assert len(calls) == 3

        # Replaying the recovered terminal ledger is idempotent.
        for record in calls[1:]:
            real_store(manifest_path).upsert(record)
        replayed = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert replayed == manifest


@pytest.mark.parametrize("stage", ["manifest", "progress", "failure_report"])
def test_persistent_terminal_prerequisite_failure_does_not_advance_gate_commit(
    tmp_path, monkeypatch, stage,
):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    output_root = root / config["outputs"]["root"]
    current = root / config["outputs"]["gate_state"]
    script.publish_gate_state(output_root, current, {"attempt_id": "previous", "status": "pass", "runs": {}})
    previous_bytes = current.read_bytes()
    if stage == "manifest":
        class BrokenManifest:
            def __init__(self, _path): pass
            def upsert(self, _record): raise OSError("persistent manifest failure")
        monkeypatch.setattr(script, "ManifestStore", BrokenManifest)
    elif stage == "progress":
        monkeypatch.setattr(
            script, "update_phase_progress",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("persistent progress failure")),
        )
    else:
        monkeypatch.setattr(
            script, "_append_failure_report",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("persistent report failure")),
        )
    trainer = _accuracy_trainer(
        {"plain": 0.5 if stage == "failure_report" else 0.6, "conditioned": 0.7}
    )
    assert _run_tiny(setup, trainer, force=True) != 0
    assert current.read_bytes() == previous_bytes
    assert {path.stem for path in (output_root / "gate_states").glob("*.json")} == {"previous"}


def test_invalid_prevalidation_seed_uses_single_safe_setup_failure_directory(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, cfg_path, _split, _dataset = setup
    config["seed"] = "../../models/escape"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert script.run_phase1(
        cfg_path, repository_root=root,
        trainer=_accuracy_trainer({"plain": 0.6, "conditioned": 0.7}),
        dataset_loader=lambda *_: (_ for _ in ()).throw(AssertionError("must not load")),
        baseline_spec=_tiny_baseline(script, config),
    ) != 0
    output_root = root / "results/v2/reproduction"
    setup_dirs = [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("setup_failed_")]
    assert len(setup_dirs) == 1
    assert (setup_dirs[0] / "result.json").is_file()
    assert not list(output_root.glob("plain_seed*"))
    assert not list(output_root.glob("conditioned_seed*"))
    assert not (root / "models" / "escape").exists()


def test_malformed_config_is_finalized_durably(tmp_path):
    script = _load_script()
    config_path = tmp_path / "configs" / "v2" / "reproduction.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: [unterminated\n", encoding="utf-8")
    assert script.run_phase1(config_path, repository_root=tmp_path) != 0
    gate_path = tmp_path / "results" / "v2" / "reproduction" / "gate_state.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "failed"
    assert "yaml" in gate["primary_error"]["message"].lower()
    assert (tmp_path / "reports" / "reproduction_failure.md").is_file()
    assert "STOP" in (tmp_path / "reports" / "v2_progress.md").read_text(encoding="utf-8")
    report = (tmp_path / "reports" / "reproduction_failure.md").read_text(encoding="utf-8")
    assert "setup failures may occur before model artifacts are created" in report
    assert "every generated artifact" not in report


def test_evaluation_publication_failure_is_finalized_and_bundle_is_preserved(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, cfg_path, _split, dataset = setup
    base = _accuracy_trainer({"plain": 0.6, "conditioned": 0.7})

    def wrong_labels(run, current_dataset, fixed_split, destination, logger):
        outcome = base(run, current_dataset, fixed_split, destination, logger)
        if run == "plain":
            outcome["labels"] = 1 - outcome["labels"]
        return outcome

    assert script.run_phase1(
        cfg_path, repository_root=root, trainer=wrong_labels,
        dataset_loader=lambda *_: dataset, baseline_spec=_tiny_baseline(script, config),
    ) != 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert "labels do not match" in gate["primary_error"]["message"]
    destination = root / gate["runs"]["plain"]["destination"]
    assert (destination / "predictions.npz").is_file()
    assert json.loads((destination / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    _assert_no_legacy_writes(root)


@pytest.mark.parametrize(
    "invalid_predictions",
    [np.array([0.0] * 199 + [0.5]), np.array([0] * 199 + [-1]), np.array([0] * 199 + [2])],
)
def test_raw_trainer_prediction_domain_is_rejected_before_publication_cast(tmp_path, invalid_predictions):
    setup = _tiny_phase1(tmp_path, count=500)
    script, root, config, cfg_path, _split, dataset = setup
    base = _accuracy_trainer({"plain": 0.6, "conditioned": 0.7})

    def invalid_trainer(run, current_dataset, fixed_split, destination, logger):
        outcome = base(run, current_dataset, fixed_split, destination, logger)
        if run == "plain":
            outcome["predictions"] = invalid_predictions
        return outcome

    assert script.run_phase1(
        cfg_path, repository_root=root, trainer=invalid_trainer,
        dataset_loader=lambda *_: dataset, baseline_spec=_tiny_baseline(script, config),
    ) != 0
    gate = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert gate["runs"]["plain"]["status"] == "failed"
    assert "predictions" in gate["primary_error"]["message"]
    assert not (root / gate["runs"]["plain"]["destination"] / "predictions.npz").exists()


def test_conditioned_model_dependency_is_present_tracked_and_matches_recorded_hash():
    script = _load_script()
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models/model_snr.py"
    assert model_path.is_file()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "models/model_snr.py"],
        cwd=root, capture_output=True, text=True,
    )
    assert tracked.returncode == 0
    expected = "802079e1d6a594a73e358d7c8c5e2800a0a428446eab9efe79eaea543ac96b2a"
    assert script._canonical_source_sha256(model_path) == expected
    assert script.validate_conditioned_dependency(root) == expected
    run_specs = sorted((root / "results/v2/reproduction").glob("conditioned_seed*/run_spec.json"))
    assert any(json.loads(path.read_text(encoding="utf-8"))["code"]["files"]["models/model_snr.py"] == expected for path in run_specs)


def test_canonical_source_digest_is_identical_for_lf_crlf_and_mixed_newlines(tmp_path):
    script = _load_script()
    variants = [b"one\ntwo\nthree\n", b"one\r\ntwo\r\nthree\r\n", b"one\r\ntwo\nthree\r\n"]
    digests = []
    for index, payload in enumerate(variants):
        path = tmp_path / f"source-{index}.py"
        path.write_bytes(payload)
        digests.append(script._canonical_source_sha256(path))
    assert len(set(digests)) == 1


def test_validate_only_rejects_legacy_checkpoint_destination(tmp_path):
    script = _load_script()
    config = {"outputs": {"root": "training/reproduction"}}
    with pytest.raises(script.ReproductionValidationError, match="results/v2/reproduction"):
        script.validate_output_root(config, tmp_path)


def _real_reproduction_config():
    root = Path(__file__).resolve().parents[2]
    return root, yaml.safe_load((root / "configs/v2/reproduction.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda c: c["dataset"].update(id="RML2018.01a"), "dataset.id"),
        (lambda c: c["dataset"].update(path="data/other.pkl"), "dataset.path"),
        (lambda c: c["dataset"].update(sha256="0" * 64), "dataset.sha256"),
        (lambda c: c["protocol"].update(baseline_spec_sha256="0" * 64), "protocol.baseline_spec_sha256"),
        (lambda c: c["historical_targets"].update(plain=0.6209), "historical_targets.plain"),
        (lambda c: c["historical_targets"].update(conditioned=0.6711), "historical_targets.conditioned"),
        (lambda c: c.update(tolerance_pp=0.5001), "tolerance_pp"),
        (lambda c: c["historical_targets"].update(plain=float("nan")), "historical_targets.plain"),
        (lambda c: c["historical_targets"].update(plain=float("inf")), "historical_targets.plain"),
        (lambda c: c.update(tolerance_pp=float("nan")), "tolerance_pp"),
        (lambda c: c.update(tolerance_pp=float("inf")), "tolerance_pp"),
        (lambda c: c.update(tolerance_pp=True), "tolerance_pp"),
    ],
)
def test_immutable_baseline_rejects_scientific_gate_mutations(mutation, error):
    script = _load_script()
    root, config = _real_reproduction_config()
    mutation(config)
    with pytest.raises(script.ReproductionValidationError, match=error):
        script.validate_config(config, root)


def test_baseline_spec_locks_complete_scientific_configuration():
    script = _load_script()
    _root, config = _real_reproduction_config()
    assert script.BASELINE_SPEC["seed"] == config["seed"]
    assert script.BASELINE_SPEC["device"] == config["device"]
    assert script.BASELINE_SPEC["training"] == config["training"]
    assert script.BASELINE_SPEC["architecture"] == config["architecture"]
    assert script.BASELINE_SPEC["models"] == config["models"]
    assert script.BASELINE_SPEC["evaluation_batch_size"] == config["evaluation_batch_size"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "optimizer", "SGD"),
        ("training", "lr", 0.002),
        ("training", "batch_size", 64),
        ("training", "patience", 9),
        ("training", "max_epochs", 99),
        ("training", "milestone_step", 2),
        ("training", "gamma", 0.25),
        ("architecture", "num_classes", 10),
        ("architecture", "num_levels", 2),
        ("architecture", "in_channels", 32),
        ("architecture", "kernel_size", 5),
        ("architecture", "latent_dim", 160),
        ("architecture", "regu_details", 0.02),
        ("architecture", "regu_approx", 0.02),
        ("architecture", "num_snr_bins", 19),
        ("architecture", "snr_embedding_dim", 7),
    ],
)
def test_complete_scientific_sections_reject_every_field_mutation(section, field, value):
    script = _load_script()
    root, config = _real_reproduction_config()
    config[section][field] = value
    with pytest.raises(script.ReproductionValidationError, match=rf"{section}\.{field}"):
        script.validate_config(config, root)


@pytest.mark.parametrize(
    ("run", "field", "value"),
    [
        ("plain", "name", "Other"), ("plain", "conditioner", "wrong"),
        ("plain", "source", "models/model_snr.py"),
        ("conditioned", "name", "Other"), ("conditioned", "conditioner", "wrong"),
        ("conditioned", "source", "models/model.py"),
    ],
)
def test_model_definitions_are_exactly_locked(run, field, value):
    script = _load_script()
    root, config = _real_reproduction_config()
    config["models"][run][field] = value
    with pytest.raises(script.ReproductionValidationError, match=rf"models\.{run}\.{field}"):
        script.validate_config(config, root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(evaluation_batch_size=32),
        lambda c: c.update(evaluation_batch_size=64.0),
        lambda c: c["training"].update(batch_size=128.0),
        lambda c: c["architecture"].update(regu_details=True),
        lambda c: c["training"].update(extra_scientific_knob=1),
        lambda c: c["architecture"].update(extra_scientific_knob=1),
        lambda c: c["models"]["plain"].update(extra_scientific_knob=1),
    ],
)
def test_scientific_config_rejects_wrong_types_and_unknown_fields(mutation):
    script = _load_script()
    root, config = _real_reproduction_config()
    mutation(config)
    with pytest.raises(script.ReproductionValidationError):
        script.validate_config(config, root)


def test_run_spec_model_construction_identity_matches_declared_constructor(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    assert _run_tiny(setup, _accuracy_trainer({"plain": 0.6, "conditioned": 0.7})) == 0
    for run in ("plain", "conditioned"):
        run_spec_path = next((root / "results/v2/reproduction").glob(f"{run}_seed*/run_spec.json"))
        run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        assert run_spec["model_construction"] == script._model_construction_identity(run, config)
        assert run_spec["architecture"] == config["architecture"]
        assert run_spec["evaluation_batch_size"] == config["evaluation_batch_size"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dataset", "path", "models/model.py"),
        ("dataset", "path", "configs/v2/reproduction.yaml"),
        ("split", "metadata", "data/RML2016.10a_dict.pkl"),
        ("split", "metadata", "models/model.py"),
        ("split", "npz", "checkpoint/legacy.pkl"),
        ("split", "npz", "configs/v2/reproduction.yaml"),
        ("outputs", "root", "models"),
        ("outputs", "gate_state", "data/gate.json"),
        ("outputs", "manifest", "results/v2/reproduction/manifest.json"),
        ("outputs", "progress", "models/progress.md"),
        ("outputs", "failure_report", "configs/v2/failure.md"),
    ],
)
def test_configurable_paths_use_strict_allowlist(section, field, value):
    script = _load_script()
    root, config = _real_reproduction_config()
    config[section][field] = value
    with pytest.raises(script.ReproductionValidationError, match=f"{section}.{field}"):
        script.validate_config(config, root)


def test_config_path_is_exact_and_reparse_components_are_rejected(tmp_path):
    script = _load_script()
    root, _config = _real_reproduction_config()
    with pytest.raises(script.ReproductionValidationError, match="config path"):
        script._validated_config_path(root, root / "configs/v2/base.yaml", script.BASELINE_SPEC)

    target = tmp_path / "target"
    target.mkdir()
    (target / "file.txt").write_text("safe", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(script.ReproductionValidationError, match="symlink|junction|reparse"):
        script._repo_path(tmp_path, "alias/file.txt", "test.path")


def test_trusted_pickle_rejects_path_and_digest_before_loader_executes(tmp_path):
    script = _load_script()
    approved = tmp_path / "data" / "RML2016.10a_dict.pkl"
    approved.parent.mkdir(parents=True)
    approved.write_bytes(b"not the trusted pickle")
    calls = []

    def forbidden_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("pickle loader must not execute")

    with pytest.raises(script.ReproductionValidationError, match="SHA-256"):
        script._load_rml_dataset(
            approved, "RML2016.10a", repository_root=tmp_path,
            approved_sha256="0" * 64, pickle_loader=forbidden_loader,
        )
    with pytest.raises(script.ReproductionValidationError, match="approved dataset path"):
        script._load_rml_dataset(
            tmp_path / "other.pkl", "RML2016.10a", repository_root=tmp_path,
            approved_sha256=hashlib.sha256(b"not the trusted pickle").hexdigest(),
            pickle_loader=forbidden_loader,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("predictions", "error"),
    [
        (np.array([0.0, 1.5]), "integer"),
        (np.array([0, -1]), "range"),
        (np.array([0, 2]), "range"),
    ],
)
def test_prediction_values_are_integral_and_in_class_range(predictions, error):
    script = _load_script()
    with pytest.raises(script.ReproductionValidationError, match=error):
        script.evaluate_predictions(
            predictions, np.array([0, 1]), np.array([-2, 0]), num_classes=2,
        )


def test_historical_epoch_control_matches_loss_patience_lr_and_accuracy_checkpoint_reference():
    script = _load_script()
    losses = [1.0, 1.1, 1.2, 1.2, 1.3, 1.4, 1.5]
    accuracies = [0.4, 0.5, 0.5, 0.49, 0.51, 0.50, 0.49]

    def reference():
        best_loss = None
        best_accuracy = 0.0
        counter = 0
        lr = 0.001
        rows = []
        for loss, accuracy in zip(losses, accuracies):
            checkpoint = accuracy >= best_accuracy
            if checkpoint:
                best_accuracy = accuracy
            if best_loss is None or loss <= best_loss:  # equality resets, matching EarlyStopping(delta=0)
                best_loss = loss
                counter = 0
            else:
                counter += 1
            lr_before = lr
            if counter and counter % 2 == 0:
                lr *= 0.5
            rows.append({
                "checkpoint": checkpoint, "best_accuracy": best_accuracy,
                "counter": counter, "lr_before": lr_before, "lr_after": lr,
                "stop": counter >= 4,
            })
            if counter >= 4:
                break
        return rows

    control = script.HistoricalEpochControl(patience=4, milestone_step=2, gamma=0.5, learning_rate=0.001)
    actual = []
    for loss, accuracy in zip(losses, accuracies):
        row = control.step(loss, accuracy)
        actual.append(row)
        if row["stop"]:
            break
    assert actual == reference()
    assert [index for index, row in enumerate(actual) if row["checkpoint"]] == [0, 1, 2, 4]


def test_index_dataset_shares_signal_storage_and_matches_historical_global_rng_batch_order():
    torch = pytest.importorskip("torch")
    script = _load_script()
    signals = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    labels = torch.arange(10, dtype=torch.int64)
    snrs = torch.arange(-10, 10, 2, dtype=torch.float32)
    indices = np.array([9, 1, 7, 3, 5, 0, 8, 2], dtype=np.int64)

    torch.manual_seed(2022)
    torch.nn.Linear(4, 3)  # historical model initialization consumes global RNG first
    reference_dataset = torch.utils.data.TensorDataset(
        signals[torch.from_numpy(indices)], labels[torch.from_numpy(indices)],
        torch.from_numpy(script._snr_bins(snrs.numpy()))[torch.from_numpy(indices)],
    )
    reference_loader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=3, shuffle=True, num_workers=0,
    )
    reference_order = torch.cat([batch[1] for batch in reference_loader]).tolist()

    torch.manual_seed(2022)
    torch.nn.Linear(4, 3)
    view = script.make_index_dataset(signals, labels, snrs, indices)
    assert view.signals.data_ptr() == signals.data_ptr()
    assert view.labels.data_ptr() == labels.data_ptr()
    loader = script.make_historical_loader(view, batch_size=3, shuffle=True)
    assert loader.generator is None
    actual_order = torch.cat([batch[1] for batch in loader]).tolist()
    assert actual_order == reference_order


def test_train_and_validation_shuffle_share_global_rng_across_multiple_epochs():
    torch = pytest.importorskip("torch")
    script = _load_script()
    train = torch.utils.data.TensorDataset(torch.arange(11, dtype=torch.int64))
    validation = torch.utils.data.TensorDataset(torch.arange(100, 107, dtype=torch.int64))

    def consume(train_loader, validation_loader):
        trace = []
        for _epoch in range(3):
            train_order = torch.cat([batch[0] for batch in train_loader]).tolist()
            validation_order = torch.cat([batch[0] for batch in validation_loader]).tolist()
            trace.append((train_order, validation_order))
        return trace

    torch.manual_seed(2022)
    torch.nn.Linear(4, 3)  # model initialization consumes global RNG before loader iteration
    reference = consume(
        torch.utils.data.DataLoader(train, batch_size=3, shuffle=True, num_workers=0),
        torch.utils.data.DataLoader(validation, batch_size=2, shuffle=True, num_workers=0),
    )

    torch.manual_seed(2022)
    torch.nn.Linear(4, 3)
    actual_loaders = script.make_historical_train_val_loaders(
        train, validation, train_batch_size=3, validation_batch_size=2,
    )
    assert actual_loaders[0].generator is None
    assert actual_loaders[1].generator is None
    actual = consume(*actual_loaders)
    assert actual == reference

    torch.manual_seed(2022)
    torch.nn.Linear(4, 3)
    wrong = consume(
        torch.utils.data.DataLoader(train, batch_size=3, shuffle=True, num_workers=0),
        torch.utils.data.DataLoader(validation, batch_size=2, shuffle=False, num_workers=0),
    )
    assert wrong[1][0] != reference[1][0]  # validation randperm changes next-epoch train order


def test_phase1_memory_estimate_is_bounded_and_rml2018_is_explicitly_out_of_scope():
    script = _load_script()
    signals = np.zeros((220_000, 2, 128), dtype=np.float32)
    labels = np.zeros(220_000, dtype=np.int64)
    snrs = np.zeros(220_000, dtype=np.float32)
    indices = np.arange(220_000, dtype=np.int64)
    measured = script.estimate_phase1_dataset_bytes(
        {"signals": signals, "labels": labels, "snrs": snrs}, indices,
    )
    assert measured == signals.nbytes + labels.nbytes + snrs.nbytes + indices.nbytes
    assert measured <= script.BASELINE_SPEC["memory"]["max_dataset_and_indices_bytes"]
    assert "steady-state" in script.BASELINE_SPEC["memory"]["measurement"]
    assert "RML2018" in script.BASELINE_SPEC["memory"]["out_of_scope"]
    assert "memmap" in script.BASELINE_SPEC["memory"]["out_of_scope"]


def test_declared_source_provenance_is_complete_and_each_dependency_changes_identity(tmp_path):
    script = _load_script()
    root = Path(__file__).resolve().parents[2]
    required = {
        "scripts/v2/phase1_reproduce.py", "v2/splits.py", "v2/contracts.py",
        "v2/manifest.py", "v2/progress.py", "v2/provenance.py",
        "models/model.py", "models/model_snr.py", "models/lifting.py", "util/utils.py",
    }
    assert set(script.SOURCE_DEPENDENCIES) == required
    for relative in script.SOURCE_DEPENDENCIES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    original = script._code_identifier(tmp_path)
    for relative in script.SOURCE_DEPENDENCIES:
        path = tmp_path / relative
        prior = path.read_bytes()
        path.write_bytes(prior + b"\n# provenance mutation\n")
        changed = script._code_identifier(tmp_path)
        assert changed["hash"] != original["hash"], relative
        with pytest.raises(script.ReproductionValidationError, match="code identity"):
            script._code_record_matches(original, changed)
        path.write_bytes(prior)


def test_strict_code_identity_uses_committed_executable_bytes_and_ignores_dirty_references(tmp_path):
    script = _load_script()
    root = Path(__file__).resolve().parents[2]
    reference_only = {
        "data_loader/data_loader.py", "util/training.py",
        "util/early_stop.py", "util/logger.py",
    }
    assert set(script.REFERENCE_SOURCE_DEPENDENCIES) == reference_only
    assert set(script.SOURCE_DEPENDENCIES).isdisjoint(reference_only)

    clean = tmp_path / "clean"
    committed_hashes = {}
    for relative in (*script.SOURCE_DEPENDENCIES, *script.REFERENCE_SOURCE_DEPENDENCIES):
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=root, check=True,
            capture_output=True,
        ).stdout
        committed_hashes[relative] = hashlib.sha256(
            committed.replace(b"\r\n", b"\n")
        ).hexdigest()
        destination = clean / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkout_bytes = committed.replace(b"\n", b"\r\n") if relative == "models/model_snr.py" else committed
        destination.write_bytes(checkout_bytes)

    strict_before = script._code_identifier(clean)
    assert strict_before["files"] == {
        relative: committed_hashes[relative] for relative in script.SOURCE_DEPENDENCIES
    }
    references_before = script._reference_source_hashes(clean)
    assert references_before["role"] == "non_executed_reference_only"
    dirty_reference = clean / "util/early_stop.py"
    dirty_reference.write_bytes(dirty_reference.read_bytes() + b"\n# user dirty reference\n")
    assert script._code_identifier(clean)["hash"] == strict_before["hash"]
    assert script._reference_source_hashes(clean)["files"] != references_before["files"]


def test_cross_process_phase_lock_has_one_owner_bounded_timeout_and_preserves_evidence(tmp_path):
    script = _load_script()
    lock_path = tmp_path / "results" / "v2" / "reproduction" / ".phase1.lock"
    gate_path = lock_path.parent / "gate_state.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(b'{"status":"preserved"}\n')
    owner = script.PhaseExecutionLock(lock_path, timeout_seconds=1.0)
    owner.acquire()
    code = (
        "import importlib.util, pathlib; "
        f"p=pathlib.Path({str(Path(script.__file__))!r}); "
        "s=importlib.util.spec_from_file_location('phase1_lock_child',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"lock=m.PhaseExecutionLock(pathlib.Path({str(lock_path)!r}),timeout_seconds=0.2); "
        "lock.acquire()"
    )
    started = __import__("time").perf_counter()
    child = subprocess.run(
        [__import__("sys").executable, "-c", code], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[2], timeout=3,
    )
    elapsed = __import__("time").perf_counter() - started
    try:
        assert child.returncode != 0
        assert "execution lock timeout" in child.stderr
        assert elapsed < 2.0
        assert gate_path.read_bytes() == b'{"status":"preserved"}\n'
    finally:
        owner.release()
    assert lock_path.exists()
    with script.PhaseExecutionLock(lock_path, timeout_seconds=0.2):
        pass


def test_phase_lock_recovers_stale_owner_file_without_waiting(tmp_path):
    script = _load_script()
    lock_path = tmp_path / "results/v2/reproduction/.phase1.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("pid=999999;started=stale\n", encoding="utf-8")
    started = __import__("time").perf_counter()
    with script.PhaseExecutionLock(lock_path, timeout_seconds=0.5):
        assert lock_path.exists()
    assert __import__("time").perf_counter() - started < 0.4
    assert "pid=" in lock_path.read_text(encoding="utf-8")


def test_gate_publication_keeps_immutable_attempt_states_and_atomically_advances_current(tmp_path):
    script = _load_script()
    output_root = tmp_path / "results" / "v2" / "reproduction"
    current = output_root / "gate_state.json"
    first = {"attempt_id": "attempt-one", "status": "failed", "runs": {"plain": {"status": "failed"}}}
    first_path = script.publish_gate_state(output_root, current, first)
    first_bytes = first_path.read_bytes()
    first_pointer = json.loads(current.read_text(encoding="utf-8"))
    assert first_pointer["attempt_state"] == first_path.relative_to(tmp_path).as_posix()
    assert first_pointer["attempt_state_sha256"] == hashlib.sha256(first_bytes).hexdigest()

    second = {"attempt_id": "attempt-two", "status": "pass", "runs": {}}
    second_path = script.publish_gate_state(output_root, current, second)
    assert first_path.read_bytes() == first_bytes
    assert second_path != first_path
    assert json.loads(current.read_text(encoding="utf-8"))["attempt_id"] == "attempt-two"
    with pytest.raises(script.ReproductionValidationError, match="immutable gate state"):
        script.publish_gate_state(output_root, current, first | {"status": "pass"})


def test_forced_attempts_keep_distinct_manifest_and_immutable_gate_evidence(tmp_path):
    setup = _tiny_phase1(tmp_path)
    script, root, config, _cfg_path, _split, _dataset = setup
    trainer = _accuracy_trainer({"plain": 0.6, "conditioned": 0.7})
    assert _run_tiny(setup, trainer, force=True) == 0
    first_pointer = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    first_state = root / first_pointer["attempt_state"]
    first_bytes = first_state.read_bytes()
    assert _run_tiny(setup, trainer, force=True) == 0
    second_pointer = json.loads((root / config["outputs"]["gate_state"]).read_text(encoding="utf-8"))
    assert second_pointer["attempt_id"] != first_pointer["attempt_id"]
    assert first_state.read_bytes() == first_bytes
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiments"]
    assert len(manifest) == 4
    assert len({item["result_json"] for item in manifest}) == 4
    assert {item["status"] for item in manifest} == {"completed"}

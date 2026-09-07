import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml


def _load_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase3", root / "scripts/v2/run_phase3.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path, monkeypatch):
    module = _load_module()
    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "repo"
    root.mkdir()
    config = yaml.safe_load((source_root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    config["outputs"] = {
        "root": "results/v2/phase3/test",
        "manifest": "results/v2/phase3/test/manifest.json",
        "progress": "reports/v2_phase3/test.md",
    }
    config_path = tmp_path / "phase3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    split = type(
        "Split",
        (),
        {
            "train_idx": np.array([0, 1]),
            "val_idx": np.array([2, 3]),
            "test_idx": np.arange(20),
            "sample_ids": np.asarray([f"sample-{index}" for index in range(20)]),
            "split_hash": module.LOCKED_SPLIT_HASH,
        },
    )()
    monkeypatch.setattr(module, "load_split", lambda *_a, **_k: split)
    monkeypatch.setattr(module, "_code_identity", lambda _root: {"hash": "test-code", "files": {}})
    monkeypatch.setattr(module, "_environment_identity", lambda: {"identity": "test-environment"})
    dataset = {
        "signals": np.zeros((20, 2, 128), dtype=np.float32),
        "labels": np.arange(20, dtype=np.int64) % 2,
        "snrs": np.asarray(module.LOCKED_SNR_VALUES, dtype=np.float32),
    }
    return module, root, config_path, dataset


def _outcome(model_id, dataset, split, destination):
    logits = np.full((len(split.test_idx), 11), -4.0, dtype=np.float32)
    labels = dataset["labels"][split.test_idx]
    logits[np.arange(len(labels)), labels] = 4.0
    (destination / "checkpoint.pt").write_bytes(("checkpoint-" + model_id).encode())
    return {
        "checkpoint": destination / "checkpoint.pt",
        "logits": logits,
        "labels": labels,
        "snrs": dataset["snrs"][split.test_idx],
        "best_epoch": 0,
        "epochs_completed": 100,
        "best_val_accuracy": 1.0,
        "duration_seconds": 0.01,
        "scheduler": {"name": "none", "reason": "Phase 2 protocol used no scheduler"},
        "complexity": {
            "total_parameters": 10,
            "additional_conditioner_parameters": 2,
            "parameter_increase_percent": 25.0,
            "flops": {"available": False, "reason": "test"},
            "latency": _fake_latency(),
            "memory": {"available": False, "reason": "test"},
        },
        "mechanism_tensors": _fake_mechanism(model_id),
    }


def _fake_mechanism(model_id):
    zeros = lambda shape: np.zeros(shape, dtype=np.float32)
    if model_id == "M0": return {}
    if model_id == "M1": return {"immediate_affine.condition_weight": zeros((320, 1))}
    if model_id == "M2": return {"immediate_affine.condition_weight": zeros((320, 20))}
    if model_id == "M3": return {
        "snr_embedding_vectors": zeros((20, 8)),
        "immediate_affine.condition_weight": zeros((320, 8)),
        "per_snr_additive_contribution": zeros((20, 320)),
    }
    if model_id == "M4": return {"per_snr_logit_bias": zeros((20, 11))}
    if model_id == "M5": return {"per_snr_gamma": zeros((20, 128)), "per_snr_beta": zeros((20, 128))}
    if model_id == "M6": return {"per_snr_gates": zeros((20, 128))}
    values = {}
    for index in range(20):
        values[f"snr_heads.{index}.0.weight"] = zeros((320, 128))
        values[f"snr_heads.{index}.0.bias"] = zeros((320,))
        values[f"snr_heads.{index}.2.weight"] = zeros((11, 320))
        values[f"snr_heads.{index}.2.bias"] = zeros((11,))
    return values


def _fake_latency():
    measurement = lambda batch, value: {"batch_size": batch, "mean_ms": value, "median_ms": value}
    return {
        "device": "cpu", "warmup": 1, "repeats": 1, "timer": "time.perf_counter",
        "bin_construction": "single_bin uses bin 0; mixed_bins cycles deterministically over bins 0..19",
        "single_bin": {"batch_1": measurement(1, 0.1), "fixed_batch": measurement(64, 1.0)},
        "mixed_bins": {"batch_1": measurement(1, 0.1), "fixed_batch": measurement(64, 1.1)},
        "summary": {"basis": "mixed_bins.fixed_batch", "mean_ms": 1.1, "median_ms": 1.1},
    }


def _resume_verifier(model_id, data, split, _candidate, _logger, *, config):
    labels = data["labels"][split.test_idx]
    logits = np.full((len(labels), 11), -4.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 4.0
    return {"logits": logits, "mechanism_tensors": _fake_mechanism(model_id), "parameter_counts": {"total_parameters": 10, "additional_conditioner_parameters": 2, "parameter_increase_percent": 25.0}}


def _source_validator(*_args, **_kwargs):
    return None


def test_phase3_runs_only_new_models_and_accounts_for_all_40_records(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    trained, reused = [], []

    def trainer(model_id, data, split, destination, _logger, *, config):
        trained.append((model_id, config["_active_seed"]))
        return _outcome(model_id, data, split, destination)

    def reuser(model_id, seed, data, split, destination, _logger, *, config):
        reused.append((model_id, seed))
        outcome = _outcome(model_id, data, split, destination)
        outcome["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        outcome["source_artifact_hashes"] = {"run_spec_sha256": "c" * 64, "result_sha256": "d" * 64, "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}
        return outcome

    code = module.run_phase3(
        config_path,
        repository_root=root,
        trainer=trainer,
        phase2_reuser=reuser,
        dataset_loader=lambda *_a: dataset,
        logger=lambda _line: None,
        smoke=True,
        completed_verifier=_resume_verifier,
        phase2_source_validator=_source_validator,
    )
    assert code == 0
    assert {item[0] for item in reused} == {"M0", "M3"}
    assert len(reused) == 10
    assert {item[0] for item in trained} == {"M1", "M2", "M4", "M5", "M6", "M7"}
    assert len(trained) == 30

    output = root / "results/v2/phase3/test"
    runs = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(runs) == 40
    result = json.loads((runs[0] / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert {"overall_accuracy", "macro_f1", "balanced_accuracy", "nll", "ece", "per_snr", "segments"} <= set(result["metrics"])
    assert result["artifacts"]["mechanism"] == "mechanism.npz"
    assert result["complexity"]["flops"]["available"] is False
    spec_value = json.loads((runs[0] / "run_spec.json").read_text(encoding="utf-8"))
    assert spec_value["provenance"]["environment"] == {"identity": "test-environment"}
    assert result["provenance"]["environment"] == {"identity": "test-environment"}
    with np.load(runs[0] / "predictions.npz", allow_pickle=False) as bundle:
        assert set(bundle.files) == set(module.PREDICTION_FIELDS)
        assert bundle["split_hash"].item() == module.LOCKED_SPLIT_HASH
        assert bundle["model_id"].item() == result["model_id"]
    with np.load(runs[0] / "mechanism.npz", allow_pickle=False) as mechanism:
        assert {"model_id", "seed", "split_hash", "preprocessing_hash", "protocol_hash", "tensor_names"} <= set(mechanism.files)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["experiments"]
    assert len(manifest) == 40
    assert all(not Path(item["result_json"]).is_absolute() for item in manifest)
    assert all(not item["checkpoint"] or not Path(item["checkpoint"]).is_absolute() for item in manifest)
    progress = (root / "reports/v2_phase3/test.md").read_text(encoding="utf-8")
    assert "40/40" in progress and "10 reused" in progress and "30 trained" in progress
    shutil.rmtree(output)


def test_prediction_bundle_is_bound_to_labels_snrs_ids_and_identity(tmp_path):
    module = _load_module()
    count = len(module.LOCKED_SNR_VALUES)
    split = type("Split", (), {"test_idx": np.arange(count)[::-1], "sample_ids": np.asarray([f"id-{index}" for index in range(count)]), "split_hash": module.LOCKED_SPLIT_HASH})()
    dataset = {"labels": np.arange(count) % 2, "snrs": np.asarray(module.LOCKED_SNR_VALUES)}
    identity = {"model_id": "M1", "seed": 2022, "split_hash": module.LOCKED_SPLIT_HASH, "preprocessing_hash": "c" * 64, "protocol_hash": "d" * 64}
    labels = dataset["labels"][split.test_idx]
    logits = np.full((count, 2), -2.0)
    logits[np.arange(count), labels] = 2.0
    path = tmp_path / "predictions.npz"
    module.write_prediction_bundle(path, {"logits": logits, "labels": labels, "snrs": dataset["snrs"][split.test_idx]}, split, identity)
    metrics = module.validate_prediction_bundle(path, split, dataset, identity=identity, num_classes=2, snr_values=module.LOCKED_SNR_VALUES)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["nll"] > 0 and metrics["ece"] >= 0
    with np.load(path, allow_pickle=False) as bundle:
        tampered = {name: bundle[name] for name in bundle.files}
    tampered["sample_ids"] = tampered["sample_ids"][::-1]
    np.savez_compressed(path, **tampered)
    with pytest.raises(ValueError, match="sample IDs"):
        module.validate_prediction_bundle(path, split, dataset, identity=identity, num_classes=2, snr_values=module.LOCKED_SNR_VALUES)


def test_nll_uses_stable_logsumexp_for_extreme_wrong_margin():
    module = _load_module()
    snrs = np.asarray(module.LOCKED_SNR_VALUES)
    labels = np.zeros(len(snrs), dtype=np.int64)
    logits = np.zeros((len(snrs), 2), dtype=np.float64)
    logits[:, 1] = 10_000.0
    metrics = module.compute_metrics(labels, logits, snrs, num_classes=2, snr_values=module.LOCKED_SNR_VALUES)
    assert metrics["nll"] == pytest.approx(10_000.0, abs=1e-9)
    assert np.isfinite(metrics["ece"])


def test_nll_avoids_cancellation_with_huge_common_logit_offset():
    module = _load_module()
    snrs = np.asarray(module.LOCKED_SNR_VALUES)
    labels = np.zeros(len(snrs), dtype=np.int64)
    logits = np.full((len(snrs), 2), 1e20, dtype=np.float64)
    metrics = module.compute_metrics(labels, logits, snrs, num_classes=2, snr_values=module.LOCKED_SNR_VALUES)
    assert metrics["nll"] == pytest.approx(np.log(2.0), abs=1e-15)


def test_completed_run_is_reused_only_after_hash_and_metric_validation(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    calls = {"train": 0, "reuse": 0}

    def trainer(model_id, data, split, destination, _logger, *, config):
        calls["train"] += 1
        return _outcome(model_id, data, split, destination)

    def reuser(model_id, seed, data, split, destination, _logger, *, config):
        calls["reuse"] += 1
        value = _outcome(model_id, data, split, destination)
        value["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        value["source_artifact_hashes"] = {"run_spec_sha256": "c" * 64, "result_sha256": "d" * 64, "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}
        return value

    kwargs = dict(repository_root=root, trainer=trainer, phase2_reuser=reuser, dataset_loader=lambda *_a: dataset, logger=lambda _x: None, smoke=True, completed_verifier=_resume_verifier, phase2_source_validator=_source_validator)
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls == {"train": 30, "reuse": 10}
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls == {"train": 30, "reuse": 10}

    candidate = next((root / "results/v2/phase3/test").glob("M1_seed2022_*"))
    result = json.loads((candidate / "result.json").read_text(encoding="utf-8"))
    result["metrics"]["overall_accuracy"] = 0.0
    (candidate / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls["train"] == 31
    assert len(list((root / "results/v2/phase3/test").glob("M1_seed2022_*"))) == 2


def test_failures_are_persisted_and_retried_without_overwrite(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    def broken(*_a, **_k):
        raise RuntimeError("injected trainer failure")
    def reuser(model_id, seed, data, split, destination, _logger, *, config):
        value = _outcome(model_id, data, split, destination)
        value["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        value["source_artifact_hashes"] = {"run_spec_sha256": "c" * 64, "result_sha256": "d" * 64, "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}
        return value
    kwargs = dict(repository_root=root, trainer=broken, phase2_reuser=reuser, dataset_loader=lambda *_a: dataset, logger=lambda _x: None, smoke=True, completed_verifier=_resume_verifier, phase2_source_validator=_source_validator)
    assert module.run_phase3(config_path, **kwargs) == 1
    output = root / "results/v2/phase3/test"
    failed = next(output.glob("M1_seed2022_*"))
    assert (failed / "failure.json").is_file()
    assert json.loads((failed / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert module.run_phase3(config_path, **kwargs) == 1
    assert len(list(output.glob("M1_seed2022_*"))) == 2
    progress = (root / "reports/v2_phase3/test.md").read_text(encoding="utf-8")
    assert "INCOMPLETE" in progress and "10 completed" in progress
    assert "30 failed" in progress and "0 pending" in progress and "BLOCKED" in progress


def test_protocol_and_output_paths_are_strict():
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = module.load_phase3_config(root / "configs/v2/phase3.yaml")
    assert module.protocol_hash(config) == module.protocol_hash(config)
    for key, value in (("seeds", [2022]), ("dataset", {"id": "other", "path": config["dataset"]["path"]}), ("training", {**config["training"], "max_epochs": 1})):
        mutated = {**config, key: value}
        with pytest.raises(ValueError):
            module.validate_phase3_config(mutated)
    with pytest.raises(ValueError, match="allowlisted"):
        module.validate_output_paths(root, {"root": "../outside", "manifest": "manifest.json", "progress": "reports/v2_progress.md"})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("snr", "snr_min_db"), -18),
        (("snr", "snr_max_db"), 20),
        (("snr", "scalar_normalization", "expression"), "wrong"),
        (("snr_bands", "low", "max_db"), -10),
        (("models", "M5", "function"), "beta only"),
        (("models", "M6", "range"), "[0, 1]"),
        (("models", "M7", "head_count"), 19),
        (("models", "M7", "head_architecture"), "128 -> 11"),
    ],
)
def test_protocol_rejects_every_conditioning_or_snr_semantic_mutation(path, value):
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        module.validate_phase3_config(config)


def test_protocol_rejects_unknown_confounding_fields():
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    config["models"]["M4"]["extra_capacity"] = True
    with pytest.raises(ValueError):
        module.validate_phase3_config(config)
    config = yaml.safe_load((root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    config["outputs"]["untracked_report"] = "reports/untracked.md"
    with pytest.raises(ValueError):
        module.validate_phase3_config(config)


def test_atomic_npz_cleanup_and_phase_lock(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.np, "savez_compressed", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        module._atomic_npz(tmp_path / "x.npz", {"x": np.array([1])})
    assert not list(tmp_path.glob("*.tmp"))

    lock_path = root / "results/v2/phase3/test/.phase3.lock"
    lock_path.parent.mkdir(parents=True)
    lock = module.PhaseExecutionLock(lock_path, timeout_seconds=0.0)
    with lock:
        assert module.run_phase3(config_path, repository_root=root, trainer=lambda *_a, **_k: pytest.fail("must not train"), dataset_loader=lambda *_a: dataset, logger=lambda _x: None, smoke=True, lock_timeout_seconds=0.0) == 3


def test_smoke_mode_requires_injected_execution(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="injected"):
        module._run_phase3_unlocked(config_path, repository_root=root, dataset_loader=lambda *_a: dataset, smoke=True, logger=lambda _x: None)


def test_mechanism_extraction_persists_realized_per_snr_controls():
    torch = pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = module.load_phase3_config(root / "configs/v2/phase3.yaml")
    expected = {
        "M3": {"snr_embedding_vectors", "immediate_affine.condition_weight", "per_snr_additive_contribution"},
        "M4": {"per_snr_logit_bias"},
        "M5": {"per_snr_gamma", "per_snr_beta"},
        "M6": {"per_snr_gates"},
        "M7": {"snr_heads.0.0.weight", "snr_heads.19.2.weight"},
    }
    for model_id, keys in expected.items():
        model = module._model(config, model_id)
        tensors = module._mechanism_tensors(model, config)
        assert keys <= set(tensors)
        assert all(np.isfinite(value).all() for value in tensors.values())
    assert module._mechanism_tensors(module._model(config, "M0"), config) == {}


def test_cli_smoke_check_exercises_forward_batch_without_artifact_writes(tmp_path):
    pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    summary = module.smoke_check(root / "configs/v2/phase3.yaml")
    assert summary == {model_id: [2, 11] for model_id in module.MODEL_IDS}
    assert list(tmp_path.iterdir()) == []


def test_smoke_check_rejects_nonfinite_logits(monkeypatch):
    torch = pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    class BadModel:
        def eval(self): return self
        def forward_batch(self, signals, *, snr_db, snr_bin):
            return torch.full((len(signals), 11), float("nan")), []
    monkeypatch.setattr(module, "_model", lambda *_a: BadModel())
    with pytest.raises(RuntimeError, match="forward_batch smoke check failed"):
        module.smoke_check(root / "configs/v2/phase3.yaml")


def test_conditioning_dataset_carries_true_snr_and_independently_validated_bin():
    torch = pytest.importorskip("torch")
    module = _load_module()
    signals = torch.zeros((3, 2, 128))
    labels = torch.tensor([0, 1, 2])
    snrs = torch.tensor([-20.0, -2.0, 18.0])
    dataset = module.make_conditioning_dataset(signals, labels, snrs, np.array([2, 0]))
    _, label, snr_db, snr_bin = dataset[0]
    assert label.item() == 2 and snr_db.item() == 18.0 and snr_bin.item() == 19
    with pytest.raises(ValueError, match="grid"):
        module.make_conditioning_dataset(signals, labels, torch.tensor([-20.0, -1.0, 18.0]), np.array([0, 1]))


def test_complexity_and_mechanism_evidence_schemas_are_exact(tmp_path):
    module = _load_module()
    valid = _outcome("M5", {"labels": np.array([0]), "snrs": np.array([-20])}, type("S", (), {"test_idx": np.array([0])})(), tmp_path)["complexity"]
    module.validate_complexity(valid, fixed_batch=64)
    with pytest.raises(ValueError):
        module.validate_complexity({**valid, "unknown": 1}, fixed_batch=64)
    broken = json.loads(json.dumps(valid)); broken["latency"]["mixed_bins"]["batch_1"]["mean_ms"] = float("nan")
    with pytest.raises(ValueError):
        module.validate_complexity(broken, fixed_batch=64)
    unfair = json.loads(json.dumps(valid)); unfair["latency"]["summary"]["basis"] = "single_bin.fixed_batch"
    with pytest.raises(ValueError, match="mixed_bins"):
        module.validate_complexity(unfair, fixed_batch=64)

    identity = {"model_id": "M5", "seed": 2022, "split_hash": "a" * 64, "preprocessing_hash": "b" * 64, "protocol_hash": "c" * 64, "checkpoint_sha256": "d" * 64}
    path = tmp_path / "mechanism.npz"
    module.write_mechanism_bundle(path, {"mechanism_tensors": _fake_mechanism("M5")}, identity)
    module.validate_mechanism_bundle(path, identity=identity, architecture={"num_snr_bins": 20, "num_classes": 11, "latent_dim": 320}, pooled_feature_dim=128, expected_tensors=_fake_mechanism("M5"))
    with np.load(path, allow_pickle=False) as bundle:
        changed = {name: bundle[name] for name in bundle.files}
    changed["tensor_000"] = changed["tensor_000"] + 1
    np.savez_compressed(path, **changed)
    with pytest.raises(ValueError, match="checkpoint-derived"):
        module.validate_mechanism_bundle(path, identity=identity, architecture={"num_snr_bins": 20, "num_classes": 11, "latent_dim": 320}, pooled_feature_dim=128, expected_tensors=_fake_mechanism("M5"))


def test_latency_uses_deterministic_mixed_bins_and_true_grid_values(monkeypatch):
    torch = pytest.importorskip("torch")
    module = _load_module()
    observed = []
    class SpyModel:
        def forward_batch(self, signals, *, snr_db, snr_bin):
            observed.append((len(signals), snr_db.cpu().clone(), snr_bin.cpu().clone()))
            return torch.zeros((len(signals), 11)), []
    monkeypatch.setattr(module, "LATENCY_WARMUP", 1)
    monkeypatch.setattr(module, "LATENCY_REPEATS", 1)
    evidence = module._latency(SpyModel(), signal_shape=(2, 128), fixed_batch=4, snr_min=-20, device=torch.device("cpu"))
    mixed_fixed = [item for item in observed if item[0] == 4 and item[2].tolist() == [0, 1, 2, 3]]
    assert mixed_fixed and mixed_fixed[-1][1].tolist() == [-20.0, -18.0, -16.0, -14.0]
    assert evidence["summary"] == {
        "basis": "mixed_bins.fixed_batch",
        "mean_ms": evidence["mixed_bins"]["fixed_batch"]["mean_ms"],
        "median_ms": evidence["mixed_bins"]["fixed_batch"]["median_ms"],
    }


def test_resume_uses_external_ledger_hash_and_checkpoint_recomputation(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    calls = {"train": 0, "verify": 0}
    def trainer(model_id, data, split, destination, _logger, *, config):
        calls["train"] += 1
        return _outcome(model_id, data, split, destination)
    def reuser(model_id, seed, data, split, destination, _logger, *, config):
        value = _outcome(model_id, data, split, destination)
        value["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        value["source_artifact_hashes"] = {"run_spec_sha256": "c" * 64, "result_sha256": "d" * 64, "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}
        return value
    def verifier(*args, **kwargs):
        calls["verify"] += 1
        return _resume_verifier(*args, **kwargs)
    kwargs = dict(repository_root=root, trainer=trainer, phase2_reuser=reuser, dataset_loader=lambda *_a: dataset, logger=lambda _x: None, smoke=True, completed_verifier=verifier, phase2_source_validator=_source_validator)
    assert module.run_phase3(config_path, **kwargs) == 0
    output = root / "results/v2/phase3/test"
    ledger = json.loads((output / "phase3_ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["records"]) == 40 and all(len(item["result_sha256"]) == 64 for item in ledger["records"])
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls["verify"] == 40
    spec_candidate = next(output.glob("M4_seed2022_*"))
    spec_path = spec_candidate / "run_spec.json"
    spec_value = json.loads(spec_path.read_text(encoding="utf-8")); spec_value["untrusted_extra"] = True
    spec_path.write_text(json.dumps(spec_value), encoding="utf-8")
    result_path = spec_candidate / "result.json"
    result_value = json.loads(result_path.read_text(encoding="utf-8")); result_value["artifacts"]["run_spec_sha256"] = module.file_sha256(spec_path)
    result_path.write_text(json.dumps(result_value), encoding="utf-8")
    ledger_path = output / "phase3_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    attempt_path = spec_candidate.resolve().relative_to(root.resolve()).as_posix()
    next(item for item in ledger["records"] if item["attempt_path"] == attempt_path)["result_sha256"] = module.file_sha256(result_path)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls["train"] == 31
    candidate = next(output.glob("M2_seed2022_*"))
    (candidate / "checkpoint.pt").write_bytes(b"tampered")
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls["train"] == 32


def test_ledger_rejects_nonhex_result_hash(tmp_path):
    module = _load_module()
    path = tmp_path / "phase3_ledger.json"
    path.write_text(json.dumps({
        "schema_version": 1, "runner_version": module.RUNNER_VERSION,
        "records": [{"model_id": "M1", "seed": 2022, "run_fingerprint": "a" * 64, "attempt_path": "results/v2/phase3/run", "result_sha256": "z" * 64, "status": "completed", "updated_at": "now"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="result hash"):
        module._load_ledger(path)


@pytest.mark.parametrize(
    ("model_id", "directory"),
    [("M0", "plain_seed2022_64befcda9f51"), ("M3", "conditioned_seed2022_e69ee4acd27d")],
)
def test_actual_phase2_reuse_layers_accept_real_artifacts_and_match_tiny_probe(model_id, directory):
    torch = pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = module.load_phase3_config(root / "configs/v2/phase3.yaml")
    source_bundle = root / f"results/v2/phase2/{directory}/predictions.npz"
    split_path = root / config["split"]["metadata"]
    data_path = root / config["dataset"]["path"]
    missing = [str(path.relative_to(root)) for path in (source_bundle, split_path, data_path) if not path.is_file()]
    if missing:
        pytest.skip("ignored Phase 2 integration artifacts are absent in this checkout: " + ", ".join(missing))
    split = module.load_split(split_path, data_path=data_path)
    with np.load(source_bundle, allow_pickle=False) as bundle:
        size = int(np.max(split.test_idx)) + 1
        labels = np.zeros(size, dtype=np.int64); snrs = np.zeros(size, dtype=np.float32)
        labels[split.test_idx] = bundle["labels"]; snrs[split.test_idx] = bundle["snrs"]
    fixed_dataset = {"labels": labels, "snrs": snrs}
    source = module._find_phase2_source(model_id, 2022, fixed_dataset, split, config, root)
    assert source.name == directory
    assert len(module._validate_phase2_prediction_rows(source_bundle, split, fixed_dataset)) == len(split.test_idx)
    unified = module._load_phase2_compatible_model(model_id, source / "checkpoint.pt", config)
    architecture = config["architecture"]
    common = {key: architecture[key] for key in ("num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim", "regu_details", "regu_approx")}
    if model_id == "M3":
        from models.model_snr import AWNSNR
        legacy = AWNSNR(**common, num_snr_bins=20, snr_emb_dim=8).eval()
    else:
        from models.model import AWN
        legacy = AWN(**common).eval()
    legacy.load_state_dict(module._load_state(source / "checkpoint.pt", torch.device("cpu")), strict=True)
    signals = torch.zeros((2, 2, 128)); bins = torch.tensor([0, 19]); snr_db = torch.tensor([-20.0, 18.0])
    with torch.no_grad():
        legacy_logits = legacy(signals, bins)[0] if model_id == "M3" else legacy(signals)[0]
        assert torch.equal(legacy_logits, unified.forward_batch(signals, snr_db=snr_db, snr_bin=bins)[0])


def test_synthetic_phase2_checkpoint_and_prediction_fixtures_cover_reuse_layers(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = module.load_phase3_config(root / "configs/v2/phase3.yaml")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(module._model(config, "M3").state_dict(), checkpoint)
    loaded = module._load_phase2_compatible_model("M3", checkpoint, config)
    assert loaded.conditioning == "M3"
    split = type("Split", (), {"test_idx": np.array([2, 0]), "sample_ids": np.array(["a", "b", "c"])})()
    dataset = {"labels": np.array([0, 1, 2]), "snrs": np.array([-20, -2, 18])}
    bundle = tmp_path / "predictions.npz"
    np.savez_compressed(bundle, predictions=np.array([2, 0]), labels=np.array([2, 0]), snrs=np.array([18, -20]), sample_ids=np.array(["c", "a"]))
    predictions = module._validate_phase2_prediction_rows(bundle, split, dataset)
    assert predictions.tolist() == [2, 0]


@pytest.mark.parametrize("model_id", ["M1", "M7"])
def test_default_trainer_bounded_real_model_updates_reloads_and_aligns_rows(model_id, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    config = module.load_phase3_config(root / "configs/v2/phase3.yaml")
    config = {
        **config,
        "training": {"optimizer": "Adam", "lr": 0.001, "batch_size": 2, "max_epochs": 1},
        "evaluation_batch_size": 4,
        "architecture": {**config["architecture"], "num_classes": 3, "in_channels": 4, "latent_dim": 8, "snr_embedding_dim": 2},
        "_active_seed": 7,
    }
    dataset = {"signals": np.zeros((20, 2, 128), dtype=np.float32), "labels": np.arange(20, dtype=np.int64) % 3, "snrs": np.asarray(module.LOCKED_SNR_VALUES, dtype=np.float32)}
    split = type("Split", (), {"train_idx": np.array([0, 1, 2, 3]), "val_idx": np.array([4, 5, 6, 7]), "test_idx": np.arange(20)})()
    original_model = module._model
    observed = {}
    def capture_model(active, selected):
        model = original_model(active, selected)
        observed["model"] = model
        observed["initial"] = {name: value.detach().clone() for name, value in model.state_dict().items() if torch.is_floating_point(value)}
        return model
    monkeypatch.setattr(module, "_model", capture_model)
    monkeypatch.setattr(module, "_complexity", lambda *_a, **_k: {
        "total_parameters": 10, "additional_conditioner_parameters": 2, "parameter_increase_percent": 25.0,
        "flops": {"available": False, "reason": "bounded test"}, "latency": _fake_latency(), "memory": {"available": False, "reason": "bounded test"},
    })
    real_loss = module._training_loss; loss_checks = []
    def capture_loss(logits, labels, regularizers):
        value = real_loss(logits, labels, regularizers)
        expected = torch.nn.functional.cross_entropy(logits, labels) + sum(regularizers)
        loss_checks.append(torch.allclose(value, expected))
        return value
    monkeypatch.setattr(module, "_training_loss", capture_loss)
    destination = tmp_path / model_id; destination.mkdir()
    outcome = module._default_trainer(model_id, dataset, split, destination, lambda _x: None, config=config)
    assert loss_checks and all(loss_checks)
    assert outcome["best_epoch"] == 0 and outcome["epochs_completed"] == 1
    assert outcome["scheduler"] == {"name": "none", "reason": "Phase 2 protocol used no scheduler"}
    assert outcome["labels"].tolist() == dataset["labels"][split.test_idx].tolist()
    assert outcome["snrs"].tolist() == dataset["snrs"][split.test_idx].tolist()
    assert outcome["logits"].shape == (len(split.test_idx), 3)
    assert any(not torch.equal(observed["initial"][name], value) for name, value in observed["model"].state_dict().items() if name in observed["initial"])
    reloaded = original_model(config, model_id)
    reloaded.load_state_dict(module._load_state(destination / "checkpoint.pt", torch.device("cpu")), strict=True)
    expected_logits = module._evaluate(reloaded, dataset, split, config, torch.device("cpu"))
    assert np.array_equal(expected_logits, outcome["logits"])
    metrics = module.compute_metrics(outcome["labels"], outcome["logits"], outcome["snrs"], num_classes=3, snr_values=module.LOCKED_SNR_VALUES)
    assert metrics["overall_accuracy"] >= 0.0


def test_checkpoint_ties_replace_the_previous_best():
    module = _load_module()
    assert module._should_checkpoint(0.5, 0.5) is True


def test_source_identity_includes_contract_and_environment_dependencies():
    module = _load_module()
    assert "v2/contracts.py" in module.SOURCE_FILES
    identity = module._environment_identity()
    assert {"python", "numpy", "torch", "torch_build", "cuda_available", "cpu", "device", "torch_num_threads", "torch_num_interop_threads", "timing"} == set(identity)


def test_environment_identity_change_forces_new_attempts(tmp_path, monkeypatch):
    module, root, config_path, dataset = _fixture(tmp_path, monkeypatch)
    calls = {"trained": 0, "reused": 0}
    def trainer(model_id, data, split, destination, _logger, *, config):
        calls["trained"] += 1
        return _outcome(model_id, data, split, destination)
    def reuser(model_id, seed, data, split, destination, _logger, *, config):
        calls["reused"] += 1
        value = _outcome(model_id, data, split, destination)
        value["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        value["source_artifact_hashes"] = {"run_spec_sha256": "c" * 64, "result_sha256": "d" * 64, "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64}
        return value
    kwargs = dict(repository_root=root, trainer=trainer, phase2_reuser=reuser, dataset_loader=lambda *_a: dataset,
        logger=lambda _x: None, smoke=True, completed_verifier=_resume_verifier, phase2_source_validator=_source_validator)
    assert module.run_phase3(config_path, **kwargs) == 0
    monkeypatch.setattr(module, "_environment_identity", lambda: {"identity": "changed-environment"})
    assert module.run_phase3(config_path, **kwargs) == 0
    assert calls == {"trained": 60, "reused": 20}
    assert len([path for path in (root / "results/v2/phase3/test").iterdir() if path.is_dir()]) == 80

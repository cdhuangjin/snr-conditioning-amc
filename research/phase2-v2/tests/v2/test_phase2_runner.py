import json
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml


def _module_and_temp_root(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", source_root / "scripts/v2/run_phase2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"; root.mkdir()
    split = type("Split", (), {"test_idx": np.array([0, 1, 2, 3]), "train_idx": np.array([0, 1]), "val_idx": np.array([2, 3]), "sample_ids": np.array(["a", "b", "c", "d"]), "split_hash": module.LOCKED_SPLIT_HASH})()
    monkeypatch.setattr(module, "load_split", lambda *_args, **_kwargs: split)
    monkeypatch.setattr(module, "_code_identity", lambda _root: {"hash": "test-code", "files": {}})
    return module, root


def test_phase2_persists_recomputable_artifacts_and_paired_summary(tmp_path, monkeypatch):
    module, root = _module_and_temp_root(tmp_path, monkeypatch)
    run_phase2 = module.run_phase2
    config_path = tmp_path / "phase2.yaml"
    output = root / "results/v2/phase2/pytest_main"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "dataset": {"id": "RML2016.10a", "path": "data/RML2016.10a_dict.pkl"},
        "seeds": [2022, 2023, 2024, 2025, 2026],
        "split": {"metadata": "splits/v2/RML2016.10a_seed2022.json", "hash": "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"},
        "training": {"optimizer": "Adam", "max_epochs": 1},
        "architecture": {"num_classes": 2},
        "treatments": {
            "plain": {"conditioner": "none", "loss": "cross_entropy", "snr_reweighting": "none"},
            "conditioned": {"conditioner": "true_snr_embedding", "loss": "cross_entropy", "snr_reweighting": "none"},
            "focal": {"conditioner": "none", "loss": "focal", "gamma": 2.0, "snr_reweighting": "none"},
            "snr_reweighted": {"conditioner": "none", "loss": "cross_entropy", "snr_reweighting": "fixed_segment_weights", "segment_weights": {"low": 2.0, "mid": 1.0, "high": 0.5}},
        },
        "outputs": {"root": "results/v2/phase2/pytest_main", "manifest": "results/v2/phase2/pytest_main/manifest.json", "progress": "reports/v2_phase2/pytest_main.md"},
    }, sort_keys=False), encoding="utf-8")

    def loader(_path, _dataset_id):
        # The runner's injected-loader path must not deserialize the large pickle.
        return {"signals": np.zeros((4, 2, 128), dtype=np.float32), "labels": np.arange(4, dtype=np.int64) % 2, "snrs": np.zeros(4, dtype=np.float32)}

    def trainer(treatment, _dataset, split, destination, _logger, *, config):
        (destination / "checkpoint.pt").write_bytes(b"checkpoint")
        labels = _dataset["labels"][split.test_idx]
        predictions = labels if treatment == "conditioned" else np.zeros_like(labels)
        return {"checkpoint": destination / "checkpoint.pt", "predictions": predictions, "labels": labels, "snrs": np.asarray(split.test_idx) * 0.0, "best_epoch": 0, "epochs_completed": 1, "best_val_accuracy": 0.5, "duration_seconds": 0.01, "parameter_count": 1}

    assert run_phase2(config_path, repository_root=root, trainer=trainer, dataset_loader=loader, logger=lambda _: None) == 0
    runs = sorted(output.glob("*_seed*"))
    assert len(runs) == 20
    result = json.loads((runs[0] / "result.json").read_text(encoding="utf-8"))
    assert {"overall_accuracy", "per_snr_accuracy", "segments", "macro_f1", "balanced_accuracy", "kappa", "prediction_concentration", "prediction_entropy", "gini_prediction_distribution", "hhi", "concentration_minus_balanced_accuracy", "per_class_recall", "dominant_predicted_class", "confusion_counts", "feature_geometry"} <= set(result["metrics"])
    with np.load(runs[0] / "predictions.npz", allow_pickle=False) as bundle:
        assert {"predictions", "labels", "snrs", "sample_ids"} == set(bundle.files)
    assert module.recompute_metrics(runs[0] / "predictions.npz", num_classes=2)["overall_accuracy"] == result["metrics"]["overall_accuracy"]
    paired = json.loads((output / "paired_summary.json").read_text(encoding="utf-8"))
    assert paired["comparisons"]["conditioned_vs_plain"]["overall_accuracy"]["n_pairs"] == 5

    # A matching completed result is reused, not replaced.
    assert run_phase2(config_path, repository_root=root, trainer=trainer, dataset_loader=loader, logger=lambda _: None) == 0
    assert len(list(output.glob("*_seed*"))) == 20
    shutil.rmtree(output)


def test_phase2_locks_protocol_and_uses_nonunit_predeclared_snr_weights():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    config = module.load_phase2_config(root / "configs/v2/phase2.yaml")
    assert module.snr_segment_weights(np.array([-10, -4, 4]), config["treatments"]["snr_reweighted"]) == [2.0, 1.0, 0.5]
    for field, replacement in (("dataset", {"id": "other", "path": "data/RML2016.10a_dict.pkl"}), ("split", {"metadata": "splits/v2/other.json"}), ("seeds", [2022])):
        mutated = dict(config); mutated[field] = replacement
        with __import__("pytest").raises(ValueError):
            module.validate_phase2_config(mutated)
    for treatment, replacement in (("plain", {"conditioner": "true_snr_embedding", "loss": "cross_entropy", "snr_reweighting": "none"}), ("conditioned", {"conditioner": "true_snr_embedding", "loss": "focal", "gamma": 2.0, "snr_reweighting": "none"}), ("focal", {"conditioner": "none", "loss": "focal", "gamma": 2.0, "snr_reweighting": "fixed_segment_weights"})):
        mutated = {**config, "treatments": {**config["treatments"], treatment: replacement}}
        with pytest.raises(ValueError, match="semantics"):
            module.validate_phase2_config(mutated)
    confounded = {**config, "treatments": {**config["treatments"], "snr_reweighted": {**config["treatments"]["snr_reweighted"], "gamma": 2.0}}}
    with pytest.raises(ValueError, match="snr_reweighted"):
        module.validate_phase2_config(confounded)


def test_phase2_rejects_absolute_or_traversal_output_paths():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="allowlisted"):
        module.validate_output_paths(root, {"root": "C:/unsafe", "manifest": "manifest.json", "progress": "reports/v2_progress.md"})
    with pytest.raises(ValueError, match="allowlisted"):
        module.validate_output_paths(root, {"root": "results/v2/phase2/../../unsafe", "manifest": "manifest.json", "progress": "reports/v2_progress.md"})


def test_code_identity_changes_when_direct_dependency_changes(tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", source_root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    for relative in module.SOURCE_FILES:
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(relative, encoding="utf-8")
    before = module._code_identity(tmp_path)
    (tmp_path / "util/utils.py").write_text("changed", encoding="utf-8")
    assert module._code_identity(tmp_path)["hash"] != before["hash"]


def test_atomic_artifact_writes_remove_temps_after_failure(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", source_root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    with pytest.raises(TypeError):
        module._atomic_json(tmp_path / "result.json", {"not_json": object()})
    monkeypatch.setattr(module.np, "savez_compressed", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        module._write_predictions(tmp_path / "predictions.npz", {"predictions": [0], "labels": [0], "snrs": [0]}, type("Split", (), {"test_idx": np.array([0]), "sample_ids": np.array(["a"])})())
    monkeypatch.setattr(Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failure")))
    with pytest.raises(OSError, match="replace failure"):
        module._atomic_json(tmp_path / "replace.json", {"ok": True})
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.npz"))


def test_phase2_execution_lock_rejects_concurrent_orchestration(tmp_path, monkeypatch):
    module, root = _module_and_temp_root(tmp_path, monkeypatch)
    source_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((source_root / "configs/v2/phase2.yaml").read_text(encoding="utf-8"))
    config["outputs"] = {"root": "results/v2/phase2/locked", "manifest": "results/v2/phase2/locked/manifest.json", "progress": "reports/v2_phase2/locked.md"}
    path = tmp_path / "phase2.yaml"; path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock = module.PhaseExecutionLock(root / "results/v2/phase2/locked/.phase2.lock", timeout_seconds=0.0)
    with lock:
        assert module.run_phase2(path, repository_root=root, trainer=lambda *_args, **_kwargs: pytest.fail("trainer must not run"), dataset_loader=lambda *_args: {}, logger=lambda _: None, lock_timeout_seconds=0.0) == 3


def test_failed_runs_write_result_and_rerun_creates_retry(tmp_path, monkeypatch):
    module, root = _module_and_temp_root(tmp_path, monkeypatch)
    source_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((source_root / "configs/v2/phase2.yaml").read_text(encoding="utf-8"))
    config["outputs"] = {"root": "results/v2/phase2/pytest_failures", "manifest": "results/v2/phase2/pytest_failures/manifest.json", "progress": "reports/v2_phase2/pytest_failures.md"}
    path = tmp_path / "phase2.yaml"; path.write_text(yaml.safe_dump(config), encoding="utf-8")
    def loader(_path, _id): return {"signals": np.zeros((1, 2, 128), dtype=np.float32), "labels": np.zeros(1, dtype=np.int64), "snrs": np.zeros(1, dtype=np.float32)}
    def broken(*_args, **_kwargs): raise RuntimeError("expected failure")
    output_root = root / "results/v2/phase2/pytest_failures"
    assert module.run_phase2(path, repository_root=root, trainer=broken, dataset_loader=loader, logger=lambda _: None) == 1
    first = sorted(output_root.glob("*_seed*")); assert len(first) == 20
    assert json.loads((first[0] / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    failure = json.loads((first[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["type"] == "RuntimeError"
    assert failure["message"] == "expected failure"
    assert "RuntimeError: expected failure" in failure["traceback"]
    assert module.run_phase2(path, repository_root=root, trainer=broken, dataset_loader=loader, logger=lambda _: None) == 1
    assert len(list(output_root.glob("*_seed*"))) == 40
    shutil.rmtree(output_root)


def test_prediction_bundle_rejects_wrong_fixed_split_labels(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    split = module.load_split(root / "splits/v2/RML2016.10a_seed2022.json", data_path=root / "data/RML2016.10a_dict.pkl")
    labels = np.zeros(220000, dtype=np.int64); snrs = np.zeros(220000, dtype=np.float32)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(path, predictions=np.zeros(len(split.test_idx), dtype=np.int64), labels=np.ones(len(split.test_idx), dtype=np.int64), snrs=snrs[split.test_idx], sample_ids=np.asarray(split.sample_ids)[split.test_idx])
    with pytest.raises(ValueError, match="labels"):
        module.validate_prediction_bundle(path, split, {"labels": labels, "snrs": snrs}, num_classes=11)


def test_default_trainer_applies_fixed_segment_weights_to_cross_entropy(tmp_path, monkeypatch):
    """Exercise the real trainer and inspect its autograd loss, not just the helper."""
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_phase2", root / "scripts/v2/run_phase2.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    split = type("Split", (), {"train_idx": np.array([0, 1, 2]), "val_idx": np.array([3, 4, 5]), "test_idx": np.array([6, 7, 8])})()
    dataset = {"signals": np.zeros((9, 2, 128), dtype=np.float32), "labels": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]), "snrs": np.array([-10, -4, 4, -10, -4, 4, -10, -4, 4], dtype=np.float32)}
    base = {"_active_seed": 7, "device": "cpu", "training": {"batch_size": 3, "lr": 0.001, "max_epochs": 1}, "evaluation_batch_size": 3, "architecture": {"num_classes": 11, "num_levels": 1, "in_channels": 64, "kernel_size": 3, "latent_dim": 320, "regu_details": 0.01, "regu_approx": 0.01, "num_snr_bins": 20, "snr_embedding_dim": 8}, "treatments": {"snr_reweighted": {"snr_reweighting": "fixed_segment_weights", "segment_weights": {"low": 2.0, "mid": 1.0, "high": 0.5}}}}
    observed_weights = []
    original_multiply = torch.Tensor.__mul__
    def capture_multiply(left, right):
        if isinstance(right, torch.Tensor) and left.ndim == right.ndim == 1 and len(left) == 3:
            candidate = right.detach().cpu().numpy()
            if np.array_equal(np.sort(candidate), np.array([0.5, 1.0, 2.0])):
                observed_weights.append(candidate.copy())
        return original_multiply(left, right)
    monkeypatch.setattr(torch.Tensor, "__mul__", capture_multiply)
    weighted_destination = tmp_path / "weighted"; weighted_destination.mkdir()
    module._default_trainer("snr_reweighted", dataset, split, weighted_destination, lambda _: None, config=base)
    assert len(observed_weights) == 1
    assert np.array_equal(np.sort(observed_weights[0]), np.array([0.5, 1.0, 2.0]))

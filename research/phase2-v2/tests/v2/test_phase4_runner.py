import csv
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from v2.provenance import deterministic_json_hash, file_sha256


MODELS = tuple(f"M{index}" for index in range(8))
SEEDS = (2022, 2023, 2024, 2025, 2026)
SNRS = tuple(range(-20, 20, 2))
PREDICTION_FIELDS = {
    "y_true", "y_pred", "logits", "snr_db", "snr_bin", "sample_ids",
    "seed", "model_id", "split_hash", "preprocessing_hash", "protocol_hash",
}


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/v2/run_phase4.py"
    spec = importlib.util.spec_from_file_location("run_phase4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_repository(tmp_path: Path, *, missing: tuple[str, int] | None = None):
    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "repo"
    phase3_root = root / "results/v2/phase3"
    phase3_root.mkdir(parents=True)
    for name in _load_module().PHASE4_SOURCE_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, target)

    phase3_config = yaml.safe_load((source_root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    phase3_config["outputs"] = {
        "root": "results/v2/phase3",
        "manifest": "manifest.json",
        "progress": "reports/v2_progress.md",
    }
    phase3_config_path = root / "configs/v2/phase3.yaml"
    phase3_config_path.parent.mkdir(parents=True)
    phase3_config_path.write_text(yaml.safe_dump(phase3_config, sort_keys=False), encoding="utf-8")
    protocol_hash = deterministic_json_hash({key: value for key, value in phase3_config.items() if key != "outputs"})
    preprocessing_hash = deterministic_json_hash(phase3_config["preprocessing"])
    split_hash = phase3_config["split"]["hash"]

    y_true = np.tile(np.arange(11, dtype=np.int64), len(SNRS))
    snr_db = np.repeat(np.asarray(SNRS, dtype=np.float32), 11)
    snr_bin = np.repeat(np.arange(len(SNRS), dtype=np.int64), 11)
    sample_ids = np.asarray([f"test-{index:04d}" for index in range(len(y_true))])
    ledger_records = []
    manifest_records = []
    runs = {}
    for seed in SEEDS:
        for model_index, model_id in enumerate(MODELS):
            if (model_id, seed) == missing:
                continue
            run_dir = phase3_root / f"{model_id}_seed{seed}_fixture"
            run_dir.mkdir()
            logits = np.full((len(y_true), 11), -3.0, dtype=np.float32)
            prediction = y_true.copy()
            if model_index == 0:
                low = snr_db <= -8
                prediction[low] = 0
            logits[np.arange(len(y_true)), prediction] = 3.0
            identity = {
                "seed": np.asarray(seed, dtype=np.int64),
                "model_id": np.asarray(model_id),
                "split_hash": np.asarray(split_hash),
                "preprocessing_hash": np.asarray(preprocessing_hash),
                "protocol_hash": np.asarray(protocol_hash),
            }
            np.savez_compressed(
                run_dir / "predictions.npz",
                y_true=y_true,
                y_pred=prediction,
                logits=logits,
                snr_db=snr_db,
                snr_bin=snr_bin,
                sample_ids=sample_ids,
                **identity,
            )
            (run_dir / "checkpoint.pt").write_bytes(f"checkpoint:{model_id}:{seed}".encode())
            np.savez_compressed(run_dir / "mechanism.npz", marker=np.asarray(1))
            fingerprint = deterministic_json_hash({"model": model_id, "seed": seed})
            run_spec = {
                "schema_version": 1,
                "runner_version": "phase3-conditioning/1",
                "created_at": "2026-09-05T00:00:00Z",
                "model_id": model_id,
                "seed": seed,
                "run_fingerprint": fingerprint,
                "protocol_hash": protocol_hash,
                "preprocessing_hash": preprocessing_hash,
                "protocol_config": phase3_config,
                "fixed_split": {"metadata": phase3_config["split"]["metadata"], "hash": split_hash},
                "execution_mode": "reused_phase2" if model_id in {"M0", "M3"} else "trained_phase3",
                "provenance": {"code": {"fixture": "code"}, "environment": {"fixture": "environment"}},
            }
            _write_json(run_dir / "run_spec.json", run_spec)
            overall_accuracy = float(np.mean(prediction == y_true))
            result = {
                "schema_version": 1,
                "runner_version": "phase3-conditioning/1",
                "status": "completed",
                "model_id": model_id,
                "seed": seed,
                "completed_at": "2026-09-05T00:00:01Z",
                "execution": {"mode": "reused" if model_id in {"M0", "M3"} else "trained", "reuse_source": None, "source_artifact_hashes": {}},
                "metrics": {"overall_accuracy": overall_accuracy, "balanced_accuracy": overall_accuracy},
                "complexity": {},
                "training": {},
                "artifacts": {
                    "run_spec": "run_spec.json",
                    "run_spec_sha256": file_sha256(run_dir / "run_spec.json"),
                    "checkpoint": "checkpoint.pt",
                    "checkpoint_sha256": file_sha256(run_dir / "checkpoint.pt"),
                    "predictions": "predictions.npz",
                    "predictions_sha256": file_sha256(run_dir / "predictions.npz"),
                    "mechanism": "mechanism.npz",
                    "mechanism_sha256": file_sha256(run_dir / "mechanism.npz"),
                },
                "provenance": {
                    "run_fingerprint": fingerprint,
                    "split_hash": split_hash,
                    "preprocessing_hash": preprocessing_hash,
                    "protocol_hash": protocol_hash,
                    "code": run_spec["provenance"]["code"],
                    "environment": run_spec["provenance"]["environment"],
                },
            }
            _write_json(run_dir / "result.json", result)
            relative = run_dir.relative_to(root).as_posix()
            ledger_records.append({
                "model_id": model_id,
                "seed": seed,
                "run_fingerprint": fingerprint,
                "attempt_path": relative,
                "result_sha256": file_sha256(run_dir / "result.json"),
                "status": "completed",
                "updated_at": "2026-09-05T00:00:02Z",
            })
            manifest_records.append({
                "experiment": "phase3",
                "dataset": "RML2016.10a",
                "split_hash": split_hash,
                "model": model_id,
                "conditioner": phase3_config["models"][model_id]["conditioner"],
                "seed": seed,
                "checkpoint": f"{relative}/checkpoint.pt",
                "result_json": f"{relative}/result.json",
                "figure_paths": [],
                "status": "completed",
                "notes": "fixture",
            })
            runs[(model_id, seed)] = run_dir

    _write_json(phase3_root / "phase3_ledger.json", {"schema_version": 1, "runner_version": "phase3-conditioning/1", "records": ledger_records})
    _write_json(root / "manifest.json", {"schema_version": 1, "experiments": manifest_records})

    phase4_config = yaml.safe_load((source_root / "configs/v2/phase4.yaml").read_text(encoding="utf-8"))
    config_path = root / "configs/v2/phase4.yaml"
    config_path.write_text(yaml.safe_dump(phase4_config, sort_keys=False), encoding="utf-8")
    return root, config_path, runs


def test_config_preregisters_multiple_thresholds_and_low_snr_definitions():
    module = _load_module()
    source_root = Path(__file__).resolve().parents[2]
    config = module.load_phase4_config(source_root / "configs/v2/phase4.yaml")

    assert len(config["threshold_sensitivity"]["concentration_cutoffs"]) >= 3
    assert len(config["threshold_sensitivity"]["ba_gap_cutoffs"]) >= 3
    assert len(config["threshold_sensitivity"]["snr_ranges"]) >= 3
    assert config["metrics"]["gap_definition"] == "prediction_concentration - balanced_accuracy"
    assert config["features"]["stronger_selection_metric"] == "best_val_accuracy"


def test_continuous_metrics_include_concentration_minus_ba_hhi_gini_and_15_bin_ece():
    module = _load_module()
    y_true = np.asarray([0, 1, 0, 1])
    logits = np.asarray([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])

    row = module.compute_continuous_metrics(y_true, logits, num_classes=2, ece_bins=15)

    assert row["prediction_concentration"] == pytest.approx(0.75)
    assert row["balanced_accuracy"] == pytest.approx(0.75)
    assert row["concentration_minus_balanced_accuracy"] == pytest.approx(0.0)
    assert row["normalized_entropy"] == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(2))
    assert row["hhi"] == pytest.approx(0.75**2 + 0.25**2)
    assert row["gini"] == pytest.approx(0.25)
    assert row["ece"] > 0.0


def test_legacy_pseudo_registry_is_blocked_by_phase3_strict_contract(tmp_path):
    module = _load_module()
    root, config_path, runs = _make_repository(tmp_path)
    config = module.load_phase4_config(config_path)

    with pytest.raises(module.UpstreamValidationError, match="strict Phase 3"):
        module.audit_phase3_registry(root, config)


def test_prediction_rejects_noninteger_snr_bin_dtype(tmp_path):
    module = _load_module()
    root, config_path, runs = _make_repository(tmp_path)
    config = module.load_phase4_config(config_path)
    phase3, split_hash, preprocessing_hash, protocol_hash = module._phase3_protocol(root, config)
    path = runs[("M0", 2022)] / "predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        values = {name: bundle[name] for name in bundle.files}
    values["snr_bin"] = values["snr_bin"].astype(np.float32)
    np.savez_compressed(path, **values)
    with pytest.raises(module.UpstreamValidationError, match="integer"):
        module._validate_prediction(
            path, model_id="M0", seed=2022, split_hash=split_hash,
            preprocessing_hash=preprocessing_hash, protocol_hash=protocol_hash,
        )


def test_stronger_model_uses_validation_only_and_records_all_candidates(tmp_path):
    module = _load_module()
    root, _config_path, runs = _make_repository(tmp_path)
    registered = []
    validation = {"M4": 0.70, "M5": 0.75, "M6": 0.80, "M7": 0.79}
    test_accuracy = {"M4": 0.99, "M5": 0.90, "M6": 0.10, "M7": 0.95}
    for model in ("M4", "M5", "M6", "M7"):
        for seed in SEEDS:
            registered.append(module.RegisteredRun(
                model_id=model, seed=seed, run_dir=runs[(model, seed)],
                result_path=runs[(model, seed)] / "result.json",
                predictions_path=runs[(model, seed)] / "predictions.npz",
                checkpoint_path=runs[(model, seed)] / "checkpoint.pt",
                result_sha256="a" * 64, predictions_sha256="b" * 64,
                checkpoint_sha256="c" * 64, run_fingerprint="d" * 64,
                split_hash="e" * 64, preprocessing_hash="f" * 64,
                protocol_hash="1" * 64, overall_accuracy=test_accuracy[model],
                best_val_accuracy=validation[model], additional_conditioner_parameters=0,
            ))
    selected, evidence = module._stronger_model(registered)
    assert selected == "M6"
    assert evidence["selection_partition"] == "validation_only"
    assert [row["model_id"] for row in evidence["candidates"]] == ["M4", "M5", "M6", "M7"]
    assert evidence["tie_break"] == ["mean_best_val_accuracy_desc", "mean_additional_conditioner_parameters_asc", "model_id_asc"]


def test_threshold_stability_treats_equal_to_non_equal_as_dependency():
    module = _load_module()
    source_root = Path(__file__).resolve().parents[2]
    config = module.load_phase4_config(source_root / "configs/v2/phase4.yaml")
    continuous = []
    for model in MODELS:
        concentration = 0.90 if model == "M0" else (0.35 if model == "M1" else 0.90)
        gap = 0.90 if model != "M1" else 0.15
        for snr in SNRS:
            continuous.append({
            "model_id": model, "seed": 2022, "snr_db": -8,
            "prediction_concentration": concentration,
            "concentration_minus_balanced_accuracy": gap,
            } | {"snr_db": snr})
    _rows, summary = module._threshold_rows(continuous, config)
    comparison = next(item for item in summary["comparisons"] if item["model_id"] == "M1" and item["snr_range"] == "registered_low_le_minus8")
    assert comparison["stable_direction"] is False
    assert comparison["directions"] == ["equal", "lower"]


def test_blocked_upstream_is_generation_atomic_and_does_not_publish_current(tmp_path):
    module = _load_module()
    root, config_path, _runs = _make_repository(tmp_path)
    logs = []
    assert module.run_phase4(config_path, repository_root=root, logger=logs.append) == 3
    output = root / "results/v2/phase4_collapse_metrics"
    attempts = list((output / "attempts").iterdir())
    assert len(attempts) == 1
    summary = json.loads((attempts[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "BLOCKED_UPSTREAM"
    assert not (output / "current.json").exists()
    assert not list(output.glob(".staging-*"))
    assert all("PROCEED_TO_PHASE_5" not in message for message in logs)


@pytest.mark.parametrize(("model_id", "expected_dimension"), [("M0", 128), ("M3", 136), ("M7", 128)])
def test_builtin_checkpoint_extractor_replays_registered_logits(tmp_path, model_id, expected_dimension):
    torch = pytest.importorskip("torch")
    from models.model_conditioning import AWNConditioned

    module = _load_module()
    source_root = Path(__file__).resolve().parents[2]
    phase3_config = yaml.safe_load((source_root / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    phase4_config = module.load_phase4_config(source_root / "configs/v2/phase4.yaml")
    architecture = phase3_config["architecture"]
    model = AWNConditioned(
        **{key: architecture[key] for key in (
            "num_classes", "num_levels", "in_channels", "kernel_size", "latent_dim",
            "regu_details", "regu_approx", "num_snr_bins", "snr_embedding_dim",
        )}, conditioning=model_id, snr_min_db=-20, snr_max_db=18,
    ).eval()
    generator = torch.Generator().manual_seed(41)
    signals = torch.randn(8, 2, 32, generator=generator)
    snr_db = torch.tensor([-20, -18, -16, -14, -12, -10, -8, -6], dtype=torch.float32)
    snr_bin = torch.arange(8, dtype=torch.long)
    with torch.no_grad():
        logits = model.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)[0].numpy().astype(np.float32)
    run_dir = tmp_path / f"{model_id}_seed2022"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    sample_ids = np.asarray([f"tiny-{index}" for index in range(8)])
    prediction = run_dir / "predictions.npz"
    np.savez_compressed(
        prediction, y_true=np.arange(8, dtype=np.int64) % 11, y_pred=logits.argmax(1), logits=logits,
        snr_db=snr_db.numpy(), snr_bin=snr_bin.numpy(), sample_ids=sample_ids,
        seed=np.asarray(2022), model_id=np.asarray(model_id), split_hash=np.asarray("a" * 64),
        preprocessing_hash=np.asarray("b" * 64), protocol_hash=np.asarray("c" * 64),
    )
    run = module.RegisteredRun(
        model_id=model_id, seed=2022, run_dir=run_dir, result_path=run_dir / "result.json",
        predictions_path=prediction, checkpoint_path=checkpoint, result_sha256="d" * 64,
        predictions_sha256=file_sha256(prediction), checkpoint_sha256=file_sha256(checkpoint),
        run_fingerprint="e" * 64, split_hash="a" * 64, preprocessing_hash="b" * 64,
        protocol_hash="c" * 64, overall_accuracy=0.0, best_val_accuracy=0.5,
        additional_conditioner_parameters=0,
    )
    split = SimpleNamespace(test_idx=np.arange(8), sample_ids=sample_ids)
    dataset = {"signals": signals.numpy(), "labels": np.arange(8) % 11, "snrs": snr_db.numpy()}
    source = module._source_identity(source_root)
    environment = {"fixture": "tiny-real-checkpoint"}
    arrays = module.extract_checkpoint_features(
        run, phase3_config=phase3_config, dataset=dataset, split=split, root=source_root,
        phase4_config=phase4_config, source=source, environment=environment, batch_size=3,
    )
    assert arrays["features"].shape == (8, expected_dimension)
    np.testing.assert_allclose(arrays["replayed_logits"], logits, rtol=1e-6, atol=1e-7)
    assert np.array_equal(arrays["replayed_logits"].argmax(1), logits.argmax(1))
    bundle_path = run_dir / "features.npz"
    np.savez_compressed(bundle_path, **arrays)
    assert module._validate_feature_bundle(
        bundle_path, run, phase4_config, source=source, environment=environment,
        expected_logits=logits,
    ).shape == (8, expected_dimension)


def test_public_runner_rejects_arbitrary_feature_callback(tmp_path):
    module = _load_module()
    with pytest.raises(TypeError, match="feature_extractor"):
        module.run_phase4("unused.yaml", repository_root=tmp_path, feature_extractor=lambda _run: np.ones((1, 128)))


def test_lock_recovers_dead_pid_but_refuses_live_owner(tmp_path):
    module = _load_module()
    lock_path = tmp_path / ".phase4.lock"
    lock_path.write_text(json.dumps({"pid": 2_147_483_647, "token": "dead"}), encoding="utf-8")
    with module._Phase4Lock(lock_path):
        assert lock_path.exists()
    lock_path.write_text(json.dumps({"pid": __import__("os").getpid(), "token": "live"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already locked"):
        with module._Phase4Lock(lock_path):
            pass


def test_distribution_metric_boundaries_and_missing_true_classes():
    module = _load_module()
    uniform_logits = np.asarray([[9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=float)
    uniform = module.compute_continuous_metrics(np.asarray([0, 1, 2]), uniform_logits, num_classes=3, ece_bins=15)
    assert uniform["prediction_concentration"] == pytest.approx(1 / 3)
    assert uniform["normalized_entropy"] == pytest.approx(1.0)
    assert uniform["hhi"] == pytest.approx(1 / 3)
    assert uniform["gini"] == pytest.approx(0.0)
    onehot = module.compute_continuous_metrics(np.asarray([0, 0]), np.asarray([[1000, 0, 0], [1000, 0, 0]]), num_classes=3, ece_bins=15)
    assert onehot["prediction_concentration"] == 1.0
    assert onehot["balanced_accuracy"] == 1.0
    assert onehot["normalized_entropy"] == pytest.approx(0.0)
    assert onehot["hhi"] == 1.0
    assert onehot["ece"] == pytest.approx(0.0)


def test_manifest_closure_rejects_unregistered_pollution(tmp_path):
    module = _load_module()
    source_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "attempt"
    output.mkdir()
    for name in ("summary.json", "mechanism_report.md", "continuous_metrics.csv", "threshold_sensitivity.csv", "feature_geometry.csv"):
        (output / name).write_text("{}\n", encoding="utf-8")
    dependency = "scripts/v2/run_phase4.py"
    module._write_manifest(
        output, "BLOCKED_UPSTREAM", fingerprint="a" * 64, root=source_root,
        dependencies={dependency: file_sha256(source_root / dependency)}, environment={"fixture": True},
        upstream={"completed_runs": 0, "required_runs": 40}, selection=None,
    )
    module.validate_phase4_manifest(output / "manifest.json", output, repository_root=source_root)
    (output / "pollution.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="polluted"):
        module.validate_phase4_manifest(output / "manifest.json", output, repository_root=source_root)

import hashlib
import importlib.util
import json
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


MODELS = [f"M{index}" for index in range(8)]
SEEDS = [2022, 2023, 2024, 2025, 2026]
SNRS = list(range(-20, 20, 2))
SPLIT_HASH = "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"
PROTOCOL_HASH = "5" * 64
PREPROCESSING_HASH = "6" * 64


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "v2/phase3_analysis.py"
    spec = importlib.util.spec_from_file_location("phase3_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(labels, predictions, snrs, classes=3):
    def group(mask):
        y = labels[mask]; pred = predictions[mask]
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (y, pred), 1)
        support = matrix.sum(1)
        recall = np.divide(np.diag(matrix), support, out=np.zeros(classes, dtype=float), where=support != 0)
        precision = np.divide(np.diag(matrix), matrix.sum(0), out=np.zeros(classes, dtype=float), where=matrix.sum(0) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
        counts = np.bincount(pred, minlength=classes); shares = counts / len(pred)
        nz = shares[shares > 0]
        entropy = -float(np.sum(nz * np.log(nz))) / np.log(classes)
        dominant = int(np.argmax(shares))
        return {
            "accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(np.mean(recall)), "prediction_concentration": float(shares.max()),
            "normalized_prediction_entropy": entropy, "gini": float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * classes)),
            "hhi": float(np.sum(shares ** 2)), "dominant_predicted_class": dominant,
            "dominant_predicted_class_ratio": float(shares[dominant]), "count": int(len(y)),
        }
    overall = group(np.ones(len(labels), dtype=bool))
    logits = np.full((len(labels), classes), -2.0, dtype=np.float32)
    logits[np.arange(len(labels)), predictions] = 2.0
    shifted = logits - logits.max(1, keepdims=True); probabilities = np.exp(shifted); probabilities /= probabilities.sum(1, keepdims=True)
    nll = float(np.mean(-np.log(probabilities[np.arange(len(labels)), labels])))
    confidence = probabilities.max(1); correct = predictions == labels; ece = 0.0
    boundaries = np.linspace(0, 1, 16)
    for index in range(15):
        mask = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1] if index < 14 else confidence <= 1)
        if mask.any(): ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    result = {
        "overall_accuracy": overall.pop("accuracy"), "macro_f1": overall.pop("macro_f1"),
        "balanced_accuracy": overall.pop("balanced_accuracy"), "nll": nll, "ece": ece,
        "ece_definition": {"type": "fixture"}, "per_snr": {}, "segments": {}, **overall,
    }
    for snr in SNRS:
        result["per_snr"][f"{float(snr):+.17g}"] = group(snrs == snr)
    for name, mask in {"low": snrs <= -8, "mid": (snrs >= -6) & (snrs <= -2), "high": snrs >= 0}.items():
        result["segments"][name] = group(mask)
    return result


def _mechanism(model, seed):
    scale = (seed - 2021) / 10
    if model == "M3":
        embedding = np.stack([np.array([snr / 20, scale, (snr / 20) ** 2]) for snr in SNRS]).astype(np.float32)
        return {"snr_embedding_vectors": embedding, "immediate_affine.condition_weight": np.ones((4, 3), np.float32), "per_snr_additive_contribution": embedding @ np.ones((3, 4), np.float32)}
    if model == "M4":
        return {"per_snr_logit_bias": np.stack([np.array([snr, -snr, scale]) for snr in SNRS]).astype(np.float32)}
    if model == "M5":
        z = np.asarray(SNRS, dtype=np.float32)[:, None]
        return {"per_snr_gamma": 1 + z * np.ones((20, 4), np.float32) / 100, "per_snr_beta": z * np.ones((20, 4), np.float32) / 50}
    if model == "M6":
        z = np.asarray(SNRS, dtype=np.float32)[:, None]
        return {"per_snr_gates": 1 + z * np.array([[0.01, -0.01, 0.005, -0.005]], np.float32)}
    if model == "M7":
        tensors = {}
        for index in range(20):
            tensors[f"snr_heads.{index}.0.weight"] = np.full((4, 3), index + scale, np.float32)
            tensors[f"snr_heads.{index}.0.bias"] = np.full(4, scale, np.float32)
            tensors[f"snr_heads.{index}.2.weight"] = np.full((3, 4), index - scale, np.float32)
            tensors[f"snr_heads.{index}.2.bias"] = np.full(3, index, np.float32)
        return tensors
    return {} if model == "M0" else {"immediate_affine.condition_weight": np.ones((4, 1 if model == "M1" else 20), np.float32)}


def _write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _legacy_phase3_fixture(tmp_path, *, missing=None):
    root = tmp_path / "repo"; source = root / "results/v2/phase3"; source.mkdir(parents=True)
    records = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(SEEDS):
            if missing == (model, seed):
                continue
            run = source / f"{model}_seed{seed}_fixture"; run.mkdir()
            sample_count = 20 * 6
            snrs = np.repeat(np.asarray(SNRS, dtype=np.float32), 6)
            labels = np.tile(np.arange(6) % 3, 20).astype(np.int64)
            correct = min(5, 2 + model_index // 2 + seed_index % 2)
            predictions = labels.copy()
            for snr_index in range(20):
                start = snr_index * 6
                predictions[start + correct:start + 6] = (labels[start + correct:start + 6] + 1) % 3
            logits = np.full((sample_count, 3), -2.0, np.float32)
            logits[np.arange(sample_count), predictions] = 2.0
            prediction_path = run / "predictions.npz"
            _write_npz(prediction_path, {
                "y_true": labels, "y_pred": predictions, "logits": logits, "snr_db": snrs,
                "snr_bin": np.repeat(np.arange(20), 6), "sample_ids": np.asarray([f"sample-{i}" for i in range(sample_count)]),
                "seed": np.asarray(seed), "model_id": np.asarray(model), "split_hash": np.asarray(SPLIT_HASH),
                "preprocessing_hash": np.asarray(PREPROCESSING_HASH), "protocol_hash": np.asarray(PROTOCOL_HASH),
            })
            checkpoint = run / "checkpoint.pt"; checkpoint.write_bytes(b"fixture checkpoint")
            checkpoint_hash = _sha256(checkpoint)
            mechanism = _mechanism(model, seed); names = sorted(mechanism)
            mechanism_arrays = {
                "model_id": np.asarray(model), "seed": np.asarray(seed), "split_hash": np.asarray(SPLIT_HASH),
                "preprocessing_hash": np.asarray(PREPROCESSING_HASH), "protocol_hash": np.asarray(PROTOCOL_HASH),
                "checkpoint_sha256": np.asarray(checkpoint_hash), "tensor_names": np.asarray(names),
            }
            mechanism_arrays.update({f"tensor_{index:03d}": mechanism[name] for index, name in enumerate(names)})
            mechanism_path = run / "mechanism.npz"; _write_npz(mechanism_path, mechanism_arrays)
            run_spec = run / "run_spec.json"; run_spec.write_text(json.dumps({"model_id": model, "seed": seed, "protocol_hash": PROTOCOL_HASH}), encoding="utf-8")
            metrics = _metrics(labels, predictions, snrs)
            result = {
                "status": "completed", "model_id": model, "seed": seed, "metrics": metrics,
                "complexity": {
                    "total_parameters": 1000 + 100 * model_index, "additional_conditioner_parameters": 100 * model_index,
                    "parameter_increase_percent": float(100 * model_index / 10),
                    "flops": {"available": False, "reason": "fixture"},
                    "latency": {"summary": {"basis": "mixed_bins.fixed_batch", "mean_ms": 1.0 + model_index / 10, "median_ms": 1.0}},
                    "memory": {"available": False, "reason": "fixture"},
                },
                "artifacts": {
                    "run_spec": "run_spec.json", "run_spec_sha256": _sha256(run_spec),
                    "checkpoint": "checkpoint.pt", "checkpoint_sha256": checkpoint_hash,
                    "predictions": "predictions.npz", "predictions_sha256": _sha256(prediction_path),
                    "mechanism": "mechanism.npz", "mechanism_sha256": _sha256(mechanism_path),
                },
                "provenance": {"split_hash": SPLIT_HASH, "protocol_hash": PROTOCOL_HASH, "preprocessing_hash": PREPROCESSING_HASH},
            }
            result_path = run / "result.json"; result_path.write_text(json.dumps(result), encoding="utf-8")
            records.append({
                "experiment": "phase3", "dataset": "RML2016.10a", "split_hash": SPLIT_HASH,
                "model": model, "conditioner": model, "seed": seed, "checkpoint": checkpoint.relative_to(root).as_posix(),
                "result_json": result_path.relative_to(root).as_posix(), "figure_paths": [], "status": "completed", "notes": "fixture",
            })
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "experiments": records}), encoding="utf-8")
    return root, manifest


def _load_phase3_test_helpers():
    root = Path(__file__).resolve().parents[2]
    path = root / "tests/v2/test_phase3_runner.py"
    spec = importlib.util.spec_from_file_location("phase3_runner_fixture_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase3_fixture(tmp_path, analysis_module, monkeypatch, *, missing=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    helpers = _load_phase3_test_helpers()
    phase3, root, config_path, dataset = helpers._fixture(tmp_path, monkeypatch)
    original_mechanism = helpers._fake_mechanism

    def scientifically_valid_mechanism(model_id):
        value = original_mechanism(model_id)
        if model_id == "M6":
            value["per_snr_gates"] = np.ones((20, 128), dtype=np.float32)
        return value

    helpers._fake_mechanism = scientifically_valid_mechanism

    trained = lambda model_id, data, split, destination, _logger, *, config: helpers._outcome(model_id, data, split, destination)

    def reused(model_id, seed, data, split, destination, _logger, *, config):
        outcome = helpers._outcome(model_id, data, split, destination)
        outcome["reuse_source"] = f"results/v2/phase2/{model_id}_seed{seed}"
        outcome["source_artifact_hashes"] = {
            "run_spec_sha256": "c" * 64, "result_sha256": "d" * 64,
            "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64,
        }
        return outcome

    assert phase3.run_phase3(
        config_path, repository_root=root, trainer=trained, phase2_reuser=reused,
        dataset_loader=lambda *_args: dataset, logger=lambda _line: None, smoke=True,
        completed_verifier=helpers._resume_verifier,
        phase2_source_validator=helpers._source_validator,
    ) == 0
    split = phase3.load_split("unused")
    monkeypatch.setattr(analysis_module, "_load_fixed_data", lambda *_args, **_kwargs: (dataset, split))
    monkeypatch.setattr(analysis_module, "_COMPLETED_VERIFIER_OVERRIDE", helpers._resume_verifier)
    monkeypatch.setattr(analysis_module, "_PHASE2_SOURCE_VALIDATOR_OVERRIDE", helpers._source_validator)
    output = root / "results/v2/phase3/test"
    if missing is not None:
        model, seed = missing
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiments"] = [record for record in manifest["experiments"] if (record["model"], record["seed"]) != missing]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        ledger_path = output / "phase3_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["records"] = [record for record in ledger["records"] if (record["model_id"], record["seed"]) != missing]
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return root, output / "manifest.json"


def test_dry_run_requires_exact_complete_registered_matrix_and_writes_nothing(tmp_path, monkeypatch):
    module = _load_module()
    root, manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    report = module.analyze_phase3(repository_root=root, source_manifest=manifest, dry_run=True)
    assert report == {"ready": True, "completed_count": 40, "missing": [], "errors": [], "would_publish": "results/v2/phase3_conditioning"}
    assert not (root / "results/v2/phase3_conditioning").exists()

    incomplete_root, incomplete_manifest = _phase3_fixture(tmp_path / "incomplete", module, monkeypatch, missing=("M7", 2026))
    incomplete = module.analyze_phase3(repository_root=incomplete_root, source_manifest=incomplete_manifest, dry_run=True)
    assert incomplete["ready"] is False and incomplete["completed_count"] == 39
    assert incomplete["missing"] == [{"model_id": "M7", "seed": 2026}]
    with pytest.raises(module.AnalysisNotReadyError, match="39/40"):
        module.analyze_phase3(repository_root=incomplete_root, source_manifest=incomplete_manifest)
    assert not (incomplete_root / "results/v2/phase3_conditioning").exists()


def test_legacy_three_class_fixture_without_ledger_or_canonical_run_specs_is_rejected(tmp_path):
    module = _load_module()
    root, manifest = _legacy_phase3_fixture(tmp_path)

    report = module.analyze_phase3(repository_root=root, source_manifest=manifest, dry_run=True)

    assert report["ready"] is False
    assert any("ledger" in error.lower() for error in report["errors"])


def test_default_source_manifest_matches_phase3_configured_registry_location(tmp_path, monkeypatch):
    module = _load_module()
    root, nested_manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    configured_manifest = root / "manifest.json"
    nested_manifest.replace(configured_manifest)
    configured_ledger = root / "results/v2/phase3/phase3_ledger.json"
    configured_ledger.write_bytes((root / "results/v2/phase3/test/phase3_ledger.json").read_bytes())
    report = module.analyze_phase3(repository_root=root, dry_run=True)
    assert report["ready"] is True and report["completed_count"] == 40


def test_shared_registry_ignores_records_from_other_phases(tmp_path, monkeypatch):
    module = _load_module()
    root, manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    registry = json.loads(manifest.read_text(encoding="utf-8"))
    registry["experiments"].insert(0, {
        "experiment": "phase2", "dataset": "RML2016.10a", "split_hash": SPLIT_HASH, "model": "plain",
        "conditioner": "none", "seed": 2022, "checkpoint": "results/v2/phase2/checkpoint.pt",
        "result_json": "results/v2/phase2/result.json", "figure_paths": [], "status": "completed", "notes": "other phase",
    })
    manifest.write_text(json.dumps(registry), encoding="utf-8")
    assert module.analyze_phase3(repository_root=root, source_manifest=manifest, dry_run=True)["ready"] is True


@pytest.mark.filterwarnings("error")
def test_complete_analysis_derives_matched_seed_tables_and_all_mechanism_geometry(tmp_path, monkeypatch):
    module = _load_module()
    root, manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    source_hash_before = _sha256(manifest)
    outcome = module.analyze_phase3(repository_root=root, source_manifest=manifest)
    assert outcome["status"] == "published" and len(outcome["analysis_fingerprint"]) == 64
    output = root / "results/v2/phase3_conditioning"
    required = {
        "config.json", "conditioning_summary.json", "conditioning_summary.csv", "paired_statistics.json",
        "per_snr_metrics.csv", "complexity.csv", "embedding_geometry.npz", "conditional_bias.npz",
        "film_parameters.npz", "gating_parameters.npz", "per_bin_head_geometry.npz", "mechanism_report.md", "manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert _sha256(manifest) == source_hash_before

    paired = json.loads((output / "paired_statistics.json").read_text(encoding="utf-8"))
    assert len(paired["comparisons"]) == 10
    comparison = paired["comparisons"]["M3_minus_M0"]["overall_accuracy"]
    assert comparison["seeds"] == SEEDS and comparison["n_pairs"] == 5
    assert len(comparison["differences"]) == 5
    assert {"baseline_mean", "baseline_std", "treatment_mean", "treatment_std", "difference_mean", "difference_std", "ci95_low", "ci95_high", "cohen_dz", "p_value_secondary", "seed_direction"} <= set(comparison)
    differences = np.asarray(comparison["differences"])
    assert comparison["seed_direction"] == {
        "positive": int(np.sum(differences > 0)),
        "zero": int(np.sum(differences == 0)),
        "negative": int(np.sum(differences < 0)),
    }
    assert set(paired["comparisons"]["M7_minus_M3"]) == {"overall_accuracy", "macro_f1", "balanced_accuracy"}

    summary = json.loads((output / "conditioning_summary.json").read_text(encoding="utf-8"))
    assert set(summary["models"]) == set(MODELS)
    assert summary["low_snr_definition"] == {"preregistered": True, "max_db": -8, "values_db": [-20, -18, -16, -14, -12, -10, -8]}
    assert summary["interpolation_diagnostic"] == {
        "status": "unsupported", "reason": "all discrete 2 dB SNR bins are observed during training; no strict unseen or intermediate condition exists in this dataset protocol", "scores": None,
    }
    with (output / "conditioning_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 8 * 5
    with (output / "per_snr_metrics.csv").open(newline="", encoding="utf-8") as handle:
        snr_rows = list(csv.DictReader(handle))
    assert len(snr_rows) == 8 * 20 * 8
    assert {row["metric"] for row in snr_rows} == {"accuracy", "macro_f1", "balanced_accuracy", "prediction_concentration", "normalized_prediction_entropy", "gini", "hhi", "dominant_predicted_class_ratio"}
    with (output / "complexity.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 40

    geometry_fields = {
        "embedding_geometry.npz": {"seeds", "snr_db", "embedding_vectors", "adjacent_euclidean_distance", "pairwise_euclidean_distance", "cosine_similarity", "embedding_norm", "pca_scores"},
        "conditional_bias.npz": {"seeds", "snr_db", "bias_vectors", "bias_magnitude", "pairwise_euclidean_distance", "class_shift_from_snr_mean"},
        "film_parameters.npz": {"seeds", "snr_db", "gamma", "beta", "gamma_mean", "gamma_std", "beta_mean", "beta_std", "adjacent_gamma_change", "adjacent_beta_change"},
        "gating_parameters.npz": {"seeds", "snr_db", "gates", "gate_sparsity", "normalized_gate_entropy", "adjacent_gate_change"},
        "per_bin_head_geometry.npz": {"seeds", "snr_db", "flattened_head_parameters", "cosine_similarity", "frobenius_distance", "classifier_weight_norm", "boundary_parameter_drift"},
    }
    for filename, fields in geometry_fields.items():
        with np.load(output / filename, allow_pickle=False) as bundle:
            assert set(bundle.files) == fields | {"analysis_fingerprint", "model_id", "source_result_sha256"}
            assert bundle["analysis_fingerprint"].item() == outcome["analysis_fingerprint"]
            assert bundle["source_result_sha256"].shape == (5,)
            for field in fields - {"seeds", "snr_db"}:
                assert bundle[field].size and np.isfinite(bundle[field]).all()

    report = (output / "mechanism_report.md").read_text(encoding="utf-8")
    assert "W_f f(x) + W_e e(z) + b" in report
    assert "feature × condition interaction" in report
    assert "不具备严格 unseen/intermediate experimental support" in report
    assert "LeakyReLU" in report
    assert "Best overall conditioner:" in report
    assert "Claim allowed:" in report and "Claim forbidden:" in report
    assert "does not establish equivalence" in report


def test_publication_figures_are_machine_derived_and_existing_output_is_strictly_reused(tmp_path, monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    module = _load_module()
    root, source_manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    original_source = source_manifest.read_bytes()
    first = module.analyze_phase3(repository_root=root, source_manifest=source_manifest)
    output = root / "results/v2/phase3_conditioning"
    figure_names = {
        "overall_conditioner_comparison", "per_snr_accuracy", "per_snr_balanced_accuracy",
        "per_snr_concentration", "per_snr_entropy", "paired_improvement_forest",
        "per_snr_dominant_class_ratio", "delta_accuracy_vs_snr", "delta_balanced_accuracy_vs_snr",
        "delta_concentration_vs_snr", "accuracy_vs_complexity", "conditioner_mechanisms",
    }
    figures = output / "figures"
    assert {path.stem for path in figures.glob("*.png")} == figure_names
    assert {path.stem for path in figures.glob("*.pdf")} == figure_names
    for path in figures.glob("*.png"):
        with image_module.open(path) as image:
            assert image.width >= 1000 and image.info["dpi"][0] >= 299

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["analysis_fingerprint"] == first["analysis_fingerprint"]
    assert len(manifest["source_artifacts"]) == 40
    assert all(not Path(item["result_json"]).is_absolute() and len(item["result_sha256"]) == 64 for item in manifest["source_artifacts"])
    assert all((output / name).is_file() and _sha256(output / name) == digest for name, digest in manifest["files"].items())
    assert module.analyze_phase3(repository_root=root, source_manifest=source_manifest)["status"] == "reused"

    unexpected = output / "unexpected.txt"; unexpected.write_text("not registered", encoding="utf-8")
    with pytest.raises(module.AnalysisConflictError, match="unregistered"):
        module.analyze_phase3(repository_root=root, source_manifest=source_manifest)
    unexpected.unlink()

    changed_registry = json.loads(original_source)
    changed_registry["experiments"][0]["notes"] += " changed fixture"
    source_manifest.write_text(json.dumps(changed_registry), encoding="utf-8")
    assert _sha256(source_manifest) != hashlib.sha256(original_source).hexdigest()
    with pytest.raises(module.AnalysisConflictError, match="fingerprint"):
        module.analyze_phase3(repository_root=root, source_manifest=source_manifest)
    source_manifest.write_bytes(original_source)
    (output / "conditioning_summary.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(module.AnalysisConflictError, match="hash"):
        module.analyze_phase3(repository_root=root, source_manifest=source_manifest)
    assert not list((output.parent).glob(".phase3_conditioning.staging.*"))


def test_cli_dry_run_reports_readiness_and_incompleteness_without_publication(tmp_path, monkeypatch, capsys):
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts/v2/analyze_phase3.py"
    spec = importlib.util.spec_from_file_location("phase3_analysis_cli", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # The synthetic fixture's data and checkpoint verifiers are process-local.
    # Exercise the real parser and analyzer together with those fixture adapters.
    module = sys.modules[cli.analyze_phase3.__module__]
    root, manifest = _phase3_fixture(tmp_path / "complete", module, monkeypatch)
    assert cli.main(["--repository-root", str(root), "--source-manifest", str(manifest), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True
    assert not (root / "results/v2/phase3_conditioning").exists()

    incomplete_root, incomplete_manifest = _phase3_fixture(tmp_path / "incomplete", module, monkeypatch, missing=("M2", 2024))
    assert cli.main(["--repository-root", str(incomplete_root), "--source-manifest", str(incomplete_manifest), "--dry-run"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False and report["missing"] == [{"model_id": "M2", "seed": 2024}]
    assert not (incomplete_root / "results/v2/phase3_conditioning").exists()


def test_cli_subprocess_rejects_synthetic_registry_without_fixture_adapters(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts/v2/analyze_phase3.py"
    root, manifest = _phase3_fixture(tmp_path, _load_module(), monkeypatch)
    completed = subprocess.run(
        [sys.executable, str(script), "--repository-root", str(root), "--source-manifest", str(manifest), "--dry-run"],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ready"] is False and report["errors"]
    assert any("RML2016.10a_seed2022.json" in error for error in report["errors"])
    assert not (root / "results/v2/phase3_conditioning").exists()


@pytest.mark.parametrize("mutation", ["nll", "mechanism_identity", "mechanism_shape", "protocol_drift", "complexity"])
def test_dry_run_rejects_self_consistent_but_scientifically_invalid_registered_artifacts(tmp_path, monkeypatch, mutation):
    module = _load_module()
    root, manifest = _phase3_fixture(tmp_path, module, monkeypatch)
    result_path = next((root / "results/v2/phase3/test").glob("M5_seed2022_*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "nll":
        result["metrics"]["nll"] += 0.2
    elif mutation in {"mechanism_identity", "mechanism_shape"}:
        mechanism_path = result_path.parent / "mechanism.npz"
        with np.load(mechanism_path, allow_pickle=False) as bundle:
            arrays = {field: bundle[field] for field in bundle.files}
        if mutation == "mechanism_identity": arrays["model_id"] = np.asarray("M6")
        else: arrays["tensor_001"] = arrays["tensor_001"][:-1]
        np.savez_compressed(mechanism_path, **arrays)
        result["artifacts"]["mechanism_sha256"] = _sha256(mechanism_path)
    elif mutation == "protocol_drift":
        prediction_path = result_path.parent / "predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as bundle:
            arrays = {field: bundle[field] for field in bundle.files}
        arrays["protocol_hash"] = np.asarray("9" * 64)
        np.savez_compressed(prediction_path, **arrays)
        result["artifacts"]["predictions_sha256"] = _sha256(prediction_path)
        result["provenance"]["protocol_hash"] = "9" * 64
    else:
        result["complexity"]["total_parameters"] = float("nan")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    # Close the independent ledger over the mutated result so this test reaches
    # scientific validation rather than stopping at the outer integrity check.
    ledger_path = manifest.parent / "phase3_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    matching = [record for record in ledger["records"] if record["attempt_path"] == result_path.parent.relative_to(root).as_posix()]
    assert len(matching) == 1
    matching[0]["result_sha256"] = _sha256(result_path)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    report = module.analyze_phase3(repository_root=root, source_manifest=manifest, dry_run=True)
    assert report["ready"] is False
    expected_error = {
        "nll": "recomputable", "mechanism_identity": "mechanism",
        "mechanism_shape": "mechanism", "protocol_drift": "provenance is invalid",
        "complexity": "parameter counts must be positive integer",
    }[mutation]
    assert expected_error in " ".join(report["errors"]).lower(), report["errors"]

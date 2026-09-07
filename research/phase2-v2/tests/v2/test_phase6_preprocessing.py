import json
from pathlib import Path

import numpy as np
import pytest

from v2.phase6.audit import fit_transform, transform, frame_features, verify_artifacts


def test_global_statistics_use_only_training_frames():
    x = np.arange(32, dtype=float).reshape(4, 2, 4)
    state = fit_transform(x, np.array([0, 1]), "train_global")
    changed = x.copy(); changed[2:] = 1e9
    other = fit_transform(changed, np.array([0, 1]), "train_global")
    np.testing.assert_array_equal(state["mean"], other["mean"])
    np.testing.assert_array_equal(state["scale"], other["scale"])
    np.testing.assert_allclose(transform(x[:2], state).mean(axis=(0, 2)), 0, atol=1e-7)


@pytest.mark.parametrize("mode", ["legacy_identity", "train_global", "per_frame_rms"])
def test_streaming_is_equal_to_batch_and_future_invariant(mode):
    x = np.random.default_rng(4).normal(size=(6, 2, 16))
    state = fit_transform(x, np.array([0, 1, 2]), mode)
    batch = transform(x[3:], state)
    streamed = np.concatenate([transform(frame[None], state) for frame in x[3:]])
    np.testing.assert_array_equal(batch, streamed)
    future_changed = x[3:].copy(); future_changed[1:] *= 100
    np.testing.assert_array_equal(transform(future_changed, state)[0], batch[0])
    np.testing.assert_allclose(frame_features(batch), np.concatenate([frame_features(frame[None]) for frame in batch]))


def test_per_frame_rms_removes_positive_global_gain_and_zero_is_finite():
    x = np.random.default_rng(4).normal(size=(5, 2, 16))
    state = fit_transform(x, np.array([0, 1]), "per_frame_rms")
    np.testing.assert_allclose(transform(x, state), transform(13*x, state), atol=2e-7)
    assert np.isfinite(frame_features(transform(np.zeros_like(x), state))).all()


def test_manifest_verifier_rejects_tampering(tmp_path):
    (tmp_path / "one.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"one.json": "0"*64}, "dependencies": {}}))
    with pytest.raises(ValueError, match="hash"):
        verify_artifacts(tmp_path, tmp_path)


def test_probe_fit_and_selection_never_read_test_targets_or_features():
    from v2.phase6.experiment import select_probe
    rng = np.random.default_rng(22)
    x = rng.normal(size=(40, 3)); y = x[:, 0] * 2 + 1
    config = {"ridge_alphas": [0.1, 1.0], "logistic_cs": [0.1, 1.0], "logistic_max_iter": 1000, "logistic_tol": 1e-6, "deterministic_seed": 2022}
    model, scaler, evidence = select_probe(x[:20], y[:20], x[20:30], y[20:30], "ridge", config)
    mutated = x.copy(); mutated[30:] = 1e12
    other, other_scaler, other_evidence = select_probe(mutated[:20], y[:20], mutated[20:30], y[20:30], "ridge", config)
    np.testing.assert_array_equal(model.coef_, other.coef_)
    np.testing.assert_array_equal(scaler.mean_, x[:20].mean(axis=0))
    assert evidence == other_evidence

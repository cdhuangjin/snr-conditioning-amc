"""Shared loaders reused by the learned-SNR-estimator pipeline.

Everything that touches the locked split, the canonical dataset, the Phase 3
checkpoints, or the Phase 6 Ridge estimator is delegated to the existing
validated loaders so this supplement cannot drift from the published protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SEEDS = [2022, 2023, 2024, 2025, 2026]
CLIP_MIN_DB = -20.0
CLIP_MAX_DB = 18.0
SNR_VALUES_DB = list(range(-20, 20, 2))


def _project(root: Path) -> Any:
    import sys

    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_project(root: str | Path) -> tuple[Any, Any, Any, dict[str, Any], Any]:
    """Return ``(phase3, p3config, runs, dataset, split)`` from the canonical loaders."""
    root = _project(Path(root))
    from v2.phase3_analysis import _load_fixed_data, _load_phase3_contracts, _load_registered_runs

    phase3 = _load_phase3_contracts()
    p3config = phase3.load_phase3_config(root / "configs/v2/phase3.yaml")
    runs, missing, errors = _load_registered_runs(root, (root / "manifest.json").resolve())
    if missing or errors:
        raise ValueError(f"Phase 3 registered runs invalid: missing={missing} errors={errors}")
    dataset, split = _load_fixed_data(root, p3config, phase3)
    return phase3, p3config, runs, dataset, split


def load_ridge_estimator(root: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the canonical Phase 6 Ridge estimate ``raw`` for every frame.

    ``raw`` is indexed exactly like ``dataset['signals']`` (all 220k frames),
    so ``raw[split.test_idx]`` are the test Ridge estimates.
    """
    root = _project(Path(root))
    from v2.phase8.evidence import check_gates, load_estimator
    import yaml

    config = yaml.safe_load((root / "configs/v2/phase8.yaml").read_text(encoding="utf-8"))
    phase6, _gates = check_gates(root, config)
    _phase3, _p3config, _runs, dataset, split = load_project(root)
    return load_estimator(phase6, dataset, split)


def load_frozen_model(
    root: str | Path,
    p3config: dict[str, Any],
    runs: dict[tuple[str, int], dict[str, Any]],
    model_id: str,
    seed: int,
    device: str,
) -> torch.nn.Module:
    """Load a frozen Phase 3 ``model_id`` checkpoint for one seed into ``eval`` mode."""
    root = _project(Path(root))
    from v2.phase8.runner import get_model

    return get_model(_load_phase3(root), p3config, runs[(model_id, seed)], model_id, torch.device(device))


def _load_phase3(root: Path) -> Any:
    from v2.phase3_analysis import _load_phase3_contracts

    return _load_phase3_contracts()


def estimator_loss(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Point metrics for a single estimator output against ground-truth SNR (dB)."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("invalid estimator/label arrays")
    error = pred - truth
    return {
        "mae_db": float(np.mean(np.abs(error))),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "bias_db": float(np.mean(error)),
    }


def pearson_r(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.std(ddof=1) == 0 or truth.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(pred, truth)[0, 1])


def clip_and_bins(estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip to ``[-20,18]`` and map to the nearest even bin (Phase 8 semantics)."""
    from v2.phase8.core import conditions

    db, bins = conditions(estimates)
    return db, bins


def batch_predict(model: torch.nn.Module, signals: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    """Run the TinyCNN estimator over an ``(N,2,128)`` array on a single device."""
    model.eval()
    device = torch.device(device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            x = torch.as_tensor(signals[start : start + batch_size], device=device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def pooled_features(model: torch.nn.Module, signals: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """Extract attended pooled features (before conditioning) for frozen replay."""
    device = torch.device(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            x = torch.as_tensor(signals[start : start + batch_size], device=device)
            out.append(model.extract_pooled_features(x)[0].cpu().numpy())
    return np.concatenate(out)

"""Pure transformations and independently checkable preprocessing artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODES = ("legacy_identity", "train_global", "per_frame_rms")
FEATURE_NAMES = ("energy", "l2_norm", "variance_i", "variance_q", "centered_complex_variance", "peak_to_average", "log_energy")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _signals(x):
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != 2 or value.shape[2] == 0 or not np.isfinite(value).all():
        raise ValueError("signals require finite [frames,2,time] input")
    return value


def fit_transform(x, train_indices, mode, epsilon=1e-12):
    if mode not in MODES:
        raise ValueError("unknown preprocessing mode")
    ids = np.asarray(train_indices)
    if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(x)):
        raise ValueError("fit requires unique valid train indices")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive finite")
    # Never reduce over x before selecting training rows.
    selected = _signals(np.asarray(x)[ids])
    state = {"mode": mode, "epsilon": float(epsilon), "fit_partition": "train", "fit_indices_sha256": array_hash(ids), "fit_count": len(ids), "mean": [0., 0.], "scale": [1., 1.], "axes": ["frame", "time"], "fit_dtype": "float64", "output_dtype": "float32"}
    if mode == "train_global":
        state["mean"] = selected.mean(axis=(0, 2)).tolist()
        state["scale"] = np.maximum(selected.std(axis=(0, 2), ddof=0), epsilon).tolist()
    return state


def transform(x, state):
    value = _signals(x)
    if state["mode"] == "train_global":
        value = (value - np.asarray(state["mean"])[None, :, None]) / np.asarray(state["scale"])[None, :, None]
    elif state["mode"] == "per_frame_rms":
        rms = np.sqrt(np.mean(np.sum(value ** 2, axis=1), axis=1))
        value = value / np.maximum(rms, state["epsilon"])[:, None, None]
    elif state["mode"] != "legacy_identity":
        raise ValueError("unknown preprocessing mode")
    return value.astype(np.float32)


def frame_features(x, epsilon=1e-12):
    value = _signals(x)
    power = (value ** 2).sum(axis=1)
    energy = power.mean(axis=1)
    variance = value.var(axis=2, ddof=0)
    return np.column_stack([energy, np.sqrt(power.sum(axis=1)), variance[:, 0], variance[:, 1], variance.sum(axis=1), power.max(axis=1) / np.maximum(energy, epsilon), np.log(np.maximum(energy, epsilon))])


def verify_artifacts(output, root):
    output, root = Path(output).resolve(), Path(root).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("artifact hash closure incomplete or polluted")
    for name, expected in manifest["dependencies"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"dependency hash mismatch: {name}")
    return manifest

"""Train the TinyCNN-SNR estimator on the locked train/validation split."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from models.tinycnn_snr import TinyCNN_SNR
from .shared import estimator_loss, load_project, pearson_r


def _loader_predict(model, signals, indices, device, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            x = torch.as_tensor(signals[idx], device=device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def train_seed(
    root,
    out_root,
    seed,
    *,
    project=None,
    device="cuda",
    batch_size=512,
    lr=1e-3,
    max_epochs=50,
    patience=5,
    workers=0,
    smoke=False,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if project is None:
        phase3, p3config, runs, dataset, split = load_project(root)
    else:
        phase3, p3config, runs, dataset, split = project
    signals = dataset["signals"]
    snr = dataset["snrs"].astype(np.float32)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    val_idx = np.asarray(split.val_idx, dtype=np.int64)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = TinyCNN_SNR().to(device)
    train_signals = signals[train_idx]
    ch_mean = train_signals.mean(axis=(0, 2))
    ch_scale = np.maximum(train_signals.std(axis=(0, 2), ddof=0), 1e-6)
    model.set_input_stats(ch_mean, ch_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.0)

    if smoke:
        max_epochs = min(max_epochs, 2)
        train_idx = train_idx[: min(8192, len(train_idx))]

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    waited = 0
    epochs = []
    start_wall = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        train_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            x = torch.as_tensor(signals[idx], device=device)
            y = torch.as_tensor(snr[idx], device=device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(idx)
            seen += len(idx)
        train_loss /= max(seen, 1)

        val_pred = _loader_predict(model, signals, val_idx, device, 512)
        val_mae = float(np.mean(np.abs(val_pred - snr[val_idx])))
        epochs.append({"epoch": epoch, "train_huber": train_loss, "val_mae": val_mae})
        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
        print(f"[seed {seed}] epoch {epoch}/{max_epochs} train_huber={train_loss:.4f} val_mae={val_mae:.4f} best={best_val_mae:.4f}", flush=True)
        if waited >= patience:
            break

    if best_state is None:
        raise RuntimeError("no improving validation epoch for estimator training")
    model.load_state_dict(best_state)

    def pred(indices):
        return _loader_predict(model, signals, indices, device, 512)

    pred_all = np.zeros(len(signals), dtype=np.float32)
    for partition, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        pred_all[indices] = pred(indices)

    checkpoint = {
        "seed": seed,
        "state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
        "architecture": "TinyCNN_SNR",
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "epochs_completed": len(epochs),
        "parameter_count": TinyCNN_SNR.parameter_count(),
        "input_mean": ch_mean.tolist(),
        "input_scale": ch_scale.tolist(),
        "training": {
            "optimizer": "adam",
            "lr": lr,
            "batch_size": batch_size,
            "loss": "huber_delta_1",
            "max_epochs": max_epochs,
            "patience": patience,
            "smoke": smoke,
        },
    }
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_dir / f"tinycnn_snr_seed{seed}.pt")

    raw = out_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw / f"estimator_predictions_seed{seed}.npz",
        seed=np.asarray(seed),
        sample_ids=np.asarray(split.sample_ids),
        true_snr_db=snr,
        predicted_snr_db=pred_all,
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
    )

    result = {
        "seed": seed,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "epochs_completed": len(epochs),
        "duration_seconds": time.time() - start_wall,
        "train_huber_final": epochs[-1]["train_huber"] if epochs else None,
        "parameter_count": TinyCNN_SNR.parameter_count(),
        "test": estimator_loss(pred_all[test_idx], snr[test_idx]) | {"pearson_r": pearson_r(pred_all[test_idx], snr[test_idx])},
        "validation": estimator_loss(pred_all[val_idx], snr[val_idx]),
        "epochs": epochs,
    }
    (out_root / "raw" / f"train_result_seed{seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result

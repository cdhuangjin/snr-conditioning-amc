"""Minimal CNN regression from a single RML2016.10a I/Q frame to SNR in dB.

This is an estimator-strength control, not an SNR-estimation SOTA effort.  The
architecture is deliberately small (well under 100k parameters) and uses only
BatchNorm over channels so any input normalisation is train-derived and can
never see test statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN_SNR(nn.Module):
    """TinyCNN-SNR: ``[2,128]`` frame -> predicted SNR (dB)."""

    def __init__(self) -> None:
        super().__init__()
        # Train-derived per-channel normalisation buffers.  Before training the
        # caller must call ``set_input_stats`` with train-only statistics; until
        # then these default to identity so the module is safe standalone.
        self.register_buffer("input_mean", torch.zeros(2))
        self.register_buffer("input_scale", torch.ones(2))
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def set_input_stats(self, mean, scale, epsilon: float = 1e-6) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if mean.shape != (2,) or scale.shape != (2,):
            raise ValueError("input statistics must be per-channel length 2")
        if ((scale < epsilon) | ~torch.isfinite(scale)).any():
            raise ValueError("input scale must be finite and positive")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward a ``(batch, 2, 128)`` frame tensor to ``(batch,)`` SNR (dB)."""
        mean = self.input_mean.view(1, 2, 1)
        scale = self.input_scale.view(1, 2, 1)
        x = (x - mean) / scale
        return self.net(x).squeeze(-1)

    @staticmethod
    def parameter_count() -> int:
        return sum(p.numel() for p in TinyCNN_SNR().parameters())

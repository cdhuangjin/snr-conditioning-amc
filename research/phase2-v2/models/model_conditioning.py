"""Unified, inspectable SNR-conditioning controls for the Phase 3 AWN study."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import AWN


CONDITIONING_IDS = tuple(f"M{index}" for index in range(8))
_SCALAR_CONDITIONERS = frozenset({"M1", "M5", "M6"})
_DISCRETE_CONDITIONERS = frozenset({"M2", "M3", "M4", "M7"})


class AWNConditioned(AWN):
    """AWN backbone with one of eight controlled conditioning mechanisms.

    All mechanisms use the same convolution, lifting, pooling, and SE-attention
    backbone as :class:`models.model.AWN`.  M1--M3 concatenate their condition
    at the attended pooled-feature location.  At the immediate affine layer,
    embedding+linear concatenation provides z-dependent additive terms; it is
    not arbitrary per-bin feature normals.  M4 adds only class-logit
    bias, M5 applies feature-wise FiLM, M6 applies a feature-wise gate, and M7
    is the explicit capacity upper control with one complete classifier head
    per SNR bin over a shared feature extractor.

    M3 deliberately retains the Phase 2 ``AWNSNR`` names ``snr_embedding`` and
    ``fc`` and their tensor layout.  Strict state-dict loading requires matching
    ``num_classes``, ``num_levels``, ``in_channels``, ``kernel_size``,
    ``latent_dim``, ``num_snr_bins``, and ``snr_embedding_dim``; matching
    regularization coefficients are additionally required for identical loss
    semantics.  ``forward_batch`` is the Phase 3 runner contract: callers pass
    both true ``snr_db`` and discrete ``snr_bin`` for every M0--M7 batch, while
    the legacy-compatible ``forward`` retains optional positional SNR bins.
    Inspection metadata binds the SNR normalization bounds and reports exact
    total and additional parameter counts relative to an equivalent M0.
    """

    def __init__(
        self,
        num_classes: int,
        num_levels: int = 1,
        in_channels: int = 64,
        kernel_size: int = 3,
        latent_dim: int = 320,
        regu_details: float = 0.01,
        regu_approx: float = 0.01,
        *,
        conditioning: str = "M0",
        num_snr_bins: int = 20,
        snr_embedding_dim: int = 8,
        snr_min_db: float = -20.0,
        snr_max_db: float = 18.0,
    ) -> None:
        if conditioning not in CONDITIONING_IDS:
            raise ValueError(f"conditioning must be one of {CONDITIONING_IDS}, got {conditioning!r}")
        num_classes = self._validated_count("num_classes", num_classes)
        num_levels = self._validated_count("num_levels", num_levels)
        in_channels = self._validated_count("in_channels", in_channels)
        kernel_size = self._validated_count("kernel_size", kernel_size, odd=True)
        latent_dim = self._validated_count("latent_dim", latent_dim)
        num_snr_bins = self._validated_count("num_snr_bins", num_snr_bins, minimum=2)
        snr_embedding_dim = self._validated_count("snr_embedding_dim", snr_embedding_dim)
        snr_min_db = self._validated_bound("snr_min_db", snr_min_db)
        snr_max_db = self._validated_bound("snr_max_db", snr_max_db)
        if snr_min_db >= snr_max_db:
            raise ValueError("SNR bounds require snr_min_db < snr_max_db")
        super().__init__(
            num_classes=num_classes,
            num_levels=num_levels,
            in_channels=in_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            regu_details=regu_details,
            regu_approx=regu_approx,
        )
        self.conditioning = conditioning
        self.num_snr_bins = int(num_snr_bins)
        self.snr_embedding_dim = int(snr_embedding_dim)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)

        if conditioning == "M1":
            self.fc = self._classifier(self.out_channels + 1)
        elif conditioning == "M2":
            self.fc = self._classifier(self.out_channels + self.num_snr_bins)
        elif conditioning == "M3":
            self.snr_embedding = nn.Embedding(self.num_snr_bins, self.snr_embedding_dim)
            self.fc = self._classifier(self.out_channels + self.snr_embedding_dim)
        elif conditioning == "M4":
            self.logit_bias = nn.Embedding(self.num_snr_bins, self.num_classes)
            nn.init.zeros_(self.logit_bias.weight)
        elif conditioning == "M5":
            self.film = nn.Linear(1, 2 * self.out_channels)
            nn.init.zeros_(self.film.weight)
            with torch.no_grad():
                self.film.bias[: self.out_channels].fill_(1.0)
                self.film.bias[self.out_channels :].zero_()
        elif conditioning == "M6":
            self.gate = nn.Linear(1, self.out_channels)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif conditioning == "M7":
            del self.fc
            self.snr_heads = nn.ModuleList(
                self._classifier(self.out_channels) for _ in range(self.num_snr_bins)
            )

    @staticmethod
    def _validated_count(name: str, value: int, *, minimum: int = 1, odd: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if odd and result % 2 == 0:
            raise ValueError(f"{name} must be odd")
        return result

    @staticmethod
    def _validated_bound(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _classifier(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, self.num_classes),
        )

    def normalize_snr(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Map finite floating SNR values from configured bounds to ``[-1, 1]``."""
        if not isinstance(snr_db, torch.Tensor):
            raise TypeError("snr_db must be a torch.Tensor")
        if snr_db.ndim != 1:
            raise ValueError("snr_db must be a one-dimensional batch tensor")
        if not torch.is_floating_point(snr_db):
            raise TypeError("snr_db must have a floating dtype")
        if not bool(torch.isfinite(snr_db).all()):
            raise ValueError("snr_db values must be finite")
        if bool(((snr_db < self.snr_min_db) | (snr_db > self.snr_max_db)).any()):
            raise ValueError("snr_db lies outside the declared normalization range")
        midpoint = (self.snr_min_db + self.snr_max_db) / 2.0
        half_range = (self.snr_max_db - self.snr_min_db) / 2.0
        return (snr_db - midpoint) / half_range

    def _require_bins(self, snr_bin: torch.Tensor | None, *, device: torch.device | None = None) -> torch.Tensor:
        if snr_bin is None:
            raise ValueError(f"{self.conditioning} requires snr_bin")
        if not isinstance(snr_bin, torch.Tensor):
            raise TypeError("snr_bin must be a torch.Tensor")
        if snr_bin.ndim != 1:
            raise ValueError("snr_bin must be a one-dimensional batch tensor")
        if snr_bin.dtype != torch.long:
            raise TypeError("snr_bin must have torch.long dtype")
        if device is not None and snr_bin.device != device:
            raise ValueError(f"snr_bin device {snr_bin.device} does not match data device {device}")
        if bool(((snr_bin < 0) | (snr_bin >= self.num_snr_bins)).any()):
            raise ValueError(f"snr_bin must be in [0, {self.num_snr_bins - 1}]")
        return snr_bin

    @staticmethod
    def _require_batch_size(condition: torch.Tensor, features: torch.Tensor) -> None:
        if condition.shape[0] != features.shape[0]:
            raise ValueError("SNR condition batch size must match pooled features")

    def encode_condition(
        self,
        *,
        snr_bin: torch.Tensor | None = None,
        snr_db: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the declared scalar, one-hot, embedding, or bin encoding."""
        if self.conditioning in _SCALAR_CONDITIONERS:
            if snr_db is None:
                raise ValueError(f"{self.conditioning} requires snr_db")
            return self.normalize_snr(snr_db).unsqueeze(1)
        if self.conditioning in _DISCRETE_CONDITIONERS:
            bins = self._require_bins(snr_bin)
            if self.conditioning == "M2":
                return F.one_hot(bins, num_classes=self.num_snr_bins).to(dtype=torch.float32)
            if self.conditioning == "M3":
                return self.snr_embedding(bins.to(device=self.snr_embedding.weight.device))
            return bins
        raise ValueError("M0 has no condition encoding")

    def extract_pooled_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the unchanged AWN backbone and return attended pooled features."""
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(2)
        x = self.conv2(x)
        regularizers: list[torch.Tensor] = []
        pooled_details: list[torch.Tensor] = []
        for level in self.levels:
            x, details, regularizer = level(x)
            regularizers.append(regularizer)
            pooled_details.append(self.avgpool(details))
        pooled_details.append(self.avgpool(x))
        features = torch.cat(pooled_details, 1)
        features = features.view(-1, features.size(1))
        features = torch.mul(self.SE_attention_score(features), features)
        return features, regularizers

    def apply_feature_conditioning(
        self,
        features: torch.Tensor,
        *,
        snr_bin: torch.Tensor | None = None,
        snr_db: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply only the mechanism that acts at the pooled-feature location."""
        if self.conditioning == "M0":
            return features
        if self.conditioning in {"M1", "M2", "M3"}:
            raw_condition = snr_db if self.conditioning == "M1" else snr_bin
            if isinstance(raw_condition, torch.Tensor) and raw_condition.device != features.device:
                raise ValueError(
                    f"SNR condition device {raw_condition.device} does not match feature device {features.device}"
                )
            encoded = self.encode_condition(snr_bin=snr_bin, snr_db=snr_db)
            if encoded.device != features.device:
                raise ValueError(f"SNR condition device {encoded.device} does not match feature device {features.device}")
            encoded = encoded.to(dtype=features.dtype)
            self._require_batch_size(encoded, features)
            return torch.cat((features, encoded), dim=1)
        if self.conditioning in {"M4", "M7"}:
            bins = self._require_bins(snr_bin, device=features.device)
            self._require_batch_size(bins, features)
            return features
        if snr_db is None:
            raise ValueError(f"{self.conditioning} requires snr_db")
        if not isinstance(snr_db, torch.Tensor):
            raise TypeError("snr_db must be a torch.Tensor")
        if snr_db.device != features.device:
            raise ValueError(f"snr_db device {snr_db.device} does not match feature device {features.device}")
        scalar = self.normalize_snr(snr_db).unsqueeze(1).to(dtype=features.dtype)
        self._require_batch_size(scalar, features)
        if self.conditioning == "M5":
            gamma, beta = self.film(scalar).chunk(2, dim=1)
            return gamma * features + beta
        gate = 2.0 * torch.sigmoid(self.gate(scalar))
        return gate * features

    def classify_pooled(
        self,
        features: torch.Tensor,
        *,
        snr_bin: torch.Tensor | None = None,
        snr_db: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify already-extracted features using the selected control."""
        conditioned = self.apply_feature_conditioning(features, snr_bin=snr_bin, snr_db=snr_db)
        if self.conditioning == "M4":
            bins = self._require_bins(snr_bin, device=features.device)
            return self.fc(conditioned) + self.logit_bias(bins)
        if self.conditioning == "M7":
            bins = self._require_bins(snr_bin, device=features.device)
            logits = conditioned.new_zeros((conditioned.shape[0], self.num_classes))
            for selected_bin in torch.unique(bins, sorted=True):
                indices = torch.nonzero(bins == selected_bin, as_tuple=False).flatten()
                selected_features = conditioned.index_select(0, indices)
                selected_logits = self.snr_heads[int(selected_bin.item())](selected_features)
                logits = logits.index_copy(0, indices, selected_logits)
            return logits
        return self.fc(conditioned)

    def parameter_counts(self) -> dict[str, int]:
        """Report total capacity and additions beyond an architecture-matched M0.

        The M0-equivalent count is the shared feature extractor plus one normal
        AWN classifier.  Therefore M7's additional count includes only
        ``num_snr_bins - 1`` heads: one head is the baseline-equivalent head.
        """
        feature_modules = (self.conv1, self.conv2, self.levels, self.SE_attention_score)
        feature_count = sum(parameter.numel() for module in feature_modules for parameter in module.parameters())
        baseline_head_count = (
            self.out_channels * self.latent_dim
            + self.latent_dim
            + self.latent_dim * self.num_classes
            + self.num_classes
        )
        m0_equivalent = feature_count + baseline_head_count
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total": total,
            "m0_equivalent": m0_equivalent,
            "additional_conditioner": total - m0_equivalent,
        }

    def conditioner_metadata(self) -> dict[str, Any]:
        """Return stable, serialization-friendly semantics for inspection."""
        common: dict[str, Any] = {
            "id": self.conditioning,
            "pooled_feature_dim": self.out_channels,
            "snr_min_db": self.snr_min_db,
            "snr_max_db": self.snr_max_db,
            "parameter_counts": self.parameter_counts(),
        }
        common["location"] = {
            "M0": "none",
            "M1": "attended_pooled_features",
            "M2": "attended_pooled_features",
            "M3": "attended_pooled_features",
            "M4": "class_logits",
            "M5": "attended_pooled_features",
            "M6": "attended_pooled_features",
            "M7": "classifier_head_selection",
        }[self.conditioning]
        details: dict[str, dict[str, Any]] = {
            "M0": {"conditioner": "none"},
            "M1": {"conditioner": "normalized_scalar_concat", "condition_dim": 1},
            "M2": {"conditioner": "one_hot_snr_concat", "condition_dim": self.num_snr_bins},
            "M3": {"conditioner": "learned_snr_embedding_concat", "condition_dim": self.snr_embedding_dim},
            "M4": {"conditioner": "class_logit_bias", "feature_interaction": "none"},
            "M5": {"conditioner": "film", "modulation_dimension": self.out_channels},
            "M6": {
                "conditioner": "feature_gate",
                "gate_function": "2 * sigmoid(W z + b)",
                "gate_range": "(0, 2)",
                "modulation_dimension": self.out_channels,
            },
            "M7": {
                "conditioner": "per_snr_classifier_heads",
                "shared_feature_extractor": True,
                "classifier_head_count": self.num_snr_bins,
            },
        }
        return common | details[self.conditioning]

    def conditioner_parameters(self) -> dict[str, torch.Tensor]:
        """Expose only tensors that parameterize the selected conditioning path."""
        if self.conditioning == "M0":
            return {}
        if self.conditioning in {"M1", "M2", "M3"}:
            width = {"M1": 1, "M2": self.num_snr_bins, "M3": self.snr_embedding_dim}[self.conditioning]
            result = {"immediate_affine.condition_weight": self.fc[0].weight[:, -width:]}
            if self.conditioning == "M3":
                result["snr_embedding.weight"] = self.snr_embedding.weight
            return result
        if self.conditioning == "M4":
            return {"logit_bias.weight": self.logit_bias.weight}
        if self.conditioning == "M5":
            return {f"film.{name}": parameter for name, parameter in self.film.named_parameters()}
        if self.conditioning == "M6":
            return {f"gate.{name}": parameter for name, parameter in self.gate.named_parameters()}
        return {
            f"snr_heads.{name}": parameter
            for name, parameter in self.snr_heads.named_parameters()
        }

    def forward(
        self,
        x: torch.Tensor,
        snr_bin: torch.Tensor | None = None,
        *,
        snr_db: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(logits, regularizers)`` with Phase 2 positional-bin compatibility."""
        features, regularizers = self.extract_pooled_features(x)
        logits = self.classify_pooled(features, snr_bin=snr_bin, snr_db=snr_db)
        return logits, regularizers

    def forward_batch(
        self,
        signals: torch.Tensor,
        *,
        snr_db: torch.Tensor,
        snr_bin: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Phase 3 runner API accepting both true SNR values and fixed bins.

        Both tensors are validated for every model, then each conditioner uses
        its declared representation.  This gives the runner one call shape
        without weakening M0/M3's Phase 2-compatible ``forward`` signature.
        """
        if not isinstance(signals, torch.Tensor):
            raise TypeError("signals must be a torch.Tensor")
        if signals.ndim != 3:
            raise ValueError("signals must have NCT shape")
        if not torch.is_floating_point(signals):
            raise TypeError("signals must have a floating dtype")
        if not isinstance(snr_db, torch.Tensor):
            raise TypeError("snr_db must be a torch.Tensor")
        if snr_db.device != signals.device:
            raise ValueError(f"snr_db device {snr_db.device} does not match signals device {signals.device}")
        normalized_snr = self.normalize_snr(snr_db)
        bins = self._require_bins(snr_bin, device=signals.device)
        self._require_batch_size(normalized_snr, signals)
        self._require_batch_size(bins, signals)
        return self.forward(signals, bins, snr_db=snr_db)

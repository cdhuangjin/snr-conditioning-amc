"""Empirical prediction-distribution and representation-geometry metrics.

The scalar functions intentionally return plain ``float`` values so they remain
easy to persist.  :func:`compute_diagnostics` wraps those values in
``MetricResult`` objects and supplies a reason whenever a value is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """A metric value plus an explanation when the value is undefined."""

    value: float
    reason: str | None = None

    @property
    def defined(self) -> bool:
        return bool(np.isfinite(self.value))

    def as_dict(self) -> dict[str, float | str | None]:
        return {"value": self.value, "reason": self.reason}


def _as_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(array.dtype, np.number):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    elif any(value is None or (isinstance(value, float) and not np.isfinite(value)) for value in array):
        raise ValueError(f"{name} must contain only finite, non-missing labels")
    return array


def _as_features(features: Any, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError("features, labels, and predictions must have the same length")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def _unique(values: np.ndarray) -> list[Any]:
    try:
        return np.unique(values).tolist()
    except TypeError as error:
        raise ValueError("labels must be mutually comparable") from error


def _normalise_class_labels(class_labels: Iterable[Any] | None, fallback: np.ndarray) -> list[Any]:
    if class_labels is None:
        return _unique(fallback)
    classes = _as_vector(list(class_labels), "class_labels")
    unique = _unique(classes)
    if len(unique) != len(classes):
        raise ValueError("class_labels must not contain duplicates")
    if not unique:
        raise ValueError("class_labels must not be empty")
    return unique


def _canonical_class_universe(
    labels: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> list[Any] | None:
    """Resolve the class universe once for an overall/grouped comparison."""

    if labels.size == 0 and class_labels is None:
        return None
    return _normalise_class_labels(class_labels, labels)


def _outside_universe(values: np.ndarray, classes: Sequence[Any]) -> list[Any]:
    return [value for value in _unique(values) if not any(value == candidate for candidate in classes)]


def _validate_universe(values: np.ndarray, classes: Sequence[Any]) -> None:
    outside = _outside_universe(values, classes)
    if outside:
        raise ValueError(f"labels outside class_labels: {outside}")


def _class_counts(values: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    return np.asarray([np.count_nonzero(values == label) for label in classes], dtype=float)


def prediction_concentration(
    predictions: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return the largest empirical prediction share."""

    predicted = _as_vector(predictions, "predictions")
    if predicted.size == 0:
        return float("nan")
    classes = _normalise_class_labels(class_labels, predicted)
    if class_labels is not None:
        _validate_universe(predicted, classes)
    return float(_class_counts(predicted, classes).max() / predicted.size)


def _balanced_accuracy_result(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> MetricResult:
    if labels.size == 0:
        return MetricResult(float("nan"), "empty_group")
    classes = _normalise_class_labels(class_labels, labels)
    _validate_universe(labels, classes)
    if class_labels is not None:
        _validate_universe(predictions, classes)
    recalls: list[float] = []
    for label in classes:
        mask = labels == label
        if not mask.any():
            return MetricResult(float("nan"), f"class_without_true_samples:{label}")
        recalls.append(float(np.mean(predictions[mask] == label)))
    return MetricResult(float(np.mean(recalls)))


def balanced_accuracy(
    labels: Any,
    predictions: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return macro recall over the exact true or explicitly supplied classes.

    With no ``class_labels``, the universe is the set of true labels, never the
    set of predictions.  Thus a classifier cannot inflate the score merely by
    omitting classes from its predictions.
    """

    true = _as_vector(labels, "labels")
    predicted = _as_vector(predictions, "predictions")
    if len(true) != len(predicted):
        raise ValueError("labels and predictions must have the same length")
    return _balanced_accuracy_result(true, predicted, class_labels).value


def _concentration_gap_result(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> MetricResult:
    if labels.size == 0:
        return MetricResult(float("nan"), "empty_group")
    classes = _normalise_class_labels(class_labels, labels)
    _validate_universe(labels, classes)
    if class_labels is not None:
        _validate_universe(predictions, classes)
    true_share = _class_counts(labels, classes).max() / labels.size
    predicted_classes = classes if class_labels is not None else _unique(predictions)
    predicted_share = _class_counts(predictions, predicted_classes).max() / predictions.size
    return MetricResult(float(predicted_share - true_share))


def concentration_gap(
    labels: Any,
    predictions: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return predicted maximum share minus true-label maximum share.

    This distribution-relative definition is zero for every perfect classifier,
    including datasets whose class frequencies are not uniform.
    """

    true = _as_vector(labels, "labels")
    predicted = _as_vector(predictions, "predictions")
    if len(true) != len(predicted):
        raise ValueError("labels and predictions must have the same length")
    return _concentration_gap_result(true, predicted, class_labels).value


def _prediction_entropy_result(
    predictions: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> MetricResult:
    if predictions.size == 0:
        return MetricResult(float("nan"), "empty_group")
    classes = _normalise_class_labels(class_labels, predictions)
    if class_labels is not None:
        _validate_universe(predictions, classes)
    if len(classes) < 2:
        return MetricResult(float("nan"), "normalized_entropy_requires_at_least_two_classes")
    probabilities = _class_counts(predictions, classes) / predictions.size
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return MetricResult(entropy / float(np.log(len(classes))))


def prediction_entropy(
    predictions: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return normalized Shannon entropy of empirical prediction shares."""

    predicted = _as_vector(predictions, "predictions")
    return _prediction_entropy_result(predicted, class_labels).value


def _gini_result(
    predictions: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> MetricResult:
    if predictions.size == 0:
        return MetricResult(float("nan"), "empty_group")
    classes = _normalise_class_labels(class_labels, predictions)
    if class_labels is not None:
        _validate_universe(predictions, classes)
    shares = _class_counts(predictions, classes) / predictions.size
    mean_share = float(np.mean(shares))
    pairwise_difference = np.abs(shares[:, None] - shares[None, :]).sum()
    value = pairwise_difference / (2.0 * len(shares) ** 2 * mean_share)
    return MetricResult(float(value))


def gini_prediction_distribution(
    predictions: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return the Gini coefficient across class prediction shares."""

    predicted = _as_vector(predictions, "predictions")
    return _gini_result(predicted, class_labels).value


def _effective_rank_from_eigenvalues_result(eigenvalues: Any) -> MetricResult:
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("eigenvalues must contain only finite values")
    if values.size == 0:
        return MetricResult(float("nan"), "empty_spectrum")
    if np.any(values < 0.0):
        raise ValueError("eigenvalues must be non-negative")
    total = float(values.sum())
    if total == 0.0:
        return MetricResult(float("nan"), "zero_spectrum")
    probabilities = values[values > 0.0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return MetricResult(float(np.exp(entropy)))


def effective_rank_from_eigenvalues(eigenvalues: Any) -> float:
    """Return entropy effective rank for a non-negative spectrum."""

    return _effective_rank_from_eigenvalues_result(eigenvalues).value


def _effective_rank_result(features: np.ndarray) -> MetricResult:
    if features.shape[0] == 0:
        return MetricResult(float("nan"), "empty_group")
    if features.shape[0] < 2:
        return MetricResult(float("nan"), "requires_at_least_two_samples")
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return _effective_rank_from_eigenvalues_result(singular_values**2)


def effective_rank(features: Any) -> float:
    """Return effective rank of the centered feature covariance spectrum."""

    return _effective_rank_result(_as_features(features)).value


def _top_singular_value_share_result(features: np.ndarray) -> MetricResult:
    if features.shape[0] == 0:
        return MetricResult(float("nan"), "empty_group")
    if features.shape[0] < 2:
        return MetricResult(float("nan"), "requires_at_least_two_samples")
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(singular_values.sum())
    if total == 0.0:
        return MetricResult(float("nan"), "zero_spectrum")
    return MetricResult(float(singular_values[0] / total))


def top_singular_value_share(features: Any) -> float:
    """Return the leading centered singular value divided by their sum."""

    return _top_singular_value_share_result(_as_features(features)).value


def _class_separability_result(
    features: np.ndarray,
    labels: np.ndarray,
    class_labels: Iterable[Any] | None,
) -> MetricResult:
    if labels.size == 0:
        return MetricResult(float("nan"), "empty_group")
    classes = _normalise_class_labels(class_labels, labels)
    _validate_universe(labels, classes)
    observed = [label for label in classes if np.any(labels == label)]
    if len(observed) < 2:
        return MetricResult(float("nan"), "requires_at_least_two_observed_classes")
    missing = [label for label in classes if not np.any(labels == label)]
    if missing:
        return MetricResult(float("nan"), f"class_without_true_samples:{missing[0]}")

    global_mean = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in classes:
        class_features = features[labels == label]
        class_mean = class_features.mean(axis=0)
        between += len(class_features) * float(np.sum((class_mean - global_mean) ** 2))
        within += float(np.sum((class_features - class_mean) ** 2))
    if within == 0.0:
        return MetricResult(float("nan"), "zero_within_class_scatter")
    return MetricResult(float(between / within))


def class_separability(
    features: Any,
    labels: Any,
    *,
    class_labels: Iterable[Any] | None = None,
) -> float:
    """Return trace between-class scatter divided by within-class scatter."""

    true = _as_vector(labels, "labels")
    matrix = _as_features(features, expected_rows=len(true))
    return _class_separability_result(matrix, true, class_labels).value


def compute_diagnostics(
    labels: Any,
    predictions: Any,
    *,
    features: Any | None = None,
    class_labels: Iterable[Any] | None = None,
) -> dict[str, MetricResult]:
    """Compute all diagnostics for one overall or grouped sample set."""

    true = _as_vector(labels, "labels")
    predicted = _as_vector(predictions, "predictions")
    if len(true) != len(predicted):
        raise ValueError("labels and predictions must have the same length")
    classes = _canonical_class_universe(true, class_labels)
    if true.size == 0:
        empty = MetricResult(float("nan"), "empty_group")
        return {name: empty for name in _DIAGNOSTIC_NAMES}
    assert classes is not None
    _validate_universe(true, classes)
    if class_labels is not None:
        _validate_universe(predicted, classes)

    concentration = MetricResult(prediction_concentration(predicted, class_labels=classes))
    diagnostics = {
        "prediction_concentration": concentration,
        "balanced_accuracy": _balanced_accuracy_result(true, predicted, classes),
        "concentration_gap": _concentration_gap_result(true, predicted, classes),
        "prediction_entropy": _prediction_entropy_result(predicted, classes),
        "gini_prediction_distribution": _gini_result(predicted, classes),
    }
    if features is None:
        absent = MetricResult(float("nan"), "features_not_provided")
        diagnostics.update(
            effective_rank=absent,
            top_singular_value_share=absent,
            class_separability=absent,
        )
    else:
        matrix = _as_features(features, expected_rows=len(true))
        diagnostics.update(
            effective_rank=_effective_rank_result(matrix),
            top_singular_value_share=_top_singular_value_share_result(matrix),
            class_separability=_class_separability_result(matrix, true, classes),
        )
    return diagnostics


def diagnostics_by_snr(
    labels: Any,
    predictions: Any,
    snr_db: Any,
    *,
    features: Any | None = None,
    class_labels: Iterable[Any] | None = None,
    snr_values: Iterable[float] | None = None,
) -> dict[Any, dict[str, MetricResult]]:
    """Compute diagnostics independently for each requested SNR group."""

    true = _as_vector(labels, "labels")
    predicted = _as_vector(predictions, "predictions")
    snr = _as_vector(snr_db, "snr_db")
    if not (len(true) == len(predicted) == len(snr)):
        raise ValueError("labels, predictions, and snr_db must have the same length")
    canonical_classes = _canonical_class_universe(true, class_labels)
    if canonical_classes is not None:
        _validate_universe(true, canonical_classes)
        _validate_universe(predicted, canonical_classes)
    matrix = None if features is None else _as_features(features, expected_rows=len(true))
    requested = _unique(snr) if snr_values is None else _as_vector(list(snr_values), "snr_values").tolist()
    if len(_unique(np.asarray(requested))) != len(requested):
        raise ValueError("snr_values must not contain duplicates")

    grouped: dict[Any, dict[str, MetricResult]] = {}
    for value in requested:
        key = value.item() if isinstance(value, np.generic) else value
        mask = snr == value
        grouped[key] = compute_diagnostics(
            true[mask],
            predicted[mask],
            features=None if matrix is None else matrix[mask],
            class_labels=canonical_classes,
        )
    return grouped


def diagnostic_report(
    labels: Any,
    predictions: Any,
    snr_db: Any,
    *,
    features: Any | None = None,
    class_labels: Iterable[Any] | None = None,
    snr_values: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Return the two-level overall/per-SNR diagnostic report."""

    true = _as_vector(labels, "labels")
    canonical_classes = _canonical_class_universe(true, class_labels)
    return {
        "overall": compute_diagnostics(
            true,
            predictions,
            features=features,
            class_labels=canonical_classes,
        ),
        "per_snr": diagnostics_by_snr(
            true,
            predictions,
            snr_db,
            features=features,
            class_labels=canonical_classes,
            snr_values=snr_values,
        ),
    }


_DIAGNOSTIC_NAMES = (
    "prediction_concentration",
    "balanced_accuracy",
    "concentration_gap",
    "prediction_entropy",
    "gini_prediction_distribution",
    "effective_rank",
    "top_singular_value_share",
    "class_separability",
)

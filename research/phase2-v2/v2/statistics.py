"""Paired central-comparison statistics for fixed-seed experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _paired_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def paired_summary(baseline: Any, treatment: Any) -> dict[str, Any]:
    """Summarize treatment-minus-baseline paired differences.

    The paired t-test p value is deliberately named ``p_value_secondary`` and
    emitted last: confidence intervals and effect sizes are the primary
    evidence.  Undefined inference is represented by NaN plus a reason rather
    than a silently substituted zero.
    """

    baseline_values = _paired_vector(baseline, "baseline")
    treatment_values = _paired_vector(treatment, "treatment")
    if len(baseline_values) != len(treatment_values):
        raise ValueError("baseline and treatment must have the same length")

    n_pairs = len(baseline_values)
    differences = treatment_values - baseline_values
    nan = float("nan")
    undefined: dict[str, str] = {}

    baseline_mean = float(np.mean(baseline_values)) if n_pairs else nan
    treatment_mean = float(np.mean(treatment_values)) if n_pairs else nan
    difference_mean = float(np.mean(differences)) if n_pairs else nan
    if n_pairs >= 2:
        baseline_std = float(np.std(baseline_values, ddof=1))
        treatment_std = float(np.std(treatment_values, ddof=1))
        difference_std = float(np.std(differences, ddof=1))
    else:
        baseline_std = treatment_std = difference_std = nan

    ci95_low = ci95_high = cohen_dz = p_value = nan
    if n_pairs == 0:
        undefined["descriptive_statistics"] = "empty_pairs"
        undefined["paired_inference"] = "requires_at_least_two_pairs"
    elif n_pairs == 1:
        undefined["sample_standard_deviation"] = "requires_at_least_two_pairs"
        undefined["paired_inference"] = "requires_at_least_two_pairs"
    elif difference_std == 0.0:
        undefined["paired_inference"] = "zero_difference_variance"
    else:
        standard_error = difference_std / float(np.sqrt(n_pairs))
        critical = float(stats.t.ppf(0.975, df=n_pairs - 1))
        margin = critical * standard_error
        ci95_low = difference_mean - margin
        ci95_high = difference_mean + margin
        cohen_dz = difference_mean / difference_std
        p_value = float(stats.ttest_rel(treatment_values, baseline_values).pvalue)

    return {
        "n_pairs": n_pairs,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "treatment_mean": treatment_mean,
        "treatment_std": treatment_std,
        "differences": differences.tolist(),
        "difference_mean": difference_mean,
        "difference_std": difference_std,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "cohen_dz": cohen_dz,
        "undefined": undefined,
        "p_value_secondary": p_value,
    }

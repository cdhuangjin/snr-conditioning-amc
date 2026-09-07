import math

import numpy as np
import pytest


def test_paired_summary_known_differences_and_ordered_secondary_p_value():
    from v2.statistics import paired_summary

    summary = paired_summary(
        baseline=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        treatment=np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
    )

    assert summary["n_pairs"] == 5
    assert summary["baseline_mean"] == pytest.approx(3.0)
    assert summary["baseline_std"] == pytest.approx(np.sqrt(2.5))
    assert summary["treatment_mean"] == pytest.approx(6.0)
    assert summary["treatment_std"] == pytest.approx(np.sqrt(10.0))
    assert summary["differences"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert summary["difference_mean"] == pytest.approx(3.0)
    assert summary["difference_std"] == pytest.approx(np.sqrt(2.5))
    assert summary["ci95_low"] == pytest.approx(1.0367568385)
    assert summary["ci95_high"] == pytest.approx(4.9632431615)
    assert summary["cohen_dz"] == pytest.approx(3.0 / np.sqrt(2.5))
    assert summary["p_value_secondary"] == pytest.approx(0.0132355996)
    assert list(summary)[-1] == "p_value_secondary"


def test_paired_summary_n_one_retains_descriptive_difference_and_explains_nan_inference():
    from v2.statistics import paired_summary

    summary = paired_summary([0.4], [0.5])

    assert summary["differences"] == pytest.approx([0.1])
    assert summary["difference_mean"] == pytest.approx(0.1)
    for field in [
        "baseline_std",
        "treatment_std",
        "difference_std",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "p_value_secondary",
    ]:
        assert math.isnan(summary[field])
    assert summary["undefined"]["paired_inference"] == "requires_at_least_two_pairs"


def test_paired_summary_empty_input_returns_reasoned_nan_not_silent_zero():
    from v2.statistics import paired_summary

    summary = paired_summary([], [])
    assert summary["n_pairs"] == 0
    assert summary["differences"] == []
    assert math.isnan(summary["difference_mean"])
    assert summary["undefined"]["descriptive_statistics"] == "empty_pairs"
    assert math.isnan(summary["p_value_secondary"])


@pytest.mark.parametrize(
    ("baseline", "treatment"),
    [
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_zero_variance_differences_are_marked_undefined_without_infinities(baseline, treatment):
    from v2.statistics import paired_summary

    summary = paired_summary(baseline, treatment)
    assert summary["difference_std"] == 0.0
    for field in ["ci95_low", "ci95_high", "cohen_dz", "p_value_secondary"]:
        assert math.isnan(summary[field])
    assert summary["undefined"]["paired_inference"] == "zero_difference_variance"


@pytest.mark.parametrize(
    ("baseline", "treatment", "message"),
    [
        ([1.0, 2.0], [1.0], "same length"),
        ([[1.0], [2.0]], [[1.0], [2.0]], "one-dimensional"),
        ([1.0, np.nan], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, np.inf], "finite"),
    ],
)
def test_paired_summary_rejects_invalid_inputs(baseline, treatment, message):
    from v2.statistics import paired_summary

    with pytest.raises(ValueError, match=message):
        paired_summary(baseline, treatment)

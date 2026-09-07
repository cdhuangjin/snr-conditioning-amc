import math

import numpy as np
import pytest


def test_single_class_classifier_has_unit_concentration_and_expected_gap():
    from v2.metrics import balanced_accuracy, concentration_gap, prediction_concentration

    labels = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.zeros_like(labels)

    assert prediction_concentration(predictions) == 1.0
    assert balanced_accuracy(labels, predictions, class_labels=[0, 1, 2]) == pytest.approx(1 / 3)
    assert concentration_gap(labels, predictions, class_labels=[0, 1, 2]) == pytest.approx(2 / 3)


def test_perfect_classifier_has_zero_concentration_gap_even_when_labels_are_imbalanced():
    from v2.metrics import concentration_gap

    labels = np.array([0, 0, 0, 1, 2])
    assert concentration_gap(labels, labels, class_labels=[0, 1, 2]) == 0.0


def test_balanced_accuracy_uses_true_class_universe_when_predictions_omit_classes():
    from v2.metrics import balanced_accuracy

    labels = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.zeros_like(labels)
    assert balanced_accuracy(labels, predictions) == pytest.approx(1 / 3)


def test_uniform_prediction_distribution_has_zero_gini_and_unit_normalized_entropy():
    from v2.metrics import gini_prediction_distribution, prediction_entropy

    predictions = np.array([0, 1, 2] * 10)
    assert gini_prediction_distribution(predictions) == 0.0
    assert prediction_entropy(predictions, class_labels=[0, 1, 2]) == pytest.approx(1.0)


def test_concentrated_prediction_distribution_uses_explicit_class_universe():
    from v2.metrics import gini_prediction_distribution

    predictions = np.zeros(12, dtype=int)
    assert gini_prediction_distribution(predictions, class_labels=[0, 1, 2]) == pytest.approx(2 / 3)


def test_effective_rank_known_spectrum():
    from v2.metrics import effective_rank_from_eigenvalues

    assert effective_rank_from_eigenvalues(np.array([1.0, 1.0, 0.0])) == pytest.approx(2.0)


def test_geometry_metrics_are_invariant_to_nonzero_feature_scale():
    from v2.metrics import class_separability, effective_rank, effective_rank_from_eigenvalues, top_singular_value_share

    features = np.array([[-2.0], [-1.0], [1.0], [2.0]]) * 1e-150
    labels = np.array([0, 0, 1, 1])
    assert effective_rank_from_eigenvalues(np.array([1e-300, 1e-300])) == pytest.approx(2.0)
    assert effective_rank(features) == pytest.approx(1.0)
    assert top_singular_value_share(features) == pytest.approx(1.0)
    assert class_separability(features, labels, class_labels=[0, 1]) == pytest.approx(9.0)


def test_effective_rank_rejects_negative_eigenvalues_at_any_scale():
    from v2.metrics import effective_rank_from_eigenvalues

    with pytest.raises(ValueError, match="non-negative"):
        effective_rank_from_eigenvalues(np.array([1e-20, -1e-20]))


def test_effective_rank_and_top_singular_share_for_rank_one_features():
    from v2.metrics import effective_rank, top_singular_value_share

    features = np.array([[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]])
    assert effective_rank(features) == pytest.approx(1.0)
    assert top_singular_value_share(features) == pytest.approx(1.0)


def test_class_separability_is_between_to_within_scatter_ratio():
    from v2.metrics import class_separability

    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    assert class_separability(features, labels, class_labels=[0, 1]) == pytest.approx(9.0)


def test_single_observed_class_reports_undefined_separability_with_reason():
    from v2.metrics import compute_diagnostics

    diagnostics = compute_diagnostics(
        labels=np.zeros(3, dtype=int),
        predictions=np.zeros(3, dtype=int),
        features=np.array([[0.0], [1.0], [2.0]]),
        class_labels=[0],
    )
    result = diagnostics["class_separability"]
    assert math.isnan(result.value)
    assert result.reason == "requires_at_least_two_observed_classes"


def test_missing_expected_true_class_is_not_silently_dropped_from_balanced_accuracy():
    from v2.metrics import compute_diagnostics

    diagnostics = compute_diagnostics(
        labels=np.array([0, 0, 1, 1]),
        predictions=np.array([0, 0, 1, 1]),
        class_labels=[0, 1, 2],
    )
    result = diagnostics["balanced_accuracy"]
    assert math.isnan(result.value)
    assert result.reason == "class_without_true_samples:2"


def test_per_snr_api_includes_requested_empty_group_with_nan_reasons():
    from v2.metrics import diagnostics_by_snr

    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 0, 0, 1])
    snr_db = np.array([-8, -8, 0, 0])
    features = np.array([[0.0], [1.0], [2.0], [3.0]])

    grouped = diagnostics_by_snr(
        labels,
        predictions,
        snr_db,
        features=features,
        class_labels=[0, 1],
        snr_values=[-8, 0, 8],
    )

    assert grouped[-8]["prediction_concentration"].value == 1.0
    assert grouped[0]["balanced_accuracy"].value == pytest.approx(1.0)
    assert all(math.isnan(result.value) for result in grouped[8].values())
    assert {result.reason for result in grouped[8].values()} == {"empty_group"}


def test_report_exposes_overall_and_per_snr_layers():
    from v2.metrics import diagnostic_report

    report = diagnostic_report(
        labels=np.array([0, 1, 0, 1]),
        predictions=np.array([0, 0, 0, 1]),
        snr_db=np.array([-8, -8, 0, 0]),
        class_labels=[0, 1],
    )
    assert set(report) == {"overall", "per_snr"}
    assert report["overall"]["prediction_concentration"].value == pytest.approx(0.75)
    assert set(report["per_snr"]) == {-8, 0}


def test_report_infers_class_universe_once_so_missing_snr_class_is_not_silently_dropped():
    from v2.metrics import diagnostic_report

    report = diagnostic_report(
        labels=np.array([0, 1, 0, 1, 2, 2]),
        predictions=np.array([0, 1, 0, 1, 2, 2]),
        snr_db=np.array([-8, -8, 0, 0, 0, 0]),
    )

    low = report["per_snr"][-8]
    assert math.isnan(low["balanced_accuracy"].value)
    assert low["balanced_accuracy"].reason == "class_without_true_samples:2"
    assert low["prediction_entropy"].value == pytest.approx(np.log(2) / np.log(3))
    assert low["gini_prediction_distribution"].value == pytest.approx(1 / 3)


def test_direct_per_snr_api_infers_one_global_class_universe():
    from v2.metrics import diagnostics_by_snr

    grouped = diagnostics_by_snr(
        labels=np.array([0, 1, 0, 1, 2, 2]),
        predictions=np.array([0, 1, 0, 1, 2, 2]),
        snr_db=np.array([-8, -8, 0, 0, 0, 0]),
    )

    assert grouped[-8]["balanced_accuracy"].reason == "class_without_true_samples:2"
    assert grouped[-8]["prediction_entropy"].value == pytest.approx(np.log(2) / np.log(3))


def test_report_uses_the_same_explicit_class_universe_at_both_levels():
    from v2.metrics import diagnostic_report

    report = diagnostic_report(
        labels=np.array([0, 1, 0, 1]),
        predictions=np.array([0, 1, 0, 1]),
        snr_db=np.array([-8, -8, 0, 0]),
        class_labels=[2, 0, 1],
    )

    assert report["overall"]["balanced_accuracy"].reason == "class_without_true_samples:2"
    for metrics in report["per_snr"].values():
        assert metrics["balanced_accuracy"].reason == "class_without_true_samples:2"
        assert metrics["prediction_entropy"].value == pytest.approx(np.log(2) / np.log(3))
        assert metrics["gini_prediction_distribution"].value == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda m: m.prediction_concentration(np.array([0.0, np.nan])), "finite"),
        (lambda m: m.balanced_accuracy(np.array([0, 1]), np.array([0])), "same length"),
        (lambda m: m.effective_rank(np.ones((2, 2, 1))), "two-dimensional"),
        (
            lambda m: m.class_separability(np.ones((2, 2)), np.array([0, 2]), class_labels=[0, 1]),
            "outside class_labels",
        ),
    ],
)
def test_metric_inputs_are_validated(call, message):
    import v2.metrics as metrics

    with pytest.raises(ValueError, match=message):
        call(metrics)


def test_empty_scalar_metric_is_nan_and_diagnostic_api_supplies_reason():
    from v2.metrics import compute_diagnostics, prediction_concentration

    assert math.isnan(prediction_concentration(np.array([], dtype=int)))
    diagnostics = compute_diagnostics(np.array([], dtype=int), np.array([], dtype=int), class_labels=[0, 1])
    assert diagnostics["prediction_concentration"].reason == "empty_group"

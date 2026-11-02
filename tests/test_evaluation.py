"""Regression cases for OOF threshold policies and reported uncertainty."""

import json

import numpy as np
import pytest

from secom.evaluation import (
    bootstrap_ap,
    choose_threshold,
    operating_metrics,
    ranking_metrics,
    wilson_interval,
)


def test_threshold_maximizes_f1_on_pooled_oof_predictions():
    # Each pair represents a held-out fold; the policy uses their pooled scores.
    y = np.concatenate(([0, 1], [0, 1]))
    scores = np.concatenate(([0.1, 0.4], [0.6, 0.9]))

    threshold = choose_threshold(y, scores)

    assert threshold == 0.4
    assert operating_metrics(y, scores, threshold)["f1"] == pytest.approx(0.8)
    for order in ([3, 1, 0, 2], [2, 0, 3, 1]):
        assert choose_threshold(y[order], scores[order]) == threshold


def test_exact_f1_tie_chooses_highest_observed_threshold():
    y = [1, 0, 0, 1]
    scores = [0.1, 0.3, 0.6, 0.9]

    assert operating_metrics(y, scores, 0.1)["f1"] == pytest.approx(2 / 3)
    assert operating_metrics(y, scores, 0.9)["f1"] == pytest.approx(2 / 3)
    assert choose_threshold(y, scores) == 0.9


def test_nearly_equal_f1_values_are_not_treated_as_exact_ties():
    # Adding one positive and two negatives improves F1 by about 0.0000055.
    y = np.concatenate((np.ones(200), np.zeros(200), [1, 0, 0]))
    scores = np.concatenate((np.full(400, 0.9), np.full(3, 0.1)))
    low_f1 = 402 / 604
    high_f1 = 400 / 601

    assert low_f1 > high_f1
    assert np.isclose(low_f1, high_f1)
    assert choose_threshold(y, scores) == 0.1


def test_exact_count_tie_survives_different_floating_point_rounding():
    labels = [0, 0, 1, 1, 1, 0, 0, 1]
    scores = [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    assert choose_threshold(labels, scores) == 0.5
    assert operating_metrics(labels, scores, 0.5)["alarm_count"] == 5


@pytest.mark.parametrize("score", [-2.0, 0.0, 0.37, 1.0])
def test_constant_scores_choose_a_real_threshold_without_using_pr_terminal_point(score):
    y = [0, 0, 0, 1]
    scores = np.full(4, score)

    threshold = choose_threshold(y, scores)
    metrics = operating_metrics(y, scores, threshold)

    assert threshold == score
    assert np.isfinite(threshold)
    assert metrics["alarm_count"] == 4
    assert metrics["f1"] == pytest.approx(0.4)


def test_equal_scores_cannot_be_split_into_different_threshold_decisions():
    y = [1, 0, 1, 0, 0]
    scores = [0.8, 0.8, 0.8, 0.2, 0.1]

    threshold = choose_threshold(y, scores)
    metrics = operating_metrics(y, scores, threshold)

    assert threshold == 0.8
    assert metrics["alarm_count"] == 3
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["f1"] == pytest.approx(0.8)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_threshold_rejects_nonfinite_scores(bad_score):
    with pytest.raises(ValueError, match="finite scores"):
        choose_threshold([0, 1], [0.2, bad_score])


@pytest.mark.parametrize("y", [[0, 0], [1, 1], [-1, 1]])
def test_threshold_requires_both_binary_label_classes(y):
    with pytest.raises(ValueError, match="both label classes"):
        choose_threshold(y, [0.2, 0.8])


def test_operating_metrics_match_hand_counted_confusion_and_wilson_interval():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    scores = [0.8, 0.6, 0.7, 0.2, 0.9, 0.55, 0.5, 0.3, 0.1, 0.0]

    metrics = operating_metrics(y, scores, threshold=0.6)

    assert metrics == {
        "threshold": 0.6,
        "precision": pytest.approx(3 / 4),
        "recall": pytest.approx(3 / 4),
        "f1": pytest.approx(3 / 4),
        "accuracy": pytest.approx(8 / 10),
        "balanced_accuracy": pytest.approx(19 / 24),
        "true_negatives": 5,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives": 3,
        "alarm_count": 4,
        "alarm_rate": pytest.approx(0.4),
        "failure_support": 4,
        "pass_support": 6,
        "pass_precision": pytest.approx(5 / 6),
        "pass_recall": pytest.approx(5 / 6),
        "pass_f1": pytest.approx(5 / 6),
        "recall_wilson_95": pytest.approx([0.30064184258240184, 0.9544127391902995]),
    }
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("threshold,alarms", [(0.5, 0), (0.1, 4)])
def test_no_alarm_and_all_alarm_metrics_are_finite(threshold, alarms):
    metrics = operating_metrics([0, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], threshold)

    assert metrics["alarm_count"] == alarms
    assert metrics["alarm_rate"] == alarms / 4
    assert metrics["failure_support"] == 1
    assert metrics["pass_support"] == 3
    if alarms == 0:
        assert metrics["true_negatives"] == 3
        assert metrics["false_negatives"] == 1
        assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
        assert metrics["recall_wilson_95"] == pytest.approx([0, 0.7934506856227626])
    else:
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 3
        assert metrics["precision"] == pytest.approx(0.25)
        assert metrics["recall"] == 1
        assert metrics["f1"] == pytest.approx(0.4)
        assert metrics["pass_precision"] == metrics["pass_recall"] == metrics["pass_f1"] == 0
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "successes,expected",
    [
        (0, [0.0, 0.2775327998628892]),
        (10, [0.7224672001371107, 1.0]),
    ],
)
def test_wilson_interval_remains_informative_at_extreme_observed_recalls(successes, expected):
    assert wilson_interval(successes, 10) == pytest.approx(expected)


def test_recall_interval_is_undefined_without_failure_support():
    assert wilson_interval(0, 0) is None
    with pytest.warns(UserWarning, match="y_pred contains classes not in y_true"):
        metrics = operating_metrics([0, 0], [0.1, 0.9], threshold=0.5)
    assert metrics["recall_wilson_95"] is None


def test_ranking_uses_average_precision_for_a_hand_ranked_example():
    # Positives occupy ranks 1 and 3: AP = (1 + 2/3) / 2.
    metrics = ranking_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])

    assert metrics == {
        "average_precision": pytest.approx(5 / 6),
        "roc_auc": pytest.approx(3 / 4),
        "failure_prevalence": 0.5,
    }


def test_tied_ranking_scores_have_prevalence_ap_and_chance_roc_auc():
    metrics = ranking_metrics([0, 0, 0, 1], [0.4] * 4)

    assert metrics["average_precision"] == pytest.approx(0.25)
    assert metrics["roc_auc"] == 0.5
    assert metrics["failure_prevalence"] == 0.25


@pytest.mark.parametrize("seed", [4, 91])
def test_stratified_bootstrap_keeps_observed_prevalence_with_constant_scores(seed):
    y = np.array([0] * 12 + [1] * 3)

    assert bootstrap_ap(y, np.full(15, 0.4), repetitions=37, seed=seed) == pytest.approx([0.2, 0.2])


def test_bootstrap_perfect_ranking_has_degenerate_ap_interval():
    assert bootstrap_ap([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9], 25, 7) == [1, 1]


def test_bootstrap_resamples_reproducibly_without_mutating_fixed_scores():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.9, 0.3, 0.7, 0.8])
    original_y, original_scores = y.copy(), scores.copy()

    interval = bootstrap_ap(y, scores, repetitions=80, seed=42)

    assert interval == bootstrap_ap(y, scores, repetitions=80, seed=42)
    lower, upper = interval
    assert 0 <= lower < ranking_metrics(y, scores)["average_precision"] < upper <= 1
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(scores, original_scores)


@pytest.mark.parametrize("y,repetitions", [([0, 0], 5), ([1, 1], 5), ([0, 1], 0), ([0, 1], -1)])
def test_bootstrap_requires_both_classes_and_positive_repetitions(y, repetitions):
    with pytest.raises(ValueError, match="both classes and at least one repetition"):
        bootstrap_ap(y, [0.2, 0.8], repetitions, seed=0)

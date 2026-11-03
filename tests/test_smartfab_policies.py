"""Capacity, exact OOF F2 selection, and chronology contracts for smartfab."""

import json
from fractions import Fraction
from hashlib import sha256
from math import floor

import numpy as np
import pandas as pd
import pytest

from secom.evaluation import wilson_interval
from smartfab.policies import (
    batch_ranks,
    capacity_mask,
    choose_f2_threshold,
    chronological_partition,
    evaluate_policy,
    forward_splits,
    threshold_mask,
)


@pytest.mark.parametrize("fraction", [0, 0.14, 0.2, 0.49, 0.999, 1])
def test_capacity_is_exactly_floor_fraction_times_batch_size(fraction):
    scores = np.array([-100, 2, 3, 3, 3, 8, 1e10])
    row_ids = np.array([101, 67, 99, 24, 0, -2, 12])
    original_scores, original_ids = scores.copy(), row_ids.copy()

    mask = capacity_mask(scores, row_ids, fraction)

    assert mask.dtype == bool
    assert mask.sum() == floor(fraction * len(scores))
    if mask.any() and not mask.all():
        assert scores[mask].min() >= scores[~mask].max()
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(row_ids, original_ids)


def test_tied_capacity_uses_scores_then_seeded_row_hashes_without_labels():
    scores = np.array([20, 10, 10, 10, 10, -3])
    row_ids = np.array([99, 12, -7, 45, 6, 101])
    tied_order = sorted(
        range(1, 5), key=lambda i: sha256(f"2026:{row_ids[i]}".encode("ascii")).digest()
    )
    expected = np.zeros(6, dtype=bool)
    expected[[0, tied_order[0], tied_order[1]]] = True

    mask = capacity_mask(scores, row_ids, 0.5)

    np.testing.assert_array_equal(mask, expected)
    # Changing failure labels can change measured recall, never review selection.
    assert evaluate_policy(expected, mask)["recall"] == 1
    assert evaluate_policy(~expected, mask)["recall"] == 0
    np.testing.assert_array_equal(capacity_mask(scores, row_ids, 0.5), mask)
    for permutation in (np.arange(6)[::-1], np.array([2, 0, 5, 3, 1, 4])):
        permuted = capacity_mask(scores[permutation], row_ids[permutation], 0.5)
        np.testing.assert_array_equal(permuted, expected[permutation])


def test_constant_score_baseline_has_reproducible_hash_order_across_permutations():
    row_ids = np.arange(60)
    expected = sorted(
        row_ids, key=lambda row_id: sha256(f"2026:{row_id}".encode("ascii")).digest()
    )[:12]
    scores = np.full(60, -2.5)
    mask = capacity_mask(scores, row_ids, 0.2)

    assert set(row_ids[mask]) == set(expected)
    permutation = np.random.default_rng(13).permutation(60)
    np.testing.assert_array_equal(
        capacity_mask(scores[permutation], row_ids[permutation], 0.2), mask[permutation]
    )
    assert not np.array_equal(capacity_mask(scores, row_ids, 0.2, seed=7), mask)


def test_capacity_keeps_large_integer_row_ids_distinct():
    ids = [2**80, 2**80 + 1, -(2**80), -(2**80) + 1]
    mask = capacity_mask([1] * 4, ids, 0.5)
    reverse_mask = capacity_mask([1] * 4, ids[::-1], 0.5)

    assert mask.sum() == 2
    np.testing.assert_array_equal(mask, reverse_mask[::-1])


@pytest.mark.parametrize(
    "row_ids",
    [[1], [[1, 2]], [1, 1], [1.0, 2.0], [True, 2], ["1", "2"], [1, None]],
)
def test_capacity_rejects_misaligned_noninteger_or_duplicate_row_ids(row_ids):
    with pytest.raises(ValueError, match="Row IDs"):
        capacity_mask([0.1, 0.2], row_ids, 0.2)


@pytest.mark.parametrize("seed", [2.5, None, "2026", True])
def test_capacity_rejects_noninteger_seeds(seed):
    with pytest.raises(ValueError, match="seed"):
        capacity_mask([1, 2], [1, 2], 0.2, seed=seed)


@pytest.mark.parametrize(
    "fraction", [-0.1, 1.1, np.nan, np.inf, -np.inf, [0.2], None, np.complex128(0.2 + 1j)]
)
def test_capacity_and_threshold_selection_reject_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="fraction"):
        capacity_mask([0, 1], [0, 1], fraction)
    with pytest.raises(ValueError, match="max_alert_fraction"):
        choose_f2_threshold([0, 1], [0, 1], fraction)


@pytest.mark.parametrize(
    "scores", [[], [[1, 2]], [0, np.nan], [0, np.inf], [-np.inf, 0], [1j, 0], ["bad", 1]]
)
@pytest.mark.parametrize("policy", ["capacity", "ranks", "threshold", "f2"])
def test_score_apis_reject_empty_nonvector_or_nonfinite_scores(scores, policy):
    with pytest.raises(ValueError, match="Scores"):
        if policy == "capacity":
            capacity_mask(scores, [0, 1], 0.2)
        elif policy == "ranks":
            batch_ranks(scores)
        elif policy == "threshold":
            threshold_mask(scores, None)
        else:
            choose_f2_threshold([0, 1], scores)


def test_batch_ranks_are_average_percentiles_with_larger_scores_higher():
    scores = np.array([-10, 4, 4, 0, 20])
    expected = np.array([1, 3.5, 3.5, 2, 5]) / 5

    np.testing.assert_array_equal(batch_ranks(scores), expected)
    permutation = [4, 1, 0, 2, 3]
    np.testing.assert_array_equal(batch_ranks(scores[permutation]), expected[permutation])
    np.testing.assert_array_equal(batch_ranks([7] * 5), np.full(5, 0.6))
    np.testing.assert_array_equal(batch_ranks([7]), [1])


def test_policy_metrics_match_hand_counts_and_existing_wilson_interval():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    mask = [1, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    metrics = evaluate_policy(y, mask)

    assert metrics == {
        "true_positives": 3,
        "false_negatives": 1,
        "false_positives": 2,
        "true_negatives": 4,
        "precision": pytest.approx(3 / 5),
        "recall": pytest.approx(3 / 4),
        "f1": pytest.approx(2 / 3),
        "f2": pytest.approx(5 / 7),
        "alarm_count": 5,
        "alarm_rate": 0.5,
        "reviews_per_caught": pytest.approx(5 / 3),
        "failure_support": 4,
        "pass_support": 6,
        "recall_wilson_95": wilson_interval(3, 4),
    }
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "y,mask,precision,recall,f1,f2,reviews",
    [
        ([0, 1, 0], [0, 0, 0], 0, 0, 0, 0, None),
        ([0, 0, 0], [0, 0, 0], 0, None, 0, 0, None),
        ([0, 0, 0], [1, 0, 1], 0, None, 0, 0, None),
        ([1, 1, 1], [1, 0, 1], 1, 2 / 3, 4 / 5, 5 / 7, 1),
        ([1, 1, 1], [0, 0, 0], 0, 0, 0, 0, None),
    ],
)
def test_zero_alerts_and_single_class_metrics(y, mask, precision, recall, f1, f2, reviews):
    metrics = evaluate_policy(y, mask)

    assert metrics["precision"] == precision
    assert metrics["recall"] == recall
    assert metrics["f1"] == f1
    assert metrics["f2"] == f2
    assert metrics["reviews_per_caught"] == reviews
    assert metrics["alarm_count"] == sum(mask)
    assert metrics["failure_support"] == sum(y)
    assert metrics["pass_support"] == len(y) - sum(y)
    assert metrics["recall_wilson_95"] == wilson_interval(
        sum(a * b for a, b in zip(y, mask, strict=True)), sum(y)
    )
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "y,mask", [([], []), ([[0, 1]], [0, 1]), ([0, 1], [1]), ([0, 2], [0, 1]), ([0, 1], [0, np.nan])]
)
def test_metrics_reject_invalid_labels_or_masks(y, mask):
    with pytest.raises(ValueError, match="binary"):
        evaluate_policy(y, mask)


@pytest.mark.parametrize("y", [[0], [[0, 1]], [0, 2], [0, np.nan]])
def test_threshold_selection_requires_aligned_binary_labels(y):
    with pytest.raises(ValueError, match="Labels"):
        choose_f2_threshold(y, [0, 1])


def test_threshold_selection_matches_hand_counted_f2_and_floor_capacity():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.arange(9, -1, -1)

    result = choose_f2_threshold(y, scores, 0.49)

    assert result["threshold"] == 7
    assert result["max_reviews"] == 4
    assert result["metrics"]["alarm_count"] == 3
    assert result["metrics"]["true_positives"] == 2
    assert result["metrics"]["f2"] == pytest.approx(10 / 15)
    assert result["metrics"] == evaluate_policy(y, scores >= 7)
    # Default capacity is 20%; the second row adds a false positive only.
    default = choose_f2_threshold(y, scores)
    assert default["max_reviews"] == 2
    assert default["threshold"] == 9


@pytest.mark.parametrize("negative_count,expected", [(195, 9.0), (196, 9.0), (197, -3.0)])
def test_f2_near_ties_and_exact_ties_use_integer_counts(negative_count, expected):
    # At 196 negatives both thresholds give exactly 5/6. One extra negative
    # makes the lower threshold better by ~0.00000345, inside np.isclose.
    y = np.r_[np.ones(200), np.zeros(negative_count), 1, np.zeros(5)]
    scores = np.r_[np.full(200 + negative_count, 9.0), np.full(6, -3.0)]
    high = Fraction(1000, 4 * 201 + 200 + negative_count)
    low = Fraction(1005, 4 * 201 + 206 + negative_count)
    assert np.isclose(float(high), float(low))
    assert (low > high) == (negative_count == 197)
    assert (low == high) == (negative_count == 196)

    result = choose_f2_threshold(y, scores, 1)

    assert result["threshold"] == expected
    assert result["metrics"]["f2"] == float(max(high, low))
    permutation = np.random.default_rng(3).permutation(len(y))
    assert choose_f2_threshold(y[permutation], scores[permutation], 1) == result


@pytest.mark.parametrize("score", [-np.finfo(float).max, -2.0, 0.0, 0.37, 2.0, np.finfo(float).max])
def test_constant_score_over_capacity_means_no_alerts_without_infinite_threshold(score):
    result = choose_f2_threshold([0, 0, 0, 0, 1], [score] * 5)

    assert result["threshold"] is None
    assert result["max_reviews"] == 1
    assert result["metrics"]["alarm_count"] == 0
    assert result["metrics"]["recall"] == 0
    assert result["metrics"]["reviews_per_caught"] is None
    json.dumps(result, allow_nan=False)
    allowed = choose_f2_threshold([0, 0, 0, 0, 1], [score] * 5, 1)
    assert allowed["threshold"] == score
    assert allowed["metrics"]["alarm_count"] == 5


def test_threshold_groups_are_atomic_and_zero_f2_prefers_no_alerts():
    assert choose_f2_threshold([1, 1, 0, 0, 0], [9, 9, 9, 0, -1], 0.4)["threshold"] is None
    assert choose_f2_threshold([0, 0, 0, 1, 1], [9, 8, 7, 6, 5], 0.4)["threshold"] is None
    assert choose_f2_threshold([0, 1], [0, 1], 0)["threshold"] is None
    assert choose_f2_threshold([0, 0], [-1, 8], 1)["threshold"] is None
    all_failures = choose_f2_threshold([1] * 10, np.arange(10))
    assert all_failures["threshold"] == 8
    assert all_failures["metrics"]["alarm_count"] == 2


@pytest.mark.parametrize("size", [13, 37, 100])
@pytest.mark.parametrize("fraction", [0, 0.2, 0.53, 1])
def test_f2_selection_matches_exhaustive_exact_rational_search(size, fraction):
    rng = np.random.default_rng(size)
    y = rng.integers(0, 2, size=size)
    scores = rng.integers(-5, 6, size=size).astype(float)
    capacity = floor(fraction * size)
    candidates = []
    for threshold in np.unique(scores):
        mask = scores >= threshold
        alerts = int(mask.sum())
        tp = int(y[mask].sum())
        if alerts <= capacity and tp:
            candidates.append((Fraction(5 * tp, 4 * int(y.sum()) + alerts), float(threshold)))
    expected = max(candidates)[1] if candidates else None

    result = choose_f2_threshold(y, scores, fraction)

    assert result == {
        "threshold": expected,
        "metrics": evaluate_policy(y, threshold_mask(scores, expected)),
        "max_reviews": capacity,
    }
    assert result["metrics"]["alarm_count"] <= capacity


def test_threshold_mask_accepts_any_finite_range_and_includes_equal_scores():
    scores = [-np.finfo(float).max, -20, 8, 8, np.finfo(float).max]

    np.testing.assert_array_equal(threshold_mask(scores, 8), [False, False, True, True, True])
    np.testing.assert_array_equal(threshold_mask(scores, None), [False] * 5)
    np.testing.assert_array_equal(threshold_mask(scores, -np.finfo(float).max), [True] * 5)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, [1], "bad", 1j, np.complex128(1j)])
def test_threshold_mask_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="threshold"):
        threshold_mask([0, 1], threshold)


def test_chronological_partition_returns_sorted_original_positions():
    dates = pd.date_range("2026-01-01", periods=10)
    permutation = np.array([9, 3, 0, 8, 1, 6, 2, 7, 5, 4])
    timestamps = dates[permutation]
    order = np.argsort(permutation, kind="stable")

    development, assessment = chronological_partition(timestamps)

    np.testing.assert_array_equal(development, order[:8])
    np.testing.assert_array_equal(assessment, order[8:])
    assert timestamps[development].max() < timestamps[assessment].min()
    half, rest = chronological_partition(timestamps, 0.5)
    np.testing.assert_array_equal(half, order[:5])
    np.testing.assert_array_equal(rest, order[5:])


@pytest.mark.parametrize(
    "counts,fraction,expected_development",
    [([2, 5, 3], 0.8, 7), ([7, 1, 1, 1], 0.8, 8), ([3, 4, 3], 0.5, 3)],
)
def test_partition_rounds_observation_counts_to_whole_timestamp_groups(
    counts, fraction, expected_development
):
    dates = np.repeat(pd.date_range("2026-01-01", periods=len(counts)).to_numpy(), counts)
    permutation = np.random.default_rng(22).permutation(len(dates))
    timestamps = dates[permutation]

    development, assessment = chronological_partition(timestamps, fraction)

    assert len(development) == expected_development
    assert set(permutation[development]) == set(range(expected_development))
    order = np.argsort(timestamps, kind="stable")
    np.testing.assert_array_equal(np.r_[development, assessment], order)
    assert timestamps[development].max() < timestamps[assessment].min()


def test_forward_splits_use_relative_indices_and_only_development_rows():
    dates = pd.date_range("2026-01-01", periods=25)
    permutation = np.random.default_rng(5).permutation(25)
    timestamps = dates[permutation]
    development, assessment = chronological_partition(timestamps)
    development_times = timestamps[development]
    # Shuffle development again: fold positions refer to this input, not global rows.
    relative_order = np.random.default_rng(8).permutation(20)
    development, development_times = development[relative_order], development_times[relative_order]
    chronological = np.argsort(development_times, kind="stable")

    splits = forward_splits(development_times)

    assert len(splits) == 3
    for (train, valid), end in zip(splits, [8, 12, 16], strict=True):
        np.testing.assert_array_equal(train, chronological[:end])
        np.testing.assert_array_equal(valid, chronological[end : end + 4])
        assert development_times[train].max() < development_times[valid].min()
        assert not set(train) & set(valid)
        assert not set(development[np.r_[train, valid]]) & set(assessment)


@pytest.mark.parametrize(
    "counts,expected_cuts", [([3, 4, 5, 4, 4], [7, 12, 16]), ([17, 1, 1, 1], [17, 18, 19])]
)
def test_forward_splits_keep_timestamp_ties_whole_and_validation_windows_nonempty(
    counts, expected_cuts
):
    dates = np.repeat(pd.date_range("2026-01-01", periods=len(counts)).to_numpy(), counts)
    permutation = np.random.default_rng(17).permutation(len(dates))
    timestamps = dates[permutation]
    order = np.argsort(timestamps, kind="stable")
    splits = forward_splits(timestamps)

    assert len(splits) == 3
    stops = [*expected_cuts[1:], len(dates)]
    validations = []
    for (train, valid), cut, stop in zip(splits, expected_cuts, stops, strict=True):
        np.testing.assert_array_equal(train, order[:cut])
        np.testing.assert_array_equal(valid, order[cut:stop])
        assert len(train) and len(valid)
        assert timestamps[train].max() < timestamps[valid].min()
        assert not set(train) & set(valid)
        validations.extend(valid)
    assert len(validations) == len(set(validations))


def test_chronological_ties_are_compared_as_utc_instants():
    timestamps = ["2026-01-01T02:00:00+02:00", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]

    development, assessment = chronological_partition(timestamps, 0.5)

    np.testing.assert_array_equal(development, [0, 1])
    np.testing.assert_array_equal(assessment, [2])


@pytest.mark.parametrize(
    "timestamps",
    [
        [],
        [["2026-01-01", "2026-01-02"]],
        ["bad", "2026-01-01"],
        [None, "2026-01-01"],
        [np.datetime64("NaT", "ns")],
        [1, 2, 3, 4],
    ],
)
def test_time_apis_reject_invalid_timestamps(timestamps):
    with pytest.raises(ValueError, match="Timestamps"):
        chronological_partition(timestamps)
    with pytest.raises(ValueError, match="Timestamps"):
        forward_splits(timestamps)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, np.nan, np.inf, [0.8]])
def test_partition_requires_an_internal_finite_fraction(fraction):
    with pytest.raises(ValueError, match="development_fraction"):
        chronological_partition(pd.date_range("2026-01-01", periods=5), fraction)


def test_time_apis_reject_too_few_distinct_times_for_nonempty_splits():
    with pytest.raises(ValueError, match="two distinct"):
        chronological_partition(["2026-01-01"] * 5)
    with pytest.raises(ValueError, match="four distinct"):
        forward_splits(pd.date_range("2026-01-01", periods=3))

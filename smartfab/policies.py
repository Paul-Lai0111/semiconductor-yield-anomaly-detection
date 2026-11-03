"""Batch review policies and time splits without learning from assessment labels.

The primary policy is top-k review at a 20% batch capacity. F2 threshold selection
is a secondary policy fitted on development out-of-fold scores only. Scores may
have any finite range; neither policy assumes calibrated probabilities.
"""

from hashlib import sha256
from math import floor
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from secom.evaluation import wilson_interval


def _score_vector(scores) -> np.ndarray:
    message = "Scores must be a nonempty one-dimensional array of finite real numbers."
    try:
        if np.iscomplexobj(scores):
            raise ValueError(message)
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(message)
    return values


def _fraction(value, name: str) -> float:
    message = f"{name} must be a finite scalar in [0, 1]."
    try:
        if np.ndim(value) != 0 or np.iscomplexobj(value):
            raise ValueError(message)
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(message)
    return value


def _binary_vector(values, name: str, size: int | None = None) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not values.size
        or (size is not None and values.size != size)
        or not np.isin(values, [0, 1]).all()
    ):
        raise ValueError(f"{name} must be nonempty, aligned, one-dimensional binary values.")
    return values.astype(bool)


def capacity_mask(scores, row_ids, fraction, seed=2026) -> np.ndarray:
    """Select exactly floor(fraction * n) rows, highest scores first.

    Ties use ascending SHA-256 of the ASCII string ``seed:row_id``, then row ID
    in the event of a hash collision. Integer IDs identify rows across input
    permutations. No labels enter this decision, including for constant scores.
    """
    scores = _score_vector(scores)
    fraction = _fraction(fraction, "fraction")
    row_ids = np.asarray(row_ids, dtype=object)
    if (
        row_ids.ndim != 1
        or row_ids.size != scores.size
        or any(
            not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            for value in row_ids
        )
    ):
        raise ValueError("Row IDs must be aligned one-dimensional unique integers.")
    ids = [int(value) for value in row_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be aligned one-dimensional unique integers.")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer.")
    hashes = [sha256(f"{int(seed)}:{row_id}".encode("ascii")).digest() for row_id in ids]
    order = sorted(range(scores.size), key=lambda i: (-scores[i], hashes[i], ids[i]))
    mask = np.zeros(scores.size, dtype=bool)
    mask[order[: floor(fraction * scores.size)]] = True
    return mask


def batch_ranks(scores) -> np.ndarray:
    """Return average ranks / n within this batch, with larger scores ranked higher."""
    scores = _score_vector(scores)
    return rankdata(scores, method="average") / scores.size


def blend_scores(members) -> np.ndarray:
    """Average ranks before normalizing so equal rank sums remain exact score ties."""
    arrays = [_score_vector(values) for values in members]
    if len(arrays) < 2 or any(len(values) != len(arrays[0]) for values in arrays):
        raise ValueError("A rank blend needs at least two aligned score vectors.")
    ranks = np.sum([rankdata(values, method="average") for values in arrays], axis=0)
    return ranks / (len(arrays) * len(arrays[0]))


def evaluate_policy(y, mask) -> dict:
    """Summarize binary failure labels (1) and review decisions.

    Precision and F scores use zero for zero denominators. Recall and its Wilson
    interval are None without failures; reviews_per_caught is None without a TP.
    """
    y = _binary_vector(y, "Labels")
    mask = _binary_vector(mask, "Mask", y.size)
    tp = int(np.count_nonzero(y & mask))
    fn = int(np.count_nonzero(y & ~mask))
    fp = int(np.count_nonzero(~y & mask))
    tn = int(np.count_nonzero(~y & ~mask))
    positives, alerts = tp + fn, tp + fp
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (positives + alerts) if positives + alerts else 0.0,
        "f2": 5 * tp / (4 * positives + alerts) if 4 * positives + alerts else 0.0,
        "alarm_count": alerts,
        "alarm_rate": alerts / y.size,
        "reviews_per_caught": alerts / tp if tp else None,
        "failure_support": positives,
        "pass_support": tn + fp,
        "recall_wilson_95": wilson_interval(tp, positives),
    }


def threshold_mask(scores, threshold) -> np.ndarray:
    """Review scores >= a finite threshold; None explicitly means no alerts."""
    scores = _score_vector(scores)
    if threshold is None:
        return np.zeros(scores.size, dtype=bool)
    try:
        if np.ndim(threshold) != 0 or np.iscomplexobj(threshold):
            raise ValueError
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("threshold must be a finite scalar or None.") from None
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite scalar or None.")
    return scores >= threshold


def choose_f2_threshold(y, scores, max_alert_fraction=0.2) -> dict:
    """Fit a secondary threshold using development OOF labels and scores only.

    Whole observed score groups must fit floor(max_alert_fraction * n) reviews.
    Exact F2 ties retain the higher threshold. If no legal threshold catches a
    failure, return None (no alerts), including when a constant score exceeds
    capacity. The result contains threshold, metrics, and max_reviews.
    """
    scores = _score_vector(scores)
    y = _binary_vector(y, "Labels", scores.size)
    fraction = _fraction(max_alert_fraction, "max_alert_fraction")
    max_reviews = floor(fraction * scores.size)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    cumulative_tp = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), scores.size - 1]
    positives = int(np.count_nonzero(y))
    best_numerator, best_denominator = 0, 1
    threshold = None
    for end in ends:
        alerts = int(end) + 1
        if alerts > max_reviews:
            break
        numerator = 5 * int(cumulative_tp[end])
        denominator = 4 * positives + alerts
        # Python integer cross-products distinguish near ties and preserve exact ties.
        if numerator * best_denominator > best_numerator * denominator:
            best_numerator, best_denominator = numerator, denominator
            threshold = float(ordered[end])
    return {
        "threshold": threshold,
        "metrics": evaluate_policy(y, threshold_mask(scores, threshold)),
        "max_reviews": max_reviews,
    }


def _ordered_time_groups(timestamps) -> tuple[np.ndarray, np.ndarray]:
    message = "Timestamps must be a nonempty one-dimensional array of valid datetimes."
    values = np.asarray(timestamps)
    if (
        values.ndim != 1
        or not values.size
        or any(isinstance(value, (Real, np.bool_)) for value in values)
    ):
        raise ValueError(message)
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, format="mixed", errors="raise", utc=True))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if dates.isna().any():
        raise ValueError(message)
    order = np.argsort(dates.asi8, kind="stable")
    ordered = dates.asi8[order]
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    return order, boundaries


def chronological_partition(timestamps, development_fraction=0.8) -> tuple[np.ndarray, np.ndarray]:
    """Return (development, assessment) original positions, stably sorted by time.

    Round floor(development_fraction * n) to the nearest internal time-group
    boundary, preferring the earlier boundary on equal distances. Both parts
    must be nonempty; tied timestamps never straddle the boundary.
    """
    fraction = _fraction(development_fraction, "development_fraction")
    if not 0 < fraction < 1:
        raise ValueError("development_fraction must be strictly between 0 and 1.")
    order, boundaries = _ordered_time_groups(timestamps)
    if not boundaries.size:
        raise ValueError("Chronological partition needs at least two distinct timestamps.")
    target = floor(fraction * order.size)
    cut = int(boundaries[np.argmin(np.abs(boundaries - target))])
    return order[:cut], order[cut:]


def forward_splits(timestamps) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return three expanding (train, validation) splits relative to the input.

    Supply development timestamps only. Target train ends are 40%, 60%, and
    80% of observations, with the next 20% for validation. Round each target
    down to a row count, then to the nearest time-group boundary (earlier on
    ties), reserving enough later groups for three nonempty validation windows.
    Input order is stable within ties; every train time precedes its validation.
    """
    order, boundaries = _ordered_time_groups(timestamps)
    if boundaries.size < 3:
        raise ValueError("Forward splits need at least four distinct timestamps.")
    cuts = []
    first_boundary = 0
    for step, numerator in enumerate((2, 3, 4)):
        candidates = boundaries[first_boundary : boundaries.size - 2 + step]
        target = numerator * order.size // 5
        choice = int(np.argmin(np.abs(candidates - target)))
        cuts.append(int(candidates[choice]))
        first_boundary += choice + 1
    cuts.append(order.size)
    return [(order[: cuts[i]], order[cuts[i] : cuts[i + 1]]) for i in range(3)]

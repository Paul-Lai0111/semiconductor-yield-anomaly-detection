"""Threshold policies and uncertainty summaries with explicit selection boundaries."""

from math import sqrt

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y, scores) -> float:
    """Maximize development OOF F1; choose the highest threshold on exact ties."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires finite scores and both label classes.")
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("Labels and scores must be aligned one-dimensional arrays.")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    cumulative_tp = np.cumsum(y[order])
    boundaries = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    positives = int(y.sum())
    best_numerator, best_denominator = -1, 1
    best_threshold = float(ordered[0])
    for end in boundaries:
        numerator = 2 * int(cumulative_tp[end])
        denominator = int(end) + 1 + positives
        # Integer cross-products preserve true F1 ties despite floating-point rounding.
        # Descending score order means retaining a tie keeps the highest threshold.
        if numerator * best_denominator > best_numerator * denominator:
            best_numerator, best_denominator = numerator, denominator
            best_threshold = float(ordered[end])
    return best_threshold


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total == 0:
        return None
    z = float(norm.ppf((1 + confidence) / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def operating_metrics(y, scores, threshold: float) -> dict:
    y = np.asarray(y)
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "alarm_count": tp + fp,
        "alarm_rate": float(predicted.mean()),
        "failure_support": tp + fn,
        "pass_support": tn + fp,
        "pass_precision": tn / (tn + fn) if tn + fn else 0.0,
        "pass_recall": tn / (tn + fp) if tn + fp else 0.0,
        "pass_f1": 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0,
        "recall_wilson_95": wilson_interval(tp, tp + fn),
    }


def ranking_metrics(y, scores) -> dict:
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "failure_prevalence": float(np.asarray(y).mean()),
    }


def bootstrap_ap(y, scores, repetitions: int, seed: int) -> list[float]:
    """Fixed-score, class-stratified bootstrap; conditional on observed prevalence."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(negative) or not len(positive) or repetitions < 1:
        raise ValueError("Bootstrap needs both classes and at least one repetition.")
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        rows = np.concatenate(
            (
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            )
        )
        estimates[index] = average_precision_score(y[rows], scores[rows])
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]

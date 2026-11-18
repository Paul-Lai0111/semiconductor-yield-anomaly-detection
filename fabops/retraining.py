"""Prequential (test-then-train) simulation of retraining policies.

The whole timeline is replayed in chronological batches. For each batch:

1. Score it with the model currently in production (the model has never
   seen this batch).
2. Review the top ``review_fraction`` of the batch and count failures caught.
3. Decide whether to retrain before the next batch, using only information
   available at that moment:
   * sensor values of every batch scored so far (available immediately);
   * labels only for batches at least ``label_delay`` batches old
     (pass/fail results arrive later than sensor data in a fab).

Policies:
* ``frozen``           - never retrain after the initial fit.
* ``periodic``         - retrain every ``period`` batches.
* ``drift_triggered``  - retrain when the share of informative sensors with
                         PSI > 0.25 versus the current model's training window
                         reaches ``drift_threshold``.
* ``always``           - retrain after every batch (cost upper bound).

Retraining always uses an expanding window of all labeled rows available.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from fabops.drift import DriftReference, drift_share, informative_sensors, sensor_drift
from smartfab.policies import capacity_mask, evaluate_policy

POLICIES = ("frozen", "periodic", "drift_triggered", "always")


@dataclass(frozen=True)
class RetrainConfig:
    batch_size: int = 100
    n_batches: int = 11
    label_delay: int = 1
    review_fraction: float = 0.20
    period: int = 3
    drift_threshold: float = 0.10
    bins: int = 10
    seed: int = 2026

    def __post_init__(self):
        if self.batch_size < 10 or self.n_batches < 2 or self.label_delay < 0 or self.period < 1:
            raise ValueError("Invalid retraining configuration.")


def batch_layout(n_rows: int, config: RetrainConfig) -> tuple[int, list[tuple[int, int]]]:
    """Initial training rows and (start, end) positions of each scored batch."""
    initial = n_rows - config.batch_size * config.n_batches
    if initial < config.batch_size:
        raise ValueError("Too few rows for the initial training window.")
    return initial, [
        (initial + b * config.batch_size, initial + (b + 1) * config.batch_size)
        for b in range(config.n_batches)
    ]


def labeled_end(batch: int, initial: int, batches, label_delay: int) -> int:
    """Exclusive chronological position up to which labels are known after scoring ``batch``."""
    last_labeled = batch - label_delay
    return initial if last_labeled < 0 else batches[last_labeled][1]


def simulate(
    X: pd.DataFrame,
    y: np.ndarray,
    timestamps,
    fit: Callable,
    score: Callable,
    config: RetrainConfig | None = None,
    policies=POLICIES,
) -> pd.DataFrame:
    """Replay the timeline for each policy; return one row per (policy, batch)."""
    config = config or RetrainConfig()
    order = np.argsort(pd.DatetimeIndex(timestamps).asi8, kind="stable")
    Xo, yo = X.iloc[order].reset_index(drop=True), np.asarray(y)[order]
    row_ids = order  # original positions remain the stable tie-break IDs
    initial, batches = batch_layout(len(order), config)
    rows = []
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy}.")
        train_end = initial
        model = fit(Xo.iloc[:train_end], yo[:train_end])
        reference = DriftReference.fit(Xo.iloc[:train_end], config.bins)
        informative = informative_sensors(Xo.iloc[:train_end])
        retrains = 0
        for b, (start, end) in enumerate(batches):
            batch_X, batch_y = Xo.iloc[start:end], yo[start:end]
            scores = score(model, batch_X)
            mask = capacity_mask(scores, row_ids[start:end], config.review_fraction, config.seed)
            result = evaluate_policy(batch_y, mask)
            share = drift_share(sensor_drift(reference, batch_X), informative)
            available = labeled_end(b, initial, batches, config.label_delay)
            wants = {
                "frozen": False,
                "periodic": (b + 1) % config.period == 0,
                "drift_triggered": share >= config.drift_threshold,
                "always": True,
            }[policy]
            retrained = bool(wants and available > train_end and b < len(batches) - 1)
            rows.append(
                {
                    "policy": policy,
                    "batch": b,
                    "first_timestamp": pd.DatetimeIndex(timestamps)[order[start]].isoformat(),
                    "rows": end - start,
                    "failures": int(batch_y.sum()),
                    "caught": result["true_positives"],
                    "reviews": result["alarm_count"],
                    "train_rows_used": train_end,
                    "drift_share_vs_training": share,
                    "retrain_requested": bool(wants),
                    "retrained_after_batch": retrained,
                }
            )
            if retrained:
                train_end = available
                model = fit(Xo.iloc[:train_end], yo[:train_end])
                reference = DriftReference.fit(Xo.iloc[:train_end], config.bins)
                informative = informative_sensors(Xo.iloc[:train_end])
                retrains += 1
    return pd.DataFrame(rows)


def summarize(batches: pd.DataFrame, baseline: str = "frozen", group: str = "policy") -> pd.DataFrame:
    """Totals per policy plus a paired per-batch comparison against ``baseline``."""
    keys = [c for c in ("model", group) if c in batches]
    out = []
    for key, part in batches.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key, strict=True))
        base = batches
        for column in keys[:-1]:
            base = base[base[column] == record[column]]
        base = base[base[group] == baseline].set_index("batch")
        diff = part.set_index("batch").caught - base.caught
        wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
        record.update(
            failures=int(part.failures.sum()),
            caught=int(part.caught.sum()),
            reviews=int(part.reviews.sum()),
            recall=float(part.caught.sum() / max(part.failures.sum(), 1)),
            retrains=int(part.retrained_after_batch.sum()),
            caught_minus_baseline=int(diff.sum()),
            batches_better=wins,
            batches_worse=losses,
            sign_test_p=float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
        )
        out.append(record)
    return pd.DataFrame(out)

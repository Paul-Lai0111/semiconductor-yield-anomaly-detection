import numpy as np
import pandas as pd
import pytest

from fabops.drift import (
    DriftReference,
    benjamini_hochberg,
    drift_share,
    expected_null_psi,
    informative_sensors,
    psi,
    sensor_drift,
)
from fabops.retraining import RetrainConfig, batch_layout, labeled_end, simulate, summarize


def frame(n, shift=0.0, seed=0, p=20):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, p))
    values[:, : p // 2] += shift
    return pd.DataFrame(values, columns=[f"s{i}" for i in range(p)])


def test_psi_zero_for_identical_and_large_for_shift():
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0
    reference = DriftReference.fit(frame(2000))
    same = sensor_drift(reference, frame(2000, seed=1))
    moved = sensor_drift(reference, frame(2000, shift=1.5, seed=1))
    assert same.psi.max() < 0.05
    assert moved.psi[:10].min() > 0.25 and moved.psi[10:].max() < 0.05
    assert drift_share(moved) == pytest.approx(0.5)
    assert moved.ks_alert[:10].all()


def test_missing_values_drift_through_missing_bin():
    base = frame(1000)
    later = frame(1000, seed=2)
    later.iloc[:400, 0] = np.nan
    table = sensor_drift(DriftReference.fit(base), later)
    assert table.loc[0, "psi_alert"] and table.loc[0, "missing_rate_change"] == pytest.approx(0.4)


def test_null_psi_approximation_order_of_magnitude():
    reference = DriftReference.fit(frame(500, p=200))
    noise = sensor_drift(reference, frame(100, seed=3, p=200)).psi.mean()
    assert 0.5 * expected_null_psi(11, 500, 100) < noise < 2 * expected_null_psi(11, 500, 100)


def test_benjamini_hochberg_and_informative():
    q = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.isnan(q[3]) and q[0] <= q[2] <= q[1] <= 1
    X = frame(50)
    X["constant"] = 3.0
    assert not informative_sensors(X)[-1]
    with pytest.raises(ValueError):
        sensor_drift(DriftReference.fit(X), frame(50))


def test_label_delay_never_uses_unscored_labels():
    config = RetrainConfig(batch_size=50, n_batches=4, label_delay=1)
    initial, batches = batch_layout(400, config)
    assert initial == 200 and batches[0] == (200, 250)
    assert labeled_end(0, initial, batches, 1) == 200
    assert labeled_end(2, initial, batches, 1) == 300
    assert labeled_end(2, initial, batches, 0) == 350


def test_simulation_policies_and_training_windows():
    n = 400
    X = frame(n)
    rng = np.random.default_rng(0)
    y = (X.s0 + rng.normal(scale=0.5, size=n) > 1.2).astype(int).to_numpy()
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    seen = []

    def fit(Xt, yt):
        seen.append(len(Xt))
        return float(Xt.s0.mean())

    def score(model, Xb):
        return Xb.s0.to_numpy() - model

    config = RetrainConfig(batch_size=50, n_batches=4, period=2, drift_threshold=0.0)
    batches = simulate(X, y, times, fit, score, config)
    by = batches.groupby("policy")
    assert by.retrained_after_batch.sum().to_dict() == {
        "always": 2, "drift_triggered": 2, "frozen": 0, "periodic": 1,
    }
    # Training windows only ever contain labels from batches at least one batch old.
    for _, part in by:
        for row in part.itertuples():
            assert row.train_rows_used <= 200 + 50 * max(row.batch - 1, 0)
    summary = summarize(batches)
    assert set(summary.policy) == {"frozen", "periodic", "drift_triggered", "always"}
    assert (summary.loc[summary.policy.eq("frozen"), "caught_minus_baseline"] == 0).all()
    assert (summary.reviews == 4 * 10).all()

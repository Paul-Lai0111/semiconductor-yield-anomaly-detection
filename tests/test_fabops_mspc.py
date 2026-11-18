import numpy as np
import pandas as pd
import pytest

from fabops.mspc import MSPCModel, spe_limit, t2_limit, top_contributors


def correlated(n=300, p=12, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, p))
    values = latent @ loadings + 0.1 * rng.normal(size=(n, p))
    return pd.DataFrame(values, columns=[f"sensor_{i:03d}" for i in range(p)])


def test_failures_are_excluded_from_fit():
    X = correlated()
    y = np.zeros(len(X), dtype=int)
    y[:50] = 1
    model = MSPCModel(variance=0.9).fit(X, y)
    assert model.n_train_ == 250


def test_t2_matches_mahalanobis_in_score_space():
    X = correlated()
    model = MSPCModel(variance=0.95).fit(X)
    Z = model._standardize(X)
    scores = model.pca_.transform(Z)
    expected = np.sum(scores**2 / model.pca_.explained_variance_, axis=1)
    np.testing.assert_allclose(model.statistics(X).t2, expected, rtol=1e-8)


def test_contributions_sum_to_statistics():
    X = correlated()
    model = MSPCModel(variance=0.9).fit(X)
    stats = model.statistics(X)
    np.testing.assert_allclose(model.contributions(X, "spe").sum(axis=1), stats.spe, rtol=1e-8)
    np.testing.assert_allclose(model.contributions(X, "t2").sum(axis=1), stats.t2, rtol=1e-8)
    assert (model.contributions(X, "spe").to_numpy() >= 0).all()


def test_broken_correlation_raises_spe_and_points_to_sensor():
    X = correlated()
    model = MSPCModel(variance=0.95).fit(X)
    faulty = X.iloc[[0]].copy()
    faulty["sensor_005"] += 8 * X["sensor_005"].std()
    base = model.statistics(X.iloc[[0]])
    after = model.statistics(faulty)
    assert after.spe[0] > 10 * base.spe[0]
    assert after.spe[0] > model.spe_limit_
    top = top_contributors(model.contributions(faulty, "spe"), 1)[0][0][0]
    assert top == "sensor_005"


def test_in_control_false_alarm_rate_near_alpha():
    X = correlated(n=2000, seed=1)
    model = MSPCModel(variance=0.95, alpha=0.99).fit(X.iloc[:1000])
    rate = model.alarms(X.iloc[1000:]).alarm.mean()
    assert rate < 0.06


def test_limits_and_validation():
    assert t2_limit(3, 100, 0.99) > t2_limit(3, 100, 0.95) > 0
    assert spe_limit(np.random.default_rng(0).chisquare(5, 1000), 0.99) > 5
    with pytest.raises(ValueError):
        MSPCModel().fit(correlated(n=5))
    model = MSPCModel().fit(correlated())
    with pytest.raises(ValueError):
        model.statistics(correlated().iloc[:, ::-1])
    with pytest.raises(ValueError):
        model.statistics(correlated()).score("bogus")


def test_missing_values_and_constant_sensors():
    X = correlated()
    X["sensor_999"] = 1.0
    X.iloc[::7, 2] = np.nan
    model = MSPCModel().fit(X)
    assert "sensor_999" not in model.feature_names_
    assert np.isfinite(model.statistics(X).combined).all()

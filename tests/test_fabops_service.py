import io

import numpy as np
import pandas as pd
import pytest

from fabops.drift import DriftReference, informative_sensors
from fabops.mspc import MSPCModel
from fabops.service import SCHEMA, batch_drift, score_batch, validate_batch
from smartfab.models import fit_candidate

SPEC = {"id": "knn_test", "family": "knn", "params": {"k": 5, "n_neighbors": 5, "weights": "distance"}, "novelty": False}


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"sensor_{i:03d}" for i in range(8)])
    y = (X.sensor_000 + 0.3 * rng.normal(size=200) > 1.3).astype(int).to_numpy()
    return {
        "schema": SCHEMA,
        "sensor_columns": X.columns.tolist(),
        "knn_spec": SPEC,
        "knn": fit_candidate(SPEC, X, y),
        "mspc": MSPCModel(variance=0.8).fit(X, y),
        "mspc_statistic": "spe",
        "drift_reference": DriftReference.fit(X),
        "informative": informative_sensors(X),
        "seed": 2026,
        "default_review_fraction": 0.2,
        "watchlist": [{"sensor": "sensor_000", "tier": "stable", "consensus_freq": 0.9}],
        "provenance": {"v3_winner": "knn_test", "training_rows": 200, "training_last_timestamp": "2008-10-02"},
    }


def batch(n=50, seed=1, shift=0.0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 8)) + shift, columns=[f"sensor_{i:03d}" for i in range(8)])
    frame.insert(0, "unit_id", np.arange(1000, 1000 + n))
    return frame


def test_score_batch_builds_exact_queue(bundle):
    queue = score_batch(bundle, batch(), 0.2, "v3_selected_knn")
    assert queue.review.sum() == 10
    assert queue.queue_rank.tolist() == list(range(1, 51))
    assert queue.loc[queue.review.eq(1), "priority_score"].min() >= queue.loc[queue.review.eq(0), "priority_score"].max()
    for policy in ("mspc", "hybrid_knn_mspc"):
        assert score_batch(bundle, batch(), 0.1, policy).review.sum() == 5


def test_row_order_does_not_change_decisions(bundle):
    first = score_batch(bundle, batch()).set_index("unit_id").sort_index()
    shuffled = batch().sample(frac=1, random_state=3)
    second = score_batch(bundle, shuffled).set_index("unit_id").sort_index()
    pd.testing.assert_series_equal(first.review, second.review)


def test_validation_rejects_bad_batches(bundle):
    with pytest.raises(ValueError, match="columns"):
        validate_batch(bundle, batch().drop(columns="sensor_003"))
    with pytest.raises(ValueError, match="columns"):
        validate_batch(bundle, batch().assign(label=0))
    bad = batch()
    bad.loc[0, "unit_id"] = bad.loc[1, "unit_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_batch(bundle, bad)
    text = batch().astype(object)
    text.loc[0, "sensor_001"] = "abc"
    with pytest.raises(ValueError, match="numbers"):
        validate_batch(bundle, text)
    empty_row = batch()
    empty_row.iloc[0, 1:] = np.nan
    with pytest.raises(ValueError, match="observed"):
        validate_batch(bundle, empty_row)
    with pytest.raises(ValueError):
        score_batch(bundle, batch(), 1.5)


def test_drift_warning(bundle):
    assert not batch_drift(bundle, batch(200))["warning"]
    assert batch_drift(bundle, batch(200, shift=2.0))["warning"]


def test_api_endpoints(bundle):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fabops.api import create_app

    client = TestClient(create_app(bundle))
    assert client.get("/health").json()["bundle_loaded"]
    assert client.get("/model").json()["v3_winner"] == "knn_test"
    records = batch(20).replace({np.nan: None}).to_dict(orient="records")
    response = client.post("/score?review_fraction=0.25&policy=mspc", json={"records": records})
    assert response.status_code == 200
    body = response.json()
    assert body["reviews"] == 5 and len(body["queue"]) == 20
    csv = batch(20).to_csv(index=False).encode()
    response = client.post("/score/csv", files={"file": ("batch.csv", io.BytesIO(csv), "text/csv")})
    assert response.status_code == 200 and response.json()["reviews"] == 4
    assert client.post("/score", json={"records": [{"unit_id": 1}]}).status_code == 422
    assert client.post("/score?policy=bogus", json={"records": records}).status_code == 422
    assert "psi_alert_share" in client.post("/drift", json={"records": records}).json()


def test_api_without_bundle_is_degraded(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fabops.api import create_app

    monkeypatch.setenv("FABOPS_BUNDLE", str(tmp_path / "missing.joblib"))
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/model").status_code == 503

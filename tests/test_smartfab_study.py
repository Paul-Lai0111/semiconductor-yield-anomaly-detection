"""Regression checks for temporal search, the freeze boundary, and tied rank blends."""

import json
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import smartfab.reporting as reporting
import smartfab.study as study
from secom.data import Dataset
from smartfab.models import CANDIDATES
from smartfab.policies import blend_scores, capacity_mask, chronological_partition


def test_opposite_rankings_produce_exact_ties_before_hash_tie_break():
    scores = blend_scores([np.arange(12), np.arange(12)[::-1]])
    assert np.unique(scores).size == 1
    ids = np.arange(12)
    np.testing.assert_array_equal(
        capacity_mask(scores, ids, 0.2), capacity_mask(np.ones(12), ids, 0.2)
    )


def test_search_freezes_before_assessment_and_preserves_temporal_membership(tmp_path, monkeypatch):
    values, labels = make_classification(
        n_samples=250, n_features=12, n_informative=5, weights=[0.7, 0.3], random_state=9
    )
    X = pd.DataFrame(values, columns=[f"sensor_{i:03d}" for i in range(12)])
    dataset = Dataset(X, labels, pd.date_range("2008-01-01", periods=250), {})
    development, assessment = chronological_partition(dataset.timestamps)
    output = tmp_path / "chronological_study"
    monkeypatch.setattr(study, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(reporting, "generate_report", lambda _: None)
    original_fit = study.scored_fit
    fits = []

    def observe_fit(spec, train_X, train_y, target, *args):
        assert set(train_X.index) <= set(development)
        assert not set(train_X.index).intersection(target.index)
        if set(target.index).intersection(assessment):
            frozen = json.loads((output / "selection.json").read_text())
            assert frozen["winner"] and frozen["members"]
            assert (
                frozen["protocol_sha256"]
                == sha256((output / "protocol.json").read_bytes()).hexdigest()
            )
            assert set(target.index) == set(assessment)
        fits.append(set(train_X.index))
        return original_fit(spec, train_X, train_y, target, *args)

    monkeypatch.setattr(study, "scored_fit", observe_fit)
    chosen = {"v2_rf_knn_smote", "rf_leaf2_balanced", "svc_c1_gamma_scale", "dummy_prior"}
    catalog = [deepcopy(spec) for spec in CANDIDATES if spec["id"] in chosen]
    for spec in catalog:
        if "n_estimators" in spec["params"]:
            spec["params"]["n_estimators"] = 2
    result = study.run_study(
        study.SearchConfig(bootstrap=10), tmp_path / "unused", output, catalog=catalog
    )
    assert result["assessment_rows"] == len(assessment)
    assert result["last_training_timestamp"] < result["first_assessment_timestamp"]
    assert result["assessment"]["selected"]["capacity"]["alarm_count"] == int(len(assessment) * 0.2)
    assert len(fits) >= len(catalog) * 3 + 15
    summary = pd.read_csv(output / "search_summary.csv")
    assert result["winner"] == summary.iloc[0].candidate
    assert len(summary) == len(catalog) + 1
    forward = pd.read_csv(output / "forward_predictions.csv")
    assert set(forward.row_id) <= set(development)
    assert not forward.duplicated(["candidate", "row_id"]).any()
    for _, group in forward.groupby("candidate"):
        assert len(group) == len(forward.loc[forward.candidate == summary.candidate.iloc[0]])
    oof = pd.read_csv(output / "finalist_oof.csv")
    assert oof.groupby(["candidate", "row_id"]).size().eq(1).all()
    assert set(oof.row_id) == set(development)
    scenarios = pd.read_csv(output / "cost_scenarios.csv")
    assert np.isfinite(scenarios.simulated_total_cost_units).all()
    assert (
        scenarios.loc[scenarios.strategy == "random_same_capacity_expected", "difference_vs_random"]
        == 0
    ).all()
    assert (output / "artifacts/selected_model.joblib").is_file()
    np.testing.assert_array_equal(X, values)

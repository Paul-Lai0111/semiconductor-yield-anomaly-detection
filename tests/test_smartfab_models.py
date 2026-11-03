"""Regressions for training-only features, novelty fitting, and failure scores."""

import copy
import json
import warnings
from collections import Counter
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from threadpoolctl import threadpool_limits

from secom.modeling import make_pipeline as make_v2_pipeline
from smartfab.models import (
    CANDIDATES,
    FoldANOVA,
    NativeSensorMapper,
    SensorPruner,
    fit_candidate,
    make_pipeline,
    predict_score,
)


def small_spec(identifier):
    spec = copy.deepcopy(next(spec for spec in CANDIDATES if spec["id"] == identifier))
    if "n_estimators" in spec["params"]:
        spec["params"]["n_estimators"] = 3
    if spec["family"] == "histgb":
        spec["params"]["max_iter"] = 5
    return spec


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):
        yield


@pytest.fixture
def training_data():
    rng = np.random.default_rng(53)
    y = np.array([0] * 60 + [1] * 20)
    X = pd.DataFrame(
        rng.normal(size=(80, 8)),
        columns=[3, 10, 21, 144, 305, 589, 500, 501],
        index=[f"wafer_{i}" for i in range(80)],
    )
    X[3] += 3 * y
    X[10] *= 1200
    X.loc[X.index[::4], 10] = np.nan
    X.loc[X.index[::3], 21] = np.nan
    X[144] = 7.0
    X[305] = np.nan
    X[589] = 8.0
    X.loc[X.index[::2], 589] = np.nan
    return X, y


def test_catalog_contains_the_explicit_grids_and_serializable_unique_specs():
    counts = {
        "v2": 1,
        "rf": 4,
        "extra_trees": 6,
        "balanced_rf": 2,
        "xgb": 8,
        "histgb": 4,
        "svc": 6,
        "linear_svc": 3,
        "logistic": 6,
        "knn": 4,
        "lda": 1,
        "gaussian_nb": 2,
        "isolation_forest": 1,
        "one_class_svm": 1,
        "dummy": 1,
    }
    # The enumerated requirements sum to 50, including v2 and the prior dummy.
    assert len(CANDIDATES) == sum(counts.values()) == 50
    assert Counter(spec["family"] for spec in CANDIDATES) == counts
    assert len({spec["id"] for spec in CANDIDATES}) == len(CANDIDATES)
    assert len({json.dumps((s["family"], s["params"]), sort_keys=True) for s in CANDIDATES}) == 50
    assert json.loads(json.dumps(CANDIDATES)) == list(CANDIDATES)
    for spec in CANDIDATES:
        assert set(spec) == {"id", "family", "params", "novelty"}
        assert isinstance(spec["novelty"], bool)
        assert spec["novelty"] == (spec["family"] in ("isolation_forest", "one_class_svm"))
        if "n_estimators" in spec["params"]:
            assert spec["params"]["n_estimators"] == 300
        assert ("smote" in make_pipeline(spec).named_steps) == (spec["family"] == "v2")


@pytest.mark.parametrize("identifier", [spec["id"] for spec in CANDIDATES])
def test_every_candidate_fits_scores_and_serializes(training_data, identifier):
    X, y = training_data
    original = X.copy(deep=True)
    catalog_before = copy.deepcopy(CANDIDATES)
    spec = small_spec(identifier)
    pipeline = fit_candidate(spec, X, y, seed=23, jobs=1)
    query = X.iloc[[0, 7, 61, 79]].copy()
    scores = predict_score(spec, pipeline, query)

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert pipeline.n_features_in_ == 8
    assert pipeline.named_steps["model"].n_features_in_ > 0
    if spec["novelty"]:
        np.testing.assert_allclose(scores, -pipeline.score_samples(query))
    elif hasattr(pipeline, "predict_proba"):
        np.testing.assert_allclose(scores, pipeline.predict_proba(query)[:, 1])
    else:
        np.testing.assert_allclose(scores, pipeline.decision_function(query))
    if spec["family"] == "dummy":
        np.testing.assert_allclose(scores, 0.25)

    names = pipeline[:-1].get_feature_names_out()
    assert len(names) == pipeline.named_steps["model"].n_features_in_
    buffer = BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    np.testing.assert_allclose(predict_score(spec, joblib.load(buffer), query), scores)
    assert CANDIDATES == catalog_before
    pd.testing.assert_frame_equal(X, original)


def test_pruning_cutoff_retains_missingness_signal_and_original_feature_ids():
    X = pd.DataFrame(
        {
            3: np.arange(20),
            10: [1.0] * 20,
            21: [np.nan] * 20,
            144: [8.0] + [np.nan] * 19,
            589: [5.0, np.nan] * 10,
        }
    )
    pruner = SensorPruner().fit(X)
    assert pruner.get_feature_names_out().tolist() == [3, 144, 589]
    pd.testing.assert_frame_equal(pruner.transform(X), X[[3, 144, 589]].astype(float))
    assert SensorPruner(max_missing=0.94).fit(X).get_feature_names_out().tolist() == [3, 589]
    query = X.iloc[:2].copy()
    query[10] = [1, 2]
    query[21] = [5, 6]
    assert pruner.transform(query).columns.tolist() == [3, 144, 589]


@pytest.mark.parametrize("change", ["reorder", "rename", "drop", "append"])
def test_prediction_rejects_changed_original_sensor_schema(training_data, change):
    X, y = training_data
    spec = small_spec("rf_leaf2_none")
    pipeline = fit_candidate(spec, X, y, jobs=1)
    if change == "reorder":
        query = X.iloc[:, ::-1]
    elif change == "rename":
        query = X.rename(columns={3: 999})
    elif change == "drop":
        query = X.drop(columns=305)
    else:
        query = X.assign(extra=0)
    with pytest.raises(ValueError, match="sensor"):
        predict_score(spec, pipeline, query)


def test_missing_indicator_survives_when_all_observed_values_are_constant():
    X = pd.DataFrame({589: [8.0, np.nan] * 20, 144: [7.0] * 40})
    y = np.array([0, 1] * 20)
    spec = small_spec("logistic_c1_balanced")
    pipeline = fit_candidate(spec, X, y, jobs=1)
    assert pipeline.named_steps["sensors"].get_feature_names_out().tolist() == [589]
    assert pipeline[:-1].get_feature_names_out().tolist() == ["missingindicator_589"]
    assert (
        predict_score(spec, pipeline, X.iloc[[1]])[0]
        > predict_score(spec, pipeline, X.iloc[[0]])[0]
    )


def test_anova_clamps_k_drops_undefined_columns_and_selects_training_signal():
    y = np.array([0, 1] * 20)
    X = pd.DataFrame({3: np.arange(40), 10: np.ones(40), 589: y})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector = FoldANOVA(k=100).fit(X, y)
    assert not caught
    assert selector.get_feature_names_out().tolist() == [3, 589]
    best = FoldANOVA(k=1).fit(X, y)
    assert best.get_feature_names_out().tolist() == [589]
    query = X.iloc[:2].copy()
    query[3] = [1e9, -1e9]
    np.testing.assert_array_equal(best.transform(query), query[[589]].to_numpy())
    # Refitting the same transformer must use the original k, not an earlier clamp.
    selector.fit(X.assign(extra=y), y)
    assert selector.get_feature_names_out().tolist() == [3, 589, "extra"]


@pytest.mark.parametrize("identifier", ["logistic_c1_balanced", "v2_rf_knn_smote"])
def test_prediction_cannot_relearn_mask_imputer_scaler_or_donors(training_data, identifier):
    X, y = training_data
    spec = small_spec(identifier)
    pipeline = fit_candidate(spec, X, y, jobs=1)
    preprocessing = pipeline[:-2] if spec["family"] == "v2" else pipeline[:-1]
    state_before = joblib.hash(preprocessing)
    query = X.iloc[:3].copy()
    query.loc[:, :] = 1e6
    query[3] = np.nan
    query[10] = -1e6
    transformed = preprocessing.transform(query)
    assert np.isfinite(transformed).all()
    predict_score(spec, pipeline, query)
    assert joblib.hash(preprocessing) == state_before


@pytest.mark.parametrize("k", ["all", 100])
def test_native_mapper_keeps_raw_nans_and_training_missing_indicators(k):
    X = pd.DataFrame(
        {3: [0.0, 1.0, 2.0, 3.0], 10: [0.0, np.nan, 2.0, 4.0], 589: [8.0, np.nan, 8.0, np.nan]}
    )
    mapper = NativeSensorMapper(k=k).fit(X, [0, 0, 1, 1])
    names = mapper.get_feature_names_out().tolist()
    assert names == ([3, 10, 589] if k == "all" else [3, 10]) + [
        "missingindicator_10",
        "missingindicator_589",
    ]
    transformed = mapper.transform(X)
    selected = mapper.selected_indices_
    np.testing.assert_allclose(transformed[:, : len(selected)], X.iloc[:, selected], equal_nan=True)
    np.testing.assert_array_equal(transformed[:, -2:], X[[10, 589]].isna().astype(float))
    state_before = joblib.hash(mapper)
    query = pd.DataFrame({3: [np.nan], 10: [1e9], 589: [np.nan]})
    assert np.isnan(mapper.transform(query)[0, 0])
    assert mapper.get_feature_names_out().tolist() == names
    assert joblib.hash(mapper) == state_before


def test_native_top_k_keeps_indicator_when_no_imputed_sensor_varies():
    X = pd.DataFrame({589: [8.0, np.nan] * 20})
    mapper = NativeSensorMapper(k=100).fit(X, [0, 1] * 20)
    assert mapper.get_feature_names_out().tolist() == ["missingindicator_589"]
    np.testing.assert_array_equal(mapper.transform(X).ravel(), X[589].isna().astype(float))


@pytest.mark.parametrize("identifier", ["xgb_k100_depth2_weight1", "histgb_k100_leaves7"])
def test_native_pipeline_uses_training_only_selection_and_feature_ids(training_data, identifier):
    X, y = training_data
    spec = small_spec(identifier)
    pipeline = fit_candidate(spec, X, y, jobs=1)
    names = pipeline[:-1].get_feature_names_out().tolist()
    assert 3 in names
    assert "missingindicator_589" in names
    assert 144 not in names
    assert 305 not in names
    assert np.isnan(pipeline[:-1].transform(X)).any()
    state_before = joblib.hash(pipeline[:-1])
    query = X.iloc[:4].copy()
    query[144] = np.arange(4)
    query[305] = 1e9
    query[10] = np.nan
    predict_score(spec, pipeline, query)
    assert joblib.hash(pipeline[:-1]) == state_before


@pytest.mark.parametrize("negatives,positives", [(60, 20), (20, 20)])
def test_xgb_training_ratio_is_fold_local(training_data, negatives, positives):
    X, _ = training_data
    spec = small_spec("xgb_k100_depth2_weighttrain_ratio")
    labels = np.array([0] * negatives + [1] * positives)
    pipeline = fit_candidate(spec, X.iloc[: len(labels)], labels, jobs=1)
    assert pipeline.named_steps["model"].scale_pos_weight == negatives / positives
    predict_score(spec, pipeline, X.iloc[-7:])
    assert pipeline.named_steps["model"].scale_pos_weight == negatives / positives
    assert spec["params"]["scale_pos_weight"] == "train_ratio"


@pytest.mark.parametrize("label", [0, 1])
def test_xgb_training_ratio_rejects_a_missing_class(training_data, label):
    X, _ = training_data
    spec = small_spec("xgb_kall_depth2_weighttrain_ratio")
    with pytest.raises(ValueError, match="Both classes"):
        fit_candidate(spec, X, np.full(len(X), label), jobs=1)


@pytest.mark.parametrize("identifier", ["isolation_forest_pass", "one_class_svm_pass"])
def test_entire_novelty_pipeline_excludes_failures_and_scores_outliers_higher(identifier):
    rng = np.random.default_rng(29)
    passes = pd.DataFrame(rng.normal(size=(60, 4)), columns=[3, 10, 21, 589])
    passes[589] = 8.0
    failures = pd.DataFrame(np.full((20, 4), 100.0), columns=passes.columns)
    failures.loc[::2, 21] = np.nan
    X = pd.concat([passes, failures], ignore_index=True)
    labels = np.array([0] * 60 + [1] * 20)
    spec = small_spec(identifier)
    pipeline = fit_candidate(spec, X, labels, seed=31, jobs=1)
    pass_only = fit_candidate(spec, passes, np.zeros(60), seed=31, jobs=1)

    assert pipeline.named_steps["sensors"].get_feature_names_out().tolist() == [3, 10, 21]
    assert pipeline.named_steps["impute"].indicator_.features_.tolist() == []
    np.testing.assert_allclose(pipeline.named_steps["scale"].center_, passes[[3, 10, 21]].median())
    np.testing.assert_allclose(predict_score(spec, pipeline, X), predict_score(spec, pass_only, X))
    scores = predict_score(spec, pipeline, X)
    assert np.median(scores[60:]) > np.median(scores[:60])
    if "pca" in pipeline.named_steps:
        assert pipeline.named_steps["pca"].n_samples_ == 60
    with pytest.raises(ValueError, match="passing training wafers"):
        fit_candidate(spec, failures, np.ones(20), jobs=1)


def test_v2_reference_matches_original_pipeline_exactly(training_data):
    X, y = training_data
    spec = small_spec("v2_rf_knn_smote")
    reference = make_v2_pipeline("rf_knn_smote", seed=37, trees=3, jobs=1).fit(X, y)
    pipeline = fit_candidate(spec, X, y, seed=37, jobs=1)
    assert list(pipeline.named_steps) == list(reference.named_steps)
    np.testing.assert_allclose(predict_score(spec, pipeline, X), reference.predict_proba(X)[:, 1])
    assert joblib.hash(pipeline) == joblib.hash(reference)


def test_convergence_warnings_are_not_suppressed(training_data):
    X, y = training_data
    spec = small_spec("logistic_c1_balanced")
    spec["params"]["max_iter"] = 1
    with pytest.warns(ConvergenceWarning):
        pipeline = fit_candidate(spec, X, y, jobs=1)
    assert np.isfinite(predict_score(spec, pipeline, X)).all()


@pytest.mark.parametrize("bad_scores", [[np.nan], [np.inf], [[0.5]], [0.5, 0.5]])
def test_nonfinite_or_malformed_scores_are_rejected(bad_scores):
    class InvalidPipeline:
        def decision_function(self, X):
            return bad_scores

    with pytest.raises(ValueError, match="finite 1D"):
        predict_score({"novelty": False}, InvalidPipeline(), pd.DataFrame({3: [1.0]}))


def test_pruner_rejects_empty_sensor_sets_and_requires_fit():
    with pytest.raises(ValueError, match="No sensor"):
        SensorPruner().fit(pd.DataFrame({3: [1, 1], 10: [np.nan, np.nan]}))
    with pytest.raises(NotFittedError):
        SensorPruner().transform(pd.DataFrame({3: [1, 2]}))
    with pytest.raises(ValueError, match="Unknown candidate"):
        make_pipeline({"family": "unknown", "params": {}})

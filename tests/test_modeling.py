"""Small synthetic regressions for fold-local preprocessing and model fitting."""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import NotFittedError

from secom.modeling import SensorFilter, fit_pipeline, make_pipeline


@pytest.fixture
def sensor_frame():
    return pd.DataFrame(
        {
            "sensor_003": np.arange(8, dtype=float),
            "sensor_010": [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan],
            "sensor_021": [1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan],
            "sensor_144": np.ones(8),
            "sensor_305": np.full(8, np.nan),
            "sensor_589": [8, np.nan, 8, np.nan, 8, np.nan, 8, np.nan],
        },
        index=[f"wafer_{index}" for index in range(8)],
    )


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(53)
    y = np.array([0] * 24 + [1] * 12)
    X = pd.DataFrame(
        rng.normal(size=(36, 6)),
        columns=[
            "sensor_003",
            "sensor_010",
            "sensor_021",
            "sensor_144",
            "sensor_305",
            "sensor_589",
        ],
    )
    X["sensor_003"] += 1.5 * y
    X["sensor_010"] *= 1200
    X.loc[:17, "sensor_010"] = np.nan
    X["sensor_144"] = 9.0
    X["sensor_305"] = np.nan
    X.loc[:18, "sensor_589"] = np.nan
    return X, y


@pytest.mark.parametrize(
    "max_missing,expected_names",
    [
        (0.0, ["sensor_003"]),
        (0.5, ["sensor_003", "sensor_010"]),
        (1.0, ["sensor_003", "sensor_010", "sensor_021"]),
    ],
)
def test_sensor_filter_missing_cutoff_is_inclusive_and_excludes_constant_columns(
    sensor_frame, max_missing, expected_names
):
    original = sensor_frame.copy(deep=True)
    sensors = SensorFilter(max_missing=max_missing).fit(sensor_frame)

    assert sensors.get_feature_names_out().tolist() == expected_names
    np.testing.assert_allclose(
        sensors.transform(sensor_frame), sensor_frame[expected_names].to_numpy(), equal_nan=True
    )
    pd.testing.assert_frame_equal(sensor_frame, original)


def test_sensor_filter_retains_original_numeric_feature_ids(sensor_frame):
    sensor_frame.columns = [3, 10, 21, 144, 305, 589]
    sensors = SensorFilter().fit(sensor_frame)

    assert sensors.get_feature_names_out().tolist() == [3, 10]
    np.testing.assert_allclose(
        sensors.transform(sensor_frame), sensor_frame[[3, 10]].to_numpy(), equal_nan=True
    )


@pytest.mark.parametrize(
    "change", ["reorder", "rename", "drop", "append", "array_width", "one_row"]
)
def test_sensor_filter_rejects_wrong_columns_or_input_dimensions(sensor_frame, change):
    sensors = SensorFilter().fit(sensor_frame)
    if change == "reorder":
        invalid = sensor_frame.iloc[:, ::-1]
    elif change == "rename":
        invalid = sensor_frame.rename(columns={"sensor_003": "sensor_004"})
    elif change == "drop":
        invalid = sensor_frame.drop(columns="sensor_305")
    elif change == "append":
        invalid = sensor_frame.assign(sensor_999=0)
    elif change == "array_width":
        invalid = sensor_frame.to_numpy()[:, :-1]
    else:
        invalid = sensor_frame.iloc[0].to_numpy()

    with pytest.raises(ValueError, match="sensor"):
        sensors.transform(invalid)


def test_sensor_filter_accepts_arrays_with_training_column_order(sensor_frame):
    sensors = SensorFilter().fit(sensor_frame)

    np.testing.assert_allclose(
        sensors.transform(sensor_frame.to_numpy()),
        sensor_frame[["sensor_003", "sensor_010"]].to_numpy(),
        equal_nan=True,
    )


def test_sensor_filter_rejects_a_training_fold_without_usable_sensors():
    X = pd.DataFrame(
        {
            "all_missing": [np.nan] * 6,
            "constant": [3.0] * 6,
            "sparse_nonconstant": [1, 2, np.nan, np.nan, np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="No nonconstant sensor"):
        SensorFilter().fit(X)


@pytest.mark.parametrize("max_missing", [-0.1, 1.1])
def test_sensor_filter_rejects_invalid_missingness_cutoffs(sensor_frame, max_missing):
    with pytest.raises(ValueError, match="max_missing"):
        SensorFilter(max_missing=max_missing).fit(sensor_frame)


def test_sensor_filter_requires_fitting_before_transform_or_feature_names(sensor_frame):
    sensors = SensorFilter()

    with pytest.raises(NotFittedError):
        sensors.transform(sensor_frame)
    with pytest.raises(NotFittedError):
        sensors.get_feature_names_out()


@pytest.mark.parametrize("name", ["logistic_median", "rf_knn"])
def test_validation_changes_cannot_relearn_sensor_mask_or_scaling(sensor_frame, name):
    original = sensor_frame.copy(deep=True)
    pipeline = fit_pipeline(name, sensor_frame, [0] * 5 + [1] * 3, seed=17, trees=3, jobs=1)
    sensors = pipeline.named_steps["sensors"]
    scale = pipeline.named_steps["scale"]
    mask = sensors.keep_mask_.copy()
    mean, variance = scale.mean_.copy(), scale.var_.copy()
    counts = scale.n_samples_seen_.copy()

    np.testing.assert_array_equal(mask, [True, True, False, False, False, False])
    np.testing.assert_allclose(mean, [3.5, 4.0])
    np.testing.assert_allclose(variance, [5.25, 5.0])
    np.testing.assert_array_equal(counts, [8, 4])

    validation = sensor_frame.iloc[:3].copy()
    # Previously rejected sensors now vary; retained sensors are constant or missing.
    validation.loc[:, :] = np.arange(18, dtype=float).reshape(3, 6) + 100_000
    validation["sensor_003"] = np.nan
    validation["sensor_010"] = -100_000.0
    transformed = pipeline[:-1].transform(validation)
    assert transformed.shape == (3, 2)
    assert np.isfinite(transformed).all()
    validation["sensor_003"] = 1_000_000.0
    validation["sensor_010"] = np.nan
    assert pipeline.predict_proba(validation).shape == (3, 2)

    np.testing.assert_array_equal(sensors.keep_mask_, mask)
    assert sensors.get_feature_names_out().tolist() == ["sensor_003", "sensor_010"]
    np.testing.assert_array_equal(scale.mean_, mean)
    np.testing.assert_array_equal(scale.var_, variance)
    np.testing.assert_array_equal(scale.n_samples_seen_, counts)
    pd.testing.assert_frame_equal(sensor_frame, original)


def test_knn_neighbors_use_training_standardized_sensor_geometry():
    # Raw distances prefer the five donors with target=0. After standardization,
    # the large-scale sensor contributes little and the target=100 donors are closest.
    X = pd.DataFrame(
        {
            "sensor_017": [0.0] * 5 + [1000.0] * 5 + [100_000.0],
            "sensor_203": [10.0] * 5 + [0.0] * 5 + [10.0],
            "sensor_507": [0.0] * 5 + [100.0] * 5 + [0.0],
        }
    )
    pipeline = fit_pipeline("rf_knn", X, [0] * 7 + [1] * 4, seed=3, trees=3, jobs=1)
    query = pd.DataFrame([[0.0, 0.0, np.nan]], columns=X.columns)

    imputed = pipeline.named_steps["scale"].inverse_transform(pipeline[:-1].transform(query))

    np.testing.assert_allclose(imputed, [[0.0, 0.0, 100.0]], atol=1e-10)


def test_pca_learns_training_components_and_keeps_them_for_validation():
    coordinate = np.arange(-10, 11, dtype=float)
    X = pd.DataFrame(
        {"sensor_003": coordinate, "sensor_010": 10 * coordinate, "sensor_021": coordinate / 100}
    )
    pipeline = fit_pipeline("rf_knn_pca", X, [0] * 14 + [1] * 7, seed=11, trees=3, jobs=1)
    pca = pipeline.named_steps["pca"]
    components, mean = pca.components_.copy(), pca.mean_.copy()
    variance_ratio = pca.explained_variance_ratio_.copy()

    assert pca.n_components_ == 1
    np.testing.assert_allclose(variance_ratio, [1.0])
    validation = pd.DataFrame([[1e6, -1e6, 4e6], [-1e6, 3e6, 0], [0, -2e6, 1e6]], columns=X.columns)
    assert pipeline[:-1].transform(validation).shape == (3, 1)
    pipeline.predict_proba(validation)

    np.testing.assert_array_equal(pca.components_, components)
    np.testing.assert_array_equal(pca.mean_, mean)
    np.testing.assert_array_equal(pca.explained_variance_ratio_, variance_ratio)
    assert pca.n_samples_ == len(X)


def test_smote_resamples_training_only_and_preserves_validation_rows(
    classification_data, monkeypatch
):
    X, y = classification_data
    original_fit_resample = SMOTE.fit_resample
    calls = []

    def record_fit_resample(self, features, labels):
        resampled_X, resampled_y = original_fit_resample(self, features, labels)
        calls.append((features.shape, resampled_X.shape, np.bincount(resampled_y).tolist()))
        assert np.isfinite(features).all()
        return resampled_X, resampled_y

    monkeypatch.setattr(SMOTE, "fit_resample", record_fit_resample)
    pipeline = fit_pipeline("rf_knn_smote", X, y, seed=19, trees=3, jobs=1)
    assert calls == [((36, 3), (48, 3), [24, 24])]
    validation = X.iloc[[0, 7, 25, 30]].copy()
    original = validation.copy(deep=True)
    received = []
    original_predict_proba = pipeline.named_steps["model"].predict_proba

    def record_predict_proba(features):
        received.append(features.copy())
        return original_predict_proba(features)

    monkeypatch.setattr(pipeline.named_steps["model"], "predict_proba", record_predict_proba)
    assert pipeline.predict_proba(validation).shape == (4, 2)
    assert len(received) == 1
    assert received[0].shape == (4, 3)
    np.testing.assert_array_equal(received[0], pipeline[:-2].transform(validation))
    assert len(calls) == 1
    pd.testing.assert_frame_equal(validation, original)


@pytest.mark.parametrize(
    "name",
    ["dummy", "logistic_median", "rf_median", "rf_knn", "rf_knn_pca", "rf_knn_smote", "xgb_knn"],
)
def test_candidates_fit_predict_and_serialize_on_small_synthetic_data(classification_data, name):
    X, y = classification_data
    pipeline = fit_pipeline(name, X, y, seed=23, trees=3, jobs=1)
    validation = X.iloc[[1, 10, 25, 35]].copy()
    probabilities = pipeline.predict_proba(validation)

    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    if name == "dummy":
        np.testing.assert_allclose(probabilities, [[2 / 3, 1 / 3]] * 4)
    else:
        assert pipeline.named_steps["sensors"].get_feature_names_out().tolist() == [
            "sensor_003",
            "sensor_010",
            "sensor_021",
        ]

    buffer = BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict_proba(validation), probabilities)


@pytest.mark.parametrize("negatives,positives,expected_weight", [(18, 6, 3.0), (12, 12, 1.0)])
def test_xgboost_class_weight_uses_only_the_current_training_labels(
    classification_data, negatives, positives, expected_weight
):
    X, _ = classification_data
    training = X.iloc[: negatives + positives].copy()
    y = np.array([0] * negatives + [1] * positives)
    pipeline = fit_pipeline("xgb_knn", training, y, seed=29, trees=3, jobs=1)

    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == expected_weight
    # Scoring a differently sized held-out set must leave the fitted ratio intact.
    pipeline.predict_proba(X.iloc[-7:])
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == expected_weight


@pytest.mark.parametrize("label", [0, 1])
def test_xgboost_rejects_single_class_training_before_fitting(classification_data, label):
    X, _ = classification_data

    with pytest.raises(ValueError, match="Both classes"):
        fit_pipeline("xgb_knn", X.iloc[:8], np.full(8, label), seed=31, trees=3, jobs=1)


def test_unknown_model_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown candidate"):
        make_pipeline("unsupported", seed=0, trees=3, jobs=1)

"""Identical fold-local preprocessing for each controlled model comparison."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

CANDIDATES = (
    "dummy",
    "logistic_median",
    "rf_median",
    "rf_knn",
    "rf_knn_pca",
    "rf_knn_smote",
    "xgb_knn",
)
LABELS = {
    "dummy": "Prevalence dummy",
    "logistic_median": "Balanced logistic / Median",
    "rf_median": "Balanced RF / Median",
    "rf_knn": "Balanced RF / KNN",
    "rf_knn_pca": "Balanced RF / KNN + PCA90",
    "rf_knn_smote": "RF / KNN + SMOTE",
    "xgb_knn": "Weighted XGBoost / KNN",
}


class SensorFilter(TransformerMixin, BaseEstimator):
    """Learn missingness and constant-column masks only from the training partition."""

    def __init__(self, max_missing: float = 0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        if not 0 <= self.max_missing <= 1:
            raise ValueError("max_missing must lie in [0, 1].")
        frame = pd.DataFrame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.keep_mask_ = (
            (frame.isna().mean() <= self.max_missing) & (frame.nunique(dropna=True) > 1)
        ).to_numpy()
        if not self.keep_mask_.any():
            raise ValueError("No nonconstant sensor meets the training missingness cutoff.")
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Prediction data must contain the original sensor columns in order.")
        if isinstance(X, pd.DataFrame) and not np.array_equal(X.columns, self.feature_names_in_):
            raise ValueError("Prediction sensor names/order differ from training.")
        return values[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        return self.feature_names_in_[self.keep_mask_]


def make_pipeline(name: str, seed: int, trees: int, jobs: int) -> Pipeline:
    if name not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {name}")
    if name == "dummy":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])

    steps = [
        ("sensors", SensorFilter()),
        # NaNs survive scaling. KNN distances therefore use training-standardized coordinates.
        ("scale", StandardScaler()),
        (
            "impute",
            SimpleImputer(strategy="median") if "median" in name else KNNImputer(n_neighbors=5),
        ),
    ]
    if name == "rf_knn_pca":
        steps.append(("pca", PCA(n_components=0.90, svd_solver="full")))
    if name == "rf_knn_smote":
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))

    if name == "logistic_median":
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    elif name == "xgb_knn":
        model = XGBClassifier(
            n_estimators=trees,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            tree_method="hist",
            eval_metric="logloss",
            importance_type="gain",
            random_state=seed,
            n_jobs=jobs,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=jobs,
        )
    return Pipeline([*steps, ("model", model)])


def fit_pipeline(name: str, X, y, seed: int, trees: int, jobs: int) -> Pipeline:
    pipeline = make_pipeline(name, seed, trees, jobs)
    if name == "xgb_knn":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if not counts.all():
            raise ValueError("Both classes are required to fit weighted XGBoost.")
        pipeline.set_params(model__scale_pos_weight=float(counts[0] / counts[1]))
    pipeline.fit(X, y)
    return pipeline

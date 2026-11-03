"""Serializable candidate grids and training-only preprocessing for failure scores.

Labels are 0 for pass and 1 for failure. Every score increases toward failure.
The explicit grids contain 50 candidates, including the frozen v2 reference and
the prior dummy; a rank blend belongs to the study, not this catalog.
"""

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from secom.modeling import make_pipeline as make_v2_pipeline


def _candidate(identifier, family, params, novelty=False):
    return {"id": identifier, "family": family, "params": params, "novelty": novelty}


CANDIDATES = (
    _candidate("v2_rf_knn_smote", "v2", {"n_estimators": 300}),
    *(
        _candidate(
            f"rf_leaf{leaf}_{weight or 'none'}",
            "rf",
            {
                "n_estimators": 300,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
                "class_weight": weight,
            },
        )
        for leaf in (2, 8)
        for weight in (None, "balanced")
    ),
    *(
        _candidate(
            f"extra_trees_leaf{leaf}_{weight or 'none'}",
            "extra_trees",
            {
                "n_estimators": 300,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
                "class_weight": weight,
            },
        )
        for leaf in (1, 4, 12)
        for weight in (None, "balanced")
    ),
    *(
        _candidate(
            f"balanced_rf_leaf{leaf}",
            "balanced_rf",
            {
                "n_estimators": 300,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
                "sampling_strategy": "all",
                "replacement": True,
                "bootstrap": False,
            },
        )
        for leaf in (2, 8)
    ),
    *(
        _candidate(
            f"xgb_k{k}_depth{depth}_weight{weight}",
            "xgb",
            {
                "k": k,
                "max_depth": depth,
                "scale_pos_weight": weight,
                "n_estimators": 300,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "reg_lambda": 10.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "importance_type": "gain",
            },
        )
        for k in ("all", 100)
        for depth in (2, 3)
        for weight in (1, "train_ratio")
    ),
    *(
        _candidate(
            f"histgb_k{k}_leaves{leaves}",
            "histgb",
            {
                "k": k,
                "max_leaf_nodes": leaves,
                "learning_rate": 0.05,
                "l2_regularization": 10.0,
                "max_iter": 300,
                "early_stopping": False,
                "class_weight": "balanced",
                "categorical_features": None,
            },
        )
        for k in ("all", 100)
        for leaves in (7, 15)
    ),
    *(
        _candidate(
            f"svc_c{c:g}_gamma_{gamma}",
            "svc",
            {
                "k": 100,
                "C": c,
                "kernel": "rbf",
                "gamma": gamma,
                "class_weight": "balanced",
                "cache_size": 512,
                "max_iter": 10000,
            },
        )
        for c in (0.1, 1.0, 10.0)
        for gamma in ("scale", "auto")
    ),
    *(
        _candidate(
            f"linear_svc_c{c:g}",
            "linear_svc",
            {"k": 100, "C": c, "class_weight": "balanced", "dual": "auto", "max_iter": 10000},
        )
        for c in (0.01, 0.1, 1.0)
    ),
    *(
        _candidate(
            f"logistic_c{c:g}_{weight or 'none'}",
            "logistic",
            {"k": 100, "C": c, "class_weight": weight, "solver": "lbfgs", "max_iter": 3000},
        )
        for c in (0.01, 0.1, 1.0)
        for weight in (None, "balanced")
    ),
    *(
        _candidate(
            f"knn_k{k}_neighbors{neighbors}",
            "knn",
            {"k": k, "n_neighbors": neighbors, "weights": "distance"},
        )
        for k in (30, 100)
        for neighbors in (5, 21)
    ),
    _candidate("lda_shrinkage", "lda", {"k": 100, "solver": "lsqr", "shrinkage": "auto"}),
    *(
        _candidate(
            f"gaussian_nb_var{variance:g}",
            "gaussian_nb",
            {"k": 100, "var_smoothing": variance},
        )
        for variance in (1e-9, 1e-7)
    ),
    _candidate(
        "isolation_forest_pass",
        "isolation_forest",
        {"n_estimators": 300, "contamination": "auto"},
        novelty=True,
    ),
    _candidate(
        "one_class_svm_pass",
        "one_class_svm",
        {"nu": 0.05, "kernel": "rbf", "gamma": "scale", "max_iter": 10000, "pca_variance": 0.90},
        novelty=True,
    ),
    _candidate("dummy_prior", "dummy", {"strategy": "prior"}),
)


def _feature_names(X):
    if isinstance(X, pd.DataFrame):
        return np.asarray(X.columns, dtype=object)
    return np.asarray([f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object)


def _sensor_values(X, names):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Prediction data must contain the training sensor columns in order.")
    if isinstance(X, pd.DataFrame) and not np.array_equal(X.columns, names):
        raise ValueError("Prediction sensor names/order differ from training.")
    return values


class SensorPruner(TransformerMixin, BaseEstimator):
    """Keep observed variation or varying missingness, learned from training only."""

    def __init__(self, max_missing=0.95):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        if not 0 <= self.max_missing <= 1:
            raise ValueError("max_missing must lie in [0, 1].")
        frame = pd.DataFrame(X)
        missing = frame.isna().mean()
        varying = (frame.nunique(dropna=True) > 1) | ((missing > 0) & (missing < 1))
        self.feature_names_in_ = _feature_names(frame)
        self.n_features_in_ = frame.shape[1]
        self.keep_mask_ = ((missing <= self.max_missing) & varying).to_numpy()
        if not self.keep_mask_.any():
            raise ValueError("No sensor meets the training variation and missingness cutoffs.")
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        values = _sensor_values(X, self.feature_names_in_)
        return pd.DataFrame(
            values[:, self.keep_mask_],
            columns=(
                X.columns[self.keep_mask_]
                if isinstance(X, pd.DataFrame)
                else self.get_feature_names_out()
            ),
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("Input sensor names differ from training.")
        return self.feature_names_in_[self.keep_mask_].copy()


class FoldANOVA(TransformerMixin, BaseEstimator):
    """Clamp top-k to defined, varying training features; resolve ties in sensor order."""

    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        if self.k != "all" and (not isinstance(self.k, int) or self.k < 1):
            raise ValueError("k must be a positive integer or 'all'.")
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = _feature_names(X)
        varying = np.flatnonzero(np.ptp(values, axis=0) > 0)
        self.scores_ = np.full(self.n_features_in_, np.nan)
        if len(varying):
            # Perfect class separation has an infinite, valid ANOVA score.
            with np.errstate(divide="ignore", invalid="ignore"):
                self.scores_[varying] = f_classif(values[:, varying], y)[0]
        defined = np.flatnonzero(~np.isnan(self.scores_))
        count = len(defined) if self.k == "all" else min(self.k, len(defined))
        ranked = defined[np.argsort(-self.scores_[defined], kind="stable")]
        self.keep_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.keep_mask_[ranked[:count]] = True
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        return _sensor_values(X, self.feature_names_in_)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        if len(names) != self.n_features_in_:
            raise ValueError("Feature count differs from training.")
        return names[self.keep_mask_].copy()


class NativeSensorMapper(TransformerMixin, BaseEstimator):
    """Use training medians for ANOVA only; retain raw NaNs for native models."""

    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y):
        self.feature_names_in_ = _feature_names(X)
        self.n_features_in_ = len(self.feature_names_in_)
        values = _sensor_values(X, self.feature_names_in_)
        missing = np.isnan(values)
        self.indicator_indices_ = np.flatnonzero(missing.any(axis=0) & ~missing.all(axis=0))
        if self.k == "all":
            self.selected_indices_ = np.arange(self.n_features_in_)
        else:
            self.imputer_ = SimpleImputer(strategy="median").fit(values)
            self.selector_ = FoldANOVA(self.k).fit(self.imputer_.transform(values), y)
            self.selected_indices_ = np.flatnonzero(self.selector_.keep_mask_)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        values = _sensor_values(X, self.feature_names_in_)
        indicators = np.isnan(values[:, self.indicator_indices_]).astype(float)
        return np.column_stack((values[:, self.selected_indices_], indicators))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_indices_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        if not np.array_equal(names, self.feature_names_in_):
            raise ValueError("Input sensor names differ from training.")
        return np.asarray(
            [
                *names[self.selected_indices_],
                *(f"missingindicator_{names[i]}" for i in self.indicator_indices_),
            ],
            dtype=object,
        )


def make_pipeline(spec, seed=2026, jobs=2):
    """Build an unfitted pipeline without mutating its JSON-serializable spec.

    Use fit_candidate to resolve XGBoost's training ratio and pass-only fitting.
    The v2 reference delegates to its original builder, including KNN and SMOTE.
    """
    family = spec["family"]
    params = dict(spec["params"])
    if family == "v2":
        return make_v2_pipeline("rf_knn_smote", seed, params["n_estimators"], jobs)

    k = params.pop("k", None)
    pca_variance = params.pop("pca_variance", None)
    if family in ("rf", "extra_trees", "balanced_rf"):
        estimator = {
            "rf": RandomForestClassifier,
            "extra_trees": ExtraTreesClassifier,
            "balanced_rf": BalancedRandomForestClassifier,
        }[family]
        model = estimator(**params, random_state=seed, n_jobs=jobs)
    elif family == "xgb":
        if params.get("scale_pos_weight") == "train_ratio":
            params["scale_pos_weight"] = 1.0
        model = XGBClassifier(**params, random_state=seed, n_jobs=jobs)
    elif family == "histgb":
        model = HistGradientBoostingClassifier(**params, random_state=seed)
    elif family == "svc":
        model = SVC(**params, random_state=seed)
    elif family == "linear_svc":
        model = LinearSVC(**params, random_state=seed)
    elif family == "logistic":
        model = LogisticRegression(**params, random_state=seed)
    elif family == "knn":
        model = KNeighborsClassifier(**params, n_jobs=jobs)
    elif family == "lda":
        model = LinearDiscriminantAnalysis(**params)
    elif family == "gaussian_nb":
        model = GaussianNB(**params)
    elif family == "isolation_forest":
        model = IsolationForest(**params, random_state=seed, n_jobs=jobs)
    elif family == "one_class_svm":
        model = OneClassSVM(**params)
    elif family == "dummy":
        model = DummyClassifier(**params, random_state=seed)
    else:
        raise ValueError(f"Unknown candidate family: {family}")

    steps = [("sensors", SensorPruner())]
    if family in ("xgb", "histgb"):
        steps.append(("features", NativeSensorMapper(k=k)))
    else:
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
        if family in (
            "svc",
            "linear_svc",
            "logistic",
            "knn",
            "lda",
            "gaussian_nb",
            "isolation_forest",
            "one_class_svm",
        ):
            steps.append(("scale", RobustScaler()))
        if k is not None:
            steps.append(("select", FoldANOVA(k=k)))
        if pca_variance is not None:
            steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    return Pipeline([*steps, ("model", model)])


def fit_candidate(spec, X, y, seed=2026, jobs=2):
    """Fit on the supplied training partition, excluding failures for novelty models."""
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be a 1D array of pass=0/fail=1 matching X.")
    pipeline = make_pipeline(spec, seed=seed, jobs=jobs)
    if spec["novelty"]:
        passes = labels == 0
        if not passes.any():
            raise ValueError("Novelty fitting requires passing training wafers.")
        pipeline.fit(X.iloc[passes])
    else:
        if spec["family"] == "xgb" and spec["params"].get("scale_pos_weight") == "train_ratio":
            counts = np.bincount(labels.astype(int), minlength=2)
            if not counts.all():
                raise ValueError("Both classes are required for the training class ratio.")
            pipeline.set_params(model__scale_pos_weight=float(counts[0] / counts[1]))
        pipeline.fit(X, labels)
    return pipeline


def predict_score(spec, pipeline, X):
    """Return one finite score per wafer, with larger scores indicating failure."""
    if spec["novelty"]:
        scores = -pipeline.score_samples(X)
    elif hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        positive = np.flatnonzero(pipeline.classes_ == 1)
        scores = probabilities[:, positive[0]] if len(positive) else np.zeros(len(X))
    else:
        scores = pipeline.decision_function(X)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be a finite 1D array matching X.")
    return scores

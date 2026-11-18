"""Multivariate statistical process control (MSPC) for fault detection.

This is the classic FDC baseline used on fab tool data: fit a PCA model on
in-control (passing) observations only, then monitor two statistics.

* Hotelling T^2 - distance *inside* the PCA model plane, scaled by each
  component's variance. Large T^2 = known correlation structure, unusual size.
* SPE (Q statistic) - squared reconstruction residual *outside* the model
  plane. Large SPE = the correlation structure itself broke.

Both statistics decompose exactly into per-sensor contributions, which is
what process engineers read when an alarm fires.

Preprocessing is learned from training passes only: the same training-local
sensor pruning as v3, median imputation and z-score autoscaling.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from smartfab.models import SensorPruner

SCORES = ("t2", "spe", "combined")


@dataclass(frozen=True)
class MSPCResult:
    t2: np.ndarray
    spe: np.ndarray
    combined: np.ndarray

    def score(self, name: str) -> np.ndarray:
        if name not in SCORES:
            raise ValueError(f"score must be one of {SCORES}.")
        return getattr(self, name)


def t2_limit(n_components: int, n_samples: int, alpha: float = 0.99) -> float:
    """Phase-II Hotelling T^2 limit for a new observation (F distribution)."""
    a, n = n_components, n_samples
    if not 0 < a < n - 1:
        raise ValueError("Need 0 < n_components < n_samples - 1.")
    factor = a * (n - 1) * (n + 1) / (n * (n - a))
    return float(factor * stats.f.ppf(alpha, a, n - a))


def spe_limit(training_spe, alpha: float = 0.99) -> float:
    """Box (1954) / Nomikos-MacGregor g*chi2(h) approximation of the SPE limit."""
    values = np.asarray(training_spe, dtype=float)
    mean, variance = float(values.mean()), float(values.var(ddof=1))
    if mean <= 0 or variance <= 0:
        return float(values.max())
    g, h = variance / (2 * mean), 2 * mean**2 / variance
    return float(g * stats.chi2.ppf(alpha, h))


class MSPCModel(BaseEstimator):
    """PCA monitoring model fitted on passing observations only.

    ``variance`` is the cumulative explained-variance target for the number
    of retained components; ``alpha`` is the control-limit confidence level.
    """

    def __init__(self, variance=0.90, alpha=0.99, max_missing=0.95):
        self.variance = variance
        self.alpha = alpha
        self.max_missing = max_missing

    def fit(self, X, y=None):
        """Fit on X; if labels are given, rows with y == 1 are excluded."""
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if y is not None:
            labels = np.asarray(y)
            if labels.shape != (len(frame),):
                raise ValueError("Labels must align with X.")
            frame = frame.loc[labels == 0]
        if len(frame) < 10:
            raise ValueError("MSPC needs at least 10 in-control training observations.")
        if not 0 < self.variance < 1 or not 0.5 < self.alpha < 1:
            raise ValueError("variance must lie in (0, 1) and alpha in (0.5, 1).")
        self.sensor_columns_ = list(frame.columns)
        self.pruner_ = SensorPruner(self.max_missing).fit(frame)
        kept = self.pruner_.transform(frame)
        self.feature_names_ = np.asarray(kept.columns, dtype=object)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(kept)
        filled = self.imputer_.transform(kept)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0, ddof=1)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        Z = (filled - self.mean_) / self.scale_
        max_components = min(Z.shape[0] - 2, Z.shape[1])
        self.pca_ = PCA(n_components=self.variance, svd_solver="full").fit(Z)
        if self.pca_.n_components_ > max_components:
            self.pca_ = PCA(n_components=max_components, svd_solver="full").fit(Z)
        self.n_components_ = int(self.pca_.n_components_)
        self.n_train_ = int(Z.shape[0])
        self.eigenvalues_ = self.pca_.explained_variance_
        training = self._statistics(Z)
        self.t2_limit_ = t2_limit(self.n_components_, self.n_train_, self.alpha)
        self.spe_limit_ = spe_limit(training[1], self.alpha)
        # Empirical limits: non-Gaussian sensors make the theoretical ones optimistic.
        self.t2_empirical_limit_ = float(np.quantile(training[0], self.alpha))
        self.spe_empirical_limit_ = float(np.quantile(training[1], self.alpha))
        return self

    def _standardize(self, X):
        check_is_fitted(self, "pca_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.sensor_columns_)
        if list(frame.columns) != self.sensor_columns_:
            raise ValueError("Sensor columns differ from the MSPC training schema.")
        kept = self.pruner_.transform(frame)
        return (self.imputer_.transform(kept) - self.mean_) / self.scale_

    def _statistics(self, Z):
        P = self.pca_.components_.T
        scores = Z @ P
        t2 = np.sum(scores**2 / self.eigenvalues_, axis=1)
        residual = Z - scores @ P.T
        spe = np.sum(residual**2, axis=1)
        return t2, spe, scores, residual

    def statistics(self, X) -> MSPCResult:
        t2, spe, _, _ = self._statistics(self._standardize(X))
        return MSPCResult(t2, spe, t2 / self.t2_limit_ + spe / self.spe_limit_)

    def alarms(self, X, empirical=False) -> pd.DataFrame:
        """Classic control-chart decision: alarm when either statistic exceeds its limit."""
        result = self.statistics(X)
        t2_lim = self.t2_empirical_limit_ if empirical else self.t2_limit_
        spe_lim = self.spe_empirical_limit_ if empirical else self.spe_limit_
        return pd.DataFrame(
            {
                "t2": result.t2,
                "spe": result.spe,
                "t2_alarm": result.t2 > t2_lim,
                "spe_alarm": result.spe > spe_lim,
                "alarm": (result.t2 > t2_lim) | (result.spe > spe_lim),
            }
        )

    def contributions(self, X, statistic="spe") -> pd.DataFrame:
        """Per-sensor contributions that sum exactly to each row's T^2 or SPE.

        SPE: squared residual of each scaled sensor (non-negative).
        T^2: partial decomposition sum_a (t_a / lambda_a) * p_ja * z_j, which
        sums to T^2 but can be negative for individual sensors.
        """
        Z = self._standardize(X)
        t2, spe, scores, residual = self._statistics(Z)
        if statistic == "spe":
            values = residual**2
        elif statistic == "t2":
            weights = (scores / self.eigenvalues_) @ self.pca_.components_
            values = weights * Z
        else:
            raise ValueError("statistic must be 'spe' or 't2'.")
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(values, columns=self.feature_names_, index=index)

    def summary(self) -> dict:
        check_is_fitted(self, "pca_")
        return {
            "variance_target": self.variance,
            "alpha": self.alpha,
            "n_train_passes": self.n_train_,
            "n_sensors_monitored": len(self.feature_names_),
            "n_components": self.n_components_,
            "explained_variance": float(self.pca_.explained_variance_ratio_.sum()),
            "t2_limit": self.t2_limit_,
            "spe_limit": self.spe_limit_,
            "t2_empirical_limit": self.t2_empirical_limit_,
            "spe_empirical_limit": self.spe_empirical_limit_,
        }


def top_contributors(contributions: pd.DataFrame, n: int = 5) -> list[list[tuple[str, float]]]:
    """Return the n largest-magnitude contributing sensors for each row."""
    values = contributions.to_numpy()
    names = contributions.columns.to_numpy()
    order = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :n]
    return [
        [(str(names[j]), float(values[i, j])) for j in row] for i, row in enumerate(order)
    ]

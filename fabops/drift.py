"""Per-sensor distribution drift between a reference window and a later batch.

Three signals per sensor:

* PSI (population stability index) on reference-quantile bins, with missing
  values as their own bin, so a sensor that starts dropping out also drifts.
* Two-sample KS on observed values, with Benjamini-Hochberg adjustment over
  sensors.
* Change in missing-value rate.

Sampling noise matters with small batches. Under no drift,
PSI * n*m/(n+m) is approximately chi-square with (bins - 1) degrees of
freedom, so the expected noise PSI is about (bins - 1) * (1/n + 1/m). For a
100-row batch with 10 bins that is about 0.1, which is why the usual
"PSI > 0.25" rule is applied together with a sensor-share trigger rather
than to any single sensor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

PSI_ALERT = 0.25
EPSILON = 1e-4


@dataclass
class DriftReference:
    """Bin edges and reference proportions learned from one window only."""

    columns: list
    edges: list
    proportions: list
    missing_rate: np.ndarray
    observed: list
    n_rows: int

    @classmethod
    def fit(cls, frame: pd.DataFrame, bins: int = 10) -> "DriftReference":
        if bins < 2 or frame.empty:
            raise ValueError("Need at least two bins and a nonempty reference.")
        edges, proportions, observed = [], [], []
        for column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            present = values[~np.isnan(values)]
            if present.size:
                inner = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1]))
            else:
                inner = np.array([])
            edges.append(inner)
            proportions.append(_proportions(values, inner))
            observed.append(present)
        return cls(
            list(frame.columns),
            edges,
            proportions,
            frame.isna().mean().to_numpy(),
            observed,
            len(frame),
        )


def _proportions(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    """Bin proportions; the last bin holds missing values."""
    missing = np.isnan(values)
    counts = np.bincount(
        np.searchsorted(inner_edges, values[~missing], side="right"),
        minlength=inner_edges.size + 1,
    )
    counts = np.append(counts, missing.sum()).astype(float)
    return counts / max(values.size, 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not finite.size:
        return q
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * finite.size / np.arange(1, finite.size + 1)
    q[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def sensor_drift(reference: DriftReference, batch: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sensor with PSI, KS, missing-rate change and alert flags."""
    if list(batch.columns) != reference.columns:
        raise ValueError("Batch columns must match the drift reference.")
    rows = []
    for index, column in enumerate(reference.columns):
        values = batch[column].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        base = reference.observed[index]
        if present.size >= 2 and base.size >= 2 and np.ptp(np.r_[present, base]) > 0:
            ks = stats.ks_2samp(base, present)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            ks_stat, ks_p = np.nan, np.nan
        rows.append(
            {
                "sensor": column,
                "psi": psi(reference.proportions[index], _proportions(values, reference.edges[index])),
                "ks_statistic": ks_stat,
                "ks_p": ks_p,
                "missing_rate_reference": float(reference.missing_rate[index]),
                "missing_rate_batch": float(np.isnan(values).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["ks_q"] = benjamini_hochberg(table["ks_p"])
    table["missing_rate_change"] = table.missing_rate_batch - table.missing_rate_reference
    table["psi_alert"] = table.psi > PSI_ALERT
    table["ks_alert"] = table.ks_q < 0.01
    return table


def drift_share(table: pd.DataFrame, informative: np.ndarray | None = None) -> float:
    """Share of (informative) sensors with PSI above the alert level."""
    mask = np.ones(len(table), dtype=bool) if informative is None else informative
    return float(table.loc[mask, "psi_alert"].mean()) if mask.any() else 0.0


def informative_sensors(reference_frame: pd.DataFrame) -> np.ndarray:
    """Sensors that vary in the reference window; constant sensors cannot drift in PSI."""
    nunique = reference_frame.nunique(dropna=True).to_numpy()
    missing = reference_frame.isna().mean().to_numpy()
    return (nunique > 1) & (missing < 0.95)


def expected_null_psi(bins: int, n_reference: int, n_batch: int) -> float:
    """Approximate mean PSI from sampling noise alone (chi-square approximation)."""
    return (bins - 1) * (1 / n_reference + 1 / n_batch)

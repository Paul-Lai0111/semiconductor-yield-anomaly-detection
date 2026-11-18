"""How stable is "which sensors matter"? Bootstrap and time-slice rank stability.

A single feature-importance run on 87 failures is not trustworthy. Each
ranking method is re-run on stratified bootstrap resamples and on
consecutive time slices of the development data. A sensor earns a place on
the engineer watchlist only if it keeps appearing near the top.

Ranking methods (all learned inside each resample):

* ``anova`` - univariate F statistic, failure vs pass, median-imputed.
* ``random_forest`` - impurity importance of a 200-tree forest.
* ``mspc_spe`` - log ratio of mean SPE contribution, failures vs passes, from
  an MSPC model fitted on the resample's passes. This is the multivariate
  "which residuals blow up when a unit fails" view.

The output is a prioritised inspection list with stability evidence, not a
root-cause claim: correlation with recorded failures, on anonymised sensors.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from fabops.mspc import MSPCModel
from smartfab.models import SensorPruner

METHODS = ("anova", "random_forest", "mspc_spe")


def method_scores(method: str, X: pd.DataFrame, y: np.ndarray, seed: int, jobs: int = 2) -> pd.Series:
    """Importance per original sensor (pruned sensors get NaN)."""
    if method == "mspc_spe":
        model = MSPCModel(variance=0.90).fit(X, y)
        contrib = model.contributions(X, "spe")
        fail = contrib.loc[y == 1].mean()
        passing = contrib.loc[y == 0].mean()
        values = np.log((fail + 1e-6) / (passing + 1e-6))
        return values.reindex(X.columns)
    pruner = SensorPruner().fit(X)
    kept = pruner.transform(X)
    filled = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(kept)
    if method == "anova":
        varying = np.ptp(filled, axis=0) > 0
        values = np.full(filled.shape[1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            values[varying] = f_classif(filled[:, varying], y)[0]
    elif method == "random_forest":
        forest = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=jobs
        ).fit(filled, y)
        values = forest.feature_importances_
    else:
        raise ValueError(f"Unknown method {method}.")
    return pd.Series(values, index=kept.columns).reindex(X.columns)


def top_set(scores: pd.Series, k: int) -> set:
    return set(scores.dropna().sort_values(ascending=False, kind="stable").index[:k])


def ranks(scores: pd.Series) -> pd.Series:
    return scores.rank(ascending=False, method="average", na_option="bottom")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def stratified_bootstrap(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(np.flatnonzero(y == c), np.count_nonzero(y == c), replace=True) for c in (0, 1)]
    return np.sort(np.concatenate(parts))


def time_slices(n_rows: int, n_slices: int) -> list[np.ndarray]:
    return [np.arange(n_rows)[s] for s in np.array_split(np.arange(n_rows), n_slices)]


def run_stability(
    X: pd.DataFrame,
    y: np.ndarray,
    n_bootstrap: int = 50,
    n_slices: int = 4,
    k: int = 20,
    seed: int = 2026,
    jobs: int = 2,
    methods=METHODS,
) -> dict:
    """X and y must be in chronological order (time slices are consecutive rows)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    samples = [stratified_bootstrap(y, rng) for _ in range(n_bootstrap)]
    frequency, median_rank, rank_iqr, full_rank = {}, {}, {}, {}
    boot_jaccard, slice_jaccard, slice_frequency = {}, {}, {}
    full_tops = {}
    for method in methods:
        full = method_scores(method, X, y, seed, jobs)
        full_rank[method] = ranks(full)
        full_tops[method] = top_set(full, k)
        tops, rank_rows = [], []
        for b, rows in enumerate(samples):
            scores = method_scores(method, X.iloc[rows], y[rows], seed + b, jobs)
            tops.append(top_set(scores, k))
            rank_rows.append(ranks(scores))
        rank_table = pd.concat(rank_rows, axis=1)
        frequency[method] = pd.Series(
            {sensor: np.mean([sensor in t for t in tops]) for sensor in X.columns}
        )
        median_rank[method] = rank_table.median(axis=1)
        rank_iqr[method] = rank_table.quantile(0.75, axis=1) - rank_table.quantile(0.25, axis=1)
        boot_jaccard[method] = float(np.mean([jaccard(a, b) for a, b in combinations(tops, 2)]))
        slice_tops = []
        for rows in time_slices(len(y), n_slices):
            if y[rows].sum() < 3:
                continue
            slice_tops.append(top_set(method_scores(method, X.iloc[rows], y[rows], seed, jobs), k))
        slice_jaccard[method] = (
            float(np.mean([jaccard(a, b) for a, b in combinations(slice_tops, 2)]))
            if len(slice_tops) > 1
            else float("nan")
        )
        slice_frequency[method] = pd.Series(
            {sensor: np.mean([sensor in t for t in slice_tops]) for sensor in X.columns}
        )
    cross = {
        f"{a}__{b}": jaccard(full_tops[a], full_tops[b]) for a, b in combinations(methods, 2)
    }
    table = pd.DataFrame({"sensor": X.columns})
    for method in methods:
        table[f"{method}_top{k}_freq"] = frequency[method].to_numpy()
        table[f"{method}_median_rank"] = median_rank[method].to_numpy()
        table[f"{method}_rank_iqr"] = rank_iqr[method].to_numpy()
        table[f"{method}_full_rank"] = full_rank[method].to_numpy()
        table[f"{method}_slice_freq"] = slice_frequency[method].to_numpy()
    freq_cols = [f"{m}_top{k}_freq" for m in methods]
    table["consensus_freq"] = table[freq_cols].mean(axis=1)
    table["methods_freq_ge_0.5"] = (table[freq_cols] >= 0.5).sum(axis=1)
    table["slice_freq"] = table[[f"{m}_slice_freq" for m in methods]].mean(axis=1)
    table["missing_rate"] = X.isna().mean().to_numpy()
    table["tier"] = np.select(
        [
            (table.consensus_freq >= 0.5) & (table["methods_freq_ge_0.5"] >= 2),
            table.consensus_freq >= 0.25,
        ],
        ["stable", "moderate"],
        "exploratory",
    )
    table = table.sort_values(["consensus_freq", "slice_freq"], ascending=False, kind="stable")
    return {
        "table": table.reset_index(drop=True),
        "bootstrap_jaccard": boot_jaccard,
        "time_slice_jaccard": slice_jaccard,
        "cross_method_jaccard": cross,
        "k": k,
        "n_bootstrap": n_bootstrap,
        "n_slices": n_slices,
    }

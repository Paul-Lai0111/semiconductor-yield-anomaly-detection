"""Freeze development choices before producing held-out estimates."""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

from secom.data import load_dataset
from secom.evaluation import bootstrap_ap, choose_threshold, operating_metrics, ranking_metrics
from secom.modeling import CANDIDATES, fit_pipeline, make_pipeline


@dataclass(frozen=True)
class StudyConfig:
    seed: int = 2026
    folds: int = 5
    trees: int = 300
    jobs: int = 2
    bootstrap: int = 2000
    temporal_check: bool = True
    test_fraction: float = 0.2

    def __post_init__(self):
        if self.folds < 2 or self.trees < 1 or self.bootstrap < 1 or self.jobs < 1:
            raise ValueError(
                "Use at least two folds and positive trees, bootstrap draws, and jobs."
            )
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must lie strictly between zero and one.")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cross_validate(name, X, y, folds, config):
    scores = np.full(len(y), np.nan)
    seen = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (train, validation) in enumerate(folds):
        if np.intersect1d(train, validation).size:
            raise ValueError("Training and validation indices overlap.")
        start = perf_counter()
        pipeline = fit_pipeline(
            name, X.iloc[train], y[train], config.seed + fold, config.trees, config.jobs
        )
        scores[validation] = pipeline.predict_proba(X.iloc[validation])[:, 1]
        seen[validation] += 1
        rows.append(
            {
                "candidate": name,
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_failures": int(y[train].sum()),
                "validation_failures": int(y[validation].sum()),
                **ranking_metrics(y[validation], scores[validation]),
                "retained_sensors": (
                    int(pipeline["sensors"].keep_mask_.sum()) if name != "dummy" else None
                ),
                "pca_components": int(pipeline["pca"].n_components_)
                if "pca" in pipeline.named_steps
                else None,
                "seconds": round(perf_counter() - start, 3),
            }
        )
        print(
            f"  {name} fold {fold + 1}/{len(folds)} AP={rows[-1]['average_precision']:.4f}",
            flush=True,
        )
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Every development row must receive exactly one finite OOF prediction.")
    threshold = choose_threshold(y, scores)
    fold_ap = [row["average_precision"] for row in rows]
    summary = {
        "candidate": name,
        "mean_fold_ap": float(np.mean(fold_ap)),
        "std_fold_ap": float(np.std(fold_ap, ddof=1)),
        "pooled_oof_ap": ranking_metrics(y, scores)["average_precision"],
        "oof_threshold": threshold,
        "oof_default_f1": operating_metrics(y, scores, 0.5)["f1"],
        "oof_selected_f1": operating_metrics(y, scores, threshold)["f1"],
        "oof_selected_precision": operating_metrics(y, scores, threshold)["precision"],
        "oof_selected_recall": operating_metrics(y, scores, threshold)["recall"],
    }
    return scores, rows, summary


def temporal_sensitivity(dataset, development, config):
    ordered = development[np.argsort(dataset.timestamps[development], kind="stable")]
    boundary = int(len(ordered) * 0.8)
    # Equal timestamps must not straddle a time boundary.
    while (
        boundary > 0
        and dataset.timestamps[ordered[boundary - 1]] == dataset.timestamps[ordered[boundary]]
    ):
        boundary -= 1
    early, later = ordered[:boundary], ordered[boundary:]
    temporal_folds = list(TimeSeriesSplit(n_splits=3).split(early))
    scores = np.full(len(early), np.nan)
    for fold, (train, validation) in enumerate(temporal_folds):
        if dataset.timestamps[early[train]].max() >= dataset.timestamps[early[validation]].min():
            raise ValueError("Temporal folds contain overlapping timestamps.")
        pipeline = fit_pipeline(
            "xgb_knn",
            dataset.X.iloc[early[train]],
            dataset.y[early[train]],
            config.seed + fold,
            config.trees,
            config.jobs,
        )
        scores[validation] = pipeline.predict_proba(dataset.X.iloc[early[validation]])[:, 1]
    predicted_rows = np.flatnonzero(np.isfinite(scores))
    threshold = choose_threshold(dataset.y[early[predicted_rows]], scores[predicted_rows])
    pipeline = fit_pipeline(
        "xgb_knn", dataset.X.iloc[early], dataset.y[early], config.seed, config.trees, config.jobs
    )
    later_scores = pipeline.predict_proba(dataset.X.iloc[later])[:, 1]
    return {
        "candidate": "xgb_knn",
        "purpose": "Predeclared development-only chronological sensitivity, not model selection.",
        "early_rows": len(early),
        "later_rows": len(later),
        "early_failures": int(dataset.y[early].sum()),
        "later_failures": int(dataset.y[later].sum()),
        "last_training_timestamp": dataset.timestamps[early].max().isoformat(),
        "first_validation_timestamp": dataset.timestamps[later].min().isoformat(),
        "threshold_source": "Three expanding-window folds within the early development portion.",
        **ranking_metrics(dataset.y[later], later_scores),
        "default": operating_metrics(dataset.y[later], later_scores, 0.5),
        "selected": operating_metrics(dataset.y[later], later_scores, threshold),
    }, pd.DataFrame({"row_id": later, "label": dataset.y[later], "score": later_scores})


def output_paths(output: Path) -> tuple[Path, Path]:
    published = Path(__file__).resolve().parents[1] / "results"
    root = output.parent if output.resolve() == published else output
    return root / "images", root / "artifacts"


def run_study(config: StudyConfig, data_dir: Path, output: Path, *, overwrite: bool = False):
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} is not empty. Choose a new --output or explicitly --overwrite."
        )
    if overwrite and not config.temporal_check:
        (output / "temporal_predictions.csv").unlink(missing_ok=True)
    figure_dir, artifacts = output_paths(output)
    started = datetime.now(UTC).isoformat()
    dataset = load_dataset(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    source_files = [
        project_root / "main.py",
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        *sorted((project_root / "secom").glob("*.py")),
    ]
    source_hash = sha256()
    for path in source_files:
        source_hash.update(path.relative_to(project_root).as_posix().encode())
        source_hash.update(path.read_bytes())

    development, holdout = train_test_split(
        np.arange(len(dataset.y)),
        test_size=config.test_fraction,
        stratify=dataset.y,
        random_state=config.seed,
    )
    if np.intersect1d(development, holdout).size:
        raise ValueError("Development and holdout rows overlap.")
    X_dev, y_dev = dataset.X.iloc[development], dataset.y[development]
    folds = list(
        StratifiedKFold(config.folds, shuffle=True, random_state=config.seed).split(X_dev, y_dev)
    )
    assignment = np.full(len(dataset.y), -1)
    for fold, (_, validation) in enumerate(folds):
        assignment[development[validation]] = fold
    pd.DataFrame(
        {
            "row_id": np.arange(len(dataset.y)),
            "partition": np.where(assignment >= 0, "development", "holdout"),
            "validation_fold": assignment,
            "label": dataset.y,
            "timestamp": dataset.timestamps,
        }
    ).to_csv(output / "split_manifest.csv", index=False)

    oof, development_rows, fold_rows = {}, [], []
    for name in CANDIDATES:
        print(f"Development comparison: {name}", flush=True)
        oof[name], details, summary = cross_validate(name, X_dev, y_dev, folds, config)
        fold_rows.extend(details)
        development_rows.append(summary)
    # Stable candidate order breaks exact ties. No holdout scores exist at this point.
    selected = max(development_rows, key=lambda row: row["mean_fold_ap"])["candidate"]
    thresholds = {row["candidate"]: row["oof_threshold"] for row in development_rows}
    protocol = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hash.hexdigest(),
        "config": asdict(config),
        "candidates_in_tie_break_order": list(CANDIDATES),
        "model_parameters": {
            name: {
                key: "NaN" if isinstance(value, float) and np.isnan(value) else value
                for key, value in make_pipeline(name, config.seed, config.trees, config.jobs)[
                    "model"
                ]
                .get_params()
                .items()
            }
            for name in CANDIDATES
        },
        "fold_seed_policy": "Estimator/SMOTE seed = seed + zero-based fold; full-development refit uses seed.",
        "xgboost_weight_policy": "Training pass count divided by training failure count, recomputed for each fit.",
        "preprocessing": {
            "max_training_missing_fraction": 0.5,
            "drop_training_constants": True,
            "scaling": "StandardScaler before imputation; NaNs retained during scaling.",
            "imputation": "Median or KNN: 5 neighbors, uniform NaN-aware standardized distance.",
            "pca": "90% training variance, full SVD; component count learned per fit.",
            "smote": "Training only, 5 neighbors, default minority balancing.",
        },
        "model_selection": "Maximum mean development-fold average precision; listed-order exact ties.",
        "threshold_selection": "Maximum pooled development OOF F1; highest-threshold exact ties.",
        "selected_candidate": selected,
        "thresholds": thresholds,
        "holdout_policy": "Evaluate all frozen candidates for audit; do not change the CV-selected winner.",
        "dataset_history": "SECOM was explored previously. This protocol excludes holdout rows from fitting and selection; it does not erase historical exploration.",
    }
    write_json(output / "protocol.json", protocol)
    protocol_hash = sha256((output / "protocol.json").read_bytes()).hexdigest()
    pd.DataFrame(development_rows).to_csv(output / "development_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "candidate": name,
                    "row_id": development,
                    "label": y_dev,
                    "fold": assignment[development],
                    "score": oof[name],
                }
            )
            for name in CANDIDATES
        ]
    ).to_csv(output / "oof_predictions.csv", index=False)
    print(
        f"FROZEN: CV selection={selected}; thresholds saved before holdout evaluation.", flush=True
    )

    fitted, holdout_scores, holdout_results, metric_rows = {}, {}, {}, []
    X_test, y_test = dataset.X.iloc[holdout], dataset.y[holdout]
    for name in CANDIDATES:
        fitted[name] = fit_pipeline(name, X_dev, y_dev, config.seed, config.trees, config.jobs)
        scores = holdout_scores[name] = fitted[name].predict_proba(X_test)[:, 1]
        result = {
            **ranking_metrics(y_test, scores),
            "ap_bootstrap_95": bootstrap_ap(y_test, scores, config.bootstrap, config.seed),
            "default": operating_metrics(y_test, scores, 0.5),
            "selected": operating_metrics(y_test, scores, thresholds[name]),
        }
        holdout_results[name] = result
        for policy in ("default", "selected"):
            metric_rows.append(
                {
                    "candidate": name,
                    "policy": policy,
                    "average_precision": result["average_precision"],
                    **result[policy],
                }
            )
        print(
            f"Holdout {name}: AP={result['average_precision']:.4f}, frozen-threshold F1={result['selected']['f1']:.4f}",
            flush=True,
        )
    if sha256((output / "protocol.json").read_bytes()).hexdigest() != protocol_hash:
        raise ValueError("Frozen protocol was modified during holdout evaluation.")
    pd.DataFrame(metric_rows).to_csv(output / "holdout_metrics.csv", index=False)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "candidate": name,
                    "row_id": holdout,
                    "label": y_test,
                    "score": holdout_scores[name],
                    "threshold": thresholds[name],
                    "prediction": (holdout_scores[name] >= thresholds[name]).astype(int),
                }
            )
            for name in CANDIDATES
        ]
    ).to_csv(output / "holdout_predictions.csv", index=False)

    xgb = fitted["xgb_knn"]
    importance = pd.DataFrame(
        {
            "sensor": xgb["sensors"].get_feature_names_out(),
            "gain_importance": xgb["model"].feature_importances_,
        }
    ).sort_values("gain_importance", ascending=False, kind="stable")
    importance.to_csv(output / "feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "sensor": dataset.X.columns,
            "development_missing_fraction": X_dev.isna().mean().to_numpy(),
            "development_distinct_observed_values": X_dev.nunique().to_numpy(),
            "retained_on_full_development": xgb["sensors"].keep_mask_,
        }
    ).to_csv(output / "data_quality.csv", index=False)

    sensitivity = None
    if config.temporal_check:
        print("Development-only chronological XGBoost sensitivity check", flush=True)
        sensitivity, temporal_predictions = temporal_sensitivity(dataset, development, config)
        temporal_predictions.to_csv(output / "temporal_predictions.csv", index=False)
    metrics = {
        "schema_version": 1,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hash.hexdigest(),
        "protocol_sha256": protocol_hash,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "imbalanced-learn",
                    "xgboost",
                    "matplotlib",
                    "joblib",
                )
            },
        },
        "dataset": dataset.manifest,
        "config": asdict(config),
        "development_rows": len(development),
        "development_failures": int(y_dev.sum()),
        "holdout_rows": len(holdout),
        "holdout_failures": int(y_test.sum()),
        "selected_candidate": selected,
        "development": development_rows,
        "holdout": holdout_results,
        "temporal_sensitivity": sensitivity,
        "uncertainty_scope": "95% recall Wilson intervals and class-stratified fixed-score AP bootstrap percentile intervals. Conditional on this sample/model and observed prevalence; not training, selection, historical-exploration or future-drift uncertainty.",
    }
    metrics["outputs"] = {
        "figures": str(figure_dir.resolve().relative_to(output.resolve()))
        if figure_dir.resolve().is_relative_to(output.resolve())
        else "../images",
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": fitted[selected],
            "threshold": thresholds[selected],
            "sensor_columns": dataset.X.columns.tolist(),
            "candidate": selected,
            "config": asdict(config),
            "protocol_sha256": protocol_hash,
        },
        artifacts / "selected_model.joblib",
        compress=3,
    )
    from secom.reporting import create_figures, write_results

    create_figures(
        figure_dir,
        dataset,
        development,
        holdout,
        fitted,
        oof,
        holdout_scores,
        metrics,
    )
    metrics["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "metrics.json", metrics)
    write_results(output / "RESULTS.md", metrics)
    print(
        f"Complete. Read {output / 'RESULTS.md'}; model saved to {artifacts / 'selected_model.joblib'}",
        flush=True,
    )
    return metrics

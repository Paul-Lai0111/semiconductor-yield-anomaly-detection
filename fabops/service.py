"""Deployable scoring bundle and the batch-scoring logic shared by API and dashboard.

Build the bundle after both studies have run:

    uv run python -m fabops.service build

The bundle refits the *frozen* v3 winner and the *frozen* MSPC configuration on
the development window only (the same data both studies trained on), plus a
drift reference for that window. Nothing is re-selected.

Only load bundles you built yourself: joblib deserialization executes code.
"""

import argparse
import json
import re
from hashlib import sha256
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fabops.drift import DriftReference, drift_share, informative_sensors, sensor_drift
from fabops.mspc import MSPCModel, top_contributors
from secom.data import load_dataset
from smartfab.models import fit_candidate, predict_score
from smartfab.policies import blend_scores, capacity_mask, chronological_partition

SCHEMA = "fabops-service-1"
POLICIES = ("v3_selected_knn", "mspc", "hybrid_knn_mspc")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "fab_monitoring_results" / "artifacts" / "service_bundle.joblib"
DRIFT_WARNING_SHARE = 0.10


def build_bundle(data_dir: Path, v3_dir: Path, fab_dir: Path, seed: int = 2026, jobs: int = 2) -> dict:
    selection_bytes = (v3_dir / "selection.json").read_bytes()
    mspc_bytes = (fab_dir / "mspc_selection.json").read_bytes()
    selection, mspc_selection = json.loads(selection_bytes), json.loads(mspc_bytes)
    spec = selection["member_specs"][0]
    dataset = load_dataset(data_dir)
    development, _ = chronological_partition(dataset.timestamps, 0.80)
    X, y = dataset.X.iloc[development], dataset.y[development]
    knn = fit_candidate(spec, X, y, seed=seed, jobs=jobs)
    mspc = MSPCModel(variance=mspc_selection["variance"]).fit(X, y)
    watchlist = []
    stability_path = fab_dir / "sensor_stability.csv"
    if stability_path.exists():
        table = pd.read_csv(stability_path)
        watchlist = table.loc[table.tier.ne("exploratory"), ["sensor", "tier", "consensus_freq"]].head(20).to_dict(orient="records")
    return {
        "schema": SCHEMA,
        "sensor_columns": dataset.X.columns.tolist(),
        "knn_spec": spec,
        "knn": knn,
        "mspc": mspc,
        "mspc_statistic": mspc_selection["statistic"],
        "drift_reference": DriftReference.fit(X),
        "informative": informative_sensors(X),
        "seed": seed,
        "default_review_fraction": selection["primary_review_fraction"],
        "watchlist": watchlist,
        "provenance": {
            "v3_winner": selection["winner"],
            "v3_selection_sha256": sha256(selection_bytes).hexdigest(),
            "mspc_candidate": mspc_selection["candidate"],
            "mspc_selection_sha256": sha256(mspc_bytes).hexdigest(),
            "training_rows": int(len(development)),
            "training_last_timestamp": dataset.timestamps[development].max().isoformat(),
            "mspc": mspc.summary(),
        },
    }


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} bundle.")
    return bundle


def validate_batch(bundle: dict, frame: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Return stable unit IDs and a float sensor frame in training column order."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Batch must be a nonempty table with unique column names.")
    columns = bundle["sensor_columns"]
    extra = set(frame.columns) - set(columns) - {"unit_id"}
    missing = set(columns) - set(frame.columns)
    if extra or missing:
        raise ValueError(
            f"Batch columns must be unit_id plus the {len(columns)} training sensors "
            f"(missing {len(missing)}, unexpected {sorted(extra)[:5]})."
        )
    if "unit_id" in frame:
        raw = frame["unit_id"].tolist()
        if any(
            not ((isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)))
                 or (isinstance(v, str) and re.fullmatch(r"[+-]?[0-9]+", v.strip())))
            for v in raw
        ):
            raise ValueError("unit_id must contain integers.")
        ids = [int(v) for v in raw]
        if len(set(ids)) != len(ids):
            raise ValueError("unit_id values must be unique.")
    else:
        ids = list(range(len(frame)))
    sensors = frame.loc[:, columns].replace("", np.nan)
    try:
        sensors = sensors.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError):
        raise ValueError("Sensor values must be numbers or blank.") from None
    if np.isinf(sensors.to_numpy()).any():
        raise ValueError("Sensor values must be finite or blank.")
    if sensors.isna().all(axis=1).any():
        raise ValueError("Every unit needs at least one observed sensor value.")
    return ids, sensors.reset_index(drop=True)


def score_batch(bundle: dict, frame: pd.DataFrame, review_fraction: float | None = None,
                policy: str = "v3_selected_knn", top_n: int = 3) -> pd.DataFrame:
    """Score one completed batch and build the inspection queue."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    fraction = bundle["default_review_fraction"] if review_fraction is None else float(review_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("review_fraction must lie in (0, 1].")
    ids, sensors = validate_batch(bundle, frame)
    knn_score = predict_score(bundle["knn_spec"], bundle["knn"], sensors)
    mspc = bundle["mspc"]
    statistics = mspc.statistics(sensors)
    mspc_score = statistics.score(bundle["mspc_statistic"])
    priority = {
        "v3_selected_knn": knn_score,
        "mspc": mspc_score,
        "hybrid_knn_mspc": blend_scores([knn_score, mspc_score]) if len(ids) > 1 else knn_score,
    }[policy]
    review = capacity_mask(priority, ids, fraction, bundle["seed"])
    contributors = top_contributors(mspc.contributions(sensors, "spe"), top_n)
    result = pd.DataFrame(
        {
            "unit_id": ids,
            "priority_score": priority,
            "review": review.astype(int),
            "knn_score": knn_score,
            "t2": statistics.t2,
            "spe": statistics.spe,
            "t2_alarm": (statistics.t2 > mspc.t2_limit_).astype(int),
            "spe_alarm": (statistics.spe > mspc.spe_limit_).astype(int),
            "top_spe_sensors": [", ".join(name for name, _ in row) for row in contributors],
        }
    )
    result["queue_rank"] = result.priority_score.rank(ascending=False, method="first").astype(int)
    return result.sort_values("queue_rank").reset_index(drop=True)


def batch_drift(bundle: dict, frame: pd.DataFrame, top: int = 10) -> dict:
    _, sensors = validate_batch(bundle, frame)
    table = sensor_drift(bundle["drift_reference"], sensors)
    informative = bundle["informative"]
    share = drift_share(table, informative)
    worst = table.loc[informative].sort_values("psi", ascending=False).head(top)
    return {
        "rows": len(sensors),
        "informative_sensors": int(informative.sum()),
        "psi_alert_share": share,
        "warning": bool(share >= DRIFT_WARNING_SHARE),
        "warning_rule": f"share of informative sensors with PSI > 0.25 >= {DRIFT_WARNING_SHARE:.0%}",
        "top_sensors": worst[["sensor", "psi", "ks_statistic", "missing_rate_change"]]
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }


def write_demo_batch(data_dir: Path, output_dir: Path) -> Path:
    """Export the temporal-assessment rows as an unlabeled demo batch (labels separately)."""
    dataset = load_dataset(data_dir)
    _, assessment = chronological_partition(dataset.timestamps, 0.80)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = dataset.X.iloc[assessment].copy()
    batch.insert(0, "unit_id", assessment)
    path = output_dir / "demo_batch.csv"
    batch.to_csv(path, index=False)
    pd.DataFrame({"unit_id": assessment, "label": dataset.y[assessment],
                  "timestamp": dataset.timestamps[assessment]}).to_csv(output_dir / "demo_labels.csv", index=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the fab-monitoring service bundle.")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--v3-dir", type=Path, default=Path("manufacturing_results"))
    parser.add_argument("--fab-dir", type=Path, default=Path("fab_monitoring_results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    bundle = build_bundle(args.data_dir, args.v3_dir, args.fab_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    demo = write_demo_batch(args.data_dir, args.output.parent)
    print(f"Wrote {args.output} and demo batch {demo}")


if __name__ == "__main__":
    main()

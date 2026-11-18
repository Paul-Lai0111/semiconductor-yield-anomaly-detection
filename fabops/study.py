"""Fab-monitoring study: MSPC baseline, drift, retraining policies, sensor stability.

Run after the frozen v3 study:

    uv run python -m fabops.study --overwrite

Inputs are the hash-checked SECOM files and the frozen v3 artifacts in
``manufacturing_results/``. The v3 winner is never re-selected here. The
MSPC configuration is selected with the *same* development-only objective as
v3 (failures caught at a 20% review budget on the three forward blocks),
then frozen before the temporal assessment block is scored.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from threadpoolctl import threadpool_limits

from fabops.drift import DriftReference, informative_sensors, sensor_drift
from fabops.mspc import SCORES, MSPCModel
from fabops.retraining import RetrainConfig, simulate, summarize
from fabops.stability import run_stability
from secom.data import load_dataset
from smartfab.models import fit_candidate, predict_score
from smartfab.policies import (
    blend_scores,
    capacity_mask,
    chronological_partition,
    evaluate_policy,
    forward_splits,
)
from smartfab.study import SearchConfig, ranking, score_blocks

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "manufacturing_results"
VARIANCES = (0.80, 0.90, 0.95)
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


@dataclass(frozen=True)
class FabConfig:
    seed: int = 2026
    jobs: int = 2
    review_fraction: float = 0.20
    alpha: float = 0.99
    stability_bootstrap: int = 50
    stability_slices: int = 4
    stability_top_k: int = 20
    retrain: RetrainConfig = RetrainConfig()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_frozen_v3():
    """Read the frozen v3 selection and verify it against the v3 metrics hash."""
    metrics = json.loads((V3 / "metrics.json").read_text())
    if file_hash(V3 / "selection.json") != metrics["selection_sha256"]:
        raise ValueError("manufacturing_results/selection.json differs from the frozen v3 run.")
    selection = json.loads((V3 / "selection.json").read_text())
    if selection["is_rank_blend"]:
        raise ValueError("This study expects a single frozen v3 winner.")
    return selection, metrics


def mspc_id(variance: float, statistic: str) -> str:
    return f"mspc_var{int(round(variance * 100))}_{statistic}"


def mspc_search(X, y, dev_ids, dev_times, config):
    """Score every (variance, statistic) MSPC variant on the v3 forward blocks."""
    splits = forward_splits(dev_times)
    search_config = SearchConfig(seed=config.seed, review_fraction=config.review_fraction)
    summaries, block_rows, block_scores = [], [], {}
    for variance in VARIANCES:
        stats_by_block = []
        for train, validation in splits:
            model = MSPCModel(variance=variance, alpha=config.alpha).fit(X.iloc[train], y[train])
            stats_by_block.append(model.statistics(X.iloc[validation]))
        for statistic in SCORES:
            identifier = mspc_id(variance, statistic)
            scores = [s.score(statistic) for s in stats_by_block]
            rows, summary = score_blocks(identifier, scores, splits, dev_ids, y, search_config)
            summary.update(family="mspc", variance=variance, statistic=statistic)
            summaries.append(summary)
            block_rows.extend(rows)
            block_scores[identifier] = scores
    return splits, summaries, block_rows, block_scores


def forward_reference_scores(candidate: str, splits, dev_ids):
    """Re-read the frozen v3 forward-validation scores of one candidate, in block order."""
    forward = pd.read_csv(V3 / "forward_predictions.csv", float_precision="round_trip")
    forward = forward.loc[forward.candidate.eq(candidate)]
    scores = []
    for block, (_, validation) in enumerate(splits):
        part = forward.loc[forward.block.eq(block)].set_index("row_id")
        scores.append(part.loc[dev_ids[validation], "score"].to_numpy())
    return scores


def drift_noise_floor(window: pd.DataFrame, config: FabConfig, repeats: int = 20) -> dict:
    """PSI-alert share when nothing changes (random rows) vs. time order, inside one window."""
    size = config.retrain.batch_size
    rng = np.random.default_rng(config.seed)
    random_shares = []
    for _ in range(repeats):
        held = rng.choice(len(window), size, replace=False)
        rest = np.setdiff1d(np.arange(len(window)), held)
        reference = DriftReference.fit(window.iloc[rest], config.retrain.bins)
        table = sensor_drift(reference, window.iloc[held])
        random_shares.append(float(table.psi_alert[informative_sensors(window.iloc[rest])].mean()))
    reference = DriftReference.fit(window.iloc[:-size], config.retrain.bins)
    table = sensor_drift(reference, window.iloc[-size:])
    temporal = float(table.psi_alert[informative_sensors(window.iloc[:-size])].mean())
    return {
        "random_split_share_mean": float(np.mean(random_shares)),
        "random_split_share_max": float(np.max(random_shares)),
        "last_batch_vs_earlier_share": temporal,
        "note": "Same initial window: random held-out rows show sampling noise; the last rows in time show temporal drift.",
    }


def run(config: FabConfig, data_dir: Path, output: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output is not empty; pass --overwrite to replace it.")
    if output.resolve() in {ROOT / p for p in ("manufacturing_results", "results", "smartfab", "secom", "fabops")}:
        raise ValueError("Fab-monitoring output must be separate from sources and v3 evidence.")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir)
    selection, v3_metrics = load_frozen_v3()
    winner_spec = selection["member_specs"][0]
    development, assessment = chronological_partition(dataset.timestamps, 0.80)
    manifest = pd.read_csv(V3 / "split_manifest.csv")
    if not np.array_equal(
        np.sort(development), manifest.loc[manifest.partition.eq("development"), "row_id"].to_numpy()
    ):
        raise ValueError("Chronological split differs from the frozen v3 manifest.")
    X_dev, y_dev = dataset.X.iloc[development], dataset.y[development]
    X_new, y_new = dataset.X.iloc[assessment], dataset.y[assessment]

    sources = sorted((ROOT / "fabops").glob("*.py"))
    protocol = {
        "schema_version": 1,
        "frozen_before_analysis_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "v3_selection_sha256": file_hash(V3 / "selection.json"),
        "v3_protocol_sha256": v3_metrics["protocol_sha256"],
        "v3_winner": selection["winner"],
        "mspc_grid": {"variance": list(VARIANCES), "statistic": list(SCORES)},
        "mspc_selection_rule": "Same as v3: most forward-block failures caught at exact 20% review, then mean block AP, then grid order. Development data only.",
        "mspc_fit": "PCA on training passes only; training-local pruning, median imputation, z-scoring. T2 limit from F distribution, SPE limit from Box g*chi2(h), alpha=0.99.",
        "hybrid": "Predeclared equal rank blend of the frozen v3 winner and the selected MSPC score; reported, never used to re-select.",
        "drift": "Per-sensor PSI (10 reference-quantile bins + missing bin), KS with BH adjustment, missing-rate change.",
        "retraining": "Prequential replay of the full timeline in 100-row batches, labels delayed by one batch; frozen / periodic / drift-triggered / always.",
        "stability": "Stratified bootstraps and consecutive time slices of development data; ANOVA, random forest and MSPC-SPE rankings.",
        "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
    }
    write_json(output / "protocol.json", protocol)

    with threadpool_limits(limits=config.jobs):
        # 1. MSPC selection on development forward blocks, compared with v3 forward scores.
        print("1/4 MSPC forward-block search", flush=True)
        splits, summaries, block_rows, mspc_blocks = mspc_search(
            X_dev, y_dev, development, dataset.timestamps[development], config
        )
        ordered = sorted(
            enumerate(summaries), key=lambda item: (-item[1]["true_positives"], -item[1]["mean_block_ap"], item[0])
        )
        chosen = ordered[0][1]
        search_config = SearchConfig(seed=config.seed, review_fraction=config.review_fraction)
        references = {}
        for name in (selection["winner"], "v2_rf_knn_smote"):
            references[name] = forward_reference_scores(name, splits, development)
            rows, summary = score_blocks(name, references[name], splits, development, y_dev, search_config)
            summary.update(family="v3_reference", variance=None, statistic=None)
            summaries.append(summary)
            block_rows.extend(rows)
        hybrid_blocks = [
            blend_scores([a, b])
            for a, b in zip(references[selection["winner"]], mspc_blocks[chosen["candidate"]], strict=True)
        ]
        rows, summary = score_blocks("hybrid_knn_mspc", hybrid_blocks, splits, development, y_dev, search_config)
        summary.update(family="hybrid", variance=None, statistic=None)
        summaries.append(summary)
        block_rows.extend(rows)
        search = pd.DataFrame(summaries).sort_values(
            ["true_positives", "mean_block_ap"], ascending=False, kind="stable"
        )
        search.to_csv(output / "mspc_search.csv", index=False)
        pd.DataFrame(block_rows).to_csv(output / "mspc_search_blocks.csv", index=False)
        mspc_selection = {
            "frozen_before_assessment_utc": datetime.now(UTC).isoformat(),
            "candidate": chosen["candidate"],
            "variance": chosen["variance"],
            "statistic": chosen["statistic"],
            "development_true_positives": chosen["true_positives"],
            "development_failures": chosen["failure_support"],
            "development_mean_block_ap": chosen["mean_block_ap"],
            "protocol_sha256": file_hash(output / "protocol.json"),
        }
        write_json(output / "mspc_selection.json", mspc_selection)
        print(f"  FROZEN MSPC: {chosen['candidate']}", flush=True)

        # 2. Temporal assessment: same frozen boundary, same budget as v3.
        print("2/4 Temporal assessment", flush=True)
        mspc = MSPCModel(variance=chosen["variance"], alpha=config.alpha).fit(X_dev, y_dev)
        stats_new = mspc.statistics(X_new)
        stats_dev = mspc.statistics(X_dev)
        v3_frozen = pd.read_csv(V3 / "assessment_predictions.csv", float_precision="round_trip")
        frozen_scores = {
            name: v3_frozen.loc[v3_frozen.strategy.eq(name)].set_index("row_id").loc[assessment, "score"].to_numpy()
            for name in ("selected", "v2_reference")
        }
        strategies = {
            "mspc": stats_new.score(chosen["statistic"]),
            "v3_selected_knn": frozen_scores["selected"],
            "v2_reference": frozen_scores["v2_reference"],
        }
        strategies["hybrid_knn_mspc"] = blend_scores([strategies["v3_selected_knn"], strategies["mspc"]])
        for statistic in SCORES:
            strategies[f"mspc_{statistic}_only"] = stats_new.score(statistic)
        comparison, budgets, masks = {}, [], {}
        for name, scores in strategies.items():
            mask = capacity_mask(scores, assessment, config.review_fraction, config.seed)
            masks[name] = mask
            comparison[name] = {**ranking(y_new, scores), "capacity": evaluate_policy(y_new, mask)}
            for budget in BUDGETS:
                budgets.append(
                    {
                        "strategy": name,
                        "review_fraction": budget,
                        **evaluate_policy(y_new, capacity_mask(scores, assessment, budget, config.seed)),
                    }
                )
        reviews = int(np.floor(len(y_new) * config.review_fraction))
        random_range = hypergeom.ppf([0.025, 0.975], len(y_new), int(y_new.sum()), reviews).tolist()
        failures = y_new == 1
        overlap = {
            "knn_only": int((masks["v3_selected_knn"] & ~masks["mspc"] & failures).sum()),
            "mspc_only": int((masks["mspc"] & ~masks["v3_selected_knn"] & failures).sum()),
            "both": int((masks["mspc"] & masks["v3_selected_knn"] & failures).sum()),
            "neither": int((~masks["mspc"] & ~masks["v3_selected_knn"] & failures).sum()),
        }
        classic = {}
        for label, empirical in (("theoretical_limits", False), ("empirical_limits", True)):
            dev_alarm = mspc.alarms(X_dev, empirical)
            new_alarm = mspc.alarms(X_new, empirical)
            classic[label] = {
                "development_pass_alarm_rate_in_sample": float(dev_alarm.alarm[y_dev == 0].mean()),
                "assessment_pass_alarm_rate": float(new_alarm.alarm[y_new == 0].mean()),
                "assessment_t2_alarms": int(new_alarm.t2_alarm.sum()),
                "assessment_spe_alarms": int(new_alarm.spe_alarm.sum()),
                **evaluate_policy(y_new, new_alarm.alarm.to_numpy()),
            }
        pd.DataFrame(budgets).to_csv(output / "assessment_budgets.csv", index=False)
        timeline = pd.DataFrame(
            {
                "row_id": np.r_[development, assessment],
                "timestamp": np.r_[dataset.timestamps[development], dataset.timestamps[assessment]],
                "partition": ["development"] * len(development) + ["temporal_assessment"] * len(assessment),
                "label": np.r_[y_dev, y_new],
                "t2": np.r_[stats_dev.t2, stats_new.t2],
                "spe": np.r_[stats_dev.spe, stats_new.spe],
                "combined": np.r_[stats_dev.combined, stats_new.combined],
            }
        ).sort_values("timestamp", kind="stable")
        timeline.to_csv(output / "mspc_timeline.csv", index=False)
        predictions = pd.DataFrame(
            {"row_id": assessment, "label": y_new, **{f"{k}_score": v for k, v in strategies.items()},
             **{f"{k}_review": m.astype(int) for k, m in masks.items() if k in ("mspc", "v3_selected_knn", "hybrid_knn_mspc")}}
        )
        predictions.to_csv(output / "assessment_predictions.csv", index=False)
        spe_contrib = mspc.contributions(X_new, "spe")
        t2_contrib = mspc.contributions(X_new, "t2")
        flagged = masks["mspc"]
        pareto = (
            pd.DataFrame(
                {
                    "spe_contribution_share": (spe_contrib[flagged].sum() / spe_contrib[flagged].to_numpy().sum()),
                    "t2_contribution_share": (t2_contrib[flagged].clip(lower=0).sum() / t2_contrib[flagged].clip(lower=0).to_numpy().sum()),
                    "times_top5_spe": pd.Series(
                        np.bincount(
                            np.argsort(-spe_contrib[flagged].to_numpy(), axis=1)[:, :5].ravel(),
                            minlength=spe_contrib.shape[1],
                        ),
                        index=spe_contrib.columns,
                    ),
                }
            )
            .rename_axis("sensor")
            .sort_values("spe_contribution_share", ascending=False)
            .reset_index()
        )
        pareto.to_csv(output / "alarm_pareto.csv", index=False)
        example_rows = np.flatnonzero(failures & flagged)
        example_index = (
            int(example_rows[np.argmax(strategies["mspc"][example_rows])]) if example_rows.size else int(np.argmax(strategies["mspc"]))
        )
        example = pd.DataFrame(
            {
                "sensor": spe_contrib.columns,
                "spe_contribution": spe_contrib.iloc[example_index].to_numpy(),
                "t2_contribution": t2_contrib.iloc[example_index].to_numpy(),
            }
        )
        example.to_csv(output / "example_contribution.csv", index=False)

        # 3. Drift and prequential retraining.
        print("3/4 Drift monitoring and retraining simulation", flush=True)
        reference = DriftReference.fit(X_dev)
        drift_table = sensor_drift(reference, X_new)
        informative = informative_sensors(X_dev)
        drift_table["informative"] = informative
        drift_table.sort_values("psi", ascending=False).to_csv(output / "drift_assessment.csv", index=False)

        def fit_knn(X, y):
            return fit_candidate(winner_spec, X, y, seed=config.seed, jobs=config.jobs)

        def score_knn(model, X):
            return predict_score(winner_spec, model, X)

        def fit_mspc(X, y):
            return MSPCModel(variance=chosen["variance"], alpha=config.alpha).fit(X, y)

        def score_mspc(model, X):
            return model.statistics(X).score(chosen["statistic"])

        replay = []
        for model_name, fit, score in (("v3_selected_knn", fit_knn, score_knn), ("mspc", fit_mspc, score_mspc)):
            part = simulate(dataset.X, dataset.y, dataset.timestamps, fit, score, config.retrain)
            part.insert(0, "model", model_name)
            replay.append(part)
        replay = pd.concat(replay, ignore_index=True)
        replay.to_csv(output / "retraining_batches.csv", index=False)
        retrain_summary = summarize(replay)
        retrain_summary.to_csv(output / "retraining_summary.csv", index=False)
        # Drift of each batch versus the initial window, independent of policy.
        initial_rows = len(dataset.y) - config.retrain.batch_size * config.retrain.n_batches
        order = np.argsort(dataset.timestamps.asi8, kind="stable")
        initial_reference = DriftReference.fit(dataset.X.iloc[order[:initial_rows]])
        initial_informative = informative_sensors(dataset.X.iloc[order[:initial_rows]])
        drift_batches = []
        for b in range(config.retrain.n_batches):
            start = initial_rows + b * config.retrain.batch_size
            rows = order[start : start + config.retrain.batch_size]
            table = sensor_drift(initial_reference, dataset.X.iloc[rows])
            table = table.assign(batch=b, informative=initial_informative)
            drift_batches.append(table)
        drift_batches = pd.concat(drift_batches, ignore_index=True)
        drift_batches.to_csv(output / "drift_by_batch.csv", index=False)
        noise_floor = drift_noise_floor(dataset.X.iloc[order[:initial_rows]], config)

        # 4. Sensor ranking stability on chronologically ordered development data.
        print("4/4 Sensor ranking stability", flush=True)
        dev_order = np.argsort(dataset.timestamps[development].asi8, kind="stable")
        stability = run_stability(
            X_dev.iloc[dev_order],
            y_dev[dev_order],
            n_bootstrap=config.stability_bootstrap,
            n_slices=config.stability_slices,
            k=config.stability_top_k,
            seed=config.seed,
            jobs=config.jobs,
        )
        watch = stability["table"]
        # Descriptive follow-up on later data: does the failure/pass gap persist?
        later = X_new[watch.sensor]
        pooled = later.std(ddof=1).replace(0, np.nan)
        watch["assessment_std_mean_difference"] = (
            (later[failures].mean() - later[~failures].mean()) / pooled
        ).to_numpy()
        watch["assessment_psi"] = drift_table.set_index("sensor").loc[watch.sensor, "psi"].to_numpy()
        watch.to_csv(output / "sensor_stability.csv", index=False)

    metrics = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_hash(output / "protocol.json"),
        "mspc_selection_sha256": file_hash(output / "mspc_selection.json"),
        "mspc_model": mspc.summary(),
        "mspc_selected": chosen["candidate"],
        "assessment_rows": int(len(y_new)),
        "assessment_failures": int(y_new.sum()),
        "review_budget": reviews,
        "random_expected_caught": reviews * float(y_new.mean()),
        "random_95_range": random_range,
        "assessment": comparison,
        "catch_overlap_at_20pct": overlap,
        "classic_control_limits": classic,
        "example_contribution_row_id": int(assessment[example_index]),
        "example_contribution_label": int(y_new[example_index]),
        "drift_assessment": {
            "informative_sensors": int(informative.sum()),
            "psi_alert_share": float(drift_table.psi_alert[informative].mean()),
            "ks_alert_share": float(drift_table.ks_alert[informative].mean()),
            "sensors_missing_rate_up_10pts": int((drift_table.missing_rate_change > 0.10).sum()),
        },
        "drift_noise_floor": noise_floor,
        "drift_by_batch_share": drift_batches.loc[drift_batches.informative]
        .groupby("batch")
        .psi_alert.mean()
        .tolist(),
        "retraining": retrain_summary.to_dict(orient="records"),
        "stability": {k: v for k, v in stability.items() if k != "table"},
        "watchlist_tiers": watch.tier.value_counts().to_dict(),
    }
    write_json(output / "metrics.json", metrics)
    from fabops.reporting import generate_report

    generate_report(output)
    print(json.dumps({k: metrics[k] for k in ("mspc_selected", "catch_overlap_at_20pct")}, indent=2))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("fab_monitoring_results"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run(FabConfig(jobs=args.jobs), args.data_dir, args.output, args.overwrite)


if __name__ == "__main__":
    main()

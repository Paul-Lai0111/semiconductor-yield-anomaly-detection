"""Search a fixed catalog using forward validation, then assess frozen review policies."""

import argparse
import json
import platform
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from secom.data import load_dataset
from secom.evaluation import bootstrap_ap
from smartfab.models import CANDIDATES, fit_candidate, predict_score
from smartfab.policies import (
    blend_scores,
    capacity_mask,
    choose_f2_threshold,
    chronological_partition,
    evaluate_policy,
    forward_splits,
    threshold_mask,
)

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 2026
    jobs: int = 2
    review_fraction: float = 0.20
    development_fraction: float = 0.80
    bootstrap: int = 2000

    def __post_init__(self):
        if self.jobs < 1 or self.bootstrap < 1 or not 0 < self.review_fraction < 1:
            raise ValueError(
                "Positive jobs/bootstrap and an interior review fraction are required."
            )
        if not 0 < self.development_fraction < 1:
            raise ValueError("development_fraction must lie strictly between zero and one.")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def ranking(y, scores):
    if len(np.unique(y)) != 2:
        return {"average_precision": None, "roc_auc": None}
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def scored_fit(spec, X, y, target, seed, jobs):
    started = perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model = fit_candidate(spec, X, y, seed=seed, jobs=jobs)
        scores = predict_score(spec, model, target)
    return model, scores, round(perf_counter() - started, 3), [str(w.message) for w in recorded]


def score_blocks(identifier, scores_by_block, splits, row_ids, y, config):
    decisions, labels, rows = [], [], []
    for block, ((train, validation), scores) in enumerate(
        zip(splits, scores_by_block, strict=True)
    ):
        ids = row_ids[validation]
        mask = capacity_mask(scores, ids, config.review_fraction, config.seed)
        result = evaluate_policy(y[validation], mask)
        rows.append(
            {
                "candidate": identifier,
                "block": block,
                "train_rows": len(train),
                "validation_rows": len(validation),
                **ranking(y[validation], scores),
                **result,
            }
        )
        decisions.extend(mask)
        labels.extend(y[validation])
    combined = evaluate_policy(np.asarray(labels), np.asarray(decisions))
    ap = [row["average_precision"] for row in rows if row["average_precision"] is not None]
    summary = {
        "candidate": identifier,
        "mean_block_ap": float(np.mean(ap)) if ap else 0.0,
        **combined,
    }
    return rows, summary


def search_development(X, y, timestamps, row_ids, catalog, config, output):
    splits = forward_splits(timestamps)
    summaries, block_rows, predictions = [], [], []
    scores = {}
    failed = []
    for index, spec in enumerate(catalog):
        identifier = spec["id"]
        print(f"Search {index + 1}/{len(catalog)}: {identifier}", flush=True)
        blocks, timings, messages = [], [], []
        try:
            for block, (train, validation) in enumerate(splits):
                _, predicted, seconds, notes = scored_fit(
                    spec,
                    X.iloc[train],
                    y[train],
                    X.iloc[validation],
                    config.seed + block,
                    config.jobs,
                )
                blocks.append(predicted)
                timings.append(seconds)
                messages.extend(notes)
            rows, summary = score_blocks(identifier, blocks, splits, row_ids, y, config)
            summary.update(
                family=spec["family"], fit_seconds=sum(timings), warning_count=len(messages)
            )
            for block, row in enumerate(rows):
                row["fit_seconds"] = timings[block]
            summaries.append(summary)
            block_rows.extend(rows)
            scores[identifier] = blocks
            for block, ((_, validation), predicted) in enumerate(zip(splits, blocks, strict=True)):
                predictions.append(
                    pd.DataFrame(
                        {
                            "candidate": identifier,
                            "block": block,
                            "row_id": row_ids[validation],
                            "label": y[validation],
                            "score": predicted,
                        }
                    )
                )
            print(
                f"  validation recall@20%={summary['recall']:.3f}, AP={summary['mean_block_ap']:.3f}",
                flush=True,
            )
            if messages:
                failed.append(
                    {
                        "candidate": identifier,
                        "status": "warning",
                        "messages": sorted(set(messages)),
                    }
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failed.append({"candidate": identifier, "status": "failed", "messages": [str(error)]})
            print(f"  unavailable: {error}", flush=True)

    supervised = {
        spec["id"] for spec in catalog if not spec["novelty"] and spec["family"] != "dummy"
    }
    # Deterministic catalog order breaks exact metric ties, not measured runtime.
    ordered = sorted(summaries, key=lambda row: (-row["true_positives"], -row["mean_block_ap"]))
    leaders = [row["candidate"] for row in ordered if row["candidate"] in supervised][:3]
    if len(leaders) < 2:
        raise ValueError("At least two supervised candidates must finish the fixed search.")
    blend = "rank_blend_top2"
    blended = [blend_scores([scores[name][block] for name in leaders[:2]]) for block in range(len(splits))]
    rows, summary = score_blocks(blend, blended, splits, row_ids, y, config)
    summary.update(family="rank_blend", fit_seconds=0.0, warning_count=0)
    summaries.append(summary)
    block_rows.extend(rows)
    scores[blend] = blended
    for block, ((_, validation), predicted) in enumerate(zip(splits, blended, strict=True)):
        predictions.append(
            pd.DataFrame(
                {
                    "candidate": blend,
                    "block": block,
                    "row_id": row_ids[validation],
                    "label": y[validation],
                    "score": predicted,
                }
            )
        )
    ordered = sorted(summaries, key=lambda row: (-row["true_positives"], -row["mean_block_ap"]))
    table = pd.DataFrame(ordered)
    table.insert(0, "selection_rank", np.arange(1, len(table) + 1))
    table.to_csv(output / "search_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(output / "search_blocks.csv", index=False)
    pd.concat(predictions).to_csv(output / "forward_predictions.csv", index=False)
    write_json(output / "fit_notes.json", failed)
    assignments = []
    for block, (train, validation) in enumerate(splits):
        assignments.extend(
            {"block": block, "row_id": int(row_ids[index]), "role": role}
            for role, indices in (("train", train), ("validation", validation))
            for index in indices
        )
    pd.DataFrame(assignments).to_csv(output / "forward_membership.csv", index=False)
    return ordered[0]["candidate"], leaders, scores, splits, table


def finalist_oof(X, y, ids, specs, config, output, filename="finalist_oof.csv"):
    splits = list(StratifiedKFold(5, shuffle=True, random_state=config.seed).split(X, y))
    scores, records = {}, []
    for spec in specs:
        predicted = np.full(len(y), np.nan)
        for fold, (train, validation) in enumerate(splits):
            _, values, _, _ = scored_fit(
                spec, X.iloc[train], y[train], X.iloc[validation], config.seed + fold, config.jobs
            )
            predicted[validation] = values
            records.append(
                pd.DataFrame(
                    {
                        "candidate": spec["id"],
                        "fold": fold,
                        "row_id": ids[validation],
                        "label": y[validation],
                        "score": values,
                    }
                )
            )
        if not np.isfinite(predicted).all():
            raise ValueError("Incomplete finalist OOF coverage.")
        scores[spec["id"]] = predicted
    pd.concat(records).to_csv(output / filename, index=False)
    return scores, splits


def scenario_table(y, candidates, config):
    rows = []
    n, failures = len(y), int(np.sum(y))
    reviews = int(np.floor(n * config.review_fraction))
    strategies = {
        "no_additional_review": (0, 0.0, 0.0),
        "review_everything": (n, float(failures), float(n - failures)),
        "random_same_capacity_expected": (
            reviews,
            reviews * failures / n,
            reviews * (1 - failures / n),
        ),
    }
    for identifier, metrics in candidates.items():
        strategies[identifier] = (
            metrics["alarm_count"],
            metrics["true_positives"],
            metrics["false_positives"],
        )
    for escape_cost in (1, 2, 5, 10, 20, 50, 100):
        for effectiveness in (0.25, 0.50, 0.75, 1.0):
            for hold_cost in (0.0, 2.0):
                costs = {}
                for name, (count, caught, false) in strategies.items():
                    costs[name] = (
                        count
                        + (failures - effectiveness * caught) * escape_cost
                        + false * hold_cost
                    )
                random_cost = costs["random_same_capacity_expected"]
                for name, value in costs.items():
                    rows.append(
                        {
                            "strategy": name,
                            "escape_cost_per_review_cost": escape_cost,
                            "assumed_intervention_effectiveness": effectiveness,
                            "passing_hold_cost_per_review_cost": hold_cost,
                            "simulated_total_cost_units": value,
                            "difference_vs_random": value - random_cost,
                            "scope": "Hypothetical review/intervention cost units, not measured production cost or throughput.",
                        }
                    )
    return pd.DataFrame(rows)


def run_study(config, data_dir, output, overwrite=False, catalog=CANDIDATES):
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            "Output is not empty; choose another --output or explicitly --overwrite."
        )
    root = Path(__file__).resolve().parents[1]
    protected = [
        root / name
        for name in ("results", "secom", "smartfab", "scripts", "images", "docs", "supplementary")
    ]
    if any(
        output.resolve() == p
        or output.resolve().is_relative_to(p)
        or p.is_relative_to(output.resolve())
        for p in protected
    ):
        raise ValueError("New study output must be separate from sources and historical evidence.")
    output.mkdir(parents=True, exist_ok=True)
    for file in output.rglob("*"):
        if file.is_symlink() or (file.is_file() and file.stat().st_nlink > 1):
            raise ValueError("Refusing linked output files.")
    dataset = load_dataset(data_dir)
    development, assessment = chronological_partition(
        dataset.timestamps, config.development_fraction
    )
    X, y = dataset.X.iloc[development], dataset.y[development]
    source_paths = [
        *sorted((root / "smartfab").glob("*.py")),
        *sorted((root / "secom").glob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    source_hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in source_paths}
    split = pd.DataFrame(
        {
            "row_id": np.arange(len(dataset.y)),
            "timestamp": dataset.timestamps,
            "partition": np.where(
                np.isin(np.arange(len(dataset.y)), development),
                "development",
                "temporal_assessment",
            ),
            "label": dataset.y,
        }
    )
    split.to_csv(output / "split_manifest.csv", index=False)
    protocol = {
        "schema_version": 3,
        "frozen_before_search_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "dataset": dataset.manifest,
        "catalog": list(catalog),
        "source_sha256": source_hashes,
        "split_manifest_sha256": file_hash(output / "split_manifest.csv"),
        "primary_objective": "Highest total failures caught under exact per-block 20% review capacity on three expanding chronological development blocks.",
        "tie_breaks": "Higher mean block AP, then fixed catalog order; blend comes last on exact ties.",
        "batch_ties": "SHA256(seed:original_row_id), ascending; labels never break ties.",
        "blend": "Equal mean within-batch percentile ranks from the two highest-ranked supervised base candidates; no weight tuning.",
        "secondary_threshold": "Five-fold development OOF maximum F2 with at most 20% OOF alerts; does not replace capacity selection.",
        "assessment_policy": "Freeze winner, blend members and threshold before scoring the latest temporal block. No model or threshold reselection from assessment results.",
        "dataset_history": "SECOM and its labels were explored in v2. This is a protected within-v3 temporal boundary, not historically unseen or external data.",
        "timing_limit": "Batch ranking requires the complete assessment batch. Availability of sensor features and labels at a real production decision time is not verified.",
    }
    write_json(output / "protocol.json", protocol)
    protocol_hash = file_hash(output / "protocol.json")
    with threadpool_limits(limits=config.jobs):
        winner, leaders, _, _, search = search_development(
            X, y, dataset.timestamps[development], development, catalog, config, output
        )
        specs = {spec["id"]: spec for spec in catalog}
        finalist_scores, oof_splits = finalist_oof(
            X, y, development, [specs[name] for name in leaders], config, output
        )
        is_blend = winner == "rank_blend_top2"
        members = leaders[:2] if is_blend else [winner]
        if is_blend:
            winner_oof = np.empty(len(y))
            for _, validation in oof_splits:
                winner_oof[validation] = blend_scores([finalist_scores[name][validation] for name in members])
        elif winner in finalist_scores:
            winner_oof = finalist_scores[winner]
        else:
            additional, _ = finalist_oof(X, y, development, [specs[winner]], config, output, "winner_oof.csv")
            winner_oof = additional[winner]
        threshold = choose_f2_threshold(y, winner_oof, config.review_fraction)
        selection = {
            "frozen_before_assessment_utc": datetime.now(UTC).isoformat(),
            "winner": winner,
            "members": members,
            "member_specs": [specs[name] for name in members],
            "is_rank_blend": is_blend,
            "supervised_finalists": leaders,
            "threshold_policy": threshold,
            "primary_review_fraction": config.review_fraction,
            "development_selection": search.iloc[0].to_dict(),
            "protocol_sha256": protocol_hash,
        }
        write_json(output / "selection.json", selection)
        selection_hash = file_hash(output / "selection.json")
        print(f"FROZEN BEFORE TEMPORAL ASSESSMENT: {winner}, members={members}", flush=True)

        required = list(dict.fromkeys([*members, "v2_rf_knn_smote", "dummy_prior"]))
        models, raw_assessment = {}, {}
        for identifier in required:
            models[identifier], raw_assessment[identifier], _, _ = scored_fit(
                specs[identifier], X, y, dataset.X.iloc[assessment], config.seed, config.jobs
            )
        winner_scores = blend_scores([raw_assessment[name] for name in members]) if is_blend else raw_assessment[winner]
        assessment_scores = {
            "selected": winner_scores,
            "v2_reference": raw_assessment["v2_rf_knn_smote"],
            "dummy_reference": raw_assessment["dummy_prior"],
        }
        target = dataset.y[assessment]
        results, prediction_rows, budgets = {}, [], []
        for name, values in assessment_scores.items():
            mask = capacity_mask(values, assessment, config.review_fraction, config.seed)
            metrics = evaluate_policy(target, mask)
            results[name] = {**ranking(target, values), "capacity": metrics}
            for budget in BUDGETS:
                budgets.append(
                    {
                        "strategy": name,
                        "review_fraction": budget,
                        **evaluate_policy(
                            target, capacity_mask(values, assessment, budget, config.seed)
                        ),
                    }
                )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "strategy": name,
                        "row_id": assessment,
                        "label": target,
                        "score": values,
                        "review": mask.astype(int),
                    }
                )
            )
        results["selected"]["fixed_threshold"] = evaluate_policy(
            target, threshold_mask(winner_scores, threshold["threshold"])
        )
        results["selected"]["ap_bootstrap_95"] = bootstrap_ap(
            target, winner_scores, config.bootstrap, config.seed
        )
        pd.concat(prediction_rows).to_csv(output / "assessment_predictions.csv", index=False)
        pd.DataFrame(budgets).to_csv(output / "assessment_budgets.csv", index=False)
        scenario_table(
            target, {key: value["capacity"] for key, value in results.items()}, config
        ).to_csv(output / "cost_scenarios.csv", index=False)
        model_dir = output / "artifacts"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(
            {
                "schema_version": 3,
                "winner": winner,
                "is_rank_blend": is_blend,
                "members": [{"spec": specs[name], "pipeline": models[name]} for name in members],
                "sensor_columns": dataset.X.columns.tolist(),
                "review_fraction": config.review_fraction,
                "seed": config.seed,
                "secondary_threshold": threshold["threshold"],
                "protocol_sha256": protocol_hash,
                "selection_sha256": selection_hash,
            },
            model_dir / "selected_model.joblib",
            compress=3,
        )

    random_expected = int(
        np.floor(len(assessment) * config.review_fraction)
    ) * float(target.mean())
    random_interval = hypergeom.ppf(
        [0.025, 0.975],
        len(target),
        int(target.sum()),
        int(np.floor(len(target) * config.review_fraction)),
    ).tolist()
    metrics = {
        "schema_version": 3,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "winner": winner,
        "members": members,
        "protocol_sha256": protocol_hash,
        "selection_sha256": selection_hash,
        "development_rows": len(development),
        "development_failures": int(y.sum()),
        "assessment_rows": len(assessment),
        "assessment_failures": int(target.sum()),
        "last_training_timestamp": dataset.timestamps[development].max().isoformat(),
        "first_assessment_timestamp": dataset.timestamps[assessment].min().isoformat(),
        "random_same_capacity_expected_caught": random_expected,
        "random_same_capacity_95_range": random_interval,
        "assessment": results,
        "environment": {
            "python": platform.python_version(),
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
                )
            },
        },
        "limits": [
            "Previously explored public dataset; within-run temporal assessment only.",
            "Small failure count and shared development blocks leave selection uncertainty.",
            "Batch ranking, uncalibrated scores and unverified decision-time feature availability.",
            "Cost scenarios assume intervention effectiveness; no measured yield, throughput, or financial benefit.",
        ],
    }
    if protocol_hash != file_hash(output / "protocol.json") or selection_hash != file_hash(
        output / "selection.json"
    ):
        raise ValueError("Frozen protocol or selection changed during assessment.")
    write_json(output / "metrics.json", metrics)
    from smartfab.reporting import generate_report

    generate_report(output)
    print(json.dumps({"winner": winner, "assessment": results["selected"]}, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("manufacturing_results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    run_study(SearchConfig(jobs=args.jobs), args.data_dir, args.output, args.overwrite)


if __name__ == "__main__":
    main()

"""Post-hoc stability and review-capacity checks using development observations only."""

import argparse
import json
import platform
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from secom.data import load_dataset
from secom.evaluation import operating_metrics
from secom.experiment import StudyConfig, cross_validate, write_json
from secom.modeling import LABELS

STABILITY_CANDIDATES = ("rf_median", "rf_knn", "rf_knn_smote", "xgb_knn")
DEFAULT_SEEDS = (17, 43, 89)
BUDGETS = (0.05, 0.10, 0.20)


def review_mask(scores, fraction: float) -> np.ndarray:
    """Admit whole descending score groups until the block's review budget is exhausted."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Capacity analysis needs a nonempty finite one-dimensional score array.")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Review fraction must lie in [0, 1].")
    values, counts = np.unique(scores, return_counts=True)
    cumulative = np.cumsum(counts[::-1])
    permitted = values[::-1][cumulative <= int(np.floor(len(scores) * fraction))]
    return scores >= permitted[-1] if len(permitted) else np.zeros(len(scores), dtype=bool)


def capacity_tables(oof: pd.DataFrame, budgets=BUDGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks, summaries = [], []
    for name, candidate in oof.groupby("candidate", sort=False):
        for budget in budgets:
            labels, reviewed = [], []
            max_reviews = 0
            for fold, frame in candidate.groupby("fold", sort=True):
                mask = review_mask(frame.score.to_numpy(), budget)
                maximum = int(np.floor(len(frame) * budget))
                metrics = operating_metrics(frame.label.to_numpy(), mask.astype(int), 0.5)
                metrics.pop("threshold")
                blocks.append(
                    {
                        "candidate": name,
                        "budget_fraction": budget,
                        "fold": int(fold),
                        "rows": len(frame),
                        "maximum_reviews": maximum,
                        "unused_capacity": maximum - int(mask.sum()),
                        **metrics,
                    }
                )
                labels.extend(frame.label)
                reviewed.extend(mask)
                max_reviews += maximum
            metrics = operating_metrics(np.asarray(labels), np.asarray(reviewed, dtype=int), 0.5)
            metrics.pop("threshold")
            summaries.append(
                {
                    "candidate": name,
                    "budget_fraction": budget,
                    "rows": len(labels),
                    "maximum_reviews": max_reviews,
                    "unused_capacity": max_reviews - int(np.sum(reviewed)),
                    **metrics,
                }
            )
    return pd.DataFrame(blocks), pd.DataFrame(summaries)


def validate_evidence(evidence: Path, dataset) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    protocol = json.loads((evidence / "protocol.json").read_text())
    metrics = json.loads((evidence / "metrics.json").read_text())
    if sha256((evidence / "protocol.json").read_bytes()).hexdigest() != metrics["protocol_sha256"]:
        raise ValueError("Original metrics and protocol hashes disagree.")
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in [
        root / "main.py",
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "secom").glob("*.py")),
    ]:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != protocol["source_sha256"]:
        raise ValueError("Original experiment source changed; use its frozen implementation.")
    if dataset.manifest["sha256"] != metrics["dataset"]["sha256"]:
        raise ValueError("Raw source hashes differ from the original study.")
    manifest = pd.read_csv(evidence / "split_manifest.csv")
    if (
        manifest.row_id.duplicated().any()
        or set(manifest.row_id) != set(range(len(dataset.y)))
        or set(manifest.partition) != {"development", "holdout"}
        or not pd.api.types.is_integer_dtype(manifest.row_id)
    ):
        raise ValueError("Invalid original split manifest.")
    manifest = manifest.set_index("row_id").sort_index()
    if not np.array_equal(manifest.label.to_numpy(), dataset.y):
        raise ValueError("Saved labels do not match raw source row identities.")
    original = protocol["config"]
    expected_dev, _ = train_test_split(
        np.arange(len(dataset.y)),
        test_size=original["test_fraction"],
        stratify=dataset.y,
        random_state=original["seed"],
    )
    if set(manifest.index[manifest.partition == "development"]) != set(expected_dev):
        raise ValueError("Saved development membership differs from the frozen outer split.")
    expected_folds = np.full(len(dataset.y), -1)
    for fold, (_, validation) in enumerate(
        StratifiedKFold(original["folds"], shuffle=True, random_state=original["seed"]).split(
            expected_dev, dataset.y[expected_dev]
        )
    ):
        expected_folds[expected_dev[validation]] = fold
    if not np.array_equal(manifest.validation_fold, expected_folds):
        raise ValueError("Saved validation folds differ from the frozen fold assignments.")
    oof = pd.read_csv(evidence / "oof_predictions.csv", float_precision="round_trip")
    dev = manifest.loc[manifest.partition == "development"]
    if set(oof.candidate) != set(protocol["candidates_in_tie_break_order"]):
        raise ValueError("Original OOF candidates are incomplete.")
    for _, frame in oof.groupby("candidate"):
        ordered = frame.set_index("row_id").sort_index()
        if (
            frame.row_id.duplicated().any()
            or not ordered.index.equals(dev.index)
            or not np.array_equal(ordered.label, dev.label)
            or not np.array_equal(ordered.fold, dev.validation_fold)
            or not np.isfinite(ordered.score).all()
        ):
            raise ValueError(
                "OOF records must match the original development membership and folds."
            )
    return protocol, dev, oof


def repeated_comparisons(dataset, development, seeds, folds, trees, jobs):
    # Canonical row order is deliberate: these seeds are new partitions, not a replay of seed 2026.
    row_ids = np.asarray(sorted(development), dtype=int)
    X, y = dataset.X.iloc[row_ids], dataset.y[row_ids]
    summaries, fold_rows, predictions, assignments = [], [], [], []
    for repeat, seed in enumerate(seeds):
        config = StudyConfig(seed=seed, folds=folds, trees=trees, jobs=jobs, temporal_check=False)
        shared = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
        membership = np.empty(len(y), dtype=int)
        for fold, (_, validation) in enumerate(shared):
            membership[validation] = fold
        assignments.append(
            pd.DataFrame({"repeat": repeat, "seed": seed, "row_id": row_ids, "fold": membership})
        )
        for name in STABILITY_CANDIDATES:
            print(f"Repeat {repeat + 1}/{len(seeds)}, seed {seed}: {name}", flush=True)
            scores, details, summary = cross_validate(name, X, y, shared, config)
            summaries.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    **summary,
                    "oof_alarm_rate": float(np.mean(scores >= summary["oof_threshold"])),
                }
            )
            fold_rows.extend({"repeat": repeat, "seed": seed, **row} for row in details)
            predictions.append(
                pd.DataFrame(
                    {
                        "candidate": name,
                        "repeat": repeat,
                        "seed": seed,
                        "row_id": row_ids,
                        "fold": membership,
                        "label": y,
                        "score": scores,
                    }
                )
            )
    summary = pd.DataFrame(summaries)
    summary["rank_among_four"] = (
        summary.groupby("repeat").mean_fold_ap.rank(method="min", ascending=False).astype(int)
    )
    return summary, pd.DataFrame(fold_rows), pd.concat(predictions), pd.concat(assignments)


def create_report(output, provenance, repeats, capacity):
    paired = repeats.pivot(index="repeat", columns="candidate", values="mean_fold_ap")
    differences = paired.rf_knn_smote - paired.rf_knn
    positive_repeats = int((differences > 0).sum())
    selected_capacity = capacity.loc[
        capacity.candidate == provenance["original_selected_candidate"]
    ]
    lines = [
        "# Development Stability And Review Capacity",
        "",
        "Supplementary, post-hoc development-only analysis. The original selected model, threshold, holdout predictions, and reported test metrics are unchanged.",
        "",
        f"Original protocol SHA-256: `{provenance['original_protocol_sha256']}`.",
        "",
        "## Takeaways",
        "",
        f"- SMOTE has higher mean AP than plain KNN/RF in **{positive_repeats}/{len(differences)}** matched repeats. The differences are descriptive; the highest-ranked candidate can still change across partitions.",
        "- For the original selected pipeline, the review budget exposes a concrete workload/detection tradeoff: "
        + "; ".join(
            f"**{row.budget_fraction:.0%} cap: {row.alarm_count} reviews, {row.true_positives}/{row.failure_support} failures caught**"
            for row in selected_capacity.itertuples()
        )
        + ".",
        "- These completed checks strengthen the interpretation of the original study without replacing its selected model or reusing its holdout for another search.",
        "",
        "![Development stability and review workload, with observed takeaways](development_diagnostics.png)",
        "",
        "## Sensitivity To Fold Assignment And Estimator Randomness",
        "",
        "The same original development observations are reused with three new five-fold partitions. Each candidate uses the same partitions within a repeat. Splitter seed is the listed seed; estimator and SMOTE seed is that seed plus fold index. Rows are ordered by original row ID. Only four existing candidates are checked, with the original fixed hyperparameters.",
        "",
        "| Candidate | "
        + " | ".join(f"Seed {seed}: mean AP (rank)" for seed in provenance["seeds"])
        + " | OOF threshold range |",
        "|---|" + "---:|" * (len(provenance["seeds"]) + 1),
    ]
    for name in STABILITY_CANDIDATES:
        frame = repeats.loc[repeats.candidate == name]
        values = [f"{row.mean_fold_ap:.4f} ({row.rank_among_four})" for row in frame.itertuples()]
        lines.append(
            f"| {LABELS[name]} | "
            + " | ".join(values)
            + f" | {frame.oof_threshold.min():.4f}-{frame.oof_threshold.max():.4f} |"
        )
    lines.extend(
        [
            "",
            "Paired mean-AP differences (SMOTE minus plain KNN/RF), using matching repeats: "
            + ", ".join(f"**{value:+.4f}**" for value in differences)
            + ".",
            "",
            "Ranks are among these four candidates. The repeats reuse observations and overlapping training samples; fold SD, rank counts, and AP ranges are descriptive, not independent confidence intervals, p-values, or probabilities of superiority. Thresholds are optimized development diagnostics and do not establish transfer to a full-development refit. These checks do not replace the original selection or justify a new holdout search.",
            "",
            "## Hypothetical Review Capacity",
            "",
            "The original OOF scores are ranked within each original validation fold. Each block may review at most floor(budget * block rows). Equal scores move together: a group that would exceed capacity and every lower group are excluded. A constant-score model can therefore issue no reviews. Aggregate metrics are computed from summed confusion counts, not averages of fold precision or recall.",
            "",
            "| Candidate | Budget per fold | Maximum reviews | Actual reviews | Unused | TP | FP | FN | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in capacity.itertuples():
        lines.append(
            f"| {LABELS[row.candidate]} | {row.budget_fraction:.0%} | {row.maximum_reviews} | {row.alarm_count} | {row.unused_capacity} | {row.true_positives} | {row.false_positives} | {row.false_negatives} | {row.precision:.4f} | {row.recall:.4f} |"
        )
    lines.extend(
        [
            "",
            "Budget flooring is per block; a 5% budget across the original five folds allows 60 reviews in total, before tie exclusions. These random validation folds are hypothetical blocks, not actual factory inspection batches. Reusing development outcomes for model selection and policy exploration makes retrospectively preferred policies optimistic. No inspection costs, prevented losses, or future capacity compliance are established.",
            "",
            "## Evidence And Reproduction",
            "",
            "- `provenance.json`: original evidence hashes, supplementary source hashes, package versions, fixed configuration and seed policy.",
            "- `repeat_metrics.csv`, `repeat_fold_metrics.csv`, `repeat_assignments.csv`, `repeat_oof_predictions.csv`: every repeated score and membership for reconstruction.",
            "- `capacity_by_fold.csv`, `capacity_summary.csv`: review constraints and confusion counts from the original OOF predictions.",
            "",
            "Run `uv run python -m scripts.development_diagnostics`. An existing output directory requires `--overwrite`. The analysis reads original development membership, not a new outer split. It never fits on or predicts the original holdout. The raw-data loader still validates the full published source files.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in STABILITY_CANDIDATES:
        frame = repeats.loc[repeats.candidate == name]
        axes[0].plot(range(1, len(frame) + 1), frame.mean_fold_ap, "o-", label=LABELS[name])
    axes[0].set(
        xticks=range(1, len(provenance["seeds"]) + 1),
        xticklabels=provenance["seeds"],
        xlabel="Development repeat seed",
        ylabel="Mean fold average precision",
        title=f"SMOTE > plain KNN/RF in {positive_repeats}/{len(differences)} repeats\nTop-ranked candidate still depends on the split",
    )
    selected = provenance["original_selected_candidate"]
    frame = capacity.loc[capacity.candidate == selected]
    positions = np.arange(len(frame))
    axes[1].bar(
        positions, frame.true_positives, label="Failures caught", color="#3574a8", width=0.56
    )
    axes[1].bar(
        positions,
        frame.false_positives,
        bottom=frame.true_positives,
        label="Passing rows reviewed",
        color="#d5dce4",
        width=0.56,
    )
    for position, row in enumerate(frame.itertuples()):
        axes[1].text(
            position,
            row.alarm_count + 5,
            f"{row.true_positives}/{row.failure_support} caught\n{row.false_positives} pass reviews",
            ha="center",
            fontsize=8,
        )
    axes[1].set(
        xlabel="Maximum review fraction per original validation fold",
        ylabel="Reviewed development observations",
        xticks=positions,
        xticklabels=[f"{value:.0%}" for value in frame.budget_fraction],
        title="More capacity catches more failures\nExtra passing-row reviews remain substantial",
        ylim=(0, max(frame.alarm_count) * 1.3),
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Post-hoc development diagnostics | original holdout results unchanged", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(output / "development_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(
    evidence: Path, data_dir: Path, output: Path, *, seeds=DEFAULT_SEEDS, overwrite=False
):
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output is not empty; choose a new directory or --overwrite.")
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(seed < 0 for seed in seeds):
        raise ValueError("Supply three distinct nonnegative repeat seeds.")
    # A custom output must not overwrite any original evidence or source directory.
    root = Path(__file__).resolve().parents[1]
    protected = [
        evidence.resolve(),
        root / "secom",
        root / "scripts",
        root / "images",
        data_dir.resolve(),
    ]
    if any(
        output.resolve() == path
        or output.resolve().is_relative_to(path)
        or path.is_relative_to(output.resolve())
        for path in protected
    ):
        raise ValueError("Supplementary output must be separate from source and original evidence.")
    destinations = (
        "provenance.json",
        "repeat_metrics.csv",
        "repeat_fold_metrics.csv",
        "repeat_oof_predictions.csv",
        "repeat_assignments.csv",
        "capacity_by_fold.csv",
        "capacity_summary.csv",
        "REPORT.md",
        "development_diagnostics.png",
    )
    for name in destinations:
        path = output / name
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink > 1)
        ):
            raise ValueError("Supplementary destinations must be ordinary unlinked files.")
    dataset = load_dataset(data_dir)
    protocol, development, oof = validate_evidence(evidence, dataset)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__),
        root / "secom/modeling.py",
        root / "secom/evaluation.py",
        root / "secom/experiment.py",
    ]
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "Post-hoc development-only descriptive diagnostics; no model reselection or holdout evaluation.",
        "original_protocol_sha256": sha256((evidence / "protocol.json").read_bytes()).hexdigest(),
        "original_source_sha256": protocol["source_sha256"],
        "original_selected_candidate": protocol["selected_candidate"],
        "input_sha256": {
            name: sha256((evidence / name).read_bytes()).hexdigest()
            for name in ("split_manifest.csv", "oof_predictions.csv", "metrics.json")
        },
        "supplementary_source_sha256": {
            path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "raw_file_sha256": dataset.manifest["sha256"],
        "seeds": list(seeds),
        "folds": 5,
        "trees": protocol["config"]["trees"],
        "candidates": list(STABILITY_CANDIDATES),
        "development_rows": len(development),
        "development_failures": int(development.label.sum()),
        "seed_policy": "Canonical ascending row IDs; splitter=repeat seed; estimator/SMOTE=repeat seed + fold index.",
        "review_budgets": list(BUDGETS),
        "review_policy": "Within original validation folds; whole tied score groups; floor per-block capacity.",
        "environment": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")
            },
        },
    }
    write_json(output / "provenance.json", provenance)
    with threadpool_limits(limits=protocol["config"]["jobs"]):
        repeats, folds, predictions, assignments = repeated_comparisons(
            dataset,
            development.index,
            seeds,
            5,
            protocol["config"]["trees"],
            protocol["config"]["jobs"],
        )
    repeats.to_csv(output / "repeat_metrics.csv", index=False)
    folds.to_csv(output / "repeat_fold_metrics.csv", index=False)
    predictions.to_csv(output / "repeat_oof_predictions.csv", index=False)
    assignments.to_csv(output / "repeat_assignments.csv", index=False)
    blocks, capacity = capacity_tables(oof)
    blocks.to_csv(output / "capacity_by_fold.csv", index=False)
    capacity.to_csv(output / "capacity_summary.csv", index=False)
    create_report(output, provenance, repeats, capacity)
    provenance["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "provenance.json", provenance)
    print(f"Supplement complete: {output / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("supplementary/development"))
    parser.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_diagnostics(
        args.evidence, args.data_dir, args.output, seeds=args.seeds, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()

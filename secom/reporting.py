"""Generate every published figure and result table from the current experiment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

from secom.modeling import CANDIDATES, LABELS

COLORS = ["#9ca3af", "#7c3aed", "#3574a8", "#168a8a", "#8f6f36", "#bf6d29", "#173f5f"]


def create_figures(directory, dataset, development, holdout, fitted, oof, test_scores, metrics):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 160,
            "savefig.facecolor": "white",
            "axes.labelcolor": "#263442",
            "text.color": "#263442",
        }
    )

    def save(figure, filename):
        figure.savefig(directory / filename, bbox_inches="tight")
        plt.close(figure)

    dev = {row["candidate"]: row for row in metrics["development"]}
    results = metrics["holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    positions = np.arange(len(CANDIDATES))
    for index, name in enumerate(CANDIDATES):
        axes[0].errorbar(
            dev[name]["mean_fold_ap"],
            index,
            xerr=dev[name]["std_fold_ap"],
            fmt="o",
            color=COLORS[index],
            capsize=3,
        )
        low, high = results[name]["ap_bootstrap_95"]
        # Percentile bootstrap intervals need not contain the original point estimate.
        axes[1].hlines(index, low, high, color=COLORS[index], linewidth=2)
        axes[1].plot(results[name]["average_precision"], index, "o", color=COLORS[index])
    axes[0].set_yticks(
        positions,
        [
            LABELS[name] + (" *" if name == metrics["selected_candidate"] else "")
            for name in CANDIDATES
        ],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Development: mean fold AP +/- fold SD")
    axes[1].set_title("Holdout: AP with conditional 95% bootstrap interval")
    for ax, prevalence in zip(
        axes, [dataset.y[development].mean(), dataset.y[holdout].mean()], strict=True
    ):
        ax.axvline(prevalence, color="#9ca3af", linestyle="--", label="Failure prevalence")
        ax.set_xlabel("Average precision (AP)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Controlled comparisons | * selected using development folds only", y=1.02)
    fig.tight_layout()
    save(fig, "model_comparison.png")

    selected = metrics["selected_candidate"]
    names = list(dict.fromkeys([selected, "xgb_knn", "rf_median"]))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name in names:
        precision, recall, _ = precision_recall_curve(dataset.y[holdout], test_scores[name])
        ax.step(
            recall,
            precision,
            where="post",
            label=f"{LABELS[name]} | AP={results[name]['average_precision']:.3f}",
        )
        point = results[name]["selected"]
        ax.scatter(point["recall"], point["precision"], s=48)
    ax.axhline(
        dataset.y[holdout].mean(),
        color="#9ca3af",
        linestyle="--",
        label="Holdout failure prevalence",
    )
    ax.set(
        xlabel="Failure recall",
        ylabel="Failure precision",
        xlim=(0, 1.02),
        ylim=(0, 1.03),
        title="Held-out precision-recall curves | dots use frozen OOF thresholds",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "pr_curve.png")

    precision, recall, thresholds = precision_recall_curve(dataset.y[development], oof["xgb_knn"])
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for values, label in [(precision[:-1], "Precision"), (recall[:-1], "Recall"), (f1, "F1")]:
        ax.plot(thresholds, values, label=label)
    ax.axvline(
        dev["xgb_knn"]["oof_threshold"],
        color="#263442",
        linestyle="--",
        label="OOF-selected threshold",
    )
    ax.set(
        xlabel="XGBoost score threshold (not calibrated probability)",
        ylabel="Development OOF metric",
        title="Operating-point choice uses development labels only",
        ylim=(0, 1.03),
    )
    ax.legend()
    fig.tight_layout()
    save(fig, "threshold_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    maximum = max(results[selected][policy]["true_negatives"] for policy in ["default", "selected"])
    for ax, policy in zip(axes, ["default", "selected"], strict=True):
        r = results[selected][policy]
        matrix = np.array(
            [
                [r["true_negatives"], r["false_positives"]],
                [r["false_negatives"], r["true_positives"]],
            ]
        )
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(
                col,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > maximum * 0.6 else "#263442",
                fontsize=16,
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Pass", "Fail"],
            yticklabels=["Pass", "Fail"],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{'Default' if policy == 'default' else 'Frozen OOF'} threshold = {r['threshold']:.4f}",
        )
    fig.suptitle(f"Holdout confusion counts | {LABELS[selected]}")
    fig.tight_layout()
    save(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, label in [("rf_median", "Median"), ("rf_knn", "KNN")]:
        transformed = fitted[name][:-1].transform(dataset.X.iloc[development])
        variance = np.cumsum(PCA(svd_solver="full").fit(transformed).explained_variance_ratio_)
        count = int(np.searchsorted(variance, 0.9) + 1)
        ax.plot(
            np.arange(1, len(variance) + 1), variance, label=f"{label}: 90% at {count} components"
        )
    ax.axhline(0.9, linestyle="--", color="#9ca3af")
    ax.set(
        xlabel="Components",
        ylabel="Cumulative explained variance",
        title="PCA diagnostic | fitted on full development data only",
        ylim=(0, 1.03),
    )
    ax.legend()
    fig.tight_layout()
    save(fig, "pca_variance_comparison.png")

    sensors = fitted["xgb_knn"]["sensors"].get_feature_names_out()
    gains = fitted["xgb_knn"]["model"].feature_importances_
    order = np.argsort(-gains, kind="stable")[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(sensors[order], gains[order], color="#3574a8")
    ax.set(
        xlabel="Normalized mean split gain (training data)",
        title="XGBoost: 15 highest gain-importance sensors",
    )
    fig.text(
        0.5,
        0.01,
        "Anonymous sensor IDs; importance is not physical causality or a validated reduced model.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "feature_importance.png")

    missing = dataset.X.iloc[development].isna().mean().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(missing)), missing, color="#3574a8")
    ax.axhline(0.5, color="#bf6d29", linestyle="--", label="Predeclared missingness cutoff")
    ax.set_xticks(np.arange(len(missing)), missing.index, rotation=65, ha="right", fontsize=8)
    ax.set(
        ylabel="Missing fraction",
        title="Most incomplete sensors | development data only",
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()
    save(fig, "missingness.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, rows, scores, title in [
        (axes[0], development, oof["xgb_knn"], "Development OOF"),
        (axes[1], holdout, test_scores["xgb_knn"], "Holdout after refit"),
    ]:
        for label, text, color in [(0, "Pass", "#3574a8"), (1, "Fail", "#bf6d29")]:
            ax.hist(
                scores[dataset.y[rows] == label],
                bins=np.linspace(0, 1, 26),
                color=color,
                alpha=0.7,
                label=text,
            )
        ax.axvline(
            dev["xgb_knn"]["oof_threshold"],
            color="#263442",
            linestyle="--",
            label="Frozen threshold",
        )
        ax.set(xlabel="Weighted-model score", ylabel="Observations", title=title, yscale="symlog")
        ax.legend(fontsize=8)
    fig.suptitle(
        "XGBoost score distributions | OOF-to-refit threshold transfer can change behavior"
    )
    fig.tight_layout()
    save(fig, "score_distribution.png")


def write_results(path: Path, metrics: dict):
    selected = metrics["selected_candidate"]
    selected_result = metrics["holdout"][selected]
    chosen = selected_result["selected"]
    low, high = selected_result["ap_bootstrap_95"]
    lines = [
        "# Reproducible Experiment Results",
        "",
        "Generated by `main.py`; tables and figures use the same run. Do not edit numbers by hand.",
        "",
        f"- Completed (UTC): `{metrics['completed_at_utc']}`",
        f"- Source SHA-256: `{metrics['source_sha256']}`",
        f"- Frozen protocol SHA-256: `{metrics['protocol_sha256']}`",
        f"- Development: {metrics['development_rows']} observations / {metrics['development_failures']} failures.",
        f"- Holdout: {metrics['holdout_rows']} observations / {metrics['holdout_failures']} failures.",
        f"- CV-selected candidate: **{LABELS[selected]}** (`{selected}`).",
        "",
        "The selected candidate is determined by mean development-fold average precision (AP). All holdout comparisons below were produced after freezing that choice and every candidate's threshold. A different holdout ranking does not change the selection.",
        "",
        "## Development Comparison",
        "",
        f"Fold SD describes variation across {metrics['config']['folds']} overlapping training sets; it is not a confidence interval. OOF threshold F1 is optimized on these development labels and is optimistic.",
        "",
        "| Candidate | Mean fold AP | Fold SD | Pooled OOF AP | OOF threshold | OOF selected F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["development"]:
        lines.append(
            f"| {LABELS[row['candidate']]} | {row['mean_fold_ap']:.4f} | {row['std_fold_ap']:.4f} | {row['pooled_oof_ap']:.4f} | {row['oof_threshold']:.6f} | {row['oof_selected_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Holdout Comparison",
            "",
            "AP evaluates ranking. Precision, recall, F1, and alarms below use each candidate's development-selected threshold. AP is `average_precision_score`, not trapezoidal PR area. Model scores have not been calibrated as physical failure probabilities.",
            "",
            "| Candidate | AP | F1 @ 0.5 | Frozen precision | Frozen recall | Frozen F1 | Alarms | TP / FN / FP / TN |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in CANDIDATES:
        result, r = metrics["holdout"][name], metrics["holdout"][name]["selected"]
        lines.append(
            f"| {LABELS[name]} | {result['average_precision']:.4f} | {result['default']['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['alarm_count']} | {r['true_positives']} / {r['false_negatives']} / {r['false_positives']} / {r['true_negatives']} |"
        )
    recall_low, recall_high = chosen["recall_wilson_95"]
    lines.extend(
        [
            "",
            "## Selected-Candidate Uncertainty",
            "",
            f"- AP: **{selected_result['average_precision']:.4f}**, conditional bootstrap 95% interval **[{low:.4f}, {high:.4f}]**.",
            f"- Failure recall: **{chosen['recall']:.4f}**, Wilson 95% interval **[{recall_low:.4f}, {recall_high:.4f}]**.",
            f"- Caught **{chosen['true_positives']} of {chosen['failure_support']}** failures; **{chosen['false_positives']}** false alarms among {chosen['pass_support']} passing observations.",
            f"- One additional detected failure changes holdout recall by **{1 / metrics['holdout_failures']:.4f}**.",
            "",
            metrics["uncertainty_scope"],
            "",
            "## Chronological Sensitivity",
            "",
        ]
    )
    temporal = metrics["temporal_sensitivity"]
    if temporal:
        r = temporal["selected"]
        lines.extend(
            [
                "A fixed XGBoost candidate is trained on earlier development timestamps and evaluated on the latest development portion. Three expanding-window folds inside the earlier portion choose its threshold. This check does not use the main holdout and does not select the main model.",
                "",
                f"- Earlier: {temporal['early_rows']} observations / {temporal['early_failures']} failures; later: {temporal['later_rows']} observations / {temporal['later_failures']} failures.",
                f"- Last training timestamp: `{temporal['last_training_timestamp']}`; first later timestamp: `{temporal['first_validation_timestamp']}`.",
                f"- Later AP: **{temporal['average_precision']:.4f}**; prevalence: **{temporal['failure_prevalence']:.4f}**.",
                f"- Frozen threshold: **{r['threshold']:.6f}**; precision **{r['precision']:.4f}**, recall **{r['recall']:.4f}**, F1 **{r['f1']:.4f}**.",
            ]
        )
    else:
        lines.append("Not run (`--skip-temporal`).")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Development and holdout AP comparisons]({metrics['outputs']['figures']}/model_comparison.png)",
            "",
            f"![Held-out precision-recall curves]({metrics['outputs']['figures']}/pr_curve.png)",
            "",
            f"![Selected candidate confusion counts]({metrics['outputs']['figures']}/confusion_matrix.png)",
            "",
            "## Reproducibility And Limits",
            "",
            "`protocol.json` freezes choices before evaluation. `split_manifest.csv`, `fold_metrics.csv`, `oof_predictions.csv`, and `holdout_predictions.csv` permit independent metric reconstruction. `metrics.json` contains raw-file hashes, package versions, configuration, and intervals.",
            "",
            "SECOM was explored before this revision. These holdout rows are excluded from fitting and selection in this protocol, not historically unseen. A single small random split is not external validation. The temporal check is descriptive; no live factory cost, calibrated risk, causal sensor attribution, or production reliability is established. Do not compare the old test-tuned F1 directly with this protocol as if only the model changed.",
            "",
        ]
    )
    path.write_text("\n".join(lines))

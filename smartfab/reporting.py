"""Render a completed smartfab study without fitting or selecting any models."""

import json
from datetime import timedelta
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve

INK = "#263746"
BLUE = "#1764a0"
ORANGE = "#c87319"
GRAY = "#73808c"
STRATEGIES = ("selected", "v2_reference", "dummy_reference")
LABELS = ("Selected", "v2 reference", "Dummy reference")
COLORS = (BLUE, ORANGE, GRAY)


def _number(value, digits=3):
    return f"{value:.{digits}f}" if value is not None and np.isfinite(value) else "N/A"


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _percent(value):
    return f"{100 * value:.1f}%" if value is not None and np.isfinite(value) else "N/A"


def _counts(result):
    """Compute reporting rates from the confusion counts, including undefined rates."""
    tp, fp = int(result["true_positives"]), int(result["false_positives"])
    fn, tn = int(result["false_negatives"]), int(result["true_negatives"])
    n, positives, reviews = tp + fp + fn + tn, tp + fn, tp + fp
    precision = _ratio(tp, reviews)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
        "positives": positives,
        "reviews": reviews,
        "recall": _ratio(tp, positives),
        "precision": precision,
        "review_rate": _ratio(reviews, n),
        "lift": _ratio(precision, positives / n) if precision is not None and n else None,
    }


def generate_report(output: Path):
    """Read the nine study result files and write RESULTS.md and six PNG figures.

    All comparisons describe the stored run. This function never reads raw sensor
    data, changes the frozen selection, or supplies results before a run completes.
    """
    output = Path(output)
    metrics, protocol, selection = (
        json.loads((output / name).read_text(encoding="utf-8"))
        for name in ("metrics.json", "protocol.json", "selection.json")
    )
    search, blocks, predictions, budgets, costs, manifest = (
        pd.read_csv(output / name)
        for name in (
            "search_summary.csv",
            "search_blocks.csv",
            "assessment_predictions.csv",
            "assessment_budgets.csv",
            "cost_scenarios.csv",
            "split_manifest.csv",
        )
    )
    winner = selection["winner"]
    results = metrics["assessment"]
    counts = {name: _counts(results[name]["capacity"]) for name in STRATEGIES}
    chosen, reference = counts["selected"], counts["v2_reference"]
    n, positives, reviews = chosen["n"], chosen["positives"], chosen["reviews"]
    fraction = selection["primary_review_fraction"]
    budget_label = f"{fraction:.0%}"
    prevalence = positives / n
    expected = float(metrics["random_same_capacity_expected_caught"])
    random_low, random_high = metrics["random_same_capacity_95_range"]
    random_precision = _ratio(expected, reviews)
    search = search.sort_values("selection_rank", kind="stable")
    top = search.head(12)
    winner_blocks = blocks.loc[blocks["candidate"] == winner].sort_values("block")
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"], format="mixed", utc=True)
    development = manifest.loc[manifest["partition"] == "development"]
    assessment = manifest.loc[manifest["partition"] == "temporal_assessment"]
    cutoff = assessment["timestamp"].min()
    ap_interval = results["selected"]["ap_bootstrap_95"]
    for name in STRATEGIES:
        rows = predictions.loc[predictions["strategy"] == name]
        if rows.empty or not np.isfinite(rows["score"].to_numpy(dtype=float)).all():
            raise ValueError(f"Assessment scores must be nonempty and finite: {name}.")
    takeaway = (
        f"Selected flags {chosen['tp']}/{positives} failures at {budget_label} capacity; "
        f"v2 flags {reference['tp']}/{positives}"
    )
    cost_slice = costs.loc[
        (costs["strategy"] == "selected")
        & np.isclose(costs["passing_hold_cost_per_review_cost"], 0)
    ]
    ratios = [1, 2, 5, 10, 20, 50, 100]
    effectiveness = [0.25, 0.50, 0.75, 1.0]
    cost_grid = cost_slice.pivot(
        index="assumed_intervention_effectiveness",
        columns="escape_cost_per_review_cost",
        values="difference_vs_random",
    ).reindex(index=effectiveness, columns=ratios)
    differences = cost_grid.to_numpy(dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError(
            "The selected, zero-hold-cost sensitivity grid must contain 28 finite cells."
        )
    tied_cost = np.isclose(differences, 0)
    benefits = int(np.count_nonzero((differences < 0) & ~tied_cost))
    losses = int(np.count_nonzero((differences > 0) & ~tied_cost))
    equal_costs = int(tied_cost.sum())
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name, note):
        fig.text(0.5, 0.025, fill(note, 115), ha="center", va="bottom", fontsize=8, color=INK)
        try:
            fig.savefig(figures / f"{name}.png", dpi=200, facecolor="white")
        finally:
            plt.close(fig)

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAY,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.alpha": 0.15,
        "axes.axisbelow": True,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 6), sharey=True)
        positions = np.arange(4)
        labels = [*LABELS, "Random\n(expected)"]
        colors = [*COLORS, "#b3bdc5"]
        caught = [counts[name]["tp"] for name in STRATEGIES] + [expected]
        false = [counts[name]["fp"] for name in STRATEGIES] + [reviews - expected]
        precision = [counts[name]["precision"] or 0 for name in STRATEGIES]
        precision.append(random_precision or 0)
        for ax, values, title, xlabel in zip(
            axes,
            (caught, false, precision),
            ("Failures flagged", "Passing items flagged", "Review precision"),
            ("Flagged failures (count)", "False alerts (count)", "Failures / reviews (%)"),
            strict=True,
        ):
            bars = ax.barh(positions, values, color=colors, height=0.56)
            bars[-1].set_hatch("///")
            ax.set(title=title, xlabel=xlabel)
            ax.grid(axis="x")
            ax.set_xlim(0, max(max(values) * 1.4, 0.1 if title == "Review precision" else 1))
            for index, value in enumerate(values):
                if title == "Failures flagged":
                    text = f"{value:.1f}/{positives}" if index == 3 else f"{int(value)}/{positives}"
                elif title == "Review precision":
                    text = _percent(value) if reviews else "N/A"
                else:
                    text = f"{value:.1f}" if index == 3 else str(int(value))
                label_position = value
                offset, alignment, text_color = 5, "left", INK
                if title == "Failures flagged" and index == 3 and random_high is not None:
                    label_position = max(value, random_high)
                if title == "Review precision" and value >= 0.85:
                    offset, alignment = -5, "right"
                    if index == 0:
                        text_color = "white"
                ax.annotate(
                    text,
                    (label_position, index),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    va="center",
                    ha=alignment,
                    color=text_color,
                    fontsize=9,
                )
        if random_low is not None and random_high is not None:
            axes[0].hlines(3, random_low, random_high, color=INK, linewidth=2)
            axes[0].plot([random_low, random_high], [3, 3], "|", color=INK)
            axes[0].set_xlim(right=max(axes[0].get_xlim()[1], random_high * 1.2 + 0.5))
        axes[0].set_yticks(positions, labels)
        axes[0].invert_yaxis()
        axes[2].xaxis.set_major_formatter(PercentFormatter(1))
        axes[2].set_xlim(0, 1.05)
        fig.suptitle(f"Selected model: {winner}\n{takeaway}", fontsize=12)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.77, bottom=0.22, wspace=0.35)
        save(
            fig,
            "outcome_summary",
            f"Same completed batch: {reviews}/{n} reviews per strategy. Random whisker: stored 95% "
            "hypergeometric catch range, not a confidence interval for model improvement.",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, label, color, marker, line in zip(
            STRATEGIES, LABELS, COLORS, ("o", "s", "^"), ("-", "--", ":"), strict=True
        ):
            rows = budgets.loc[budgets["strategy"] == name].sort_values("review_fraction")
            ax.plot(
                100 * rows["review_fraction"],
                rows["true_positives"],
                color=color,
                marker=marker,
                linestyle=line,
                label=label,
                linewidth=2,
            )
            if name == "selected":
                for row in rows.itertuples():
                    total = int(row.true_positives + row.false_negatives)
                    ax.annotate(
                        f"{int(row.true_positives)}/{total}",
                        (100 * row.review_fraction, row.true_positives),
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )
        sweep = np.sort(budgets["review_fraction"].unique())
        ax.plot(
            100 * sweep,
            np.floor(n * sweep) * prevalence,
            color=INK,
            linestyle="-.",
            label="Random expectation",
            linewidth=1.5,
        )
        ax.axvline(100 * fraction, color=GRAY, linewidth=1, alpha=0.6)
        ax.set(
            xlabel="Review budget (% of completed batch; rounded down to whole items)",
            ylabel="Failures flagged (count)",
            xticks=100 * sweep,
            xlim=(0, max(sweep) * 100 + 5),
            ylim=(0, max(positives * 1.2, 1)),
            title=fill(takeaway, 65),
        )
        ax.grid()
        ax.legend(loc="upper left", fontsize=9)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.22)
        save(
            fig,
            "detection_vs_capacity",
            "Selected labels are flagged failures / all assessment failures. "
            "Budget changes are retrospective ranking scenarios; they do not measure factory capacity or throughput.",
        )

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 6), sharey=True)
        positions = np.arange(len(top))
        development_counts = [_counts(row) for row in top.to_dict("records")]
        row_colors = [BLUE if name == winner else "#a8bdce" for name in top["candidate"]]
        recalls = [
            row["recall"] if row["recall"] is not None else np.nan for row in development_counts
        ]
        axes[0].barh(positions, recalls, color=row_colors, height=0.6)
        axes[1].barh(positions, top["mean_block_ap"], color=row_colors, height=0.6)
        for index, row in enumerate(development_counts):
            axes[0].text(
                0.98, index, f"{row['tp']}/{row['positives']}", ha="right", va="center", fontsize=8
            )
        axes[0].set_yticks(positions, [fill(name, 30) for name in top["candidate"]], fontsize=8)
        axes[0].invert_yaxis()
        for ax, title, xlabel in zip(
            axes,
            ("Failure recall at fixed capacity", "Mean forward-block AP"),
            (
                f"Development failure recall at {budget_label} reviews (%)",
                "Average precision (unitless)",
            ),
            strict=True,
        ):
            ax.set(title=title, xlabel=xlabel, xlim=(0, 1.05))
            ax.grid(axis="x")
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        selected_development = _counts(search.loc[search["candidate"] == winner].iloc[0])
        fig.suptitle(
            f"Forward development: {winner}\n"
            f"{selected_development['tp']}/{selected_development['positives']} failures flagged; "
            f"top {len(top)} by frozen selection order",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.28, right=0.97, top=0.83, bottom=0.22, wspace=0.18)
        save(
            fig,
            "model_search",
            "Selection maximizes total forward-block true positives, then mean block AP, "
            "then catalog order. These reused development results do not measure assessment performance.",
        )

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        weekly_index = pd.date_range(
            manifest["timestamp"].min().normalize()
            - timedelta(days=manifest["timestamp"].min().weekday()),
            manifest["timestamp"].max().normalize(),
            freq="W-MON",
        )
        bottom = np.zeros(len(weekly_index))
        maximum_rate = 0.0
        for rows, label, color, marker in (
            (development, "Development", BLUE, "o"),
            (assessment, "Temporal assessment", ORANGE, "s"),
        ):
            weekly = (
                rows.set_index("timestamp")["label"]
                .resample("W-MON", label="left", closed="left")
                .agg(["sum", "count"])
            )
            weekly = weekly.reindex(weekly_index, fill_value=0)
            rate = weekly["sum"].div(weekly["count"].replace(0, np.nan))
            maximum_rate = max(maximum_rate, float(rate.max()))
            axes[0].plot(
                weekly_index,
                rate,
                color=color,
                marker=marker,
                markersize=4,
                label=f"{label}: {int(rows['label'].sum())}/{len(rows)} failures",
            )
            axes[1].bar(
                weekly_index,
                weekly["count"],
                bottom=bottom,
                width=6.5,
                align="edge",
                color=color,
                label=label,
                hatch="//" if label == "Temporal assessment" else None,
            )
            bottom += weekly["count"].to_numpy()
        for ax in axes:
            ax.axvline(cutoff, color=INK, linestyle="--", linewidth=1.3)
            ax.axvspan(
                cutoff, manifest["timestamp"].max() + timedelta(days=7), color=ORANGE, alpha=0.06
            )
            ax.grid(axis="y")
        axes[0].set(
            ylabel="Weekly failure fraction (%)", ylim=(0, min(max(maximum_rate * 1.3, 0.1), 1.05))
        )
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].set(
            ylabel="Weekly observations (count)",
            xlabel="Observation week starting Monday (dataset timestamp)",
        )
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.suptitle(
            "Recorded failure prevalence: development "
            f"{_percent(development['label'].mean())}, assessment {_percent(prevalence)}\n"
            f"Assessment starts {cutoff.date().isoformat()} (dashed boundary)",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.22, hspace=0.13)
        save(
            fig,
            "temporal_evaluation",
            "Weeks crossing the boundary are split by partition; volume bars are stacked. "
            "Only recorded timestamps and labels are available here; no sensor missingness or causal drift is inferred.",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, label, color, line, marker in zip(
            STRATEGIES, LABELS, COLORS, ("-", "--", ":"), ("o", "s", "^"), strict=True
        ):
            rows = predictions.loc[predictions["strategy"] == name]
            if rows["label"].nunique() == 2:
                precision_values, recall_values, _ = precision_recall_curve(
                    rows["label"], rows["score"]
                )
                ax.step(
                    recall_values,
                    precision_values,
                    where="post",
                    color=color,
                    linestyle=line,
                    linewidth=2,
                    label=f"{label}: AP={_number(results[name]['average_precision'])}",
                )
            if counts[name]["recall"] is not None and counts[name]["precision"] is not None:
                ax.scatter(
                    counts[name]["recall"],
                    counts[name]["precision"],
                    color=color,
                    marker=marker,
                    s=55,
                    edgecolors="white",
                    zorder=4,
                )
        ax.axhline(
            prevalence,
            color=INK,
            linestyle="-.",
            linewidth=1,
            label=f"Failure prevalence: {prevalence:.1%}",
        )
        if positives in (0, n):
            ax.text(
                0.5,
                0.5,
                "PR comparison unavailable: assessment has one label class",
                transform=ax.transAxes,
                ha="center",
                wrap=True,
            )
        ax.set(
            xlabel="Failure recall (%)",
            ylabel="Failure precision (%)",
            xlim=(0, 1.03),
            ylim=(0, 1.05),
            title="Selected assessment AP "
            f"{_number(results['selected']['average_precision'])}; prevalence {prevalence:.1%}",
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid()
        ax.legend(loc="upper right", fontsize=9)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.22)
        save(
            fig,
            "pr_curve",
            f"Markers use the frozen {budget_label} batch policy, including tie-breaking; "
            "tied-score markers need not lie on a threshold curve. AP is average precision, not trapezoidal PR area.",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        limit = max(float(np.abs(differences).max()), 1)
        cmap = LinearSegmentedColormap.from_list("review_cost", [BLUE, "#ffffff", ORANGE])
        image = ax.imshow(
            differences,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            aspect="auto",
        )
        for (row, column), value in np.ndenumerate(differences):
            ax.text(
                column,
                row,
                f"{value:,.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > limit * 0.65 else INK,
            )
        ax.set(
            xticks=np.arange(len(ratios)),
            xticklabels=ratios,
            yticks=np.arange(len(effectiveness)),
            yticklabels=[f"{value:.0%}" for value in effectiveness],
            xlabel="Assumed escaped-failure cost / one-review cost (ratio)",
            ylabel="Assumed intervention effectiveness (%)",
            title=f"Selected vs random simulated cost: {benefits} lower, {equal_costs} equal, {losses} higher\n"
            "Passing-item hold cost fixed at 0 review-cost units",
        )
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.045,
            pad=0.04,
            label="Selected minus expected random cost (review-cost units)",
        )
        fig.subplots_adjust(left=0.13, right=0.90, top=0.81, bottom=0.24)
        save(
            fig,
            "cost_sensitivity",
            "Blue/negative = simulated benefit; orange/positive = simulated loss. "
            "One review costs 1 normalized unit; effectiveness and cost ratios are assumptions, not measured factory costs.",
        )

    def interval_text(interval):
        if interval is None:
            return "N/A (undefined)"
        return f"[{_number(interval[0])}, {_number(interval[1])}]"

    delta_v2 = chosen["tp"] - reference["tp"]
    delta_random = chosen["tp"] - expected
    threshold = selection["threshold_policy"]
    threshold_development = _counts(threshold["metrics"])
    threshold_assessment = _counts(results["selected"]["fixed_threshold"])
    threshold_value = (
        _number(threshold["threshold"], 6)
        if threshold["threshold"] is not None
        else "None (no alerts)"
    )
    catalog_count = len(protocol["catalog"])
    blended_count = int((search["candidate"] == "rank_blend_top2").any())
    lines = [
        "# SECOM Smart Manufacturing: Frozen Temporal Assessment",
        "",
        "Generated from this completed run's result files; all numbers and figures are derived from those files.",
        "",
        f"**{takeaway}.** The selected policy flags {_percent(chosen['recall'])} of recorded failures "
        f"using {reviews}/{n} reviews ({_percent(chosen['review_rate'])} of the batch), with "
        f"{chosen['fp']} passing items flagged and precision {_percent(chosen['precision'])}.",
        "",
        f"The count difference is **{delta_v2:+d}** flagged failures versus v2 and "
        f"**{delta_random:+.2f}** versus the random expectation of {expected:.2f}; "
        "these are descriptive comparisons on this assessment batch, not evidence of a guaranteed improvement.",
        "",
        "## Scope And Dates",
        "",
        f"- Completed (UTC): `{metrics['completed_at_utc']}`.",
        f"- Protocol frozen (UTC): `{protocol['frozen_before_search_utc']}`.",
        f"- Selection frozen (UTC): `{selection['frozen_before_assessment_utc']}`.",
        f"- Development: {len(development)} observations, {int(development['label'].sum())} failures "
        f"({_percent(development['label'].mean())}); `{development['timestamp'].min().isoformat()}` "
        f"through `{development['timestamp'].max().isoformat()}`.",
        f"- Latest temporal assessment: {n} observations, {positives} failures ({prevalence:.1%}); "
        f"`{assessment['timestamp'].min().isoformat()}` through `{assessment['timestamp'].max().isoformat()}`.",
        "",
        "The latest chronological block is protected from fitting, candidate selection, blend-member selection, "
        "and threshold selection within this run. SECOM and its labels were historically explored in v2; "
        "this is a protected within-run temporal boundary, not historically unseen or external validation.",
        "",
        "## Frozen Development Selection",
        "",
        f"Selected candidate: **`{winner}`**. Frozen members: "
        + ", ".join(f"`{member}`" for member in selection["members"])
        + ".",
        "",
        f"The protocol catalog contains **{catalog_count} base candidates**, plus "
        f"**{blended_count} development-derived rank blend**, giving "
        f"**{catalog_count + blended_count} search entries**; {len(search)} entries completed and appear "
        "in the search summary. The standard catalog is 50 base candidates plus one rank blend (51 total). "
        "The blend averages within-batch percentile ranks of the two highest-ranked supervised base "
        "candidates, with no weight tuning.",
        "",
        f"The primary objective is the **highest total true positives at exact per-block {budget_label} review "
        f"capacity** across {winner_blocks['block'].nunique()} expanding chronological development blocks. "
        "Ties use higher mean block AP, then fixed catalog order; the blend comes last on exact ties. "
        "The assessment results do not select or reselect a model, threshold, blend, or operating budget.",
        "",
        "| Rank | Candidate | Forward failures flagged / total | Recall | Mean block AP |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in top.to_dict("records"):
        values = _counts(row)
        lines.append(
            f"| {int(row['selection_rank'])} | `{row['candidate']}` | {values['tp']}/{values['positives']} "
            f"| {_percent(values['recall'])} | {_number(row['mean_block_ap'])} |"
        )
    lines.extend(
        [
            "",
            "| Selected candidate forward block | Training rows | Validation rows | Failures flagged / total | Block AP |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in winner_blocks.to_dict("records"):
        values = _counts(row)
        lines.append(
            f"| {int(row['block']) + 1} | {int(row['train_rows'])} | {int(row['validation_rows'])} "
            f"| {values['tp']}/{values['positives']} | {_number(row['average_precision'])} |"
        )
    lines.extend(
        [
            "",
            "![Top development candidates by the frozen selection order](figures/model_search.png)",
            "",
            f"The development search selected `{winner}` with {selected_development['tp']}/"
            f"{selected_development['positives']} failures flagged, and the figure compares the top "
            f"{len(top)} candidates' recall and mean block AP. These reused development blocks support "
            "selection and are not an independent performance estimate or an assessment-based leaderboard.",
            "",
            "## Primary Completed-Batch Policy",
            "",
            f"Every strategy ranks the same completed assessment batch and reviews exactly "
            f"`floor({fraction:g} * {n}) = {reviews}` items, with highest scores first. Tied scores use "
            "the stored seed and original row IDs through SHA-256 ordering, never labels; the prior dummy "
            "therefore represents one deterministic tie-broken allocation, while theoretical random review "
            "is an expectation over allocations.",
            "",
            "| Strategy | Reviews | TP | FN | FP | TN | Recall | Precision | Lift vs prevalence | AP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, label in zip(STRATEGIES, LABELS, strict=True):
        values = counts[name]
        lines.append(
            f"| {label} | {values['reviews']} | {values['tp']} | {values['fn']} | {values['fp']} "
            f"| {values['tn']} | {_percent(values['recall'])} | {_percent(values['precision'])} "
            f"| {_number(values['lift'])} | {_number(results[name]['average_precision'])} |"
        )
    lines.extend(
        [
            f"| Random (theoretical expectation) | {reviews} | {expected:.2f} | {positives - expected:.2f} "
            f"| {reviews - expected:.2f} | {n - positives - reviews + expected:.2f} "
            f"| {_percent(_ratio(expected, positives))} | {_percent(random_precision)} "
            f"| {_number(_ratio(random_precision, prevalence) if random_precision is not None else None)} | N/A |",
            "",
            "Precision is TP / (TP + FP), recall is TP / (TP + FN), and lift is precision / assessment "
            "failure prevalence, all calculated from the actual confusion counts. Undefined rates are N/A; "
            "a constant-score dummy's AP is the assessment prevalence when both label classes are present.",
            "",
            f"The stored theoretical random-review expectation is **{expected:.2f} failures**, with stored "
            f"95% hypergeometric catch range **[{_number(random_low, 0)}, {_number(random_high, 0)}]** "
            f"for sampling {reviews} of {n} items without replacement with {positives} failures. "
            "This range describes the number caught by a random allocation, not a confidence interval "
            "for a model difference or a significance test.",
            "",
            "![Equal-capacity failure detection and false-alert workload](figures/outcome_summary.png)",
            "",
            f"At the same {budget_label} budget, the selected model flags {chosen['tp']}/{positives} failures "
            f"and {chosen['fp']} passing items, compared with {reference['tp']}/{positives} failures for v2 "
            f"and {expected:.2f} failures expected under random review. These are batch review outcomes "
            "on recorded labels, and additional flagged failures do not establish higher production yield.",
            "",
            "## Detection Across Review Budgets",
            "",
            "![Failure counts across the five recorded review budgets](figures/detection_vs_capacity.png)",
            "",
            "The lines show failures flagged by each frozen score ranking at 5%, 10%, 20%, 30%, and 50% "
            "review budgets, with selected-model labels giving caught count / total failures. This is a "
            "completed-batch ranking sensitivity analysis, not a measured increase in production capacity, "
            "a new model selection, or a live arrival-by-arrival review guarantee.",
            "",
            "## Temporal Composition",
            "",
            "![Weekly recorded failure fractions and observation volumes](figures/temporal_evaluation.png)",
            "",
            f"Weekly failure fractions and observation counts show development prevalence "
            f"{_percent(development['label'].mean())} and assessment prevalence {prevalence:.1%}, with "
            "the dashed line marking the first assessment timestamp and boundary weeks split by partition. "
            "The available manifest contains timestamps and labels only, so this plot cannot establish "
            "sensor missingness, feature drift, causality, or feature availability at decision time.",
            "",
            "## Ranking And Conditional Uncertainty",
            "",
            f"Selected assessment AP: **{_number(results['selected']['average_precision'])}**, with the "
            f"stored conditional 95% bootstrap interval **{interval_text(ap_interval)}**. This interval "
            "uses a fixed-score, class-stratified bootstrap conditional on the fitted model and observed "
            "assessment prevalence; it does not include "
            "candidate-search uncertainty, training variation, temporal dependence, or uncertainty in "
            "future production conditions.",
            "",
            f"Selected recall: **{_percent(chosen['recall'])}**, with stored Wilson 95% interval "
            f"**{interval_text(results['selected']['capacity']['recall_wilson_95'])}**. "
            + (
                f"One additional flagged failure changes recall by {100 / positives:.2f} percentage points."
                if positives
                else "Recall is undefined because this assessment has no failures."
            ),
            "",
            "![Assessment precision-recall curves and fixed-capacity review points](figures/pr_curve.png)",
            "",
            f"The selected, v2, and dummy curves compare score rankings against assessment failure "
            f"prevalence {prevalence:.1%}, and markers show the frozen {budget_label} review decisions. "
            "AP is average precision rather than trapezoidal PR area, scores are not calibrated "
            "probabilities, and tie-broken capacity points need not lie on a score-threshold curve.",
            "",
            "## Strictly Secondary F2 Threshold",
            "",
            f"`selection.threshold_policy` records a five-fold stratified development OOF maximum-F2 "
            f"threshold of **{threshold_value}**, with at most **{threshold['max_reviews']}** development "
            f"OOF alerts ({budget_label} of development rows, rounded down). The development policy "
            f"actually issues {threshold_development['reviews']} alerts "
            f"({_percent(threshold_development['review_rate'])}), flags {threshold_development['tp']}/"
            f"{threshold_development['positives']} failures, and has F2 "
            f"{_number(threshold['metrics']['f2'])} on the labels used to choose it.",
            "",
            f"Applied unchanged to assessment scores, this secondary threshold issues "
            f"**{threshold_assessment['reviews']}/{n} alerts "
            f"({_percent(threshold_assessment['review_rate'])})**, flags "
            f"{threshold_assessment['tp']}/{positives} failures, produces {threshold_assessment['fp']} "
            f"false alerts, and has precision {_percent(threshold_assessment['precision'])}, recall "
            f"{_percent(threshold_assessment['recall'])}, and F2 "
            f"{_number(results['selected']['fixed_threshold']['f2'])}. "
            + (
                f"Its assessment alert rate exceeds the primary {budget_label} budget. "
                if threshold_assessment["reviews"] > reviews
                else ""
            )
            + f"The OOF alert constraint does not guarantee at most {budget_label} alerts after refitting "
            "or in live use; this threshold is strictly secondary and never replaces the primary batch-capacity result.",
            "",
            "## Hypothetical Cost Sensitivity",
            "",
            "The stored simulation uses `C = reviews + (P - a * TP) * escapeCost + FP * holdCost`, "
            "where P is the assessment failure count, a is assumed intervention effectiveness, and "
            "one review costs one normalized unit. An intervention averts only the assumed fraction "
            "of flagged failures; false alerts incur the assumed additional passing-item hold cost. "
            "These assumptions are not observed costs, estimated intervention effects, or measured savings.",
            "",
            "![Selected-minus-random cost under explicit hypothetical assumptions](figures/cost_sensitivity.png)",
            "",
            f"With passing-item hold cost fixed at zero, selected-minus-expected-random cost is lower in "
            f"{benefits}, equal in {equal_costs}, and higher in {losses} of the 28 combinations of escape-cost "
            "ratios [1, 2, 5, 10, 20, 50, 100] and intervention effectiveness [25%, 50%, 75%, 100%]. "
            "Negative blue cells indicate simulated benefit and positive orange cells simulated loss "
            "under those assumptions; neither color establishes actual financial, yield, or throughput gains.",
            "",
            "The illustrative slice below uses escape cost 20 and effectiveness 50%, with passing-item "
            "hold cost 0 or 2, all in normalized review-cost units. The complete stored grid includes "
            "no additional review, review everything, theoretical random review, the selected model, "
            "v2, and the dummy; this slice is not an optimized or empirically estimated cost setting.",
            "",
            "| Stored strategy | Passing-item hold cost | Simulated total cost | Difference vs expected random |",
            "|---|---:|---:|---:|",
        ]
    )
    illustrative = costs.loc[
        np.isclose(costs["escape_cost_per_review_cost"], 20)
        & np.isclose(costs["assumed_intervention_effectiveness"], 0.5)
    ].sort_values(["passing_hold_cost_per_review_cost", "strategy"])
    for row in illustrative.itertuples():
        lines.append(
            f"| `{row.strategy}` | {row.passing_hold_cost_per_review_cost:g} "
            f"| {row.simulated_total_cost_units:.2f} | {row.difference_vs_random:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence And Interpretation Boundaries",
            "",
            "The positive evidence is directly observable: a frozen development-selected ranking "
            f"allocates exactly {reviews} reviews and identifies {chosen['tp']} recorded failures "
            f"with {_number(chosen['lift'])} lift over assessment prevalence. Comparisons with the "
            "reference and random expectation above retain their observed direction, including ties or losses.",
            "",
            "Budget refers to ranking an already completed batch, not expanding factory capacity. "
            "Availability of sensor features and labels at a real production decision time is unverified. "
            "Increased flagged failures must not be interpreted as increased production yield; no causal "
            "intervention, yield improvement, throughput increase, deployment reliability, or state-of-the-art "
            "performance is established by this retrospective study.",
            "",
            "Small failure counts and repeated comparisons on shared development blocks leave selection "
            "uncertainty that the conditional AP interval does not resolve. Prior public-dataset exploration "
            "limits historical independence even though the current assessment boundary is frozen within this run.",
            "",
            f"- Protocol SHA-256: `{metrics['protocol_sha256']}`.",
            f"- Selection SHA-256: `{metrics['selection_sha256']}`.",
            "- Frozen configuration and catalog: [protocol.json](protocol.json).",
            "- Frozen winner, members, and secondary threshold: [selection.json](selection.json).",
            "- Counts, AP, intervals, dates, and environment: [metrics.json](metrics.json).",
            "- Development search: [search_summary.csv](search_summary.csv) and [search_blocks.csv](search_blocks.csv).",
            "- Assessment scores and primary decisions: [assessment_predictions.csv](assessment_predictions.csv).",
            "- Recorded review-budget sweep: [assessment_budgets.csv](assessment_budgets.csv).",
            "- Complete hypothetical strategy grid: [cost_scenarios.csv](cost_scenarios.csv).",
            "- Recorded timestamps, partitions, and labels: [split_manifest.csv](split_manifest.csv).",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

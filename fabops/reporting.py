"""Figures and RESULTS.md for a completed fab-monitoring run (no fitting here)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, BLUE, ORANGE, GRAY, GREEN = "#263746", "#1764a0", "#c87319", "#9aa5ae", "#3d8b5f"
NAMES = {
    "v3_selected_knn": ("ML: frozen v3 KNN", BLUE, "o", "-"),
    "v2_reference": ("ML: v2 RF reference", GRAY, "s", ":"),
    "mspc": ("MSPC (selected)", ORANGE, "^", "-"),
    "hybrid_knn_mspc": ("Hybrid rank blend", GREEN, "D", "--"),
}
POLICY_STYLE = {
    "frozen": (GRAY, "-"),
    "periodic": (BLUE, "-"),
    "drift_triggered": (ORANGE, "--"),
    "always": (GREEN, ":"),
}


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": "#c9d0d6",
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": "#e8ecef",
            "grid.linewidth": 0.8,
        }
    )


def _save(fig, path, note):
    fig.tight_layout()
    fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=8.5, color=INK, wrap=True)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def generate_report(output: Path) -> None:
    _style()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metrics = json.loads((output / "metrics.json").read_text())
    budgets = pd.read_csv(output / "assessment_budgets.csv")
    timeline = pd.read_csv(output / "mspc_timeline.csv", parse_dates=["timestamp"])
    pareto = pd.read_csv(output / "alarm_pareto.csv")
    example = pd.read_csv(output / "example_contribution.csv")
    replay = pd.read_csv(output / "retraining_batches.csv")
    summary = pd.read_csv(output / "retraining_summary.csv")
    drift = pd.read_csv(output / "drift_by_batch.csv")
    stability = pd.read_csv(output / "sensor_stability.csv")
    n, failures = metrics["assessment_rows"], metrics["assessment_failures"]

    # 1. Industry baseline vs ML at every budget.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fractions = np.array(sorted(budgets.review_fraction.unique()))
    expected = np.floor(n * fractions) * failures / n
    ax.fill_between(100 * fractions, 0, expected, color=GRAY, alpha=0.15, label="Random review (expected)")
    for name, (label, color, marker, line) in NAMES.items():
        part = budgets.loc[budgets.strategy.eq(name)].sort_values("review_fraction")
        ax.plot(100 * part.review_fraction, part.true_positives, marker=marker, linestyle=line, color=color, label=label)
    ax.axvline(20, color=INK, linewidth=1, alpha=0.4)
    ax.set(xlabel="Inspection budget (% of the 314-unit assessment batch)", ylabel=f"Failures caught (of {failures})",
           title="Industry-standard MSPC vs supervised ranking on later data", ylim=(0, failures))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, figures / "mspc_vs_ml.png",
          "Frozen before assessment. MSPC is pass-only (no failure labels); ML rankings use labels. 17 failures: one unit ≈ 6 recall points.")

    # 2. Control charts across the full timeline.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    limits = metrics["mspc_model"]
    cut = timeline.loc[timeline.partition.eq("temporal_assessment"), "timestamp"].min()
    for ax, column, limit, title in (
        (axes[0], "t2", limits["t2_limit"], "Hotelling T²"),
        (axes[1], "spe", limits["spe_limit"], "SPE (Q statistic)"),
    ):
        passes, fails = timeline.loc[timeline.label.eq(0)], timeline.loc[timeline.label.eq(1)]
        ax.scatter(passes.timestamp, passes[column], s=6, color=GRAY, alpha=0.6, label="pass")
        ax.scatter(fails.timestamp, fails[column], s=16, color=ORANGE, label="recorded failure")
        ax.axhline(limit, color=INK, linestyle="--", linewidth=1, label="99% limit")
        ax.axvline(cut, color=BLUE, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_ylabel(f"{title} (log)")
    axes[0].text(cut, axes[0].get_ylim()[1], "  assessment →", color=BLUE, va="top", fontsize=9)
    axes[0].set_title("MSPC control charts: model fitted on development passes only")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    classic = metrics["classic_control_limits"]["theoretical_limits"]
    _save(fig, figures / "control_charts.png",
          f"With textbook 99% limits, {classic['assessment_pass_alarm_rate']:.0%} of later passing units alarm "
          f"(expected ≈1%): non-Gaussian sensors and drift make fixed limits unusable without recalibration.")

    # 3. Contribution plot and alarm Pareto.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    top = example.reindex(example.spe_contribution.abs().sort_values(ascending=False).index).head(12)[::-1]
    axes[0].barh(top.sensor, top.spe_contribution, color=ORANGE)
    axes[0].set(title=f"SPE contribution plot: unit {metrics['example_contribution_row_id']}"
                f" ({'failure' if metrics['example_contribution_label'] else 'pass'})", xlabel="Squared scaled residual")
    head = pareto.head(12)[::-1]
    axes[1].barh(head.sensor, 100 * head.spe_contribution_share, color=BLUE)
    axes[1].set(title="Alarm Pareto: share of SPE across queued units", xlabel="% of total SPE in the 20% queue")
    _save(fig, figures / "contributions.png",
          "Contributions decompose each statistic exactly by sensor; they say where to look first, not the root cause.")

    # 4. Drift over time with the sampling-noise floor.
    fig, ax = plt.subplots(figsize=(9, 4.4))
    share = drift.loc[drift.informative].groupby("batch").psi_alert.mean()
    rates = replay.loc[(replay.model.eq("v3_selected_knn")) & replay.policy.eq("frozen")].set_index("batch")
    ax.bar(share.index, 100 * rates.failures / rates.rows, color=GRAY, alpha=0.35, label="failure rate in batch (%)")
    ax.plot(share.index, 100 * share, color=ORANGE, marker="o", label="sensors with PSI > 0.25 vs initial window (%)")
    floor = metrics["drift_noise_floor"]
    ax.axhline(100 * floor["random_split_share_max"], color=INK, linestyle=":", label="sampling-noise ceiling (random split)")
    ax.axhline(10, color=BLUE, linestyle="--", linewidth=1, label="predeclared retrain trigger (10%)")
    ax.set(xlabel="Prequential batch (100 units each, chronological)", ylabel="%", ylim=(0, 100),
           title="Sensor drift is continuous, not an occasional event")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    _save(fig, figures / "drift_over_time.png",
          f"Random 100-row splits of the same window flag ≤{floor['random_split_share_max']:.0%} of sensors; "
          f"time-ordered batches flag {share.min():.0%}–{share.max():.0%}. A share-based trigger therefore fires every batch.")

    # 5. Retraining policies.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, model, title in ((axes[0], "v3_selected_knn", "Supervised KNN"), (axes[1], "mspc", "MSPC")):
        for policy, (color, line) in POLICY_STYLE.items():
            part = replay.loc[replay.model.eq(model) & replay.policy.eq(policy)].sort_values("batch")
            row = summary.loc[summary.model.eq(model) & summary.policy.eq(policy)].iloc[0]
            ax.plot(part.batch, part.caught.cumsum(), color=color, linestyle=line, linewidth=2,
                    label=f"{policy.replace('_', ' ')}: {row.caught} caught, {row.retrains} retrains")
        total = replay.loc[replay.model.eq(model) & replay.policy.eq("frozen")].sort_values("batch")
        ax.plot(total.batch, 0.2 * total.failures.cumsum(), color=GRAY, linewidth=1, alpha=0.7, label="random (expected)")
        ax.set(title=title, xlabel="Prequential batch")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Cumulative failures caught at 20% review")
    _save(fig, figures / "retraining_policies.png",
          f"Test-then-train replay of {int(replay.rows.sum() / replay.groupby(['model', 'policy']).ngroups)} units, labels "
          f"delayed one batch. Smallest paired sign-test p vs frozen: {summary.loc[summary.policy.ne('frozen'), 'sign_test_p'].min():.2f}.")

    # 6. Sensor ranking stability.
    k = metrics["stability"]["k"]
    head = stability.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    methods = (("anova", BLUE, "o"), ("random_forest", GREEN, "s"), ("mspc_spe", ORANGE, "^"))
    for method, color, marker in methods:
        ax.scatter(head[f"{method}_top{k}_freq"], head.sensor, color=color, marker=marker, label=method.replace("_", " "), s=36)
    ax.scatter(head.consensus_freq, head.sensor, color=INK, marker="|", s=200, label="consensus")
    ax.set(xlim=(-0.03, 1.03), xlabel=f"Share of {metrics['stability']['n_bootstrap']} bootstraps in the top {k}",
           title="Which sensors stay on top? Bootstrap rank stability")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    jac = metrics["stability"]["bootstrap_jaccard"]
    _save(fig, figures / "sensor_stability.png",
          "Mean pairwise top-20 Jaccard across bootstraps: " + ", ".join(f"{m} {v:.2f}" for m, v in jac.items()) + ".")

    write_results(output, metrics, budgets, summary, stability, pareto)


def _row(name, result):
    capacity = result["capacity"]
    ap = result.get("average_precision")
    return (f"| {NAMES.get(name, (name,))[0]} | {capacity['true_positives']} / {capacity['failure_support']} "
            f"| {capacity['false_positives']} | {100 * capacity['precision']:.1f}% | {ap:.3f} |")


def write_results(output, metrics, budgets, summary, stability, pareto):
    assessment = metrics["assessment"]
    classic = metrics["classic_control_limits"]
    overlap = metrics["catch_overlap_at_20pct"]
    floor = metrics["drift_noise_floor"]
    stab = metrics["stability"]
    model = metrics["mspc_model"]
    lines = [
        "# Fab-monitoring results",
        "",
        "Generated by `python -m fabops.study`. Every number below comes from the CSV/JSON files in this folder.",
        "",
        "## 1. MSPC baseline vs supervised ranking (same frozen split, same 20% budget)",
        "",
        f"Selected MSPC on development forward blocks: **`{metrics['mspc_selected']}`** "
        f"({model['n_components']} components, {model['explained_variance']:.0%} variance, "
        f"{model['n_sensors_monitored']} sensors, fitted on {model['n_train_passes']} development passes).",
        "",
        "| Strategy (assessment, 62 reviews) | Failures caught | Passing units flagged | Review precision | AP |",
        "|---|---:|---:|---:|---:|",
        *(_row(name, assessment[name]) for name in ("v3_selected_knn", "v2_reference", "mspc", "hybrid_knn_mspc")),
        f"| Random review (expected) | {metrics['random_expected_caught']:.2f} / {metrics['assessment_failures']} "
        f"| – | {100 * metrics['assessment_failures'] / metrics['assessment_rows']:.1f}% | – |",
        "",
        f"Catch overlap at 20%: KNN only {overlap['knn_only']}, MSPC only {overlap['mspc_only']}, "
        f"both {overlap['both']}, neither {overlap['neither']}.",
        "",
        "Classic control-chart decision (alarm when T² or SPE exceeds its 99% limit):",
        "",
        "| Limits | Development pass alarm rate (in-sample) | Assessment pass alarm rate | Alarms | Failures caught |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {label.replace('_', ' ')} | {v['development_pass_alarm_rate_in_sample']:.1%} | "
            f"{v['assessment_pass_alarm_rate']:.1%} | {v['alarm_count']} | {v['true_positives']} / {v['failure_support']} |"
            for label, v in classic.items()
        ),
        "",
        "![MSPC vs ML](figures/mspc_vs_ml.png)",
        "",
        "![Control charts](figures/control_charts.png)",
        "",
        "![Contributions](figures/contributions.png)",
        "",
        "Top alarm-Pareto sensors (share of SPE in the 20% MSPC queue): "
        + ", ".join(f"`{r.sensor}` {100 * r.spe_contribution_share:.1f}%" for r in pareto.head(5).itertuples())
        + ".",
        "",
        "## 2. Drift monitoring",
        "",
        f"- Assessment block vs development: **{metrics['drift_assessment']['psi_alert_share']:.0%}** of "
        f"{metrics['drift_assessment']['informative_sensors']} informative sensors have PSI > 0.25; "
        f"{metrics['drift_assessment']['ks_alert_share']:.0%} fail KS at BH q < 0.01.",
        f"- Noise floor: random 100-row splits of one window flag {floor['random_split_share_mean']:.1%} on average "
        f"(max {floor['random_split_share_max']:.1%}); the last 100 rows *in time* of the same window flag "
        f"{floor['last_batch_vs_earlier_share']:.0%}.",
        "- Per prequential batch vs the initial window: "
        + ", ".join(f"{100 * s:.0f}%" for s in metrics["drift_by_batch_share"]) + ".",
        "",
        "![Drift](figures/drift_over_time.png)",
        "",
        "## 3. Retraining policies (prequential, labels delayed one batch)",
        "",
        "| Model | Policy | Caught / failures | Retrains | vs frozen | Batches better / worse | Sign-test p |",
        "|---|---|---:|---:|---:|---:|---:|",
        *(
            f"| {r.model} | {r.policy} | {r.caught} / {r.failures} | {r.retrains} | {r.caught_minus_baseline:+d} "
            f"| {r.batches_better} / {r.batches_worse} | {r.sign_test_p:.2f} |"
            for r in summary.itertuples()
        ),
        "",
        "![Retraining](figures/retraining_policies.png)",
        "",
        "## 4. Sensor ranking stability",
        "",
        f"Top-{stab['k']} Jaccard, mean over bootstrap pairs: "
        + ", ".join(f"{m} {v:.2f}" for m, v in stab["bootstrap_jaccard"].items())
        + "; across time slices: "
        + ", ".join(f"{m} {v:.2f}" for m, v in stab["time_slice_jaccard"].items())
        + "; between methods (full development data): "
        + ", ".join(f"{m.replace('__', ' vs ')} {v:.2f}" for m, v in stab["cross_method_jaccard"].items())
        + ".",
        "",
        f"Watchlist tiers: {metrics['watchlist_tiers']}.",
        "",
        "| Sensor | Tier | Consensus bootstrap freq | ANOVA | RF | MSPC-SPE | Time-slice freq | Later std. mean diff | Later PSI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| `{r.sensor}` | {r.tier} | {r.consensus_freq:.2f} | {getattr(r, f'anova_top{stab['k']}_freq'):.2f} "
            f"| {getattr(r, f'random_forest_top{stab['k']}_freq'):.2f} | {getattr(r, f'mspc_spe_top{stab['k']}_freq'):.2f} "
            f"| {r.slice_freq:.2f} | {r.assessment_std_mean_difference:+.2f} | {r.assessment_psi:.2f} |"
            for r in stability.head(15).itertuples()
        ),
        "",
        "![Stability](figures/sensor_stability.png)",
        "",
        "## Limits",
        "",
        "- 17 assessment failures and 46 replay failures: differences of a few units are within noise.",
        "- SECOM sensors are anonymised; contributions and rankings are inspection priorities, not root causes.",
        "- Replay retraining uses the full timeline, including the v3 assessment block, as a separate operational simulation; it never feeds back into model selection.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))

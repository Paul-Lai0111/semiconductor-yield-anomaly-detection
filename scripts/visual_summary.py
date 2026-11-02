"""Render a closing evidence figure from the frozen study and development supplement."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render(root: Path):
    evidence = root / "results"
    supplement = root / "supplementary/development"
    metrics = json.loads((evidence / "metrics.json").read_text())
    if sha256((evidence / "protocol.json").read_bytes()).hexdigest() != metrics["protocol_sha256"]:
        raise ValueError("Frozen protocol does not match the reported experiment.")
    provenance = json.loads((supplement / "provenance.json").read_text())
    if provenance["original_protocol_sha256"] != metrics["protocol_sha256"]:
        raise ValueError("Supplementary results refer to a different original experiment.")
    selected = metrics["holdout"][metrics["selected_candidate"]]
    default, frozen = selected["default"], selected["selected"]
    repeats = pd.read_csv(supplement / "repeat_metrics.csv")
    paired = repeats.pivot(index="repeat", columns="candidate", values="mean_fold_ap")
    differences = paired.rf_knn_smote - paired.rf_knn
    quality = pd.read_csv(evidence / "data_quality.csv")
    palette = {
        "ink": "#183448",
        "muted": "#556774",
        "blue": "#26739d",
        "orange": "#bc642f",
        "light": "#dce4ea",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": palette["ink"],
            "axes.labelcolor": palette["muted"],
            "xtick.color": palette["muted"],
            "ytick.color": palette["ink"],
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.6))
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.37, top=0.73, wspace=0.73)
    fig.text(
        0.055,
        0.94,
        "A useful ranking still needs an explicit review policy.",
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.055,
        0.895,
        f"SECOM | {metrics['dataset']['rows']:,} observations | {metrics['dataset']['sensor_columns']} sensor columns | supervised failure classification",
        fontsize=12,
        color=palette["muted"],
    )
    fig.text(
        0.055,
        0.84,
        f"FROZEN HOLDOUT: {metrics['holdout_rows']} observations / {metrics['holdout_failures']} failures     SELECTED PIPELINE: Random Forest + KNN + SMOTE",
        fontsize=10,
        weight="bold",
    )

    ax = axes[0]
    values = [metrics["holdout"]["dummy"]["average_precision"], selected["average_precision"]]
    ax.barh([1, 0], values, height=0.46, color=[palette["light"], palette["blue"]])
    low, high = selected["ap_bootstrap_95"]
    ax.hlines(0, low, high, color=palette["ink"], linewidth=2)
    ax.plot([low, high], [0, 0], "|", color=palette["ink"], markersize=10)
    ax.text(values[0] + 0.015, 1, f"{values[0]:.3f}", va="center", fontsize=12)
    ax.text(values[1], -0.4, f"{values[1]:.3f} AP", ha="center", fontsize=13, weight="bold")
    ax.set(
        yticks=[1, 0],
        yticklabels=["Constant\nscore", "Selected\npipeline"],
        xlim=(0, 0.55),
        ylim=(-0.7, 1.7),
        xlabel="Average precision (AP)",
    )
    ax.set_title("01 / Ranking contains signal", loc="left", pad=22, weight="bold", fontsize=12)
    ax.text(
        0,
        -0.32,
        f"Conditional 95% AP interval:\n{low:.3f}-{high:.3f}; uncertainty remains wide.",
        transform=ax.transAxes,
        va="top",
        color=palette["muted"],
        fontsize=10,
    )

    ax = axes[1]
    tp = [default["true_positives"], frozen["true_positives"]]
    fn = [default["false_negatives"], frozen["false_negatives"]]
    ax.barh([1, 0], tp, height=0.46, color=palette["blue"], label="Caught")
    ax.barh([1, 0], fn, left=tp, height=0.46, color=palette["light"], label="Missed")
    for y, caught, missed in zip([1, 0], tp, fn, strict=True):
        ax.text(caught + missed / 2, y, f"{missed} missed", ha="center", va="center", fontsize=9)
        ax.text(caught / 2, y + 0.34, f"{caught} caught", ha="center", fontsize=10, weight="bold")
    ax.set(
        yticks=[1, 0],
        yticklabels=["Default\n0.5000", f"Frozen\n{frozen['threshold']:.4f}"],
        xlim=(0, metrics["holdout_failures"]),
        ylim=(-0.7, 1.7),
        xlabel="Failures in the holdout (21 total)",
    )
    ax.set_title(
        f"02 / {tp[1] - tp[0]:+d} failures caught", loc="left", pad=22, weight="bold", fontsize=12
    )
    ax.text(
        0,
        -0.32,
        "Same model; threshold selected on\nout-of-fold development scores.",
        transform=ax.transAxes,
        va="top",
        color=palette["muted"],
        fontsize=10,
    )

    ax = axes[2]
    fp = [default["false_positives"], frozen["false_positives"]]
    ax.barh([1, 0], tp, height=0.46, color=palette["blue"], label="Failure reviewed")
    ax.barh([1, 0], fp, left=tp, height=0.46, color=palette["orange"], label="Passing row reviewed")
    for y, caught, passing in zip([1, 0], tp, fp, strict=True):
        ax.text(
            caught + passing + 1, y, f"{caught + passing}", va="center", weight="bold", fontsize=12
        )
    ax.set(
        yticks=[1, 0],
        yticklabels=["Default", "Frozen"],
        xlim=(0, max(frozen["alarm_count"] * 1.18, 1)),
        ylim=(-0.7, 1.7),
        xlabel=f"Reviews among {metrics['holdout_rows']} observations",
    )
    ax.set_title("03 / Reviews are the tradeoff", loc="left", pad=22, weight="bold", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=8, frameon=False)
    ax.text(
        0,
        -0.32,
        f"{frozen['alarm_count']} reviews: {frozen['true_positives']} failures, {frozen['false_positives']} passes.\nPrecision {frozen['precision']:.1%}; recall {frozen['recall']:.1%}.",
        transform=ax.transAxes,
        va="top",
        color=palette["muted"],
        fontsize=10,
    )

    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.12)

    fig.text(0.055, 0.20, "WHAT I DELIVERED", fontsize=10, weight="bold", color=palette["muted"])
    fig.text(
        0.055,
        0.16,
        f"7 controlled pipelines  /  35 primary CV fits  /  60 supplementary CV fits  /  {int(quality.retained_on_full_development.sum())} retained sensors",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.055,
        0.105,
        f"Supplementary development finding: SMOTE outperforms plain KNN/RF in {int((differences > 0).sum())}/{len(differences)} paired repeats; the top model changes across splits.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.055,
        "Scope: one small frozen holdout; supplementary repeats reuse development data. No factory yield improvement, calibrated risk, or cost savings is claimed.",
        fontsize=9,
        color=palette["muted"],
    )
    output = root / "images/findings_summary.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    inputs = [
        evidence / "metrics.json",
        evidence / "protocol.json",
        evidence / "data_quality.csv",
        supplement / "repeat_metrics.csv",
        supplement / "provenance.json",
    ]
    receipt = {
        "renderer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "output_sha256": sha256(output.read_bytes()).hexdigest(),
        "original_protocol_sha256": metrics["protocol_sha256"],
        "selected_candidate": metrics["selected_candidate"],
    }
    (supplement / "visual_summary_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Rendered {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    render(args.root)


if __name__ == "__main__":
    main()

"""Inspection-priority monitor: one page a process engineer could open each shift.

    uv pip install -r requirements-service.txt   # serving overlay, once
    uv run streamlit run dashboard/app.py

Uses the local service bundle if it exists; otherwise (e.g. on Streamlit
Community Cloud) it downloads the hash-checked SECOM files and rebuilds the
bundle from the frozen selections committed in the repository.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fabops.service import (  # noqa: E402
    DEFAULT_BUNDLE,
    batch_drift,
    build_bundle,
    load_bundle,
    score_batch,
    validate_batch,
)
from secom.data import load_dataset  # noqa: E402
from smartfab.policies import capacity_mask, chronological_partition  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
POLICY_LABELS = {
    "v3_selected_knn": "ML ranking (frozen v3 KNN)",
    "mspc": "MSPC (PCA T² + SPE)",
    "hybrid_knn_mspc": "Hybrid rank blend",
}
INK, BLUE, ORANGE, GRAY = "#263746", "#1764a0", "#c87319", "#9aa5ae"

st.set_page_config(page_title="Fab inspection monitor", layout="wide")


@st.cache_resource(show_spinner="Loading frozen models…")
def get_bundle():
    if DEFAULT_BUNDLE.is_file():
        return load_bundle(DEFAULT_BUNDLE)
    return build_bundle(DATA_DIR, ROOT / "manufacturing_results", ROOT / "fab_monitoring_results")


@st.cache_data(show_spinner="Loading demo batch…")
def demo_batch():
    dataset = load_dataset(DATA_DIR)
    _, assessment = chronological_partition(dataset.timestamps, 0.80)
    frame = dataset.X.iloc[assessment].copy()
    frame.insert(0, "unit_id", assessment)
    labels = pd.DataFrame(
        {"unit_id": assessment, "label": dataset.y[assessment], "timestamp": dataset.timestamps[assessment]}
    )
    return frame.reset_index(drop=True), labels


bundle = get_bundle()

with st.sidebar:
    st.header("Batch")
    upload = st.file_uploader("Upload a completed batch (CSV, 590 sensors + optional unit_id)", type="csv")
    st.header("Policy")
    policy = st.radio("Ranking", list(POLICY_LABELS), format_func=POLICY_LABELS.get)
    budget = st.slider("Inspection budget (% of batch)", 5, 50, int(100 * bundle["default_review_fraction"]), step=5)
    show_outcomes = False
    if upload is None:
        show_outcomes = st.toggle(
            "Show recorded outcomes (demo replay only)", value=True,
            help="The demo batch is the frozen temporal-assessment block, whose pass/fail labels are known.",
        )
    st.caption(
        f"Models trained on {bundle['provenance']['training_rows']} development units up to "
        f"{bundle['provenance']['training_last_timestamp'][:10]}. Nothing is retrained in this app."
    )

if upload is not None:
    batch = pd.read_csv(upload, dtype=str, keep_default_na=False)
    labels = None
    source = f"Uploaded batch · {upload.name}"
else:
    batch, labels = demo_batch()
    source = "Demo batch · SECOM temporal-assessment block (Oct 2–17, 2008)"

try:
    queue = score_batch(bundle, batch, budget / 100, policy)
    drift = batch_drift(bundle, batch)
except ValueError as error:
    st.error(f"Batch rejected: {error}")
    st.stop()

if labels is not None and show_outcomes:
    queue = queue.merge(labels, on="unit_id", how="left")

st.title("Fab inspection monitor")
st.caption(f"{source} · ranking: {POLICY_LABELS[policy]}")

mspc = bundle["mspc"]
alarms = int(((queue.t2_alarm == 1) | (queue.spe_alarm == 1)).sum())
cols = st.columns(4)
cols[0].metric("Units in batch", f"{len(queue)}")
cols[1].metric("Inspection queue", f"{int(queue.review.sum())}", f"{budget}% budget", delta_color="off")
cols[2].metric("MSPC limit alarms (99%)", f"{alarms}", f"{alarms / len(queue):.0%} of batch", delta_color="off")
cols[3].metric(
    "Sensor drift",
    "WARNING" if drift["warning"] else "OK",
    f"{drift['psi_alert_share']:.0%} of sensors PSI > 0.25",
    delta_color="inverse" if drift["warning"] else "off",
)
if "label" in queue:
    caught = int(queue.loc[queue.review.eq(1), "label"].sum())
    total = int(queue.label.sum())
    expected = queue.review.sum() * total / len(queue)
    st.info(
        f"Replay: this queue contains **{caught} of {total}** recorded failures; "
        f"random inspection of the same number of units would catch **{expected:.1f}** on average."
    )

tabs = st.tabs(["Inspection queue", "Alarm Pareto", "Control charts", "Drift", "Budget frontier", "Sensor watchlist"])

with tabs[0]:
    st.subheader("Today's inspection queue")
    st.caption("Ordered by priority score. Scores are uncalibrated rankings, not failure probabilities.")
    shown = queue.loc[queue.review.eq(1)].copy()
    columns = ["queue_rank", "unit_id", "priority_score", "t2", "spe", "t2_alarm", "spe_alarm", "top_spe_sensors"]
    if "label" in shown:
        shown["recorded_outcome"] = shown.label.map({0: "pass", 1: "FAIL"})
        columns.append("recorded_outcome")
    st.dataframe(shown[columns], hide_index=True, use_container_width=True)
    st.download_button("Download full scored batch (CSV)", queue.to_csv(index=False), "inspection_queue.csv")

with tabs[1]:
    st.subheader("Which sensors drive the alarms?")
    st.caption("Count of queued units in which a sensor is one of the three largest SPE contributors.")
    names = queue.loc[queue.review.eq(1), "top_spe_sensors"].str.split(", ").explode()
    pareto = names.value_counts().head(15).rename_axis("sensor").reset_index(name="units")
    pareto["cumulative"] = pareto.units.cumsum() / max(names.size, 1)
    bars = alt.Chart(pareto).mark_bar(color=BLUE).encode(
        x=alt.X("sensor:N", sort=None, title=None), y=alt.Y("units:Q", title="Queued units"),
        tooltip=["sensor", "units"],
    )
    line = alt.Chart(pareto).mark_line(color=ORANGE, point=True).encode(
        x=alt.X("sensor:N", sort=None), y=alt.Y("cumulative:Q", axis=alt.Axis(format="%", title="Cumulative share")),
    )
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)

    st.subheader("Contribution plot for one unit")
    unit = st.selectbox("Unit", queue.loc[queue.review.eq(1), "unit_id"].tolist())
    ids, sensors = validate_batch(bundle, batch)
    position = list(queue.unit_id).index(unit)
    row_index = ids.index(unit)
    statistic = st.radio("Statistic", ["spe", "t2"], format_func={"spe": "SPE (residual)", "t2": "Hotelling T²"}.get, horizontal=True)
    contribution = mspc.contributions(sensors.iloc[[row_index]], statistic).iloc[0]
    top = contribution.reindex(contribution.abs().sort_values(ascending=False).index).head(15)
    chart = alt.Chart(top.rename("contribution").rename_axis("sensor").reset_index()).mark_bar().encode(
        x=alt.X("contribution:Q"), y=alt.Y("sensor:N", sort=None, title=None),
        color=alt.condition("datum.contribution > 0", alt.value(ORANGE), alt.value(BLUE)),
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Unit {unit}: T² = {queue.t2.iloc[position]:.1f} (limit {mspc.t2_limit_:.1f}), "
        f"SPE = {queue.spe.iloc[position]:.0f} (limit {mspc.spe_limit_:.0f}). "
        "Large contributions point to where to look first; they are not a confirmed root cause."
    )

with tabs[2]:
    st.subheader("Hotelling T² and SPE control charts")
    chart_data = queue.sort_values("unit_id").reset_index(drop=True)
    chart_data["sequence"] = np.arange(len(chart_data))
    outcome = alt.value(GRAY)
    if "label" in chart_data:
        chart_data["outcome"] = chart_data.label.map({0: "pass", 1: "fail"})
        outcome = alt.Color("outcome:N", scale=alt.Scale(domain=["pass", "fail"], range=[GRAY, ORANGE]))
    for column, limit, title in (("t2", mspc.t2_limit_, "Hotelling T²"), ("spe", mspc.spe_limit_, "SPE (Q)")):
        points = alt.Chart(chart_data).mark_circle(size=28).encode(
            x=alt.X("sequence:Q", title="Unit order in batch"),
            y=alt.Y(f"{column}:Q", scale=alt.Scale(type="log"), title=f"{title} (log)"),
            color=outcome, tooltip=["unit_id", column],
        )
        rule = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=INK, strokeDash=[5, 4]).encode(y="limit:Q")
        st.altair_chart(points + rule, use_container_width=True)
    st.caption("Dashed line: 99% theoretical limit. Points far above the line on many units suggest the limits, "
               "not the process, need revisiting (non-Gaussian sensors, drift).")

with tabs[3]:
    st.subheader("Is this batch still like the training data?")
    message = (f"{drift['psi_alert_share']:.0%} of {drift['informative_sensors']} informative sensors have PSI > 0.25 "
               f"(rule: {drift['warning_rule']}).")
    (st.warning if drift["warning"] else st.success)(message)
    top = pd.DataFrame(drift["top_sensors"])
    st.altair_chart(
        alt.Chart(top).mark_bar(color=ORANGE).encode(x=alt.X("psi:Q", title="PSI vs training window"),
                                                     y=alt.Y("sensor:N", sort=None, title=None)),
        use_container_width=True,
    )
    st.dataframe(top, hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Detection vs inspection workload")
    if "label" in queue:
        rows = []
        for name in POLICY_LABELS:
            scored = score_batch(bundle, batch, 0.5, name)
            scored = scored.merge(labels, on="unit_id")
            for pct in range(5, 55, 5):
                mask = capacity_mask(scored.priority_score.to_numpy(), scored.unit_id.tolist(), pct / 100, bundle["seed"])
                rows.append({"policy": POLICY_LABELS[name], "budget": pct, "caught": int(scored.label[mask].sum())})
        total = int(labels.label.sum())
        for pct in range(5, 55, 5):
            rows.append({"policy": "Random (expected)", "budget": pct,
                         "caught": int(np.floor(len(labels) * pct / 100)) * total / len(labels)})
        frontier = pd.DataFrame(rows)
        chart = alt.Chart(frontier).mark_line(point=True).encode(
            x=alt.X("budget:Q", title="Inspection budget (% of batch)"),
            y=alt.Y("caught:Q", title=f"Recorded failures caught (of {total})"),
            color=alt.Color("policy:N", scale=alt.Scale(range=[BLUE, ORANGE, "#3d8b5f", GRAY])),
        )
        rule = alt.Chart(pd.DataFrame({"budget": [budget]})).mark_rule(color=INK, strokeDash=[4, 4]).encode(x="budget:Q")
        st.altair_chart(chart + rule, use_container_width=True)
        st.caption("Replay on the frozen assessment block: 17 failures, so one unit changes recall by about 6 points.")
    else:
        st.write("Upload labels are not used; the frontier is shown for the demo replay only.")

with tabs[5]:
    st.subheader("Sensors to check first")
    st.caption("Sensors that stay near the top across 50 bootstraps and time slices of the development data "
               "(ANOVA, random forest, MSPC-SPE). A prioritised checklist, not a root cause.")
    st.dataframe(pd.DataFrame(bundle["watchlist"]), hide_index=True, use_container_width=True)

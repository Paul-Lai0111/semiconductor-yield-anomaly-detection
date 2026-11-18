"""Fab inspection monitor: the batch read as a wafer bin map.

    uv pip install -r requirements-service.txt   # serving overlay, once
    uv run streamlit run dashboard/app.py

Uses the local service bundle if it exists; otherwise (e.g. on Streamlit
Community Cloud) it downloads the hash-checked SECOM files and rebuilds the
bundle from the frozen selections committed in the repository.
"""

import json
import sys
from hashlib import sha256
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.stats import hypergeom  # noqa: E402

from fabops.service import (  # noqa: E402
    DEFAULT_BUNDLE,
    DRIFT_WARNING_SHARE,
    batch_drift,
    build_bundle,
    load_bundle,
    score_batch,
    validate_batch,
)
from secom.data import load_dataset  # noqa: E402
from smartfab.policies import capacity_mask, chronological_partition  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
REPO = "https://github.com/Paul-Lai0111/semiconductor-yield-anomaly-detection"
POLICIES = {
    "v3_selected_knn": "ML ranking (frozen KNN)",
    "mspc": "MSPC (PCA, SPE)",
    "hybrid_knn_mspc": "Hybrid rank blend",
}

# World tokens. Bin colours carry state; graphite carries ink and structure.
PANEL, GRAPHITE, MUTED, HAIRLINE = "#F4F6F7", "#1E2328", "#5B656E", "#CBD1D6"
PASS_LIGHT, FAIL, ALARM, STEEL = "#B9D9C2", "#C43C3C", "#C9851F", "#4A6A8A"
BINS = {
    "Caught failure": (FAIL, "square"),
    "False alarm": (GRAPHITE, "square"),
    "Missed failure": (FAIL, "outline"),
    "Not queued": (PASS_LIGHT, "square"),
    "Queued": (GRAPHITE, "square"),
}

st.set_page_config(page_title="Fab inspection monitor", layout="wide")

st.markdown(
    f"""
<style>
::selection {{ background: {PASS_LIGHT}; color: {GRAPHITE}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1480px; }}
.lede {{ font-size: 1.06rem; line-height: 1.5; max-width: 68ch; margin: -0.2rem 0 0.35rem; }}
.lede b {{ font-weight: 650; }}
.byline {{ color: {MUTED}; font-size: 0.86rem; margin-bottom: 1.1rem; max-width: 92ch; }}
.byline a {{ color: {GRAPHITE}; text-underline-offset: 3px; }}
.readout {{ width: 100%; border-collapse: collapse; background: {PANEL}; border: 1px solid {HAIRLINE};
  border-radius: 6px; overflow: hidden; font-variant-numeric: tabular-nums; }}
.readout th {{ text-align: left; font-weight: 500; color: {MUTED}; font-size: 0.82rem; padding: 12px 16px;
  border-bottom: 1px solid {HAIRLINE}; width: 36%; vertical-align: top; }}
.readout td {{ padding: 10px 16px 12px; border-bottom: 1px solid {HAIRLINE}; }}
.readout tr:last-child th, .readout tr:last-child td {{ border-bottom: 0; }}
.readout .v {{ font-size: 1.45rem; font-weight: 650; letter-spacing: -0.01em; }}
.readout .v small {{ font-size: 0.9rem; font-weight: 500; color: {MUTED}; }}
.readout .n {{ display: block; font-size: 0.82rem; color: {MUTED}; margin-top: 2px; }}
.chip {{ display: inline-block; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.04em; padding: 2px 8px;
  border-radius: 4px; vertical-align: 3px; margin-left: 8px; }}
.chip.warn {{ background: #F3E2C4; color: #7A4A0C; }} .chip.ok {{ background: #D5EBDC; color: #1F6136; }}
.mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.86em; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.82rem; color: {MUTED}; margin: -4px 0 4px 6px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 11px; height: 11px; display: inline-block; border-radius: 1px; background: var(--c); }}
.legend i.o {{ background: transparent; box-shadow: inset 0 0 0 2px var(--c); }}
.st-key-wafer {{ aspect-ratio: 1 / 1; max-width: 600px; width: 100%; height: auto !important;
  flex: 0 0 auto !important; }}
.st-key-wafer > div, .st-key-wafer [data-testid="stVegaLiteChart"] {{ height: 100%; }}
.note {{ color: {MUTED}; font-size: 0.8rem; margin: 2px 0 0 6px; max-width: 72ch; }}
.unit-head {{ font-size: 1.02rem; font-weight: 650; margin: 0.2rem 0 0.1rem; }}
.unit-sub {{ color: {MUTED}; font-size: 0.84rem; margin-bottom: 0.4rem; }}
.prose {{ max-width: 70ch; line-height: 1.6; }}
.prose h3 {{ margin-top: 1.4rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def wafer_theme():
    return {
        "config": {
            "background": "transparent",
            "font": "Public Sans",
            "view": {"stroke": None},
            "axis": {
                "labelColor": MUTED, "titleColor": MUTED, "titleFontWeight": 500, "titleFontSize": 12,
                "labelFontSize": 11, "gridColor": "#D9DDE1", "domainColor": HAIRLINE, "tickColor": HAIRLINE,
            },
            "legend": {"labelColor": MUTED, "titleColor": MUTED, "labelFontSize": 12, "orient": "bottom"},
            "title": {"color": GRAPHITE, "fontSize": 13, "fontWeight": 600, "anchor": "start"},
        }
    }


alt.theme.register("wafer", enable=True)(wafer_theme)


# ---------------------------------------------------------------- data and scoring

@st.cache_resource(show_spinner="Loading the frozen models (a first visit rebuilds them, about a minute)…")
def get_bundle():
    if DEFAULT_BUNDLE.is_file():
        return load_bundle(DEFAULT_BUNDLE)
    return build_bundle(DATA_DIR, ROOT / "manufacturing_results", ROOT / "fab_monitoring_results")


@st.cache_data(show_spinner="Loading the demo batch…")
def demo_batch():
    dataset = load_dataset(DATA_DIR)
    _, assessment = chronological_partition(dataset.timestamps, 0.80)
    frame = dataset.X.iloc[assessment].copy()
    frame.insert(0, "unit_id", assessment)
    labels = pd.DataFrame({"unit_id": assessment, "label": dataset.y[assessment]})
    return frame.reset_index(drop=True), labels


@st.cache_data(show_spinner=False)
def scored(_bundle, _batch, digest, fraction, policy):
    return score_batch(_bundle, _batch, fraction, policy)


@st.cache_data(show_spinner=False)
def drift_of(_bundle, _batch, digest):
    return batch_drift(_bundle, _batch)


@st.cache_data(show_spinner=False)
def sensors_of(_bundle, _batch, digest):
    return validate_batch(_bundle, _batch)


def wafer_layout(n: int) -> pd.DataFrame:
    """Die cells inside a circle, ordered along a serpentine probe path."""
    side = int(np.ceil(np.sqrt(n * 4 / np.pi))) + 1
    while True:
        cells = []
        centre = (side - 1) / 2
        for row in range(side):
            cols = range(side) if row % 2 == 0 else range(side - 1, -1, -1)
            for col in cols:
                dx, dy = abs(col - centre) + 0.41, abs(row - centre) + 0.41
                if dx ** 2 + dy ** 2 <= (side / 2 - 0.15) ** 2:
                    cells.append((col - centre, centre - row))
        if len(cells) >= n:
            return pd.DataFrame(cells[:n], columns=["x", "y"]).assign(radius=side / 2)
        side += 1


@st.cache_data(show_spinner=False)
def evidence():
    fab = json.loads((ROOT / "fab_monitoring_results" / "metrics.json").read_text())
    data = json.loads((ROOT / "manufacturing_results" / "protocol.json").read_text())["dataset"]
    return fab, data


bundle = get_bundle()
mspc = bundle["mspc"]
fab_metrics, dataset_facts = evidence()
noise_ceiling = fab_metrics["drift_noise_floor"]["random_split_share_max"]

# ---------------------------------------------------------------- sidebar controls

with st.sidebar:
    st.markdown("### Batch controls")
    st.caption("Rank a completed batch of SECOM units for inspection.")
    upload = st.file_uploader(
        "Score your own batch", type="csv",
        help="CSV with the 590 sensor columns and an optional integer unit_id. Blank cells are missing values.",
    )
    policy = st.radio("Ranking", list(POLICIES), format_func=POLICIES.get)
    budget = st.slider("Inspection budget", 5, 50, int(100 * bundle["default_review_fraction"]), step=5,
                       format="%d%%")
    show_outcomes = False
    if upload is None:
        show_outcomes = st.toggle(
            "Show recorded outcomes", value=True,
            help="The demo batch is the frozen temporal-assessment block, so its pass/fail labels are known.",
        )
    st.divider()
    st.caption(
        f"Models were fitted on {bundle['provenance']['training_rows']:,} earlier units, up to "
        f"{bundle['provenance']['training_last_timestamp'][:10]}, and frozen. Nothing retrains here."
    )
    st.caption(f"[Source on GitHub]({REPO}) · [Engineering report]({REPO}/blob/main/docs/16_engineering_report.md)")

if upload is not None:
    batch = pd.read_csv(upload, dtype=str, keep_default_na=False)
    labels = None
    source = f"Uploaded batch, {escape(upload.name)}"
else:
    batch, labels = demo_batch()
    source = "Demo replay of the SECOM temporal-assessment block, Oct 2–17, 2008"
digest = sha256(pd.util.hash_pandas_object(batch, index=False).values.tobytes()).hexdigest()

try:
    queue = scored(bundle, batch, digest, budget / 100, policy)
    drift = drift_of(bundle, batch, digest)
    ids, sensors = sensors_of(bundle, batch, digest)
except ValueError as error:
    st.title("Fab inspection monitor")
    st.error(f"This batch can't be scored: {error} Fix the file and upload it again, or remove it to see the demo.")
    st.stop()

if labels is not None and show_outcomes:
    queue = queue.merge(labels, on="unit_id", how="left")
has_outcomes = "label" in queue
n_units, n_queued = len(queue), int(queue.review.sum())
alarm = (queue.t2_alarm == 1) | (queue.spe_alarm == 1)
order = pd.Series(range(n_units), index=ids)  # batch order, which is time order for the demo

# ---------------------------------------------------------------- opening: the finding

st.title("Fab inspection monitor")
if has_outcomes:
    caught = int(queue.loc[queue.review.eq(1), "label"].sum())
    failures = int(queue.label.sum())
    expected = n_queued * failures / n_units
    low, high = hypergeom.ppf([0.025, 0.975], n_units, failures, n_queued).astype(int)
    rival = "v3_selected_knn" if policy == "mspc" else "mspc"
    rival_queue = scored(bundle, batch, digest, budget / 100, rival).merge(labels, on="unit_id")
    rival_caught = int(rival_queue.loc[rival_queue.review.eq(1), "label"].sum())
    rival_name = "the ML ranking" if rival == "v3_selected_knn" else "MSPC limits"
    st.markdown(
        f"<p class='lede'>Inspect <b>{n_queued} of {n_units}</b> units. The {POLICIES[policy].split(' (')[0]} "
        f"queue contains <b>{caught} of {failures}</b> recorded failures. Inspecting the same number at random "
        f"finds {expected:.1f} on average; {rival_name} find{"s" if rival == "v3_selected_knn" else ""} {rival_caught}.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<p class='lede'>Inspect <b>{n_queued} of {n_units}</b> units, ranked by {POLICIES[policy]}. "
        "Recorded outcomes are not used for uploaded batches.</p>",
        unsafe_allow_html=True,
    )
st.markdown(
    f"<div class='byline'>{source}. A frozen, chronologically validated ranking decides which units to inspect "
    f"when only part of a batch can be reviewed. <a href='{REPO}'>How it was built</a><br>"
    "Change the budget, ranking or batch in the sidebar, or with » on a phone.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- wafer bin map + readout

layout = wafer_layout(n_units)
dies = queue.assign(position=queue.unit_id.map(order)).sort_values("position").reset_index(drop=True)
dies = pd.concat([dies, layout], axis=1)
if has_outcomes:
    dies["bin"] = np.select(
        [dies.review.eq(1) & dies.label.eq(1), dies.review.eq(1), dies.label.eq(1)],
        ["Caught failure", "False alarm", "Missed failure"], "Not queued",
    )
    dies["outcome"] = dies.label.map({0: "pass", 1: "recorded failure"})
else:
    dies["bin"] = np.where(dies.review.eq(1), "Queued", "Not queued")
    dies["outcome"] = "not used"

if "unit_pick" not in st.session_state or st.session_state.unit_pick not in set(queue.unit_id):
    st.session_state.unit_pick = int(queue.unit_id.iloc[0])

left, right = st.columns([0.58, 0.42], gap="large")
with left:
    radius = float(layout.radius.iloc[0])
    scale = alt.Scale(domain=[-radius - 0.6, radius + 0.6], nice=False)
    angle = np.linspace(-np.pi / 2 + 0.06, 3 * np.pi / 2 - 0.06, 240)
    outline = pd.DataFrame({"x": radius * np.cos(angle), "y": radius * np.sin(angle)})
    notch = pd.DataFrame({"x": [outline.x.iloc[-1], 0.0, outline.x.iloc[0]],
                          "y": [outline.y.iloc[-1], -radius + 0.7, outline.y.iloc[0]]})
    ring_data = pd.concat([outline, notch], ignore_index=True).assign(i=lambda d: range(len(d)))
    # Cells are rectangles in data units, so they tile at any rendered size.
    half = 0.41
    dies = dies.assign(x0=dies.x - half, x1=dies.x + half, y0=dies.y - half, y1=dies.y + half,
                       hollow=dies.bin.map(lambda b: BINS[b][1] == "outline"))
    pick = alt.selection_point(fields=["unit_id"], name="die", on="click", clear=False)
    present = [b for b in BINS if b in set(dies.bin)]
    x, y = alt.X("x:Q", scale=scale, axis=None), alt.Y("y:Q", scale=scale, axis=None)
    box = {"x": alt.X("x0:Q", scale=scale, axis=None), "x2": "x1:Q",
           "y": alt.Y("y0:Q", scale=scale, axis=None), "y2": "y1:Q"}
    colour = alt.Scale(domain=present, range=[BINS[b][0] for b in present])
    ring = alt.Chart(ring_data).mark_line(color=GRAPHITE, strokeWidth=1.2, opacity=0.5).encode(x=x, y=y, order="i:Q")
    cells = alt.Chart(dies).mark_rect(strokeWidth=1.6, cursor="pointer").encode(
        **box,
        fill=alt.condition("datum.hollow", alt.value("transparent"), alt.Color("bin:N", scale=colour, legend=None)),
        stroke=alt.condition("datum.hollow", alt.Color("bin:N", scale=colour, legend=None), alt.value(None)),
        tooltip=[
            alt.Tooltip("unit_id:Q", title="Unit"), alt.Tooltip("queue_rank:Q", title="Queue rank"),
            alt.Tooltip("priority_score:Q", title="Priority", format=".3f"), alt.Tooltip("bin:N", title="Bin"),
            alt.Tooltip("top_spe_sensors:N", title="Top SPE sensors"),
        ],
    ).add_params(pick)
    chosen = dies.loc[dies.unit_id.eq(st.session_state.unit_pick)]
    halo = alt.Chart(chosen.assign(x0=chosen.x - 0.78, x1=chosen.x + 0.78, y0=chosen.y - 0.78, y1=chosen.y + 0.78)
                     ).mark_rect(fill="transparent", stroke=GRAPHITE, strokeWidth=2.2).encode(**box)
    wafer = alt.layer(ring, cells, halo).properties(width="container", height="container")
    event = st.altair_chart(wafer, on_select="rerun", key="wafer", theme=None, width="stretch", height="stretch")
    selected = (event.selection.get("die") or [{}]) if event and event.selection else [{}]
    clicked = selected[0].get("unit_id") if selected else None
    if clicked is not None and clicked != st.session_state.get("last_click"):
        st.session_state.last_click = clicked
        st.session_state.unit_pick = int(clicked)
        st.rerun()
    legend = "".join(
        f"<span><i class='{'o' if BINS[b][1] == 'outline' else ''}' style='--c:{BINS[b][0]}'></i>{b} "
        f"<b class='mono'>{int((dies.bin == b).sum())}</b></span>"
        for b in present
    )
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>Each cell is one unit, laid along a serpentine probe path in time order. SECOM records "
        "no die positions, so this is a batch map, not a spatial wafer map. Click a cell to inspect it.</div>",
        unsafe_allow_html=True,
    )

with right:
    rows = [("Inspection queue", f"<span class='v'>{n_queued} <small>of {n_units} units</small></span>"
             f"<span class='n'>{budget}% budget, exactly ⌊{budget}% × {n_units}⌋; ties broken by unit ID</span>")]
    if has_outcomes:
        rows.append(("Recorded failures found", f"<span class='v'>{caught} <small>of {failures}</small></span>"
                     f"<span class='n'>Random inspection: {expected:.1f} expected, 95% range {low}–{high}</span>"))
    rows.append(("MSPC limit alarms",
                 f"<span class='v'>{int(alarm.sum())} <small>units, {alarm.mean():.0%}</small></span>"
                 "<span class='n'>T² or SPE above its 99% limit; about 1% is expected in control</span>"))
    chip = "<span class='chip warn'>WARNING</span>" if drift["warning"] else "<span class='chip ok'>OK</span>"
    rows.append(("Sensor drift",
                 f"<span class='v'>{drift['psi_alert_share']:.0%} <small>of {drift['informative_sensors']} "
                 f"sensors</small>{chip}</span><span class='n'>PSI above 0.25 against the training window; "
                 f"warning at {DRIFT_WARNING_SHARE:.0%}</span>"))
    st.markdown(
        "<table class='readout'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>",
        unsafe_allow_html=True,
    )
    st.write("")

    queued_ids = queue.loc[queue.review.eq(1), "unit_id"].tolist()
    all_ids = queue.unit_id.tolist()

    def step(delta):
        browse = queued_ids if st.session_state.unit_pick in queued_ids else all_ids
        position = browse.index(st.session_state.unit_pick)
        st.session_state.unit_pick = browse[(position + delta) % len(browse)]

    nav = st.columns([1, 1, 2.4])
    nav[0].button("Previous", on_click=step, args=(-1,), width="stretch")
    nav[1].button("Next", on_click=step, args=(1,), width="stretch")
    nav[2].selectbox("Unit", all_ids, key="unit_pick", label_visibility="collapsed",
                     format_func=lambda u: f"Unit {u}")
    unit = st.session_state.unit_pick
    row = queue.loc[queue.unit_id.eq(unit)].iloc[0]
    status = "In the queue" if row.review else "Not queued"
    outcome = f", recorded {'failure' if row.label else 'pass'}" if has_outcomes else ""
    st.markdown(
        f"<div class='unit-head'>Unit <span class='mono'>{unit}</span> · rank {int(row.queue_rank)} of {n_units}</div>"
        f"<div class='unit-sub'>{status}{outcome} · T² {row.t2:,.0f} (limit {mspc.t2_limit_:,.0f}) · "
        f"SPE {row.spe:,.0f} (limit {mspc.spe_limit_:,.0f})</div>",
        unsafe_allow_html=True,
    )
    contribution = mspc.contributions(sensors.iloc[[ids.index(unit)]], "spe").iloc[0]
    top = contribution.sort_values(ascending=False).head(8).rename("share").rename_axis("sensor").reset_index()
    top["share"] = top["share"] / max(contribution.sum(), 1e-12)
    st.altair_chart(
        alt.Chart(top).mark_bar(color=GRAPHITE, height=14).encode(
            x=alt.X("share:Q", title="Share of this unit's SPE", axis=alt.Axis(format="%")),
            y=alt.Y("sensor:N", sort=None, title=None, axis=alt.Axis(labelFont="JetBrains Mono", minExtent=96, labelLimit=0)),
            tooltip=["sensor", alt.Tooltip("share:Q", format=".1%")],
        ).properties(height=220),
        width="stretch", theme=None,
    )
    st.caption("Largest SPE contributors: where to look first, not a confirmed root cause.")

# ---------------------------------------------------------------- working tabs

tabs = st.tabs([f"Queue · {n_queued}", f"Alarms · {int(alarm.sum())}", "Control charts",
                f"Drift · {drift['psi_alert_share']:.0%}", "Budget", f"Watchlist · {len(bundle['watchlist'])}",
                "About"])

with tabs[0]:
    shown = queue.loc[queue.review.eq(1)].copy()
    columns = {"queue_rank": "Rank", "unit_id": "Unit", "priority_score": "Priority", "t2": "T²", "spe": "SPE",
               "top_spe_sensors": "Top SPE sensors"}
    if has_outcomes:
        shown["outcome"] = shown.label.map({0: "pass", 1: "FAIL"})
        columns["outcome"] = "Recorded"
    if shown.empty:
        st.info(f"The budget is too small for this batch: ⌊{budget}% × {n_units}⌋ = 0 units. Raise the budget.")
    else:
        table = shown[list(columns)].rename(columns=columns).round({"T²": 0, "SPE": 0})
        styled = table.style.format({"Priority": "{:.3f}", "T²": "{:,.0f}", "SPE": "{:,.0f}"})
        if has_outcomes:
            styled = styled.map(lambda v: f"color: {FAIL}; font-weight: 650" if v == "FAIL" else "", subset=["Recorded"])
        st.dataframe(styled, hide_index=True, width="stretch")
    st.download_button("Download the scored batch (CSV)", queue.to_csv(index=False), "inspection_queue.csv")
    st.caption("Priority is an uncalibrated ranking score, not a failure probability.")

with tabs[1]:
    st.markdown("#### Which sensors drive the queue?")
    names = queue.loc[queue.review.eq(1), "top_spe_sensors"].str.split(", ").explode()
    pareto = names.value_counts().head(15).rename_axis("sensor").reset_index(name="units")
    pareto["cumulative"] = pareto.units.cumsum() / max(names.size, 1)
    bars = alt.Chart(pareto).mark_bar(color=GRAPHITE).encode(
        x=alt.X("sensor:N", sort=None, title=None, axis=alt.Axis(labelFont="JetBrains Mono", labelAngle=-40)),
        y=alt.Y("units:Q", title="Queued units with this sensor in their top three"), tooltip=["sensor", "units"])
    line = alt.Chart(pareto).mark_line(color=ALARM, point=alt.OverlayMarkDef(color=ALARM, filled=True)).encode(
        x=alt.X("sensor:N", sort=None), y=alt.Y("cumulative:Q", axis=alt.Axis(format="%", title="Cumulative share")))
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=340),
                    width="stretch", theme=None)
    st.caption(f"{int(alarm.sum())} of {n_units} units exceed a 99% MSPC limit. On this process the limits fire on "
               "most later units because the sensors drift; the Drift tab shows how much.")

with tabs[2]:
    chart_data = dies[["position", "unit_id", "t2", "spe", "outcome"]].copy()
    if has_outcomes:
        domain = ["pass", "recorded failure"]
        shape = alt.Shape("outcome:N", scale=alt.Scale(domain=domain, range=["circle", "triangle-up"]),
                          legend=alt.Legend(title=None))
        color = alt.Color("outcome:N", scale=alt.Scale(domain=domain, range=["#8C969F", FAIL]),
                          legend=alt.Legend(title=None))
    else:
        shape, color = alt.value("circle"), alt.value("#8C969F")
    for column, limit, title in (("t2", mspc.t2_limit_, "Hotelling T²"), ("spe", mspc.spe_limit_, "SPE (Q statistic)")):
        points = alt.Chart(chart_data).mark_point(filled=True, size=26, opacity=0.85).encode(
            x=alt.X("position:Q", title="Unit order in batch"),
            y=alt.Y(f"{column}:Q", scale=alt.Scale(type="log"), title=f"{title}, log scale"),
            color=color, shape=shape, tooltip=["unit_id", alt.Tooltip(f"{column}:Q", format=",.0f")])
        rule = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=GRAPHITE, strokeDash=[5, 4]).encode(
            y="limit:Q")
        st.altair_chart((points + rule).properties(height=260, title=f"{title}, 99% limit dashed"),
                        width="stretch", theme=None)

with tabs[3]:
    st.markdown("#### Is this batch still like the training window?")
    share = f"{drift['psi_alert_share']:.0%} of {drift['informative_sensors']} informative sensors have PSI above 0.25."
    if drift["warning"]:
        message = (f"{share} Random splits of one training window flag at most {noise_ceiling:.1%}, so this is real "
                   "drift, not sampling noise.")
    else:
        message = f"{share} That is below the {DRIFT_WARNING_SHARE:.0%} warning level."
    (st.warning if drift["warning"] else st.success)(message)
    top = pd.DataFrame(drift["top_sensors"])
    st.altair_chart(
        alt.Chart(top).mark_bar(color=ALARM, height=16).encode(
            x=alt.X("psi:Q", title="PSI against the training window"),
            y=alt.Y("sensor:N", sort=None, title=None, axis=alt.Axis(labelFont="JetBrains Mono", minExtent=96, labelLimit=0)),
            tooltip=["sensor", alt.Tooltip("psi:Q", format=".2f")]).properties(height=300),
        width="stretch", theme=None)

with tabs[4]:
    if has_outcomes:
        rows = []
        for name in POLICIES:
            ranked = scored(bundle, batch, digest, 0.5, name).merge(labels, on="unit_id")
            for pct in range(5, 55, 5):
                mask = capacity_mask(ranked.priority_score.to_numpy(), ranked.unit_id.tolist(), pct / 100,
                                     bundle["seed"])
                rows.append({"policy": POLICIES[name], "budget": pct, "found": int(ranked.label[mask].sum())})
        for pct in range(5, 55, 5):
            rows.append({"policy": "Random (expected)", "budget": pct,
                         "found": np.floor(n_units * pct / 100) * failures / n_units})
        frontier = pd.DataFrame(rows)
        names = [*POLICIES.values(), "Random (expected)"]
        chart = alt.Chart(frontier).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("budget:Q", title="Inspection budget, % of batch"),
            y=alt.Y("found:Q", title=f"Recorded failures found, of {failures}"),
            color=alt.Color("policy:N", scale=alt.Scale(domain=names, range=[GRAPHITE, ALARM, STEEL, "#9AA3AB"]),
                            legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash("policy:N", legend=None,
                                      scale=alt.Scale(domain=names, range=[[1, 0], [1, 0], [6, 3], [2, 3]])))
        rule = alt.Chart(pd.DataFrame({"budget": [budget]})).mark_rule(color=GRAPHITE, strokeDash=[3, 3]).encode(
            x="budget:Q")
        st.altair_chart((chart + rule).properties(height=360), width="stretch", theme=None)
        st.caption(f"The dashed rule is the budget in the sidebar. With {failures} failures, one unit moves recall "
                   f"by about {100 / max(failures, 1):.0f} points.")
    else:
        st.info("The detection frontier needs recorded outcomes, so it is shown for the demo batch only.")

with tabs[5]:
    st.markdown("#### Sensors to check first")
    watch = pd.DataFrame(bundle["watchlist"]).rename(
        columns={"sensor": "Sensor", "tier": "Stability", "consensus_freq": "Top-20 frequency"})
    st.dataframe(watch, hide_index=True, width="content",
                 column_config={"Top-20 frequency": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Share of 50 bootstrap resamples in which a sensor ranks in the top 20, averaged over ANOVA, random "
               "forest and MSPC-SPE. Rankings barely survive across time slices, so treat this as a checklist.")

with tabs[6]:
    about = {
        "ml": fab_metrics["assessment"]["v3_selected_knn"]["capacity"]["true_positives"],
        "mspc": fab_metrics["assessment"]["mspc"]["capacity"]["true_positives"],
        "failures": fab_metrics["assessment_failures"],
        "random": fab_metrics["random_expected_caught"],
    }
    st.markdown(
        f"""<div class='prose'>
<h3>What this is</h3>
<p>A side project on the UCI SECOM dataset: {dataset_facts["rows"]:,} production units, {dataset_facts["sensor_columns"]}
anonymised sensors, {dataset_facts["failures"]} recorded failures.
The question is the one a fab faces every shift: if only part of a completed batch can be inspected, which units
go first?</p>
<h3>How the ranking was chosen</h3>
<p>Fifty model configurations were compared on the earliest 80% of time, using three expanding forward-validation
blocks. The selection rule, the winner and its threshold were saved and hashed before the latest 20% of units was
scored once. The demo on this page replays that final block.</p>
<h3>What it found</h3>
<p>At the frozen 20% budget the ranking finds {about["ml"]} of {about["failures"]} later failures, against
{about["random"]:.1f} for random inspection and {about["mspc"]} for MSPC, the PCA-based control charts fabs typically run. The sensors drift continuously, so MSPC's fixed limits alarm
on most later units, and periodic retraining helps the ranking in replay without reaching statistical
significance.</p>
<h3>Limits</h3>
<p>{about["failures"]} failures leave wide uncertainty. Sensors are anonymised, so contributions are inspection hints, not
root causes. No yield, throughput or cost benefit was measured.</p>
<p><a href='{REPO}'>Code, tests and protocol on GitHub</a> ·
<a href='{REPO}/blob/main/docs/16_engineering_report.md'>Engineering report</a></p>
</div>""",
        unsafe_allow_html=True,
    )

---
version: 1
slug: "dashboard-app-py"
primary_target: "dashboard/app.py"
related_targets: []
---

# Surface: Fab inspection monitor (dashboard/app.py)

Scope: the public Streamlit app. Mode: Operate (a working tool), with a recruiter-first opening.
Audience: recruiters and interviewers from the portfolio first; process engineers second. English only.
Task: understand the finding in seconds, then operate the queue: budget, policy, unit diagnostics, drift.
Constraints: Streamlit + Altair only; dashboard/ and .streamlit/ only (fabops/ is hash-frozen). Every number from fabops/service output or committed results.

## Direction contract

THESIS: The batch is read the way a yield engineer reads a lot: as a wafer bin map. Refuses the KPI-tile grid over tabs that every Streamlit portfolio ships.

OWN-WORLD: Polished-silicon grey ground (#E4E7EA) with lighter panels (#F4F6F7); graphite (#1E2328) ink and sidebar rail; bin colours do the state work: bin green pass, bin red recorded failure, bin amber MSPC alarm / drift; hairline #CBD1D6 rules. Public Sans for UI, JetBrains Mono only for measurements, unit IDs and sensor names. Square cells, 6px radius panels, no shadows beyond one soft lift.

STORY: Visitor reads one sentence (inspect 62 of 314; ranking finds 5 of 17 vs 3.4 random, MSPC 2), sees it on the wafer, moves the budget and watches the queue boundary sweep across the map, clicks a die to see why it was flagged.

FIRST VIEWPORT: Graphite sidebar with batch, ranking and budget controls. Main: title plus the finding sentence; left 58% the wafer bin map of all units on a serpentine probe path in time order, notch at bottom, legend with shape plus colour; right 42% a readout spec table (queue, failures found with random range, MSPC alarms, drift state) and the selected-die detail. Tabs with live counts below.

FORM: Yield Wafer Map, position 1 on the ordered list (IMPECCABLE’S PICK). seed:6502010410b735aa
Signature interaction: budget slider redraws which dies are queued; clicking a die selects it for the contribution detail (prev/next in queue order).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

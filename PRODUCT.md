# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Primary: recruiters, hiring managers and technical interviewers who open the public Streamlit app from Paul Lai's portfolio, résumé or GitHub README. They have a few minutes, may not know semiconductor FDC, and need to understand within seconds what the tool does, what it found, and that it is a working tool rather than a static report.
Secondary: process / yield engineers evaluating whether the workflow (inspection queue, MSPC diagnostics, drift checks) would fit a fab shift routine.

## Product Purpose
A fab inspection-priority monitor built on UCI SECOM. Given a completed batch of units with 590 sensor readings, it ranks which units to inspect under a fixed review budget, explains alarms with MSPC (Hotelling T² / SPE) contributions, and flags sensor drift against the training window. Success: a visitor leaves knowing the result (supervised ranking catches 5 / 17 later failures in 62 reviews vs 3.4 at random and 2 for MSPC) and has tried the tool themselves.

## Positioning
Every number on screen comes from a frozen, hash-verified temporal protocol, and the app states its own limits: uncalibrated scores, small failure counts, anonymised sensors, no root-cause claims. Honest, inspectable evidence is the point, not a high score.

## Operating Context
Public Streamlit Community Cloud app (https://semiconductor-yield-anomaly-detection-paullai.streamlit.app/), kept awake by the private streamlit-keepalive workflow. Cold start downloads SECOM and rebuilds the bundle. Default content is a demo replay of the frozen 314-unit temporal-assessment block, whose labels are known; visitors may also upload their own CSV.

## Capabilities and Constraints
- Stack: Streamlit (dashboard/app.py) on top of the `fabops` library; Altair charts; theme in .streamlit/config.toml. Custom CSS via st.markdown is allowed; no separate frontend build.
- Policies: frozen v3 KNN ranking, MSPC SPE ranking, hybrid rank blend. Budget 5–50%.
- Views: inspection queue, alarm Pareto + per-unit contribution plot, T²/SPE control charts, drift, budget frontier, sensor watchlist.
- Terminology: "unit", "inspection queue", "review budget", "recorded failure", "T²", "SPE", "PSI".
- `fabops/*.py`, `smartfab/`, `secom/`, `pyproject.toml`, `uv.lock` are hash-frozen; dashboard changes must stay in `dashboard/` and `.streamlit/`.
- English only.

## Brand Commitments
Author identity shown on the app and README: Paul Lai, M.S. in Data Science, UC San Diego, holai@ucsd.edu, GitHub Paul-Lai0111.

## Evidence on Hand
fab_monitoring_results/ (metrics.json, RESULTS.md, figures), manufacturing_results/ (v3), docs/16_engineering_report.md. No users, testimonials, deployments or measured yield/cost savings exist; never imply them.

## Product Principles
1. Lead with the finding, then let the visitor operate the tool.
2. Every number is traceable and honest; show uncertainty and the random baseline next to every catch count.
3. Diagnostics point where to look, never claim root cause.
4. Engineer-grade clarity: scanable, dense where it helps, no decoration that hides data.

## Accessibility & Inclusion
Charts must not rely on colour alone (pass/fail also distinguished by shape or label); readable contrast in the light theme.

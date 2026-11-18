# Engineering report: inspection prioritisation for a wafer-level sensor stream

*Format: problem → current practice → proposal → measured benefit → risks → go-live conditions.
Every number links to a generated file; nothing is estimated by hand.*

## 1. Problem

A line produces units with 590 in-line sensor readings each. About 6.6% later fail.
Inspection capacity is limited: assume **20% of a completed batch** can be reviewed.
The question is which units to review so that the most recorded failures are found,
and what it costs to keep that decision working as the process changes.

Data: UCI SECOM, 1,567 units, 104 failures, 13 weeks, anonymised sensors, no lot/tool/wafer IDs.
Evaluation boundary: earliest 80% (1,253 units, 87 failures) for all development decisions,
latest 20% (314 units, 17 failures) scored once after every choice was frozen.

## 2. Current practice: MSPC fault detection

Fabs typically run multivariate statistical process control on tool data: a PCA model of
in-control behaviour, Hotelling T² (unusual size within the normal correlation structure)
and SPE/Q (the correlation structure itself breaks), with 99% control limits and
contribution plots for diagnosis.

Implemented in [`fabops/mspc.py`](../fabops/mspc.py): fit on development **passes only**,
training-local sensor pruning, median imputation, z-scoring; T² limit from the F distribution,
SPE limit from Box's g·χ²(h) approximation. Nine variants (80/90/95% variance × T²/SPE/combined)
were scored on the same three forward development blocks as the v3 model search and the
best (`mspc_var95_spe`, 167 components) frozen before assessment.

| Assessment block, 62 reviews (20%) | Failures caught / 17 | Review precision |
|---|---:|---:|
| Supervised ranking (frozen v3 KNN) | **5** | 8.1% |
| MSPC, SPE ranking | 2 | 3.2% |
| Hybrid rank blend (KNN + MSPC) | 3 | 4.8% |
| Random review (expected) | 3.4 | 5.4% |

Run as a classic control chart (alarm when T² or SPE exceeds the 99% limit), MSPC alarms on
**72% of later passing units** (9% in-sample on development passes, versus the nominal 1%).
With empirically calibrated limits the later false-alarm rate is still 64%.

**Reading:** on this process the "abnormal versus normal operation" signal that MSPC measures is
dominated by drift, not by the conditions that precede a recorded failure. The two detectors
mostly catch *different* failures (KNN only 4, MSPC only 1, both 1), so MSPC adds diagnostic
value but not detection value at this budget.

Source: [fab_monitoring_results/RESULTS.md](../fab_monitoring_results/RESULTS.md) §1.

## 3. Proposal

1. **Order the inspection queue with the supervised ranking** (frozen v3 KNN), exactly
   `floor(20% × batch)` units per completed batch.
2. **Keep MSPC as the diagnostic view, not the detector.** For each queued unit show the top SPE/T²
   contributors, and a shift-level alarm Pareto, so engineers know where to look first.
3. **Retrain on a fixed schedule** (every 3 batches of 100) on all labelled data, rather than on a
   drift trigger.
4. **Monitor drift continuously but report it as context**, with the warning rule calibrated to
   this process (see §5).

Delivered as a batch-scoring API (FastAPI), a container, and an engineer-facing dashboard
(inspection queue, alarm Pareto, control charts, drift, budget slider, sensor watchlist).

## 4. Measured benefit

| Evidence | Result | Caveat |
|---|---|---|
| Frozen temporal assessment | 5 / 17 failures in 62 reviews vs 3.4 expected at random | Random 95% range is 1–7; not proof of superiority |
| Supervised vs MSPC | 5 vs 2 at 20%, 8 vs 3 at 30%, 10 vs 3 at 50% | One failure ≈ 6 recall points |
| Prequential replay, 1,100 units, label delay 1 batch | Frozen 10 / 46 → periodic (3 retrains) 14 / 46 → always (9 retrains) 15 / 46 | Sign test p = 0.45–0.69; direction consistent, not significant |
| Sensor-ranking stability | Top-20 Jaccard ≈ 0.2 across bootstraps, ≈ 0.01 across time slices; 1 stable sensor (`sensor_059`) | Rankings are not a reliable root-cause list |

Not measured: prevented defects, yield, throughput, money. The cost grid in
[manufacturing_results/cost_scenarios.csv](../manufacturing_results/cost_scenarios.csv) shows the
conditions under which inspection pays off; it is an assumption model.

## 5. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Continuous drift | 58–82% of sensors exceed PSI 0.25 in every 100-unit batch vs the initial window; random-split noise floor ≤ 7.5% | Periodic retraining; drift warnings shown relative to a rolling baseline, not the original training window |
| Drift trigger degenerates | The predeclared 10% sensor-share trigger fired on every batch, so "drift-triggered" = "always retrain" | Calibrate the trigger on historical batch-to-batch variation, or trigger on score-distribution/performance shift |
| Fixed MSPC limits | 72% later false alarms with textbook limits | Recalibrate limits on a recent in-control window; robust scaling for heavy-tailed sensors (SPE share is dominated by 3 sensors) |
| Small failure counts | 17 assessment, 46 replay failures | Shadow mode long enough to accumulate ≥100 failures before claiming lift |
| Spurious "important sensors" | Low rank stability; methods disagree (cross-method Jaccard ≤ 0.05) | Present the watchlist with stability tiers; never as root cause |
| Model artifacts | joblib executes code on load | Only load bundles built by this pipeline; hashes recorded in provenance |

## 6. Go-live conditions

The model can be trialled only if all of these hold:

1. **Measurement timing.** All 590 sensors are available before the inspection decision for that
   batch. SECOM does not say when each reading is taken.
2. **Effective intervention.** Reviewing a flagged unit can actually prevent or contain a failure;
   otherwise a better ranking does not change outcomes.
3. **Labels arrive.** Pass/fail outcomes return within about one batch so periodic retraining can run.
4. **Shadow mode first.** Score live batches without acting, compare the queue with current
   practice, and require the lift over random review to hold on new failures.
5. **Schema and drift guardrails.** The API rejects missing/extra sensors, non-numeric values and
   fully empty rows; drift warnings and MSPC alarm rates are tracked per batch.
6. **Ownership.** A named process engineer reviews the alarm Pareto and the watchlist each week,
   and decides limit recalibration.

## 7. Decision

Proceed to a **shadow-mode pilot** with supervised ranking plus MSPC diagnostics and scheduled
retraining. Do not deploy MSPC fixed limits as the alarm rule on this process, and do not present
feature rankings as root causes.

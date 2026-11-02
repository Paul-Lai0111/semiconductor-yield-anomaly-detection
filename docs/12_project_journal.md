# 12 - Project Journal

## Why I kept the diagnostic story

I began with a practical puzzle: a high overall accuracy could coexist with poor
failure detection. The original repository followed a useful sequence of attempts
to understand that gap: remove sparse and constant columns, compare median and KNN
imputation, inspect PCA, try SMOTE, fit weighted XGBoost, and move the threshold.
I want to retain those questions because they explain why the current comparison
exists.

The original narrative was more confident than its evaluation justified. It
treated an operating-point result as proof of a geometric mechanism and presented
a test-selected threshold as a final result. The revision keeps the attempts but
makes the evidence boundary explicit. This is a record of decisions visible in
the repository, not an invented account of dates, effort, or factory outcomes.

## What the existing code showed

The original tracked `main.py` was a single sequential script. It fetched remote
files, filtered features before splitting, made one stratified split with seed
42, and compared several transformed Random Forest inputs. It then fit weighted
XGBoost, searched the test precision-recall curve for a maximum-F1 threshold, and
reported performance on that same test set.

Several details changed how I interpret the old results:

- The sensor mask was learned before splitting, so evaluation feature
  distributions influenced preprocessing even without labels.
- The median branch fitted separate scalers on training and test data, making
  their coordinate systems inconsistent.
- KNN imputation preceded scaling, so raw numerical units determined neighbors.
- PCA classification used a fixed component count taken from an earlier variance
  diagnostic, rather than learning a 90% rule within each training fold.
- There was no shared-fold development comparison, and test labels chose the
  reported XGBoost threshold.
- The PR figure used trapezoidal integration; its area should not be relabeled as
  scikit-learn AP.

The earlier images and numeric summaries belong to that exploratory setup. I do
not use them as evidence for the revised protocol or assume that correcting the
setup must improve the scores.

## What I learned about the original explanations

The useful distinction is between a result and its possible cause. Zero failure
F1 says there were no true positives at that threshold. It does not prove zero
ranking signal. PCA can remove predictive directions, but its poor thresholded
score alone does not demonstrate that this happened. SMOTE can produce unhelpful
interpolations, but feature count alone does not prove its neighbors were invalid.

I also narrowed the engineering claims. Missing measurements do not diagnose dead
sensors. Tree models generally do not need scaling for their splits. Positive
weights do not calibrate probabilities, F1 does not measure factory cost, and
gain does not identify a physical cause. These corrections make the project more
useful because the remaining questions can actually be checked.

## What changed when I ran the revised experiment

SMOTE is no longer a discarded attempt. RF/KNN/SMOTE has the highest mean
development-fold AP under the fixed selection rule, with substantial fold
variation. The plain KNN/RF comparison is close enough that I do not interpret
the ordering as proof of general superiority.

The holdout also gives an instructive disagreement: Median/RF has a better frozen
F1 than the development-selected pipeline. I leave the selected candidate alone.
Changing it after reading this table would violate the rule I set out to test.

Weighted XGBoost does not win this comparison. Its separate chronological check
also exposes weak transfer at the earlier-selected operating point. That result
stays visible rather than being repaired by choosing a threshold on later labels.
The exact measurements, intervals, learned feature counts, and importance ranks
are in the [generated evidence](00_navigation.md#code-and-generated-evidence).

Both raw-file hashes are pinned and checked on cached loads. A regression case
rejects corrupted cached data. Other regression cases cover validation isolation,
training-only SMOTE, schema preservation, F1 ties affected by floating-point
rounding, frozen-before-holdout orchestration, and isolated output destinations.

The completed run leaves limits: few failures, dependence that
cannot be checked without lot/tool identifiers, uncertain future prevalence,
unknown alert costs, and previous exposure to this public dataset. The temporal
check asks a narrower transfer question; it does not remove those limits.

## What the supplementary checks settled

I checked four existing candidates on three new shared five-fold development
partitions, using seeds **17, 43, and 89** and fixed hyperparameters. These
**60 additional CV fits** change fold assignment and estimator randomness
together. They reuse the original **1,253 development rows and 83 failures**;
they do not reopen the original holdout.

SMOTE minus plain RF/KNN mean fold AP is **+0.0183, +0.0447, and +0.0433** in
the new repeats. The consistent positive difference changes how I describe the
old SMOTE attempt: it is an observed advantage for this pipeline over its plain
RF/KNN reference. XGBoost leads at seed 17 and SMOTE at 43 and 89, among the four
checked candidates. The leader is sensitive to the repeat, and reused data do not
provide independent significance evidence or a replacement winner.

I also ranked the **original OOF scores** inside each original validation fold
under 5%, 10%, and 20% review budgets. The selected pipeline uses **60, 125, and
250 reviews**, catching **15, 20, and 39 of 83 failures**, with **45, 105, and
211 passing observations reviewed**. At 125 reviews, plain RF/KNN catches **29**
versus SMOTE's **20**. This made the distinction between AP and capacity concrete:
an AP-selected candidate need not catch the most failures at a particular review
limit. The blocks are random folds, not factory inspection batches. The
[supplementary chapter](13_development_diagnostics.md) records the rules and raw
evidence; no original selection, threshold, or holdout result is replaced.

## Concrete design choices

| Choice | Reason and consequence |
| --- | --- |
| Seed 2026, stratified 20% holdout | Fix the main within-run evaluation boundary before fitting or selecting |
| Five identical development folds for seven candidates | Make comparisons use the same validation observations |
| Dummy and balanced logistic baselines | Check whether added complexity improves on simple alternatives |
| Training-only filter, scaler, imputer, PCA, and sampler | Keep evaluation distributions out of learned preprocessing |
| Scaling before both imputers | Use standardized KNN distances and a consistent median comparison |
| Learned PCA90 dimension | Test a variance-retention rule rather than inherit an exploratory count |
| Same 300-tree, minimum-leaf-2 Random Forest settings | Keep the RF preprocessing comparisons controlled |
| Fixed regularized XGBoost with per-fit class ratio | Avoid holdout tuning and prevent global prevalence from setting training weights |
| Mean fold AP selection, candidate-order exact ties | Choose the ranking rule without switching to whichever metric looks best |
| Pooled OOF maximum-F1 threshold, highest-threshold exact ties | Define one reproducible operating-point rule while acknowledging its optimism |
| Freeze before evaluating every holdout candidate | Show the full comparison without retrofitting the winner |
| Development-only, fixed-XGBoost temporal check | Examine forward transfer without opening the main holdout for another selection |
| Generated reports plus pipeline/threshold/schema/config artifact | Keep reported decisions connected to predictions and a usable fitted object |

## Changelog from the original repository

| Area | Original approach | Revised contract |
| --- | --- | --- |
| Entry point | Sequential all-in-one script | Simple `main.py` CLI with explicit configuration |
| Source handling | Remote reads without pinned source verification | Cached raw files, expected SHA-256 validation, and provenance manifest |
| Development boundary | One split; global sensor filtering | Shared stratified folds; every learned preprocessing step inside the fit |
| Imputation geometry | KNN before scaling; median train/test scalers fitted separately | Training scaler before either imputer, reused for validation and prediction |
| Baselines | Mostly transformed Random Forest comparisons | Ordered seven-candidate set including dummy and logistic references |
| PCA | Historical fixed component count | Per-fit 90% variance rule |
| Model choice | XGBoost presented as final by narrative | Mean fold AP decides; exact ties use candidate order |
| Threshold | Maximum F1 on the same test labels used for reporting | Development OOF maximum F1; highest threshold on exact ties; frozen before holdout |
| Holdout comparison | Test data also used for threshold tuning | Post-selection evaluation of all candidates without changing the selected winner |
| Time | Label-file timestamps unused | Separate development-only forward sensitivity check of fixed XGBoost |
| Evidence | Console reports and three explanatory figures | Structured metrics, split/prediction exports, frozen protocol, generated report/figures, and selected-model payload |
| Claims | Geometric causes and engineering benefits inferred from sparse evidence | Historical exploration distinguished from the frozen run; no causal or factory-outcome claims |
| Development diagnostics | Single comparison used to support broad method claims | Three descriptive repeat comparisons and three hypothetical review budgets, separate from the frozen study |
| Inference | Predictions embedded in the experiment script | Saved-pipeline batch command with schema validation, frozen-rule alerts, and hash metadata |
| Closing narrative | A proposed maintenance direction from anonymous sensors | Measured findings, quantified outputs, chart takeaways, and explicit interpretation limits |

Existing chapter filenames and their diagnostic progression are preserved. The
new protocol and this journal add the evaluation contract and a record of the
changes. The generated predictions and report carry the numerical evidence.

## What I completed

The original comparison contains **7 pipelines x 5 shared folds = 35 primary
CV fits**. The supplement adds **4 x 3 x 5 = 60 CV fits**, and the capacity
analysis adds **21 aggregate candidate/budget comparisons** without another fit.
These fit counts exclude full-development refits and the separate chronological
check. The learned full-development filter retains **446 of 590 sensors**;
the actual fold masks are learned separately.

I completed **8 original figures, 1 development diagnostic, and 1 findings
summary**, with explicit model names, data partitions, and takeaways. The
[executed notebook](../main.ipynb) preserves **16 executed code cells** that
audit the original saved predictions, metrics, and provenance. Closing Markdown
connects that audit to the supplementary findings. The batch interface uses the
saved pipeline and frozen threshold to export scores and alerts with metadata;
it demonstrates reproducible inference without measuring a factory deployment.

My main measured result is the same-model threshold comparison: **1 to 10 of 21
holdout failures caught**, with **1 to 52 reviews**, when moving from 0.5 to the
development-frozen threshold **0.242222**. That is **9 more catches for 51 more
reviews**, leaving **42 false alarms and 11 misses**. F1 changes from **0.0909
to 0.2740** and AP remains **0.2727**. The weak fixed-XGBoost chronological
result also remains visible: **AP 0.0967 and zero recall** at its earlier-selected
threshold, rather than a revised rule chosen on later outcomes.

The [findings and impact case study](14_findings_and_impact.md) closes the
investigation with those results and their limitations. My contribution is a
controlled, reconstructable study and a reusable scoring interface; yield gains,
prevented losses, and physical sensor causes were not measured.

# 07 - Decision Threshold and Operating Point

## Ranking and alerts answer different questions

A candidate produces a failure score. The decision rule is:

```text
alert = 1 if score >= threshold, otherwise 0
```

Changing the threshold changes alerts, not the underlying ranking or AP. Lowering
it can recover failures below 0.5, but also adds false positives. A score below
0.5 does not establish that the model failed to rank observations, and rare
failures are not mathematically prevented from receiving high scores.

The value 0.5 is a useful common diagnostic point. Its probabilistic interpretation
requires calibrated probabilities and a suitable decision objective. Class-weighted
or oversampled models do not automatically satisfy those conditions.

## The historical evaluation problem

The original script used test scores and test labels to find the threshold with
the largest F1, then reported F1 on those same observations. That is threshold
selection on the evaluation set. The optimized score is historical exploratory
evidence, not an independent test result, regardless of whether the threshold
looks numerically plausible.

I do not reuse that threshold in the revised protocol. A new split improves the
evaluation procedure but does not turn a previously explored public dataset into
unseen evidence.

## The development-only rule

Each candidate receives one OOF score per development row from a model fitted
without that row. Candidate selection uses mean validation-fold AP. Thresholds
are then selected from each candidate's pooled development OOF scores, using:

```text
F1(t) = 2 * TP(t) / (2 * TP(t) + FP(t) + FN(t))
t_star = threshold with maximum pooled OOF F1
exact F1 ties -> choose the highest threshold
```

The implementation in [evaluation.py](../secom/evaluation.py) considers distinct
observed scores in descending order, with tied scores moving together. Integer
confusion-count cross-products compare F1 exactly, so floating-point rounding
cannot reverse the tie rule. It considers the same observed operating points as
the PR curve; the terminal PR endpoint has no threshold. Predictions use `>=`,
including scores exactly equal to the selected threshold.

The highest-threshold tie rule is deterministic. Among exactly tied maxima it
chooses the option issuing no more alerts than lower tied thresholds. It does not
add a general precision constraint, capacity limit, or uncertainty tolerance to
the objective.

## What is frozen

Before holdout evaluation, the run records the selected candidate, all development
thresholds, and the configuration in `results/protocol.json`. Every candidate is
refitted on all development rows, then evaluated on the same holdout with its
already selected threshold and at the common 0.5 diagnostic point. Reporting all
seven candidates after selection makes the comparison inspectable; it does not
authorize choosing a different winner from the holdout table.

The selected-model artifact stores the selected pipeline and its frozen threshold.
The full-development fit can produce a score distribution different from the
fold models. That is one reason to evaluate the frozen rule on the holdout rather
than silently adjusting it there.

## Optimism and uncertainty

OOF predictions avoid fitting an observation's own fold model on that observation.
They do not remove all selection bias. The candidate was chosen using these
development data, and the maximum F1 threshold was chosen using the same OOF
labels on which its F1 is summarized. The maximum OOF F1 is optimistic. Pooled OOF
AP is also not an independent estimate of the selected procedure. A fully nested
assessment would be a different, more expensive experiment.

The holdout provides a later evaluation boundary within this run. Its small failure
support still limits precision of the estimate. The report includes failure
support, confusion counts, and a Wilson interval for recall. Its class-stratified,
fixed-score bootstrap AP interval conditions on the observed prevalence and does
not include variation from retraining or model selection.

## A policy still needs an operating objective

I use F1 because costs and inspection capacity are unspecified. It is not a claim
that the chosen alarm burden is acceptable in a factory. Given real costs, an
alternative objective could minimize `C_FN * FN + C_FP * FP`; given a review budget,
it could constrain alarm count. Either would need to be defined and selected on
development data before another evaluation.

Exact thresholds, alarm rates, and performance are recorded in the
[generated results](00_navigation.md#code-and-generated-evidence). The next chapter
explains how weighted XGBoost changes score fitting, a separate decision from
threshold selection.

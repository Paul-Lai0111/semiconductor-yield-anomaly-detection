# 01 - Problem Framing and Metrics

## The question behind the project

I want to know whether the supplied process measurements can rank failures above
passes, and what happens when I convert that ranking into alerts. This is a
supervised study of the UCI SECOM labels. The dataset does not supply inspection
capacity, a validated factory cost matrix, or evidence that acting on an alert
would improve yield.

Missing a failure and inspecting a passing observation plausibly have different
consequences. Their relative costs are unknown here. I therefore report both
failure detection and alarm burden instead of assigning a financial benefit.

## Dataset and target

| Property | Value |
| --- | --- |
| Observations | 1,567 |
| Anonymized sensor columns | 590 |
| Passes | 1,463 |
| Failures | 104 |
| Failure prevalence | 104 / 1,567, approximately 6.6% |
| Target mapping | Raw -1 becomes pass 0; raw +1 becomes failure 1 |
| Timestamps | Separate metadata, excluded from predictors |

Source: [UCI SECOM](https://doi.org/10.24432/C54305). The
[data loader](../secom/data.py) validates the expected shape and labels. The
[protocol](11_experiment_protocol.md) describes cached source validation and row
assignments.

An always-pass prediction has approximately 93.4% accuracy on the full dataset
while detecting no failures. That arithmetic is a baseline property of the
labels, not a new model result. Accuracy remains useful context, but it cannot
answer the failure-detection question by itself.

## Ranking before a decision threshold

The primary selection metric is **average precision (AP)**, computed with
`average_precision_score`. Using the points on the precision-recall curve:

```text
AP = sum over recall increments of (recall increment * precision)
```

AP summarizes precision as recall increases. It is not the trapezoidal area
computed by applying `auc` to precision and recall coordinates. I use the name
"AP" throughout so those definitions cannot silently change between comparisons.
For a constant-score prevalence dummy, AP equals the evaluated partition's failure
prevalence. That is a useful reference, not a guarantee that a small lift is stable.

The candidate is selected by **mean AP across the same five development validation
folds**. Pooled OOF AP is a separate summary: merging scores from several fitted
models can give a different answer from averaging their fold AP values. Neither
the winning development AP nor pooled OOF AP is an independent estimate after
using these data for selection.

ROC-AUC is also reported. It measures positive-negative ordering and remains a
valid ranking statistic under imbalance. Its false-positive rate uses all actual
passes as the denominator, so it can look favorable while the number of false
alarms is impractical. Precision, alarm counts, and prevalence make that consequence
visible; ROC-AUC need not be dismissed as mathematically invalid.

## Decisions and workload

With failure as the positive class, `TP`, `FP`, `FN`, and `TN` give:

| Measure | Definition | Interpretation |
| --- | --- | --- |
| Precision | TP / (TP + FP) | Fraction of alerts that are failures |
| Recall | TP / (TP + FN) | Fraction of failures alerted |
| F1 | 2 TP / (2 TP + FP + FN) | A precision-recall compromise |
| Alarm count | TP + FP | Number of observations sent for review |
| Alarm rate | (TP + FP) / N | Fraction of observations sent for review |
| Balanced accuracy | (failure recall + pass recall) / 2 | Equal emphasis on the two class recalls |

The development threshold maximizes pooled OOF failure-class F1. F1 is a transparent
working rule when costs are unspecified; it does not encode a factory loss
function, an explicit preference for recall, or an inspection-capacity constraint.
It also ignores true negatives. A different real operating objective would require
a different, prospectively defined threshold rule.

## What a zero F1 does and does not show

The original exploration reported zero failure F1 at some default-threshold
settings. Zero F1 means no true positives at that operating point. It does not
alone prove that every prediction was pass, that ranking was random, or that PCA
removed the failure signal. Confusion counts identify whether alerts were issued;
AP and the precision-recall curve assess ranking independently of that threshold.

The revised report includes both the default 0.5 operating point and development
thresholds, together with confusion counts. The
[generated results index](00_navigation.md#code-and-generated-evidence) is the place
to read them. The next chapter addresses which inputs may enter those comparisons.

# 06 - Revisiting PCA and SMOTE

## What the earlier attempts taught me

The original investigation tried PCA and SMOTE and reported poor failure-class
metrics at the evaluated operating points. I initially treated those outcomes as
evidence that changing the representation or training balance could not help.
That conclusion was too strong: one thresholded report cannot distinguish weak
ranking from a poor threshold, and it cannot establish a geometric cause.

The revised experiment keeps both candidates. Their historical results explain
why I want a controlled comparison; they do not determine the winner.

## What SMOTE adds

SMOTE constructs a synthetic minority observation between a minority training row
and one of its minority neighbors:

```text
x_new = x_i + u * (x_neighbor - x_i), where 0 <= u <= 1
```

The implementation uses `SMOTE(k_neighbors=5, random_state=2026)` at the default
seed, after fold-local sensor filtering, scaling, and KNN imputation. It resamples
the training partition only. Validation and holdout rows remain real observations
with their original prevalence.

Synthetic rows increase the number of training examples presented to the model,
but do not add independently measured failures. They rely on the assumption that
interpolating between chosen minority neighbors gives useful training examples.
That assumption may be poor when failures arise from multiple processes or when
distances are noisy. Neither condition is proven simply by counting dimensions.

## Controlled comparisons

| Reference | Alternative | Change under investigation |
| --- | --- | --- |
| `rf_knn` | `rf_knn_pca` | Add fold-fitted PCA retaining 90% training variance |
| `rf_knn` | `rf_knn_smote` | Add training-only minority interpolation |

Every Random Forest uses 300 trees, `min_samples_leaf=2`,
`max_features="sqrt"`, and `class_weight="balanced"`, with the same seed and
shared folds. Other settings remain at library defaults recorded by the
environment. There is no separate, more favorable forest configuration for a
particular preprocessing choice.

One detail matters for interpretation: `class_weight="balanced"` is calculated
from the labels actually passed to the forest. With the default binary SMOTE
policy, resampled classes are equal, so their balanced weights are equal too.
The SMOTE branch therefore does not additionally apply the original class ratio
as a second weight. It compares resampling with the original balanced-weight
training policy, rather than isolating interpolation while holding every
effective sample weight fixed.

## Leakage boundaries

SMOTE must run inside the training pipeline. Oversampling all development rows
before cross-validation would let validation observations influence synthetic
training data. Imputing or fitting PCA before forming folds creates related
information leaks even without using validation labels directly.

At prediction time, the fitted pipeline filters, scales, imputes, and predicts.
It does not synthesize new validation or holdout observations. This preserves the
meaning of precision and alarm rate at the real evaluation prevalence.

## What a poor result would mean

Lower AP across shared folds would count against the candidate under these
settings. It would not prove that all oversampling or dimensionality reduction
methods fail on semiconductor data. Lower F1 at 0.5 with similar AP would instead
point toward a different operating-point behavior. A large apparent gain at the
best OOF threshold still needs the frozen holdout evaluation because that
threshold was selected on those OOF labels.

The current run has not supplied new numbers. I will compare the generated
development and fold results before describing either candidate as helpful or
unhelpful. The next step in the diagnostic arc is to make threshold selection
explicit.

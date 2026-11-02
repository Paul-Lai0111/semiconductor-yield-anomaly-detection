# 05 - PCA as a Structural Probe

## Why I tried compression

After removing unusable training columns and filling the remaining gaps, there
are still many possible sensor relationships and relatively few failures. I tried
PCA to ask whether a smaller representation could retain much of the observed
variation and help a classifier generalize.

There are two distinct questions: how compressible is the feature matrix, and
does that compression help failure prediction? A cumulative explained-variance
curve addresses the first. Validation predictions are needed for the second.

## What PCA optimizes

PCA centers the completed training matrix and finds orthogonal directions of
decreasing variance. Retaining leading components minimizes squared reconstruction
error for a given dimension. It does not use labels.

Consequently, retaining 90% of variance means retaining 90% of the measured
variation in this transformed training matrix. It does not mean retaining 90% of
failure information. A predictive direction can have low variance, and a
high-variance direction can be unrelated to the target.

Imbalance can affect the pooled covariance because passes are more numerous, but
sample counts alone do not establish which directions dominate. Within-class
spread and differences between class means also matter. I cannot infer that PCA
erased a minority signal just from the class ratio.

## The revised candidate

`rf_knn_pca` uses:

```text
SensorFilter -> StandardScaler -> KNNImputer(k=5)
             -> PCA(n_components=0.90, svd_solver="full")
             -> balanced Random Forest
```

All transformations are fitted on the current fold's training rows. The component
count is learned from that training matrix to meet the 90% variance rule. The
count can differ between folds and again when refitting on all development rows.
There is no hard-coded component count carried forward from the old exploration.

`rf_knn` is the reference candidate. It shares the imputer, folds, and Random
Forest settings, so the comparison asks what adding this PCA transformation does
under the fixed protocol. The full candidate configuration is in
[chapter 11](11_experiment_protocol.md).

## Why the classifier can change even if variance is retained

PCA both rotates the coordinates and removes some directions. Random Forest uses
axis-aligned splits, so it is not invariant to rotation: a simple split on an
original sensor can become a combination of several component coordinates. A
performance difference therefore does not isolate information loss alone. It can
also reflect how the classifier expresses boundaries in the new coordinates.

Conversely, compression might reduce noisy variation and help. I leave that as an
empirical possibility rather than deciding in advance that PCA must fail.

## Reading the evidence

The historical PCA attempt was described as a failure after a poor thresholded
classification report. That is an exploratory observation, not proof of the
mechanism. Zero F1 at 0.5 is compatible with useful ordering below that threshold.

For the revised comparison, inspect mean fold AP, fold variation, pooled OOF
ranking, and both default and development-selected operating points. Component
counts describe the fitted representation; they do not establish predictive
quality. Exact values are recorded in the
[generated results](00_navigation.md#code-and-generated-evidence).

PCA stays in the candidate set as a testable alternative. The next chapter gives
SMOTE the same treatment instead of treating earlier unsuccessful attempts as
settled explanations.

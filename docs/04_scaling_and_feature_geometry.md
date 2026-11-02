# 04 - Scaling and Feature Geometry

## Why the order matters

The original KNN branch imputed raw measurements and scaled afterward. That made
neighbor selection depend on the original numerical units. Scaling after neighbor
selection cannot repair a neighbor choice already dominated by a large-scale
sensor. This is a concrete implementation issue, separate from whether KNN later
outperforms the median baseline.

The revised non-dummy pipelines all filter sensors, scale observed values, and
then impute. The median branch uses the same order, so the imputer comparison
does not also change the location of scaling.

## What is learned from training data

For retained sensor `j`, `StandardScaler` applies:

```text
z_j = (x_j - training_mean_j) / training_standard_deviation_j
```

Its fit ignores NaNs and its transform preserves them. The subsequent imputer
fills gaps in standardized coordinates. Means and standard deviations are learned
on the current training rows only and reused for validation rows. After imputation,
the completed columns need not have exactly zero mean and unit variance; the
scaler was fitted to the observed entries, not to the completed matrix.

The historical median branch fitted a separate scaler on test data. That used a
different coordinate system at evaluation time and allowed test distribution
information into preprocessing. A single fitted pipeline now carries training
statistics through to prediction.

## Which methods care about scale?

| Method | Why scale matters here |
| --- | --- |
| KNN imputation | Distances depend on the relative numerical spread of sensor coordinates |
| PCA | Variance directions depend on feature units; standardization changes the covariance structure being summarized |
| SMOTE | Neighbors and interpolation are constructed in the completed standardized space |
| Logistic regression | Feature scaling affects optimization and the meaning of its regularization penalty |
| Random Forest and tree XGBoost | Splits generally do not require standardization; these pipelines include it for their upstream steps and consistent comparisons |

Tree splits depend on value ordering and candidate cut points. Positive affine
rescaling generally preserves those partitions. Numerical precision and histogram
implementation details can produce small differences, but unscaled inputs do not
inherently give tree boosting an elongated optimization surface requiring scaling.
That explanation belongs to different kinds of optimization.

## What standardization leaves unresolved

Equalizing numerical spread does not make all sensors equally informative. It can
also give a noisy low-variance sensor more influence in a distance calculation.
Outliers still affect means and standard deviations, and correlations remain.
Standardization does not establish normality, calibrated scores, class separation,
or a physically meaningful neighborhood.

High dimensionality makes those assumptions worth checking. It is a reason to
compare KNN, median, and PCA under shared folds, rather than a sufficient
explanation for any observed failure.

## Decision retained in the protocol

I use `StandardScaler` before both imputers and keep its fit local to each training
partition. I am not claiming a measured scaling benefit: there is no unscaled
ablation in the seven-candidate set. The change makes the intended KNN geometry
explicit and fixes the earlier train/test inconsistency. The
[PCA chapter](05_pca_as_structural_probe.md) examines what variance preservation
can and cannot establish after that preprocessing.

# 02 - Data Pathology and Feature Pruning

## What the input can tell me

SECOM combines many anonymized columns with missing entries and columns that can
be constant in a training sample. I need a consistent input policy before comparing
imputers and classifiers. I cannot infer a sensor's physical condition from a
missing-value count: absence could reflect logging, process routing, measurement
availability, or equipment problems. Those explanations require metadata that this
dataset does not provide.

The input keeps all 590 original sensor columns in their original order. The loader
names them `sensor_000` through `sensor_589`; these are identifiers, not process
descriptions. Labels and timestamps remain outside the predictor matrix.

## The implemented filter

For every non-dummy candidate, `SensorFilter` in
[modeling.py](../secom/modeling.py) learns a mask from the current training
partition. A sensor is retained only when both conditions hold:

```text
training missing fraction <= 0.50
number of distinct observed training values > 1
```

Exactly 50% missing is allowed. More than 50% missing is removed. A column with no
observed values or only one distinct observed value is removed too. The cutoff is
a fixed practical choice to limit dependence on imputation; it is not a statistical
law establishing that a sensor becomes unusable at 50%.

A column with one observed value has no observed value variation available for
learning a split. Its missingness pattern might still contain information, but
this experiment does not add missingness indicators. Removing the column therefore
makes a simplifying assumption rather than proving it has no possible predictive
use. Constant coordinates also do not automatically distort Euclidean distances:
their pairwise differences are zero. Removing them reduces unnecessary processing.

## Where fitting is allowed

| Fit stage | Data allowed to determine the mask | Data that only receives the mask |
| --- | --- | --- |
| Development fold | That fold's training rows | That fold's validation rows |
| Full-development refit | All development rows | Holdout rows |
| Temporal training window | Earlier training rows only | The corresponding later validation/check rows |

Applying a filter learned from all rows before splitting would let validation and
holdout feature distributions influence the model. No labels are needed for this
form of leakage. Keeping the filter inside the pipeline enforces the same boundary
as scaling and imputation.

Retained columns can differ across folds. The final refit can also retain a
different set from an individual fold. The old documentation's fixed cleaned
dimension is therefore not an invariant of the revised experiment. Exact counts
must come from the current run, not from a full-dataset exploratory summary.

## Missingness is also an assumption

Median and KNN imputation both replace unavailable measurements with estimates.
Neither identifies why a value is missing. Missingness related to failures or
process regimes could affect performance, and aggressive filtering could discard
useful but rarely observed signals. A later missingness-indicator comparison would
be reasonable, but it would be a new experiment rather than an explanation already
verified by this one.

High feature count relative to the number of failures raises concerns about noisy
distances and unstable fits. It does not by itself prove class overlap, identify
the failure mechanism, or establish that SMOTE must fail. Those claims need
comparative evidence.

## Prediction-time contract

The fitted filter expects the original 590 sensor columns in order. With a pandas
DataFrame, it also checks column names and their order. New missing entries in a
retained sensor are handled by the fitted imputer; validation or prediction data
does not cause the sensor mask to be relearned. Even if an observed prediction
column is entirely missing, the training mask remains fixed.

The saved selected-model payload includes the pipeline and original sensor-column
order so prediction code can enforce this contract. The
[imputation chapter](03_imputation_strategy_diagnostics.md) explains what happens
after filtering.

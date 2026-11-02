# 08 - Weighted XGBoost as a Candidate

## Why it remains in the study

The original project ended with weighted XGBoost and a threshold chosen on the
same test predictions. That exploration made boosting worth revisiting, but did
not establish an independently evaluated winner. I keep the existing chapter
filename for links; the revised protocol allows any of the seven candidates to
be selected by development mean fold AP.

XGBoost is a reasonable candidate for nonlinear relationships in tabular sensor
data. Trees can express interactions without requiring a distance-based
classifier. Whether that helps here must be measured against the dummy, logistic,
and Random Forest alternatives.

## What the positive weight changes

For a binary logistic objective with failure score `p`, the weighted data loss is:

```text
L = -sum_i [w * y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
w = number of training passes / number of training failures
```

Tree regularization is applied in addition to this data loss. Positive-class
weighting changes the gradient and Hessian contributions from failure rows,
which affects split selection and leaf values. It does not directly optimize F1
or impose a particular false-negative cost in money.

`fit_pipeline` in [modeling.py](../secom/modeling.py) recomputes
`scale_pos_weight` from the labels of the current training partition. A fold uses
its training labels; the full-development refit uses development labels; a
temporal window uses only its earlier training labels. No holdout prevalence is
used to set this weight.

Weighting can change both ranking and score scale. It guarantees neither better
AP nor calibrated probabilities. A weighted loss generally targets a different
effective class distribution, so I treat `predict_proba` output as a score until
calibration has been separately assessed.

## Fixed configuration

| Parameter | Default study value |
| --- | --- |
| Candidate | `xgb_knn` |
| Preprocessing | Fold-local sensor filter, StandardScaler, KNN imputation with 5 neighbors |
| Trees | 300 |
| Maximum depth | 3 |
| Learning rate | 0.05 |
| Minimum child weight | 5 |
| L2 leaf regularization, `reg_lambda` | 10.0 |
| Row subsampling | 0.8 |
| Column subsampling per tree | 0.8 |
| Tree method | `hist` |
| Evaluation metric | `logloss` |
| Importance type | `gain` |
| Random seed | 2026 |
| Positive weight | Training passes / training failures, recomputed per fit |

These are fixed comparison settings, not the result of a new hyperparameter
search. There is no holdout-driven early stopping. The shared 300-tree setting
with Random Forest does not equalize model capacity or computation; the study
compares these specified pipelines rather than every possible tuned model family.

## Selection and saved model

Mean fold AP selects the candidate, with exact ties resolved by the listed
candidate order. Each candidate's OOF F1 threshold is selected on development data
and frozen before the holdout is evaluated. A favorable XGBoost holdout result
cannot replace a different development-selected candidate.

`artifacts/selected_model.joblib` contains the selected pipeline, threshold,
`sensor_columns`, and `config`. It is a reproducibility artifact from the study,
not evidence of production readiness. XGBoost scores and the actual selected
candidate are recorded in the
[generated report](00_navigation.md#code-and-generated-evidence).

## Temporal sensitivity is a separate question

The temporal check deliberately fixes `xgb_knn` rather than selecting a second
winner. Within development data, it uses earlier observations to choose a
threshold through three expanding time splits, refits on the earlier block, and
evaluates the later 20%. It asks whether this particular pipeline transfers
forward within the available history. It does not validate whichever candidate
wins the main comparison, and it is not independent of the project's exploratory
history. [Chapter 11](11_experiment_protocol.md#temporal-sensitivity-check) gives the
exact boundaries.

The remaining question is what the fitted model can tell me about its inputs.
[Chapter 09](09_model_interpretability_and_engineering_translation.md) separates
predictive importance from a physical explanation.

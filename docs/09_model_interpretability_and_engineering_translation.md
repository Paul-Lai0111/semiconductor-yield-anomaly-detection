# 09 - Model Interpretation and Engineering Limits

## What I want to learn from importance

After comparing predictions, I want to inspect which inputs the fitted model
uses. That can reveal dependence on a small set of columns, suggest checks for
measurement artifacts, and guide a more focused follow-up analysis. It cannot
identify a tool subsystem when sensor meanings are unavailable.

The earlier documentation turned a ranked sensor list into a maintenance roadmap.
That went beyond the evidence. The revised study treats importance as a model
diagnostic and exports exact sensor ranks from the current development-trained XGBoost fit.

## Read the importance definition first

For XGBoost configured with `importance_type="gain"`, importance describes the
average improvement in the training objective for splits using a feature. The
scikit-learn interface normalizes its reported importances. This is distinct
from total gain accumulated across all splits, and from an effect on held-out
performance.

A Random Forest's impurity importance or a logistic regression coefficient has a
different definition. Their numerical values are not interchangeable with gain.
The selected model can also be the dummy or a PCA pipeline, so a table must be
read with its model and representation in mind. PCA coordinates are combinations
of retained sensors; a component's tree importance is not directly an original
sensor's importance. A back-projection through loadings would be an additional
approximation, not a measured sensor effect.

The generated `results/feature_importance.csv` and corresponding figure are
descriptive outputs of XGBoost even when a different candidate is selected. Read
them with that model identity and frozen run context
in [the results index](00_navigation.md#code-and-generated-evidence). The old top
sensor list is not carried forward as a new finding.

## Sources of ambiguity

Correlated sensors can substitute for one another, splitting or concentrating
importance differently across fits. Imputed values also affect which splits are
available. A high-importance column might reflect process conditions, a logging
pattern, or a proxy for another variable. A low-importance column may be redundant
with a stronger correlated column rather than physically irrelevant.

Global gain does not give the direction of an association. It also does not
explain why a particular observation received an alert. Local attribution,
held-out permutation importance, and stability across resamples would answer
additional questions; they are not established merely by exporting a gain table.
Using the holdout to design a smaller feature set would consume it for another
selection decision and require a new evaluation boundary.

## A useful next investigation

With access to process metadata, I would first map anonymous columns to actual
measurements, check units and acquisition times, and verify that each value would
be available when an alert is intended. I would then inspect missingness and
distributions for candidate signals across process regimes, lots, and time.
Those checks could distinguish a repeatable association from a data artifact.

Only after that would I ask a process engineer whether a proposed mechanism is
plausible and what measurement or controlled intervention could test it. No such
mapping, intervention, sensor-specific limit, or maintenance outcome has been
demonstrated here. A high gain value does not justify replacing or recalibrating
a sensor.

## What remains outside this project

This offline public-data study does not establish real-time availability, stable
future prevalence, calibrated failure probabilities, a review capacity, or reduced
factory losses. The timestamps permit a limited sensitivity check, but no supplied
lot or tool grouping is used to establish independence between observations.
Related rows could appear across a random split.

The engineering value at this stage is a reproducible way to inspect ranking,
alarm burden, and model dependence on inputs. Claims about physical causes or
yield improvement would require evidence beyond this dataset and experiment.
The [review questions](10_questions_you_will_get_asked.md) collect the remaining
statistical and practical limits.

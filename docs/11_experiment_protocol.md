# 11 - Experiment Protocol

## Status and scope

This is the contract for the revised experiment. It specifies decisions before
the holdout evaluation rather than filling in a preferred outcome afterward.
Exact performance, the selected candidate, learned sensor masks, and learned PCA
dimensions are recorded in the [generated results](../results/RESULTS.md). The
earlier test-tuned results remain historical exploration and are not reused as
current evidence.

The study is supervised classification on UCI SECOM. It compares seven fixed
pipelines, selects one on development data, and reports a frozen holdout
evaluation. It does not claim independence from the project's exploratory history.

## Source data and provenance

Source: [UCI SECOM](https://doi.org/10.24432/C54305), downloaded from the
[published archive](https://archive.ics.uci.edu/static/public/179/secom.zip).
The raw files are `secom.data` and `secom_labels.data`, cached under `data/raw`
by default. The source is attributed as CC BY 4.0 in the data manifest.

The data contract is 1,567 rows by 590 sensor columns, 1,463 passes, and 104
failures. Raw labels -1 and +1 map to 0 and 1. Timestamps are parsed separately
from the label file and never enter the predictor matrix. Original row identity
and sensor order must survive splitting and prediction export.

Source integrity requires SHA-256 validation against pinned expected file hashes,
including when reusing cached files. Merely computing and recording the current
files' hashes establishes provenance but does not validate them against a trusted
reference. Before accepting a run, verify that the expected-hash mapping in
`secom/data.py` is populated and both source files match it. Shape and label checks
are additional validation, not substitutes for checksums.

## Main split and shared folds

1. Load and validate the raw data without learning preprocessing statistics.
2. Reserve a stratified 20% holdout using seed 2026. The remaining 80% is development
   data. Record actual row assignments and class support in the split manifest.
3. Within development, use `StratifiedKFold(n_splits=5, shuffle=True,
   random_state=2026)`. Materialize the folds once and reuse them for all candidates.
4. Fit every candidate independently on each fold's training rows and score only
   that fold's validation rows. Each development row receives one OOF score per
   candidate.

The holdout is excluded from candidate selection, preprocessing fits, class-weight
calculation, threshold selection, and the temporal check. A stratified random
split approximates transfer among observations from the available population; it
does not enforce separation by time, tool, or lot.

## Candidate order and settings

Exact ties in mean fold AP resolve to the first candidate in this order:

| Order | Identifier | Pipeline after input |
| --- | --- | --- |
| 1 | `dummy` | `DummyClassifier(strategy="prior")`, no learned sensor preprocessing |
| 2 | `logistic_median` | Filter, scale, median imputation, balanced logistic regression |
| 3 | `rf_median` | Filter, scale, median imputation, balanced Random Forest |
| 4 | `rf_knn` | Filter, scale, KNN imputation, balanced Random Forest |
| 5 | `rf_knn_pca` | Filter, scale, KNN imputation, PCA90, balanced Random Forest |
| 6 | `rf_knn_smote` | Filter, scale, KNN imputation, SMOTE, Random Forest |
| 7 | `xgb_knn` | Filter, scale, KNN imputation, weighted XGBoost |

All learned steps belong to the current training partition:

- Filter: retain sensors with missing fraction at most 0.50 and more than one
  distinct observed value. Learn the mask again for each fit.
- Scale: `StandardScaler` fits observed training entries, preserves NaNs, and
  precedes both median and KNN imputation.
- Impute: `SimpleImputer(strategy="median")` or `KNNImputer(n_neighbors=5)` with
  default NaN-aware distance and uniform weighting. Use training statistics/donors
  for validation and holdout transformations.
- PCA90: `PCA(n_components=0.90, svd_solver="full")`. Learn the component count on
  the current completed training matrix; do not prescribe a historical count.
- SMOTE: `SMOTE(k_neighbors=5, random_state=2026)` with default sampling policy,
  applied only to training rows after imputation. Never resample evaluation rows.

For main CV fold `k` (zero based), estimator and SMOTE seeds are `2026 + k`.
Full-development refits use seed 2026. The temporal check uses the same fold-offset
rule. Settings below describe the default run; exact parameters are saved in the
frozen protocol and locked package versions.

Logistic regression uses `C=1.0`, `class_weight="balanced"`, `max_iter=3000`, and
seed 2026. Every Random Forest uses 300 trees, `min_samples_leaf=2`,
`max_features="sqrt"`, `class_weight="balanced"`, and seed 2026. Balanced weights
in the SMOTE branch are computed on the resampled labels and are equal when
resampling has equalized class counts.

XGBoost uses 300 trees, depth 3, learning rate 0.05, `min_child_weight=5`,
`reg_lambda=10.0`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`,
`eval_metric="logloss"`, `importance_type="gain"`, and seed 2026. For every fit,
`scale_pos_weight` is training pass count divided by training failure count.
There is no holdout-based early stopping or hyperparameter search. Unspecified
estimator settings follow the installed versions captured by the environment.

## Selection and freeze

1. Compute validation AP for each candidate on each of the five shared folds.
   Select the candidate with the largest arithmetic mean fold AP, using the
   candidate order above for exact ties.
2. For each candidate, pool its development OOF scores and select the threshold
   maximizing failure-class F1. On exact F1 ties, choose the highest threshold.
   This is separate from selecting the candidate by mean fold AP.
3. Write `results/protocol.json` with the configuration, selection, and development
   thresholds before holdout evaluation. This is a within-run freeze, not a claim
   of preregistration before the project's original exploration.
4. Refit each of the seven pipelines on all development data. Evaluate each on
   the holdout using its frozen threshold and the common 0.5 diagnostic threshold.
5. Retain the development-selected winner even if another candidate has a better
   holdout score. Save the selected full-development pipeline and its threshold.

Threshold selection considers distinct observed scores and compares F1 using
integer confusion counts. Exact ties retain the higher threshold without a
floating-point tolerance. The terminal PR point has no matching threshold and is
not a candidate. Scores equal to the threshold count as positive. AP is
`average_precision_score`, not trapezoidal PR integration.

Development fold AP variation is descriptive. The winning development score and
pooled OOF AP are not independent estimates after selection. The maximum pooled
OOF F1 is optimistic because those same OOF labels chose its threshold. The
holdout is the later evaluation boundary for the frozen procedure within this run.

## Temporal sensitivity check

This check uses only development data and fixes `xgb_knn` in advance:

1. Sort development observations by timestamp. Reserve the last 20% as the later
   temporal check block; use the first 80% as the earlier block.
2. Apply `TimeSeriesSplit(n_splits=3)` within the earlier block. For each expanding
   training window, fit all preprocessing and the XGBoost weight from that window
   only, then predict its later validation window.
3. Pool only the forward-validation predictions to choose an F1 threshold, using
   the same highest-threshold exact-tie rule. The initial training prefix has no
   forward OOF prediction and is not backfilled with training predictions.
4. Refit `xgb_knn` on the entire earlier block and apply the fixed temporal
   threshold to the later block. Do not tune on later labels.

This is a time-ordered sensitivity analysis of a fixed candidate. It is not
stratified, not a competition among the seven candidates, and not a new independent
holdout. Development observations used here also participate in the main analysis.
Rows adjacent in the development timestamp order may have intervening source rows
reserved for the main holdout, so these are not necessarily contiguous full-source
time windows. The check does not remove exploratory-history or possible group
dependence. Any missing class support limits which metrics or fits are defined;
an unavailable check must be reported rather than replaced with a random split.

## Reporting and uncertainty

Report ranking metrics AP and ROC-AUC with failure prevalence. For operating
points, report precision, recall, F1, accuracy, balanced accuracy, confusion counts,
alarm count/rate, and class support. The default threshold is diagnostic; the
development-selected threshold is the frozen operating policy.

`evaluation.py` supplies a 95% Wilson interval for failure recall and a
class-stratified bootstrap AP interval, with 2,000 repetitions by default. The AP
bootstrap resamples fixed scores within each observed class and therefore
conditions on observed prevalence. It does not refit models or repeat selection.
Neither interval captures all uncertainty from the training sample, pipeline
choices, temporal drift, or earlier exploration. Fold variability is also not a
formal confidence interval because training folds overlap.

All output destinations are linked in
[the navigation index](00_navigation.md#code-and-generated-evidence): structured
metrics, development/holdout/fold tables, split membership, OOF and holdout
predictions, feature importance, frozen protocol, generated `RESULTS.md`, and
current-run `images/*.png`. Do not combine an old figure with a new metric table.

`artifacts/selected_model.joblib` carries `pipeline`, `threshold`, `sensor_columns`,
and `config`. It stores the full-development refit, not a model subsequently
trained on development plus holdout. Preserve the original sensor schema when
using it for prediction.

## Invocation and environment

[pyproject.toml](../pyproject.toml) requires Python 3.12 and declares the modeling
dependencies; `uv.lock` records the resolved environment. With that environment
already prepared, the default invocation from the repository root is:

```sh
uv run python main.py --overwrite
```

The simple CLI in [main.py](../main.py) exposes:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data/raw` | Raw source cache |
| `--output` | `results` | Results directory |
| `--seed` | `2026` | Random split and estimator seed |
| `--folds` | `5` | Number of shared stratified development folds |
| `--trees` | `300` | Tree count for both Random Forest and XGBoost |
| `--jobs` | `2` | Estimator parallelism |
| `--bootstrap` | `2000` | Fixed-score AP bootstrap repetitions |
| `--skip-temporal` | Not set | Omit the temporal check when supplied |
| `--overwrite` | Not set | Explicitly replace an existing nonempty run directory |

Changing seed, folds, trees, or omitting the temporal check changes the documented
default experiment and must be visible in the saved configuration and report.
Published results occupy the default directory, so reproducing them requires
`--overwrite`. A custom `--output` keeps metrics, figures, and model together in
that directory; it does not overwrite published assets. Skipping the temporal
check during an explicit replacement removes its previous generated predictions.

# 10 - Questions for Reviewing This Study

These are the questions I need to answer when interpreting the experiment. The
[generated results](00_navigation.md#code-and-generated-evidence) supply the
numerical answers.

## Is this anomaly detection or classification?

The repository name uses anomaly detection, but the implemented comparison is
supervised failure classification. The target is the supplied SECOM label. No
candidate learns a novelty detector from passes alone, and the evaluation does
not establish detection of previously unseen physical failure modes.

## Why include a dummy and logistic regression?

The prevalence dummy exposes whether the ranking beats a constant score and how
the threshold rule behaves without discrimination. Balanced logistic regression
is a simple supervised reference. If a more complicated pipeline adds little
development AP, complexity needs a better justification. Neither baseline is
included merely to make the tree models look successful.

## Why not use deep learning or a much larger search?

I chose a bounded set of tabular models to investigate preprocessing, imbalance,
and threshold selection on a small public dataset. That scope makes comparisons
inspectable. It does not prove that neural networks, undersampling, native-missing
XGBoost, or other models cannot work. More candidates would add selection
opportunities and require a suitable evaluation plan.

## Why select by AP but choose a threshold by F1?

AP compares rankings without committing to one cutoff. F1 supplies a fixed
development operating-point rule when costs are unknown. The candidate with the
best mean fold AP need not have the best maximum OOF F1. The protocol deliberately
does not switch objectives after seeing which candidate benefits.

## What exactly improved when I moved the threshold?

For the **same fitted RF/KNN/SMOTE model and holdout scores**, threshold **0.5**
catches **1 of 21** failures with **1 review and no false alarms**. The
development-frozen threshold **0.242222** catches **10 of 21** with **52 reviews,
42 false alarms, and 11 misses**. F1 changes from **0.0909 to 0.2740**, while
**AP stays 0.2727**. The measured gain is **9 additional catches for 51 additional
reviews**, not improved ranking or a quantified economic return.

The threshold was chosen from development OOF labels. These counts do not compare
the revised protocol with the historical test-tuned project; that would confound
multiple evaluation and implementation changes.
[Saved counts](../results/holdout_metrics.csv).

## Is AP the same as PR-AUC? Is ROC-AUC wrong here?

The implementation reports scikit-learn average precision, which weights
precision by recall increments. Trapezoidal integration of a PR curve can differ,
so an unspecified "PR-AUC" label would be ambiguous. ROC-AUC remains a valid
ranking summary; it is secondary because it does not directly express the number
of passing observations that will be alerted at a chosen cutoff.

## Does zero F1 mean the model learned nothing?

It means there were no true positives at that operating point. Confusion counts
are needed to distinguish no alerts from only false alerts. AP is needed to
assess ranking. A single zero-F1 report cannot prove that PCA erased signal or
that SMOTE generated physically invalid observations.

## Does OOF prediction make the development score unbiased?

Each row is scored by a fold model that did not train on that row. However, the
same development data select the candidate and the threshold. Winning development
AP is selected, pooled OOF AP is not independent, and maximum OOF F1 is optimistic.
Shared folds make candidate comparisons more controlled; they do not substitute
for nested evaluation or new data.

## Did this project already use the test data?

Yes. The historical script tuned a threshold on its test labels and reported the
result there. Those results are exploratory. The revised run freezes selection
before evaluating its seeded holdout, but the dataset has already influenced the
project's design. I would describe the new holdout as protected within the revised
run, not as data never seen during the project's history.

## What if a different model wins on the holdout table?

The development-selected candidate remains selected. All seven candidates are
evaluated after selection for transparency, not for a second contest. Switching
the winner or tuning a new threshold based on that table would turn the holdout
into development data. Any such follow-up must be labeled exploratory and needs
another evaluation boundary.

## Why scale trees, and why put scaling before imputation?

The trees do not generally require scaling. KNN and SMOTE distances, PCA variance,
and logistic regularization do depend on scale. Scaling observed values before
imputation makes the intended KNN distance meaningful in standardized coordinates.
The median branch uses the same ordering. This also fixes the old median branch's
separate train/test scaler fits.

## Why use both class weights and SMOTE?

All Random Forest candidates keep `class_weight="balanced"` in their settings.
In the SMOTE branch, those weights are computed from the resampled labels. Once
the classes are balanced by resampling, the weights are equal. This is not a
second multiplication by the original imbalance ratio. The
[SMOTE chapter](06_algorithmic_failure_pca_and_smote.md) explains the comparison's
remaining interpretation limits.

## Are weighted scores calibrated? Does F1 prove business value?

Neither follows. Positive weights alter the training objective; calibration has
not been established. F1 combines precision and recall while ignoring true
negatives and explicit costs. A real decision rule would need review capacity,
costs, timing of available inputs, and prospective evidence that intervention
helps. The project reports alarm burden instead of inventing those quantities.

## What does the temporal check establish?

It checks fixed `xgb_knn` within development data: earlier time splits choose its
threshold and the later block evaluates that rule. On **251 later rows with 14
failures**, AP is **0.0967** against prevalence **0.0558**, and recall is **zero**
at the earlier-selected threshold **0.267113**. I leave that result visible.

This is not another model selection process or an assessment of RF/KNN/SMOTE's
temporal performance. It is a fixed-XGBoost sensitivity check, and it is not
independent of earlier exploration. The random holdout and temporal check address
different assumptions; neither removes possible lot or tool dependence.
[Chronological results](../results/RESULTS.md#chronological-sensitivity).

## Can feature importance tell an engineer what to repair?

Importance identifies model dependence under a particular representation. Gain
is not a physical cause, a signed sensor effect, or an explanation of each alert.
Anonymous sensor IDs cannot be mapped to subsystems without additional metadata.
Repair or maintenance recommendations would need process knowledge and further
measurement.

## What did the supplementary checks establish?

The three new development repeats rank XGBoost first at seed 17 and RF/KNN/SMOTE
first at 43 and 89, among the four checked candidates. SMOTE-minus-plain-RF/KNN
mean fold AP is positive in all three repeats: **+0.0183, +0.0447, and +0.0433**.
The same development rows and overlapping training sets are reused, so these are
descriptive results, not independent intervals or evidence for replacing the
original selected model. The [diagnostics chapter](13_development_diagnostics.md)
explains the seed policy and raw evidence.

Capacity answers a different question: at the 10% per-fold budget, plain RF/KNN
catches **29 of 83** development failures with **125 reviews**, versus SMOTE's
**20**. For SMOTE, the 5%, 10%, and 20% per-fold budgets produce **60, 125, and
250 reviews**, catching **15, 20, and 39 failures**, with **45, 105, and 211
passing observations reviewed**. The limits use `floor(budget * fold_rows)` and
preserve whole score ties. These random-fold blocks do not establish real
inspection-batch behavior, costs, or prevented losses; this retrospective policy
comparison does not retune the holdout.

## What is the completed project's impact?

I produced **7 pipelines compared over 35 primary CV fits**, frozen decisions and
predictions, and **60 supplementary CV fits** for development sensitivity.
The data filter retains **446 of 590 sensors** on full development data. I also
completed **21 aggregate capacity comparisons**, **8 original figures plus 1
development diagnostic and 1 findings summary**, a notebook audit with **16
executed code cells**, and a batch-scoring interface with schema checks and hash metadata.
Fit counts exclude full-development refits and the separate chronological check;
fold sensor masks are learned separately.

The selected model catches **10 of 21**
holdout failures with **42 false alarms in 52 reviews**, at **F1 0.2740** and
**AP 0.2727**. That quantifies detection and review burden. It does not measure
manufacturing yield improvement, economic return, or production reliability.
The [case study](14_findings_and_impact.md) ties the completed outputs and each
chart's takeaway to saved evidence.

## How should I read the closing visualizations?

The [findings summary](../images/findings_summary.png) separates held-out ranking,
the same-model threshold comparison, and hypothetical development review burden.
The [supplementary figure](../supplementary/development/development_diagnostics.png)
shows changing repeat ranks and the capacity tradeoff. I state the model, data
partition, decision rule, and takeaway for each plot. Mixing the holdout's
**21 failures** with development's **83**, or treating XGBoost's importance and
threshold plots as explanations of the selected Random Forest, would change the
meaning of the evidence. The figures close the study with measured findings and
limits.

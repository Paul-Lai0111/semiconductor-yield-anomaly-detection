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
threshold and the later block evaluates that rule. It is not a second model
selection process, not an assessment of every possible selected model, and not
independent of earlier exploration. The random holdout and temporal check address
different assumptions. A good result in either does not remove possible lot or
tool dependence.

## Can feature importance tell an engineer what to repair?

Importance identifies model dependence under a particular representation. Gain
is not a physical cause, a signed sensor effect, or an explanation of each alert.
Anonymous sensor IDs cannot be mapped to subsystems without additional metadata.
Repair or maintenance recommendations would need process knowledge and further
measurement.

## What would be the most useful next evidence?

The frozen run now provides predictions, split membership, fold variation,
confusion counts, and uncertainty. Beyond this dataset, I would value
later process data with lot/tool identifiers, sensor definitions, and input
availability times. That would let me examine dependence and prospective transfer
before expanding the model search. The
[journal](12_project_journal.md) records the current decisions and open questions.

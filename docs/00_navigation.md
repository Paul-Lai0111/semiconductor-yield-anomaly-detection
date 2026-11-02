# 00 - Diagnostic Documentation Navigation

## What I am investigating

I started this side project to understand why a classifier could report high
accuracy on semiconductor process measurements while missing the failure class.
That led me through missing-data handling, scaling, PCA, SMOTE, class weighting,
and decision thresholds. The useful outcome is a comparison I can inspect and
reproduce, with clear limits on what the data supports.

UCI SECOM contains 1,567 observations and 590 anonymized sensor features: 1,463
passes and 104 failures. Despite the repository's anomaly-detection name, the
experiment is supervised binary classification using the supplied failure labels.
It does not establish a factory deployment or an unsupervised novelty detector.

## Evidence status

The original investigation reused test predictions and test labels to choose a
threshold. Its results are **historical exploratory results**. They motivated the
current questions, but cannot serve as an independent estimate of performance.
Some earlier explanations also inferred physical or geometric causes from a
single thresholded score; the chapters below separate observations from hypotheses.

The revised protocol reserves a stratified 20% holdout with seed 2026. Seven
candidates share five development folds. Mean fold average precision selects the
candidate; pooled development out-of-fold predictions select its F1 threshold.
The selection is frozen before holdout evaluation. Exact performance numbers,
the selected candidate, retained feature counts, and PCA component counts are
recorded in the generated evidence below. A new split does not erase earlier
exposure to this public dataset.

I also completed post-hoc development-only repeat and review-capacity checks.
They describe sensitivity and workload using reused development observations;
they do not replace the original winner or retune the holdout operating point.
For the completed project's contribution and measured results, start with
[chapter 14](14_findings_and_impact.md).

The main takeaway is concrete: the selected RF/KNN/SMOTE pipeline has holdout
**AP 0.2727** versus the constant-score **0.0669** reference. Moving the same
model from 0.5 to the development-frozen threshold catches **1 -> 10 of 21
failures**, with **1 -> 52 reviews**, **42 false alarms**, and **11 remaining
misses**. The case study pairs those counts with four findings, completed outputs,
visual takeaways, and the limits of this previously explored public dataset.

## Documentation map

| Chapter | Question it addresses |
| --- | --- |
| [01 - Problem framing and metrics](01_problem_framing_and_metrics.md) | What do ranking, failure detection, and alarm burden measure? |
| [02 - Data pathology and pruning](02_data_pathology_and_feature_pruning.md) | Which sensor filters are learned, and where may they be fitted? |
| [03 - Imputation diagnostics](03_imputation_strategy_diagnostics.md) | What does median versus KNN imputation actually compare? |
| [04 - Scaling and geometry](04_scaling_and_feature_geometry.md) | Why does scaling precede imputation, including for tree pipelines? |
| [05 - PCA as a structural probe](05_pca_as_structural_probe.md) | What can explained variance tell me about prediction? |
| [06 - PCA and SMOTE comparisons](06_algorithmic_failure_pca_and_smote.md) | How do I revisit earlier unsuccessful attempts without assuming their cause? |
| [07 - Threshold and operating point](07_decision_threshold_and_operating_point.md) | How is a decision threshold selected without using holdout labels? |
| [08 - Weighted XGBoost candidate](08_cost_sensitive_xgboost_final_model.md) | What does weighting change, and how does XGBoost enter selection? |
| [09 - Interpretation and engineering limits](09_model_interpretability_and_engineering_translation.md) | What can feature importance support? |
| [10 - Review questions](10_questions_you_will_get_asked.md) | Which assumptions and limitations need a direct answer? |
| [11 - Experiment protocol](11_experiment_protocol.md) | What are the exact split, fit, selection, and reporting rules? |
| [12 - Project journal](12_project_journal.md) | What changed from the original repository, and why? |
| [13 - Development diagnostics](13_development_diagnostics.md) | What do repeated folds and hypothetical review budgets reveal? |
| [14 - Findings and impact](14_findings_and_impact.md) | What did I build, what did it produce, and what do the figures establish? |

Chapters 01 through 09 retain the diagnostic order of the original investigation.
For the reproducibility contract, start with chapter 11. Chapter 12 records the
reasoning behind the revision and what changed after executing the protocol.
Chapter 13 explains the supplementary analysis boundary. Chapter 14 is the
completed case study: the question, my work, comparison method, four findings,
quantified contributions, figure guide, limitations, and conclusion.
Existing chapter filenames are retained, including the historical "final_model"
filename for chapter 08; that filename does not preselect XGBoost.

## Code and generated evidence

[main.py](../main.py) is the command-line entry point.
[modeling.py](../secom/modeling.py) defines the candidates and preprocessing;
[evaluation.py](../secom/evaluation.py) defines AP, threshold selection, and
uncertainty summaries. Environment constraints live in
[pyproject.toml](../pyproject.toml).

The completed default run generates the following evidence. Figures and metrics
belong to the same source/protocol hashes; an image alone is not proof of that
connection.

| Output | Purpose |
| --- | --- |
| [metrics.json](../results/metrics.json) | Structured results and run context |
| [development_metrics.csv](../results/development_metrics.csv) | Development candidate comparison |
| [holdout_metrics.csv](../results/holdout_metrics.csv) | Post-selection holdout comparison |
| [fold_metrics.csv](../results/fold_metrics.csv) | Fold-level scores and variation |
| [split_manifest.csv](../results/split_manifest.csv) | Row assignments for auditing the split |
| [oof_predictions.csv](../results/oof_predictions.csv) | Development scores used for threshold selection |
| [holdout_predictions.csv](../results/holdout_predictions.csv) | Scores from refitted candidates on the holdout |
| [feature_importance.csv](../results/feature_importance.csv) | Model-specific feature importance |
| [protocol.json](../results/protocol.json) | Selection and configuration frozen before holdout evaluation |
| [RESULTS.md](../results/RESULTS.md) | Generated human-readable report |
| [images](../images/) | Eight original frozen-run figures |
| [Executed notebook](../main.ipynb) | Sixteen executed code cells reconstruct metrics and audit the original saved evidence |
| `artifacts/selected_model.joblib` | Generated locally: selected pipeline, threshold, sensor order, and configuration |
| [Supplementary report](../supplementary/development/REPORT.md) | Post-hoc development-only repeated comparisons and review counts |
| [Repeat metrics](../supplementary/development/repeat_metrics.csv) | Four candidates across three shared five-fold repeats |
| [Capacity summary](../supplementary/development/capacity_summary.csv) | Seven candidates at three hypothetical review budgets |
| [Supplementary provenance](../supplementary/development/provenance.json) | Original evidence hashes and separate supplementary configuration/source hashes |
| [Development diagnostics figure](../supplementary/development/development_diagnostics.png) | Descriptive repeat AP and selected-pipeline capacity tradeoff |
| [Findings summary](../images/findings_summary.png) | Summary of completed results, generated separately from saved metrics |

The supplementary and batch-inference scripts sit outside the original frozen
experiment source-hash set. They document their own source hashes. The
[batch command](../README.md#use-the-saved-pipeline) exports scores, frozen-rule
alerts, and metadata without refitting the model.

The visual evidence totals **8 original figures + 1 supplementary development
diagnostic + 1 findings summary**. The additional summary does not replace an
original experiment figure or change the frozen run's provenance. Each plot's
takeaway, model identity, and evidence boundary are explained in
[chapter 14](14_findings_and_impact.md).

Read performance claims from the generated report together with its protocol and
prediction files. This documentation explains how to interpret that evidence; it
does not substitute remembered scores for a completed run.

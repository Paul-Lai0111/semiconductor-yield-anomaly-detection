# 03 – Imputation Strategy Diagnostics

## 1. Why Imputation Strategy Matters

After removing high-sparsity and zero-variance features,  
the dataset still contains missing values.

At this stage, we face a critical design decision:

How do we fill remaining gaps without destroying anomaly structure?

This is not a cosmetic preprocessing step.

Imputation changes the geometry of the feature space.

And geometry determines classification boundaries.

---

## 2. Two Candidate Strategies

We evaluated two strategies:

| Pipeline | Strategy |
|-----------|------------|
| A | Global Median Imputation |
| B | K-Nearest Neighbors (KNN) Imputation |

Both pipelines operate on the same 446-dimensional space.

---

## 3. What Each Algorithm Does Under the Hood

### 3.1 Median Imputation (Global Statistic)

For each feature:

Missing value → replaced with feature median

Mathematically:

x_missing = median(x_feature)

This assumes:

Missingness is random  
Feature distribution is stable  
Global central tendency is meaningful  

It ignores local correlations between sensors.

---

### 3.2 KNN Imputation (Local Correlation)

For each sample with missing values:

1. Find k nearest complete samples
2. Use their feature values to estimate missing entry

This preserves:

Local geometry  
Inter-sensor correlation  
Failure-mode clusters  

In semiconductor systems:

Failure states often produce correlated sensor fluctuations.

KNN attempts to preserve this.

---

## 4. PCA Variance Diagnostic

We compared macro variance structure using PCA.

| Pipeline | Components Needed for 90% Variance |
|------------|------------------------------------|
| Median | 129 |
| KNN | 129 |

Observation:

Global variance structure remains identical.

Interpretation:

Since high-sparsity features were removed earlier,  
remaining missingness is sparse.

Therefore, macro geometry is stable.

This tells us:

The real difference between imputation strategies is local, not global.

---

## 5. Downstream Model Evidence (Random Forest, Threshold=0.15)

### Pipeline A – Median

| Metric | Value |
|----------|--------|
| Precision | 0.15 |
| Recall | 0.10 |
| F1 | 0.1176 |

---

### Pipeline B – KNN

| Metric | Value |
|----------|--------|
| Precision | 0.23 |
| Recall | 0.14 |
| F1 | 0.1765 |

---

## 6. Why KNN Performed Better

Median imputation smooths the minority class.

Defect samples often have subtle deviations.

Replacing missing entries with median pulls them toward normal cluster.

This reduces separability.

KNN imputation preserves:

Local sensor correlation structure.

It respects the idea:

Similar wafers likely share failure modes.

---

## 7. What Would Break If We Used Median?

- Minority anomalies shrink toward majority center
- PCA later suppresses minority variance further
- SMOTE interpolates from already-smoothed points
- Decision boundary collapses faster

Median increases overlap.

KNN preserves local heterogeneity.

---

## 8. Why KNN Still Doesn't Solve Everything

Despite improvement:

F1 remains below 0.18

This tells us:

The core issue is not imputation alone.

It is:

Class imbalance  
Probability compression  
High-dimensional overlap  

Imputation selection improves signal preservation.

It does not fix rare-event learning dynamics.

---

## 9. Decision Made at This Stage

We select:

KNN Imputation as architectural backbone.

Because:

It preserves local structure  
It improves minority F1 by ~50%  
It is geometrically consistent  

But we acknowledge:

The imbalance problem remains unsolved.

Next stage:

Scaling and geometry normalization.
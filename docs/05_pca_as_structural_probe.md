# 05 – PCA as Structural Probe (Not a Solution)

## 1. Why We Tried PCA

After:

- Feature pruning (590 → 446)
- KNN imputation
- Standardization

We still face:

- 446-dimensional space
- Severe class imbalance (93% vs 7%)
- Overlapping class distributions

High dimensionality increases:

- variance complexity
- overfitting risk
- instability of decision boundaries

Therefore, we explored PCA.

But not as blind dimensionality reduction.

We used PCA as a structural diagnostic tool.

---

## 2. What PCA Actually Does

PCA finds orthogonal directions that maximize variance.

Formally:

Find vector w that maximizes:

Var(wᵀX)

subject to:

||w|| = 1

Key property:

PCA is variance-driven, not label-aware.

It does NOT consider class labels.

It only preserves directions of maximum global variance.

---

## 3. PCA Results

We computed cumulative explained variance.

| Pipeline | Components for 90% Variance |
|-----------|----------------------------|
| Median | 129 |
| KNN | 129 |

Observation:

90% of total variance can be compressed from 446 → 129 dimensions.

This appears promising.

But variance preservation ≠ class separability preservation.

---

## 4. Why PCA Is Dangerous Under Imbalance

In a 93:7 dataset:

Total variance is dominated by majority class.

Let:

X = X_majority ∪ X_minority

Since majority samples are ~14x more frequent,
their covariance structure dominates global variance.

Therefore:

Principal components align with majority distribution.

Minority-specific fluctuations are treated as noise.

Mathematically:

If minority variance direction contributes only 7% of total energy,
PCA suppresses it.

---

## 5. Empirical Evidence: Complete Collapse

After reducing to 129 dimensions:

Random Forest performance:

| Metric | Value |
|----------|--------|
| Precision (defect) | 0.00 |
| Recall (defect) | 0.00 |
| F1 | 0.00 |

The model predicts every sample as normal.

Why?

Because PCA erased subtle minority structure.

---

## 6. What Broke

PCA optimizes:

Global reconstruction error.

We need:

Minority detection sensitivity.

These objectives are not aligned.

PCA compresses:

Low-variance but high-discriminative signals.

In rare-event detection,
important signals are often low-variance.

---

## 7. Why PCA Still Matters

Even though PCA failed as solution,
it revealed something critical:

The minority signal is extremely weak relative to global variance.

This confirms:

The problem is not just dimensionality.
It is signal suppression.

PCA failure was informative.

It diagnosed variance imbalance.

---

## 8. Decision Made at This Stage

We reject PCA as a solution.

We preserve full 446-dimensional space.

We shift strategy from:

Topology modification

to:

Objective function modification.

Next step:

Investigate synthetic balancing (SMOTE),
and understand why topology manipulation fails.
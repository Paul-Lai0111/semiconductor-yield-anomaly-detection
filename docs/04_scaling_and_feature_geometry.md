# 04 – Scaling and Feature Geometry

## 1. Why Scaling Is Not Cosmetic

After selecting KNN imputation, the feature matrix contains:

- 1,567 samples
- 446 features
- heterogeneous sensor scales

Example:

Some sensors:
- values around 3,000
- others around 0.01
- others around 100

Without scaling:

High-magnitude sensors dominate distance calculations and variance structure.

This is not a modeling detail.
It changes the geometry of the feature space.

---

## 2. What StandardScaler Actually Does

StandardScaler performs:

For each feature:

x_scaled = (x - mean) / standard deviation

This transforms each feature to:

Mean = 0  
Standard deviation = 1  

This ensures all sensors operate on the same statistical baseline.

---

## 3. What Breaks If We Skip Scaling

### 3.1 KNN Imputation Distortion

KNN uses Euclidean distance:

d(x, y) = sqrt(sum((x_i - y_i)^2))

If Sensor A ranges [0, 3000]
and Sensor B ranges [0, 0.01]

Then Sensor A dominates distance.

Neighbor selection becomes magnitude-driven,
not structure-driven.

---

### 3.2 PCA Variance Bias

PCA maximizes variance:

Max Var(w^T X)

If large-scale sensors dominate variance,
principal components align with magnitude,
not discriminative structure.

Minority anomaly signals become buried.

---

### 3.3 Optimization Instability

For boosting algorithms:

Gradient updates depend on feature scale.

Unscaled data can cause:

- elongated loss surface
- unstable splits
- suboptimal convergence

---

## 4. Why Tree Models “Technically” Don’t Require Scaling

Common statement:

"Tree-based models do not require scaling."

This is partially true.

Tree splits are scale-invariant in 1D.

But:

- KNN imputation is scale-sensitive.
- PCA is scale-sensitive.
- Distance-based reasoning depends on scaling.
- Even tree ensembles benefit from balanced feature spread.

Since scaling is required upstream,
we standardize globally for consistency.

---

## 5. High-Dimensional Geometry After Scaling

After scaling:

The dataset becomes:

446-dimensional normalized hyperspace.

But this introduces a new reality:

In high dimensions:

- Distances concentrate
- Nearest neighbors become less meaningful
- Class overlap increases
- Rare clusters shrink in relative volume

This geometric property foreshadows:

Why SMOTE will later fail.

Because interpolation in high-dimensional space does not preserve class manifold integrity.

---

## 6. Evidence From Observed Model Behavior

After scaling:

Random Forest still collapses to predicting majority class under default threshold.

This tells us:

The issue is not magnitude distortion anymore.

The issue is:

Loss function bias under imbalance.

---

## 7. Decision Made at This Stage

We commit to:

- Scaling as mandatory preprocessing
- Preserving 446 dimensions
- Using scaling before PCA and model training

We have corrected magnitude bias.

Next, we investigate dimensionality reduction.

PCA is not yet a solution.

It is a diagnostic tool.
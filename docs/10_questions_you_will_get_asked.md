# 10 Q and A

This chapter anticipates technical, statistical, and engineering questions that may arise during project review.

---

## 1. Why Didn’t You Use Deep Learning?

**Answer:**

- Dataset size: 1,567 samples is too small for deep neural networks.
- Tabular sensor data is better handled by tree ensembles.
- High-dimensional but low-sample regime favors gradient boosting.
- Interpretability is critical in manufacturing.

Deep learning would likely overfit and reduce interpretability.

---

## 2. Why Not Use ROC-AUC Instead of PR?

ROC-AUC is misleading under imbalance.

False Positive Rate:

FPR = FP / (FP + TN)

Since TN is large (1,463),
FPR remains small even if FP increases significantly.

PR curve directly measures:

Precision and Recall of defect class.

Which aligns with business objective.

---

## 3. Why Not Keep PCA to Reduce Overfitting?

Because PCA is variance-driven.

Minority defect signals are low-variance.

PCA suppresses them.

Empirical evidence:

F1 collapsed to 0.00 after PCA reduction.

Dimensionality reduction must preserve discriminative power,
not just variance.

---

## 4. Why Did SMOTE Fail?

Two reasons:

1. Minority class is not well-separated.
2. High dimensionality (446D) causes convex hull explosion.

Interpolation across overlapping manifolds
produces synthetic samples indistinguishable from majority.

SMOTE assumes cluster structure.
This dataset violates that assumption.

---

## 5. Why Not Use Undersampling Instead?

Undersampling would reduce majority samples from 1,463
to roughly 104.

This would:

- Discard real signal
- Increase variance
- Risk unstable model

Given limited data,
discarding majority information is not ideal.

---

## 6. Why Use scale_pos_weight Instead of Class Weighting in Random Forest?

Random Forest class_weight shifts split criteria,
but boosting adjusts gradient magnitude.

Boosting focuses sequentially on hard examples.

Weighted gradient descent is more effective
for rare-event emphasis.

---

## 7. Why Lower Threshold So Aggressively?

Threshold ≈ 0.0159 appears extreme.

But under rare-event detection:

True defect probabilities are naturally small.

Threshold tuning is not cheating.

It defines operating policy.

In production,
threshold reflects inspection capacity.

---

## 8. Is 0.36 F1 Actually Good?

Baseline defect rate: 6.6%

Random precision baseline: 0.066

Model precision: 0.26

≈ 4x improvement over random.

Recall: 0.57

Intercepting over half of defects
is operationally meaningful.

Given extreme imbalance and overlap,
this performance is realistic.

---

## 9. How Would You Improve Further?

Potential directions:

- SHAP value analysis per prediction
- Semi-supervised anomaly detection
- Cost-sensitive hyperparameter tuning
- Time-series modeling if wafer order is available
- Ensemble stacking

However,
improvement must preserve interpretability.

---

## 10. What Is the Core Lesson of This Project?

This project is not about:

Applying XGBoost.

It is about:

Understanding why models fail under imbalance,
why topology manipulation is insufficient,
why gradient dynamics matter,
why threshold defines policy,
and how to translate ML into engineering action.

---

## 11. Final Reflection

The system built here is:

- Not perfect
- Not fully causal
- Not a physics simulator

It is a structured defect risk ranking engine
optimized under real-world constraints.

The true strength of this project
is not the final model.

It is the reasoning process behind every decision.
# 08 – Final Model: Cost-Sensitive XGBoost

## 1. Why We Chose XGBoost

After observing failure patterns in:

- PCA (variance suppression)
- SMOTE (manifold distortion)
- Default Random Forest (majority collapse)

We required a model that:

- Handles high-dimensional tabular data
- Captures nonlinear interactions
- Supports class weighting
- Is robust under noise
- Performs well in imbalanced settings

XGBoost satisfies these requirements.

It is a gradient boosting framework
that sequentially minimizes residual errors.

---

## 2. What XGBoost Does Under the Hood

XGBoost optimizes:

Logistic loss:

L = - Σ [ y log(p) + (1-y) log(1-p) ]

Where:

p = sigmoid(f(x))

It builds trees sequentially,
each correcting the gradient residual of previous trees.

Key concept:

Gradient = prediction error signal.

Under imbalance,
majority class dominates gradient updates.

Therefore:

Without intervention,
model favors majority.

---

## 3. Cost-Sensitive Learning via scale_pos_weight

To counter imbalance,
we set:

scale_pos_weight ≈ (negative samples / positive samples)

This modifies the loss:

L = - Σ [ w * y log(p) + (1-y) log(1-p) ]

Where:

w > 1 for positive (defect) class.

Effect:

False Negatives become more expensive.

Gradients for defect misclassification increase.

The model is forced to pay attention to minority samples.

---

## 4. Why Weighting Alone Was Not Enough

With cost-sensitive learning
but default threshold 0.5:

F1 remained 0.00.

Why?

Because:

Even though gradient dynamics shift,
probability outputs remain compressed.

Thus:

Weighting improves ranking quality,
but decision threshold must still be tuned.

This confirms:

Training objective and operating threshold are independent axes.

---

## 5. Final Optimized Performance

Using:

- KNN imputation
- Standardization
- Full 446 dimensions
- scale_pos_weight
- PR-based threshold optimization

Best threshold ≈ 0.0159

Final performance:

| Metric | Value |
|----------|--------|
| Precision | 0.26 |
| Recall | 0.57 |
| F1 | 0.36 |
| Accuracy | 0.86 |

Interpretation:

57% of defects intercepted.

Precision 0.26 means:

About 1 in 4 flagged wafers is truly defective.

Baseline defect rate was 6.6%.

This represents ~4x precision lift over random.

---

## 6. Why This Is a Meaningful Improvement

Under rare-event detection:

Perfect precision is unrealistic.

The real question is:

Does recall gain justify inspection cost?

In semiconductor manufacturing:

Intercepting 57% of defects
may save significant downstream cost.

The tradeoff is operationally favorable.

---

## 7. Why XGBoost Succeeded

XGBoost succeeded because:

- It models nonlinear feature interactions.
- It handles sparse structure well.
- Weighted gradients amplify minority signal.
- It produces calibrated ranking probabilities.
- Threshold tuning converts ranking into policy.

Unlike PCA or SMOTE,
it does not distort topology.

It changes optimization dynamics.

---

## 8. What This Model Is NOT

It is not:

- A perfect defect detector.
- A calibrated probability estimator.
- A generative model of manufacturing process.

It is:

A risk ranking engine
converted into inspection policy.

---

## 9. Decision at Final Model Stage

We commit to:

- Cost-sensitive XGBoost
- PR-based threshold optimization
- Full feature space retention
- No synthetic sampling
- No dimensionality compression

Next step:

Translate model importance into engineering insight.
# 07 – Decision Threshold and Operating Point

## 1. The Hidden Problem: Default Threshold = 0.5

After:

- Feature pruning
- KNN imputation
- Scaling
- PCA failure
- SMOTE failure

We trained XGBoost with class weighting.

But under default threshold (0.5):

F1 (defect) = 0.00

The model still predicts all samples as normal.

This is not a model failure.

It is a decision policy failure.

---

## 2. What the 0.5 Threshold Actually Means

Binary classifier outputs probability:

P(y = 1 | x)

Using threshold = 0.5 means:

We require the model to believe defect is more likely than normal.

In highly imbalanced problems:

True defect probability is naturally low.

Even real defective wafers may receive:

0.01  
0.03  
0.07  

Requiring > 0.5 is mathematically unrealistic.

---

## 3. Probability Compression in Rare Event Learning

In imbalanced datasets:

The model minimizes log-loss:

Loss = -[y log(p) + (1-y) log(1-p)]

Because majority dominates training,
the optimal probability distribution shifts toward normal class.

This causes probability compression:

Predicted probabilities cluster near 0.

Even strong defect signals may not exceed 0.1.

This explains observed test outputs:

Top predicted defect probabilities were around:

0.35  
0.38  
0.52 (rare)

But most true defects remained below 0.5.

---

## 4. Precision-Recall Curve as Operating Map

Instead of fixing threshold at 0.5,
we evaluated the Precision-Recall curve.

PR curve directly maps:

Threshold → (Precision, Recall)

This allows us to select an operating point
aligned with business objective.

Remember:

False Negative cost >> False Positive cost

Therefore:

Recall must be prioritized.

---

## 5. Empirical Threshold Optimization

We searched across thresholds.

Best threshold found:

≈ 0.0159

At this threshold:

| Metric | Value |
|----------|--------|
| Precision | 0.26 |
| Recall | 0.57 |
| F1 | 0.36 |
| Accuracy | 0.86 |

Comparison:

| Setting | F1 |
|----------|------|
| Default 0.5 | 0.00 |
| Optimized Threshold | 0.36 |

This is the first successful break of the collapse.

---

## 6. Why Lowering Threshold Works

Lower threshold:

Expands positive prediction region.

We are no longer asking:

"Is this wafer more likely defective than normal?"

We are asking:

"Does this wafer show early defect signal?"

This reframes the classifier as:

Risk ranking system.

Not binary truth oracle.

---

## 7. What Breaks If We Ignore Threshold Tuning

If threshold remains 0.5:

- Recall remains zero
- System detects no defects
- Production failure continues

In rare-event detection:

Threshold is part of the model.

It is not post-processing.

---

## 8. Operating Point as Business Decision

Selecting threshold is equivalent to:

Selecting inspection budget.

Higher threshold:
- Fewer inspections
- More missed defects

Lower threshold:
- More inspections
- Fewer missed defects

This is a controllable tradeoff.

Therefore:

Model training and decision threshold must be separated.

---

## 9. Decision Made at This Stage

We conclude:

- Threshold tuning is mandatory.
- Default 0.5 is invalid under imbalance.
- PR-based optimization aligns with business objective.

However:

Threshold tuning alone does not solve gradient bias.

We must also modify objective function.

Next chapter:

Cost-sensitive XGBoost as final model.
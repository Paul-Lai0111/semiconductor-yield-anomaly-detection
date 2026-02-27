# 01 – Problem Framing and Metric Selection

## 1. Industrial Context: What Are We Actually Optimizing?

This is not a generic binary classification problem.

This is a semiconductor yield interception system.

In semiconductor manufacturing, yield is directly tied to profit margin.  
A defective wafer that escapes detection can result in:

- downstream assembly contamination
- field returns
- contractual penalties
- reputational damage

In contrast, a false alarm (flagging a good wafer for inspection) incurs:

- inspection labor cost
- temporary production slowdown

Therefore:

False Negative cost >> False Positive cost

This asymmetry fundamentally changes how we define “good performance.”

---

## 2. Dataset Structure

| Metric | Value |
|--------|--------|
| Total Samples | 1,567 |
| Raw Sensors | 590 |
| Defective Units | 104 |
| Normal Units | 1,463 |
| Defect Ratio | 6.6% |

Class imbalance:

| Class | Count | Percentage |
|--------|--------|------------|
| Normal (0) | 1,463 | 93.4% |
| Defective (1) | 104 | 6.6% |

This immediately tells us:

A trivial classifier predicting "all normal" yields:

Accuracy ≈ 93.4%  
F1 (defect class) = 0

Therefore, accuracy is not only useless — it is dangerous.

---

## 3. Why Accuracy Is Misleading

Accuracy is defined as:

Accuracy = (TP + TN) / (Total Samples)

In imbalanced datasets:

TN dominates the numerator.

For this dataset:

If we predict all wafers as normal:

TP = 0  
FN = 104  
TN = 1,463  
FP = 0  

Accuracy ≈ 93%

But the system detects zero defects.

In production, that system would be catastrophic.

---

## 4. Metric Selection Logic

We must select metrics aligned with business cost.

### Candidate Metrics

- Accuracy → rejected
- ROC-AUC → misleading under extreme imbalance
- Precision → important (controls inspection load)
- Recall → critical (defect interception)
- F1-score → harmonic balance between precision and recall
- PR-AUC → preferred under rare-event detection

### Why PR Curve Instead of ROC Curve?

ROC measures:

True Positive Rate vs False Positive Rate

But in imbalanced settings:

False Positive Rate = FP / (FP + TN)

Since TN is very large (1,463), FPR remains artificially small even if FP increases significantly.

Precision-Recall curve instead focuses on:

Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)

This directly reflects:

- How many detected defects are actually real?
- How many total defects are we intercepting?

This aligns with operational decision-making.

---

## 5. Operating Objective Definition

We are solving:

Maximize Recall_defect  
Subject to acceptable Precision_defect

Not:

Maximize Accuracy

This means:

The model is a ranking function.

The decision threshold becomes a policy variable.

---

## 6. Baseline Diagnostic: The 0 F1 Collapse

Initial Random Forest results:

| Model | Threshold | F1 (Defect) |
|--------|-----------|------------|
| RF + Median | 0.5 | 0.00 |
| RF + KNN | 0.5 | 0.00 |

Interpretation:

The model learned to classify everything as normal.

Why?

Because minimizing overall classification error under imbalance incentivizes ignoring the minority class.

The loss function is implicitly biased toward majority accuracy.

---

## 7. Core Problem Statement

We are not just training a classifier.

We are designing a defect interception policy under:

- Rare-event suppression
- High-dimensional sensor noise
- Cost asymmetry
- Probability compression

Therefore, the rest of this report investigates:

Why standard methods collapse  
Why probability outputs shrink  
Why topology manipulation fails  
Why cost-sensitive learning shifts the gradient  
Why threshold tuning is mandatory  

---

## 8. Decision Made at This Stage

1. Reject accuracy as optimization target
2. Adopt F1-score (defect class) as primary metric
3. Use Precision-Recall curve for threshold analysis
4. Treat threshold as a controllable operating parameter

This framing dictates every modeling decision that follows.
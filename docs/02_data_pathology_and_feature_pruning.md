# 02 – Data Pathology and Feature Pruning

## 1. Why Data Pathology Must Be Diagnosed First

Before training any model, we must answer:

Are we modeling signal, or modeling sensor failure?

The SECOM dataset contains:

| Metric | Value |
|--------|--------|
| Total Samples | 1,567 |
| Raw Features | 590 |
| Total Missing Values | 41,951 |

This immediately indicates:

The dataset is not clean industrial telemetry.
It is a noisy sensor log with structural defects.

If we skip pathology analysis:

- The model will overfit to missingness artifacts
- Synthetic patterns may dominate real signal
- Dimensionality will amplify noise
- Imputation may hallucinate structure

Therefore, feature pruning is not optional.
It is mathematically required.

---

## 2. High-Sparsity Sensors

We observed extreme missingness in several sensors.

Top 15 most corrupted sensors:

| Sensor ID | Missing Count |
|------------|----------------|
| 292 | 1429 |
| 293 | 1429 |
| 157 | 1429 |
| 158 | 1429 |
| 220 | 1341 |
| 358 | 1341 |
| 85 | 1341 |
| 492 | 1341 |
| 382 | 1018 |
| 384 | 1018 |
| 244 | 1018 |
| 111 | 1018 |
| 516 | 1018 |
| 517 | 1018 |
| 110 | 1018 |

For a dataset of 1,567 samples:

1,429 missing values = 91% missing

These are effectively dead sensors.

---

## 3. Why >50% Missing Is a Hard Cut

We removed 28 high-sparsity features.

Why 50%?

### 3.1 Statistical Reasoning

If more than half of values are missing:

The imputer generates more synthetic values than real observations.

The model learns:

Imputer behavior > physical sensor behavior

This violates the principle:

"Model must learn physical process, not data repair artifacts."

---

### 3.2 Bias Amplification

High sparsity often means:

- Sensor only activates in rare state
- Sensor malfunction
- Batch-specific logging failure

Imputation converts sparse rare-event signals into dense artificial noise.

This increases overlap between classes.

---

### 3.3 What Breaks If We Keep Them?

- KNN distance metrics distort
- PCA variance dominated by imputation variance
- SMOTE interpolates across artificial values
- Tree splits may form around missingness artifacts

---

## 4. Zero-Variance Sensors

We removed 116 constant features.

Why?

Information gain of constant feature:

IG = 0

Entropy contribution = 0

Tree-based models ignore them, but:

High-dimensional geometry worsens distance metrics.

Constant features:

- Increase dimensionality
- Increase computational cost
- Worsen curse of dimensionality
- Add no discriminative value

Removing them is mandatory.

---

## 5. Dimensionality After Pruning

| Stage | Feature Count |
|--------|---------------|
| Raw | 590 |
| After >50% Missing Removal | 562 |
| After Zero-Variance Removal | 446 |

Final working feature space:

446 dimensions

Still extremely high-dimensional relative to:

1,567 samples

This ratio alone predicts:

Distance concentration  
Class overlap amplification  
Minority suppression  

---

## 6. Curse of Dimensionality Implications

In high-dimensional space:

- Distances between points become similar
- Nearest neighbor loses meaning
- Decision boundaries become unstable
- Minority clusters dissolve into majority manifold

This explains why:

SMOTE later fails catastrophically.

Because in 446 dimensions:

Interpolation does not preserve class structure.

---

## 7. Evidence From Observed Model Behavior

Before pruning:

Models collapsed to predicting majority class.

After pruning:

The collapse still occurred.

This tells us:

The core problem is not just noise.
It is geometric and statistical imbalance.

But pruning was necessary to prevent:

Imputation hallucination  
Variance distortion  
Dead channel bias  

---

## 8. Decision Made at This Stage

We commit to:

1. Remove >50% missing sensors
2. Remove zero-variance sensors
3. Accept 446 dimensions as minimal viable signal space
4. Proceed to imputation diagnostics

We have reduced structural noise.
We have not yet solved imbalance.

That comes next.
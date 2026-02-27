# 06 – Algorithmic Failure: PCA and SMOTE

## 1. Why We Tried SMOTE

After PCA failure, we preserved full 446-dimensional space.

But the imbalance problem remained:

| Class | Count |
|--------|--------|
| Normal | 1,463 |
| Defect | 104 |

Training split:

| Class | Count |
|--------|--------|
| Normal | 1,170 |
| Defect | 83 |

The minority class is severely underrepresented.

To correct this, we applied SMOTE:

Synthetic Minority Over-sampling Technique.

---

## 2. What SMOTE Does Under the Hood

For each minority sample:

1. Find k nearest minority neighbors.
2. Interpolate between them.
3. Generate synthetic samples along connecting line segments.

Mathematically:

x_new = x_i + λ(x_j - x_i)

where λ ∈ (0,1)

This creates artificial minority points inside minority convex hull.

Assumption:

Minority class forms coherent local clusters.

---

## 3. Why SMOTE Fails in High Dimensions

Our feature space:

446 dimensions

In high-dimensional space:

- Distances concentrate
- Neighborhood structure becomes unstable
- Minority clusters dissolve
- Convex hull expands dramatically

Additionally:

The minority class is embedded inside majority manifold.

This creates class overlap.

SMOTE interpolates across this overlapping region.

---

## 4. Empirical Evidence: Still 0 F1

After applying SMOTE:

Training distribution:

| Class | Count |
|--------|--------|
| Normal | 1,170 |
| Defect | 1,170 |

Perfectly balanced.

Yet test performance:

| Metric | Value |
|----------|--------|
| Precision | 0.00 |
| Recall | 0.00 |
| F1 | 0.00 |

The model still predicts everything as normal.

---

## 5. Why This Happens

Two structural reasons:

### 5.1 Minority Manifold Collapse

Minority points are not forming clean clusters.

They are scattered near majority region.

Interpolation does not strengthen boundary.

It blurs it further.

---

### 5.2 Curse of Dimensionality

In 446 dimensions:

Convex hull volume grows exponentially.

Synthetic samples fill large empty regions.

These regions often resemble majority space.

Thus:

SMOTE may create "fake anomalies" that look normal.

The classifier learns nothing useful.

---

## 6. Key Insight: Topology Manipulation Is Not Enough

We attempted:

- PCA (dimensional compression)
- SMOTE (data balancing)

Both modify data topology.

Both failed.

This tells us:

The core issue is not data representation.

It is loss function bias.

---

## 7. Why the Model Still Predicts All Normal

Under default threshold 0.5:

The model must assign >50% probability to classify defect.

In rare-event detection:

True defect probabilities are naturally low.

Even true defects may receive probabilities like:

0.02  
0.05  
0.08  

But 0.5 is unreachable.

Thus:

All predictions collapse to majority.

This reveals a second problem:

Probability compression.

---

## 8. What This Failure Teaches Us

We learned:

- Dimensionality reduction suppresses minority variance.
- Synthetic sampling distorts manifold geometry.
- Class overlap cannot be fixed by interpolation.
- Default decision threshold is unrealistic.

The solution is not:

Change data.

The solution is:

Change objective and decision policy.

---

## 9. Decision Made at This Stage

We abandon topology manipulation strategies.

We move to:

Cost-sensitive learning.

We will modify gradient dynamics
to penalize False Negatives more heavily.

And we will treat threshold as operating policy.

Next chapter:

Decision threshold analysis.
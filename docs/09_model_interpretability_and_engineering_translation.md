# 09 – From Model to Engineering Roadmap

## 1. Why Interpretability Is Mandatory

A machine learning model that outputs only predictions is not useful in manufacturing.

Engineers do not act on probabilities.
They act on root causes.

Therefore:

The model must answer:

Which sensors drive defect risk?

And:

How can this be translated into preventative maintenance (PM) strategy?

---

## 2. Feature Importance Extraction

Using XGBoost’s gain-based importance,
we extracted the top contributing sensors.

Top 15 sensors by importance:

| Rank | Sensor ID | Importance |
|------|------------|------------|
| 1 | 271 | 0.048 |
| 2 | 130 | 0.038 |
| 3 | 319 | 0.037 |
| 4 | 199 | 0.031 |
| 5 | 56 | 0.027 |
| 6 | 9 | 0.024 |
| 7 | 103 | 0.022 |
| 8 | 433 | 0.021 |
| 9 | 214 | 0.019 |
| 10 | 489 | 0.019 |
| 11 | 53 | 0.017 |
| 12 | 26 | 0.017 |
| 13 | 31 | 0.015 |
| 14 | 470 | 0.014 |
| 15 | 202 | 0.014 |

These 15 sensors explain a significant proportion of model split gain.

This narrows monitoring focus from:

446 sensors → 15 critical signals.

---

## 3. What “Importance” Means in XGBoost

XGBoost importance (gain-based):

Measures total improvement in loss function
brought by splits using that feature.

It does NOT mean:

Causality  
Physical mechanism  

It means:

This sensor helps partition defect vs normal in tree structure.

---

## 4. Engineering Interpretation Strategy

We translate importance into action in three steps:

### Step 1 – Sensor Grouping

Cluster top sensors by:

- Tool module
- Physical subsystem
- Process step

This identifies concentration of failure risk.

---

### Step 2 – Distribution Analysis

For each top sensor:

Plot:

Defect vs Normal distribution

Look for:

- Shifted mean
- Increased variance
- Bimodal behavior

This reveals physical instability.

---

### Step 3 – Early Warning Threshold

Instead of using model threshold alone:

We can derive:

Sensor-specific anomaly thresholds.

This creates:

Pre-model alert system.

---

## 5. Why This Matters

Instead of saying:

“The model predicts risk”

We can say:

“These 3 subsystems show abnormal fluctuation patterns
associated with yield loss.”

This transforms:

Black-box ML → Actionable PM roadmap

---

## 6. From Detection to Prevention

The pipeline can be used for:

1. Real-time risk scoring
2. Tool health monitoring
3. Sensor drift detection
4. Preventative maintenance scheduling

If a wafer receives high risk score:

Engineers can inspect:

The top contributing sensors for that specific prediction.

This bridges:

Data science → Process engineering.

---

## 7. What This Model Still Cannot Do

- It cannot determine physical causality.
- It cannot simulate process behavior.
- It cannot replace domain expertise.

It highlights statistical risk drivers.

Final decisions remain engineering-driven.

---

## 8. Final Engineering Outcome

We reduced complexity from:

590 raw sensors  
→ 446 validated features  
→ 15 high-impact sensors  

We built:

A defect interception system  
AND  
A sensor prioritization roadmap.

Next chapter:

Anticipated interview and technical challenges.
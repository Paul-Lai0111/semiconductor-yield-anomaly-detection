# 00 – Diagnostic Documentation Navigation

## Purpose of This Documentation

This documentation is not written as a tutorial.

It is written as a structured diagnostic report that mirrors the actual Python pipeline implemented in `main.py`.

The goal is not to demonstrate how to use machine learning libraries.

The goal is to demonstrate how to design, debug, and justify a defect interception system under real industrial constraints.

---

## What This Project Is Actually About

This repository documents the full investigation process behind building a semiconductor yield anomaly detection system under:

- Severe class imbalance (93% vs 7%)
- High-dimensional sensor noise (590 raw features)
- Heavy missingness and dead channels
- Asymmetric business cost (False Negatives >> False Positives)
- Suppressed probability outputs in rare-event learning

The final solution is:

Cost-sensitive XGBoost with precision-recall based threshold optimization.

But the final model is not the story.

The story is the diagnostic journey.

---

## How This Documentation Is Structured

Each chapter answers six core questions:

1. Why is this step necessary?
2. What breaks if we skip it?
3. What is the algorithm doing under the hood?
4. Why can it fail in this specific dataset?
5. What empirical evidence did we observe?
6. What decision did we make next, and why?

This structure ensures that every modeling choice is justified both statistically and operationally.

---

## Documentation Map

The documentation follows the exact logical progression of the pipeline.

### 01 – Problem Framing and Metric Selection
Defines the business objective, cost asymmetry, and why accuracy is misleading.

### 02 – Data Pathology and Feature Pruning
Diagnoses missingness, zero-variance sensors, and dimensional noise.

### 03 – Imputation Strategy Diagnostics
Compares Median vs KNN imputation under geometric and statistical reasoning.

### 04 – Scaling and Feature Geometry
Explains why standardization is required for PCA, KNN, and stable optimization.

### 05 – PCA as Structural Probe (Not a Solution)
Uses PCA to analyze variance structure and minority signal suppression.

### 06 – Algorithmic Failure: PCA and SMOTE
Explains why topology modification fails in high-dimensional overlapping spaces.

### 07 – Decision Threshold and Operating Point
Demonstrates why default 0.5 probability thresholds collapse in rare-event learning.

### 08 – Final Model: Cost-Sensitive XGBoost
Derives the weighted loss function and explains why this shifts the gradient dynamics.

### 09 – From Model to Engineering Roadmap
Translates feature importance into actionable preventive maintenance strategy.

### 10 – Interview Defense and Edge Cases
Anticipates technical questions and challenges the robustness of decisions.

---

## Philosophy Behind the Diagnostic Approach

Many ML projects present only the final model.

This documentation deliberately preserves:

- failed attempts
- statistical dead-ends
- misleading metrics
- suppressed probability behavior
- geometry-induced collapse

Because in real production systems, failures are the most informative signals.

The project demonstrates not algorithm usage, but system reasoning.

---

## How to Read This Report

If you are reviewing this project as:

- A hiring manager → read 01, 06, 08, 09
- A machine learning engineer → read 03 through 08
- An interviewer → read 10
- A manufacturing engineer → read 09

---

## Final Statement

This repository is not a demonstration of algorithm usage.

It is a structured exploration of how to design a defect interception system under:

- severe imbalance
- high-dimensional sensor noise
- asymmetric business cost
- suppressed probability outputs

The final system, cost-sensitive XGBoost with optimized threshold selection, represents a principled balance between statistical rigor and manufacturing practicality.
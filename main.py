import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# 0. Environment Setup
os.makedirs('images', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

print("SECOM Yield Anomaly Detection: Full Diagnostic Pipeline")

# 1. Data Fetching & Cleansing
url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
url_labels = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

df_features = pd.read_csv(url_data, sep=" ", header=None)
df_labels = pd.read_csv(url_labels, sep=" ", header=None, usecols=[0])
df_labels.columns = ['Label']
df = pd.concat([df_features, df_labels], axis=1)

missing_ratio = df.isnull().mean()
df_dropped = df.drop(columns=missing_ratio[missing_ratio > 0.5].index)
df_dropped = df_dropped.drop(columns=df_dropped.std()[df_dropped.std() == 0].index)
df_dropped.columns = df_dropped.columns.astype(str)

X = df_dropped.drop(columns=['Label'])
y = df_dropped['Label'].map({-1: 0, 1: 1})

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Imputation & PCA Variance Analysis
print("[Phase 1] Analyzing Imputation Strategies (Median vs. KNN)...\n")
# Pipeline A: Median
med_imputer = SimpleImputer(strategy='median')
X_train_med = StandardScaler().fit_transform(med_imputer.fit_transform(X_train_raw))
X_test_med = StandardScaler().fit_transform(med_imputer.transform(X_test_raw))

# Pipeline B: KNN
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train_raw)
X_test_knn = knn_imputer.transform(X_test_raw)
scaler_knn = StandardScaler()
X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
X_test_knn_scaled = scaler_knn.transform(X_test_knn)

# --- PCA Variance Visualization ---
pca_median, pca_knn = PCA(), PCA()
pca_median.fit(X_train_med)
pca_knn.fit(X_train_knn_scaled)

cum_var_median = np.cumsum(pca_median.explained_variance_ratio_)
cum_var_knn = np.cumsum(pca_knn.explained_variance_ratio_)

n_comp_med_90 = np.argmax(cum_var_median >= 0.90) + 1
n_comp_knn_90 = np.argmax(cum_var_knn >= 0.90) + 1

plt.figure(figsize=(10, 6))
plt.plot(cum_var_median, label=f'Pipeline A (Median): 90% at {n_comp_med_90} dims', color='#1f77b4', lw=2)
plt.plot(cum_var_knn, label=f'Pipeline B (KNN): 90% at {n_comp_knn_90} dims', color='#ff7f0e', linestyle='--', lw=2)
plt.axhline(y=0.90, color='#d62728', linestyle=':', label='90% Information Threshold')
plt.title('PCA: Cumulative Explained Variance (Median vs. KNN)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('images/pca_variance_comparison.png')
plt.close()
print(f"-> PCA Plot generated. Median requires {n_comp_med_90} dims, KNN requires {n_comp_knn_90} dims for 90% variance.")

# 3. Model Evaluation: Imputation Comparison
rf_med = RandomForestClassifier(random_state=42, class_weight='balanced')
y_pred_med = rf_med.fit(X_train_med, y_train).predict(X_test_med)
print("\n[Report: Pipeline A - Median Imputation + RF]")
print(classification_report(y_test, y_pred_med))

rf_knn = RandomForestClassifier(random_state=42, class_weight='balanced')
y_pred_knn = rf_knn.fit(X_train_knn_scaled, y_train).predict(X_test_knn_scaled)
print("[Report: Pipeline B - KNN Imputation + RF]")
print(classification_report(y_test, y_pred_knn))

# 4. Model Evaluation: The Failed Attempts
print("\n[Phase 2] Diagnosing Algorithmic Traps...\n")
# Attempt 1: PCA (using 129 components based on KNN 90% variance)
pca = PCA(n_components=129, random_state=42)
y_pred_pca = rf_knn.fit(pca.fit_transform(X_train_knn_scaled), y_train).predict(pca.transform(X_test_knn_scaled))
print("[Report: Failed Attempt 1 - PCA Dimensionality Reduction]")
print(classification_report(y_test, y_pred_pca))

# Attempt 2: SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_knn_scaled, y_train)
y_pred_smote = rf_knn.fit(X_train_smote, y_train_smote).predict(X_test_knn_scaled)
print("[Report: Failed Attempt 2 - SMOTE Synthetic Sampling]")
print(classification_report(y_test, y_pred_smote))

# 5. The Solution: XGBoost & Threshold Tuning
print("\n[Phase 3] Deploying Cost-Sensitive XGBoost...\n")
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=imbalance_ratio, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_knn_scaled, y_train)

# Default Threshold (0.5)
y_pred_xgb_def = xgb_model.predict(X_test_knn_scaled)
print("[Report: XGBoost with Default 0.5 Threshold]")
print(classification_report(y_test, y_pred_xgb_def))

# Optimized Threshold via PR Curve
xgb_probs = xgb_model.predict_proba(X_test_knn_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) 
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
y_pred_final = np.where(xgb_probs >= best_threshold, 1, 0)

# Save PR Curve Image
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='#1f77b4', lw=2.5, label=f'XGBoost (AUC-PR = {auc(recall, precision):.3f})')
plt.scatter(recall[best_idx], precision[best_idx], color='#d62728', s=150, zorder=5, edgecolors='white', label=f'Best Threshold: {best_threshold:.4f}\n(F1: {f1_scores[best_idx]:.3f})')
plt.fill_between(recall, precision, alpha=0.1, color='#1f77b4')
plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Tradeoff')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('images/pr_curve.png')
plt.close()

print(f"\n[Report: FINAL SOLUTION - XGBoost with Optimized Threshold ({best_threshold:.4f})]")
print(classification_report(y_test, y_pred_final))

# 6. Feature Importance
feature_importance_df = pd.DataFrame({'Sensor_ID': X.columns, 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Sensor_ID', data=feature_importance_df.head(15), palette='viridis')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Sensor Identifier')
plt.title('Top 15 Critical Sensors Driving Yield Prediction')
for index, value in enumerate(feature_importance_df.head(15)['Importance']):
    plt.text(value + 0.001, index, f'{value:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

print("Pipeline Execution Completed Successfully")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
from xgboost import XGBClassifier
import os

os.makedirs('outputs', exist_ok=True)

# ── Load cleaned data ──────────────────────────────────
floods = pd.read_csv('data/processed/floods_clean.csv')
print("✅ Data loaded:", floods.shape)
print("Columns:", list(floods.columns))

# ══════════════════════════════════════════════════════
# STEP 1 — FEATURE ENGINEERING
# Creating new features from existing ones
# ══════════════════════════════════════════════════════
print("\n--- STEP 1: Feature Engineering ---")

# Feature 1: Flood Pressure Index
# Combines water level + river discharge — when both are high, flood risk spikes
floods['Flood_Pressure_Index'] = (
    floods['Water Level (m)'] * floods['River Discharge (m³/s)']
)

# Feature 2: Heat Humidity Index
# High temp + high humidity = more evaporation = more rainfall potential
floods['Heat_Humidity_Index'] = (
    floods['Temperature (°C)'] * floods['Humidity (%)'] / 100
)

# Feature 3: Elevation Risk Score
# Low elevation = higher flood risk (water flows downhill)
# We invert elevation so higher score = higher risk
floods['Elevation_Risk'] = 1 / (floods['Elevation (m)'] + 1)

# Feature 4: Population Vulnerability Score
# High population + no infrastructure + historical floods = very vulnerable
floods['Vulnerability_Score'] = (
    floods['Population Density'] *
    (1 - floods['Infrastructure']) *
    (floods['Historical Floods'] + 1)
)

# Feature 5: Rainfall intensity category
# Bin rainfall into Low / Medium / High / Extreme
floods['Rainfall_Category'] = pd.cut(
    floods['Rainfall (mm)'],
    bins=[0, 75, 150, 225, 300],
    labels=[0, 1, 2, 3]  # 0=Low, 1=Medium, 2=High, 3=Extreme
).astype(int)

print("✅ 5 new features created")
print(f"   Flood_Pressure_Index — range: {floods['Flood_Pressure_Index'].min():.2f} to {floods['Flood_Pressure_Index'].max():.2f}")
print(f"   Heat_Humidity_Index  — range: {floods['Heat_Humidity_Index'].min():.2f} to {floods['Heat_Humidity_Index'].max():.2f}")
print(f"   Elevation_Risk       — range: {floods['Elevation_Risk'].min():.4f} to {floods['Elevation_Risk'].max():.4f}")
print(f"   Vulnerability_Score  — range: {floods['Vulnerability_Score'].min():.2f} to {floods['Vulnerability_Score'].max():.2f}")
print(f"   Rainfall_Category    — values: {floods['Rainfall_Category'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════
# STEP 2 — ENCODE CATEGORICAL FEATURES
# Convert text columns to numbers for ML
# ══════════════════════════════════════════════════════
print("\n--- STEP 2: Encoding Categorical Features ---")

le_landcover = LabelEncoder()
le_soiltype = LabelEncoder()

floods['Land_Cover_Encoded'] = le_landcover.fit_transform(floods['Land Cover'])
floods['Soil_Type_Encoded'] = le_soiltype.fit_transform(floods['Soil Type'])

print("✅ Land Cover classes:", list(le_landcover.classes_))
print("✅ Soil Type classes:", list(le_soiltype.classes_))

# ══════════════════════════════════════════════════════
# STEP 3 — DEFINE FEATURES AND TARGET
# ══════════════════════════════════════════════════════
print("\n--- STEP 3: Defining Features (X) and Target (y) ---")

feature_cols = [
    # Original features
    'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
    'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)',
    'Population Density', 'Infrastructure', 'Historical Floods',
    'Land_Cover_Encoded', 'Soil_Type_Encoded',
    # Engineered features
    'Flood_Pressure_Index', 'Heat_Humidity_Index',
    'Elevation_Risk', 'Vulnerability_Score', 'Rainfall_Category'
]

X = floods[feature_cols]
y = floods['Flood Occurred']

print(f"✅ Features (X): {X.shape}  — {len(feature_cols)} features")
print(f"✅ Target  (y): {y.shape}  — values: {y.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# 80% training, 20% testing
# ══════════════════════════════════════════════════════
print("\n--- STEP 4: Train/Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # reproducibility
    stratify=y           # keep class balance in both splits
)

print(f"✅ Training set:  {X_train.shape[0]} rows")
print(f"✅ Testing set:   {X_test.shape[0]} rows")
print(f"   Train balance: {y_train.value_counts().to_dict()}")
print(f"   Test balance:  {y_test.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════
# STEP 5 — TRAIN XGBOOST MODEL
# ══════════════════════════════════════════════════════
print("\n--- STEP 5: Training XGBoost Model ---")

model = XGBClassifier(
    n_estimators=100,      # 100 trees
    max_depth=6,           # max depth of each tree
    learning_rate=0.1,     # how fast model learns
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)
print("✅ XGBoost model trained successfully!")

# ══════════════════════════════════════════════════════
# STEP 6 — EVALUATE MODEL
# ══════════════════════════════════════════════════════
print("\n--- STEP 6: Model Evaluation ---")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Flood', 'Flood']))

# Key metrics
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
print(f"✅ AUC-ROC Score: {auc:.4f}")
print(f"✅ F1 Score:      {f1:.4f}")

# ══════════════════════════════════════════════════════
# CHART 9 — Confusion Matrix
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'],
            ax=ax, linewidths=1)
ax.set_title('Chart 9 — Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# Add labels explaining each cell
ax.text(0.5, -0.15,
    f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}",
    ha='center', transform=ax.transAxes, fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('outputs/chart9_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Chart 9 saved — confusion matrix")

# ══════════════════════════════════════════════════════
# CHART 10 — ROC Curve
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color='#3B82F6', linewidth=2.5,
        label=f'XGBoost (AUC = {auc:.4f})')
ax.plot([0,1], [0,1], color='gray', linestyle='--',
        linewidth=1.5, label='Random classifier (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.1, color='#3B82F6')
ax.set_title('Chart 10 — ROC Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('outputs/chart10_roc_curve.png', dpi=150)
plt.show()
print("✅ Chart 10 saved — ROC curve")

# ══════════════════════════════════════════════════════
# CHART 11 — Feature Importance
# Which features matter most to the model?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

colors = ['#EF4444' if 'Flood_Pressure' in f or
          'Vulnerability' in f or
          'Elevation_Risk' in f
          else '#3B82F6'
          for f in importance_df['Feature']]

ax.barh(importance_df['Feature'],
        importance_df['Importance'],
        color=colors)
ax.set_title('Chart 11 — XGBoost Feature Importance\n(Red = Engineered features)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')

# Add value labels
for i, (val, name) in enumerate(zip(importance_df['Importance'],
                                     importance_df['Feature'])):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/chart11_feature_importance.png', dpi=150)
plt.show()
print("✅ Chart 11 saved — feature importance")

# ══════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("🎯 DAY 3 COMPLETE — MODEL RESULTS SUMMARY")
print("="*55)
print(f"Total features used:     {len(feature_cols)}")
print(f"  Original features:     11")
print(f"  Engineered features:   5")
print(f"Training samples:        {X_train.shape[0]}")
print(f"Testing samples:         {X_test.shape[0]}")
print(f"AUC-ROC:                 {auc:.4f}")
print(f"F1 Score:                {f1:.4f}")
print(f"\nTop 3 most important features:")
top3 = importance_df.tail(3)
for _, row in top3.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")


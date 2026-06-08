import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
from xgboost import XGBClassifier
import pickle, os

os.makedirs('outputs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ── Load rainfall data ─────────────────────────────────
rainfall = pd.read_csv('data/processed/rainfall_clean.csv')
print("✅ Rainfall loaded:", rainfall.shape)

# ══════════════════════════════════════════════════════
# STEP 1 — BUILD ROW-LEVEL FEATURES
# Each row = one subdivision × one year
# 4116 rows = 4116 training samples
# ══════════════════════════════════════════════════════
print("\n--- STEP 1: Feature Engineering (per year per subdivision) ---")

df = rainfall.copy()
df = df.sort_values(['SUBDIVISION', 'YEAR'])

# Feature 1: Monsoon intensity ratio
# How dominant is monsoon vs full year
df['monsoon_ratio'] = df['Jun-Sep'] / (df['ANNUAL'] + 1)

# Feature 2: 3-year rolling average rainfall
df['rolling_3yr'] = df.groupby('SUBDIVISION')['ANNUAL'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Feature 3: Year-over-year change
df['annual_change'] = df.groupby('SUBDIVISION')['ANNUAL'].transform(
    lambda x: x.diff().fillna(0)
)

# Feature 4: Dry season ratio
df['dry_ratio'] = df['Jan-Feb'] / (df['ANNUAL'] + 1)

# Feature 5: Pre-monsoon buildup (Mar-May)
df['premonsoon_ratio'] = df['Mar-May'] / (df['ANNUAL'] + 1)

# Feature 6: Post-monsoon (Oct-Dec)
df['postmonsoon_ratio'] = df['Oct-Dec'] / (df['ANNUAL'] + 1)

# Feature 7: Rainfall anomaly vs subdivision average
subdiv_mean = df.groupby('SUBDIVISION')['ANNUAL'].transform('mean')
df['rainfall_anomaly'] = df['ANNUAL'] - subdiv_mean

# Feature 8: Extreme month — max single month rainfall
month_cols = ['JAN','FEB','MAR','APR','MAY','JUN',
              'JUL','AUG','SEP','OCT','NOV','DEC']
df['max_month_rainfall'] = df[month_cols].max(axis=1)

# Feature 9: July intensity (peak monsoon month)
df['july_intensity'] = df['JUL'] / (df['ANNUAL'] + 1)

print(f"✅ Features engineered. Dataset shape: {df.shape}")

# ══════════════════════════════════════════════════════
# STEP 2 — CREATE TARGET LABEL
# High flood risk year = annual rainfall in top 25%
# for THAT subdivision (not global — avoids leakage)
# ══════════════════════════════════════════════════════
print("\n--- STEP 2: Creating Target Label ---")

# Label = 1 if that year's rainfall is in top 25%
# for that subdivision's historical distribution
df['High_Risk_Year'] = df.groupby('SUBDIVISION')['ANNUAL'].transform(
    lambda x: (x > x.quantile(0.75)).astype(int)
)

print("✅ Label distribution:")
print(df['High_Risk_Year'].value_counts())
print(df['High_Risk_Year'].value_counts(normalize=True).round(3) * 100)

# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════
print("\n--- STEP 3: Train/Test Split ---")

feature_cols = [
    'ANNUAL', 'Jun-Sep', 'monsoon_ratio', 'rolling_3yr',
    'annual_change', 'dry_ratio', 'premonsoon_ratio',
    'postmonsoon_ratio', 'rainfall_anomaly',
    'max_month_rainfall', 'july_intensity'
]

X = df[feature_cols]
y = df['High_Risk_Year']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train: {X_train.shape[0]} rows")
print(f"✅ Test:  {X_test.shape[0]} rows")
print(f"   Train balance: {y_train.value_counts().to_dict()}")
print(f"   Test balance:  {y_test.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════
# STEP 4 — BASELINE MODEL
# ══════════════════════════════════════════════════════
print("\n--- STEP 4: Baseline Model ---")

baseline = XGBClassifier(
    n_estimators=50, max_depth=3,
    learning_rate=0.1, random_state=42,
    verbosity=0, eval_metric='logloss'
)
baseline.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:,1])
base_f1  = f1_score(y_test, baseline.predict(X_test))
print(f"✅ Baseline AUC: {base_auc:.4f}")
print(f"✅ Baseline F1:  {base_f1:.4f}")

# ══════════════════════════════════════════════════════
# STEP 5 — IMPROVED MODEL
# ══════════════════════════════════════════════════════
print("\n--- STEP 5: Improved XGBoost ---")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)
f1  = f1_score(y_test, y_pred)

print(f"✅ Improved AUC: {auc:.4f}")
print(f"✅ Improved F1:  {f1:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal Year','High Risk Year']))

# ══════════════════════════════════════════════════════
# STEP 6 — CROSS VALIDATION
# ══════════════════════════════════════════════════════
print("\n--- STEP 6: 5-Fold Cross Validation ---")
cv = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"✅ CV AUC scores: {cv.round(4)}")
print(f"✅ Mean CV AUC:   {cv.mean():.4f} ± {cv.std():.4f}")

# ══════════════════════════════════════════════════════
# CHART 12 — Model Improvement Story
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
models_names = [
    f'Previous Baseline\n(Synthetic data)\nAUC=0.49',
    f'Baseline\n(Real rainfall)\nAUC={base_auc:.2f}',
    f'Improved\n(Engineered features)\nAUC={auc:.2f}'
]
aucs = [0.4880, base_auc, auc]
colors = ['#EF4444', '#F97316', '#22C55E']
bars = axes[0].bar(models_names, aucs,
                   color=colors, width=0.4, edgecolor='white')
axes[0].axhline(0.5, color='gray', linestyle='--',
                linewidth=1.5, label='Random (0.5)')
axes[0].set_ylim([0, 1.1])
axes[0].set_title('Model Improvement Journey',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('AUC-ROC Score')
axes[0].legend()
for bar, val in zip(bars, aucs):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 val + 0.03, f'{val:.3f}',
                 ha='center', fontsize=11, fontweight='bold')

# ROC curves
fpr_b, tpr_b, _ = roc_curve(y_test,
    baseline.predict_proba(X_test)[:,1])
fpr_i, tpr_i, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr_b, tpr_b, color='#F97316',
             linewidth=2, linestyle='--',
             label=f'Baseline (AUC={base_auc:.3f})')
axes[1].plot(fpr_i, tpr_i, color='#22C55E',
             linewidth=2.5,
             label=f'Improved (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1], color='gray',
             linestyle=':', label='Random (0.5)')
axes[1].fill_between(fpr_i, tpr_i, alpha=0.1, color='#22C55E')
axes[1].set_title('ROC Curve Comparison',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

fig.suptitle('Chart 12 — Full Model Improvement Story',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/chart12_improvement.png', dpi=150)
plt.show()
print("✅ Chart 12 saved")

# ══════════════════════════════════════════════════════
# CHART 13 — Feature Importance
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

colors = ['#22C55E' if f in ['rainfall_anomaly',
          'rolling_3yr','monsoon_ratio']
          else '#3B82F6'
          for f in imp_df['Feature']]

ax.barh(imp_df['Feature'], imp_df['Importance'],
        color=colors, edgecolor='white')
ax.set_title('Chart 13 — Feature Importance\n(Green = key engineered features)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, val in enumerate(imp_df['Importance']):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/chart13_feature_importance.png', dpi=150)
plt.show()
print("✅ Chart 13 saved")

# ══════════════════════════════════════════════════════
# CHART 14 — Confusion Matrix
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal','High Risk'],
            yticklabels=['Normal','High Risk'],
            ax=ax, linewidths=1,
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Chart 14 — Confusion Matrix (Improved)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall    = tp/(tp+fn) if (tp+fn) > 0 else 0
ax.text(0.5, -0.15,
    f"TN={tn}  FP={fp}  FN={fn}  TP={tp}"
    f"  |  Precision={precision:.2f}  Recall={recall:.2f}",
    ha='center', transform=ax.transAxes,
    fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('outputs/chart14_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Chart 14 saved")

# ══════════════════════════════════════════════════════
# SAVE MODEL + RISK SCORES FOR DAY 4 MAP
# ══════════════════════════════════════════════════════
with open('data/processed/flood_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Risk scores per subdivision for map
df['Risk_Probability'] = model.predict_proba(X)[:,1]
risk_map = df.groupby('SUBDIVISION').agg(
    avg_risk_prob     = ('Risk_Probability', 'mean'),
    avg_annual_rain   = ('ANNUAL', 'mean'),
    high_risk_years   = ('High_Risk_Year', 'sum'),
    total_years       = ('High_Risk_Year', 'count')
).reset_index()
risk_map['risk_pct'] = (
    risk_map['high_risk_years'] /
    risk_map['total_years'] * 100
).round(1)
risk_map.to_csv(
    'data/processed/subdivision_risk_scores.csv', index=False)

print("\n✅ Model saved")
print("✅ Risk scores saved for Day 4 map")
print(f"\nTop 5 highest risk subdivisions:")
print(risk_map.nlargest(5, 'avg_risk_prob')[
    ['SUBDIVISION','avg_annual_rain',
     'avg_risk_prob','risk_pct']].to_string(index=False))

# ══════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print(" DAY 3B COMPLETE — FINAL SUMMARY")
print("="*55)
print(f"Previous baseline AUC: 0.4880 (synthetic data)")
print(f"New baseline AUC:      {base_auc:.4f} (real data)")
print(f"Improved model AUC:    {auc:.4f}")
print(f"Total AUC improvement: +{auc - 0.4880:.4f}")
print(f"F1 Score:              {f1:.4f}")
print(f"CV Mean AUC:           {cv.mean():.4f} ± {cv.std():.4f}")
print(f"Training samples:      {X_train.shape[0]}")
print(f"Test samples:          {X_test.shape[0]}")
print(f"Features used:         {len(feature_cols)}")
print(f"\n🚀 Next: Day 4 — Folium Map Visualization")
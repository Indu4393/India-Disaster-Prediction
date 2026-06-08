import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load cleaned datasets ──────────────────────────────
floods = pd.read_csv('data/processed/floods_clean.csv')
rainfall = pd.read_csv('data/processed/rainfall_clean.csv')
earthquakes = pd.read_csv('data/processed/earthquakes_clean.csv')

# Fix earthquake datetime
earthquakes['Origin Time'] = pd.to_datetime(
    earthquakes['Origin Time'].str.replace(' IST', '', regex=False),
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
)
earthquakes['year'] = earthquakes['Origin Time'].dt.year
earthquakes['month'] = earthquakes['Origin Time'].dt.month

os.makedirs('outputs', exist_ok=True)
print("✅ Datasets loaded. Starting EDA...\n")

# ══════════════════════════════════════════════════════
# CHART 1 — Flood class balance (pie chart)
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
counts = floods['Flood Occurred'].value_counts()
colors = ['#EF4444', '#22C55E']
ax.pie(counts, labels=['Flood (1)', 'No Flood (0)'],
       autopct='%1.1f%%', colors=colors,
       startangle=90, textprops={'fontsize': 13})
ax.set_title('Chart 1 — Flood Class Balance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/chart1_class_balance.png', dpi=150)
plt.show()
print("✅ Chart 1 saved — class balance")

# ══════════════════════════════════════════════════════
# CHART 2 — Rainfall distribution (histogram)
# What it shows: how rainfall is spread — is it normal or skewed?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(floods['Rainfall (mm)'], bins=40,
        color='#3B82F6', edgecolor='white', alpha=0.85)
ax.axvline(floods['Rainfall (mm)'].mean(),
           color='red', linestyle='--', linewidth=2,
           label=f"Mean: {floods['Rainfall (mm)'].mean():.1f} mm")
ax.axvline(floods['Rainfall (mm)'].median(),
           color='orange', linestyle='--', linewidth=2,
           label=f"Median: {floods['Rainfall (mm)'].median():.1f} mm")
ax.set_title('Chart 2 — Rainfall Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Rainfall (mm)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/chart2_rainfall_distribution.png', dpi=150)
plt.show()
print("✅ Chart 2 saved — rainfall distribution")

# ══════════════════════════════════════════════════════
# CHART 3 — Rainfall vs Flood Occurred (boxplot)
# What it shows: do flood zones get more rain? 
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
flood_groups = [
    floods[floods['Flood Occurred'] == 0]['Rainfall (mm)'],
    floods[floods['Flood Occurred'] == 1]['Rainfall (mm)']
]
bp = ax.boxplot(flood_groups, labels=['No Flood (0)', 'Flood (1)'],
                patch_artist=True,
                boxprops=dict(facecolor='#BFDBFE'),
                medianprops=dict(color='red', linewidth=2))
ax.set_title('Chart 3 — Rainfall vs Flood Occurrence', fontsize=14, fontweight='bold')
ax.set_ylabel('Rainfall (mm)')
ax.set_xlabel('Flood Occurred')
plt.tight_layout()
plt.savefig('outputs/chart3_rainfall_vs_flood.png', dpi=150)
plt.show()
print("✅ Chart 3 saved — rainfall vs flood boxplot")

# ══════════════════════════════════════════════════════
# CHART 4 — Correlation heatmap (flood dataset)
# What it shows: which features are most related to flooding?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
numeric_cols = floods.select_dtypes(include=np.number)
corr = numeric_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Chart 4 — Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/chart4_correlation_heatmap.png', dpi=150)
plt.show()
print("✅ Chart 4 saved — correlation heatmap")

# ══════════════════════════════════════════════════════
# CHART 5 — Top 10 rainfall subdivisions (bar chart)
# What it shows: which regions get the most rain in India?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
top10 = rainfall.groupby('SUBDIVISION')['ANNUAL'].mean().nlargest(10)
colors_bar = ['#1E40AF'] * 3 + ['#3B82F6'] * 7
ax.barh(top10.index[::-1], top10.values[::-1], color=colors_bar[::-1])
ax.set_title('Chart 5 — Top 10 Highest Rainfall Subdivisions (Avg Annual)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Average Annual Rainfall (mm)')
for i, v in enumerate(top10.values[::-1]):
    ax.text(v + 20, i, f'{v:.0f}mm', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/chart5_top_rainfall_subdivisions.png', dpi=150)
plt.show()
print("✅ Chart 5 saved — top rainfall subdivisions")

# ══════════════════════════════════════════════════════
# CHART 6 — India rainfall trend over 100 years (line chart)
# What it shows: is India's rainfall increasing or decreasing?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
yearly_rain = rainfall.groupby('YEAR')['ANNUAL'].mean()
ax.plot(yearly_rain.index, yearly_rain.values,
        color='#3B82F6', linewidth=1.5, alpha=0.7)
# Rolling average to show trend
rolling = yearly_rain.rolling(10).mean()
ax.plot(rolling.index, rolling.values,
        color='#EF4444', linewidth=2.5,
        label='10-year rolling average')
ax.set_title('Chart 6 — India Annual Rainfall Trend (1901–2015)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Average Annual Rainfall (mm)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/chart6_rainfall_trend.png', dpi=150)
plt.show()
print("✅ Chart 6 saved — rainfall trend over 100 years")

# ══════════════════════════════════════════════════════
# CHART 7 — Monthly rainfall pattern (monsoon analysis)
# What it shows: which months get most rain — monsoon visible?
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
month_cols = ['JAN','FEB','MAR','APR','MAY','JUN',
              'JUL','AUG','SEP','OCT','NOV','DEC']
monthly_avg = rainfall[month_cols].mean()
colors_month = ['#93C5FD'] * 5 + ['#1D4ED8'] * 4 + ['#93C5FD'] * 3
bars = ax.bar(month_cols, monthly_avg.values, color=colors_month, edgecolor='white')
ax.set_title('Chart 7 — Average Monthly Rainfall Across India (Monsoon Pattern)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Average Rainfall (mm)')
ax.set_xlabel('Month')
# Highlight monsoon
ax.axvspan(4.5, 8.5, alpha=0.15, color='blue', label='Monsoon Season (Jun–Sep)')
ax.legend()
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/chart7_monthly_rainfall_monsoon.png', dpi=150)
plt.show()
print("✅ Chart 7 saved — monthly monsoon pattern")

# ══════════════════════════════════════════════════════
# CHART 8 — Earthquake magnitude distribution
# What it shows: most earthquakes are minor — power law distribution
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: histogram of magnitudes
axes[0].hist(earthquakes['Magnitude'], bins=30,
             color='#F97316', edgecolor='white', alpha=0.85)
axes[0].axvline(4.0, color='red', linestyle='--',
                linewidth=2, label='Magnitude 4.0 (felt by humans)')
axes[0].set_title('Magnitude Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Magnitude')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Right: earthquakes by month
monthly_eq = earthquakes['month'].value_counts().sort_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
axes[1].bar([month_names[m-1] for m in monthly_eq.index],
            monthly_eq.values, color='#F97316', edgecolor='white')
axes[1].set_title('Earthquakes by Month', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Count')

fig.suptitle('Chart 8 — Earthquake Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/chart8_earthquake_analysis.png', dpi=150)
plt.show()
print("✅ Chart 8 saved — earthquake analysis")

# ══════════════════════════════════════════════════════
# SUMMARY — Key findings from EDA
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("📊 KEY EDA FINDINGS — your interview talking points")
print("="*55)
print(f"1. Class balance: perfectly balanced at 50/50 — no SMOTE needed")
print(f"2. Avg rainfall in flood zones: {floods[floods['Flood Occurred']==1]['Rainfall (mm)'].mean():.1f}mm")
print(f"   Avg rainfall in non-flood zones: {floods[floods['Flood Occurred']==0]['Rainfall (mm)'].mean():.1f}mm")
print(f"3. Highest rainfall: {rainfall.groupby('SUBDIVISION')['ANNUAL'].mean().idxmax()}")
print(f"4. Monsoon (Jun-Sep) contributes: {(rainfall['Jun-Sep'].mean()/rainfall['ANNUAL'].mean()*100):.1f}% of annual rain")
print(f"5. Total earthquakes recorded: {len(earthquakes)}")
print(f"6. Avg earthquake magnitude: {earthquakes['Magnitude'].mean():.2f}")
print(f"7. Most seismically active month: {month_names[earthquakes['month'].value_counts().idxmax()-1]}")

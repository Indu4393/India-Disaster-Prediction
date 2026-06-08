import pandas as pd
import numpy as np
import folium
from folium import plugins
import json
import os

os.makedirs('outputs', exist_ok=True)

# ── Load data ──────────────────────────────────────────
risk_scores = pd.read_csv('data/processed/subdivision_risk_scores.csv')
rainfall = pd.read_csv('data/processed/rainfall_clean.csv')
earthquakes = pd.read_csv('data/processed/earthquakes_clean.csv')

# Fix earthquake datetime
earthquakes['Origin Time'] = pd.to_datetime(
    earthquakes['Origin Time'].str.replace(' IST','', regex=False),
    format='%Y-%m-%d %H:%M:%S', errors='coerce'
)
earthquakes['year'] = earthquakes['Origin Time'].dt.year

print("✅ Risk scores:", risk_scores.shape)
print("✅ Rainfall:", rainfall.shape)
print("✅ Earthquakes:", earthquakes.shape)
print("\nRisk scores sample:")
print(risk_scores.head())

# ══════════════════════════════════════════════════════
# STEP 1 — PREPARE RISK DATA
# ══════════════════════════════════════════════════════
print("\n--- STEP 1: Preparing Risk Data ---")

# Normalize risk probability to 0-100 score
risk_scores['risk_score'] = (
    risk_scores['avg_risk_prob'] * 100
).round(1)

# Risk category labels
def risk_category(score):
    if score >= 30:   return 'High Risk'
    elif score >= 25: return 'Medium Risk'
    else:             return 'Low Risk'

risk_scores['risk_category'] = risk_scores['risk_score'].apply(risk_category)

print("✅ Risk categories:")
print(risk_scores['risk_category'].value_counts())
print("\nTop 10 highest risk subdivisions:")
print(risk_scores.nlargest(10, 'risk_score')[
    ['SUBDIVISION','avg_annual_rain','risk_score','risk_category']
].to_string(index=False))

# ══════════════════════════════════════════════════════
# STEP 2 — SUBDIVISION COORDINATES
# Map each subdivision to lat/lon for markers
# ══════════════════════════════════════════════════════
print("\n--- STEP 2: Mapping Coordinates ---")

# Approximate center coordinates for each subdivision
subdivision_coords = {
    'ANDAMAN & NICOBAR ISLANDS': [11.7401, 92.6586],
    'ARUNACHAL PRADESH':         [28.2180, 94.7278],
    'ASSAM & MEGHALAYA':         [26.2006, 92.9376],
    'BIHAR':                     [25.0961, 85.3131],
    'CHHATTISGARH':              [21.2787, 81.8661],
    'COASTAL ANDHRA PRADESH':    [16.5062, 80.6480],
    'COASTAL KARNATAKA':         [13.1986, 75.0554],
    'EAST MADHYA PRADESH':       [23.4734, 80.4400],
    'EAST RAJASTHAN':            [26.4499, 74.6399],
    'EAST UTTAR PRADESH':        [26.3000, 82.8500],
    'GANGETIC WEST BENGAL':      [23.0000, 88.0000],
    'GUJARAT REGION':            [22.2587, 71.1924],
    'HARYANA DELHI & CHANDIGARH':[28.7041, 77.1025],
    'HIMACHAL PRADESH':          [31.1048, 77.1734],
    'JAMMU & KASHMIR':           [33.7782, 76.5762],
    'JHARKHAND':                 [23.6102, 85.2799],
    'KERALA':                    [10.8505, 76.2711],
    'KONKAN & GOA':              [15.4909, 73.8278],
    'LAKSHADWEEP':               [10.5667, 72.6417],
    'MADHYA MAHARASHTRA':        [18.9220, 75.7139],
    'MARATHWADA':                [19.3919, 76.1420],
    'MATATHWADA':                [19.3919, 76.1420],
    'NAGA MANI MIZO TRIPURA':    [25.4670, 93.8313],
    'NORTH INTERIOR KARNATAKA':  [15.3173, 75.7139],
    'ORISSA':                    [20.9517, 85.0985],
    'PUNJAB':                    [31.1471, 75.3412],
    'RAYALASEEMA':               [14.4426, 79.1288],
    'SAURASHTRA & KUTCH':        [22.6924, 71.5222],
    'SHWB & SIKKIM':             [27.3314, 88.6138],
    'SOUTH INTERIOR KARNATAKA':  [12.9716, 77.5946],
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': [26.7271, 88.3953],
    'TAMIL NADU':                [11.1271, 78.6569],
    'TELANGANA':                 [17.1232, 79.2088],
    'UTTARAKHAND':               [30.0668, 79.0193],
    'VIDARBHA':                  [20.7002, 78.6979],
    'WEST MADHYA PRADESH':       [23.0000, 77.0000],
    'WEST RAJASTHAN':            [27.0238, 70.0000],
    'WEST UTTAR PRADESH':        [29.0000, 78.0000],
    'ANDHRA PRADESH':            [15.9129, 79.7400],
    'NORTH ANDHRA PRADESH':      [18.0000, 83.0000],
}

# Add coordinates to risk_scores
risk_scores['lat'] = risk_scores['SUBDIVISION'].map(
    lambda x: subdivision_coords.get(x, [None,None])[0]
)
risk_scores['lon'] = risk_scores['SUBDIVISION'].map(
    lambda x: subdivision_coords.get(x, [None,None])[1]
)

# Drop rows with no coordinates
risk_mapped = risk_scores.dropna(subset=['lat','lon'])
print(f"✅ Mapped {len(risk_mapped)} / {len(risk_scores)} subdivisions")

# ══════════════════════════════════════════════════════
# STEP 3 — BUILD BASE MAP
# ══════════════════════════════════════════════════════
print("\n--- STEP 3: Building Folium Map ---")

# India centered map
m = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles='CartoDB positron'
)

# ══════════════════════════════════════════════════════
# LAYER 1 — Flood Risk Circles per Subdivision
# ══════════════════════════════════════════════════════
flood_layer = folium.FeatureGroup(name='🌊 Flood Risk Zones')

def risk_color(category):
    return {'High Risk': '#DC2626',
            'Medium Risk': '#F97316',
            'Low Risk': '#22C55E'}.get(category, '#94A3B8')

def risk_radius(score):
    return max(20000, score * 8000)

for _, row in risk_mapped.iterrows():
    # Outer circle — risk zone
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=risk_radius(row['risk_score']),
        color=risk_color(row['risk_category']),
        fill=True,
        fill_opacity=0.25,
        weight=2,
        popup=folium.Popup(
            f"""
            <div style='font-family:Arial; width:200px'>
            <b style='color:{risk_color(row['risk_category'])}'>
            {row['risk_category']}</b><br>
            <b>{row['SUBDIVISION']}</b><br>
            <hr style='margin:4px 0'>
            Risk Score: <b>{row['risk_score']:.1f}/100</b><br>
            Avg Annual Rain: <b>{row['avg_annual_rain']:.0f}mm</b><br>
            High Risk Years: <b>{row['high_risk_years']:.0f}</b>
            out of {row['total_years']:.0f}<br>
            Risk %: <b>{row['risk_pct']}%</b>
            </div>
            """,
            max_width=220
        ),
        tooltip=f"{row['SUBDIVISION']} | {row['risk_category']} | Score: {row['risk_score']:.1f}"
    ).add_to(flood_layer)

    # Center dot
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,
        color=risk_color(row['risk_category']),
        fill=True,
        fill_opacity=1.0,
        weight=1
    ).add_to(flood_layer)

flood_layer.add_to(m)

# ══════════════════════════════════════════════════════
# LAYER 2 — Earthquake Markers
# ══════════════════════════════════════════════════════
print("Adding earthquake layer...")
eq_layer = folium.FeatureGroup(name='🔴 Earthquake Events')

# Sample top 200 significant earthquakes (magnitude >= 4)
significant_eq = earthquakes[
    earthquakes['Magnitude'] >= 4.0
].head(200)

for _, row in significant_eq.iterrows():
    mag = row['Magnitude']
    # Size scales with magnitude
    radius = (mag - 3) * 4

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=max(3, radius),
        color='#7C3AED',
        fill=True,
        fill_opacity=0.6,
        weight=1,
        popup=folium.Popup(
            f"""
            <div style='font-family:Arial; width:180px'>
            <b>🔴 Earthquake</b><br>
            <hr style='margin:4px 0'>
            Magnitude: <b>{mag}</b><br>
            Depth: <b>{row['Depth']}km</b><br>
            Location: {row['Location'][:40]}<br>
            Year: <b>{row['year']}</b>
            </div>
            """,
            max_width=200
        ),
        tooltip=f"M{mag} earthquake"
    ).add_to(eq_layer)

eq_layer.add_to(m)

# ══════════════════════════════════════════════════════
# LAYER 3 — Top 5 Highest Risk Markers (highlighted)
# ══════════════════════════════════════════════════════
top5_layer = folium.FeatureGroup(name='⭐ Top 5 Highest Risk Zones')

top5 = risk_mapped.nlargest(5, 'risk_score')
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(
            f"""
            <div style='font-family:Arial; width:200px'>
            <b style='color:#DC2626'>⭐ Rank #{rank} Highest Risk</b><br>
            <b>{row['SUBDIVISION']}</b><br>
            <hr style='margin:4px 0'>
            Risk Score: <b>{row['risk_score']:.1f}/100</b><br>
            Avg Rainfall: <b>{row['avg_annual_rain']:.0f}mm</b><br>
            Risk %: <b>{row['risk_pct']}%</b>
            </div>
            """,
            max_width=220
        ),
        tooltip=f"#{rank} {row['SUBDIVISION']}",
        icon=folium.Icon(
            color='red',
            icon='warning-sign',
            prefix='glyphicon'
        )
    ).add_to(top5_layer)

top5_layer.add_to(m)

# ══════════════════════════════════════════════════════
# STEP 4 — ADD MAP FEATURES
# ══════════════════════════════════════════════════════

# Layer control — toggle layers on/off
folium.LayerControl(collapsed=False).add_to(m)

# Legend
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background-color: white; padding: 15px; border-radius: 10px;
     border: 2px solid #ccc; font-family: Arial; font-size: 13px;
     box-shadow: 3px 3px 6px rgba(0,0,0,0.2);">
<b style="font-size:14px">🗺️ Disaster Risk Map</b><br>
<b>India Flood Risk Predictor</b><br>
<hr style="margin: 6px 0">
<b>Flood Risk Zones:</b><br>
🔴 High Risk (&gt;30% score)<br>
🟠 Medium Risk (25-30%)<br>
🟢 Low Risk (&lt;25%)<br>
<hr style="margin: 6px 0">
🟣 Earthquakes (M≥4.0)<br>
⭐ Top 5 Highest Risk<br>
<hr style="margin: 6px 0">
<i style="font-size:11px">Click any circle for details</i>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Title
title_html = """
<div style="position: fixed; top: 10px; left: 50%;
     transform: translateX(-50%); z-index: 1000;
     background-color: white; padding: 10px 20px;
     border-radius: 8px; border: 2px solid #3B82F6;
     font-family: Arial; text-align: center;
     box-shadow: 3px 3px 6px rgba(0,0,0,0.2);">
<b style="font-size:16px; color:#1D4ED8">
🌊 India Disaster Risk Predictor</b><br>
<span style="font-size:11px; color:#666">
XGBoost Model | AUC: 0.9954 | 115 Years of Data
</span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ══════════════════════════════════════════════════════
# STEP 5 — SAVE MAP
# ══════════════════════════════════════════════════════
map_path = 'outputs/india_disaster_risk_map.html'
m.save(map_path)
print(f"\n✅ Map saved: {map_path}")
print(f"   Open this file in your browser!")

# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("🎯 DAY 4 COMPLETE — MAP SUMMARY")
print("="*55)
print(f"Total subdivisions mapped:  {len(risk_mapped)}")
print(f"Earthquake events shown:    {len(significant_eq)}")
print(f"Map layers:                 3 (Flood Risk, Earthquakes, Top 5)")
print(f"Interactive features:       Popups, Tooltips, Layer Control")
print(f"\nRisk breakdown:")
print(risk_mapped['risk_category'].value_counts().to_string())
print(f"\nTop 5 highest risk zones:")
print(risk_mapped.nlargest(5,'risk_score')[
    ['SUBDIVISION','risk_score','risk_category']
].to_string(index=False))
print(f"\n📂 Open: outputs/india_disaster_risk_map.html")
print(f"🚀 Next: Day 5 — Streamlit Dashboard")
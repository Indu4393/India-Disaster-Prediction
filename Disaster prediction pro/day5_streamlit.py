import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import pickle
import os

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="India Disaster Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1D4ED8;
        text-align: center;
        padding: 10px 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1D4ED8, #3B82F6);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .risk-high   { color: #DC2626; font-weight: 700; }
    .risk-medium { color: #F97316; font-weight: 700; }
    .risk-low    { color: #16A34A; font-weight: 700; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E3A5F;
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
        margin: 20px 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    floods    = pd.read_csv('data/processed/floods_clean.csv')
    rainfall  = pd.read_csv('data/processed/rainfall_clean.csv')
    eq        = pd.read_csv('data/processed/earthquakes_clean.csv')
    risk      = pd.read_csv('data/processed/subdivision_risk_scores.csv')

    eq['Origin Time'] = pd.to_datetime(
        eq['Origin Time'].str.replace(' IST','', regex=False),
        format='%Y-%m-%d %H:%M:%S', errors='coerce'
    )
    eq['year']  = eq['Origin Time'].dt.year
    eq['month'] = eq['Origin Time'].dt.month

    risk['risk_score'] = (risk['avg_risk_prob'] * 100).round(1)
    risk['risk_category'] = risk['risk_score'].apply(
        lambda s: 'High Risk' if s >= 30
        else 'Medium Risk' if s >= 25
        else 'Low Risk'
    )
    return floods, rainfall, eq, risk

@st.cache_resource
def load_model():
    with open('data/processed/flood_risk_model.pkl','rb') as f:
        return pickle.load(f)

floods, rainfall, earthquakes, risk_scores = load_data()
model = load_model()

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
             width=120)
    st.markdown("### 🌊 India Disaster Risk")
    st.markdown("**ML-powered flood & earthquake risk prediction**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "🗺️ Risk Map",
         "📊 EDA Charts",
         "🤖 ML Model",
         "🔮 Live Predictor",
         "📋 About Project"]
    )
    st.divider()
    st.markdown("**Model Performance**")
    st.metric("AUC-ROC",  "0.9954", "+0.507 vs baseline")
    st.metric("F1 Score",  "0.9431")
    st.metric("Accuracy", "97.0%")
    st.metric("CV Score", "0.9927 ± 0.004")

# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-title">🌊 India Disaster Risk Predictor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">XGBoost ML Model | 115 Years of NDMA Data | AUC: 0.9954</div>',
                unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("10,000", "Flood Records"),
        ("4,116",  "Rainfall Records"),
        ("2,719",  "Earthquake Events"),
        ("115",    "Years of Data"),
        ("0.9954", "Model AUC"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Project story
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">📖 Project Story</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        This project predicts **flood risk zones** across India using
        115 years of historical rainfall data from NDMA.

        **The problem:** Floods cause ₹1 lakh crore damage in India
        annually. Early warning saves lives and property.

        **The approach:**
        - Loaded 4 real datasets (flood, rainfall, earthquake, GeoJSON)
        - Engineered 11 temporal features from 115 years of data
        - Built XGBoost classifier — improved AUC from **0.49 → 0.9954**
        - Visualized predictions on interactive India map

        **Key finding:** `rainfall_anomaly` (how much a year deviates
        from the subdivision's historical average) is the strongest
        predictor — more important than absolute rainfall.
        """)

    with col_r:
        st.markdown('<div class="section-header">🎯 Model Journey</div>',
                    unsafe_allow_html=True)

        journey_data = pd.DataFrame({
            'Model': ['Synthetic\nBaseline', 'Real Data\nBaseline',
                      'Improved\nXGBoost'],
            'AUC':   [0.488, 0.995, 0.9954],
            'Color': ['#EF4444', '#F97316', '#22C55E']
        })

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(journey_data['Model'], journey_data['AUC'],
                      color=journey_data['Color'],
                      width=0.4, edgecolor='white')
        ax.axhline(0.5, color='gray', linestyle='--',
                   linewidth=1.5, label='Random (0.5)')
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('AUC-ROC Score')
        ax.set_title('Model Improvement Journey', fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, journey_data['AUC']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.03, f'{val:.3f}',
                    ha='center', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Risk summary table
    st.markdown('<div class="section-header">🗂️ Risk Summary by Subdivision</div>',
                unsafe_allow_html=True)

    display_df = risk_scores[[
        'SUBDIVISION','avg_annual_rain',
        'risk_score','risk_category',
        'high_risk_years','total_years','risk_pct'
    ]].sort_values('risk_score', ascending=False).reset_index(drop=True)

    display_df.columns = ['Subdivision','Avg Rain (mm)',
                          'Risk Score','Category',
                          'High Risk Years','Total Years','Risk %']

    def color_category(val):
        if val == 'High Risk':
            return 'color: #DC2626; font-weight: bold'
        elif val == 'Medium Risk':
            return 'color: #F97316; font-weight: bold'
        return 'color: #16A34A; font-weight: bold'

    
    st.dataframe(
        display_df.style.map(
    color_category, subset=['Category']
        ).format({
            'Avg Rain (mm)': '{:.0f}',
            'Risk Score': '{:.1f}',
            'Risk %': '{:.1f}%'
        }),
        use_container_width=True,
        height=400
    )

# ══════════════════════════════════════════════════════
# PAGE 2 — RISK MAP
# ══════════════════════════════════════════════════════
elif page == "🗺️ Risk Map":
    st.markdown('<div class="section-header">🗺️ Interactive India Disaster Risk Map</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        show_floods = st.checkbox("🌊 Show Flood Risk Zones", value=True)
    with col2:
        show_eq = st.checkbox("🔴 Show Earthquakes", value=True)
    with col3:
        show_top5 = st.checkbox("⭐ Show Top 5 Risk Zones", value=True)

    # Build map
    m = folium.Map(location=[20.5937, 78.9629],
                   zoom_start=5, tiles='CartoDB positron')

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
        'NAGA MANI MIZO TRIPURA':    [25.4670, 93.8313],
        'NORTH INTERIOR KARNATAKA':  [15.3173, 75.7139],
        'ORISSA':                    [20.9517, 85.0985],
        'PUNJAB':                    [31.1471, 75.3412],
        'RAYALASEEMA':               [14.4426, 79.1288],
        'SAURASHTRA & KUTCH':        [22.6924, 71.5222],
        'SOUTH INTERIOR KARNATAKA':  [12.9716, 77.5946],
        'SUB HIMALAYAN WEST BENGAL & SIKKIM': [26.7271, 88.3953],
        'TAMIL NADU':                [11.1271, 78.6569],
        'TELANGANA':                 [17.1232, 79.2088],
        'UTTARAKHAND':               [30.0668, 79.0193],
        'VIDARBHA':                  [20.7002, 78.6979],
        'WEST MADHYA PRADESH':       [23.0000, 77.0000],
        'WEST RAJASTHAN':            [27.0238, 70.0000],
        'WEST UTTAR PRADESH':        [29.0000, 78.0000],
        'GUJARAT REGION':            [22.2587, 71.1924],
    }

    def risk_color(cat):
        return {'High Risk':'#DC2626',
                'Medium Risk':'#F97316',
                'Low Risk':'#22C55E'}.get(cat,'#94A3B8')

    risk_scores['lat'] = risk_scores['SUBDIVISION'].map(
        lambda x: subdivision_coords.get(x,[None,None])[0])
    risk_scores['lon'] = risk_scores['SUBDIVISION'].map(
        lambda x: subdivision_coords.get(x,[None,None])[1])
    risk_mapped = risk_scores.dropna(subset=['lat','lon'])

    if show_floods:
        for _, row in risk_mapped.iterrows():
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=max(20000, row['risk_score'] * 8000),
                color=risk_color(row['risk_category']),
                fill=True, fill_opacity=0.25, weight=2,
                popup=folium.Popup(
                    f"<b>{row['SUBDIVISION']}</b><br>"
                    f"Risk: <b>{row['risk_category']}</b><br>"
                    f"Score: <b>{row['risk_score']:.1f}</b><br>"
                    f"Rain: <b>{row['avg_annual_rain']:.0f}mm</b>",
                    max_width=200),
                tooltip=f"{row['SUBDIVISION']} | {row['risk_score']:.1f}"
            ).add_to(m)
            folium.CircleMarker(
                location=[row['lat'],row['lon']],
                radius=5,
                color=risk_color(row['risk_category']),
                fill=True, fill_opacity=1
            ).add_to(m)

    if show_eq:
        sig_eq = earthquakes[earthquakes['Magnitude'] >= 4.0].head(150)
        for _, row in sig_eq.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=max(3, (row['Magnitude']-3)*4),
                color='#7C3AED', fill=True,
                fill_opacity=0.6, weight=1,
                tooltip=f"M{row['Magnitude']}"
            ).add_to(m)

    if show_top5:
        top5 = risk_mapped.nlargest(5,'risk_score')
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            folium.Marker(
                location=[row['lat'],row['lon']],
                tooltip=f"#{rank} {row['SUBDIVISION']}",
                icon=folium.Icon(color='red',
                                 icon='warning-sign',
                                 prefix='glyphicon')
            ).add_to(m)

    st_folium(m, width=1200, height=550)

# ══════════════════════════════════════════════════════
# PAGE 3 — EDA CHARTS
# ══════════════════════════════════════════════════════
elif page == "📊 EDA Charts":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌊 Flood Data", "🌧 Rainfall Data", "🌍 Earthquake Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Class balance
            fig, ax = plt.subplots(figsize=(5,4))
            counts = floods['Flood Occurred'].value_counts()
            ax.pie(counts, labels=['Flood','No Flood'],
                   autopct='%1.1f%%',
                   colors=['#EF4444','#22C55E'])
            ax.set_title('Class Balance', fontweight='bold')
            st.pyplot(fig); plt.close()

        with col2:
            # Correlation heatmap
            fig, ax = plt.subplots(figsize=(6,5))
            num = floods.select_dtypes(include=np.number)
            mask = np.triu(np.ones_like(num.corr(), dtype=bool))
            sns.heatmap(num.corr(), mask=mask, annot=True,
                        fmt='.2f', cmap='RdYlGn',
                        center=0, ax=ax,
                        annot_kws={'size':7})
            ax.set_title('Feature Correlations', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Monthly pattern
            fig, ax = plt.subplots(figsize=(6,4))
            month_cols = ['JAN','FEB','MAR','APR','MAY','JUN',
                          'JUL','AUG','SEP','OCT','NOV','DEC']
            monthly = rainfall[month_cols].mean()
            colors  = ['#93C5FD']*5 + ['#1D4ED8']*4 + ['#93C5FD']*3
            ax.bar(month_cols, monthly, color=colors)
            ax.axvspan(4.5,8.5,alpha=0.1,color='blue',
                       label='Monsoon')
            ax.set_title('Monthly Rainfall Pattern', fontweight='bold')
            ax.set_ylabel('Avg Rainfall (mm)')
            ax.legend()
            for i,(m,v) in enumerate(zip(month_cols,monthly)):
                ax.text(i,v+2,f'{v:.0f}',ha='center',fontsize=7)
            st.pyplot(fig); plt.close()

        with col2:
            # Rainfall trend
            fig, ax = plt.subplots(figsize=(6,4))
            yearly = rainfall.groupby('YEAR')['ANNUAL'].mean()
            ax.plot(yearly.index, yearly.values,
                    color='#93C5FD', linewidth=1, alpha=0.7)
            ax.plot(yearly.index,
                    yearly.rolling(10).mean(),
                    color='#EF4444', linewidth=2.5,
                    label='10yr rolling avg')
            ax.set_title('India Rainfall Trend 1901-2015',
                         fontweight='bold')
            ax.set_ylabel('Avg Annual Rainfall (mm)')
            ax.legend()
            st.pyplot(fig); plt.close()

        # Top subdivisions
        fig, ax = plt.subplots(figsize=(10,4))
        top10 = rainfall.groupby('SUBDIVISION')['ANNUAL'].mean().nlargest(10)
        ax.barh(top10.index[::-1], top10.values[::-1], color='#1D4ED8')
        ax.set_title('Top 10 Highest Rainfall Subdivisions',
                     fontweight='bold')
        ax.set_xlabel('Avg Annual Rainfall (mm)')
        for i,v in enumerate(top10.values[::-1]):
            ax.text(v+20,i,f'{v:.0f}mm',va='center',fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.hist(earthquakes['Magnitude'], bins=30,
                    color='#F97316', edgecolor='white')
            ax.axvline(4.0, color='red', linestyle='--',
                       label='M4.0 (felt by humans)')
            ax.set_title('Magnitude Distribution', fontweight='bold')
            ax.set_xlabel('Magnitude')
            ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            monthly_eq = earthquakes['month'].value_counts().sort_index()
            month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
            ax.bar([month_names[m-1] for m in monthly_eq.index],
                   monthly_eq.values, color='#F97316')
            ax.set_title('Earthquakes by Month', fontweight='bold')
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════
# PAGE 4 — ML MODEL
# ══════════════════════════════════════════════════════
elif page == "🤖 ML Model":
    st.markdown('<div class="section-header">🤖 XGBoost Model Details</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (label, val, delta) in zip(
        [col1,col2,col3,col4],
        [("AUC-ROC","0.9954","+0.507"),
         ("F1 Score","0.9431","+0.943"),
         ("Precision","0.93",""),
         ("Recall","0.96","")]
    ):
        col.metric(label, val, delta)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Model Improvement Journey**")
        fig, ax = plt.subplots(figsize=(6,3.5))
        names = ['Synthetic\nBaseline','Real Data\nBaseline',
                 'Improved\nXGBoost']
        aucs  = [0.488, 0.995, 0.9954]
        colors= ['#EF4444','#F97316','#22C55E']
        bars  = ax.bar(names, aucs, color=colors,
                       width=0.4, edgecolor='white')
        ax.axhline(0.5,color='gray',linestyle='--',
                   linewidth=1.5,label='Random')
        ax.set_ylim([0,1.15])
        ax.legend()
        for bar,val in zip(bars,aucs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    val+0.03,f'{val:.3f}',
                    ha='center',fontsize=11,fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("**Feature Importance**")
        feature_cols = [
            'ANNUAL','Jun-Sep','monsoon_ratio','rolling_3yr',
            'annual_change','dry_ratio','premonsoon_ratio',
            'postmonsoon_ratio','rainfall_anomaly',
            'max_month_rainfall','july_intensity'
        ]
        # Use placeholder importances from your actual run
        importances = [0.071,0.060,0.019,0.053,0.085,
                       0.022,0.028,0.022,0.583,0.039,0.019]
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(6,5))
        colors = ['#22C55E' if f in ['rainfall_anomaly','rolling_3yr','monsoon_ratio']
                  else '#3B82F6' for f in imp_df['Feature']]
        ax.barh(imp_df['Feature'], imp_df['Importance'],
                color=colors, edgecolor='white')
        ax.set_title('Feature Importance\n(Green = engineered)',
                     fontweight='bold')
        for i,val in enumerate(imp_df['Importance']):
            ax.text(val+0.005,i,f'{val:.3f}',
                    va='center',fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # CV scores
    st.markdown("**5-Fold Cross Validation Results**")
    cv_data = pd.DataFrame({
        'Fold': ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],
        'AUC':  [0.9889,  0.9948,  0.9978,  0.9962,  0.9861]
    })
    fig, ax = plt.subplots(figsize=(8,3))
    bars = ax.bar(cv_data['Fold'], cv_data['AUC'],
                  color='#3B82F6', edgecolor='white')
    ax.axhline(0.9927, color='red', linestyle='--',
               linewidth=2, label='Mean AUC: 0.9927')
    ax.set_ylim([0.97,1.01])
    ax.set_ylabel('AUC Score')
    ax.set_title('5-Fold CV — Consistent performance across all folds',
                 fontweight='bold')
    ax.legend()
    for bar,val in zip(bars,cv_data['AUC']):
        ax.text(bar.get_x()+bar.get_width()/2,
                val+0.0005,f'{val:.4f}',
                ha='center',fontsize=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown('<div class="section-header">🔮 Live Flood Risk Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Enter rainfall values for a subdivision and get instant flood risk prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        annual      = st.slider("Annual Rainfall (mm)", 200, 5000, 1500)
        jun_sep     = st.slider("Monsoon Rainfall Jun-Sep (mm)", 100, 3000, 900)
        annual_chg  = st.slider("Year-over-year change (mm)", -500, 500, 50)
    with col2:
        rolling_3yr = st.slider("3-year rolling avg (mm)", 200, 5000, 1400)
        anomaly     = st.slider("Rainfall anomaly vs avg (mm)", -800, 800, 100)
        max_month   = st.slider("Max single month rainfall (mm)", 50, 1000, 350)
    with col3:
        monsoon_r   = st.slider("Monsoon ratio", 0.3, 0.9, 0.6)
        july_int    = st.slider("July intensity", 0.05, 0.4, 0.2)
        dry_r       = st.slider("Dry season ratio", 0.01, 0.1, 0.03)
        premonsoon_r= st.slider("Pre-monsoon ratio", 0.02, 0.15, 0.06)
        postmonsoon_r=st.slider("Post-monsoon ratio", 0.05, 0.25, 0.12)

    if st.button("🔮 Predict Flood Risk", type="primary", use_container_width=True):
        input_data = np.array([[
            annual, jun_sep, monsoon_r, rolling_3yr,
            annual_chg, dry_r, premonsoon_r,
            postmonsoon_r, anomaly, max_month, july_int
        ]])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)

        risk_cat = "🔴 HIGH RISK" if prob > 0.6 else \
                   "🟠 MEDIUM RISK" if prob > 0.4 else \
                   "🟢 LOW RISK"
        color = "#DC2626" if prob > 0.6 else \
                "#F97316" if prob > 0.4 else "#16A34A"

        with col_res1:
            st.metric("Risk Probability", f"{prob*100:.1f}%")
        with col_res2:
            st.metric("Prediction", "High Risk Year" if pred==1 else "Normal Year")
        with col_res3:
            st.markdown(f"""
            <div style='background:{color};color:white;
            padding:15px;border-radius:10px;text-align:center;
            font-size:1.3rem;font-weight:700'>
            {risk_cat}
            </div>""", unsafe_allow_html=True)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Risk'], [prob], color=color, height=0.4)
        ax.barh(['Risk'], [1-prob], left=[prob],
                color='#E5E7EB', height=0.4)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim([0,1])
        ax.set_title(f'Risk Probability: {prob*100:.1f}%',
                     fontweight='bold')
        ax.set_xlabel('Probability')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(f"""
        **What this means:**
        Based on the rainfall values you entered, this subdivision
        has a **{prob*100:.1f}% probability** of experiencing a
        high-risk flood year. The model was trained on 115 years
        of historical NDMA data with AUC 0.9954.
        ⚠️ This is a research tool — always consult official
        disaster management authorities for real alerts.
        """)

# ══════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "📋 About Project":
    st.markdown('<div class="section-header">📋 About This Project</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Goal
        Build an end-to-end ML system that predicts flood risk
        zones across India using historical rainfall data,
        visualized on an interactive geospatial map.

        ### 📂 Datasets Used
        | Dataset | Source | Rows |
        |---|---|---|
        | Flood Risk India | Kaggle | 10,000 |
        | Rainfall 1901-2015 | Kaggle/NDMA | 4,116 |
        | Indian Earthquakes | Kaggle/USGS | 2,719 |
        | India Districts GeoJSON | GitHub | 820 districts |

        ### 🛠️ Tech Stack
        | Tool | Purpose |
        |---|---|
        | Pandas / NumPy | Data cleaning, EDA |
        | Matplotlib / Seaborn | Static visualization |
        | XGBoost | ML classification |
        | Folium | Interactive map |
        | Streamlit | Web dashboard |
        | Pickle | Model persistence |
        """)

    with col2:
        st.markdown("""
        ### 🔬 Key Findings
        1. **Rainfall anomaly** is the strongest predictor (importance: 0.583)
           — relative deviation matters more than absolute rainfall
        2. India's annual rainfall has **declined ~150mm since the 1960s peak**
        3. **Coastal Karnataka** receives the highest rainfall (3,414mm avg)
        4. **July** delivers the most rainfall (347mm avg) — peak monsoon month
        5. **April** is India's most seismically active month (310 events)
        6. Synthetic data gave AUC 0.49; real NDMA data gave AUC 0.9954

        ### 📈 Model Performance
        | Metric | Score |
        |---|---|
        | AUC-ROC | 0.9954 |
        | F1 Score | 0.9431 |
        | Accuracy | 97.0% |
        | Precision | 0.93 |
        | Recall | 0.96 |
        | CV Mean AUC | 0.9927 ± 0.004 |

        ### ⚠️ Responsible AI Note
        This tool is for research and educational purposes.
        Predictions should not replace official disaster
        management warnings from NDMA or IMD.
        """)
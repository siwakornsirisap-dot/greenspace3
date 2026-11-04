import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from html import escape
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Bangkok Urban Heat & Green Space Priority Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Navy, Green, and Professional Design
st.markdown("""
<style>
    .main { background-color: #0d1b2a; color: #e0e1dd; }
    .stApp { background-color: #0b132b; color: #ffffff; }
    h1, h2, h3 { color: #52b788 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .metric-card {
        background-color: #1c2541;
        border-left: 5px solid #52b788;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .metric-title { font-size: 0.9rem; color: #a3b18a; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #ffffff; }
    .stAlert { background-color: #1c2541; border: 1px solid #3a506b; color: #e0e1dd; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Preprocessing Functions
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    data_dir = Path(__file__).resolve().parent / "data"
    grid_df = pd.read_csv(data_dir / "bangkok_250m_grid_data.csv")
    district_df = pd.read_csv(data_dir / "bangkok_district_summary.csv")

    if 'system:index' in grid_df.columns:
        grid_df.rename(columns={'system:index': 'grid_id'}, inplace=True)
    else:
        grid_df['grid_id'] = grid_df.index.astype(str)
    
    # Extract Longitude and Latitude from .geo JSON column if present
    if '.geo' in grid_df.columns:
        def parse_coords(geo_str):
            try:
                data = json.loads(geo_str)
                coords = data['coordinates']
                return coords[0], coords[1] # lon, lat
            except:
                return np.nan, np.nan
        grid_df['longitude'], grid_df['latitude'] = zip(*grid_df['.geo'].apply(parse_coords))
    
    # Clean null values and extreme outliers
    grid_df = grid_df.dropna(subset=['LST', 'NDVI', 'population', 'latitude', 'longitude']).copy()
    grid_df = grid_df[(grid_df['LST'] >= 15) & (grid_df['LST'] <= 50)]
    grid_df = grid_df[(grid_df['NDVI'] >= -0.2) & (grid_df['NDVI'] <= 1.0)]
    grid_df = grid_df[grid_df['population'] >= 0]
    
    # ------------------------------------------
    # Priority Scoring Methodology (Stage 4)
    # ------------------------------------------
    # Normalize factors between 0 and 1
    lst_min, lst_max = grid_df['LST'].min(), grid_df['LST'].max()
    ndvi_min, ndvi_max = grid_df['NDVI'].min(), grid_df['NDVI'].max()
    pop_min, pop_max = grid_df['population'].min(), grid_df['population'].max()
    
    grid_df['LST_norm'] = (grid_df['LST'] - lst_min) / (lst_max - lst_min + 1e-6)
    grid_df['NDVI_norm'] = (ndvi_max - grid_df['NDVI']) / (ndvi_max - ndvi_min + 1e-6) # Inverse: low vegetation = higher priority
    grid_df['Pop_norm'] = (grid_df['population'] - pop_min) / (pop_max - pop_min + 1e-6)
    
    # Transparent Weighted Scoring (LST: 40%, NDVI: 35%, Population Exposure: 25%)
    grid_df['Priority_Score'] = (
        0.40 * grid_df['LST_norm'] +
        0.35 * grid_df['NDVI_norm'] +
        0.25 * grid_df['Pop_norm']
    ) * 100.0
    
    # Categorize into Priority Levels
    quantiles = grid_df['Priority_Score'].quantile([0.25, 0.50, 0.75]).values
    def get_category(score):
        if score >= quantiles[2]: return 'Very High'
        elif score >= quantiles[1]: return 'High'
        elif score >= quantiles[0]: return 'Medium'
        else: return 'Low'
    
    grid_df['Priority_Category'] = grid_df['Priority_Score'].apply(get_category)
    
    # Clean district summaries
    district_df = district_df.dropna(subset=['LST', 'NDVI']).copy()
    if 'ADM2_NAME' in district_df.columns:
        district_df.rename(columns={'ADM2_NAME': 'District'}, inplace=True)
    elif 'system:index' in district_df.columns:
        district_df.rename(columns={'system:index': 'District'}, inplace=True)
        
    return grid_df, district_df

@st.cache_resource
def train_random_forest_model(grid_df):
    features = ['longitude', 'latitude', 'NDVI', 'population']
    X = grid_df[features]
    y = grid_df['LST']
    
    # 80/20 Train-Test Split with reproducible seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "importances": dict(zip(features, model.feature_importances_))
    }
    
    grid_df['LST_Predicted'] = model.predict(X)
    return model, metrics, grid_df

def render_horizontal_bars(data, label_column, value_column, color, number_format):
    rows = data[[label_column, value_column]].dropna().to_dict('records')
    max_value = max((abs(float(row[value_column])) for row in rows), default=0) or 1
    row_html = []
    for row in rows:
        label = escape(str(row[label_column]))
        value = float(row[value_column])
        width = max(0, min(100, abs(value) / max_value * 100))
        row_html.append(
            f"<div class='row'><div class='label' title='{label}'>{label}</div>"
            f"<div class='track'><div class='bar' style='width:{width:.2f}%;background:{color}'></div></div>"
            f"<div class='value'>{format(value, number_format)}</div></div>"
        )
    chart_html = f"""<!doctype html><html><head><meta charset='utf-8'><style>
        body{{font-family:Arial,sans-serif;background:transparent;color:#e0e1dd;margin:8px}}
        .row{{display:flex;align-items:center;gap:10px;height:28px;margin:3px 0}}
        .label{{flex:0 0 175px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}}
        .track{{flex:1;background:#1c2541;height:14px;border-radius:4px;overflow:hidden}}
        .bar{{height:100%;border-radius:4px}}
        .value{{flex:0 0 58px;text-align:right;font-size:13px;font-variant-numeric:tabular-nums}}
        </style></head><body>{''.join(row_html)}</body></html>"""
    st.components.v1.html(chart_html, height=max(80, len(rows) * 34 + 18), scrolling=False)


# Load datasets and model
try:
    grid_df, district_df = load_and_preprocess_data()
    rf_model, ml_metrics, grid_df = train_random_forest_model(grid_df)
except Exception as e:
    st.error(f"Error loading project datasets: {e}")
    st.stop()

# ==========================================
# 3. Sidebar Navigation & Global Filters
# ==========================================
st.sidebar.title("🌳 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Homepage & Executive Overview", "Interactive Priority Map", "Tree Planting Recommendations", "District Comparison", "Model Performance & Machine Learning"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Details")
st.sidebar.info("""
**Bangkok Urban Heat & Green Space Priority Dashboard**
- **Spatial Resolution:** 250m × 250m Grid
- **Satellite Data:** Landsat 8/9 (2020–2025)
- **Target Users:** Urban Planners & BMA Policy Researchers
""")

# ==========================================
# PAGE 1: Homepage & Executive Overview
# ==========================================
if page == "Homepage & Executive Overview":
    st.title("Bangkok Urban Heat & Green Space Priority Dashboard")
    st.markdown("##### *An Academic Decision-Support System for Targeted Urban Greening Interventions*")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Land Surface Temp (LST)</div>
            <div class="metric-value">{grid_df['LST'].mean():.1f} °C</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Vegetation (NDVI)</div>
            <div class="metric-value">{grid_df['NDVI'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Analytical Grid Cells</div>
            <div class="metric-value">{len(grid_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        very_high_count = len(grid_df[grid_df['Priority_Category'] == 'Very High'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Very High Priority Zones</div>
            <div class="metric-value">{very_high_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Executive Summary & Key Findings")
    st.write("""
    Urban heat island effects present significant environmental challenges across Bangkok. This academic prototype integrates multi-year Landsat thermal satellite observations, NDVI vegetation indexes, and WorldPop population density models to answer key spatial planning questions:
    
    - **Thermal Distribution:** Satellite-derived Land Surface Temperature (LST) reveals pronounced heat concentration in dense, paved urban centers.
    - **Vegetation Deficits:** Low vegetation coverage coincides directly with commercial and residential corridors lacking public tree canopy.
    - **Priority Targeting:** Weighted priority scoring highlights areas where high surface temperature, low vegetation, and high population density intersect.
    """)

    st.warning("**Scientific & Planning Disclaimer:** High-priority grid cells represent zones for further field investigation. The system does not assess underground utilities, land ownership, tree species selection, or guaranteed microclimate cooling effects.")

# ==========================================
# PAGE 2: Interactive Priority Map
# ==========================================
elif page == "Interactive Priority Map":
    st.title("🗺️ Interactive Urban Heat & Green Space Priority Map")
    st.markdown("Explore heat exposure, vegetation deficits, and prioritized greening intervention zones.")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        category_filter = st.multiselect(
            "Filter priority categories:",
            options=['Very High', 'High', 'Medium', 'Low'],
            default=['Very High', 'High']
        )
    with col_filter2:
        max_display = st.slider("Maximum map markers:", min_value=100, max_value=2000, value=500, step=100)

    filtered_df = grid_df[grid_df['Priority_Category'].isin(category_filter)].copy()
    # Match the original dashboard's deterministic random sampling so markers
    # stay spatially dispersed instead of forming lines from adjacent grid cells.
    map_df = filtered_df.sample(n=min(max_display, len(filtered_df)), random_state=42) if not filtered_df.empty else filtered_df
    st.caption(f"Showing {len(map_df):,} map markers across the selected priority categories ({len(filtered_df):,} matching grid cells). The table and CSV include all matching cells.")

    m = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="OpenStreetMap")
    color_map = {'Very High': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
    for _, row in map_df.iterrows():
        popup_text = f"""
        <div style="min-width:220px">
          <b>Grid cell:</b> {escape(str(row['grid_id']))}<br>
          <b>Latitude:</b> {row['latitude']:.6f}<br>
          <b>Longitude:</b> {row['longitude']:.6f}<br>
          <hr style="margin:5px 0">
          <b>Priority:</b> {escape(str(row['Priority_Category']))}<br>
          <b>Score:</b> {row['Priority_Score']:.2f}/100<br>
          <b>Observed LST:</b> {row['LST']:.2f} °C (40%)<br>
          <b>NDVI:</b> {row['NDVI']:.3f} (inverted, 35%)<br>
          <b>Population density:</b> {row['population']:.3f} persons/grid (25%)<br>
          <b>Predicted LST:</b> {row['LST_Predicted']:.2f} °C
        </div>
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=4,
            color=color_map.get(row['Priority_Category'], 'blue'),
            fill_color=color_map.get(row['Priority_Category'], 'blue'),
            fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=280)
        ).add_to(m)
    st.components.v1.html(m.get_root().render(), height=600, scrolling=False)

    st.markdown("### 📍 Priority locations and grid details")
    st.caption("Coordinates use WGS84 decimal degrees. The priority score combines normalized LST (40%), inverted NDVI (35%), and population density (25%).")
    detail_columns = ['grid_id', 'latitude', 'longitude', 'Priority_Category', 'Priority_Score', 'LST', 'NDVI', 'population', 'LST_Predicted']
    detail_df = filtered_df[detail_columns].sort_values('Priority_Score', ascending=False).rename(columns={
        'grid_id': 'Grid ID', 'latitude': 'Latitude', 'longitude': 'Longitude',
        'Priority_Category': 'Priority level', 'Priority_Score': 'Priority score',
        'LST': 'Observed LST (°C)', 'NDVI': 'NDVI',
        'population': 'Population density', 'LST_Predicted': 'Predicted LST (°C)'
    })
    st.dataframe(
        detail_df, use_container_width=True, hide_index=True,
        column_config={
            'Latitude': st.column_config.NumberColumn(format='%.6f'),
            'Longitude': st.column_config.NumberColumn(format='%.6f'),
            'Priority score': st.column_config.NumberColumn(format='%.2f'),
            'Observed LST (°C)': st.column_config.NumberColumn(format='%.2f'),
            'Predicted LST (°C)': st.column_config.NumberColumn(format='%.2f'),
            'NDVI': st.column_config.NumberColumn(format='%.3f'),
            'Population density': st.column_config.NumberColumn(format='%.3f')
        }
    )
    st.download_button(
        'Download these priority locations (CSV)',
        data=detail_df.to_csv(index=False).encode('utf-8-sig'),
        file_name='bangkok_priority_locations.csv', mime='text/csv'
    )

# ==========================================
# PAGE 3: Tree Planting Recommendations
# ==========================================
elif page == "Tree Planting Recommendations":
    st.title("🌱 Tree Planting Recommendations")
    st.markdown("Use the heat and vegetation priority score to shortlist a grid cell, then estimate a planting plan from the **actual plantable area** and site conditions.")
    st.info("The 250 m grid identifies an area to investigate; it does not identify a verified planting site. Enter the measured plantable area below. The count is a planning estimate based on Bangkok guidance, not an AI or field-survey result.")

    candidate_df = grid_df.sort_values("Priority_Score", ascending=False).head(1000).copy()
    candidate_df["location_label"] = candidate_df.apply(
        lambda row: f"{row['grid_id']} · {row['Priority_Category']} · score {row['Priority_Score']:.1f} · {row['latitude']:.5f}, {row['longitude']:.5f}",
        axis=1
    )
    selected_label = st.selectbox("Select a priority grid cell (top 1,000 by score):", candidate_df["location_label"].tolist())
    selected = candidate_df.loc[candidate_df["location_label"] == selected_label].iloc[0]

    col_site, col_area, col_width = st.columns(3)
    with col_site:
        site_type = st.selectbox("Site setting:", ["Open sunny area", "Beside buildings / partial shade", "Sidewalk or median", "Waterfront"])
    with col_area:
        plantable_area = st.number_input("Measured plantable area (m²):", min_value=1.0, max_value=50000.0, value=100.0, step=10.0)
    with col_width:
        sidewalk_width = st.number_input("Sidewalk / median width (m, if relevant):", min_value=0.0, max_value=100.0, value=4.0, step=0.5)

    species_catalog = {
        "Open sunny area": [
            ("เสลา", "Lagerstroemia loudonii", 4, 6, "แดดเต็มวัน; คู่มือ กทม. แนะนำสำหรับพื้นที่ว่าง"),
            ("แคนา", "Dolichandrone serrulata", 4, 6, "แดดเต็มวัน; คู่มือ กทม. แนะนำสำหรับพื้นที่ว่าง"),
            ("ปีบทอง", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "แดดเต็มวัน; คู่มือ กทม. แนะนำสำหรับพื้นที่ว่าง"),
            ("คอร์เดีย", "ชื่อภาษาไทยตามคู่มือ กทม.", 3, 5, "แดดเต็มวัน; ระยะปลูกถี่กว่าไม้ตัวเลือกอื่นในคู่มือ")
        ],
        "Beside buildings / partial shade": [
            ("พิกุล", "ชื่อภาษาไทยตามคู่มือ กทม.", 5, 6, "พื้นที่ใกล้อาคาร; คู่มือ กทม. ระบุระยะปลูก 5–6 ม."),
            ("ลำดวน", "ชื่อภาษาไทยตามคู่มือ กทม.", 5, 6, "พื้นที่ใกล้อาคาร; คู่มือ กทม. ระบุระยะปลูก 5–6 ม."),
            ("รวงผึ้ง", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "พื้นที่ใกล้อาคาร; คู่มือ กทม. ระบุระยะปลูก 4–6 ม."),
            ("มันปู", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "พื้นที่ใกล้อาคาร; คู่มือ กทม. ระบุระยะปลูก 4–6 ม.")
        ],
        "Sidewalk or median": [
            ("คอร์เดีย", "ชื่อภาษาไทยตามคู่มือ กทม.", 3, 5, "คู่มือ กทม. แนะนำสำหรับทางเท้า/เกาะกลาง; ตรวจสายไฟและความกว้างจริง"),
            ("เสลา", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "คู่มือ กทม. แนะนำสำหรับทางเท้า/เกาะกลาง; ตรวจสายไฟและความกว้างจริง"),
            ("ปีบทอง", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "คู่มือ กทม. แนะนำสำหรับทางเท้า/เกาะกลาง; ตรวจสายไฟและความกว้างจริง"),
            ("พิกุล", "ชื่อภาษาไทยตามคู่มือ กทม.", 5, 6, "คู่มือ กทม. แนะนำสำหรับทางเท้า/เกาะกลาง; ตรวจสายไฟและความกว้างจริง")
        ],
        "Waterfront": [
            ("มันปู", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "คู่มือ กทม. ระบุสำหรับบริเวณริมน้ำ; ตรวจระดับน้ำท่วมและดิน"),
            ("ลำดวน", "ชื่อภาษาไทยตามคู่มือ กทม.", 5, 6, "คู่มือ กทม. ระบุสำหรับบริเวณริมน้ำ; ตรวจระดับน้ำท่วมและดิน"),
            ("สังทำ", "ชื่อภาษาไทยตามคู่มือ กทม.", 5, 6, "คู่มือ กทม. ระบุสำหรับบริเวณริมน้ำ; ตรวจระดับน้ำท่วมและดิน"),
            ("สารภี", "ชื่อภาษาไทยตามคู่มือ กทม.", 4, 6, "คู่มือ กทม. ระบุสำหรับบริเวณริมน้ำ; ตรวจระดับน้ำท่วมและดิน")
        ]
    }
    choices = species_catalog[site_type]
    species_label = st.selectbox("Candidate species (choose one dominant species per street/plan):", [item[0] for item in choices])
    species = choices[[item[0] for item in choices].index(species_label)]
    spacing_min, spacing_max = species[2], species[3]
    if site_type == "Sidewalk or median":
        # BMA public-space criteria specify 4–8 m spacing for sidewalks/medians.
        spacing_min, spacing_max = 4, 8

    st.markdown("### Estimate for the selected location")
    score_col, temp_col, veg_col, pop_col = st.columns(4)
    score_col.metric("Priority", str(selected["Priority_Category"]), f"{selected['Priority_Score']:.1f}/100")
    temp_col.metric("Surface temperature", f"{selected['LST']:.1f} °C")
    veg_col.metric("NDVI", f"{selected['NDVI']:.3f}")
    pop_col.metric("Population density value", f"{selected['population']:.2f}")
    st.caption(f"Grid {selected['grid_id']} center: {selected['latitude']:.6f}, {selected['longitude']:.6f}. This coordinate is a grid center, not a surveyed planting point.")

    if site_type == "Sidewalk or median" and sidewalk_width < 3:
        st.error("Bangkok guidance says sidewalks under 3 m wide should not receive trees. The tree estimate is set to 0 until a suitable wider planting strip is verified; consider low planting that does not obstruct pedestrian access.")
        min_trees = max_trees = 0
    else:
        # Conservative area/spacing envelope; the area must be measured on site.
        min_trees = max(0, int(np.floor(plantable_area / (spacing_max ** 2))))
        max_trees = max(0, int(np.floor(plantable_area / (spacing_min ** 2))))
        if max_trees == 0:
            max_trees = 1
        st.success(f"Estimated capacity: **{min_trees}–{max_trees} trees** of **{species[0]}** in {plantable_area:,.0f} m², using {spacing_min}–{spacing_max} m spacing.")

    st.markdown(f"**Why this species:** {species[4]} The dashboard uses priority score to help rank candidate sites; it does not infer tree suitability from LST/NDVI alone.")
    st.caption("Count formula: floor(plantable area ÷ spacing²), using the selected species' spacing range in Bangkok's Million Trees public planting manual. It assumes roughly square planting layout, so boundaries, paths, roots, utilities, setbacks, canopy overlap and access can reduce the count.")
    if site_type == "Sidewalk or median":
        st.warning("For Bangkok sidewalks, the cited criteria recommend tree planting only when width is at least 3 m, and 4–8 m spacing depending on species. Check overhead wires and access to homes before planting.")

    plan_df = pd.DataFrame([{
        "Grid ID": selected["grid_id"], "Latitude (grid center)": selected["latitude"], "Longitude (grid center)": selected["longitude"],
        "Priority category": selected["Priority_Category"], "Priority score": round(float(selected["Priority_Score"]), 2),
        "Observed LST (°C)": round(float(selected["LST"]), 2), "NDVI": round(float(selected["NDVI"]), 3),
        "Site setting": site_type, "Species": species[0], "Species name reference": species[1],
        "Plantable area (m²)": plantable_area, "Spacing min (m)": spacing_min, "Spacing max (m)": spacing_max,
        "Estimated trees min": min_trees, "Estimated trees max": max_trees, "Estimate status": "Preliminary; field verification required"
    }])
    st.download_button("Download this preliminary planting plan (CSV)", data=plan_df.to_csv(index=False).encode("utf-8-sig"), file_name="preliminary_tree_planting_plan.csv", mime="text/csv")
    st.markdown("**References:** [Bangkok Million Trees planting manual (species and spacing by setting)](https://greener.bangkok.go.th/wp-content/uploads/2024/06/PlantTreeManual.pdf) · [Bangkok planting criteria (sidewalk width and access)](https://budget.bangkok.go.th/main/upload/2020/04/27/A20200427093716.pdf)")

# ==========================================
# PAGE 4: District Comparison
# ==========================================
elif page == "District Comparison":
    st.title("📊 Bangkok District Comparisons")
    st.markdown("Side-by-side district summary analysis derived from multi-year spatial aggregations.")

    st.subheader("Top 20 districts by average Land Surface Temperature (°C)")
    lst_chart = district_df.sort_values(by='LST', ascending=False).head(20)
    render_horizontal_bars(lst_chart, 'District', 'LST', '#e76f51', '.2f')

    st.subheader("20 districts with the lowest average NDVI")
    ndvi_chart = district_df.sort_values(by='NDVI', ascending=True).head(20)
    render_horizontal_bars(ndvi_chart, 'District', 'NDVI', '#52b788', '.3f')

# PAGE 4: Model Performance & Machine Learning
# ==========================================
elif page == "Model Performance & Machine Learning":
    st.title("🤖 Machine Learning Model Performance")
    st.markdown("##### *Random Forest Regression for Land Surface Temperature Estimation*")
    st.markdown("---")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Mean Absolute Error (MAE)", f"{ml_metrics['MAE']:.3f} °C")
    with col_m2:
        st.metric("Root Mean Squared Error (RMSE)", f"{ml_metrics['RMSE']:.3f} °C")
    with col_m3:
        st.metric("R² Variance Score", f"{ml_metrics['R2']:.3f}")
        
    st.markdown("### 📊 Feature Importance Breakdown")
    importance_df = pd.DataFrame({
        'Feature': list(ml_metrics['importances'].keys()),
        'Importance': list(ml_metrics['importances'].values())
    }).sort_values(by='Importance', ascending=True)
    
    render_horizontal_bars(importance_df, 'Feature', 'Importance', '#52b788', '.3f')
    st.markdown("### ⚠️ Methodological & Spatial Limitations")
    st.info("""
    - **Random Split Leakage:** Standard 80/20 train-test splits on spatial datasets can overestimate model performance due to spatial autocorrelation between adjacent 250m grid cells.
    - **Surface vs Air Temperature:** Satellite thermal sensors measure skin/surface temperature (LST), which fluctuates more dramatically than ambient air temperature.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from html import escape
import folium
from folium.plugins import HeatMap
import pydeck as pdk
from gee_satellite import get_satellite_screening, initialize_earth_engine
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
    st.markdown("Click a priority-map point to get an automatic, satellite-based preliminary planting screen for that grid cell.")
    st.info("No area, site type, or species inputs are needed. The AI model estimates surface temperature and GEE screens heat, vegetation, and land cover. Tree candidates and the capacity range are then checked against Bangkok planting guidance; they are not ML predictions because the project has no verified records of species planted at each site or successful planting outcomes.")

    map_filter_col, map_limit_col = st.columns([1, 1])
    with map_filter_col:
        tree_categories = st.multiselect(
            "Filter priority categories:",
            options=["Very High", "High", "Medium", "Low"],
            default=["Very High", "High"],
            key="tree_map_categories"
        )
    with map_limit_col:
        tree_map_limit = st.slider("Maximum map markers:", 100, 2000, 1000, 100, key="tree_map_limit")

    matching_cells = grid_df[grid_df["Priority_Category"].isin(tree_categories)].copy()
    tree_map_df = (
        matching_cells.sample(n=min(tree_map_limit, len(matching_cells)), random_state=42)
        if not matching_cells.empty else matching_cells
    ).reset_index(drop=True)
    tree_color_map = {
        "Very High": [230, 57, 70, 220], "High": [244, 162, 97, 220],
        "Medium": [255, 209, 102, 220], "Low": [82, 183, 136, 220]
    }
    tree_map_df["marker_color"] = tree_map_df["Priority_Category"].map(tree_color_map)
    st.caption(f"Click a marker to inspect that grid cell. Showing {len(tree_map_df):,} markers from {len(matching_cells):,} matching cells.")

    if tree_map_df.empty:
        st.warning("No grid cells match the selected priority categories.")
        st.stop()

    tree_layer = pdk.Layer(
        "ScatterplotLayer", data=tree_map_df, id="priority-cells",
        get_position="[longitude, latitude]", get_radius=130,
        get_fill_color="marker_color", get_line_color=[50, 50, 50, 220],
        line_width_min_pixels=1, stroked=True, filled=True,
        pickable=True, auto_highlight=True, radius_min_pixels=4
    )
    tree_deck = pdk.Deck(
        layers=[tree_layer],
        initial_view_state=pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=10, pitch=0),
        tooltip={"text": "Grid {grid_id}\nPriority: {Priority_Category} ({Priority_Score} / 100)\nLST: {LST} °C\nNDVI: {NDVI}\nPopulation: {population}"},
        map_style="light"
    )
    tree_map_event = st.pydeck_chart(
        tree_deck, key="tree_priority_map", on_select="rerun",
        selection_mode="single-object", height=580,
        alt="Clickable Bangkok priority map for selecting a grid cell to inspect for tree planting"
    )
    selected_objects = tree_map_event.selection.objects.get("priority-cells", [])
    selected_grid_id = selected_objects[0].get("grid_id") if selected_objects else None
    selected_on_current_map = (
        tree_map_df[tree_map_df["grid_id"].astype(str) == str(selected_grid_id)]
        if selected_grid_id is not None else tree_map_df.iloc[0:0]
    )
    if not selected_on_current_map.empty:
        selected_matches = grid_df[grid_df["grid_id"].astype(str) == str(selected_grid_id)]
        selected = selected_matches.iloc[0] if not selected_matches.empty else tree_map_df.iloc[0]
    else:
        selected = tree_map_df.sort_values("Priority_Score", ascending=False).iloc[0]
        st.caption("No marker selected yet; showing the highest-priority visible cell as an example. Click another marker to update it.")

    st.markdown("### 🛰️ Satellite screening (Google Earth Engine)")
    gee_ready, gee_message = initialize_earth_engine()
    satellite = None
    if gee_ready:
        try:
            with st.spinner("กำลังคำนวณค่าภาพดาวเทียมรอบจุดที่เลือกจาก Earth Engine..."):
                satellite = get_satellite_screening(float(selected["latitude"]), float(selected["longitude"]))
            sat_lst, sat_ndvi = st.columns(2)
            if satellite["lst_c"] is not None:
                sat_lst.metric("Landsat median surface temperature (last 3 years)", f"{satellite['lst_c']:.1f} °C")
            else:
                sat_lst.metric("Landsat surface temperature", "No clear pixels")
            if satellite["ndvi"] is not None:
                sat_ndvi.metric("Sentinel-2 hot-season NDVI", f"{satellite['ndvi']:.3f}")
            else:
                sat_ndvi.metric("Sentinel-2 NDVI", "No clear pixels")
            st.caption(f"Matched hot seasons: {satellite['hot_season_start']} to {satellite['hot_season_end']} (March–May each year) · Landsat scenes: {satellite['landsat_scenes']} · Sentinel-2 scenes: {satellite['sentinel_scenes']}. Landsat thermal pixels are coarse; the 250 m grid is a screening zone, not a surveyed plot.")
        except Exception as exc:
            st.warning(f"ดึงข้อมูลภาพดาวเทียมจาก Earth Engine ไม่สำเร็จ: {exc}")
    else:
        st.warning(f"ยังไม่ได้เชื่อมต่อ Earth Engine: {gee_message}. ตั้งค่า secrets ตาม README เพื่อเปิดใช้การดึงค่าภาพดาวเทียมอัตโนมัติ")

    st.markdown("### 🌳 Automatic planting screen")
    st.caption("Select a map point; GEE compares three completed hot seasons for heat/vegetation and screens recent 10 m land cover. Cloud/shadow and low-confidence pixels are excluded. The 250 m grid remains a screening zone, not a planting parcel.")

    score_col, temp_col, veg_col, pop_col = st.columns(4)
    score_col.metric("Priority", str(selected["Priority_Category"]), f"{selected['Priority_Score']:.1f}/100")
    temp_col.metric("Grid source LST", f"{selected['LST']:.1f} °C")
    veg_col.metric("Grid source NDVI", f"{selected['NDVI']:.3f}")
    pop_col.metric("Population density value", f"{selected['population']:.2f}")
    if pd.notna(selected.get("LST_Predicted")):
        st.caption(f"Random Forest AI estimate for this grid: {selected['LST_Predicted']:.1f} °C surface temperature. This estimates heat exposure; it does not predict a tree species or planting count.")
    st.caption(f"Grid {selected['grid_id']} center: {selected['latitude']:.6f}, {selected['longitude']:.6f}. Satellite summary covers an approximately 250 m cell, not a surveyed planting plot.")

    min_trees = max_trees = None
    species_recommendation = "Unavailable"
    open_area_m2 = None
    landcover_area = satellite.get("landcover_area_m2", {}) if satellite is not None else {}
    if landcover_area:
        open_area_m2 = float(landcover_area.get("grass", 0) + landcover_area.get("bare", 0))
        tree_cover_m2 = float(landcover_area.get("trees", 0))
        water_m2 = float(landcover_area.get("water", 0) + landcover_area.get("flooded_vegetation", 0))
        land_col, cover_col, water_col = st.columns(3)
        land_col.metric("Confident open-land proxy", f"{open_area_m2:,.0f} m²", "grass + bare; confidence ≥ 0.60")
        cover_col.metric("Existing tree-cover class", f"{tree_cover_m2:,.0f} m²")
        water_col.metric("Water / flooded class", f"{water_m2:,.0f} m²")

        confident_coverage = satellite.get("landcover_confident_coverage_pct")
        if confident_coverage is None or confident_coverage < 50:
            st.warning(f"พื้นที่กริดที่ Dynamic World จำแนกได้ด้วยความมั่นใจ ≥ 0.60 มีเพียง {confident_coverage:.1f}% ของกริด จึงยังไม่คำนวณจำนวนต้นจากพื้นที่ที่ข้อมูลครอบคลุมไม่พอ" if confident_coverage is not None else "ยังคำนวณสัดส่วนพื้นที่ที่จำแนกได้มั่นใจไม่ได้ จึงยังไม่ประมาณจำนวนต้น")
        elif open_area_m2 >= 36:
            # The Bangkok planting manual lists 4–6 m spacing for these species.
            min_trees = int(np.floor(open_area_m2 / (6 ** 2)))
            max_trees = int(np.floor(open_area_m2 / (4 ** 2)))
            species_recommendation = "เสลา, แคนา, ปีบทอง (ตัวเลือกสำหรับพื้นที่รับแดดเต็มวัน)"
            st.success(f"**ข้อเสนอเบื้องต้น:** พื้นที่เปิดที่ดาวเทียมจำแนกได้อาจรองรับประมาณ **{min_trees:,}–{max_trees:,} ต้น** ที่ระยะปลูก 4–6 ม. หากพื้นที่ดังกล่าวเป็นพื้นที่ปลูกจริงและได้รับอนุญาต")
            st.markdown("**ชนิดไม้ที่เข้ากับเงื่อนไขพื้นที่เปิดรับแดด:** เสลา, แคนา และปีบทอง เป็นตัวเลือกที่ กทม. ระบุสำหรับพื้นที่แดดเต็มวัน ภาพดาวเทียมยังยืนยันร่มเงารายจุด ดิน สายไฟ ท่อใต้ดิน หรือกรรมสิทธิ์ไม่ได้ จึงไม่จัดอันดับว่า species ใดดีที่สุด")
        else:
            st.warning("GEE ไม่พบพื้นที่ grass/bare มากพอสำหรับคำนวณความจุปลูกแบบ 4–6 ม. ในเซลล์นี้ จึงยังไม่แนะนำจำนวนต้น หากเป็นทางเท้าหรือเกาะกลาง ระบบภาพ 10 ม. ระบุความกว้างและแนวปลูกไม่ได้")

        confidence = satellite.get("landcover_confidence")
        confidence_text = f"{confidence:.2f}" if confidence is not None else "ไม่พร้อมใช้งาน"
        coverage_text = f"{confident_coverage:.1f}%" if confident_coverage is not None else "ไม่พร้อมใช้งาน"
        st.caption(f"Dynamic World {satellite.get('landcover_start')} to {satellite.get('landcover_end')} · scenes: {satellite.get('landcover_scenes', 0)} · median top-class confidence: {confidence_text} · grid area retained after confidence filter: {coverage_text}. Class areas count only pixels with top-class confidence ≥ 0.60. Grass/bare is a land-cover proxy, not verified plantable land. Produced for Dynamic World by Google with National Geographic Society and World Resources Institute.")
        st.warning("**อ่านเป็นค่าคัดกรอง ไม่ใช่จำนวนที่อนุมัติปลูกได้:** land cover ไม่บอกเขตที่ดินสาธารณะ ทางเดิน สายไฟ ท่อ การเว้นระยะอาคาร หรือจุดปลูกจริง พื้นที่และจำนวนจริงอาจน้อยกว่าค่านี้มาก ต้องตรวจหน้างานก่อนวางแผนปลูก")
    elif satellite is not None:
        st.warning("GEE ดึงอุณหภูมิ/NDVI ได้ แต่ยังไม่มีภาพ land-cover สำหรับจุดนี้ จึงไม่สร้างจำนวนต้นหรือชนิดไม้โดยเดา")
    else:
        st.warning("ยังไม่ได้ผลวิเคราะห์จาก GEE สำหรับจุดนี้ จึงยังไม่แนะนำจำนวนหรือชนิดต้นไม้ ตรวจการเชื่อมต่อ Earth Engine แล้วลองเปิดหน้านี้อีกครั้ง")

    plan_df = pd.DataFrame([{
        "Grid ID": selected["grid_id"], "Latitude (grid center)": selected["latitude"], "Longitude (grid center)": selected["longitude"],
        "Priority category": selected["Priority_Category"], "Priority score": round(float(selected["Priority_Score"]), 2),
        "LST (°C)": round(float(satellite["lst_c"]), 2) if satellite is not None and satellite.get("lst_c") is not None else None,
        "Sentinel-2 NDVI": round(float(satellite["ndvi"]), 3) if satellite is not None and satellite.get("ndvi") is not None else None,
        "Potential open surface (m²)": round(open_area_m2, 1) if open_area_m2 is not None else None,
        "Confident land-cover coverage (%)": round(float(satellite["landcover_confident_coverage_pct"]), 1) if satellite is not None and satellite.get("landcover_confident_coverage_pct") is not None else None,
        "Dynamic World median top-class confidence": round(float(satellite["landcover_confidence"]), 3) if satellite is not None and satellite.get("landcover_confidence") is not None else None,
        "Automatic species shortlist": species_recommendation,
        "Estimated capacity min": min_trees, "Estimated capacity max": max_trees,
        "Estimate status": "Satellite screening only; verify ownership and planting constraints in field"
    }])
    st.download_button("Download this preliminary planting plan (CSV)", data=plan_df.to_csv(index=False).encode("utf-8-sig"), file_name="preliminary_tree_planting_plan.csv", mime="text/csv")
    st.markdown("**References:** [Bangkok Million Trees planting manual (species and spacing)](https://greener.bangkok.go.th/wp-content/uploads/2024/06/PlantTreeManual.pdf) · [Bangkok species by site and sunlight](https://greener.bangkok.go.th/green-space/million-trees/) · [Dynamic World 10 m land cover](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)")
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

    st.subheader("7-step Machine Learning workflow used in this project")
    ml_steps = [
        ("1. Define the problem", "Estimate land surface temperature (LST) for each 250 m grid cell; LST is the target label."),
        ("2. Collect data", "Load the prepared Bangkok grid CSV containing LST, NDVI, population, and cell coordinates."),
        ("3. Clean the data", "Remove rows missing required fields and filter implausible LST, NDVI, and population values."),
        ("4. Prepare model features", "Use longitude, latitude, NDVI, and population as inputs; use observed LST as the target."),
        ("5. Split the data", f"Hold out 20% for evaluation and train on 80% (seed 42; {ml_metrics['train_samples']:,} training and {ml_metrics['test_samples']:,} test rows)."),
        ("6. Train the AI model", "Fit a Random Forest regressor with 100 trees and maximum depth 12, then generate grid-level LST estimates."),
        ("7. Evaluate and use predictions", "Compare held-out predictions with observed LST using MAE, RMSE, and R²; use estimates as heat-screening information."),
    ]
    for title, description in ml_steps:
        st.markdown(f"**{title}** — {description}")
    st.warning("This trained model predicts LST only. Tree species and tree counts are not trained ML targets: the current data has no verified site-level species, planting count, or survival labels. The planting screen combines GEE land-cover evidence with published Bangkok guidance, so it should not be presented as an AI species prediction.")
    
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

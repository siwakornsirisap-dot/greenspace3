"""Google Earth Engine helpers for satellite screening at a Bangkok grid cell."""

from __future__ import annotations

import json
from datetime import date

import streamlit as st


@st.cache_resource(show_spinner=False)
def initialize_earth_engine():
    """Initialize EE using Streamlit secrets; return (ready, message)."""
    try:
        try:
            import ee
        except ImportError:
            return False, "ยังไม่ได้ติดตั้ง earthengine-api (ติดตั้งด้วย pip install earthengine-api)"

        gee_config = st.secrets.get("gee", {})
        project_id = gee_config.get("project_id")
        service_account_json = gee_config.get("service_account_json")
        if not project_id:
            return False, "ยังไม่ได้ตั้งค่า Google Cloud Project สำหรับ Earth Engine"

        credentials = None
        if service_account_json:
            from google.oauth2 import service_account

            info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        # Explicitly request the OAuth token saved by ``ee.Authenticate()`` for
        # local runs. Passing ``None`` can fall through to ambient Google
        # credentials that exist in some environments but carry no user token.
        if credentials is None:
            ee.Initialize(credentials="persistent", project=project_id)
        else:
            ee.Initialize(credentials=credentials, project=project_id)
        # Trigger a small request so authentication/registration errors surface here.
        ee.String("Earth Engine connected").getInfo()
        return True, f"เชื่อมต่อ Earth Engine แล้ว (Project: {project_id})"
    except Exception as exc:  # Missing secrets, unregistered project, and auth errors.
        return False, f"เชื่อมต่อ Earth Engine ไม่สำเร็จ: {exc}"


@st.cache_data(ttl=86400, show_spinner=False)
def get_satellite_screening(latitude: float, longitude: float) -> dict:
    """Return season-matched satellite indicators and confidence-filtered land cover."""
    ready, message = initialize_earth_engine()
    if not ready:
        raise RuntimeError(message)
    import ee

    today = date.today()
    # Compare three completed Bangkok hot seasons (March–May), rather than
    # mixing wet/dry seasons in one temperature/vegetation composite.
    last_hot_season_year = today.year if today.month > 5 else today.year - 1
    first_hot_season_year = last_hot_season_year - 2
    start_date = date(first_hot_season_year, 1, 1).isoformat()
    end_date = date(last_hot_season_year + 1, 1, 1).isoformat()
    hot_season_start = date(first_hot_season_year, 3, 1).isoformat()
    hot_season_end = date(last_hot_season_year, 5, 31).isoformat()
    region = ee.Geometry.Point([longitude, latitude]).buffer(125).bounds()

    def mask_landsat(image):
        qa = image.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 0).eq(0)  # Fill
        clear = clear.And(qa.bitwiseAnd(1 << 1).eq(0))  # Dilated cloud
        clear = clear.And(qa.bitwiseAnd(1 << 2).eq(0))  # Cirrus
        clear = clear.And(qa.bitwiseAnd(1 << 3).eq(0))  # Cloud
        clear = clear.And(qa.bitwiseAnd(1 << 4).eq(0))  # Cloud shadow
        clear = clear.And(qa.bitwiseAnd(1 << 5).eq(0))  # Snow
        temperature_c = image.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        return temperature_c.rename("LST_C").updateMask(clear)

    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(3, 5, "month"))
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .map(mask_landsat)
    )

    sentinel_raw = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.calendarRange(3, 5, "month"))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 50))
    )
    cloud_score = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    joined_sentinel = ee.ImageCollection(ee.Join.saveFirst("cloud_score").apply(
        primary=sentinel_raw,
        secondary=cloud_score,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    ))

    def mask_sentinel(image):
        scl = image.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        score = ee.Image(image.get("cloud_score")).select("cs_cdf")
        clear = clear.And(score.gte(0.60))
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return ndvi.updateMask(clear)

    sentinel = joined_sentinel.map(mask_sentinel)

    combined = landsat.median().addBands(sentinel.median())
    values = combined.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=region,
        scale=30,
        bestEffort=True,
        maxPixels=100000,
    ).getInfo()
    # Dynamic World is a recent, 10 m land-cover product. Exclude low-confidence
    # pixels before summing classes; grass/bare remain only open-surface proxies.
    class_names = [
        "water", "trees", "grass", "flooded_vegetation", "crops",
        "shrub_and_scrub", "built", "bare", "snow_and_ice",
    ]
    dynamic_world = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(date(today.year - 1, today.month, min(today.day, 28)).isoformat(), end_date)
    )
    landcover_area_m2 = {}
    landcover_confidence = None
    landcover_confident_coverage_pct = None
    landcover_valid_area_m2 = None
    landcover_scenes = dynamic_world.size().getInfo()
    if landcover_scenes > 0:
        probabilities = ee.Image(dynamic_world.select(class_names).median())
        top_confidence = probabilities.reduce(ee.Reducer.max()).rename("top_confidence")
        confident = top_confidence.gte(0.60)
        class_index = (
            probabilities.toArray().arrayArgmax().arrayGet([0])
            .rename("landcover_class").updateMask(confident)
        )
        area_bands = [
            ee.Image.pixelArea().updateMask(class_index.eq(index)).rename(class_name)
            for index, class_name in enumerate(class_names)
        ]
        area_values = ee.Image.cat(area_bands).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=10,
            bestEffort=True, maxPixels=100000,
        ).getInfo()
        landcover_area_m2 = {name: area_values.get(name) or 0 for name in class_names}
        confidence_values = top_confidence.reduceRegion(
            reducer=ee.Reducer.median(), geometry=region, scale=10,
            bestEffort=True, maxPixels=100000,
        ).getInfo()
        landcover_confidence = confidence_values.get("top_confidence")
        coverage_values = ee.Image.pixelArea().updateMask(confident).rename("area").reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=10,
            bestEffort=True, maxPixels=100000,
        ).getInfo()
        landcover_valid_area_m2 = coverage_values.get("area") or 0
        landcover_confident_coverage_pct = min(100.0, landcover_valid_area_m2 / 62500.0 * 100)

    return {
        "lst_c": values.get("LST_C"),
        "ndvi": values.get("NDVI"),
        "start_date": start_date,
        "end_date": end_date,
        "hot_season_start": hot_season_start,
        "hot_season_end": hot_season_end,
        "landcover_start": date(today.year - 1, today.month, min(today.day, 28)).isoformat(),
        "landcover_end": end_date,
        "landsat_scenes": landsat.size().getInfo(),
        "sentinel_scenes": sentinel.size().getInfo(),
        "landcover_area_m2": landcover_area_m2,
        "landcover_confidence": landcover_confidence,
        "landcover_confident_coverage_pct": landcover_confident_coverage_pct,
        "landcover_valid_area_m2": landcover_valid_area_m2,
        "landcover_scenes": landcover_scenes,
    }

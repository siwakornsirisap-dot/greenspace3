# bangkok-urban-heat-green-space-dashboard
Academic research prototype mapping land surface temperature, green spaces, and priority intervention zones across Bangkok.
# Bangkok Urban Heat & Green Space Priority Dashboard

An academic research prototype and decision-support dashboard evaluating Land Surface Temperature (LST), Normalized Difference Vegetation Index (NDVI), and population density across Bangkok, Thailand.

- **Deployment:** Streamlit Community Cloud (deployment setup in progress)
- **Spatial Resolution:** 250m × 250m Analytical Grid Units (~25,000 cells)

---

## Project Architecture

The dashboard runs from the prepared grid data and can optionally request live satellite screening from Google Earth Engine when credentials are configured:

1. **Data Storage & Source Control:** Prepared grid data and application code are managed in this GitHub repository.
2. **Machine Learning & Dashboard:** Streamlit trains the Random Forest LST model and serves the interactive dashboard.
3. **Optional satellite screening:** The Tree Planting page calls the Earth Engine Python API for Landsat, Sentinel-2, and Dynamic World summaries.

---

## Repository Structure

```text
bangkok-urban-heat-green-space-dashboard/
├── app.py                           # Main Streamlit web application source code
├── gee_satellite.py                 # Earth Engine authentication and satellite queries
├── requirements.txt                 # Cloud dependencies & Python package manifest
├── README.md                        # Technical documentation & project overview
└── data/
    ├── bangkok_250m_grid_data.csv   # Aggregated 250m spatial grid dataset
    └── bangkok_district_summary.csv # Summary metrics across Bangkok's 50 districts

## Run locally

From this project directory, install dependencies and start the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Earth Engine satellite screening

The Tree Planting Recommendations page queries Landsat 8/9 surface temperature and Sentinel-2 NDVI for three matched, completed Bangkok hot seasons (March–May). Landsat fill/cloud/cirrus/shadow/snow QA bits and non-L2SP scenes are excluded. Sentinel-2 pixels are screened with SCL plus Cloud Score+ (`cs_cdf >= 0.60`). Recent Dynamic World 10 m class-area estimates include only pixels whose top-class probability is at least 0.60; the page reports the retained coverage and withholds a tree-count estimate when less than half of the grid has confident land-cover coverage. The grass/bare area is still only a potential-open-surface proxy, not a verified planting parcel. A theoretical capacity range uses Bangkok's 4–6 m spacing guidance, and no user-entered area, width, site type, or species is required.

1. Create/register a Google Cloud project for Earth Engine and enable the Earth Engine API, following Google's [authentication and initialization guide](https://developers.google.com/earth-engine/guides/auth). If using a service account, grant it access to the registered Earth Engine project as described in Google's [service-account guide](https://developers.google.com/earth-engine/guides/service_account).
2. For local use, authenticate your Google account once from PowerShell in the project folder:

   ```powershell
   earthengine authenticate
   ```

   Then create `.streamlit/secrets.toml` with the registered project ID. Local Earth Engine credentials are stored by the Earth Engine client; do not put them in the repository.

   ```toml
   [gee]
   project_id = "boxwood-theory-513712-e8"
   ```

   Start the dashboard after authentication with `streamlit run app.py`. The app can use the local Earth Engine OAuth credentials when `service_account_json` is not configured.

3. For Streamlit Community Cloud, add valid Earth Engine credentials in the app's **Settings → Secrets**. Local OAuth credentials are not available to the cloud app. The service-account JSON key could not be created for this project because Google Cloud enforces `iam.disableServiceAccountKeyCreation`. Until an approved cloud identity is configured, the dashboard can load but live GEE satellite screening will remain unavailable. Do not try to weaken that policy without the organization administrator. Never commit credentials to GitHub.

The automatic species shortlist follows Bangkok's published full-sun species guidance. Dynamic World classifies land cover; it cannot verify public ownership, contiguous planting geometry, sidewalk width, shade at the exact point, utilities, soil, or flood conditions. The displayed count is a theoretical screening range based on grass/bare pixels and species spacing, not an authorized or field-verified planting count. Do not use it as a construction plan without site/landowner verification. Dynamic World attribution: produced for the Dynamic World Project by Google in partnership with National Geographic Society and the World Resources Institute.

The Interactive Priority Map supports category and continuous-score filters. Markers and the ranked location table include grid ID, WGS84 latitude/longitude, observed and predicted LST, NDVI, population density, and the weighted priority score. The current filtered locations can be downloaded as CSV. When the marker cap is reached, the map displays the highest-scoring matching cells first.

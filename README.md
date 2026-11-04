# bangkok-urban-heat-green-space-dashboard
Academic research prototype mapping land surface temperature, green spaces, and priority intervention zones across Bangkok.
# Bangkok Urban Heat & Green Space Priority Dashboard

An academic research prototype and decision-support dashboard evaluating Land Surface Temperature (LST), Normalized Difference Vegetation Index (NDVI), and population density across Bangkok, Thailand.

- **Live Web Application:** [https://bangkok-urban-heat-green-space-dashboard-kimkimkim.streamlit.app](https://bangkok-urban-heat-green-space-dashboard-kimkimkim.streamlit.app)
- **Deployment Platform:** Streamlit Community Cloud (100% Cloud-Hosted)
- **Spatial Resolution:** 250m × 250m Analytical Grid Units (~25,000 cells)

---

## Project Architecture

This project operates on a fully cloud-first workflow requiring zero local software installation or raw satellite data downloads:

1. **Data Extraction & Aggregation:** Google Earth Engine (GEE) processes Landsat 8/9 thermal & optical bands alongside WorldPop layers directly on Google infrastructure.
2. **Data Storage & Source Control:** Aggregated grid datasets and application code are managed in this GitHub repository.
3. **Machine Learning & Dashboard:** Streamlit Community Cloud automatically builds the Python environment, executes the Random Forest Regression model, and serves the public interactive website.

---

## Repository Structure

```text
bangkok-urban-heat-green-space-dashboard/
├── app.py                           # Main Streamlit web application source code
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

The Interactive Priority Map supports category and continuous-score filters. Markers and the ranked location table include grid ID, WGS84 latitude/longitude, observed and predicted LST, NDVI, population density, and the weighted priority score. The current filtered locations can be downloaded as CSV. When the marker cap is reached, the map displays the highest-scoring matching cells first.

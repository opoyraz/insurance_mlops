# 🛡️ Insurance Risk MLOps

An end-to-end machine learning pipeline for predicting driver insurance risk scores based on driver profiles and geographic proximity to crash hotspots.

## 🏗️ Architecture

```
Data Sources                Pipeline                    Serving
─────────────              ──────────                  ────────
drivers_raw.csv  ──┐
                    ├──→ Validate → Cluster → Train → model.joblib
Socrata API      ──┘     (Pydantic)  (KMeans)  (GB)       │
                                                      ┌────┴────┐
                                                 FastAPI    Streamlit
                                                :8000/predict  :8501
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.914 |
| MAE | 1.46 |
| RMSE | 2.03 |
| Training Samples | 100,346 |
| Features | 20 |
| Clustering | KMeans (K=6, Silhouette=0.42) |

## 🗂️ Project Structure

```
insurance_mlops/
├── src/
│   ├── config.py              # Centralized settings (pydantic-settings)
│   ├── pipeline.py            # Production pipeline (8 steps)
│   ├── data/
│   │   ├── socrata_client.py  # Fetch crash data from Chicago API
│   │   └── validation.py      # DataFrame validation
│   ├── features/
│   │   ├── clustering.py      # KMeans hotspot detection
│   │   └── engineering.py     # Feature preprocessing
│   ├── models/
│   │   └── train.py           # Gradient Boosting + MLflow
│   └── api/
│       ├── main.py            # FastAPI application
│       ├── schemas.py         # Pydantic request/response models
│       └── predict.py         # Prediction logic
├── data/
│   ├── raw/                   # drivers_raw.csv, household_coords.csv
│   ├── processed/             # Cleaned CSVs, cluster summary
│   └── model/                 # model.joblib
├── notebooks/                 # Exploration notebooks
├── streamlit_app.py           # Streamlit UI (4 tabs)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt           # Full dependencies (local)
└── requirements-docker.txt    # Slim dependencies (Docker)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/insurance_mlops.git
cd insurance_mlops
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Data Files

Place these files in `data/raw/`:
- `drivers_raw.csv` — Driver profiles with risk scores
- `household_coords.csv` — Household lat/lon coordinates

### 3. Run the Pipeline

```bash
# First run: fetches crash data from Socrata, trains model
python -m src.pipeline

# Subsequent runs: uses cached crash data
python -m src.pipeline

# Force fresh crash data
python -m src.pipeline --refresh
```

### 4. Launch Streamlit

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 5. Launch API

```bash
uvicorn src.api.main:app --reload
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

## 🐳 Docker

Run all services (API + Streamlit + MLflow) with one command:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |

Stop all services:

```bash
docker-compose down
```

## 🔬 Pipeline Details

### Data Sources

- **Driver Data**: Local CSV with 100K+ driver records (age, gender, vehicle, claims history)
- **Crash Data**: Chicago's Socrata Open Data API — real-time traffic crash records with coordinates and injury severity

### Feature Engineering

1. **Crash Filtering**: 8,550 injury/fatal crashes from 52,334 total
2. **KMeans Clustering** (K=6): Identifies crash hotspot zones across Chicago
3. **Geodesic Distances**: Miles from each household to all 6 cluster centers
4. **20 features total**: 6 hotspot + 12 numeric + 1 categorical + 1 binary

### Model Selection

| Model | R² | MAE |
|-------|-----|-----|
| **Gradient Boosting** | **0.914** | **1.46** |
| XGBoost | 0.914 | 1.46 |
| Random Forest | 0.899 | 1.67 |
| Linear Regression | 0.897 | 1.69 |

Gradient Boosting chosen over XGBoost for identical performance with no extra dependency.

### Clustering Comparison

| Method | Silhouette | Result |
|--------|-----------|--------|
| **KMeans** | **0.42** | **Winner** |
| Agglomerative | 0.35 | Decent |
| DBSCAN | -1.00 | Failed (all noise) |

## 🛠️ Tech Stack

- **ML**: Scikit-learn, XGBoost
- **API**: FastAPI, Pydantic
- **UI**: Streamlit, Folium
- **Tracking**: MLflow
- **Deployment**: Docker, Docker Compose
- **Data**: Pandas, Socrata API (sodapy), Geopy

## 📝 License

MIT

# Karachi AQI Predictor

An end-to-end machine-learning system for monitoring and forecasting Karachi's
Air Quality Index (AQI). The project combines hourly PM2.5 observations with
weather data, stores the processed feature data in Hopsworks, trains
horizon-specific forecasting models, registers the models in Hopsworks Model
Registry, and serves the latest readings and forecasts through a Streamlit
dashboard.

The dashboard predicts AQI for four horizons:

- One hour ahead
- Twenty-four hours ahead
- Forty-eight hours ahead
- Seventy-two hours ahead

## What The Project Does

1. Collects weather data for Karachi from Open-Meteo.
2. Collects hourly PM2.5 data from OpenAQ.
3. Cleans, merges, and validates the source data.
4. Calculates AQI from PM2.5 measurements.
5. Builds lag, rolling-window, change, percentage-change, and cyclical time features.
6. Stores the training features in the Hopsworks Feature Store.
7. Trains and evaluates several regression approaches.
8. Registers optimized XGBoost models in Hopsworks Model Registry.
9. Reads the newest Hopsworks row in the Streamlit application.
10. Uses the registered models to generate live forecasts and explanations.

## Architecture

```text
Open-Meteo weather data ─┐
												 ├─> Cleaning and merging ─> Feature engineering
OpenAQ PM2.5 data ───────┘                                  │
																														v
																						 Hopsworks Feature Store
																														│
																														v
																						 Horizon-specific models
																														│
																														v
																						 Hopsworks Model Registry
																														│
																														v
																								Streamlit dashboard
```

The dashboard reads the latest feature-group history, recreates the engineered
features required by the registered models, downloads each registry model, and
performs inference using the model's own feature-name contract.

## Dashboard Features

The application is located at `app/app.py` and provides:

- Latest raw weather and pollution readings in dark header cards.
- Dynamic daytime sun and nighttime moon icon on the temperature card.
- Current AQI gauge with six AQI ranges from 0 to 400.
- AQI range colors for Good, Moderate, Unhealthy for Sensitive Groups,
	Unhealthy, Very Unhealthy, and Hazardous conditions.
- Hazardous and very-unhealthy alerts for current and forecast AQI values.
- Registry-backed predictions for +1h, +24h, +48h, and +72h.
- AQI and temperature trend charts with zoom, scroll zoom, reset, and range
	slider controls.
- Correlation heatmap for the displayed raw features and AQI.
- SHAP-compatible XGBoost feature contributions using native XGBoost
	`pred_contribs`, without requiring a separate SHAP installation.
- Horizon buttons for viewing feature importance for each registered model.
- Model Applied section with comparison views for XGBoost, Extra Trees,
	MLPRegressor, and PyTorch.
- Model metrics, interpretation text, and model-selection rationale.

The dashboard intentionally displays only the original weather and pollution
features. Engineered lag, rolling, change, and cyclical features are kept
internal because they are inputs for inference rather than user-facing
measurements.

## Hopsworks Configuration

The current application uses these Hopsworks resources:

| Resource | Value |
|---|---|
| Project | `anaskaaqi` |
| Feature group | `aqi_training_features` |
| Feature-group version | `2` |
| Feature view | `aqi_72_hour_forecast` |
| Model version | `1` by default |

Registered model names:

| Forecast horizon | Registry model |
|---|---|
| +1 hour | `aqi_1_hour_xgboost` |
| +24 hours | `aqi_24_hour_xgboost` |
| +48 hours | `aqi_48_hour_xgboost` |
| +72 hours | `aqi_72_hour_xgboost` |

## Project Structure

```text
AQI_Predictor/
├── app/
│   ├── app.py                 # Streamlit dashboard
│   └── requirements.txt       # Dashboard dependencies
├── data/
│   ├── raw/                   # Raw source data
│   ├── historical/            # Historical AQI and weather data
│   ├── processed/             # Cleaned and feature datasets
│   ├── 2026/                  # Current-year datasets
│   └── ci_cd/                 # Data used by automation workflows
├── models/                    # Local model artifacts and evaluation results
├── notebooks/                 # Exploratory analysis and experiments
├── registry_uploads/          # Model-registry upload artifacts
├── src/
│   ├── 2026/                  # Current-year ingestion and merge jobs
│   ├── ci_cd/                 # Automated data fetching and model registry jobs
│   ├── data_collection/       # API clients and collection configuration
│   ├── data_processing/       # Cleaning and transformation logic
│   ├── feature_Engineering/   # Feature groups, views, and feature creation
│   ├── models/                # Training, evaluation, and registration scripts
│   └── utils/                 # Shared AQI utilities
├── tests/                     # Test suite
├── requirements.txt           # Core Python dependencies
└── README.md
```

## Requirements

- Python 3.11 or a compatible recent Python version
- A Hopsworks account and API key
- OpenAQ API access
- Internet access for the external data sources and Hopsworks

The root `requirements.txt` contains the data and model dependencies. The
dashboard requirements file adds Streamlit and Plotly.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the complete dashboard environment:

```bash
pip install -r app/requirements.txt
```

## Environment Variables

Create a local `.env` file or export the variables in the shell:

```dotenv
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=anaskaaqi
OPENAQ_API_KEY=your_openaq_api_key
```

The application loads `.env` automatically through `python-dotenv`. Never
commit `.env` or publish API keys in source code, screenshots, logs, or the
README. Rotate any key that has been exposed.

Optional model configuration:

```dotenv
AQI_MODEL_VERSION=1
```

## Run The Dashboard

From the repository root:

```bash
source venv/bin/activate
streamlit run app/app.py
```

The default local URL is:

```text
http://localhost:8501
```

If that port is already in use, choose another one:

```bash
streamlit run app/app.py --server.port 8502
```

## Data And Feature Pipeline

The pipeline uses Karachi weather and PM2.5 measurements keyed by hourly
timestamps. In addition to the original fields, the model input can contain:

- AQI lag values at multiple lookback periods.
- Rolling AQI mean, standard deviation, minimum, and maximum.
- AQI differences and percentage changes.
- Hour and day-of-week sine/cosine encodings.
- Calendar fields such as hour, day, month, year, and weekend flag.

The dashboard reconstructs these features from the full Hopsworks history before
selecting the latest row. This is necessary because the registered models were
trained with a larger feature contract than the raw values shown in the UI.

## Models And Evaluation

The project evaluates multiple regression strategies. The current model
comparison shown in the dashboard is:

| Model | +1h R² | +24h R² | +48h R² | +72h R² |
|---|---:|---:|---:|---:|
| XGBoost | 0.8930 | 0.5605 | 0.5370 | 0.5283 |
| Extra Trees | 0.8837 | 0.5375 | 0.3702 | 0.5246 |
| MLP Regressor | 0.4872 | -0.2295 | -0.2977 | -0.0223 |
| PyTorch unstable run | -0.7731 | -0.7006 | -0.7667 | -0.8005 |

XGBoost is used for production forecasting because it remains positive and
comparatively stable across all four required horizons. The dashboard keeps the
alternative model results available as documented comparison views.

## Important Source Files

- `app/app.py`: dashboard, Hopsworks reads, registry model downloads, live
	inference, alerts, trend charts, correlation matrix, and explanations.
- `src/ci_cd/datafetch.py`: current data-fetching and Hopsworks update workflow.
- `src/ci_cd/xgboost_model_registry.py`: feature preparation, training, and
	model-registry workflow.
- `src/models/xgboost_model.py`: XGBoost feature engineering and training logic.
- `src/models/register.py`: local model upload and registration configuration.
- `src/feature_Engineering/feature_group.py`: Hopsworks feature-group setup.
- `src/feature_Engineering/feature_view.py`: Hopsworks feature-view setup.
- `src/utils/aqi.py`: AQI calculation utilities.

## Development Checks

Compile the Streamlit application:

```bash
./venv/bin/python -m py_compile app/app.py
```

Run the tests when available:

```bash
pytest
```

The dashboard must be started with `streamlit run app/app.py`; running
`python app/app.py` directly does not start a Streamlit application correctly.

## Security Notes

- Keep `.env` out of version control.
- Do not place API keys in README files or Python source files.
- Use separate development and production credentials where possible.
- Rotate credentials immediately if they are accidentally exposed.

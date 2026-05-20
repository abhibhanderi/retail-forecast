# Retail Sales Forecasting Dashboard

An interactive machine learning dashboard that forecasts weekly Walmart store sales using five models and an ensemble, built as a Master's in Computer Science capstone project.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)

---

## Live Demo

[Add Streamlit Cloud URL here once deployed]

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Models](#models)
- [Results](#results)
- [Architecture](#architecture)
- [Team](#team)
- [License](#license)

---

## Project Overview

Retail inventory distortion — including overstock and stockouts — costs the global economy an estimated $1.77 trillion per year. Accurate sales forecasting lets retailers reduce that waste by matching supply to demand before it happens. This project builds an end-to-end forecasting pipeline for 45 Walmart stores, training five models and a weighted ensemble on 2.5 years of weekly transaction data. A Streamlit dashboard makes the results interactive: store managers can explore forecasts by store and department, compare model accuracy side by side, and understand what drives each prediction using SHAP feature explanations.

---

## Tech Stack

| Tool | Purpose | Version |
|---|---|---|
| Python | Core language | 3.11+ |
| Streamlit | Interactive dashboard | 1.38.0 |
| Plotly | Interactive charts | 5.24.1 |
| Pandas | Data manipulation | 2.2.3 |
| NumPy | Numerical computing | 1.26.4 |
| XGBoost | Gradient boosted trees | 2.1.1 |
| LightGBM | Fast gradient boosting | 4.5.0 |
| Prophet | Facebook seasonal model | 1.1.6 |
| ARIMA (pmdarima) | Auto-selected ARIMA | 2.0.4 |
| SHAP | Model explainability | 0.46.0 |
| scikit-learn | Preprocessing utilities | 1.5.2 |
| GitHub Actions | CI/CD | — |
| Streamlit Community Cloud | Hosting | — |

---

## Dataset

**Walmart Store Sales Forecasting** — [Kaggle Competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

- **Size:** 421,570 rows, 45 stores, ~100 departments
- **Period:** February 2010 to October 2012
- **Features:** Weekly sales, temperature, fuel price, CPI, unemployment, IsHoliday, MarkDown1–5

Three CSV files — place in `data/raw/` before running the pipeline:

| File | Rows | Description |
|---|---|---|
| `train.csv` | 421,570 | Weekly sales per Store/Dept with holiday flag |
| `stores.csv` | 45 | Store type (A/B/C) and square footage |
| `features.csv` | 8,190 | Temperature, fuel price, markdowns, CPI, unemployment |

---

## Project Structure

```
retail-sales-forecast/
├── .github/
│   └── workflows/
│       └── ci.yml                   GitHub Actions: lint + fast tests on push/PR
├── .streamlit/
│   └── config.toml                  Streamlit theme and server settings
├── dashboard/
│   └── app.py                       Streamlit app: 3 tabs (Forecast, Comparison, Explorer)
├── data/
│   ├── raw/                         Raw Kaggle CSVs (git-ignored)
│   └── processed/                   Parquet splits and prediction outputs
├── docs/
│   ├── architecture.md              Pipeline diagram and design decisions
│   ├── deployment.md                Streamlit Cloud deployment guide
│   ├── screenshots/                 Dashboard screenshots for README
│   │   └── README.md                Instructions for capturing screenshots
│   └── wireframes/                  Early UI wireframes
├── models/                          Trained .pkl files (git-ignored); result JSON/CSV committed
├── notebooks/
│   └── 01_eda.ipynb                 Exploratory data analysis (14 sections)
├── src/
│   ├── preprocessing.py             Data loading, cleaning, temporal train/test split
│   ├── feature_engineering.py       Lag, rolling, date, and holiday features (35 total)
│   ├── models.py                    Six forecaster classes with shared BaseForecaster interface
│   ├── evaluation.py                MAE, RMSE, MAPE, weighted MAE metric functions
│   ├── run_evaluation.py            Walk-forward CV, ensemble weighting, result file writer
│   ├── save_predictions.py          Runs all models on test set, saves parquet outputs
│   ├── shap_analysis.py             SHAP TreeExplainer on XGBoost, global and store-level
│   └── utils.py                     Shared I/O and logging helpers
├── tests/
│   ├── conftest.py                  pytest configuration (slow mark)
│   ├── test_preprocessing.py        4 tests: missing values, encoding, split correctness
│   ├── test_evaluation.py           4 tests: MAE, MAPE exclusion, weighted MAE, model ranking
│   ├── test_feature_engineering.py  4 tests: lag/rolling leakage, holiday dates, NaN drop
│   ├── test_models.py               4 tests: MA calculation, save/load, ensemble weights/accuracy
│   ├── test_shap_analysis.py        3 tests: top-N sorted, n parameter, plain-English output
│   └── test_integration.py          5 slow tests: pipeline artefacts exist and are valid
├── .gitignore
├── CHANGELOG.md
├── CLAUDE.md                        AI assistant guidance for this codebase
├── Makefile                         All workflow commands
├── packages.txt                     Linux system packages for Streamlit Cloud (libgomp1)
├── pytest.ini                       pytest marker registration
├── README.md
└── requirements.txt
```

---

## How to Run

### Prerequisites

- Python 3.11+
- `make` — install with `choco install make` on Windows, `brew install make` on Mac
- Three Kaggle CSVs placed in `data/raw/` (see [Dataset](#dataset))

### Setup

```bash
make install     # Install all Python dependencies
make pipeline    # Full run: preprocess → features → train → evaluate → predictions → SHAP
make test        # Run the fast unit test suite
make serve       # Launch dashboard at http://localhost:8501
```

### Individual Pipeline Steps

```bash
make preprocess   # Merge CSVs and create parquet splits
make features     # Add lag, rolling, date, and holiday features
make train        # Train all six models and save .pkl files
make evaluate     # Walk-forward CV, ensemble weighting, write result files
make predictions  # Run all models on test set, save parquet outputs
make shap         # Compute SHAP values for XGBoost model
```

### Other Commands

```bash
make test-unit         # Fast tests only (excludes integration)
make test-integration  # Slow integration tests (requires pipeline to have run)
make coverage          # Test coverage report (excludes integration tests)
make lint              # flake8 style check
make check             # lint + test-unit
make clean             # Remove all generated artefacts (parquet, pkl, JSON, logs)
make reset             # Remove pipeline artefacts only (keeps raw data)
```

---

## Models

| Model | Type | Strengths | Training Scope |
|---|---|---|---|
| Moving Average (4w / 12w) | Statistical baseline | Fast, fully interpretable, no training required | Per Store/Dept trailing mean |
| ARIMA | Statistical time series | Captures autocorrelation and linear trend | Top 10 stores (store-level series) |
| Prophet | Additive decomposition | Robust to holidays and irregular promotions | Top 10 stores (store-level series) |
| XGBoost | Gradient boosted trees | Highest accuracy, captures non-linear feature interactions | All stores and departments |
| LightGBM | Gradient boosted trees | Matches XGBoost accuracy with faster training | All stores and departments |
| Ensemble | Inverse-MAPE weighted average | Reduces individual model variance and overfitting | All models combined |

---

## Results

| Model | MAE | RMSE | MAPE | Weighted MAE |
|---|---|---|---|---|
| **Ensemble** | **$1,750** | **$3,630** | **10.8%** | **$1,768** |
| LightGBM | $6,566 | $11,776 | 13.5% | $2,485 |
| XGBoost | $6,585 | $11,791 | 13.7% | $2,564 |
| Moving Average | $3,522 | $9,431 | 21.9% | $3,867 |
| ARIMA | $3,531 | $9,407 | 21.9% | $3,871 |
| Prophet | $3,520 | $9,434 | 21.9% | $3,872 |

> Metrics computed on held-out test set (Jul–Oct 2012) using walk-forward cross-validation. Weighted MAE weights holiday weeks 5× per Kaggle competition rules.

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for design decisions and full model details.

```
Raw CSV files (data/raw/)
       │
       ▼
src/preprocessing.py
  - Loads and merges 3 CSV files
  - Handles missing values
  - Encodes features
  - Temporal train/test split at 2012-07-01
       │
       ▼
data/processed/train_processed.parquet
data/processed/test_processed.parquet
       │
       ▼
src/feature_engineering.py
  - Lag features (lag_1, lag_2, lag_4, lag_12, lag_52)
  - Rolling statistics (mean, std, min, max)
  - Date features (week, month, quarter, year)
  - Holiday features (thanksgiving, superbowl, christmas)
       │
       ▼
src/models.py
  - Trains 5 models and saves as .pkl files
       │
       ├── model_moving_average_4w.pkl
       ├── model_moving_average_12w.pkl
       ├── model_arima.pkl
       ├── model_prophet.pkl
       ├── model_xgboost.pkl
       └── model_lightgbm.pkl
       │
       ▼
src/run_evaluation.py
  - Walk-forward cross-validation (3 folds)
  - Computes MAE, RMSE, MAPE, weighted_MAE
  - Builds ensemble with inverse-MAPE weights
  - Saves results_metrics.json, results_cv_folds.csv
  - Saves results_ensemble_weights.json
       │
       ▼
src/save_predictions.py
  - Runs all 6 models on test set
  - Saves predictions_test.parquet
  - Saves actuals_train.parquet
       │
       ▼
src/shap_analysis.py
  - TreeExplainer on XGBoost model
  - Saves shap_values.pkl
       │
       ▼
dashboard/app.py (Streamlit)
  - Tab 1: Forecast chart and weekly table
  - Tab 2: Model comparison metrics
  - Tab 3: Data explorer
```

---

## Team

| Person | Role |
|---|---|
| Person A | ML Pipeline (src/, notebooks/, models/) |
| Person B | Dashboard and Deployment (dashboard/, deployment) |
| Person C | Evaluation and Documentation (tests/, report, presentation) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

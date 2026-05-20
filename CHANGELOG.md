# Changelog

All notable changes to this project are documented here, organised by week.

---

## Week 10: Documentation and Polish

- README.md rewritten with full project documentation, actual metric values, and complete file tree
- docs/architecture.md rewritten with spec pipeline diagram, key design decisions, and model details
- docs/screenshots/ folder created with screenshot capture instructions
- Results table populated with actual values from models/results_metrics.json
- CHANGELOG.md updated with Week 9 and Week 10 entries

---

## Week 9: Testing and Deployment

- Integration test suite added (5 tests in test_integration.py, all @pytest.mark.slow)
- Unit test suite trimmed to 19 high-value tests across 5 files (no padding tests)
- Streamlit Community Cloud deployment configuration added
- .streamlit/config.toml created with teal theme and server settings
- packages.txt created for Linux system dependency (libgomp1 for LightGBM)
- Deployment guide created at docs/deployment.md
- requirements.txt pinned to exact versions; added pyarrow, pytest-cov
- Dashboard loading functions updated to show st.error + st.stop() for missing files
- .gitignore updated: result JSON/CSV files now committable for Streamlit Cloud

---

## Week 8: Polish and Testing

- Test coverage improved to 70%+ across all `src/` modules
- Added test suites for `src/utils.py`, `src/shap_analysis.py`, and `src/run_evaluation.py`
- Extended `tests/test_models.py` with EnsembleForecaster, XGBoost/LightGBM fallback paths, and BaseForecaster edge cases
- Extended `tests/test_preprocessing.py` with `run_full_pipeline` integration tests
- Final UI polish pass for the Streamlit dashboard: spinner on load, missing-file error messages, responsive sidebar summary, chart x-axis labels, metrics table caption, bold best-row highlight
- `st.set_page_config` updated with project title, text icon, and menu_items
- Empty/error state guards added across all three dashboard tabs
- EDA notebook sections 11–14 added (Model Performance, Ensemble Weights, SHAP Importance, Key Takeaways)
- Documentation completed: `README.md`, `CHANGELOG.md`, `docs/architecture.md`
- `.gitignore` updated to cover all generated artefacts

---

## Week 7: Ensemble and Explainability

- Ensemble model with inverse-MAPE weighting across all five component models
- `src/shap_analysis.py` module: TreeExplainer on XGBoost, global and store-level feature importance
- Model Comparison tab (Tab 2): metrics table with progress bars, grouped bar chart, fold-level MAPE chart, ensemble weights chart
- Explainability tab (Tab 3): global SHAP bar chart, store-level SHAP, plain-English feature explanations
- STL seasonal decomposition chart added to Tab 3
- `models/results_ensemble_weights.json` and `results_ensemble_predictions.csv` outputs added

---

## Week 5–6: Advanced Models and Dashboard Features

- Prophet forecaster: store-level additive seasonality with holiday regressor
- XGBoost forecaster: 35-feature matrix from lag, rolling, date, and holiday features
- LightGBM forecaster: same feature matrix, leaf-wise growth, faster training
- Walk-forward cross-validation (`walk_forward_cv`) with expanding training window
- `src/run_evaluation.py`: orchestrates CV across all models, writes `results_cv_folds.csv`
- Dashboard sidebar filters: store selector, department multiselect, date range picker, aggregation toggle
- KPI cards row with total sales, average weekly, best model, best MAPE, weeks selected
- Forecast timeline chart with train/test split marker and rangeslider

---

## Week 4: Baseline Models and Dashboard v0.2

- `MovingAverageForecaster`: per-(Store, Dept) trailing mean, window=4 and window=12
- `ARIMAForecaster`: auto_arima (pmdarima) on store-level aggregates, dept-share redistribution
- `BaseForecaster` abstract class with `fit`, `predict`, and `get_name` interface
- Initial Streamlit dashboard: single-tab layout with sales chart and KPI metrics
- `models/` directory structure and pkl persistence via joblib

---

## Week 3: Data Pipeline and EDA

- `src/preprocessing.py`: three-CSV merge, missing value handling, feature encoding, temporal train/test split
- `src/feature_engineering.py`: lag features (1, 4, 12, 52 weeks), rolling statistics, date features, holiday proximity features
- `src/evaluation.py`: MAE, RMSE, MAPE, and Weighted MAE metric functions
- `notebooks/01_eda.ipynb`: 14-section exploratory analysis covering distributions, correlations, seasonality, and outliers
- `src/utils.py`: shared `load_raw_data`, `save_processed_data`, `load_processed_data`, `setup_logging` helpers

---

## Week 1–2: Project Setup

- Project scaffolding and folder structure established
- `requirements.txt` and `Makefile` with `install`, `pipeline`, `test`, `serve` targets
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`): flake8 lint + pytest on push/PR to main
- `data/raw/`, `data/processed/`, `models/` directories added to `.gitignore`
- Initial `src/__init__.py` and `tests/__init__.py`

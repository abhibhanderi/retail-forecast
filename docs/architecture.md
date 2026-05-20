# Architecture

## Pipeline Flow

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

## Key Design Decisions

**Why temporal split instead of random split**
Retail sales data has strong weekly and seasonal patterns — a sale in November predicts future November sales far better than a random sample. Splitting randomly would leak future data into training and make every model look far more accurate than it actually is. The split is fixed at 2012-07-01 so that training (Feb 2010 – Jun 2012) always precedes the test period (Jul–Oct 2012) with no overlap.

**Why inverse-MAPE weighting for the ensemble**
Each model's weight is proportional to `1 / avg_MAPE` from cross-validation: a model with half the error gets twice the weight. This is simple, requires no additional training, and directly encodes validation performance into the blending step. It also handles large accuracy gaps well — a model with MAPE=5% naturally dominates one with MAPE=20%.

**Why TreeExplainer instead of KernelExplainer for SHAP**
TreeExplainer exploits the tree structure of XGBoost to compute exact SHAP values in polynomial time — typically under a second for 500 rows. KernelExplainer is model-agnostic but uses sampling approximations that can take minutes on the same data and produce noisier estimates. Since XGBoost is a tree-based model, TreeExplainer is both faster and more accurate.

**Why top 10 stores for ARIMA and Prophet**
ARIMA and Prophet each fit a separate model per store. Fitting all 45 stores takes roughly 20–30 minutes; limiting to the top 10 by sales volume reduces pipeline time to 3–5 minutes while covering the stores that matter most. Remaining stores receive predictions scaled from their historical department share of the nearest fitted store.

**Why parquet format for processed data**
Parquet stores columnar data with efficient compression and preserves dtypes (including datetime) across reads. A 421,570-row CSV takes ~35 MB and ~4 seconds to read; the equivalent parquet is ~6 MB and reads in under 0.5 seconds. The speed difference is significant when the dashboard reloads data on every Streamlit session start.

**Why predictions are pre-computed instead of live inference**
ARIMA and Prophet take several minutes to generate predictions; doing this on each dashboard load would make the app unusable. All models run once during `make pipeline`, and their outputs are saved to `predictions_test.parquet`. The dashboard reads the parquet file directly — no models are loaded at runtime.

---

## Model Details

### Moving Average (4-week and 12-week)

For each (Store, Dept) pair, the model predicts next week's sales as the trailing mean of the last 4 or 12 weeks of actual sales. It is chosen as the primary baseline because it requires no training and is fully transparent — anyone can verify a prediction by hand. Its main strength is speed and interpretability; its main weakness is that it cannot capture trends, seasonality, or promotional effects.

### ARIMA

ARIMA (AutoRegressive Integrated Moving Average) models the store-level weekly sales series as a linear combination of past values and past forecast errors. The order (p, d, q) is selected automatically by pmdarima's `auto_arima`, which tests multiple configurations and picks the one with the lowest AIC. Its strength is capturing autocorrelation and gradual trends in stable series; its weakness is that it is fit on store-level totals and must redistribute forecasts to departments using historical shares, which loses department-level variation.

### Prophet

Prophet decomposes sales into trend, weekly seasonality, yearly seasonality, and a holiday component. A binary holiday regressor is added for the five known high-sales weeks (Super Bowl, Labour Day, Thanksgiving, Christmas, and Black Friday). Its main strength is robustness to missing data and the ability to explicitly model irregular holiday spikes, which dominate Walmart's weekly sales variance. Its main weakness is the same as ARIMA: it is fit at store level, so department-level accuracy depends on the share redistribution step.

### XGBoost

XGBoost trains a gradient-boosted decision tree ensemble on a 35-feature matrix: lag features (lag_1, lag_4, lag_12, lag_52), rolling statistics (mean and std over 4 and 12 weeks), date features (week of year, month, quarter), and holiday indicator features. It is chosen because gradient boosting consistently outperforms statistical models on tabular time-series with rich feature sets. Its main strength is accuracy and its ability to capture non-linear interactions between features; its main weakness is that it requires the full feature matrix to be reconstructed at prediction time, including appending training tail rows to satisfy lag dependencies.

### LightGBM

LightGBM uses the identical 35-feature matrix as XGBoost but grows trees leaf-wise rather than level-wise, and uses histogram binning to approximate split points. In practice this means training runs 2–3× faster than XGBoost with comparable accuracy (within ~0.5% MAPE on this dataset). It is chosen as a second tabular model to test whether the implementation choice affects results and to provide a second vote in the ensemble. Its `num_leaves=63` setting is a deliberate cap to prevent overfitting on the smaller per-store data slices.

### Ensemble

The ensemble computes a weighted average of all five component model predictions. Weights are set to `w_i = (1/MAPE_i) / sum(1/MAPE_j)` using the average MAPE from walk-forward cross-validation, so each model's contribution is proportional to its validated accuracy. Weights are saved to `results_ensemble_weights.json` and reloaded at prediction time. The ensemble consistently achieves lower Weighted MAE than any single model because the statistical models (MA, ARIMA, Prophet) and tree models (XGBoost, LightGBM) make uncorrelated errors — averaging them cancels variance.

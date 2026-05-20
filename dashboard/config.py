from __future__ import annotations

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette — light theme only
# ---------------------------------------------------------------------------

PRIMARY   = "#2563EB"   # royal blue (charts)
TEAL      = "#0D9488"   # teal-600  (accent)
TEAL_SOFT = "#CCFBF1"   # teal-100
NAVY      = "#1B2A4A"   # dark navy (sidebar)
BLUE_SOFT = "#DBEAFE"   # light blue fill
DARK_TEXT = "#0F172A"   # near-black
MID_TEXT  = "#334155"   # slate-700
MUTED     = "#64748B"   # slate-500
BORDER    = "#E2E8F0"   # slate-200
PAGE_BG   = "#F8FAFC"   # slate-50
CARD_BG   = "#FFFFFF"
SUCCESS   = "#16A34A"
DANGER    = "#DC2626"

STORE_TYPE_COLORS = {
    "A": "#2563EB",
    "B": "#3B82F6",
    "C": "#93C5FD",
}

PLOTLY_TMPL = "plotly_white"

TRAIN_PATH          = Path(__file__).parent.parent / "data" / "processed" / "train_processed.parquet"
TEST_PATH           = Path(__file__).parent.parent / "data" / "processed" / "test_processed.parquet"
RESULTS_PATH        = Path(__file__).parent.parent / "models" / "results_metrics.json"
MODELS_DIR          = Path(__file__).parent.parent / "models"
ENSEMBLE_PREDS_PATH   = MODELS_DIR / "results_ensemble_predictions.csv"
CV_RESULTS_PATH       = MODELS_DIR / "results_cv_folds.csv"
ENSEMBLE_WEIGHTS_PATH = MODELS_DIR / "results_ensemble_weights.json"
PREDICTIONS_TEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "predictions_test.parquet"
ACTUALS_TRAIN_PATH    = Path(__file__).parent.parent / "data" / "processed" / "actuals_train.parquet"

# Test period boundaries
TEST_START = pd.Timestamp("2012-04-06")
TEST_END   = pd.Timestamp("2012-10-26")
_TRAIN_TEST_SPLIT = pd.Timestamp("2012-04-06")

# Model display name → pkl stem (matches _PKL_SAVE_NAMES in src/models.py)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Moving Average (w=4)":  "model_moving_average_4w",
    "Moving Average (w=12)": "model_moving_average_12w",
    "ARIMA":                 "model_arima",
    "Prophet":               "model_prophet",
    "XGBoost":               "model_xgboost",
    "LightGBM":              "model_lightgbm",
}
MODEL_COLORS: dict[str, str] = {
    "Moving Average (w=4)":  "#F59E0B",
    "Moving Average (w=12)": "#D97706",
    "ARIMA":                 "#8B5CF6",
    "Prophet":               "#EC4899",
    "XGBoost":               "#0D9488",
    "LightGBM":              "#0F766E",
    "Ensemble":              "#DC2626",
}
_ALL_MODEL_NAMES = ["Ensemble"] + list(MODEL_DISPLAY_NAMES.keys())

# Model display name → key used in results_metrics.json
_MODEL_RESULT_KEYS: dict[str, str] = {
    "Ensemble":              "Ensemble",
    "XGBoost":               "XGBoost(n=500,d=6)",
    "LightGBM":              "LightGBM(n=500,leaves=63)",
    "ARIMA":                 "ARIMA(auto,nonseasonal,top10)",
    "Prophet":               "Prophet(top10)",
    "Moving Average (w=4)":  "MovingAverage(w=4)",
    "Moving Average (w=12)": "MovingAverage(w=12)",
}

# Model display name → prediction column in predictions_test.parquet
MODEL_PRED_COLS: dict[str, str] = {
    "Moving Average (w=4)":  "pred_moving_average_4w",
    "Moving Average (w=12)": "pred_moving_average_12w",
    "ARIMA":                 "pred_arima",
    "Prophet":               "pred_prophet",
    "XGBoost":               "pred_xgboost",
    "LightGBM":              "pred_lightgbm",
    "Ensemble":              "pred_ensemble",
}

# baseline_results.json model key → short display label
_MODEL_SHORT_NAMES: dict[str, str] = {
    "MovingAverage(w=4)":            "MA (w=4)",
    "MovingAverage(w=12)":           "MA (w=12)",
    "ARIMA(auto,nonseasonal,top10)": "ARIMA",
    "Prophet(top10)":                "Prophet",
    "XGBoost(n=500,d=6)":            "XGBoost",
    "LightGBM(n=500,leaves=63)":     "LightGBM",
    "Ensemble":                      "Ensemble",
}

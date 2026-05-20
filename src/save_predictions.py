"""
Pre-compute all model predictions once and save to parquet files.

Eliminates live model inference from the dashboard — the dashboard reads
parquet files instead of loading .pkl models and running predict().

Outputs
-------
data/processed/predictions_test.parquet
    One row per (Store, Dept, Date) in the test period.
    Columns: Store, Dept, Date, actual,
             pred_moving_average_4w, pred_moving_average_12w,
             pred_arima, pred_prophet, pred_xgboost, pred_lightgbm,
             pred_ensemble

data/processed/actuals_train.parquet
    Train-period actuals for the full historical chart line.
    Columns: Store, Dept, Date, actual, IsHoliday

Usage
-----
    python -m src.save_predictions
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import setup_logging

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = _ROOT / "models"
_DATA = _ROOT / "data" / "processed"

# Ordered list of (column_suffix, pkl_filename)
_MODEL_FILES: list[tuple[str, str]] = [
    ("moving_average_4w", "model_moving_average_4w.pkl"),
    ("moving_average_12w", "model_moving_average_12w.pkl"),
    ("arima", "model_arima.pkl"),
    ("prophet", "model_prophet.pkl"),
    ("xgboost", "model_xgboost.pkl"),
    ("lightgbm", "model_lightgbm.pkl"),
]

_ENSEMBLE_CSV = _MODELS / "results_ensemble_predictions.csv"
_TEST_PARQUET = _DATA / "test_processed.parquet"
_TRAIN_PARQUET = _DATA / "train_processed.parquet"
_OUT_PREDS = _DATA / "predictions_test.parquet"
_OUT_ACTUALS = _DATA / "actuals_train.parquet"


def run() -> None:
    setup_logging()

    # ── Load test data ────────────────────────────────────────────────────────
    if not _TEST_PARQUET.exists():
        log.error("Test data not found: %s — run make preprocess first", _TEST_PARQUET)
        sys.exit(1)

    log.info("Loading test data from %s", _TEST_PARQUET)
    test_df = pd.read_parquet(_TEST_PARQUET)
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df = test_df.reset_index(drop=True)

    # Base output DataFrame — one row per (Store, Dept, Date)
    keep = ["Store", "Dept", "Date", "Weekly_Sales"]
    if "IsHoliday" in test_df.columns:
        keep.append("IsHoliday")
    out = test_df[keep].copy()
    out = out.rename(columns={"Weekly_Sales": "actual"})

    # ── Run predict() for each model ─────────────────────────────────────────
    for key, pkl_name in _MODEL_FILES:
        col = f"pred_{key}"
        pkl_path = _MODELS / pkl_name
        if not pkl_path.exists():
            log.warning("pkl not found, filling with NaN: %s", pkl_path)
            out[col] = np.nan
            continue
        try:
            log.info("Running predict() for %s ...", key)
            model = joblib.load(pkl_path)
            preds = np.asarray(model.predict(test_df))
            if len(preds) != len(out):
                log.error(
                    "Prediction length mismatch for %s: got %d, expected %d",
                    key, len(preds), len(out),
                )
                out[col] = np.nan
            else:
                out[col] = preds
        except Exception as exc:
            log.error("predict() failed for %s: %s", key, exc)
            out[col] = np.nan

    # ── Ensemble predictions ──────────────────────────────────────────────────
    if _ENSEMBLE_CSV.exists():
        log.info("Loading ensemble predictions from %s", _ENSEMBLE_CSV)
        try:
            ens = pd.read_csv(_ENSEMBLE_CSV, parse_dates=["Date"])
            ens["Date"] = pd.to_datetime(ens["Date"])
            ens = ens.rename(columns={"ensemble_forecast": "pred_ensemble"})
            out = out.merge(
                ens[["Store", "Dept", "Date", "pred_ensemble"]],
                on=["Store", "Dept", "Date"],
                how="left",
            )
        except Exception as exc:
            log.error("Failed to load ensemble CSV: %s", exc)
            out["pred_ensemble"] = np.nan
    else:
        log.warning("Ensemble predictions not found: %s", _ENSEMBLE_CSV)
        out["pred_ensemble"] = np.nan

    # ── Save test predictions ─────────────────────────────────────────────────
    out = out.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    out.to_parquet(_OUT_PREDS, index=False)
    log.info("Saved %d rows → %s", len(out), _OUT_PREDS)

    # ── Save train actuals ────────────────────────────────────────────────────
    if not _TRAIN_PARQUET.exists():
        log.warning("Train data not found, skipping actuals: %s", _TRAIN_PARQUET)
        return

    log.info("Loading train data from %s", _TRAIN_PARQUET)
    train_df = pd.read_parquet(_TRAIN_PARQUET)
    train_df["Date"] = pd.to_datetime(train_df["Date"])

    keep_cols = ["Store", "Dept", "Date", "Weekly_Sales"]
    if "IsHoliday" in train_df.columns:
        keep_cols.append("IsHoliday")

    actuals = (
        train_df[keep_cols]
        .copy()
        .rename(columns={"Weekly_Sales": "actual"})
        .sort_values(["Store", "Dept", "Date"])
        .reset_index(drop=True)
    )
    actuals.to_parquet(_OUT_ACTUALS, index=False)
    log.info("Saved %d rows → %s", len(actuals), _OUT_ACTUALS)

    log.info("Done. Dashboard is ready — run: make serve")


if __name__ == "__main__":
    run()

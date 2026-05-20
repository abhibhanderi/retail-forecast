from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.config import (
    TRAIN_PATH,
    TEST_PATH,
    RESULTS_PATH,
    PREDICTIONS_TEST_PATH,
    ACTUALS_TRAIN_PATH,
    CV_RESULTS_PATH,
    ENSEMBLE_WEIGHTS_PATH,
)


@st.cache_data(show_spinner="Loading predictions...", ttl=3600)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_TEST_PATH.exists():
        st.error(
            "**Missing file:** `data/processed/predictions_test.parquet`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    df = pd.read_parquet(PREDICTIONS_TEST_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading actuals...", ttl=3600)
def load_actuals() -> pd.DataFrame:
    if not ACTUALS_TRAIN_PATH.exists():
        st.error(
            "**Missing file:** `data/processed/actuals_train.parquet`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    df = pd.read_parquet(ACTUALS_TRAIN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> tuple[pd.DataFrame, bool]:
    """Load the combined train+test parquet for the Data Explorer tab."""
    if not TRAIN_PATH.exists():
        return pd.DataFrame(), False
    train = pd.read_parquet(TRAIN_PATH)
    df = pd.concat(
        [train, pd.read_parquet(TEST_PATH)] if TEST_PATH.exists() else [train],
        ignore_index=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    if "Type" not in df.columns and "Type_encoded" in df.columns:
        df["Type"] = df["Type_encoded"].map({1: "A", 2: "B", 3: "C"})
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True), True


@st.cache_data(show_spinner=False)
def _load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_metrics.json`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cv_results() -> pd.DataFrame | None:
    if not CV_RESULTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_cv_folds.csv`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        return pd.read_csv(CV_RESULTS_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_ensemble_weights() -> dict | None:
    if not ENSEMBLE_WEIGHTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_ensemble_weights.json`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        with open(ENSEMBLE_WEIGHTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

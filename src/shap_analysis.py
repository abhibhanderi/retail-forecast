from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import get_feature_columns, run_feature_engineering
from src.utils import load_processed_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level path constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = Path("models/model_xgboost.pkl")
_DEFAULT_DATA_PATH = Path("data/processed/train_processed.parquet")
_SHAP_VALUES_PATH = Path("models/shap_values.pkl")
_SHAP_FEATURE_NAMES_PATH = Path("models/shap_feature_names.pkl")

# ---------------------------------------------------------------------------
# Plain-English feature descriptions (covers all 35 engineered features)
# ---------------------------------------------------------------------------

_FEATURE_EXPLANATIONS: dict[str, str] = {
    # Lag features
    "lag_1": (
        "Last week's sales is a strong short-term signal — "
        "recent performance tends to persist from one week to the next."
    ),
    "lag_2": (
        "Sales from two weeks ago helps detect early momentum shifts "
        "or brief dips before they show up in rolling averages."
    ),
    "lag_4": (
        "Sales from one month ago captures monthly buying cycles "
        "for this store-department combination."
    ),
    "lag_12": (
        "Sales from three months ago reflects seasonal patterns "
        "at a quarterly level."
    ),
    "lag_52": (
        "Last year's sales for the same week is the strongest seasonal anchor — "
        "annual cycles dominate retail demand."
    ),
    # Rolling features
    "rolling_mean_4": (
        "The four-week average smooths out weekly noise and reveals "
        "the near-term sales trend."
    ),
    "rolling_std_4": (
        "Recent week-to-week variability — high volatility often precedes "
        "larger swings in either direction."
    ),
    "rolling_min_4": (
        "The lowest sales in the past month; the model uses this "
        "as a recent floor when projecting forward."
    ),
    "rolling_max_4": (
        "The highest sales in the past month; the model uses this "
        "as a recent ceiling when projecting forward."
    ),
    "rolling_mean_12": (
        "The three-month average captures the medium-term sales trajectory "
        "for this store-department."
    ),
    "rolling_std_12": (
        "Three-month variability — departments with consistent sales "
        "need less adjustment than highly volatile ones."
    ),
    # Date / calendar features
    "week_of_year": (
        "Which week of the year it is captures recurring seasonal patterns "
        "such as back-to-school, summer lulls, and holiday build-up."
    ),
    "month": (
        "The calendar month reflects broad seasonal demand shifts "
        "across the year."
    ),
    "quarter": (
        "Which quarter the week falls in encodes coarser seasonal effects "
        "like Q4 holiday spending."
    ),
    "year": (
        "The year captures long-term growth or decline trends "
        "in overall store performance."
    ),
    "day_of_year": (
        "The day within the year fine-tunes calendar effects "
        "beyond week or month alone."
    ),
    "is_month_start": (
        "The start of the month often sees higher spending "
        "as paydays arrive for many households."
    ),
    "is_month_end": (
        "Month-end patterns differ from mid-month, typically dipping "
        "as consumers wait for the next pay cycle."
    ),
    # Holiday features
    "weeks_to_christmas": (
        "How many weeks until Christmas — demand spikes sharply "
        "as this countdown reaches zero."
    ),
    "weeks_since_last_holiday": (
        "Weeks since the last holiday week; demand often softens briefly "
        "after a spending spike before returning to baseline."
    ),
    "is_thanksgiving_week": (
        "Thanksgiving week is one of the busiest retail periods "
        "and is flagged as a distinct high-impact event."
    ),
    "is_superbowl_week": (
        "Super Bowl week drives specific department sales "
        "such as electronics, snacks, and prepared foods."
    ),
    # Passthrough preprocessing features
    "Size": (
        "Larger stores draw higher footfall and carry more product lines, "
        "supporting higher total weekly sales."
    ),
    "Type_encoded": (
        "Store type (A, B, or C) reflects the store's market format "
        "and typical customer demographics."
    ),
    "IsHoliday": (
        "Holiday weeks are weighted five times more in the competition metric "
        "and reliably see elevated demand across most departments."
    ),
    "is_markdown_active": (
        "Whether a promotional markdown is currently running — "
        "discounts consistently lift short-term sales."
    ),
    "MarkDown1": (
        "Type-1 promotional markdown value; "
        "larger discounts in this category drive higher store traffic."
    ),
    "MarkDown2": (
        "Type-2 promotional markdown value; "
        "signals active category-level promotions."
    ),
    "MarkDown3": (
        "Type-3 promotional markdown value; "
        "signals active category-level promotions."
    ),
    "MarkDown4": (
        "Type-4 promotional markdown value; "
        "signals active category-level promotions."
    ),
    "MarkDown5": (
        "Type-5 promotional markdown value; "
        "signals active category-level promotions."
    ),
    "Temperature": (
        "Local temperature affects shopping patterns, "
        "especially in seasonal departments like gardening or clothing."
    ),
    "Fuel_Price": (
        "Fuel prices influence how willingly customers travel to the store, "
        "affecting overall foot traffic."
    ),
    "CPI": (
        "The Consumer Price Index reflects purchasing power — "
        "higher inflation shifts where and how much people choose to buy."
    ),
    "Unemployment": (
        "Local unemployment rates affect discretionary spending "
        "and overall store traffic."
    ),
}


# ---------------------------------------------------------------------------
# 1. compute_shap_values
# ---------------------------------------------------------------------------

def compute_shap_values(
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    data_path: str | Path = _DEFAULT_DATA_PATH,
    sample_size: int = 500,
) -> tuple[np.ndarray, list[str]]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP analysis. "
            "Install it with: pip install shap"
        ) from exc

    model_path = Path(model_path)
    data_path = Path(data_path)

    logger.info("Loading XGBoost model from '%s'", model_path)
    forecaster = joblib.load(model_path)
    xgb_model = forecaster._model
    if xgb_model is None:
        raise RuntimeError(
            f"The model at '{model_path}' has not been fitted (_model is None). "
            "Run the training pipeline first."
        )

    logger.info("Loading training data from '%s'", data_path)
    raw_df = load_processed_data(data_path)

    logger.info("Running feature engineering on %d rows ...", len(raw_df))
    feat_df = run_feature_engineering(raw_df)

    # get_feature_columns() is valid now that run_feature_engineering has
    # populated the module-level column registries.
    feature_names = get_feature_columns()
    avail_cols = [c for c in feature_names if c in feat_df.columns]

    n_rows = min(sample_size, len(feat_df))
    sample = feat_df.sample(n=n_rows, random_state=42)
    X_sample = sample[avail_cols].fillna(0.0).values.astype(float)
    logger.info(
        "Sampled %d rows for SHAP (requested %d, available %d).",
        n_rows, sample_size, len(feat_df),
    )

    logger.info(
        "Computing SHAP values: %d rows × %d features ...",
        X_sample.shape[0], X_sample.shape[1],
    )
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    _SHAP_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shap_values, _SHAP_VALUES_PATH)
    joblib.dump(avail_cols, _SHAP_FEATURE_NAMES_PATH)
    logger.info(
        "Saved shap_values %s -> '%s'", shap_values.shape, _SHAP_VALUES_PATH
    )
    logger.info("Saved feature_names -> '%s'", _SHAP_FEATURE_NAMES_PATH)

    logger.info("SHAP values computed: shape=%s", shap_values.shape)

    return shap_values, avail_cols


# ---------------------------------------------------------------------------
# 2. get_top_features
# ---------------------------------------------------------------------------

def get_top_features(n: int = 10) -> pd.DataFrame:
    if not _SHAP_VALUES_PATH.exists() or not _SHAP_FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            "SHAP artifacts not found. Run compute_shap_values() first. "
            f"Expected: '{_SHAP_VALUES_PATH}' and '{_SHAP_FEATURE_NAMES_PATH}'."
        )

    shap_values: np.ndarray = joblib.load(_SHAP_VALUES_PATH)
    feature_names: list[str] = joblib.load(_SHAP_FEATURE_NAMES_PATH)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    logger.info(
        "Top %d features by mean |SHAP|:\n%s", n, top_df.to_string(index=False)
    )
    return top_df


# ---------------------------------------------------------------------------
# 3. get_shap_for_store
# ---------------------------------------------------------------------------

def get_shap_for_store(
    store_id: int,
    data_path: str | Path = _DEFAULT_DATA_PATH,
    model_path: str | Path = _DEFAULT_MODEL_PATH,
) -> tuple[np.ndarray, list[str]]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP analysis. "
            "Install it with: pip install shap"
        ) from exc

    data_path = Path(data_path)
    model_path = Path(model_path)

    logger.info(
        "Loading training data from '%s' for store %d", data_path, store_id
    )
    raw_df = load_processed_data(data_path)

    store_df = raw_df[raw_df["Store"] == store_id].copy()
    if store_df.empty:
        available = sorted(raw_df["Store"].unique().tolist())
        raise ValueError(
            f"store_id={store_id} not found in '{data_path}'. "
            f"Available stores: {available}"
        )

    logger.info(
        "Store %d: %d rows — running feature engineering ...",
        store_id, len(store_df),
    )
    feat_df = run_feature_engineering(store_df)

    feature_names = get_feature_columns()
    avail_cols = [c for c in feature_names if c in feat_df.columns]
    X_store = feat_df[avail_cols].fillna(0.0).values.astype(float)

    logger.info("Loading XGBoost model from '%s'", model_path)
    forecaster = joblib.load(model_path)
    xgb_model = forecaster._model
    if xgb_model is None:
        raise RuntimeError(
            f"The model at '{model_path}' has not been fitted (_model is None)."
        )

    logger.info(
        "Computing SHAP values for store %d: %d rows × %d features ...",
        store_id, X_store.shape[0], X_store.shape[1],
    )
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_store = explainer.shap_values(X_store)

    logger.info(
        "Store %d SHAP complete: shape=%s", store_id, shap_values_store.shape
    )
    return shap_values_store, avail_cols


# ---------------------------------------------------------------------------
# 4. get_plain_english_explanation
# ---------------------------------------------------------------------------

def get_plain_english_explanation(top_features_df: pd.DataFrame) -> list[str]:
    explanations: list[str] = []
    for _, row in top_features_df.head(5).iterrows():
        feature: str = row["feature"]
        text = _FEATURE_EXPLANATIONS.get(
            feature,
            "This feature contributes meaningfully to the sales forecast.",
        )
        explanations.append(f"{feature}: {text}")
    return explanations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()

    shap_vals, feat_names = compute_shap_values()

    top_df = get_top_features(n=10)

    print(f"\n{'=' * 58}")
    print("  Top 10 Features by Mean |SHAP|")
    print(f"{'=' * 58}")
    print(top_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print(f"\n{'=' * 58}")
    print("  Plain-English Explanations (Top 5)")
    print(f"{'=' * 58}")
    for line in get_plain_english_explanation(top_df):
        feature, explanation = line.split(": ", 1)
        print(f"\n  [{feature}]")
        print(f"  {explanation}")
    print()

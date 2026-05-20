from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import (
    create_holiday_features,
    create_lag_features,
    create_rolling_features,
    run_feature_engineering,
)


def _make_df(n_weeks: int = 60, stores: tuple = (1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="W-FRI")
    rows = []
    for store in stores:
        for date in dates:
            rows.append({
                "Store": store, "Dept": 1, "Date": date,
                "Weekly_Sales": float(rng.integers(5_000, 20_000)),
                "IsHoliday": 0, "Type": "A", "Type_encoded": 1, "Size": 150_000,
                "Temperature": 55.0, "Fuel_Price": 3.5,
                "MarkDown1": 0.0, "MarkDown2": 0.0, "MarkDown3": 0.0,
                "MarkDown4": 0.0, "MarkDown5": 0.0, "is_markdown_active": 0,
                "CPI": 210.0, "Unemployment": 8.0,
            })
    return (
        pd.DataFrame(rows).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    )


def test_lag_features_no_cross_store_leakage():
    # First row of each store must have NaN lag — if not, store 2 leaks into store 1
    df = _make_df(stores=(1, 2))
    result = create_lag_features(df, lags=[1])
    for store in (1, 2):
        first = result[result["Store"] == store].iloc[0]
        assert pd.isna(first["lag_1"]), f"Store {store} first lag_1 should be NaN"


def test_rolling_features_uses_shift_to_prevent_leakage():
    # Rolling is computed on lag_1 (shifted series), not on Weekly_Sales directly.
    # Row 0 must be NaN; row 1 must equal row 0's sales (not row 1's).
    df = create_lag_features(_make_df(stores=(1,)), lags=[1])
    result = create_rolling_features(df, windows=[4])
    group = result[result["Store"] == 1].reset_index(drop=True)
    assert pd.isna(group.loc[0, "rolling_mean_4"])
    assert group.loc[1, "rolling_mean_4"] == group.loc[0, "Weekly_Sales"]


def test_is_thanksgiving_week_correct_date():
    # 2010-11-26 is the Friday after Thanksgiving 2010 (Nov 25)
    df = pd.DataFrame([{
        "Store": 1, "Dept": 1, "Date": pd.Timestamp("2010-11-26"),
        "Weekly_Sales": 10000.0, "IsHoliday": 1, "Type": "A", "Type_encoded": 1,
        "Size": 150000, "Temperature": 45.0, "Fuel_Price": 3.2,
        "MarkDown1": 0.0, "MarkDown2": 0.0, "MarkDown3": 0.0,
        "MarkDown4": 0.0, "MarkDown5": 0.0, "is_markdown_active": 0,
        "CPI": 210.0, "Unemployment": 8.0,
    }])
    assert create_holiday_features(df).loc[0, "is_thanksgiving_week"] == 1


def test_run_feature_engineering_drops_nan_rows():
    # lag_52 generates 52 NaN rows per group — must be dropped before training
    df = _make_df(stores=(1,))
    result = run_feature_engineering(df)
    assert len(result) < len(df)
    lag_cols = [c for c in result.columns if c.startswith("lag_")]
    assert result[lag_cols].isna().sum().sum() == 0

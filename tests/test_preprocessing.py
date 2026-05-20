from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    create_train_test_split,
    encode_features,
    handle_missing_values,
)


def _make_merged_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Store": [1, 1, 1, 2, 2, 2],
        "Dept": [1, 1, 1, 1, 1, 1],
        "Date": pd.to_datetime([
            "2012-01-06", "2012-01-13", "2012-07-06",
            "2012-01-06", "2012-01-13", "2012-07-06",
        ]),
        "Weekly_Sales": [10000.0, 12000.0, 9000.0, 8000.0, 8500.0, 7500.0],
        "IsHoliday": [False, False, False, False, False, False],
        "Type": ["A", "A", "A", "B", "B", "B"],
        "Size": [150000, 150000, 150000, 90000, 90000, 90000],
        "Temperature": [42.3, 38.5, 81.2, 55.1, 52.0, 78.9],
        "Fuel_Price": [3.20, 3.25, 3.80, 3.15, 3.20, 3.75],
        "MarkDown1": [None, 500.0, None, 300.0, None, None],
        "MarkDown2": [None, None, 200.0, None, None, None],
        "MarkDown3": [None, None, None, None, None, None],
        "MarkDown4": [None, None, None, None, None, None],
        "MarkDown5": [None, None, 150.0, None, None, None],
        "CPI": [210.5, None, 212.0, 190.0, None, 191.5],
        "Unemployment": [8.1, None, 7.9, 9.0, None, 8.8],
    })


def test_no_nan_in_weekly_sales_after_cleaning():
    assert handle_missing_values(_make_merged_df())["Weekly_Sales"].isna().sum() == 0


def test_markdown_nan_filled_with_zero():
    # NaN means no promotion was running, not a data quality issue
    result = handle_missing_values(_make_merged_df())
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        assert result[col].isna().sum() == 0
        assert (result[col] >= 0).all()


def test_type_encoded_valid_values():
    df = encode_features(handle_missing_values(_make_merged_df()))
    assert set(df["Type_encoded"].unique()).issubset({1, 2, 3})
    assert (df[df["Type"] == "A"]["Type_encoded"] == 1).all()
    assert (df[df["Type"] == "B"]["Type_encoded"] == 2).all()


def test_train_test_no_date_overlap():
    df = encode_features(handle_missing_values(_make_merged_df()))
    train, test = create_train_test_split(df, test_start_date="2012-07-01")
    assert len(set(train["Date"]) & set(test["Date"])) == 0

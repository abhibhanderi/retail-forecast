from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    encode_features,
    handle_missing_values,
    load_and_merge_data,
    run_full_pipeline,
)


def _write_raw_csvs(tmp_path):
    pd.DataFrame({
        "Store": [1, 1], "Dept": [1, 1],
        "Date": ["2010-02-05", "2010-02-12"],
        "Weekly_Sales": [10000.0, 12000.0],
        "IsHoliday": [0, 0],
    }).to_csv(tmp_path / "train.csv", index=False)

    pd.DataFrame({
        "Store": [1], "Type": ["A"], "Size": [150000],
    }).to_csv(tmp_path / "stores.csv", index=False)

    pd.DataFrame({
        "Store": [1, 1], "Date": ["2010-02-05", "2010-02-12"],
        "Temperature": [55.0, 56.0], "Fuel_Price": [3.5, 3.6],
        "MarkDown1": [None, None], "MarkDown2": [None, None],
        "MarkDown3": [None, None], "MarkDown4": [None, None],
        "MarkDown5": [None, None],
        "CPI": [210.0, 211.0], "Unemployment": [8.0, 8.1],
        "IsHoliday": [0, 0],
    }).to_csv(tmp_path / "features.csv", index=False)


def test_load_and_merge_data_produces_correct_columns(tmp_path):
    _write_raw_csvs(tmp_path)
    df = load_and_merge_data(tmp_path)
    assert "Weekly_Sales" in df.columns
    assert "Type" in df.columns
    assert "Temperature" in df.columns
    assert "IsHoliday" not in df.columns or len(df) == 2


def test_load_and_merge_data_row_count_matches_train(tmp_path):
    _write_raw_csvs(tmp_path)
    df = load_and_merge_data(tmp_path)
    assert len(df) == 2


def test_encode_features_all_store_types(tmp_path):
    _write_raw_csvs(tmp_path)
    df = load_and_merge_data(tmp_path)
    df = handle_missing_values(df)
    df_b = df.copy()
    df_b["Type"] = "B"
    df_c = df.copy()
    df_c["Type"] = "C"
    for df_t, expected in [(df, 1), (df_b, 2), (df_c, 3)]:
        result = encode_features(df_t)
        assert (result["Type_encoded"] == expected).all()


def test_run_full_pipeline_saves_parquets(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_csvs(raw_dir)
    out_dir = tmp_path / "out"

    train_df, test_df = run_full_pipeline(
        data_dir=raw_dir,
        output_dir=out_dir,
        test_start_date="2010-02-12",
    )

    assert (out_dir / "train_processed.parquet").exists()
    assert (out_dir / "test_processed.parquet").exists()
    assert len(train_df) + len(test_df) == 2


def test_handle_missing_values_drops_rows_with_nan_weekly_sales(tmp_path):
    _write_raw_csvs(tmp_path)
    df = load_and_merge_data(tmp_path)
    df.loc[0, "Weekly_Sales"] = float("nan")
    cleaned = handle_missing_values(df)
    assert cleaned["Weekly_Sales"].isna().sum() == 0
    assert len(cleaned) < len(df)


def test_encode_features_raises_on_unknown_type(tmp_path):
    _write_raw_csvs(tmp_path)
    df = load_and_merge_data(tmp_path)
    df = handle_missing_values(df)
    df["Type"] = "Z"
    with pytest.raises(ValueError, match="Unexpected store Type"):
        encode_features(df)

from __future__ import annotations

import pathlib

import pandas as pd

from src.utils import (
    load_processed_data,
    load_raw_data,
    save_processed_data,
    setup_logging,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({"store": [1, 2], "sales": [1000.0, 2000.0]})


def test_save_load_roundtrip(tmp_path):
    df = _make_df()
    path = tmp_path / "out.parquet"
    save_processed_data(df, path)
    loaded = load_processed_data(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_save_creates_parent_directories(tmp_path):
    df = _make_df()
    path = tmp_path / "nested" / "subdir" / "data.parquet"
    save_processed_data(df, path)
    assert path.exists()


def test_setup_logging_with_file(tmp_path):
    log_path = str(tmp_path / "run.log")
    logger = setup_logging(log_file=log_path)
    assert logger is not None
    assert pathlib.Path(log_path).exists()


def test_setup_logging_without_file():
    logger = setup_logging(log_file=None)
    assert logger is not None


def test_load_raw_data_merges_three_csvs(tmp_path):
    pd.DataFrame({
        "Store": [1], "Dept": [1], "Date": ["2010-02-05"],
        "Weekly_Sales": [10000.0], "IsHoliday": [0],
    }).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({
        "Store": [1], "Type": ["A"], "Size": [150000],
    }).to_csv(tmp_path / "stores.csv", index=False)
    pd.DataFrame({
        "Store": [1], "Date": ["2010-02-05"],
        "Temperature": [55.0], "Fuel_Price": [3.5],
        "MarkDown1": [None], "MarkDown2": [None], "MarkDown3": [None],
        "MarkDown4": [None], "MarkDown5": [None],
        "CPI": [210.0], "Unemployment": [8.0], "IsHoliday": [0],
    }).to_csv(tmp_path / "features.csv", index=False)

    df = load_raw_data(tmp_path)
    assert "Weekly_Sales" in df.columns
    assert "Type" in df.columns
    assert "Temperature" in df.columns
    assert "IsHoliday" in df.columns  # kept from train.csv

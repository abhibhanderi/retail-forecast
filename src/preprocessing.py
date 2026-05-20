from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)

# Columns that must not contain NaN after cleaning
_CRITICAL_COLUMNS = ["Store", "Dept", "Date", "Weekly_Sales"]

# MarkDown columns present in features.csv
_MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# Store-type ordinal mapping
_TYPE_MAP: dict[str, int] = {"A": 1, "B": 2, "C": 3}

# Temporal boundary for train/test split — must align with walk_forward_cv
_TEST_START_DATE = "2012-04-06"


# ---------------------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------------------

def load_and_merge_data(data_dir: str | Path = "data/raw/") -> pd.DataFrame:
    data_dir = Path(data_dir)
    logger.info("Loading raw CSVs from '%s'", data_dir)

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])

    logger.info(
        "Raw shapes — train: %s  stores: %s  features: %s",
        train.shape,
        stores.shape,
        features.shape,
    )

    # train ← stores metadata (Store is the key)
    df = train.merge(stores, on="Store", how="left")

    # df ← weekly features; drop the redundant IsHoliday from features
    df = df.merge(
        features.drop(columns=["IsHoliday"]),
        on=["Store", "Date"],
        how="left",
    )

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    logger.info("Merged DataFrame shape: %s  columns: %s", df.shape, list(df.columns))
    return df


# ---------------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)

    # --- MarkDown columns ---
    markdown_nan_counts = {col: int(df[col].isna().sum()) for col in _MARKDOWN_COLS}
    total_markdown_filled = sum(markdown_nan_counts.values())
    df[_MARKDOWN_COLS] = df[_MARKDOWN_COLS].fillna(0.0).infer_objects(copy=False)
    logger.info(
        "Filled MarkDown NaNs with 0 — %s total values filled: %s",
        total_markdown_filled,
        markdown_nan_counts,
    )

    # --- Binary markdown active flag ---
    df["is_markdown_active"] = (df[_MARKDOWN_COLS].gt(0).any(axis=1)).astype(int)
    logger.info(
        "Created 'is_markdown_active': %d rows with active promotions",
        df["is_markdown_active"].sum(),
    )

    # --- CPI: forward-fill then backward-fill within each store ---
    cpi_nan_before = int(df["CPI"].isna().sum())
    df["CPI"] = (
        df.groupby("Store")["CPI"]
        .transform(lambda s: s.ffill().bfill())
    )
    cpi_nan_after = int(df["CPI"].isna().sum())
    logger.info(
        "CPI: filled %d NaN values (remaining: %d)",
        cpi_nan_before - cpi_nan_after,
        cpi_nan_after,
    )

    # --- Unemployment: forward-fill then backward-fill within each store ---
    unemp_nan_before = int(df["Unemployment"].isna().sum())
    df["Unemployment"] = (
        df.groupby("Store")["Unemployment"]
        .transform(lambda s: s.ffill().bfill())
    )
    unemp_nan_after = int(df["Unemployment"].isna().sum())
    logger.info(
        "Unemployment: filled %d NaN values (remaining: %d)",
        unemp_nan_before - unemp_nan_after,
        unemp_nan_after,
    )

    # --- Drop rows with NaN in critical columns ---
    nan_mask = df[_CRITICAL_COLUMNS].isna().any(axis=1)
    rows_dropped = int(nan_mask.sum())
    if rows_dropped:
        logger.warning(
            "Dropping %d rows with NaN in critical columns %s",
            rows_dropped,
            _CRITICAL_COLUMNS,
        )
        df = df[~nan_mask].reset_index(drop=True)

    logger.info(
        "handle_missing_values complete -- rows: %d -> %d (dropped %d)",
        initial_rows,
        len(df),
        rows_dropped,
    )
    return df


# ---------------------------------------------------------------------------
# 3. Encode features
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Store Type ordinal encoding ---
    unknown_types = set(df["Type"].unique()) - set(_TYPE_MAP.keys())
    if unknown_types:
        raise ValueError(
            f"Unexpected store Type values found: {unknown_types}. "
            f"Expected one of {set(_TYPE_MAP.keys())}."
        )
    df["Type_encoded"] = df["Type"].map(_TYPE_MAP)
    logger.info(
        "Encoded 'Type' -> 'Type_encoded' using map %s. "
        "Value counts:\n%s",
        _TYPE_MAP,
        df["Type_encoded"].value_counts().to_dict(),
    )

    # --- IsHoliday bool → int ---
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    logger.info(
        "Converted 'IsHoliday' to int — holiday weeks: %d",
        df["IsHoliday"].sum(),
    )

    return df


# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------

def create_train_test_split(
    df: pd.DataFrame,
    test_start_date: str = _TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(test_start_date)

    train_df = df[df["Date"] < cutoff].reset_index(drop=True)
    test_df = df[df["Date"] >= cutoff].reset_index(drop=True)

    logger.info(
        "Temporal split at %s -- train: %d rows (%s to %s), test: %d rows (%s to %s)",
        test_start_date,
        len(train_df),
        train_df["Date"].min().date(),
        train_df["Date"].max().date(),
        len(test_df),
        test_df["Date"].min().date(),
        test_df["Date"].max().date(),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    data_dir: str | Path = "data/raw/",
    output_dir: str | Path = "data/processed/",
    test_start_date: str = _TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("  data_dir   : %s", data_dir)
    logger.info("  output_dir : %s", output_dir)
    logger.info("  test_split : %s", test_start_date)
    logger.info("=" * 60)

    df = load_and_merge_data(data_dir)
    logger.info(
        "[1/4] Loaded & merged  ->  %d rows, %d columns", df.shape[0], df.shape[1]
    )

    df = handle_missing_values(df)
    nan_remaining = df.isna().sum()
    nan_remaining = nan_remaining[nan_remaining > 0]
    logger.info("[2/4] Cleaned NaNs     ->  %d rows remaining", df.shape[0])
    if not nan_remaining.empty:
        logger.info("      NaN remaining:\n%s", nan_remaining.to_string())

    df = encode_features(df)
    logger.info("[3/4] Encoded features ->  columns: %s", list(df.columns))

    train_df, test_df = create_train_test_split(df, test_start_date)

    train_path = output_dir / "train_processed.parquet"
    test_path = output_dir / "test_processed.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info("Saved train -> '%s'", train_path)
    logger.info("Saved test  -> '%s'", test_path)
    logger.info("[4/4] Saved parquet files to '%s'", output_dir)

    logger.info("Pipeline complete.")
    return train_df, test_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--data-dir", default="data/raw/", help="Path to raw CSVs")
    parser.add_argument(
        "--output-dir", default="data/processed/", help="Output directory"
    )
    parser.add_argument(
        "--test-start", default=_TEST_START_DATE, help="Test split date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    run_full_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_start_date=args.test_start,
    )

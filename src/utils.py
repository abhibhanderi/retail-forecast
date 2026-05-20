import logging
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])

    # Merge train with store metadata
    df = train.merge(stores, on="Store", how="left")

    # Merge with weekly features; drop the duplicate IsHoliday from features
    df = df.merge(
        features.drop(columns=["IsHoliday"]),
        on=["Store", "Date"],
        how="left",
    )

    return df


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_processed_data(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: int = logging.INFO,
    log_file: str | None = "pipeline.log",
) -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt, handlers=handlers)

    return logging.getLogger()

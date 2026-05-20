from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name registries — the single source of truth used by
# get_feature_columns() and every function that adds columns.
# ---------------------------------------------------------------------------

_LAG_COLS: list[str] = []          # populated by create_lag_features
_ROLLING_COLS: list[str] = []      # populated by create_rolling_features
_DATE_COLS: list[str] = []         # populated by create_date_features
_HOLIDAY_COLS: list[str] = []      # populated by create_holiday_features

# Columns that come from preprocessing and are used as features but are NOT
# created in this module (pass-through features).
_PASSTHROUGH_FEATURES: list[str] = [
    "Type_encoded",
    "Size",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "is_markdown_active",
    "CPI",
    "Unemployment",
    "IsHoliday",
]


# ---------------------------------------------------------------------------
# 1. Lag features
# ---------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    lags: Sequence[int] = (1, 2, 4, 12, 52),
) -> pd.DataFrame:
    global _LAG_COLS
    df = df.copy()

    groups = df.groupby(["Store", "Dept"], sort=False)["Weekly_Sales"]
    new_cols: list[str] = []

    for lag in lags:
        col = f"lag_{lag}"
        df[col] = groups.shift(lag)
        new_cols.append(col)

    _LAG_COLS = new_cols
    logger.info("Created %d lag feature(s): %s", len(new_cols), new_cols)

    # Warn for store-dept groups too small to fill the largest lag
    max_lag = max(lags)
    group_sizes = df.groupby(["Store", "Dept"]).size()
    tiny_groups = group_sizes[group_sizes <= max_lag]
    if not tiny_groups.empty:
        logger.warning(
            "%d (Store, Dept) group(s) have <= %d rows — lag_%d will be all-NaN "
            "for these groups: %s",
            len(tiny_groups),
            max_lag,
            max_lag,
            tiny_groups.index.tolist()[:5],   # truncate for readability
        )

    return df


# ---------------------------------------------------------------------------
# 2. Rolling features
# ---------------------------------------------------------------------------

def create_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (4, 12),
) -> pd.DataFrame:
    global _ROLLING_COLS
    df = df.copy()

    min_window = min(windows)
    new_cols: list[str] = []

    for w in windows:
        groups = df.groupby(["Store", "Dept"], sort=False)["Weekly_Sales"]
        # shift(1) before rolling: window never touches the current row (no leakage)
        shifted = groups.shift(1)

        # Mean and std for every window
        df[f"rolling_mean_{w}"] = (
            shifted.transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            shifted.transform(lambda s: s.rolling(w, min_periods=2).std())
        )
        new_cols += [f"rolling_mean_{w}", f"rolling_std_{w}"]

        # Min / max only for the shortest window (keeps feature count manageable)
        if w == min_window:
            df[f"rolling_min_{w}"] = (
                shifted.transform(lambda s: s.rolling(w, min_periods=1).min())
            )
            df[f"rolling_max_{w}"] = (
                shifted.transform(lambda s: s.rolling(w, min_periods=1).max())
            )
            new_cols += [f"rolling_min_{w}", f"rolling_max_{w}"]

    _ROLLING_COLS = new_cols
    logger.info("Created %d rolling feature(s): %s", len(new_cols), new_cols)
    return df


# ---------------------------------------------------------------------------
# 3. Date / calendar features
# ---------------------------------------------------------------------------

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    global _DATE_COLS
    df = df.copy()

    dt = df["Date"].dt
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["year"] = dt.year
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)
    df["day_of_year"] = dt.day_of_year

    new_cols = [
        "week_of_year", "month", "quarter", "year",
        "is_month_start", "is_month_end", "day_of_year",
    ]
    _DATE_COLS = new_cols
    logger.info("Created %d date feature(s): %s", len(new_cols), new_cols)
    return df


# ---------------------------------------------------------------------------
# 4. Holiday features
# ---------------------------------------------------------------------------

# Easter Sunday dates for the three years covered by the dataset.
# A ±7-day window around each date captures the full Easter shopping spike.
_EASTER_DATES: list[pd.Timestamp] = [
    pd.Timestamp("2010-04-04"),
    pd.Timestamp("2011-04-24"),
    pd.Timestamp("2012-04-08"),
]


def _thanksgiving_dates(years: Sequence[int]) -> set[pd.Timestamp]:
    # Thanksgiving = 4th Thursday of November; dataset uses Friday week-end dates
    dates: set[pd.Timestamp] = set()
    for year in years:
        # First Thursday in November
        nov1 = pd.Timestamp(year=year, month=11, day=1)
        # weekday(): Mon=0 … Thu=3
        days_to_first_thu = (3 - nov1.weekday()) % 7
        first_thu = nov1 + pd.Timedelta(days=days_to_first_thu)
        fourth_thu = first_thu + pd.Timedelta(weeks=3)
        dates.add(fourth_thu + pd.Timedelta(days=1))   # Friday of that week
    return dates


def _superbowl_dates(years: Sequence[int]) -> set[pd.Timestamp]:
    # Super Bowl is on the first Sunday of February; dataset marks the
    # surrounding week as a holiday, so we flag the first two Fridays of Feb
    dates: set[pd.Timestamp] = set()
    for year in years:
        feb1 = pd.Timestamp(year=year, month=2, day=1)
        days_to_first_fri = (4 - feb1.weekday()) % 7
        first_fri = feb1 + pd.Timedelta(days=days_to_first_fri)
        dates.add(first_fri)
        dates.add(first_fri + pd.Timedelta(weeks=1))  # second Friday
    return dates


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    global _HOLIDAY_COLS
    df = df.copy()

    years = df["Date"].dt.year.unique().tolist()

    # ── weeks_to_christmas ──────────────────────────────────────────────────
    def _weeks_to_xmas(date: pd.Timestamp) -> int:
        xmas = pd.Timestamp(year=date.year, month=12, day=25)
        delta_days = (xmas - date).days
        # If Christmas has passed, look to next year
        if delta_days < 0:
            xmas = pd.Timestamp(year=date.year + 1, month=12, day=25)
            delta_days = (xmas - date).days
        return int(min(delta_days // 7, 52))

    df["weeks_to_christmas"] = df["Date"].apply(_weeks_to_xmas)
    logger.debug("Added 'weeks_to_christmas'.")

    # ── weeks_since_last_holiday ────────────────────────────────────────────
    # Iterate in (Store, Dept, Date) order; track the last holiday date per
    # group in a dict.  Avoids groupby.apply which returns a DataFrame (not
    # a Series) in pandas 2.x when the function receives grouped columns.
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    weeks_since: list[int] = []
    last_holiday_date: dict[tuple[int, int], pd.Timestamp] = {}

    for i in range(len(df)):
        key = (int(df.at[i, "Store"]), int(df.at[i, "Dept"]))
        if key in last_holiday_date:
            weeks = max(0, (df.at[i, "Date"] - last_holiday_date[key]).days // 7)
        else:
            weeks = 52   # no prior holiday seen for this group
        weeks_since.append(weeks)
        if df.at[i, "IsHoliday"] == 1:
            last_holiday_date[key] = df.at[i, "Date"]

    df["weeks_since_last_holiday"] = weeks_since
    logger.debug("Added 'weeks_since_last_holiday'.")

    # ── is_thanksgiving_week ────────────────────────────────────────────────
    thanksgiving_fridays = _thanksgiving_dates(years)
    df["is_thanksgiving_week"] = df["Date"].isin(thanksgiving_fridays).astype(int)
    logger.debug(
        "Thanksgiving Fridays found: %s",
        sorted(str(d.date()) for d in thanksgiving_fridays),
    )

    # ── is_superbowl_week ───────────────────────────────────────────────────
    superbowl_fridays = _superbowl_dates(years)
    df["is_superbowl_week"] = (
        df["Date"].isin(superbowl_fridays) | (
            (df["Date"].dt.month == 2)
            & (df["Date"].dt.day <= 14)
            & (df["IsHoliday"] == 1)
        )
    ).astype(int)
    logger.debug(
        "Super Bowl candidate Fridays: %s",
        sorted(str(d.date()) for d in superbowl_fridays),
    )

    # ── is_easter_week ──────────────────────────────────────────────────────
    df["is_easter_week"] = df["Date"].apply(
        lambda d: int(any(abs((d - e).days) <= 7 for e in _EASTER_DATES))
    )
    logger.debug("Added 'is_easter_week'.")

    new_cols = [
        "weeks_to_christmas",
        "weeks_since_last_holiday",
        "is_thanksgiving_week",
        "is_superbowl_week",
        "is_easter_week",
    ]
    _HOLIDAY_COLS = new_cols
    logger.info("Created %d holiday feature(s): %s", len(new_cols), new_cols)
    return df


# ---------------------------------------------------------------------------
# 5. Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Order matters: date/holiday before lag/rolling; lag/rolling before NaN drop
    original_rows = len(df)
    original_cols = set(df.columns)

    logger.info("Starting feature engineering -- input: %d rows, %d cols",
                original_rows, len(original_cols))

    df = create_date_features(df)
    df = create_holiday_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Drop rows where any lag or rolling feature is NaN.
    # These arise from the start of each (Store, Dept) time series.
    leakage_guard_cols = _LAG_COLS + _ROLLING_COLS
    rows_before_drop = len(df)
    df = df.dropna(subset=leakage_guard_cols).reset_index(drop=True)
    rows_dropped = rows_before_drop - len(df)

    new_feature_cols = [c for c in df.columns if c not in original_cols]
    n_new = len(new_feature_cols)

    logger.info(
        "Feature engineering done -- %d rows (dropped %d), "
        "%d new cols, %d total features",
        len(df), rows_dropped, n_new, len(get_feature_columns()),
    )
    return df


# ---------------------------------------------------------------------------
# 6. Feature column registry
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    # Call run_feature_engineering() first to populate the registry lists
    return (
        _PASSTHROUGH_FEATURES
        + _DATE_COLS
        + _HOLIDAY_COLS
        + _LAG_COLS
        + _ROLLING_COLS
    )

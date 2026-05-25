from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kaggle holiday weighting factor
_HOLIDAY_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"y_true and y_pred must be numeric arrays: {exc}") from exc

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )


# ---------------------------------------------------------------------------
# 1. Core metric functions
# ---------------------------------------------------------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate_arrays(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = float(np.mean(np.abs(y_true - y_pred)))
    logger.debug("MAE = %.4f", result)
    return result


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate_arrays(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    logger.debug("RMSE = %.4f", result)
    return result


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_abs_threshold: float = 1.0,
) -> float:
    """Weighted MAPE (WMAPE) = sum|actual-pred| / sum(actual) × 100%.

    Weighting by actual sales prevents low-volume dept-weeks (returns,
    near-zero periods) from dominating the average.  Unweighted MAPE on
    retail dept-level data routinely reaches thousands-of-percent because
    a single $5-actual / $16k-predicted row contributes 319,900%.
    """
    _validate_arrays(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Exclude net-return weeks (negative actuals) and true-zero rows.
    positive_mask = y_true >= min_abs_threshold
    excluded = int(np.sum(~positive_mask))

    if excluded > 0:
        logger.warning(
            "MAPE: %d row(s) excluded (y_true < %.1f, includes negatives/returns) "
            "(%d / %d observations used).",
            excluded,
            min_abs_threshold,
            int(np.sum(positive_mask)),
            len(y_true),
        )

    if not np.any(positive_mask):
        logger.error("MAPE: all actual values are zero -- returning nan.")
        return float("nan")

    yt = y_true[positive_mask]
    yp = y_pred[positive_mask]
    # WMAPE: weight each row by its actual sales value so high-volume
    # weeks drive the metric, not micro-sales outliers.
    result = float(np.sum(np.abs(yt - yp)) / np.sum(yt) * 100.0)
    logger.debug("WMAPE = %.4f%%", result)
    return result


def compute_weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
) -> float:
    _validate_arrays(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), _HOLIDAY_WEIGHT, 1.0)

    result = float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))
    logger.debug("Weighted MAE = %.4f  (holiday weight=%.1f)", result, _HOLIDAY_WEIGHT)
    return result


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
    }
    logger.info(
        "Metrics — MAE: %.2f | RMSE: %.2f | MAPE: %.2f%%",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
    )
    return metrics


# ---------------------------------------------------------------------------
# 2. Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    df: pd.DataFrame,
    model_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    n_splits: int = 3,
    min_train_weeks: int = 52,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if "Date" not in df.columns or "Weekly_Sales" not in df.columns:
        raise ValueError("df must contain 'Date' and 'Weekly_Sales' columns.")

    df = df.sort_values("Date").reset_index(drop=True)
    unique_dates = df["Date"].sort_values().unique()
    total_weeks = len(unique_dates)

    required_weeks = min_train_weeks + n_splits
    if total_weeks < required_weeks:
        raise ValueError(
            f"Dataset has {total_weeks} unique weeks but needs at least "
            f"{required_weeks} "
            f"(min_train_weeks={min_train_weeks} + n_splits={n_splits})."
        )

    # Divide timeline into n_splits+1 equal blocks
    block_size = total_weeks // (n_splits + 1)
    logger.info(
        "walk_forward_cv: %d weeks -> block_size=%d, n_splits=%d",
        total_weeks,
        block_size,
        n_splits,
    )

    fold_metrics: list[dict[str, float]] = []

    for fold in range(1, n_splits + 1):
        train_end_idx = fold * block_size
        test_end_idx = (fold + 1) * block_size

        train_cutoff = unique_dates[train_end_idx - 1]
        test_cutoff = unique_dates[min(test_end_idx - 1, total_weeks - 1)]

        train_df = df[df["Date"] <= train_cutoff].reset_index(drop=True)
        test_df = df[
            (df["Date"] > train_cutoff) & (df["Date"] <= test_cutoff)
        ].reset_index(drop=True)

        if len(test_df) == 0:
            logger.warning("Fold %d: empty test set -- skipping.", fold)
            continue

        logger.info(
            "Fold %d | train: %s -> %s (%d rows) | test: %s -> %s (%d rows)",
            fold,
            train_df["Date"].min().date(),
            train_df["Date"].max().date(),
            len(train_df),
            test_df["Date"].min().date(),
            test_df["Date"].max().date(),
            len(test_df),
        )

        y_pred = np.asarray(model_fn(train_df, test_df), dtype=float)

        if len(y_pred) > 0 and np.all(y_pred == y_pred[0]):
            logger.error(
                "Fold %d: all %d predictions are identical (%.2f) — model likely "
                "used a fallback mean instead of training properly. Check training "
                "data size and feature engineering output.",
                fold, len(y_pred), float(y_pred[0]),
            )

        y_true = test_df["Weekly_Sales"].to_numpy(dtype=float)
        metrics = compute_all_metrics(y_true, y_pred)
        fold_metrics.append(metrics)

        logger.info(
            "Fold %d/%d done | %s",
            fold,
            n_splits,
            format_metrics_for_display(metrics),
        )

    if not fold_metrics:
        raise ValueError("No valid folds were produced — check your data length.")

    avg_metrics: dict[str, float] = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ("MAE", "RMSE", "MAPE")
    }
    logger.info(
        "walk_forward_cv complete (%d folds) — avg %s",
        len(fold_metrics),
        {k: f"{v:.4f}" for k, v in avg_metrics.items()},
    )
    return fold_metrics, avg_metrics


# ---------------------------------------------------------------------------
# 3. Model comparison
# ---------------------------------------------------------------------------

def compare_models(results_dict: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not results_dict:
        raise ValueError("results_dict must not be empty.")

    rows = []
    for model_name, metrics in results_dict.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": metrics.get("MAE", float("nan")),
                "RMSE": metrics.get("RMSE", float("nan")),
                "MAPE (%)": metrics.get("MAPE", float("nan")),
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("MAPE (%)", ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    logger.info(
        "Model comparison (sorted by MAPE):\n%s",
        comparison.to_string(index=False),
    )
    return comparison


# ---------------------------------------------------------------------------
# 4. Display formatting
# ---------------------------------------------------------------------------

def format_metrics_for_display(metrics_dict: dict[str, float]) -> str:
    def _fmt_dollar(val: float | None) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"${val:,.2f}"

    def _fmt_pct(val: float | None) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.2f}%"

    mae_str = _fmt_dollar(metrics_dict.get("MAE"))
    rmse_str = _fmt_dollar(metrics_dict.get("RMSE"))
    mape_str = _fmt_pct(metrics_dict.get("MAPE"))

    return f"MAE: {mae_str} | RMSE: {rmse_str} | MAPE: {mape_str}"

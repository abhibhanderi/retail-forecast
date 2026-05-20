from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compare_models,
    compute_all_metrics,
    compute_mape,
    compute_rmse,
    format_metrics_for_display,
    walk_forward_cv,
)
from src.models import MovingAverageForecaster


def _cv_df(n_weeks: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": d,
         "Weekly_Sales": float(rng.integers(5_000, 15_000)),
         "IsHoliday": 0, "Type": "A", "Type_encoded": 1, "Size": 150_000}
        for d in dates
    ])


def test_compute_rmse_known_values():
    # sqrt(mean([4, 4])) = sqrt(4) = 2.0
    result = compute_rmse([0, 0], [2, 2])
    assert abs(result - 2.0) < 1e-9


def test_compute_rmse_perfect_forecast():
    assert compute_rmse([100, 200, 300], [100, 200, 300]) == 0.0


def test_compute_all_metrics_returns_three_keys():
    metrics = compute_all_metrics([100, 200], [110, 190])
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}
    assert metrics["MAE"] == 10.0


def test_walk_forward_cv_produces_correct_fold_count():
    df = _cv_df(n_weeks=60)
    ma = MovingAverageForecaster(window=4)

    def model_fn(train_df, test_df):
        return ma.fit(train_df).predict(test_df)

    folds, avg = walk_forward_cv(df, model_fn, n_splits=2, min_train_weeks=10)
    assert len(folds) == 2
    assert "MAE" in avg and "RMSE" in avg and "MAPE" in avg


def test_walk_forward_cv_raises_on_insufficient_data():
    df = _cv_df(n_weeks=8)
    with pytest.raises(ValueError, match="needs at least"):
        walk_forward_cv(df, lambda t, e: np.zeros(len(e)),
                        n_splits=3, min_train_weeks=10)


def test_walk_forward_cv_raises_on_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="Date"):
        walk_forward_cv(df, lambda t, e: np.zeros(len(e)))


def test_format_metrics_for_display_includes_all_fields():
    metrics = {"MAE": 1234.5, "RMSE": 2345.6, "MAPE": 12.34}
    result = format_metrics_for_display(metrics)
    assert "MAE" in result and "RMSE" in result and "MAPE" in result
    assert "12.34%" in result


def test_format_metrics_for_display_nan_shows_na():
    metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    result = format_metrics_for_display(metrics)
    assert "N/A" in result


def test_validate_arrays_raises_on_non_numeric():
    with pytest.raises(TypeError, match="must be numeric"):
        compute_rmse(["a", "b"], [1, 2])


def test_validate_arrays_raises_on_empty_arrays():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_rmse([], [])


def test_validate_arrays_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_rmse([1, 2, 3], [1, 2])


def test_mape_returns_nan_when_all_actuals_zero():
    result = compute_mape([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert np.isnan(result)


def test_compare_models_raises_on_empty_dict():
    with pytest.raises(ValueError, match="must not be empty"):
        compare_models({})

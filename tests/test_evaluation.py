from __future__ import annotations

import math

from src.evaluation import (
    compare_models, compute_mae, compute_mape, compute_weighted_mae,
)


def test_mae_known_values():
    # 10 and 10 error on two samples = MAE of 10
    assert compute_mae([100, 200], [110, 190]) == 10.0


def test_mape_excludes_near_zero_actuals():
    # rows where |y_true| < 1.0 are silently dropped to avoid division noise
    # only the second pair (100, 110) is used → MAPE = 10%
    result = compute_mape([0, 100], [50, 110])
    assert math.isclose(result, 10.0)


def test_weighted_mae_holiday_multiplier():
    # holiday weeks are weighted 5× (Kaggle evaluation rule)
    # row 0: error=0, holiday (weight=5); row 1: error=100, regular (weight=1)
    # weighted MAE = (5*0 + 1*100) / (5+1) = 100/6
    result = compute_weighted_mae([200, 100], [200, 200], [True, False])
    assert math.isclose(result, 100 / 6, rel_tol=1e-9)


def test_compare_models_sorted_by_weighted_mae():
    results = {
        "ARIMA": {"MAE": 1200.0, "RMSE": 1600.0, "MAPE": 9.2},
        "LightGBM": {"MAE": 750.0, "RMSE": 1050.0, "MAPE": 5.8},
        "Prophet": {"MAE": 950.0, "RMSE": 1300.0, "MAPE": 7.1},
    }
    df = compare_models(results)
    assert df.iloc[0]["Model"] == "LightGBM"
    assert df.iloc[-1]["Model"] == "ARIMA"

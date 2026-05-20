from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models import MovingAverageForecaster
from src.run_evaluation import _cv_model


def _make_weekly_df(n_weeks: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": d,
         "Weekly_Sales": float(rng.integers(5_000, 15_000)),
         "IsHoliday": 0}
        for d in dates
    ])


def test_cv_model_returns_fold_metrics_and_avg():
    # min_train_weeks=52 (default) + n_splits=2 → need 54 unique weeks; use 60
    df = _make_weekly_df(60)
    ma = MovingAverageForecaster(window=4)
    folds, avg = _cv_model(ma, df, n_splits=2)

    assert len(folds) == 2
    assert "weighted_MAE" in avg
    assert all("MAE" in f and "weighted_MAE" in f for f in folds)


def test_cv_model_avg_weighted_mae_is_mean_of_folds():
    df = _make_weekly_df(60)
    ma = MovingAverageForecaster(window=4)
    folds, avg = _cv_model(ma, df, n_splits=2)

    expected = float(np.mean([f["weighted_MAE"] for f in folds]))
    assert abs(avg["weighted_MAE"] - expected) < 1e-9


def test_run_evaluation_end_to_end_with_fast_models(tmp_path, monkeypatch):
    import src.run_evaluation as re_mod

    # min_train_weeks=52 (default) + n_splits=3 → need 55 unique weeks; use 70 train
    rng = np.random.default_rng(7)
    dates_all = pd.date_range("2010-02-05", periods=80, freq="W-FRI")
    rows = [
        {"Store": 1, "Dept": 1, "Date": d,
         "Weekly_Sales": float(rng.integers(5_000, 15_000)),
         "IsHoliday": 0}
        for d in dates_all
    ]
    pd.DataFrame(rows[:70]).to_parquet(
        tmp_path / "train_processed.parquet", index=False
    )
    pd.DataFrame(rows[70:]).to_parquet(
        tmp_path / "test_processed.parquet", index=False
    )

    # Replace slow model constructors with fast MovingAverageForecaster instances
    monkeypatch.setattr(
        re_mod, "ARIMAForecaster", lambda *a, **kw: MovingAverageForecaster(4)
    )
    monkeypatch.setattr(
        re_mod, "ProphetForecaster", lambda *a, **kw: MovingAverageForecaster(4)
    )
    monkeypatch.setattr(
        re_mod, "XGBoostForecaster", lambda *a, **kw: MovingAverageForecaster(4)
    )
    monkeypatch.setattr(
        re_mod, "LightGBMForecaster", lambda *a, **kw: MovingAverageForecaster(4)
    )

    class _FakeEnsemble:
        def __init__(self, weights_path=None):
            pass

        def predict(self, df):
            return np.full(len(df), 7000.0)

        def get_name(self):
            return "Ensemble"

    monkeypatch.setattr(re_mod, "EnsembleForecaster", _FakeEnsemble)

    re_mod.run_evaluation(data_dir=tmp_path, models_dir=tmp_path)

    assert (tmp_path / "results_metrics.json").exists()
    assert (tmp_path / "results_cv_folds.csv").exists()
    assert (tmp_path / "results_ensemble_predictions.csv").exists()

    with open(tmp_path / "results_metrics.json") as f:
        data = json.load(f)
    assert "Ensemble" in data

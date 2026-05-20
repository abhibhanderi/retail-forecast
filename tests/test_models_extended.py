from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import joblib

import src.models as models_mod
from src.models import (
    EnsembleForecaster,
    MovingAverageForecaster,
    _aggregate_store_weekly,
    run_baseline_comparison,
    train_and_evaluate_model,
)


def _make_df(n_weeks: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    train_dates = pd.date_range("2010-02-05", periods=n_weeks, freq="W-FRI")
    test_dates = pd.date_range(
        train_dates[-1] + pd.Timedelta(weeks=1), periods=4, freq="W-FRI"
    )

    def _rows(dates):
        return [
            {"Store": 1, "Dept": 1, "Date": d,
             "Weekly_Sales": float(rng.integers(4_000, 12_000)),
             "IsHoliday": 0, "Type": "A", "Type_encoded": 1, "Size": 150_000}
            for d in dates
        ]

    return pd.DataFrame(_rows(train_dates)), pd.DataFrame(_rows(test_dates))


def test_aggregate_store_weekly_sums_by_store_date():
    df = pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": pd.Timestamp("2010-02-05"),
         "Weekly_Sales": 1000.0, "IsHoliday": 0},
        {"Store": 1, "Dept": 2, "Date": pd.Timestamp("2010-02-05"),
         "Weekly_Sales": 2000.0, "IsHoliday": 1},
    ])
    agg = _aggregate_store_weekly(df)
    assert agg.loc[0, "Weekly_Sales"] == 3000.0
    assert agg.loc[0, "IsHoliday"] == 1


def test_moving_average_raises_on_unfitted_predict():
    ma = MovingAverageForecaster(window=4)
    _, test = _make_df()
    with pytest.raises(RuntimeError, match="fit()"):
        ma.predict(test)


def test_moving_average_fallback_for_unseen_store():
    train, _ = _make_df()
    ma = MovingAverageForecaster(window=4).fit(train)
    unseen_test = pd.DataFrame([{
        "Store": 99, "Dept": 99, "Date": pd.Timestamp("2010-09-17"),
        "Weekly_Sales": 0.0, "IsHoliday": 0,
        "Type": "A", "Type_encoded": 1, "Size": 150_000,
    }])
    preds = ma.predict(unseen_test)
    assert preds[0] == pytest.approx(ma._fallback_mean)


def test_moving_average_invalid_window():
    with pytest.raises(ValueError, match="window must be"):
        MovingAverageForecaster(window=0)


def test_train_and_evaluate_model_returns_metrics(tmp_path):
    train, test = _make_df()
    model = MovingAverageForecaster(window=4)
    result = train_and_evaluate_model(model, train, test, tmp_path)
    assert result["model_name"] == model.get_name()
    assert "MAE" in result and "RMSE" in result and "MAPE" in result
    assert (tmp_path / "model_moving_average_4w.pkl").exists()


def _setup_ensemble(tmp_path):
    train, _ = _make_df()
    ma4 = MovingAverageForecaster(window=4).fit(train)
    ma12 = MovingAverageForecaster(window=12).fit(train)
    joblib.dump(ma4, tmp_path / "model_moving_average_4w.pkl")
    joblib.dump(ma12, tmp_path / "model_moving_average_12w.pkl")
    pd.DataFrame({
        "model": [ma4.get_name(), ma12.get_name()],
        "fold": [1, 1],
        "MAE": [500.0, 400.0],
        "RMSE": [700.0, 600.0],
        "MAPE": [10.0, 5.0],
    }).to_csv(tmp_path / "results_cv_folds.csv", index=False)


def test_ensemble_get_name(tmp_path):
    _setup_ensemble(tmp_path)
    ens = EnsembleForecaster(weights_path=tmp_path / "results_ensemble_weights.json")
    assert ens.get_name() == "Ensemble"


def test_ensemble_raises_when_cv_csv_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="results_cv_folds.csv"):
        EnsembleForecaster(weights_path=tmp_path / "results_ensemble_weights.json")


def test_ensemble_fit_returns_self(tmp_path):
    _setup_ensemble(tmp_path)
    ens = EnsembleForecaster(weights_path=tmp_path / "results_ensemble_weights.json")
    result = ens.fit(pd.DataFrame())
    assert result is ens


def _make_processed_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2010-02-05", periods=n, freq="W-FRI")
    return pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": d,
         "Weekly_Sales": float(rng.integers(4_000, 12_000)),
         "IsHoliday": 0}
        for d in dates
    ])


def test_run_baseline_comparison_with_fast_models(tmp_path, monkeypatch):
    train_df = _make_processed_df(20)
    test_df = _make_processed_df(5)
    train_df.to_parquet(tmp_path / "train_processed.parquet", index=False)
    test_df.to_parquet(tmp_path / "test_processed.parquet", index=False)

    # Replace slow ARIMA with a fast MovingAverageForecaster
    monkeypatch.setattr(
        models_mod, "ARIMAForecaster",
        lambda *a, **kw: MovingAverageForecaster(4),
    )

    result = run_baseline_comparison(
        data_dir=tmp_path,
        models_dir=tmp_path,
        include_tabular=False,
    )

    assert isinstance(result, dict)
    assert len(result) > 0
    assert (tmp_path / "results_metrics.json").exists()
    for metrics in result.values():
        assert "MAE" in metrics and "RMSE" in metrics and "MAPE" in metrics

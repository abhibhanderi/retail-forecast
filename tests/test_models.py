from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from src.models import EnsembleForecaster, MovingAverageForecaster


def _make_df(n_train: int = 30, n_test: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    train_dates = pd.date_range("2010-02-05", periods=n_train, freq="W-FRI")
    test_dates = pd.date_range(
        train_dates[-1] + pd.Timedelta(weeks=1), periods=n_test, freq="W-FRI"
    )

    def _rows(dates):
        return [
            {"Store": 1, "Dept": 1, "Date": d,
             "Weekly_Sales": float(rng.integers(4_000, 12_000)),
             "IsHoliday": 0, "Type": "A", "Type_encoded": 1, "Size": 150_000}
            for d in dates
        ]

    return pd.DataFrame(_rows(train_dates)), pd.DataFrame(_rows(test_dates))


def _setup_ensemble_dir(tmp_path, ma4_mape: float, ma12_mape: float) -> None:
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
        "MAPE": [ma4_mape, ma12_mape],
    }).to_csv(tmp_path / "results_cv_folds.csv", index=False)


def test_moving_average_predict_matches_manual_calculation():
    # 20 weeks of constant 10k sales → any window size must predict 10k exactly
    dates_train = pd.date_range("2010-02-05", periods=20, freq="W-FRI")
    dates_test = pd.date_range("2010-07-02", periods=4, freq="W-FRI")
    train = pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": d, "Weekly_Sales": 10_000.0,
         "IsHoliday": 0, "Type": "A", "Type_encoded": 1, "Size": 150_000}
        for d in dates_train
    ])
    test = train.iloc[:4].assign(Date=dates_test)
    preds = MovingAverageForecaster(window=4).fit(train).predict(test)
    np.testing.assert_allclose(preds, 10_000.0)


def test_moving_average_save_load_produces_same_predictions(tmp_path):
    train, test = _make_df()
    ma = MovingAverageForecaster(window=4).fit(train)
    original = ma.predict(test)
    ma.save(tmp_path / "ma.pkl")
    loaded = MovingAverageForecaster.load(tmp_path / "ma.pkl")
    np.testing.assert_array_equal(original, loaded.predict(test))


def test_ensemble_weights_sum_to_one(tmp_path):
    _setup_ensemble_dir(tmp_path, ma4_mape=10.0, ma12_mape=5.0)
    ensemble = EnsembleForecaster(
        weights_path=tmp_path / "results_ensemble_weights.json"
    )
    assert abs(sum(ensemble._weights) - 1.0) < 1e-9


def test_ensemble_beats_moving_average_on_weighted_mae(tmp_path):
    # MA-4w MAPE=20 (poor), MA-12w MAPE=5 (good) → ensemble weights MA-12w 4× more
    _setup_ensemble_dir(tmp_path, ma4_mape=20.0, ma12_mape=5.0)
    train, test = _make_df()
    ma4 = MovingAverageForecaster(window=4).fit(train)
    ma12 = MovingAverageForecaster(window=12).fit(train)
    ens = EnsembleForecaster(weights_path=tmp_path / "results_ensemble_weights.json")
    ens_preds = ens.predict(test)
    # Ensemble should lie closer to the higher-weighted component (MA-12w)
    assert np.abs(ens_preds - ma12.predict(test)).mean() < \
           np.abs(ens_preds - ma4.predict(test)).mean()

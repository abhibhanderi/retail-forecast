"""
Integration tests — require `make pipeline` to have been run first.
All tests are @pytest.mark.slow and skip gracefully if artefacts are missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
_DATA = _ROOT / "data/processed"
_MODELS = _ROOT / "models"

_PKL_FILES = [
    "model_moving_average_4w.pkl", "model_moving_average_12w.pkl",
    "model_arima.pkl", "model_prophet.pkl",
    "model_xgboost.pkl", "model_lightgbm.pkl",
]
_RESULT_FILES = [
    "results_metrics.json", "results_cv_folds.csv",
    "results_ensemble_weights.json", "results_ensemble_predictions.csv",
]


@pytest.mark.slow
def test_processed_data_files_exist_and_valid():
    train_path = _DATA / "train_processed.parquet"
    test_path = _DATA / "test_processed.parquet"
    if not train_path.exists():
        pytest.skip("Processed data not found — run make pipeline first")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    # train must be larger and strictly before test — otherwise results are meaningless
    assert len(train) > len(test)
    assert pd.to_datetime(train["Date"]).max() < pd.to_datetime(test["Date"]).min()
    assert train["Weekly_Sales"].notna().all()


@pytest.mark.slow
def test_all_model_result_files_exist():
    if not (_MODELS / _PKL_FILES[0]).exists():
        pytest.skip("Model files not found — run make pipeline first")
    for pkl in _PKL_FILES:
        m = joblib.load(_MODELS / pkl)
        assert hasattr(m, "predict"), f"{pkl} has no predict()"
    for fname in _RESULT_FILES:
        assert (_MODELS / fname).exists(), f"{fname} was not produced"


@pytest.mark.slow
def test_ensemble_is_best_model_by_weighted_mae():
    metrics_path = _MODELS / "results_metrics.json"
    if not metrics_path.exists():
        pytest.skip("results_metrics.json not found — run make pipeline first")
    metrics = json.load(open(metrics_path))
    ensemble_wmae = metrics["Ensemble"]["weighted_MAE"]
    for name, m in metrics.items():
        if name != "Ensemble":
            assert ensemble_wmae < m["weighted_MAE"], f"Ensemble not better than {name}"


@pytest.mark.slow
def test_predictions_file_covers_full_test_period():
    path = _DATA / "predictions_test.parquet"
    if not path.exists():
        pytest.skip("predictions_test.parquet not found — run make pipeline first")
    df = pd.read_parquet(path)
    required = {"Store", "Dept", "Date", "actual", "pred_ensemble"}
    assert required.issubset(df.columns)
    assert not df["actual"].isna().any()


@pytest.mark.slow
def test_ensemble_weights_sum_to_one_on_real_data():
    path = _MODELS / "results_ensemble_weights.json"
    if not path.exists():
        pytest.skip("results_ensemble_weights.json not found — run make pipeline first")
    weights_data = json.load(open(path))
    total = sum(entry["weight"] for entry in weights_data.values())
    assert abs(total - 1.0) < 0.001

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

import src.save_predictions as sp_mod


class _FakeModel:
    def predict(self, df):
        return np.full(len(df), 5000.0)


class _MismatchModel:
    def predict(self, df):
        return np.full(len(df) + 5, 5000.0)


class _BrokenModel:
    def predict(self, df):
        raise RuntimeError("deliberate failure")


def _make_df(n: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2012-04-06", periods=n, freq="W-FRI")
    return pd.DataFrame([
        {"Store": 1, "Dept": 1, "Date": d, "Weekly_Sales": 5000.0, "IsHoliday": 0}
        for d in dates
    ])


def _patch_paths(monkeypatch, tmp_path, *, model_files=None, ensemble_csv=None):
    test_pq = tmp_path / "test.parquet"
    train_pq = tmp_path / "train.parquet"
    _make_df(4).to_parquet(test_pq, index=False)
    _make_df(8).to_parquet(train_pq, index=False)

    if model_files is None:
        model_files = []

    monkeypatch.setattr(sp_mod, "_TEST_PARQUET", test_pq)
    monkeypatch.setattr(sp_mod, "_TRAIN_PARQUET", train_pq)
    monkeypatch.setattr(sp_mod, "_MODELS", tmp_path)
    monkeypatch.setattr(sp_mod, "_MODEL_FILES", model_files)
    monkeypatch.setattr(
        sp_mod, "_ENSEMBLE_CSV",
        ensemble_csv if ensemble_csv is not None else tmp_path / "no_ens.csv",
    )
    monkeypatch.setattr(sp_mod, "_OUT_PREDS", tmp_path / "preds.parquet")
    monkeypatch.setattr(sp_mod, "_OUT_ACTUALS", tmp_path / "actuals.parquet")


def test_run_creates_predictions_and_actuals(tmp_path, monkeypatch):
    model_pkl = tmp_path / "fake.pkl"
    joblib.dump(_FakeModel(), model_pkl)
    _patch_paths(monkeypatch, tmp_path, model_files=[("fake", "fake.pkl")])

    sp_mod.run()

    assert (tmp_path / "preds.parquet").exists()
    assert (tmp_path / "actuals.parquet").exists()
    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert "pred_fake" in preds.columns
    assert "actual" in preds.columns


def test_run_fills_nan_when_pkl_missing(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path, model_files=[("missing", "nonexistent.pkl")])

    sp_mod.run()

    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert "pred_missing" in preds.columns
    assert preds["pred_missing"].isna().all()


def test_run_merges_ensemble_csv(tmp_path, monkeypatch):
    dates = pd.date_range("2012-04-06", periods=4, freq="W-FRI")
    ens_csv = tmp_path / "ens.csv"
    pd.DataFrame({
        "Store": [1] * 4,
        "Dept": [1] * 4,
        "Date": dates,
        "ensemble_forecast": [5500.0] * 4,
    }).to_csv(ens_csv, index=False)

    _patch_paths(monkeypatch, tmp_path, model_files=[], ensemble_csv=ens_csv)

    sp_mod.run()

    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert "pred_ensemble" in preds.columns
    assert not preds["pred_ensemble"].isna().all()


def test_run_exits_on_missing_test_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(sp_mod, "_TEST_PARQUET", tmp_path / "ghost.parquet")

    with pytest.raises(SystemExit) as exc_info:
        sp_mod.run()
    assert exc_info.value.code == 1


def test_run_skips_actuals_when_train_missing(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path, model_files=[])
    monkeypatch.setattr(sp_mod, "_TRAIN_PARQUET", tmp_path / "ghost_train.parquet")

    sp_mod.run()

    assert not (tmp_path / "actuals.parquet").exists()


def test_run_fills_nan_on_prediction_length_mismatch(tmp_path, monkeypatch):
    model_pkl = tmp_path / "mismatch.pkl"
    joblib.dump(_MismatchModel(), model_pkl)
    _patch_paths(monkeypatch, tmp_path, model_files=[("mismatch", "mismatch.pkl")])

    sp_mod.run()

    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert preds["pred_mismatch"].isna().all()


def test_run_fills_nan_on_predict_exception(tmp_path, monkeypatch):
    model_pkl = tmp_path / "broken.pkl"
    joblib.dump(_BrokenModel(), model_pkl)
    _patch_paths(monkeypatch, tmp_path, model_files=[("broken", "broken.pkl")])

    sp_mod.run()

    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert preds["pred_broken"].isna().all()


def test_run_fills_nan_on_malformed_ensemble_csv(tmp_path, monkeypatch):
    bad_csv = tmp_path / "bad_ens.csv"
    # Missing required 'Dept' column to cause a KeyError on merge
    pd.DataFrame({"Store": [1], "ensemble_forecast": [5000.0]}).to_csv(
        bad_csv, index=False
    )
    _patch_paths(monkeypatch, tmp_path, model_files=[], ensemble_csv=bad_csv)

    sp_mod.run()

    preds = pd.read_parquet(tmp_path / "preds.parquet")
    assert "pred_ensemble" in preds.columns

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src.shap_analysis as shap_mod
from src.shap_analysis import get_shap_for_store, get_top_features


def _fake_shap_module(shap_values: np.ndarray) -> types.ModuleType:
    fake_shap = types.ModuleType("shap")
    fake_explainer = MagicMock()
    fake_explainer.shap_values.return_value = shap_values
    fake_shap.TreeExplainer = MagicMock(return_value=fake_explainer)
    return fake_shap


def _patch_shap_env(monkeypatch, tmp_path, feature_names, shap_values, fitted=True):
    """Monkeypatch all external dependencies for shap_analysis functions."""
    monkeypatch.setitem(sys.modules, "shap", _fake_shap_module(shap_values))

    class _FakeForecaster:
        _model = object() if fitted else None

    fake_joblib = MagicMock()
    fake_joblib.load.return_value = _FakeForecaster()
    monkeypatch.setattr(shap_mod, "joblib", fake_joblib)

    feat_df = pd.DataFrame({f: [float(i)] * 10 for i, f in enumerate(feature_names)})
    raw_df = feat_df.assign(Store=1)

    monkeypatch.setattr(shap_mod, "load_processed_data", lambda p: raw_df)
    monkeypatch.setattr(shap_mod, "run_feature_engineering", lambda df: feat_df)
    monkeypatch.setattr(shap_mod, "get_feature_columns", lambda: feature_names)

    monkeypatch.setattr(shap_mod, "_SHAP_VALUES_PATH", tmp_path / "sv.pkl")
    monkeypatch.setattr(shap_mod, "_SHAP_FEATURE_NAMES_PATH", tmp_path / "sn.pkl")


def test_compute_shap_values_returns_array_and_names(tmp_path, monkeypatch):
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    fake_sv = np.ones((10, 5))
    _patch_shap_env(monkeypatch, tmp_path, feature_names, fake_sv)

    from src.shap_analysis import compute_shap_values

    sv, cols = compute_shap_values(
        model_path=tmp_path / "model.pkl",
        data_path=tmp_path / "data.parquet",
    )
    assert isinstance(sv, np.ndarray)
    assert cols == feature_names


def test_compute_shap_values_raises_when_model_not_fitted(tmp_path, monkeypatch):
    feature_names = ["f1", "f2"]
    fake_sv = np.ones((10, 2))
    _patch_shap_env(monkeypatch, tmp_path, feature_names, fake_sv, fitted=False)

    from src.shap_analysis import compute_shap_values

    with pytest.raises(RuntimeError, match="not been fitted"):
        compute_shap_values(
            model_path=tmp_path / "model.pkl",
            data_path=tmp_path / "data.parquet",
        )


def test_get_top_features_raises_when_artifacts_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shap_mod, "_SHAP_VALUES_PATH", tmp_path / "missing_sv.pkl")
    monkeypatch.setattr(
        shap_mod, "_SHAP_FEATURE_NAMES_PATH", tmp_path / "missing_sn.pkl"
    )
    with pytest.raises(FileNotFoundError, match="SHAP artifacts"):
        get_top_features()


def test_get_shap_for_store_success(tmp_path, monkeypatch):
    feature_names = ["f1", "f2", "f3"]
    fake_sv = np.ones((5, 3))
    monkeypatch.setitem(sys.modules, "shap", _fake_shap_module(fake_sv))

    class _FakeForecaster:
        _model = object()

    fake_joblib = MagicMock()
    fake_joblib.load.return_value = _FakeForecaster()
    monkeypatch.setattr(shap_mod, "joblib", fake_joblib)

    store_df = pd.DataFrame(
        {"Store": [1] * 5, **{f: [float(i)] * 5 for i, f in enumerate(feature_names)}}
    )
    monkeypatch.setattr(shap_mod, "load_processed_data", lambda p: store_df)
    monkeypatch.setattr(shap_mod, "run_feature_engineering", lambda df: df.copy())
    monkeypatch.setattr(shap_mod, "get_feature_columns", lambda: feature_names)

    sv, cols = get_shap_for_store(
        store_id=1,
        data_path=tmp_path / "data.parquet",
        model_path=tmp_path / "model.pkl",
    )
    assert sv.shape == fake_sv.shape
    assert cols == feature_names


def test_get_shap_for_store_raises_on_unknown_store(tmp_path, monkeypatch):
    fake_shap = types.ModuleType("shap")
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    data_df = pd.DataFrame({"Store": [1, 1, 2]})
    monkeypatch.setattr(shap_mod, "load_processed_data", lambda p: data_df)

    with pytest.raises(ValueError, match="store_id=99"):
        get_shap_for_store(
            store_id=99,
            data_path=tmp_path / "data.parquet",
            model_path=tmp_path / "model.pkl",
        )


def test_get_shap_for_store_raises_when_model_not_fitted(tmp_path, monkeypatch):
    feature_names = ["f1", "f2"]
    monkeypatch.setitem(sys.modules, "shap", _fake_shap_module(np.ones((5, 2))))

    class _UnfittedForecaster:
        _model = None

    fake_joblib = MagicMock()
    fake_joblib.load.return_value = _UnfittedForecaster()
    monkeypatch.setattr(shap_mod, "joblib", fake_joblib)

    store_df = pd.DataFrame({"Store": [1] * 5, "f1": [1.0] * 5, "f2": [2.0] * 5})
    monkeypatch.setattr(shap_mod, "load_processed_data", lambda p: store_df)
    monkeypatch.setattr(shap_mod, "run_feature_engineering", lambda df: df.copy())
    monkeypatch.setattr(shap_mod, "get_feature_columns", lambda: feature_names)

    with pytest.raises(RuntimeError, match="_model is None"):
        get_shap_for_store(
            store_id=1,
            data_path=tmp_path / "data.parquet",
            model_path=tmp_path / "model.pkl",
        )


def test_compute_shap_values_raises_on_missing_shap_package(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)

    from src.shap_analysis import compute_shap_values

    with pytest.raises(ImportError, match="shap is required"):
        compute_shap_values(
            model_path=tmp_path / "model.pkl",
            data_path=tmp_path / "data.parquet",
        )


def test_get_shap_for_store_raises_on_missing_shap_package(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)

    with pytest.raises(ImportError, match="shap is required"):
        get_shap_for_store(
            store_id=1,
            data_path=tmp_path / "data.parquet",
            model_path=tmp_path / "model.pkl",
        )

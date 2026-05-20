from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

import src.shap_analysis as shap_mod
from src.shap_analysis import get_plain_english_explanation, get_top_features


@pytest.fixture()
def fake_shap_artifacts(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    shap_values = np.abs(rng.random((50, 5)))
    feature_names = ["lag_1", "rolling_mean_4", "week_of_year", "Size", "IsHoliday"]
    sv_path = tmp_path / "shap_values.pkl"
    fn_path = tmp_path / "shap_feature_names.pkl"
    joblib.dump(shap_values, sv_path)
    joblib.dump(feature_names, fn_path)
    monkeypatch.setattr(shap_mod, "_SHAP_VALUES_PATH", sv_path)
    monkeypatch.setattr(shap_mod, "_SHAP_FEATURE_NAMES_PATH", fn_path)
    return shap_values, feature_names


def test_get_top_features_sorted_descending(fake_shap_artifacts):
    df = get_top_features(n=5)
    values = df["mean_abs_shap"].tolist()
    assert values == sorted(values, reverse=True)


def test_get_top_features_n_parameter_respected(fake_shap_artifacts):
    assert len(get_top_features(n=3)) == 3


def test_plain_english_explanation_returns_five_strings():
    df = pd.DataFrame({
        "feature": ["lag_1", "rolling_mean_4", "week_of_year", "Size", "IsHoliday"],
        "mean_abs_shap": [0.5, 0.4, 0.3, 0.2, 0.1],
    })
    result = get_plain_english_explanation(df)
    assert len(result) == 5
    assert all(isinstance(s, str) for s in result)

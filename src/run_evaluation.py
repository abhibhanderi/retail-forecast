from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, compute_weighted_mae, walk_forward_cv
from src.models import (
    ARIMAForecaster,
    EnsembleForecaster,
    LightGBMForecaster,
    MovingAverageForecaster,
    ProphetForecaster,
    XGBoostForecaster,
    _PKL_SAVE_NAMES,
)
from src.utils import load_processed_data, setup_logging

logger = logging.getLogger(__name__)

_N_SPLITS = 3


def _cv_model(
    model,
    train_df: pd.DataFrame,
    n_splits: int = _N_SPLITS,
) -> tuple[list[dict], dict]:
    # fold_capture and fold_metrics stay in sync because walk_forward_cv
    # only calls model_fn when len(test_df) > 0, so both lists grow together.
    fold_capture: list[dict] = []

    def model_fn(fold_train: pd.DataFrame, fold_test: pd.DataFrame) -> np.ndarray:
        model.fit(fold_train)
        preds = model.predict(fold_test)
        fold_capture.append(
            {
                "y_true": fold_test["Weekly_Sales"].to_numpy(dtype=float),
                "y_pred": np.asarray(preds, dtype=float),
                "is_holiday": fold_test["IsHoliday"].to_numpy(),
            }
        )
        return preds

    fold_metrics, avg_metrics = walk_forward_cv(train_df, model_fn, n_splits=n_splits)

    for fold_dict, cap in zip(fold_metrics, fold_capture):
        fold_dict["weighted_MAE"] = compute_weighted_mae(
            cap["y_true"], cap["y_pred"], cap["is_holiday"]
        )

    avg_metrics["weighted_MAE"] = float(
        np.mean([m["weighted_MAE"] for m in fold_metrics])
    )

    return fold_metrics, avg_metrics


def run_evaluation(
    data_dir: str | Path = "data/processed/",
    models_dir: str | Path = "models/",
) -> None:
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data from '%s'", data_dir)
    train_df = load_processed_data(data_dir / "train_processed.parquet")
    test_df = load_processed_data(data_dir / "test_processed.parquet")
    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df),
        train_df["Date"].min().date(),
        train_df["Date"].max().date(),
        len(test_df),
        test_df["Date"].min().date(),
        test_df["Date"].max().date(),
    )

    models = [
        MovingAverageForecaster(window=4),
        ARIMAForecaster(top_n_stores=10),
        ProphetForecaster(top_n_stores=10),
        XGBoostForecaster(),
        LightGBMForecaster(),
    ]

    all_fold_rows: list[dict] = []
    avg_by_model: dict[str, dict] = {}

    for model in models:
        model_name = model.get_name()
        logger.info("Starting walk-forward CV for %s", model_name)

        fold_metrics, avg_metrics = _cv_model(model, train_df)
        avg_by_model[model_name] = avg_metrics

        for fold_idx, fold_dict in enumerate(fold_metrics, start=1):
            all_fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "MAE": fold_dict["MAE"],
                    "RMSE": fold_dict["RMSE"],
                    "MAPE": fold_dict["MAPE"],
                    "weighted_MAE": fold_dict["weighted_MAE"],
                }
            )

        # Re-fit on the full training set and save pkl.
        # _cv_model leaves the model fitted on the last CV fold only; this
        # full-data fit is what EnsembleForecaster will load and use.
        logger.info("Fitting %s on full training set for ensemble ...", model_name)
        model.fit(train_df)
        _safe = _PKL_SAVE_NAMES.get(
            model_name,
            model_name.replace("(", "_").replace(")", "")
            .replace("=", "").replace(",", "_"),
        )
        model.save(models_dir / f"{_safe}.pkl")

    # MA(w=12) is not part of CV/ensemble but must be saved so the dashboard
    # can load it.  Fit here so the pkl is always written as src.models.*
    # (not __main__.* which happens when src.models is run directly).
    logger.info("Fitting MovingAverage(w=12) on full training set (dashboard only) ...")
    ma12 = MovingAverageForecaster(window=12)
    ma12.fit(train_df)
    ma12.save(models_dir / "model_moving_average_12w.pkl")

    # ---- fold-level CSV ----
    cv_df = pd.DataFrame(
        all_fold_rows,
        columns=["model", "fold", "MAE", "RMSE", "MAPE", "weighted_MAE"],
    )
    cv_path = models_dir / "results_cv_folds.csv"
    cv_df.to_csv(cv_path, index=False)
    logger.info("CV fold results saved -> '%s'", cv_path)

    # ---- per-model averages JSON ----
    json_path = models_dir / "results_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                name: {k: float(v) for k, v in m.items()}
                for name, m in avg_by_model.items()
            },
            f,
            indent=2,
        )
    logger.info("Results metrics saved -> '%s'", json_path)

    # ---- ensemble evaluation on held-out test set ----
    logger.info("Ensemble (inverse-MAPE weighted, evaluated on held-out test set)")
    ensemble = EnsembleForecaster(
        weights_path=models_dir / "results_ensemble_weights.json"
    )
    ensemble_preds = ensemble.predict(test_df)

    y_test = test_df["Weekly_Sales"].to_numpy(dtype=float)
    is_holiday_test = test_df["IsHoliday"].to_numpy()
    ensemble_metrics = compute_all_metrics(y_test, ensemble_preds)
    ensemble_metrics["weighted_MAE"] = compute_weighted_mae(
        y_test, ensemble_preds, is_holiday_test
    )
    logger.info(
        "Ensemble test-set metrics: %s",
        {k: f"{v:.4f}" for k, v in ensemble_metrics.items()},
    )

    # Append "Ensemble" to results_metrics.json without touching existing keys
    with open(json_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    baseline_data["Ensemble"] = {k: float(v) for k, v in ensemble_metrics.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    logger.info("Ensemble metrics appended -> '%s'", json_path)

    # Save ensemble_predictions.csv
    ens_pred_df = test_df[["Store", "Dept", "Date"]].copy()
    ens_pred_df["actual"] = y_test
    ens_pred_df["ensemble_forecast"] = ensemble_preds
    ens_pred_path = models_dir / "results_ensemble_predictions.csv"
    ens_pred_df.to_csv(ens_pred_path, index=False)
    logger.info("Ensemble predictions saved -> '%s'", ens_pred_path)

    # ---- final comparison table ----
    # avg_by_model holds CV averages for the 5 components;
    # ensemble uses test-set metrics.
    avg_by_model["Ensemble"] = ensemble_metrics
    summary = (
        pd.DataFrame([{"Model": name, **m} for name, m in avg_by_model.items()])
        .sort_values("weighted_MAE", ascending=True)
        .reset_index(drop=True)
    )
    summary.index += 1

    col_order = ["Model", "MAE", "RMSE", "MAPE", "weighted_MAE"]
    summary = summary[col_order]

    logger.info(
        "Final Comparison (CV avg for individual models | test-set for Ensemble):\n%s",
        summary.to_string(float_format=lambda x: f"{x:,.2f}", col_space=15),
    )


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Walk-forward CV for all 5 forecasting models."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/",
        help="Directory containing train/test parquet files (default: data/processed/)",
    )
    parser.add_argument(
        "--models-dir",
        default="models/",
        help="Directory for cv_results.csv and results.json (default: models/)",
    )
    args = parser.parse_args()

    run_evaluation(data_dir=args.data_dir, models_dir=args.models_dir)

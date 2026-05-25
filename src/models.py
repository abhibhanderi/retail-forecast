from __future__ import annotations

import abc
import contextlib
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation import (
    compute_all_metrics, compare_models, format_metrics_for_display,
)

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models")

# Canonical pkl stem for each model name returned by get_name()
_PKL_SAVE_NAMES: dict[str, str] = {
    "MovingAverage(w=4)":            "model_moving_average_4w",
    "MovingAverage(w=12)":           "model_moving_average_12w",
    "ARIMA(auto,nonseasonal,top10)": "model_arima",
    "Prophet(top10)":                "model_prophet",
    "XGBoost(n=500,d=6)":           "model_xgboost",
    "LightGBM(n=500,leaves=63)":    "model_lightgbm",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_store_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["Store", "Date"], sort=True)
        .agg(
            Weekly_Sales=("Weekly_Sales", "sum"),
            IsHoliday=("IsHoliday", "max"),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# 1. Abstract base class
# ---------------------------------------------------------------------------

class BaseForecaster(abc.ABC):

    @abc.abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseForecaster":
        ...

    @abc.abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        ...

    def get_name(self) -> str:
        return self.__class__.__name__

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s -> '%s'", self.get_name(), path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        model = joblib.load(Path(path))
        logger.info("Loaded %s from '%s'", model.get_name(), path)
        return model


# ---------------------------------------------------------------------------
# 2. Moving Average Forecaster
# ---------------------------------------------------------------------------

class MovingAverageForecaster(BaseForecaster):

    def __init__(self, window: int = 4) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self.window = window
        self._store_dept_means: dict[tuple[int, int], float] = {}
        self._fallback_mean: float = 0.0

    def get_name(self) -> str:
        return f"MovingAverage(w={self.window})"

    def fit(self, train_df: pd.DataFrame) -> "MovingAverageForecaster":
        self._store_dept_means = {}
        df_sorted = train_df.sort_values(["Store", "Dept", "Date"])

        groups_fitted = 0
        for (store, dept), group in df_sorted.groupby(["Store", "Dept"]):
            sales = group["Weekly_Sales"].values
            # Use the last `window` weeks; fall back to all available if fewer
            tail = sales[-self.window:] if len(sales) >= self.window else sales
            if len(tail) == 0:
                continue
            self._store_dept_means[(int(store), int(dept))] = float(np.mean(tail))
            groups_fitted += 1

        all_sales = train_df["Weekly_Sales"].values
        self._fallback_mean = float(np.mean(all_sales)) if len(all_sales) > 0 else 0.0

        logger.info(
            "%s fitted on %d (Store, Dept) groups  |  fallback mean: $%.2f",
            self.get_name(), groups_fitted, self._fallback_mean,
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not self._store_dept_means:
            raise RuntimeError(
                f"{self.get_name()} has not been fitted. Call fit() first."
            )

        unseen: set[tuple[int, int]] = set()
        preds: list[float] = []

        for _, row in test_df.iterrows():
            key = (int(row["Store"]), int(row["Dept"]))
            if key in self._store_dept_means:
                preds.append(self._store_dept_means[key])
            else:
                unseen.add(key)
                preds.append(self._fallback_mean)

        if unseen:
            logger.warning(
                "%s: %d unseen (Store, Dept) group(s) in test set — "
                "using fallback mean $%.2f. First few: %s",
                self.get_name(), len(unseen), self._fallback_mean,
                sorted(unseen)[:3],
            )

        return np.array(preds, dtype=float)


# ---------------------------------------------------------------------------
# 3. ARIMA Forecaster
# ---------------------------------------------------------------------------

class ARIMAForecaster(BaseForecaster):

    def __init__(
        self,
        auto: bool = True,
        seasonal: bool = False,
        top_n_stores: int = 10,
    ) -> None:
        self.auto = auto
        self.seasonal = seasonal
        self.top_n_stores = top_n_stores

        self._store_models: dict[int, Any] = {}       # store_id → fitted ARIMA
        self._store_histories: dict[int, np.ndarray] = {}  # store_id → training series
        self._fallback: MovingAverageForecaster = MovingAverageForecaster(window=4)
        self._fitted_stores: set[int] = set()

    def get_name(self) -> str:
        suffix = "seasonal" if self.seasonal else "nonseasonal"
        mode = "auto" if self.auto else "fixed"
        return f"ARIMA({mode},{suffix},top{self.top_n_stores})"

    def fit(self, train_df: pd.DataFrame) -> "ARIMAForecaster":
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise ImportError(
                "pmdarima is required for ARIMAForecaster. "
                "Install it with: pip install pmdarima"
            ) from exc

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):  
                return iterable

        self._fallback.fit(train_df)

        store_agg = _aggregate_store_weekly(train_df)

        store_totals = (
            store_agg.groupby("Store")["Weekly_Sales"]
            .sum()
            .sort_values(ascending=False)
        )
        selected_stores = store_totals.head(self.top_n_stores).index.tolist()

        logger.info(
            "%s: fitting ARIMA on %d store(s): %s",
            self.get_name(), len(selected_stores), selected_stores,
        )

        self._store_models = {}
        self._store_histories = {}
        self._fitted_stores = set()

        arima_kwargs: dict[str, Any] = dict(
            seasonal=self.seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
        if self.seasonal:
            arima_kwargs["m"] = 52

        for store_id in tqdm(selected_stores, desc="Fitting ARIMA", unit="store"):
            store_series = (
                store_agg[store_agg["Store"] == store_id]
                .sort_values("Date")["Weekly_Sales"]
                .values
                .astype(float)
            )

            if len(store_series) < 13:
                logger.warning(
                    "Store %d: only %d weeks of data — skipping ARIMA, "
                    "will use fallback MA.",
                    store_id, len(store_series),
                )
                continue

            self._store_histories[store_id] = store_series

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.auto:
                        model = pm.auto_arima(store_series, **arima_kwargs)
                    else:
                        model = pm.ARIMA(order=(1, 1, 1))
                        model.fit(store_series)

                self._store_models[store_id] = model
                self._fitted_stores.add(store_id)
                logger.debug(
                    "Store %d: fitted %s  (AIC=%.1f)",
                    store_id,
                    model.summary().tables[0].data[0][1]
                    if hasattr(model, "summary") else "ARIMA",
                    model.aic() if hasattr(model, "aic") else float("nan"),
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Store %d: ARIMA fitting failed (%s) — using fallback MA.",
                    store_id, exc,
                )

        logger.info(
            "%s fit complete — %d/%d stores fitted with ARIMA, "
            "%d falling back to MA.",
            self.get_name(),
            len(self._fitted_stores),
            len(selected_stores),
            len(selected_stores) - len(self._fitted_stores),
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_fallback") or not self._fallback._store_dept_means:
            raise RuntimeError(
                f"{self.get_name()} has not been fitted. Call fit() first."
            )

        store_date_forecasts: dict[tuple[int, pd.Timestamp], float] = {}

        test_agg = _aggregate_store_weekly(test_df)

        for store_id in test_agg["Store"].unique():
            store_test = (
                test_agg[test_agg["Store"] == store_id]
                .sort_values("Date")
            )
            n_periods = len(store_test)

            if store_id in self._fitted_stores:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        forecast_vals, _ = self._store_models[store_id].predict(
                            n_periods=n_periods, return_conf_int=True
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Store %d: ARIMA predict failed (%s) — using MA fallback.",
                        store_id, exc,
                    )
                    forecast_vals = np.full(
                        n_periods, self._fallback._fallback_mean
                    )
            else:
                store_ma = self._fallback._store_dept_means
                dept_means = [
                    v for (s, _d), v in store_ma.items() if s == store_id
                ]
                fallback_val = (
                    float(np.sum(dept_means)) if dept_means
                    else self._fallback._fallback_mean
                )
                forecast_vals = np.full(n_periods, fallback_val)

            for date, fval in zip(store_test["Date"].values, forecast_vals):
                store_date_forecasts[(int(store_id), pd.Timestamp(date))] = float(fval)
        dept_share: dict[tuple[int, int], float] = {}
        train_store_means: dict[int, float] = {}
        for (store, dept), dept_mean in self._fallback._store_dept_means.items():
            if store not in train_store_means:
                store_depts = [
                    v for (s, _d), v in self._fallback._store_dept_means.items()
                    if s == store
                ]
                train_store_means[store] = (
                    float(np.sum(store_depts)) if store_depts else 1.0
                )
            store_total = train_store_means[store]
            dept_share[(store, dept)] = (
                dept_mean / store_total if store_total > 0 else 1.0
            )

        preds: list[float] = []
        for _, row in test_df.iterrows():
            store = int(row["Store"])
            dept = int(row["Dept"])
            date = pd.Timestamp(row["Date"])
            store_forecast = store_date_forecasts.get(
                (store, date), self._fallback._fallback_mean
            )
            share = dept_share.get((store, dept), 1.0 / max(1, len(
                [d for (s, d) in dept_share if s == store]
            )))
            preds.append(store_forecast * share)

        return np.array(preds, dtype=float)


# ---------------------------------------------------------------------------
# 4. Prophet Forecaster
# ---------------------------------------------------------------------------

class ProphetForecaster(BaseForecaster):

    def __init__(
        self,
        top_n_stores: int = 10,
        yearly_seasonality: bool = True,
    ) -> None:
        self.top_n_stores = top_n_stores
        self.yearly_seasonality = yearly_seasonality

        self._store_models: dict[int, Any] = {}
        self._fallback: MovingAverageForecaster = MovingAverageForecaster(window=4)
        self._fitted_stores: set[int] = set()
        self._dept_share: dict[tuple[int, int], float] = {}
        self._store_train_max: dict[int, float] = {}

    def get_name(self) -> str:
        return f"Prophet(top{self.top_n_stores})"

    def fit(self, train_df: pd.DataFrame) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "prophet is required for ProphetForecaster. "
                "Install it with: pip install prophet"
            ) from exc

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("pystan").setLevel(logging.ERROR)

        self._fallback.fit(train_df)

        self._dept_share = {}
        train_store_means: dict[int, float] = {}
        for (store, dept), dept_mean in self._fallback._store_dept_means.items():
            if store not in train_store_means:
                store_depts = [
                    v for (s, _d), v in self._fallback._store_dept_means.items()
                    if s == store
                ]
                train_store_means[store] = (
                    float(np.sum(store_depts)) if store_depts else 1.0
                )
            store_total = train_store_means[store]
            self._dept_share[(store, dept)] = (
                dept_mean / store_total if store_total > 0 else 1.0
            )

        store_agg = _aggregate_store_weekly(train_df)

        # Select top-N stores by total training sales
        store_totals = (
            store_agg.groupby("Store")["Weekly_Sales"]
            .sum()
            .sort_values(ascending=False)
        )
        selected_stores = store_totals.head(self.top_n_stores).index.tolist()

        logger.info(
            "%s: fitting Prophet on %d store(s): %s",
            self.get_name(), len(selected_stores), selected_stores,
        )

        self._store_models = {}
        self._fitted_stores = set()
        self._store_train_max = {}

        for store_id in selected_stores:
            store_df = (
                store_agg[store_agg["Store"] == store_id]
                .sort_values("Date")
                .reset_index(drop=True)
            )

            if len(store_df) < 52:
                logger.warning(
                    "Store %d: only %d weeks of data (need >= 52 for reliable "
                    "yearly seasonality) -- skipping Prophet, will use fallback MA.",
                    store_id, len(store_df),
                )
                continue

            prophet_train = store_df[["Date", "Weekly_Sales", "IsHoliday"]].rename(
                columns={"Date": "ds", "Weekly_Sales": "y"}
            )

            try:
                # Disable yearly seasonality if we have < 2 full years of data:
                # with 52–103 weeks the Fourier fit is noisy; additive+no-yearly
                # is more stable, and the holiday regressor captures spike structure.
                use_yearly = self.yearly_seasonality and len(store_df) >= 104
                model = Prophet(
                    yearly_seasonality=use_yearly,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative" if use_yearly else "additive",
                )
                model.add_regressor("IsHoliday")

                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    model.fit(prophet_train)

                self._store_models[store_id] = model
                self._fitted_stores.add(store_id)
                self._store_train_max[store_id] = float(store_df["Weekly_Sales"].max())
                logger.debug(
                    "Store %d: Prophet fitted (%d rows).", store_id, len(store_df)
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Store %d: Prophet fitting failed (%s) -- using fallback MA.",
                    store_id, exc,
                )

        logger.info(
            "%s fit complete -- %d/%d stores fitted with Prophet, "
            "%d falling back to MA.",
            self.get_name(),
            len(self._fitted_stores),
            len(selected_stores),
            len(selected_stores) - len(self._fitted_stores),
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_fallback") or not self._fallback._store_dept_means:
            raise RuntimeError(
                f"{self.get_name()} has not been fitted. Call fit() first."
            )

        store_date_forecasts: dict[tuple[int, pd.Timestamp], float] = {}
        test_agg = _aggregate_store_weekly(test_df)

        for store_id in test_agg["Store"].unique():
            store_test = (
                test_agg[test_agg["Store"] == store_id]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            n_periods = len(store_test)

            if store_id in self._fitted_stores:
                try:
                    future = store_test[["Date", "IsHoliday"]].rename(
                        columns={"Date": "ds"}
                    )
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        forecast = self._store_models[store_id].predict(future)
                    raw_vals = forecast["yhat"].values
                    # Clip to [0, 3× training max] to guard against wild extrapolation
                    train_max = self._store_train_max.get(store_id, float("inf"))
                    forecast_vals = np.clip(raw_vals, 0.0, 3.0 * train_max)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Store %d: Prophet predict failed (%s) -- using MA fallback.",
                        store_id, exc,
                    )
                    forecast_vals = np.full(n_periods, self._fallback._fallback_mean)
            else:
                dept_means = [
                    v for (s, _d), v in self._fallback._store_dept_means.items()
                    if s == store_id
                ]
                # Sum dept means to get store total, not average — the store
                # dept_share splits store forecast, so mean gives ~1/n_depts× too small.
                fallback_val = (
                    float(np.sum(dept_means)) if dept_means
                    else self._fallback._fallback_mean
                )
                forecast_vals = np.full(n_periods, fallback_val)

            for date, fval in zip(store_test["Date"].values, forecast_vals):
                store_date_forecasts[(int(store_id), pd.Timestamp(date))] = float(fval)

        # Redistribute store-level forecast to department rows
        n_depts_by_store: dict[int, int] = {}
        for (s, _d) in self._dept_share:
            n_depts_by_store[s] = n_depts_by_store.get(s, 0) + 1

        preds: list[float] = []
        for _, row in test_df.iterrows():
            store = int(row["Store"])
            dept = int(row["Dept"])
            date = pd.Timestamp(row["Date"])
            store_forecast = store_date_forecasts.get(
                (store, date), self._fallback._fallback_mean
            )
            share = self._dept_share.get(
                (store, dept),
                1.0 / max(1, n_depts_by_store.get(store, 1)),
            )
            preds.append(store_forecast * share)

        return np.array(preds, dtype=float)


# ---------------------------------------------------------------------------
# 5. XGBoost Forecaster
# ---------------------------------------------------------------------------

def _build_test_features(
    train_tail: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    fallback_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.feature_engineering import run_feature_engineering

    train_part = train_tail.copy()
    train_part["__split__"] = 0
    test_part = test_df.copy()
    test_part["__split__"] = 1

    combined = pd.concat([train_part, test_part], ignore_index=True)

    with contextlib.redirect_stdout(io.StringIO()):
        combined_feat = run_feature_engineering(combined)

    test_feat = combined_feat[combined_feat["__split__"] == 1].copy()
    # reindex ensures all expected feature columns exist (fills missing with 0)
    X_test = test_feat.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)

    # Build a lookup so predictions can be merged back to original test_df order
    test_feat = test_feat.reset_index(drop=True)
    return X_test, test_feat


class XGBoostForecaster(BaseForecaster):

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self._model: Any = None
        self._feature_cols: list[str] = []
        self._train_tail: pd.DataFrame | None = None
        self._fallback_mean: float = 0.0

    def get_name(self) -> str:
        return f"XGBoost(n={self.n_estimators},d={self.max_depth})"

    def fit(self, train_df: pd.DataFrame) -> "XGBoostForecaster":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostForecaster. "
                "Install it with: pip install xgboost"
            ) from exc

        from src.feature_engineering import run_feature_engineering, get_feature_columns

        # Keep last 60 weeks per (Store, Dept) for lag computation at predict time
        self._train_tail = (
            train_df.sort_values(["Store", "Dept", "Date"])
            .groupby(["Store", "Dept"], sort=False)
            .tail(60)
            .reset_index(drop=True)
        )
        self._fallback_mean = float(train_df["Weekly_Sales"].mean())

        with contextlib.redirect_stdout(io.StringIO()):
            train_feat = run_feature_engineering(train_df)

        all_feature_cols = get_feature_columns()
        self._feature_cols = [c for c in all_feature_cols if c in train_feat.columns]

        X = train_feat[self._feature_cols].fillna(0.0)
        y = train_feat["Weekly_Sales"].values.astype(float)

        if len(X) < 2:
            raise ValueError(
                f"{self.get_name()}: feature engineering produced {len(X)} training "
                f"row(s) — cannot train. This should not happen after NaN imputation; "
                f"check feature_engineering.py."
            )

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )
        self._model.fit(X, y)
        logger.info(
            "%s fitted -- %d rows, %d features",
            self.get_name(), len(X), len(self._feature_cols),
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            if self._train_tail is not None:
                # fit() ran but FE produced too few rows to train — use fallback
                return np.full(len(test_df), self._fallback_mean)
            raise RuntimeError(
                f"{self.get_name()} has not been fitted. Call fit() first."
            )

        X_test, test_feat = _build_test_features(
            self._train_tail, test_df, self._feature_cols, self._fallback_mean
        )
        raw_preds = np.maximum(0.0, self._model.predict(X_test))

        # Align predictions back to the original test_df row order via merge
        pred_df = test_feat[["Store", "Dept", "Date"]].copy()
        pred_df["_pred"] = raw_preds

        result = test_df[["Store", "Dept", "Date"]].merge(
            pred_df, on=["Store", "Dept", "Date"], how="left"
        )
        return result["_pred"].fillna(self._fallback_mean).values.astype(float)


# ---------------------------------------------------------------------------
# 6. LightGBM Forecaster
# ---------------------------------------------------------------------------

class LightGBMForecaster(BaseForecaster):

    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self._model: Any = None
        self._feature_cols: list[str] = []
        self._train_tail: pd.DataFrame | None = None
        self._fallback_mean: float = 0.0

    def get_name(self) -> str:
        return f"LightGBM(n={self.n_estimators},leaves={self.num_leaves})"

    def fit(self, train_df: pd.DataFrame) -> "LightGBMForecaster":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMForecaster. "
                "Install it with: pip install lightgbm"
            ) from exc

        from src.feature_engineering import run_feature_engineering, get_feature_columns

        self._train_tail = (
            train_df.sort_values(["Store", "Dept", "Date"])
            .groupby(["Store", "Dept"], sort=False)
            .tail(60)
            .reset_index(drop=True)
        )
        self._fallback_mean = float(train_df["Weekly_Sales"].mean())

        with contextlib.redirect_stdout(io.StringIO()):
            train_feat = run_feature_engineering(train_df)

        all_feature_cols = get_feature_columns()
        self._feature_cols = [c for c in all_feature_cols if c in train_feat.columns]

        X = train_feat[self._feature_cols].fillna(0.0)
        y = train_feat["Weekly_Sales"].values.astype(float)

        if len(X) < 2:
            raise ValueError(
                f"{self.get_name()}: feature engineering produced {len(X)} training "
                f"row(s) — cannot train. This should not happen after NaN imputation; "
                f"check feature_engineering.py."
            )

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(X, y)
        logger.info(
            "%s fitted -- %d rows, %d features",
            self.get_name(), len(X), len(self._feature_cols),
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            if self._train_tail is not None:
                # fit() ran but FE produced too few rows to train — use fallback
                return np.full(len(test_df), self._fallback_mean)
            raise RuntimeError(
                f"{self.get_name()} has not been fitted. Call fit() first."
            )

        X_test, test_feat = _build_test_features(
            self._train_tail, test_df, self._feature_cols, self._fallback_mean
        )
        raw_preds = np.maximum(0.0, self._model.predict(X_test))

        pred_df = test_feat[["Store", "Dept", "Date"]].copy()
        pred_df["_pred"] = raw_preds

        result = test_df[["Store", "Dept", "Date"]].merge(
            pred_df, on=["Store", "Dept", "Date"], how="left"
        )
        return result["_pred"].fillna(self._fallback_mean).values.astype(float)


# ---------------------------------------------------------------------------
# 7. EnsembleForecaster
# ---------------------------------------------------------------------------

class EnsembleForecaster(BaseForecaster):

    def __init__(
        self,
        weights_path: str | Path = "models/results_ensemble_weights.json",
    ) -> None:
        self.weights_path = Path(weights_path)
        self._models_dir: Path = self.weights_path.parent

        self._component_models: list[BaseForecaster] = []
        self._weights: list[float] = []
        self._model_names: list[str] = []

        self._load_components_and_compute_weights()

    def get_name(self) -> str:
        return "Ensemble"

    def _load_components_and_compute_weights(self) -> None:
        cv_path = self._models_dir / "results_cv_folds.csv"
        if not cv_path.exists():
            raise FileNotFoundError(
                f"results_cv_folds.csv not found at '{cv_path}'. "
                "Run python -m src.run_evaluation first to generate it."
            )
        cv_df = pd.read_csv(cv_path)

        avg_mape: dict[str, float] = cv_df.groupby("model")["MAPE"].mean().to_dict()
        logger.info(
            "EnsembleForecaster: CV MAPE loaded for %d model(s) from '%s'",
            len(avg_mape), cv_path,
        )

        matched: list[tuple[str, BaseForecaster, float]] = []
        for pkl_path in sorted(self._models_dir.glob("*.pkl")):
            try:
                obj = joblib.load(pkl_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EnsembleForecaster: could not load '%s' (%s) — skipped.",
                    pkl_path.name, exc,
                )
                continue
            if not isinstance(obj, BaseForecaster):
                continue
            name = obj.get_name()
            if name not in avg_mape:
                logger.warning(
                    "EnsembleForecaster: '%s' (from %s) "
                    "not in cv_results.csv — skipped.",
                    name, pkl_path.name,
                )
                continue
            matched.append((name, obj, avg_mape[name]))
            logger.debug(
                "EnsembleForecaster: loaded '%s' avg CV MAPE=%.4f",
                name, avg_mape[name]
            )

        if not matched:
            raise RuntimeError(
                "EnsembleForecaster: no component models could be matched to "
                "cv_results.csv entries.  Ensure pkl files and cv_results.csv "
                "use identical model names (from get_name())."
            )

        raw_weights = [1.0 / mape for _, _, mape in matched]
        total = sum(raw_weights)
        norm_weights = [w / total for w in raw_weights]

        self._model_names = [name for name, _, _ in matched]
        self._component_models = [m for _, m, _ in matched]
        self._weights = norm_weights

        logger.info("  %-45s %12s  %8s", "Model", "avg CV MAPE", "weight")
        logger.info("  %s", "-" * 68)
        for (name, _, mape), w in zip(matched, norm_weights):
            logger.info("  %-45s %12.4f  %8.4f", name, mape, w)

        weights_out = {
            name: {"avg_cv_mape": float(mape), "weight": float(w)}
            for (name, _, mape), w in zip(matched, norm_weights)
        }
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.weights_path, "w", encoding="utf-8") as fh:
            json.dump(weights_out, fh, indent=2)
        logger.info(
            "EnsembleForecaster: %d component(s) loaded; weights saved -> '%s'",
            len(matched), self.weights_path,
        )

    def fit(self, train_df: pd.DataFrame) -> "EnsembleForecaster":
        logger.info(
            "EnsembleForecaster.fit() called — components are pre-fitted; no-op."
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not self._component_models:
            raise RuntimeError(
                "EnsembleForecaster has no component models. "
                "Re-initialise with a valid weights_path."
            )

        n = len(test_df)
        preds_matrix = np.zeros((n, len(self._component_models)), dtype=float)
        for i, (model, name) in enumerate(
            zip(self._component_models, self._model_names)
        ):
            try:
                col = np.asarray(model.predict(test_df), dtype=float)
                preds_matrix[:, i] = col
                logger.debug("EnsembleForecaster: collected predictions from %s.", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EnsembleForecaster: %s.predict() failed (%s) "
                    "— zeros used for this component.",
                    name, exc,
                )

        combined = preds_matrix @ np.array(self._weights, dtype=float)
        logger.info(
            "EnsembleForecaster: combined %d component(s) into final forecast.",
            len(self._component_models),
        )
        return combined


# ---------------------------------------------------------------------------
# 8. Training + evaluation harness
# ---------------------------------------------------------------------------

def train_and_evaluate_model(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str | Path = "models",
) -> dict[str, Any]:
    model_name = model.get_name()
    logger.info("--- Training %s ---", model_name)

    model.fit(train_df)

    logger.info("Predicting with %s ...", model_name)
    predictions = model.predict(test_df)

    y_true = test_df["Weekly_Sales"].to_numpy(dtype=float)
    metrics = compute_all_metrics(y_true, predictions)

    logger.info("  %-40s  %s", model_name, format_metrics_for_display(metrics))
    logger.info("%s metrics: %s", model_name, metrics)

    # Persist the fitted model
    safe_name = _PKL_SAVE_NAMES.get(
        model_name,
        model_name.replace("(", "_").replace(")", "")
        .replace("=", "").replace(",", "_"),
    )
    model_path = Path(models_dir) / f"{safe_name}.pkl"
    model.save(model_path)

    return {
        "model_name": model_name,
        **metrics,
        "predictions": predictions,
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# 8. Baseline + full model comparison runner
# ---------------------------------------------------------------------------

def run_baseline_comparison(
    data_dir: str | Path = "data/processed/",
    models_dir: str | Path = "models",
    arima_top_n: int = 10,
    include_tabular: bool = True,
    xgb_n_estimators: int = 500,
    lgb_n_estimators: int = 500,
    models_to_train: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    from src.utils import load_processed_data

    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data from '%s'", data_dir)
    train_df = load_processed_data(data_dir / "train_processed.parquet")
    test_df = load_processed_data(data_dir / "test_processed.parquet")

    logger.info("Train: %d rows  |  Test: %d rows", len(train_df), len(test_df))
    logger.info("%-42s %14s  %14s  %10s", "Model", "MAE", "RMSE", "MAPE")
    logger.info("%s", "-" * 84)

    # Build the full catalogue; filter by models_to_train when specified
    _catalogue: dict[str, BaseForecaster] = {
        "moving_average_4w":  MovingAverageForecaster(window=4),
        "moving_average_12w": MovingAverageForecaster(window=12),
        "arima": ARIMAForecaster(auto=True, seasonal=False, top_n_stores=arima_top_n),
    }
    if include_tabular:
        _catalogue["xgboost"] = XGBoostForecaster(n_estimators=xgb_n_estimators)
        _catalogue["lightgbm"] = LightGBMForecaster(n_estimators=lgb_n_estimators)

    if models_to_train:
        unknown = set(models_to_train) - set(_catalogue)
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. "
                f"Valid names: {list(_catalogue.keys())}"
            )
        models_to_run = [_catalogue[k] for k in models_to_train]
        logger.info("Selective training: %s", models_to_train)
    else:
        models_to_run = list(_catalogue.values())

    all_results: dict[str, dict[str, float]] = {}

    for model in models_to_run:
        result = train_and_evaluate_model(model, train_df, test_df, models_dir)
        all_results[result["model_name"]] = {
            "MAE":  result["MAE"],
            "RMSE": result["RMSE"],
            "MAPE": result["MAPE"],
        }

    logger.info("%s", "-" * 84)
    logger.info("Comparison table (sorted by MAPE):")
    comparison_df = compare_models(all_results)
    logger.info("%s", comparison_df.to_string(index=False))

    # Persist results — merge with existing file when only a subset was trained
    results_path = models_dir / "results_metrics.json"
    serialisable = {
        name: {k: float(v) for k, v in m.items()}
        for name, m in all_results.items()
    }
    if models_to_train and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(serialisable)
        serialisable = existing
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Results saved -> '%s'", results_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run baseline model comparison.")
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory with train/test parquet files",
    )
    parser.add_argument(
        "--models-dir", default="models/",
        help="Directory to save model artefacts",
    )
    parser.add_argument(
        "--arima-top-n", type=int, default=10,
        help="Number of stores to fit ARIMA on (default 10)",
    )
    parser.add_argument(
        "--no-tabular", action="store_true",
        help="Skip XGBoost and LightGBM (faster, baseline models only)",
    )
    parser.add_argument(
        "--xgb-n-estimators", type=int, default=500,
        help="XGBoost boosting rounds (default 500)",
    )
    parser.add_argument(
        "--lgb-n-estimators", type=int, default=500,
        help="LightGBM boosting rounds (default 500)",
    )
    parser.add_argument(
        "--models-to-train",
        nargs="+",
        default=None,
        choices=[
            "moving_average_4w", "moving_average_12w",
            "arima", "xgboost", "lightgbm",
        ],
        help=(
            "Train only the specified model(s) and skip the rest. "
            "Example: --models-to-train xgboost lightgbm"
        ),
    )
    args = parser.parse_args()

    run_baseline_comparison(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        arima_top_n=args.arima_top_n,
        include_tabular=not args.no_tabular,
        xgb_n_estimators=args.xgb_n_estimators,
        lgb_n_estimators=args.lgb_n_estimators,
        models_to_train=args.models_to_train,
    )

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Makefile shortcuts (preferred)
make install        # pip install -r requirements.txt
make pipeline       # full run: preprocess → features → train → evaluate → shap
make test           # pytest excluding slow ARIMA/Prophet tests
make test-slow      # include integration tests (fits real models — takes minutes)
make lint           # flake8 check
make serve          # streamlit run dashboard/app.py on localhost:8501
make check          # lint + test
make clean          # remove all generated artefacts (parquet, pkl, JSON, logs)

# Run individual steps manually
python -m src.preprocessing --data-dir data/raw/ --output-dir data/processed/
python -m src.models --data-dir data/processed/ --models-dir models/
# Add --no-tabular to skip XGBoost/LightGBM (faster for quick baseline check)
python -m src.run_evaluation --data-dir data/processed/ --models-dir models/
python -m src.save_predictions --data-dir data/processed/ --models-dir models/
python -m src.shap_analysis --data-dir data/processed/ --models-dir models/

# Pytest variants
pytest tests/ -v --tb=short                         # all fast tests
pytest tests/ -v -m slow                            # slow integration tests only
pytest tests/test_preprocessing.py -v               # single file
pytest tests/test_preprocessing.py::TestHandleMissingValues::test_no_nan_in_weekly_sales_after_cleaning -v

# Lint (mirrors CI exactly)
flake8 src/ tests/ dashboard/ --max-line-length=88 --exclude=__pycache__
```

## Architecture

The pipeline runs in strict sequence; each stage consumes the output of the previous one:

```
data/raw/*.csv
  └─ src/preprocessing.py        load_and_merge_data -> handle_missing_values
                                  -> encode_features -> create_train_test_split
                                  -> data/processed/*.parquet
       └─ src/feature_engineering.py    adds lag/rolling/date/holiday columns (35 total)
            └─ src/models.py            trains & persists 6 models to models/*.pkl
                 └─ src/run_evaluation.py   walk-forward CV + ensemble weights
                      └─ src/save_predictions.py   test predictions -> parquet
                           └─ src/shap_analysis.py  SHAP values -> models/*.pkl
                                └─ dashboard/app.py  Streamlit UI (3 tabs)
```

**`src/utils.py`** — shared helpers: `load_raw_data`, `save_processed_data`, `load_processed_data`, `setup_logging`. Note `preprocessing.py` has its own `load_and_merge_data` that does the same CSV loading — `utils.load_raw_data` is a lighter alternative without sort/logging.

CI runs flake8 + fast pytest on push/PR to main (`.github/workflows/ci.yml`). Slow tests are excluded from CI.

## Data

Three Kaggle CSVs live in `data/raw/` (git-ignored):

- `train.csv` — Store, Dept, Date, Weekly_Sales, IsHoliday
- `stores.csv` — Store, Type (A/B/C), Size
- `features.csv` — Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday

After merging, the canonical schema is **17 columns**. `IsHoliday` from `features.csv` is dropped (train.csv is authoritative). Processed splits are saved as parquet: `data/processed/train_processed.parquet` and `test_processed.parquet`.

## Key design decisions

- **Temporal train/test split** at `2012-04-06` — never use sklearn random split on this data; it causes leakage. Train: Feb 2010 – Mar 2012 (112 weeks). Test: Apr–Oct 2012 (26 weeks).
- **MarkDowns**: NaN → 0 (promotions were simply not running), then a derived `is_markdown_active` binary column is added.
- **CPI / Unemployment**: forward-then-backward filled _within each Store group_, not globally.
- **Type encoding**: ordinal `A=1, B=2, C=3` into `Type_encoded`; original `Type` column is kept.
- **Kaggle evaluation metric**: `weighted_mae` where holiday weeks are weighted 5× — implement this in `evaluation.py`, not plain MAE.
- **MAPE exclusion**: `compute_mape` silently drops rows where `|y_true| < 1.0` to avoid division noise from micro-sales; be aware this shrinks the denominator population.
- **Cross-validation**: `walk_forward_cv` uses an expanding training window (not k-fold) to preserve temporal order across folds (`n_splits=3`).
- All `src/` modules use `logging.getLogger(__name__)` — call `src.utils.setup_logging()` once at the entry point to activate console + file output.

## Model implementations

All six forecasters in `src/models.py` share a `BaseForecaster` abstract class with `fit()`, `predict()`, `save()`, `load()`:

- `MovingAverageForecaster` — per (Store, Dept) trailing mean; window=4 or 12; falls back to global mean for unseen groups
- `ARIMAForecaster` — store-level `auto_arima` (pmdarima); `top_n_stores` limit; dept-share redistribution
- `ProphetForecaster` — store-level Facebook Prophet; holiday regressor; dept-share redistribution
- `XGBoostForecaster` — full 35-feature matrix from `src/feature_engineering`; `train_tail` prepend at predict time to satisfy lag features
- `LightGBMForecaster` — same feature matrix as XGBoost but `LGBMRegressor`; `num_leaves=63`
- `EnsembleForecaster` — inverse-MAPE weighted average of all five component models; weights saved to `results_ensemble_weights.json`

**Feature column registry**: `src/feature_engineering.py` maintains module-level lists (`_LAG_COLS`, `_ROLLING_COLS`, `_DATE_COLS`, `_HOLIDAY_COLS`) that are populated as side effects when each `create_*` function runs. `get_feature_columns()` reads these lists. Adding a new feature column requires registering it in the appropriate list, or the tabular models will silently ignore it.

**SHAP analysis** (`src/shap_analysis.py`): uses `TreeExplainer` on the XGBoost model. Global importance is mean(|SHAP|) per feature; store-level importance filters to a single store. Values are cached to `models/` as pkl files. `_FEATURE_EXPLANATIONS` is a dict of plain-English descriptions for all 35 features consumed by dashboard Tab 3.

After `run_evaluation.py` completes, `models/results_metrics.json` is written with all six models' metrics and consumed by dashboard Tab 2. Tab 2 shows placeholder content until this file exists.

## Model artefact filenames

Pkl files in `models/` use human-readable names mapped in `src/models._PKL_SAVE_NAMES`:

| Model | pkl file |
|---|---|
| MovingAverage(w=4) | `model_moving_average_4w.pkl` |
| MovingAverage(w=12) | `model_moving_average_12w.pkl` |
| ARIMA(auto,nonseasonal,top10) | `model_arima.pkl` |
| Prophet(top10) | `model_prophet.pkl` |
| XGBoost(n=500,d=6) | `model_xgboost.pkl` |
| LightGBM(n=500,leaves=63) | `model_lightgbm.pkl` |

Result files (all git-ignored, produced by `run_evaluation.py`): `results_metrics.json`, `results_cv_folds.csv`, `results_ensemble_weights.json`, `results_ensemble_predictions.csv`.

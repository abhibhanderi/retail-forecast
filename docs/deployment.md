# Deploying to Streamlit Community Cloud

## Prerequisites

- A GitHub account with this repository pushed to a public (or private) repo
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free tier is sufficient)
- The full pipeline must have been run locally so result files exist in `models/`

## Step 1 — Run the pipeline locally and commit result files

Streamlit Community Cloud cannot run `make pipeline` during deployment. The result files must be committed to the repository.

```bash
make pipeline        # produces models/results_*.json, models/results_*.csv,
                     # data/processed/predictions_test.parquet,
                     # data/processed/actuals_train.parquet
```

Then commit the result files (they are no longer gitignored):

```bash
git add models/results_metrics.json \
        models/results_cv_folds.csv \
        models/results_ensemble_weights.json \
        models/results_ensemble_predictions.csv \
        data/processed/predictions_test.parquet \
        data/processed/actuals_train.parquet
git commit -m "add pipeline result artefacts for Streamlit Cloud"
git push
```

> **Note:** `models/*.pkl` and `models/shap_values.pkl` remain gitignored — they are too large for GitHub. The dashboard does not load pkl files directly; it only reads the result JSON/CSV/parquet files.

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **New app**.
3. Select your GitHub repository and branch.
4. Set **Main file path** to `dashboard/app.py`.
5. Click **Deploy**.

Streamlit Cloud will:
- Install system packages from `packages.txt` (`libgomp1` — required by LightGBM on Linux)
- Install Python packages from `requirements.txt`
- Launch the app at a `*.streamlit.app` URL

## Step 3 — Secrets (if needed)

If you add secrets to `.streamlit/secrets.toml` locally, replicate them in the Streamlit Cloud dashboard:

1. In your app's settings, click **Secrets**.
2. Paste the contents of your local `secrets.toml` (excluding comments).
3. Save.

The `secrets.toml` file is gitignored and must never be committed.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Red error banner: "Missing file: models/results_metrics.json" | Result files not committed | Run `make pipeline` locally and commit the output files (Step 1) |
| `ModuleNotFoundError: lightgbm` | `packages.txt` missing or `libgomp1` not listed | Verify `packages.txt` contains `libgomp1` and is committed |
| App crashes on import | Dependency version conflict | Check `requirements.txt` pins match the versions used locally |
| Slow first load | Cold start — Streamlit Cloud spins down free-tier apps after inactivity | Normal; subsequent loads are fast |

## Redeploying after pipeline changes

After running `make pipeline` to update model results:

```bash
git add models/results_*.json models/results_*.csv \
        data/processed/predictions_test.parquet \
        data/processed/actuals_train.parquet
git commit -m "update pipeline results"
git push
```

Streamlit Cloud redeploys automatically on every push to the configured branch.

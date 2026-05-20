from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import (
    chart_ensemble_weights,
    chart_fold_mape,
    chart_grouped_bar_comparison,
)
from dashboard.config import (
    BORDER,
    MID_TEXT,
    MUTED,
    PAGE_BG,
    _MODEL_SHORT_NAMES,
)
from dashboard.data_loader import (
    _load_cv_results,
    _load_ensemble_weights,
    _load_results,
)
from dashboard.helpers import _chart_card, _section


def render_model_comparison() -> None:
    results = _load_results()
    cv_df   = _load_cv_results()
    weights = _load_ensemble_weights()

    st.info(
        "This tab compares all 7 forecasting models on the held-out test set (Apr–Oct 2012). "
        "Metrics are computed using actual vs predicted weekly sales across all 45 stores. "
        "**Weighted MAE** (holiday weeks × 5) is the primary Kaggle evaluation metric. "
        "Green cells mark the best score per column."
    )

    if not results:
        _section("Expected Models")
        placeholder_rows = [
            ("Moving Average (w=4)",  "Baseline"),
            ("Moving Average (w=12)", "Baseline"),
            ("ARIMA",                 "Statistical"),
            ("Prophet",               "Statistical"),
            ("XGBoost",               "Tabular ML"),
            ("LightGBM",              "Tabular ML"),
            ("Ensemble",              "Weighted avg"),
        ]
        rows_html = "".join(
            f"""<tr>
                <td>{name}</td>
                <td><span style="color:{MUTED}">{kind}</span></td>
                <td>—</td><td>—</td><td>—</td><td>—</td>
                <td><span class="badge-pending">Pending</span></td>
            </tr>"""
            for name, kind in placeholder_rows
        )
        st.markdown(
            f"""
            <div class="chart-card" style="padding: 12px 16px;">
            <table class="metric-table">
                <thead>
                    <tr>
                        <th>Model</th><th>Type</th>
                        <th>MAE</th><th>RMSE</th><th>WMAPE</th>
                        <th>Weighted MAE</th><th>Status</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "Run the pipeline to populate this tab:\n"
            "```bash\npython -m src.preprocessing --data-dir data/raw/ --output-dir data/processed/\n"
            "python -m src.models --data-dir data/processed/ --models-dir models/\n```"
        )
        return

    # ── Best model summary ────────────────────────────────────────────────────
    def _sort_key_local(k: str) -> float:
        m = results[k]
        return m.get("weighted_MAE", m.get("MAE", float("inf")))

    best_key   = min(results, key=_sort_key_local)
    best_short = _MODEL_SHORT_NAMES.get(best_key, best_key)
    best_wmae  = results[best_key].get("weighted_MAE", results[best_key].get("MAE", 0))
    n_models   = len(results)

    _section("Metrics Table")
    st.caption(
        f"{n_models} models evaluated on the 26-week held-out test set. "
        f"Best model: {best_short} with Weighted MAE = ${best_wmae:,.0f}. "
        "Lower is better for all metrics."
    )

    # Build metrics rows with Type column
    _model_types = {
        "MovingAverage(w=4)":            "Baseline",
        "MovingAverage(w=12)":           "Baseline",
        "ARIMA(auto,nonseasonal,top10)": "Statistical",
        "Prophet(top10)":                "Statistical",
        "XGBoost(n=500,d=6)":            "Tabular ML",
        "LightGBM(n=500,leaves=63)":     "Tabular ML",
        "Ensemble":                      "Weighted Avg",
    }

    rows = []
    for key, m in results.items():
        rows.append({
            "Model":            _MODEL_SHORT_NAMES.get(key, key),
            "Type":             _model_types.get(key, "—"),
            "MAE ($)":          m.get("MAE"),
            "RMSE ($)":         m.get("RMSE"),
            "WMAPE (%)":        m.get("MAPE"),
            "Weighted MAE ($)": m.get("weighted_MAE"),
        })

    table_df = pd.DataFrame(rows)
    sort_col  = "Weighted MAE ($)" if table_df["Weighted MAE ($)"].notna().any() else "MAE ($)"
    table_df  = table_df.sort_values(sort_col).reset_index(drop=True)

    def _highlight_min(s: pd.Series) -> list[str]:
        mn = s.min()
        return [
            "background-color: #CCFBF1; color: #0F766E; font-weight: 600"
            if pd.notna(v) and v == mn else ""
            for v in s
        ]

    def _bold_best_row(df: pd.DataFrame) -> pd.DataFrame:
        _sc = "Weighted MAE ($)" if df["Weighted MAE ($)"].notna().any() else "MAE ($)"
        best_idx = df[_sc].idxmin()
        result = pd.DataFrame("", index=df.index, columns=df.columns)
        result.loc[best_idx] = "font-weight: 700"
        return result

    max_mae  = table_df["MAE ($)"].max()  if table_df["MAE ($)"].notna().any()  else 1
    max_rmse = table_df["RMSE ($)"].max() if table_df["RMSE ($)"].notna().any() else 1

    col_cfg = {
        "Model": st.column_config.TextColumn("Model", width="medium"),
        "Type":  st.column_config.TextColumn("Type", width="small"),
        "MAE ($)": st.column_config.ProgressColumn(
            "MAE ($)", format="$%d", min_value=0, max_value=int(max_mae * 1.1),
            help="Mean Absolute Error — lower is better",
        ),
        "RMSE ($)": st.column_config.ProgressColumn(
            "RMSE ($)", format="$%d", min_value=0, max_value=int(max_rmse * 1.1),
            help="Root Mean Squared Error — penalises large errors more",
        ),
        "WMAPE (%)": st.column_config.NumberColumn(
            "WMAPE (%)", format="%.1f%%",
            help="Weighted MAPE = sum|actual−pred|/sum(actual)",
        ),
        "Weighted MAE ($)": st.column_config.NumberColumn(
            "Weighted MAE ($)", format="$%d",
            help="Holiday weeks weighted 5x — the primary Kaggle metric",
        ),
    }

    st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(
        "WMAPE = sum|actual−pred| / sum(actual). "
        "Weighted MAE applies 5× holiday weight per Kaggle competition rules. "
        "MAPE excludes rows where |actual| < 1 to avoid division noise."
    )

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _section("Error Comparison Chart")
    st.caption(
        "MAE, RMSE, and Weighted MAE shown side by side. "
        "The Ensemble model is highlighted in red. Lower bars = more accurate."
    )
    _chart_card(chart_grouped_bar_comparison(results), "model_grouped_bar")

    # ── CV fold detail ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("Cross-Validation Fold Details", expanded=False):
        if cv_df is not None and not cv_df.empty:
            _section("Per-Fold Metrics")
            st.caption(
                "Walk-forward cross-validation with expanding training windows. "
                "Each fold trains on progressively more data and evaluates on the next period."
            )

            fold_cols = {c: c for c in ["model", "fold", "MAE", "RMSE", "MAPE", "weighted_MAE"]
                         if c in cv_df.columns}
            display_cv = cv_df[list(fold_cols)].copy()
            display_cv["model"] = display_cv["model"].map(
                lambda k: _MODEL_SHORT_NAMES.get(k, k)
            )

            fmt: dict = {}
            if "MAE"          in display_cv.columns: fmt["MAE"]          = lambda v: f"${v:,.0f}"
            if "RMSE"         in display_cv.columns: fmt["RMSE"]         = lambda v: f"${v:,.0f}"
            if "MAPE"         in display_cv.columns: fmt["MAPE"]         = lambda v: f"{v:.1f}%"
            if "weighted_MAE" in display_cv.columns: fmt["weighted_MAE"] = lambda v: f"${v:,.0f}"

            st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
            st.dataframe(
                display_cv.style.format(fmt),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if "MAPE" in cv_df.columns:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                _chart_card(chart_fold_mape(cv_df), "fold_mape")
        else:
            st.info(
                "Fold-level results not found. "
                "Run the pipeline to generate `models/results_cv_folds.csv`."
            )

    # ── Ensemble weights ───────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _section("Ensemble Weights")
    st.caption(
        "Models with lower average cross-validation WMAPE receive higher ensemble weights. "
        "Higher weight = bigger influence on the final blended prediction."
    )
    if weights:
        _chart_card(chart_ensemble_weights(weights), "ensemble_weights")
        st.markdown(
            f"""
            <div style="background: {PAGE_BG}; border: 1px solid {BORDER};
                        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
                        font-size: 0.83rem; color: {MID_TEXT}; line-height: 1.6;">
                <b>How weights are assigned:</b> Each model's weight is based on its
                cross-validation accuracy. Models with lower forecasting error get a higher weight.
                LightGBM and XGBoost get the highest weights (~26% each) because they achieved
                the lowest MAPE (13.5% and 13.7%) during cross-validation. Statistical models
                (ARIMA, Prophet, Moving Average) share the remaining weight equally at ~16% each
                because their MAPE was higher (around 21.9%). The ensemble combines all five
                models using these weights to produce a final prediction that is more accurate
                than any single model alone.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Ensemble weights not found. "
            "Run `python -m src.models --data-dir data/processed/ --models-dir models/` "
            "to generate `models/results_ensemble_weights.json`."
        )

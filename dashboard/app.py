from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.config import MUTED  # noqa: E402
from dashboard.data_loader import _load_ensemble_weights, _load_results, load_actuals, load_predictions  # noqa: E402
from dashboard.helpers import build_sidebar  # noqa: E402
from dashboard.styles import get_css  # noqa: E402
from dashboard.tabs.data_explorer import render_data_explorer  # noqa: E402
from dashboard.tabs.forecast import render_forecast_tab  # noqa: E402
from dashboard.tabs.model_comparison import render_model_comparison  # noqa: E402

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecast",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Retail Sales Forecasting Dashboard — "
            "Built with Streamlit and Plotly."
        )
    },
)

st.markdown(get_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Clear metric caches on every run so JSON changes are picked up immediately
_load_results.clear()
_load_ensemble_weights.clear()

# Load predictions (needed for sidebar store/dept lists)
_preds_df   = load_predictions()
_actuals_df = load_actuals()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

filters = build_sidebar(_preds_df)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

date_range = filters["date_range"]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    _hdr_start = pd.Timestamp(date_range[0]).strftime("%b %d, %Y")
    _hdr_end   = pd.Timestamp(date_range[1]).strftime("%b %d, %Y")
    _hdr_weeks = max(1, (pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])).days // 7 + 1)
else:
    _hdr_start = _hdr_end = pd.Timestamp(date_range).strftime("%b %d, %Y")
    _hdr_weeks = 1

_hdr_store   = filters["store_sel"]
_hdr_model   = filters["primary_model"]
_hdr_n_stores = _preds_df["Store"].nunique() if not _preds_df.empty else 45

st.markdown(
    f"""
    <div class="page-hdr">
        <div>
            <div class="page-hdr-title">Retail Sales Forecasting Dashboard</div>
            <div class="page-hdr-sub">
                Walmart Store Sales &nbsp;&middot;&nbsp;
                {_hdr_store} &nbsp;&middot;&nbsp;
                {_hdr_start} &mdash; {_hdr_end}
                &nbsp;&middot;&nbsp; {_hdr_weeks} weeks
            </div>
        </div>
        <div class="page-hdr-badge">
            <span class="live-badge">
                <span class="live-dot"></span>Forecast Ready
            </span>
            <div style="margin-top:6px;font-size:0.7rem;color:{MUTED};">
                Primary model: <b>{_hdr_model}</b> &nbsp;&middot;&nbsp;
                {_hdr_n_stores} stores
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_forecast, tab_models, tab_explorer = st.tabs([
    "  Forecast  ",
    "  Model Comparison  ",
    "  Data Explorer  ",
])

with tab_forecast:
    if _preds_df.empty:
        st.error(
            "**Prediction data not found.**\n\n"
            "Run the full pipeline to generate predictions:\n"
            "```bash\n"
            "python -m src.preprocessing --data-dir data/raw/ --output-dir data/processed/\n"
            "python -m src.models --data-dir data/processed/ --models-dir models/\n"
            "```"
        )
    else:
        render_forecast_tab(_preds_df, _actuals_df, filters)

with tab_models:
    render_model_comparison()

with tab_explorer:
    render_data_explorer(filters)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="dash-footer">
        <div>Retail Sales Forecasting Dashboard</div>
        <div>Built with Streamlit &nbsp;&middot;&nbsp; Walmart Store Sales Dataset (Kaggle)</div>
        <div>XGBoost &nbsp;&middot;&nbsp; LightGBM &nbsp;&middot;&nbsp; ARIMA &nbsp;&middot;&nbsp; Prophet &nbsp;&middot;&nbsp; Ensemble</div>
    </div>
    """,
    unsafe_allow_html=True,
)

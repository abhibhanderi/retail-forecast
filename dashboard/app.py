from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Colour palette — light theme only
# ---------------------------------------------------------------------------

PRIMARY   = "#2563EB"   # royal blue (charts)
TEAL      = "#0D9488"   # teal-600  (accent)
TEAL_SOFT = "#CCFBF1"   # teal-100
NAVY      = "#1B2A4A"   # dark navy (sidebar)
BLUE_SOFT = "#DBEAFE"   # light blue fill
DARK_TEXT = "#0F172A"   # near-black
MID_TEXT  = "#334155"   # slate-700
MUTED     = "#64748B"   # slate-500
BORDER    = "#E2E8F0"   # slate-200
PAGE_BG   = "#F8FAFC"   # slate-50
CARD_BG   = "#FFFFFF"
SUCCESS   = "#16A34A"
DANGER    = "#DC2626"

STORE_TYPE_COLORS = {
    "A": "#2563EB",
    "B": "#3B82F6",
    "C": "#93C5FD",
}

PLOTLY_TMPL = "plotly_white"

TRAIN_PATH          = Path(__file__).parent.parent / "data" / "processed" / "train_processed.parquet"
TEST_PATH           = Path(__file__).parent.parent / "data" / "processed" / "test_processed.parquet"
RESULTS_PATH        = Path(__file__).parent.parent / "models" / "results_metrics.json"
MODELS_DIR          = Path(__file__).parent.parent / "models"
ENSEMBLE_PREDS_PATH   = MODELS_DIR / "results_ensemble_predictions.csv"
CV_RESULTS_PATH       = MODELS_DIR / "results_cv_folds.csv"
ENSEMBLE_WEIGHTS_PATH = MODELS_DIR / "results_ensemble_weights.json"
PREDICTIONS_TEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "predictions_test.parquet"
ACTUALS_TRAIN_PATH    = Path(__file__).parent.parent / "data" / "processed" / "actuals_train.parquet"

# Test period boundaries
TEST_START = pd.Timestamp("2012-04-06")
TEST_END   = pd.Timestamp("2012-10-26")
_TRAIN_TEST_SPLIT = pd.Timestamp("2012-04-06")

# Model display name → pkl stem (matches _PKL_SAVE_NAMES in src/models.py)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Moving Average (w=4)":  "model_moving_average_4w",
    "Moving Average (w=12)": "model_moving_average_12w",
    "ARIMA":                 "model_arima",
    "Prophet":               "model_prophet",
    "XGBoost":               "model_xgboost",
    "LightGBM":              "model_lightgbm",
}
MODEL_COLORS: dict[str, str] = {
    "Moving Average (w=4)":  "#F59E0B",
    "Moving Average (w=12)": "#D97706",
    "ARIMA":                 "#8B5CF6",
    "Prophet":               "#EC4899",
    "XGBoost":               "#0D9488",
    "LightGBM":              "#0F766E",
    "Ensemble":              "#DC2626",
}
_ALL_MODEL_NAMES = ["Ensemble"] + list(MODEL_DISPLAY_NAMES.keys())

# Model display name → prediction column in predictions_test.parquet
MODEL_PRED_COLS: dict[str, str] = {
    "Moving Average (w=4)":  "pred_moving_average_4w",
    "Moving Average (w=12)": "pred_moving_average_12w",
    "ARIMA":                 "pred_arima",
    "Prophet":               "pred_prophet",
    "XGBoost":               "pred_xgboost",
    "LightGBM":              "pred_lightgbm",
    "Ensemble":              "pred_ensemble",
}

# baseline_results.json model key → short display label
_MODEL_SHORT_NAMES: dict[str, str] = {
    "MovingAverage(w=4)":            "MA (w=4)",
    "MovingAverage(w=12)":           "MA (w=12)",
    "ARIMA(auto,nonseasonal,top10)": "ARIMA",
    "Prophet(top10)":                "Prophet",
    "XGBoost(n=500,d=6)":            "XGBoost",
    "LightGBM(n=500,leaves=63)":     "LightGBM",
    "Ensemble":                      "Ensemble",
}

# ---------------------------------------------------------------------------
# Page config
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

# ---------------------------------------------------------------------------
# Global CSS — light theme
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ── Fonts ── */
        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }}

        /* ── Page background ── */
        .stApp {{
            background-color: {PAGE_BG};
        }}
        .main .block-container {{
            background-color: {PAGE_BG};
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}

        /* ── Sidebar — dark navy ── */
        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
            border-right: none;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stDateInput label {{
            color: rgba(255,255,255,0.55) !important;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}
        section[data-testid="stSidebar"] .stSelectbox > div > div,
        section[data-testid="stSidebar"] .stMultiSelect > div > div {{
            background-color: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.15) !important;
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] .stCaption p {{
            color: rgba(255,255,255,0.6) !important;
        }}
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {{
            color: rgba(255,255,255,0.85) !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.12) !important;
        }}
        section[data-testid="stSidebar"] .stExpander {{
            border-color: rgba(255,255,255,0.15) !important;
            background: rgba(255,255,255,0.05) !important;
        }}
        section[data-testid="stSidebar"] .stExpander summary p {{
            color: rgba(255,255,255,0.75) !important;
        }}

        /* ── Live dot animation ── */
        @keyframes livepulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.25; }}
        }}
        .live-dot {{
            display: inline-block;
            width: 8px; height: 8px;
            background: {SUCCESS};
            border-radius: 50%;
            animation: livepulse 2s ease-in-out infinite;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .live-badge {{
            display: inline-flex;
            align-items: center;
            background: #DCFCE7;
            color: #15803D;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 999px;
            letter-spacing: 0.04em;
        }}

        /* ── KPI card ── */
        .kpi-card {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: 18px 20px 14px 20px;
            border: 1px solid {BORDER};
            border-top: 3px solid {TEAL};
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            margin-bottom: 6px;
        }}
        .kpi-label {{
            font-size: 0.68rem;
            font-weight: 700;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {DARK_TEXT};
            line-height: 1.1;
        }}
        .kpi-delta {{
            font-size: 0.75rem;
            margin-top: 5px;
            color: {MUTED};
        }}
        .delta-up   {{ color: {SUCCESS}; font-weight: 600; }}
        .delta-down {{ color: {DANGER};  font-weight: 600; }}

        /* ── Section header ── */
        .section-hdr {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 20px 0 10px 0;
        }}
        .section-hdr-bar {{
            width: 4px;
            height: 18px;
            background: {TEAL};
            border-radius: 2px;
            flex-shrink: 0;
        }}
        .section-hdr-text {{
            font-size: 0.82rem;
            font-weight: 700;
            color: {DARK_TEXT};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        /* ── Chart card wrapper ── */
        .chart-card {{
            background: {CARD_BG};
            border-radius: 12px;
            border: 1px solid {BORDER};
            padding: 4px 8px 0 8px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            margin-bottom: 4px;
        }}

        /* ── Page header ── */
        .page-hdr {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 18px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .page-hdr-icon {{
            font-size: 2rem;
            line-height: 1;
        }}
        .page-hdr-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {DARK_TEXT};
            line-height: 1.2;
        }}
        .page-hdr-sub {{
            font-size: 0.78rem;
            color: {MUTED};
            margin-top: 2px;
        }}
        .page-hdr-badge {{
            margin-left: auto;
            text-align: right;
            font-size: 0.72rem;
            color: {MUTED};
            line-height: 1.7;
        }}

        /* ── Pill badge ── */
        .pill {{
            display: inline-block;
            background: {TEAL_SOFT};
            color: {TEAL};
            font-size: 0.68rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
            letter-spacing: 0.04em;
        }}

        .main .block-container p,
        .main .block-container span,
        .main .block-container label,
        .main .block-container small,
        .main .block-container div[class*="caption"],
        .main .block-container div[class*="Caption"] {{
            color: {DARK_TEXT} !important;
        }}

        /* ── Tabs — pill style ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {BORDER};
            border-radius: 10px;
            padding: 3px;
            border-bottom: none;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.85rem;
            font-weight: 600;
            color: {MUTED};
            padding: 7px 20px;
            border-radius: 8px;
            background: transparent;
        }}
        .stTabs [aria-selected="true"] {{
            color: {DARK_TEXT} !important;
            background: {CARD_BG} !important;
            box-shadow: 0 1px 3px rgba(15,23,42,0.12);
        }}

        /* ── Metrics comparison table ── */
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        .metric-table th {{
            background: {PAGE_BG};
            color: {MUTED};
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 10px 14px;
            border-bottom: 2px solid {BORDER};
            text-align: left;
        }}
        .metric-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid {BORDER};
            color: {MID_TEXT};
        }}
        .metric-table tr:last-child td {{ border-bottom: none; }}
        .metric-table tr:hover td {{ background: {PAGE_BG}; }}
        .badge-ready {{
            background: #DCFCE7; color: #15803D;
            padding: 2px 8px; border-radius: 999px;
            font-size: 0.68rem; font-weight: 700;
        }}
        .badge-pending {{
            background: #FEF9C3; color: #854D0E;
            padding: 2px 8px; border-radius: 999px;
            font-size: 0.68rem; font-weight: 700;
        }}

        /* ── Holiday badge ── */
        .holiday-yes {{
            background: #FEF3C7; color: #92400E;
            padding: 2px 8px; border-radius: 999px;
            font-size: 0.68rem; font-weight: 700;
        }}
        .holiday-no {{
            background: {PAGE_BG}; color: {MUTED};
            padding: 2px 7px; border-radius: 999px;
            font-size: 0.68rem;
        }}

        /* ── Dashboard footer ── */
        .dash-footer {{
            background: #F1F5F9;
            border-top: 1px solid {BORDER};
            margin-top: 36px;
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.73rem;
            color: {MUTED};
            border-radius: 0 0 8px 8px;
        }}

        /* ── Sidebar section divider label ── */
        .sb-section-label {{
            font-size: 0.65rem;
            font-weight: 700;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 14px 0 4px 0;
        }}

        /* ── st.metric overrides ── */
        [data-testid="stMetricDelta"] svg {{
            display: none;
        }}
        [data-testid="stMetricValue"] {{
            color: {DARK_TEXT} !important;
        }}
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div {{
            color: {MUTED} !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: {MID_TEXT} !important;
        }}

        /* ── Body text in main area ── */
        .main p, .main li {{
            color: {MID_TEXT} !important;
        }}

        /* ── Caption text — all selector variants Streamlit uses across versions ── */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span,
        [data-testid="stCaptionContainer"] small,
        .main .block-container small,
        .stCaption, .stCaption p, .stCaption span {{
            color: {MID_TEXT} !important;
            font-size: 0.82rem !important;
        }}

        /* ── st.info / st.warning / st.error boxes ── */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] li {{
            color: {DARK_TEXT} !important;
        }}

        /* ── Dataframe / table text ── */
        [data-testid="stDataFrame"] {{
            color: {DARK_TEXT} !important;
        }}

        /* ── Expander text ── */
        .main .stExpander p,
        .main .stExpander span {{
            color: {MID_TEXT} !important;
        }}

        /* ── Plotly axis tick and title text — force navy ── */
        .js-plotly-plot .plotly .xtick text,
        .js-plotly-plot .plotly .ytick text {{
            fill: #1B2A4A !important;
        }}
        .js-plotly-plot .plotly .g-xtitle text,
        .js-plotly-plot .plotly .g-ytitle text {{
            fill: #1B2A4A !important;
        }}

        /* ── Hide Streamlit chrome ── */
        #MainMenu {{ visibility: hidden; }}
        footer    {{ visibility: hidden; }}
        header    {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(label: str) -> None:
    st.markdown(
        f"""
        <div class="section-hdr">
            <div class="section-hdr-bar"></div>
            <div class="section-hdr-text">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_card(fig: go.Figure, key: str) -> None:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.markdown('</div>', unsafe_allow_html=True)


def _fmt_millions(v: float) -> str:
    return f"${v / 1_000_000:.2f}M"


def _fmt_thousands(v: float) -> str:
    return f"${v:,.0f}"


def _fmt_pct(v: float) -> str:
    return f"{'+'if v >= 0 else ''}{v:.1f}%"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading predictions...", ttl=3600)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_TEST_PATH.exists():
        st.error(
            "**Missing file:** `data/processed/predictions_test.parquet`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    df = pd.read_parquet(PREDICTIONS_TEST_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading actuals...", ttl=3600)
def load_actuals() -> pd.DataFrame:
    if not ACTUALS_TRAIN_PATH.exists():
        st.error(
            "**Missing file:** `data/processed/actuals_train.parquet`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    df = pd.read_parquet(ACTUALS_TRAIN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> tuple[pd.DataFrame, bool]:
    """Load the combined train+test parquet for the Data Explorer tab."""
    if not TRAIN_PATH.exists():
        return pd.DataFrame(), False
    train = pd.read_parquet(TRAIN_PATH)
    df = pd.concat(
        [train, pd.read_parquet(TEST_PATH)] if TEST_PATH.exists() else [train],
        ignore_index=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    if "Type" not in df.columns and "Type_encoded" in df.columns:
        df["Type"] = df["Type_encoded"].map({1: "A", 2: "B", 3: "C"})
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True), True


@st.cache_data(show_spinner=False)
def _load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_metrics.json`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cv_results() -> pd.DataFrame | None:
    if not CV_RESULTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_cv_folds.csv`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        return pd.read_csv(CV_RESULTS_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_ensemble_weights() -> dict | None:
    if not ENSEMBLE_WEIGHTS_PATH.exists():
        st.error(
            "**Missing file:** `models/results_ensemble_weights.json`  \n"
            "Run `make pipeline` to generate it, then restart the app."
        )
        st.stop()
    try:
        with open(ENSEMBLE_WEIGHTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared axis / layout constants for Plotly
# ---------------------------------------------------------------------------

_AX = dict(
    tickfont=dict(color=NAVY, size=11, family="Inter, sans-serif"),
    title_font=dict(color=NAVY, size=13, family="Inter, sans-serif"),
    linecolor="#D1D5DB",
    gridcolor="#E5E7EB",
)
_LEG = dict(
    font=dict(color=NAVY, size=11),
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#E0E0E0",
    borderwidth=1,
)
_LAYOUT_BASE = dict(
    template=PLOTLY_TMPL,
    margin=dict(l=12, r=12, t=44, b=12),
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=NAVY),
    title_font=dict(color=NAVY, size=15, family="Inter, sans-serif"),
    hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font_size=12, font_color=DARK_TEXT),
)


# ---------------------------------------------------------------------------
# Weekly forecast table builder
# ---------------------------------------------------------------------------

def build_forecast_table(
    preds: pd.DataFrame,
    actuals_train: pd.DataFrame,
    primary_model: str,
    compare_models: list[str],
    filters: dict,
) -> pd.DataFrame:
    """
    Aggregate predictions and actuals by Date for the selected filters.

    Returns a DataFrame with:
        Date, Actual, <primary_model>, Error_$, Error_%, IsHoliday,
        [optional compare model columns]
    """
    df = preds.copy()

    # Apply store filter
    if filters["store_sel"] != "All Stores":
        store_id = int(filters["store_sel"].split()[-1])
        df = df[df["Store"] == store_id]

    # Apply department filter
    if filters.get("dept_sel"):
        df = df[df["Dept"].isin(filters["dept_sel"])]

    # Apply date filter
    date_range = filters["date_range"]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start = end = pd.Timestamp(date_range)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    if df.empty:
        return pd.DataFrame()

    # Columns to aggregate
    agg_cols: dict[str, str] = {"actual": "sum"}
    primary_col = MODEL_PRED_COLS.get(primary_model, "pred_ensemble")
    if primary_col in df.columns:
        agg_cols[primary_col] = "sum"
    for m in compare_models:
        c = MODEL_PRED_COLS.get(m)
        if c and c in df.columns:
            agg_cols[c] = "sum"

    # Holiday: any department in the week is holiday → week is holiday
    if "IsHoliday" in df.columns:
        agg_cols["IsHoliday"] = "max"

    grouped = df.groupby("Date").agg(agg_cols).reset_index().sort_values("Date")

    # Build output table
    out = pd.DataFrame()
    out["Date"] = grouped["Date"]
    out["Actual Sales"] = grouped["actual"] if "actual" in grouped.columns else np.nan

    primary_pred_vals = grouped[primary_col] if primary_col in grouped.columns else np.nan
    out[primary_model] = primary_pred_vals

    # Error columns (primary model vs actual)
    if "actual" in grouped.columns and primary_col in grouped.columns:
        out["Error ($)"] = grouped[primary_col] - grouped["actual"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["Error (%)"] = np.where(
                np.abs(grouped["actual"]) >= 1.0,
                (grouped[primary_col] - grouped["actual"]) / grouped["actual"] * 100,
                np.nan,
            )
    else:
        out["Error ($)"] = np.nan
        out["Error (%)"] = np.nan

    # IsHoliday
    if "IsHoliday" in grouped.columns:
        out["Holiday"] = grouped["IsHoliday"].map(lambda x: "Yes" if x else "No")
    else:
        out["Holiday"] = "No"

    # Compare model columns
    for m in compare_models:
        c = MODEL_PRED_COLS.get(m)
        if c and c in grouped.columns:
            out[m] = grouped[c]

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart: Forecast with train actuals, test actuals, model predictions
# ---------------------------------------------------------------------------

def chart_forecast_with_models(
    train_weekly: pd.Series,
    test_actual: pd.Series,
    model_series: dict[str, pd.Series],
    title: str,
) -> go.Figure:
    """
    Area chart: train actuals (light fill) + test actuals (solid line) +
    model predictions (dashed). Vertical Forecast Start line at 2012-04-06.
    """
    fig = go.Figure()

    # Train actuals — light blue area
    if train_weekly is not None and not train_weekly.empty:
        fig.add_trace(go.Scatter(
            x=train_weekly.index,
            y=train_weekly.values,
            mode="lines",
            name="Historical Actuals",
            line=dict(color=PRIMARY, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Historical: $%{y:,.0f}<extra></extra>",
        ))

    # Test actuals — solid blue line (on top)
    if test_actual is not None and not test_actual.empty:
        fig.add_trace(go.Scatter(
            x=test_actual.index,
            y=test_actual.values,
            mode="lines",
            name="Test Actuals (held-out)",
            line=dict(color=PRIMARY, width=2.5),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>",
        ))

    # Model predictions — dashed lines
    for model_name, pred_series in model_series.items():
        if pred_series is None or pred_series.empty:
            continue
        color = MODEL_COLORS.get(model_name, MUTED)
        fig.add_trace(go.Scatter(
            x=pred_series.index,
            y=pred_series.values,
            mode="lines",
            name=model_name,
            line=dict(color=color, width=2, dash="dash"),
            hovertemplate=(
                f"<b>%{{x|%b %d, %Y}}</b><br>"
                f"<b>{model_name}</b>: $%{{y:,.0f}}<extra></extra>"
            ),
        ))

    # Vertical Forecast Start line
    fig.add_vline(
        x=_TRAIN_TEST_SPLIT.timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color=TEAL,
        annotation_text="Forecast Start",
        annotation_position="top right",
        annotation_font=dict(size=10, color=TEAL),
    )

    # Shaded forecast region
    all_dates = []
    for s in model_series.values():
        if s is not None and not s.empty:
            all_dates.extend(s.index.tolist())
    if test_actual is not None and not test_actual.empty:
        all_dates.extend(test_actual.index.tolist())
    if all_dates:
        x_max = max(all_dates)
        fig.add_vrect(
            x0=_TRAIN_TEST_SPLIT.timestamp() * 1000,
            x1=pd.Timestamp(x_max).timestamp() * 1000,
            fillcolor="rgba(13,148,136,0.04)",
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=440,
        xaxis=dict(
            **_AX,
            title="Date",
            showgrid=True,
            range=["2010-02-01", "2012-10-31"],
            rangeslider=dict(visible=True, thickness=0.05, range=["2010-02-01", "2012-10-31"]),
        ),
        yaxis=dict(
            **_AX,
            title="Weekly Sales ($)",
            showgrid=True,
            tickformat="$,.0f",
        ),
        legend=dict(**_LEG, orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Grouped bar comparison
# ---------------------------------------------------------------------------

def chart_grouped_bar_comparison(results: dict) -> go.Figure:
    models      = list(results.keys())
    short_names = [_MODEL_SHORT_NAMES.get(m, m) for m in models]
    is_ensemble = [m == "Ensemble" for m in models]

    mae_vals  = [results[m].get("MAE", 0)       for m in models]
    rmse_vals = [results[m].get("RMSE", 0)      for m in models]
    wmae_vals = [results[m].get("weighted_MAE") for m in models]

    def _bar_colors(base_color: str) -> list[str]:
        return [DANGER if e else base_color for e in is_ensemble]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="MAE",
        x=short_names,
        y=mae_vals,
        marker_color=_bar_colors("#3B82F6"),
        marker_line_width=0,
        text=[f"${v:,.0f}" for v in mae_vals],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>MAE: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="RMSE",
        x=short_names,
        y=rmse_vals,
        marker_color=_bar_colors("#8B5CF6"),
        marker_line_width=0,
        text=[f"${v:,.0f}" for v in rmse_vals],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>RMSE: $%{y:,.0f}<extra></extra>",
    ))

    if any(v is not None for v in wmae_vals):
        wmae_display = [v if v is not None else 0 for v in wmae_vals]
        fig.add_trace(go.Bar(
            name="Weighted MAE",
            x=short_names,
            y=wmae_display,
            marker_color=_bar_colors("#10B981"),
            marker_line_width=0,
            text=[f"${v:,.0f}" if v else "" for v in wmae_display],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Weighted MAE: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Model Performance Comparison",
                   font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=420,
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        xaxis=dict(**_AX, title=None),
        yaxis=dict(**_AX, title="Error ($)", showgrid=True, tickformat="$,.0f"),
        legend=dict(**_LEG, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: CV fold MAPE
# ---------------------------------------------------------------------------

def chart_fold_mape(cv_df: pd.DataFrame) -> go.Figure:
    _color_map = {
        "MovingAverage(w=4)":            MODEL_COLORS.get("Moving Average (w=4)", PRIMARY),
        "MovingAverage(w=12)":           MODEL_COLORS.get("Moving Average (w=12)", PRIMARY),
        "ARIMA(auto,nonseasonal,top10)": MODEL_COLORS.get("ARIMA", PRIMARY),
        "Prophet(top10)":                MODEL_COLORS.get("Prophet", PRIMARY),
        "XGBoost(n=500,d=6)":            MODEL_COLORS.get("XGBoost", PRIMARY),
        "LightGBM(n=500,leaves=63)":     MODEL_COLORS.get("LightGBM", PRIMARY),
    }

    fig = go.Figure()
    for model_name, grp in cv_df.groupby("model"):
        grp   = grp.sort_values("fold")
        short = _MODEL_SHORT_NAMES.get(model_name, model_name)
        color = _color_map.get(model_name, MUTED)
        fig.add_trace(go.Scatter(
            x=grp["fold"],
            y=grp["MAPE"],
            mode="lines+markers",
            name=short,
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{short}</b> — Fold %{{x}}<br>WMAPE: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="WMAPE per Cross-Validation Fold",
                   font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=300,
        xaxis=dict(**_AX, title="Fold", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(**_AX, title="WMAPE (%)", showgrid=True),
        legend=dict(**_LEG, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Ensemble weights
# ---------------------------------------------------------------------------

def chart_ensemble_weights(weights: dict) -> go.Figure:
    items = sorted(weights.items(), key=lambda x: x[1]["weight"])
    names = [_MODEL_SHORT_NAMES.get(k, k) for k, _ in items]
    vals  = [v["weight"]            for _, v in items]
    mapes = [v.get("avg_cv_mape", 0) for _, v in items]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, BLUE_SOFT], [1, PRIMARY]],
            showscale=False,
        ),
        marker_line_width=0,
        text=[f"{v:.1%}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=DARK_TEXT),
        customdata=mapes,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Weight: %{x:.1%}<br>"
            "Avg CV WMAPE: %{customdata:.1f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Ensemble Component Weights",
                   font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=max(200, 56 * len(items) + 80),
        xaxis=dict(**_AX, title="Weight", tickformat=".0%", showgrid=True,
                   range=[0, max(vals) * 1.3] if vals else [0, 1]),
        yaxis=dict(**_AX, title=None),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Sales by store type
# ---------------------------------------------------------------------------

def chart_sales_by_type(df: pd.DataFrame) -> go.Figure:
    stats = (
        df.groupby("Type", as_index=False)
        .agg(Total_Sales=("Weekly_Sales", "sum"), Avg_Weekly=("Weekly_Sales", "mean"))
        .sort_values("Type")
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Total Sales",
        x=stats["Type"],
        y=stats["Total_Sales"],
        marker_color=[STORE_TYPE_COLORS.get(t, PRIMARY) for t in stats["Type"]],
        marker_line_width=0,
        text=[f"${v/1e6:.1f}M" for v in stats["Total_Sales"]],
        textposition="outside",
        textfont=dict(size=11, color=DARK_TEXT),
        hovertemplate="Type <b>%{x}</b><br>Total: $%{y:,.0f}<extra></extra>",
        yaxis="y",
    ))

    fig.add_trace(go.Scatter(
        name="Avg Weekly/dept",
        x=stats["Type"],
        y=stats["Avg_Weekly"],
        mode="markers+lines",
        marker=dict(size=9, color=DARK_TEXT, symbol="diamond"),
        line=dict(color=DARK_TEXT, width=1.8, dash="dot"),
        hovertemplate="Type <b>%{x}</b><br>Avg Weekly: $%{y:,.0f}<extra></extra>",
        yaxis="y2",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Sales by Store Type",
                   font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=300,
        xaxis=dict(**_AX, title="Store Type"),
        yaxis=dict(**_AX, title="Total Sales", showgrid=True, tickformat="$,.0f"),
        yaxis2=dict(**_AX, title="Avg Weekly Sales", overlaying="y", side="right",
                    showgrid=False, tickformat="$,.0f"),
        legend=dict(**_LEG, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        bargap=0.4,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Monthly heatmap
# ---------------------------------------------------------------------------

def chart_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Year"]  = tmp["Date"].dt.year
    tmp["Month"] = tmp["Date"].dt.month

    pivot = (
        tmp.groupby(["Year", "Month"])["Weekly_Sales"]
        .sum().unstack("Month").fillna(0)
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = [month_labels[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=cols,
        y=[str(y) for y in pivot.index],
        colorscale=[[0, BLUE_SOFT], [0.5, PRIMARY], [1, "#1E3A6E"]],
        text=[[f"${v/1e6:.1f}M" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y} %{x}</b><br>$%{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickformat="$,.0f", thickness=10, outlinewidth=0),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Sales Heatmap — Month x Year",
                   font=dict(size=14, color=NAVY, weight=700), x=0.01),
        height=200,
        xaxis=dict(**_AX, title=None),
        yaxis=dict(**_AX, title=None, autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar(preds: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 22px 8px 18px 8px;
                        border-bottom: 1px solid rgba(255,255,255,0.1);
                        margin-bottom: 6px; text-align: center;">
                <div style="font-size: 1.15rem; font-weight: 800;
                            color: #FFFFFF; letter-spacing: -0.01em;">
                    Retail Sales Forecast
                </div>
                <div style="font-size: 0.72rem; color: rgba(255,255,255,0.45);
                            margin-top: 3px;">Walmart Store Sales Explorer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Store & Department ──────────────────────────────────────────────
        st.markdown(
            '<div class="sb-section-label">STORE SELECTION</div>',
            unsafe_allow_html=True,
        )
        if not preds.empty:
            all_stores = sorted(preds["Store"].unique().tolist())
            all_depts  = sorted(preds["Dept"].unique().tolist())
        else:
            all_stores = list(range(1, 46))
            all_depts  = []

        store_sel = st.selectbox(
            "Store",
            options=["All Stores"] + [f"Store {s}" for s in all_stores],
            index=0,
            help="Select a single store to see per-department breakdown in the Forecast tab.",
        )
        st.caption(f"{len(all_stores)} stores available")

        dept_sel = st.multiselect(
            "Department",
            options=all_depts,
            default=[],
            placeholder="All departments",
            help="Leave blank to include all departments.",
        )
        st.caption("Leave blank to include all departments")

        # ── Date Range ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="sb-section-label">DATE RANGE</div>',
            unsafe_allow_html=True,
        )
        # Default to full dataset range so the complete chart is visible on load
        date_range = st.date_input(
            "Date Range",
            value=(pd.Timestamp("2010-02-05").date(), TEST_END.date()),
            min_value=pd.Timestamp("2010-02-05").date(),
            max_value=TEST_END.date(),
            help="Full dataset: Feb 2010 – Oct 2012. Narrow the range to zoom the chart.",
        )

        # ── Primary Model ───────────────────────────────────────────────────
        st.markdown(
            '<div class="sb-section-label">FORECAST MODELS</div>',
            unsafe_allow_html=True,
        )
        primary_model = st.selectbox(
            "Primary Model",
            options=_ALL_MODEL_NAMES,
            index=0,   # defaults to "Ensemble"
            help="This model drives the KPI cards and the detailed weekly table.",
        )

        compare_options = [m for m in _ALL_MODEL_NAMES if m != primary_model]
        compare_models: list[str] = st.multiselect(
            "Compare With",
            options=compare_options,
            default=[],
            placeholder="Add models to compare...",
            help="Overlay additional model predictions on the chart and table.",
        )

        # ── Selection summary ─────────────────────────────────────────────────
        st.markdown(
            "<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); "
            "margin: 18px 0 10px 0'>",
            unsafe_allow_html=True,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            _d0_sb = pd.Timestamp(date_range[0])
            _d1_sb = pd.Timestamp(date_range[1])
            _n_weeks_sb = max(1, (_d1_sb - _d0_sb).days // 7 + 1)
            _start_sb = _d0_sb.strftime("%b %d, %Y")
            _end_sb   = _d1_sb.strftime("%b %d, %Y")
        else:
            _d0_sb = pd.Timestamp(date_range) if not isinstance(date_range, tuple) else pd.Timestamp(date_range[0])
            _n_weeks_sb = 1
            _start_sb = _end_sb = _d0_sb.strftime("%b %d, %Y")

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.06);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 8px; padding: 10px 12px;
                        margin: 4px 0 12px 0; line-height: 1.55;">
                <div style="font-size: 0.62rem; font-weight: 700;
                            color: rgba(255,255,255,0.4);
                            text-transform: uppercase; letter-spacing: 0.1em;
                            margin-bottom: 5px;">CURRENT VIEW</div>
                <span style="font-size: 0.78rem; font-weight: 700;
                             color: rgba(255,255,255,0.9);">{store_sel}</span><br>
                <span style="font-size: 0.72rem; color: rgba(255,255,255,0.6);">
                    {_start_sb} &rarr; {_end_sb}</span><br>
                <span style="font-size: 0.7rem; color: rgba(255,255,255,0.45);">
                    {_n_weeks_sb} forecast weeks &nbsp;&middot;&nbsp; {primary_model}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("About this app"):
            st.markdown(
                "**Retail Sales Forecasting Dashboard**\n\n"
                "Built on the Walmart Store Sales dataset from Kaggle. "
                "Models: Moving Average, ARIMA, Prophet, XGBoost, LightGBM, and Ensemble.\n\n"
                "Train/test split: **2012-04-06**\n\n"
                "Test period: **Apr 06 – Oct 26, 2012** (26 weeks)\n\n"
                "Primary metric: **Weighted MAE** (holiday weeks × 5)"
            )

        st.markdown(
            "<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); "
            "margin: 14px 0 8px 0'>",
            unsafe_allow_html=True,
        )
        if st.button("Refresh Data", use_container_width=True, help="Clear all caches and reload data from disk"):
            st.cache_data.clear()
            st.rerun()

    return {
        "store_sel":      store_sel,
        "dept_sel":       dept_sel,
        "date_range":     date_range,
        "primary_model":  primary_model,
        "compare_models": compare_models,
    }


# ---------------------------------------------------------------------------
# Helper: apply sidebar filters to a DataFrame
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter a DataFrame (with Store, Dept, Date columns) by sidebar selections."""
    out = df.copy()
    if filters["store_sel"] != "All Stores":
        out = out[out["Store"] == int(filters["store_sel"].split()[-1])]
    if filters.get("dept_sel"):
        out = out[out["Dept"].isin(filters["dept_sel"])]
    date_range = filters["date_range"]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start = end = pd.Timestamp(date_range)
    out = out[(out["Date"] >= start) & (out["Date"] <= end)]
    return out


# ---------------------------------------------------------------------------
# Tab 1: Forecast
# ---------------------------------------------------------------------------

def render_forecast_tab(preds: pd.DataFrame, actuals_train: pd.DataFrame, filters: dict) -> None:
    primary_model  = filters["primary_model"]
    compare_models = filters["compare_models"]
    store_sel      = filters["store_sel"]
    date_range     = filters["date_range"]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_start = pd.Timestamp(date_range[0])
        date_end   = pd.Timestamp(date_range[1])
    else:
        date_start = date_end = pd.Timestamp(date_range)

    in_test_period = (date_end >= TEST_START) and (date_start <= TEST_END + pd.Timedelta(days=7))

    # ── Build the weekly forecast table (used for KPIs too) ──────────────────
    forecast_table = build_forecast_table(
        preds, actuals_train, primary_model, compare_models, filters
    )

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    _section("Forecast Summary")
    st.caption(
        "Key numbers for the selected store, date range, and primary model. "
        "Accuracy metrics apply only when the selected period overlaps the test period (Apr–Oct 2012)."
    )

    primary_col_name = primary_model  # column name in forecast_table

    total_pred   = forecast_table[primary_col_name].sum() if (not forecast_table.empty and primary_col_name in forecast_table.columns) else 0.0
    weekly_avg   = forecast_table[primary_col_name].mean() if (not forecast_table.empty and primary_col_name in forecast_table.columns) else 0.0

    # Best week
    if not forecast_table.empty and primary_col_name in forecast_table.columns:
        best_idx    = forecast_table[primary_col_name].idxmax()
        best_week   = forecast_table.loc[best_idx, "Date"]
        best_val    = forecast_table.loc[best_idx, primary_col_name]
        best_week_str = pd.Timestamp(best_week).strftime("%b %d, %Y")
    else:
        best_week_str = "N/A"
        best_val      = 0.0

    # WMAPE from results_metrics.json
    results = _load_results()
    wmape_str = "N/A"
    if results and in_test_period:
        # Map primary_model display name to the results key
        # Direct key lookup
        _direct_keys = {
            "Ensemble":              "Ensemble",
            "XGBoost":               "XGBoost(n=500,d=6)",
            "LightGBM":              "LightGBM(n=500,leaves=63)",
            "ARIMA":                 "ARIMA(auto,nonseasonal,top10)",
            "Prophet":               "Prophet(top10)",
            "Moving Average (w=4)":  "MovingAverage(w=4)",
            "Moving Average (w=12)": "MovingAverage(w=12)",
        }
        result_key = _direct_keys.get(primary_model)
        if result_key and result_key in results:
            wmape_val = results[result_key].get("MAPE")
            if wmape_val is not None:
                wmape_str = f"{wmape_val:.1f}%"

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Forecasted Sales</div>
            <div class="kpi-value">{_fmt_millions(total_pred)}</div>
            <div class="kpi-delta">{primary_model}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Weekly Average Forecast</div>
            <div class="kpi-value">{_fmt_thousands(weekly_avg)}</div>
            <div class="kpi-delta">per week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Forecast Week</div>
            <div class="kpi-value">{_fmt_thousands(best_val)}</div>
            <div class="kpi-delta">{best_week_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Forecast Accuracy (WMAPE)</div>
            <div class="kpi-value">{wmape_str}</div>
            <div class="kpi-delta">{'test period' if in_test_period else 'outside test period'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Sales Forecast Chart ──────────────────────────────────────────────────
    _section("Sales Forecast Chart")

    # Build store/dept label
    dept_label = ""
    if filters.get("dept_sel"):
        d_list = filters["dept_sel"]
        dept_label = f" · Dept {', '.join(str(d) for d in d_list[:3])}"
        if len(d_list) > 3:
            dept_label += f" +{len(d_list)-3}"

    in_test_str = " · Forecast Period: Apr–Oct 2012" if in_test_period else ""
    chart_title = (
        f"Weekly Sales — All Stores{dept_label}{in_test_str}"
        if store_sel == "All Stores"
        else f"Weekly Sales — {store_sel}{dept_label}{in_test_str}"
    )

    # Aggregate train actuals
    train_weekly: pd.Series | None = None
    if not actuals_train.empty:
        t = _apply_filters(actuals_train.rename(columns={"actual": "Weekly_Sales"}), {
            **filters,
            "date_range": (pd.Timestamp("2010-02-05").date(), _TRAIN_TEST_SPLIT.date()),
        })
        if not t.empty:
            train_weekly = t.groupby("Date")["Weekly_Sales"].sum().sort_index()

    # Chart data filters: store/dept from sidebar, but date range is always the full
    # test period so the chart traces are never truncated. The sidebar date range is
    # applied later as an x-axis zoom, which prevents any visual gap when the sidebar
    # start date falls after the first available prediction week.
    _chart_filters = {
        **filters,
        "date_range": (TEST_START.date(), TEST_END.date()),
    }

    # Aggregate test actuals — always load full test period
    test_actual_series: pd.Series | None = None
    if not preds.empty and "actual" in preds.columns:
        t2 = _apply_filters(preds.rename(columns={"actual": "Weekly_Sales"}), _chart_filters)
        if not t2.empty:
            test_actual_series = t2.groupby("Date")["Weekly_Sales"].sum().sort_index()

    # Build model series dict — always load full test period
    model_series: dict[str, pd.Series] = {}
    all_models_to_show = [primary_model] + compare_models
    for m in all_models_to_show:
        pred_col = MODEL_PRED_COLS.get(m)
        if pred_col and not preds.empty and pred_col in preds.columns:
            tmp = _apply_filters(preds[["Store", "Dept", "Date", pred_col]], _chart_filters)
            if not tmp.empty:
                model_series[m] = tmp.groupby("Date")[pred_col].sum().sort_index()

    if not model_series and test_actual_series is None and train_weekly is None:
        st.warning("No data matches the current filters. Try widening the date range or selecting a different store.")
    else:
        _forecast_fig = chart_forecast_with_models(
            train_weekly, test_actual_series, model_series, chart_title
        )
        # Zoom the visible window to the sidebar date range without removing any
        # trace data — this keeps all actuals and forecasts connected with no gap
        _forecast_fig.update_xaxes(range=[
            date_start.strftime("%Y-%m-%d"),
            date_end.strftime("%Y-%m-%d"),
        ])
        _chart_card(_forecast_fig, "forecast_chart")
        st.caption(
            "Light blue area = historical training data (Feb 2010 – Dec 2011). "
            "Solid blue line = actual test sales. "
            "Dashed lines = model forecasts. "
            "The dotted vertical line marks the train/test boundary (Apr 6, 2012)."
        )

    # ── Weekly Forecast Table ─────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _section("Weekly Forecast Table")
    st.caption(
        "Detailed week-by-week predictions. "
        "Error columns compare the primary model against actual sales. "
        "Color coding: green < 15% error, yellow 15–30%, red > 30%."
    )

    if forecast_table.empty:
        st.info("No prediction data available for the selected filters and date range.")
    else:
        display_table = forecast_table.copy()

        # Format Date column for display (keep as string for nice rendering)
        display_table["Week"] = pd.to_datetime(display_table["Date"]).dt.strftime("%b %d, %Y")

        # Select and order display columns
        display_cols = ["Week", "Actual Sales", primary_col_name, "Error ($)", "Error (%)", "Holiday"]
        # Add compare model columns
        for m in compare_models:
            if m in display_table.columns:
                display_cols.append(m)

        disp = display_table[display_cols].copy()

        # Round numeric columns
        numeric_disp_cols = ["Actual Sales", primary_col_name, "Error ($)"] + [m for m in compare_models if m in disp.columns]
        for c in numeric_disp_cols:
            if c in disp.columns:
                disp[c] = disp[c].round(2)
        if "Error (%)" in disp.columns:
            disp["Error (%)"] = disp["Error (%)"].round(2)

        # Build column configs
        col_cfg: dict = {
            "Week": st.column_config.TextColumn("Week", width="medium"),
            "Actual Sales": st.column_config.NumberColumn(
                "Actual Sales",
                format="$%,.0f",
                help="Actual recorded sales for this week",
            ),
            primary_col_name: st.column_config.NumberColumn(
                f"Predicted ({primary_model})",
                format="$%,.0f",
                help=f"Forecast from {primary_model}",
            ),
            "Error ($)": st.column_config.NumberColumn(
                "Error ($)",
                format="$%+,.0f",
                help="Predicted minus Actual. Positive = over-forecast, negative = under-forecast.",
            ),
            "Error (%)": st.column_config.NumberColumn(
                "Error (%)",
                format="%+.1f%%",
                help="Percentage error. Green < 15%, yellow 15–30%, red > 30%.",
            ),
            "Holiday": st.column_config.TextColumn(
                "Holiday?",
                help="Whether this is a holiday week (weighted 5x in model evaluation)",
                width="small",
            ),
        }
        for m in compare_models:
            if m in disp.columns:
                col_cfg[m] = st.column_config.NumberColumn(
                    f"Predicted ({m})",
                    format="$%,.0f",
                )

        st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
            height=min(600, max(250, len(disp) * 36 + 50)),
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary row
        total_actual = display_table["Actual Sales"].sum() if "Actual Sales" in display_table.columns else 0
        total_pred_val = display_table[primary_col_name].sum() if primary_col_name in display_table.columns else 0
        total_err    = total_pred_val - total_actual
        total_err_pct = (total_err / total_actual * 100) if abs(total_actual) >= 1 else float("nan")
        n_holidays   = (display_table["Holiday"] == "Yes").sum() if "Holiday" in display_table.columns else 0

        err_color = SUCCESS if abs(total_err_pct) < 15 else (DANGER if abs(total_err_pct) > 30 else "#D97706")

        st.markdown(
            f"""
            <div style="background:{PAGE_BG}; border:1px solid {BORDER}; border-radius:8px;
                        padding:10px 16px; margin-top:8px; font-size:0.83rem; color:{MID_TEXT};
                        display:flex; gap:28px; flex-wrap:wrap;">
                <div><b>Total Actual:</b> <span style="color:{DARK_TEXT};font-weight:700;">{_fmt_thousands(total_actual)}</span></div>
                <div><b>Total Predicted:</b> <span style="color:{DARK_TEXT};font-weight:700;">{_fmt_thousands(total_pred_val)}</span></div>
                <div><b>Total Error:</b> <span style="color:{err_color};font-weight:700;">${total_err:+,.0f} ({total_err_pct:+.1f}%)</span></div>
                <div><b>Holiday Weeks:</b> <span style="color:{DARK_TEXT};font-weight:700;">{n_holidays}</span></div>
                <div><b>Weeks Shown:</b> <span style="color:{DARK_TEXT};font-weight:700;">{len(display_table)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Department Breakdown ──────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if store_sel == "All Stores":
        _section("Store-Level Breakdown")
        st.caption(
            "Showing top stores by predicted sales. "
            "Select a specific store in the sidebar to see department-level detail."
        )

        if not preds.empty:
            tmp_preds = _apply_filters(preds, {**filters, "dept_sel": []})
            prim_col  = MODEL_PRED_COLS.get(primary_model, "pred_ensemble")
            if prim_col in tmp_preds.columns and "actual" in tmp_preds.columns:
                store_summary = (
                    tmp_preds.groupby("Store")
                    .agg(
                        Actual=("actual", "sum"),
                        **{primary_model: (prim_col, "sum")},
                    )
                    .reset_index()
                )
                store_summary["Error ($)"]  = store_summary[primary_model] - store_summary["Actual"]
                store_summary["Error (%)"]  = (
                    store_summary["Error ($)"] / store_summary["Actual"] * 100
                ).round(1)
                store_summary = store_summary.sort_values(primary_model, ascending=False).reset_index(drop=True)
                store_summary["Store"] = "Store " + store_summary["Store"].astype(str)

                st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
                st.dataframe(
                    store_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Store":       st.column_config.TextColumn("Store"),
                        "Actual":      st.column_config.NumberColumn("Actual Sales", format="$%,.0f"),
                        primary_model: st.column_config.NumberColumn("Predicted Sales", format="$%,.0f"),
                        "Error ($)":   st.column_config.NumberColumn("Error ($)", format="$%+,.0f"),
                        "Error (%)":   st.column_config.NumberColumn("Error (%)", format="%+.1f%%"),
                    },
                    height=min(500, len(store_summary) * 36 + 50),
                )
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        _section("Department Breakdown")
        st.caption(
            f"Predicted vs actual sales per department for {store_sel}, "
            "sorted by predicted sales (highest first)."
        )

        store_id = int(store_sel.split()[-1])
        if not preds.empty:
            store_preds = preds[preds["Store"] == store_id].copy()

            date_range = filters["date_range"]
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            else:
                start = end = pd.Timestamp(date_range)
            store_preds = store_preds[(store_preds["Date"] >= start) & (store_preds["Date"] <= end)]

            if filters.get("dept_sel"):
                store_preds = store_preds[store_preds["Dept"].isin(filters["dept_sel"])]

            prim_col = MODEL_PRED_COLS.get(primary_model, "pred_ensemble")
            if prim_col in store_preds.columns and "actual" in store_preds.columns:
                dept_summary = (
                    store_preds.groupby("Dept")
                    .agg(
                        Actual=("actual", "sum"),
                        Predicted=(prim_col, "sum"),
                    )
                    .reset_index()
                )
                dept_summary["Error ($)"] = dept_summary["Predicted"] - dept_summary["Actual"]
                dept_summary["Error (%)"] = (
                    dept_summary["Error ($)"] / dept_summary["Actual"] * 100
                ).round(1)
                dept_summary = dept_summary.sort_values("Predicted", ascending=False).reset_index(drop=True)

                st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
                st.dataframe(
                    dept_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Dept":       st.column_config.NumberColumn("Dept", format="%d"),
                        "Actual":     st.column_config.NumberColumn("Actual Sales", format="$%,.0f"),
                        "Predicted":  st.column_config.NumberColumn(f"Predicted ({primary_model})", format="$%,.0f"),
                        "Error ($)":  st.column_config.NumberColumn("Error ($)", format="$%+,.0f"),
                        "Error (%)":  st.column_config.NumberColumn("Error (%)", format="%+.1f%%"),
                    },
                    height=min(500, len(dept_summary) * 36 + 50),
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Prediction or actual data not available for the selected store and filters.")
        else:
            st.info("Prediction data not loaded.")


# ---------------------------------------------------------------------------
# Tab 2: Model Comparison
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tab 3: Data Explorer
# ---------------------------------------------------------------------------

def render_data_explorer(filters: dict) -> None:
    df, ok = load_data()

    if not ok:
        st.warning(
            "Processed data not found. "
            "Run `python -m src.preprocessing --data-dir data/raw/ --output-dir data/processed/` first."
        )
        return

    st.info(
        "Explore the raw sales data used to train and evaluate the models. "
        "Use the sidebar filters to narrow down by store, department, or date range. "
        "Charts reflect the full filtered dataset."
    )

    # Apply filters
    df_filtered = df.copy()
    if filters["store_sel"] != "All Stores":
        df_filtered = df_filtered[df_filtered["Store"] == int(filters["store_sel"].split()[-1])]
    if filters.get("dept_sel"):
        df_filtered = df_filtered[df_filtered["Dept"].isin(filters["dept_sel"])]
    date_range = filters["date_range"]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start = end = pd.Timestamp(date_range)
    df_filtered = df_filtered[(df_filtered["Date"] >= start) & (df_filtered["Date"] <= end)]

    # Stats banner
    _section("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df_filtered):,}")
    col2.metric("Stores", df_filtered["Store"].nunique() if not df_filtered.empty else 0)
    col3.metric("Departments", df_filtered["Dept"].nunique() if not df_filtered.empty else 0)
    if not df_filtered.empty:
        d_min = df_filtered["Date"].min().strftime("%b %Y")
        d_max = df_filtered["Date"].max().strftime("%b %Y")
        col4.metric("Date Range", f"{d_min}–{d_max}")
    else:
        col4.metric("Date Range", "N/A")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if df_filtered.empty:
        st.warning("No data matches the current filters.")
        return

    # Monthly heatmap
    _section("Monthly Sales Heatmap")
    st.caption("Total weekly sales aggregated by month and year. Darker = higher sales.")
    if "Weekly_Sales" in df_filtered.columns:
        _chart_card(chart_monthly_heatmap(df_filtered), "heatmap_explorer")

    # Sales by store type
    if "Type" in df_filtered.columns and "Weekly_Sales" in df_filtered.columns:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _section("Sales by Store Type")
        st.caption("Total and average weekly sales for Type A (large), B (medium), and C (small) stores.")
        _chart_card(chart_sales_by_type(df_filtered), "by_type_explorer")

    # Raw data table
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _section("Raw Data Table")
    st.caption("Sample of the filtered dataset. Showing first 1,000 rows.")
    show_cols = [c for c in ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday", "Type"]
                 if c in df_filtered.columns]
    sample_df = df_filtered[show_cols].head(1000).copy()
    if "Weekly_Sales" in sample_df.columns:
        sample_df["Weekly_Sales"] = sample_df["Weekly_Sales"].round(2)

    st.markdown('<div class="chart-card" style="padding: 12px 16px;">', unsafe_allow_html=True)
    st.dataframe(
        sample_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Store":        st.column_config.NumberColumn("Store", format="%d"),
            "Dept":         st.column_config.NumberColumn("Dept",  format="%d"),
            "Date":         st.column_config.DateColumn("Date"),
            "Weekly_Sales": st.column_config.NumberColumn("Weekly Sales ($)", format="$%,.0f"),
            "IsHoliday":    st.column_config.CheckboxColumn("Holiday?"),
            "Type":         st.column_config.TextColumn("Store Type"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)
    if len(df_filtered) > 1000:
        st.caption(f"Showing 1,000 of {len(df_filtered):,} rows. Use sidebar filters to narrow down.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Clear metric caches on every run so JSON changes are picked up immediately
_load_results.clear()
_load_ensemble_weights.clear()

# Load predictions (needed for sidebar store/dept lists)
_preds_df    = load_predictions()
_actuals_df  = load_actuals()

# Sidebar
filters = build_sidebar(_preds_df)

# ── Page header ──────────────────────────────────────────────────────────────
date_range = filters["date_range"]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    _hdr_start = pd.Timestamp(date_range[0]).strftime("%b %d, %Y")
    _hdr_end   = pd.Timestamp(date_range[1]).strftime("%b %d, %Y")
    _hdr_weeks = max(1, (pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])).days // 7 + 1)
else:
    _hdr_start = _hdr_end = pd.Timestamp(date_range).strftime("%b %d, %Y")
    _hdr_weeks = 1

_hdr_store = filters["store_sel"]
_hdr_model = filters["primary_model"]
_hdr_n_stores = (
    _preds_df["Store"].nunique()
    if not _preds_df.empty
    else 45
)

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

# ── Tabs ─────────────────────────────────────────────────────────────────────
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

# ── Footer ────────────────────────────────────────────────────────────────────
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

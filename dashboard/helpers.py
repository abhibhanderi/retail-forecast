from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import (
    _ALL_MODEL_NAMES,
    MODEL_PRED_COLS,
    TEST_END,
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


def _fmt_date(dt) -> str:
    return pd.Timestamp(dt).strftime("%b %d, %Y")


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
        _DS_START = pd.Timestamp("2010-02-05").date()
        _DS_END = TEST_END.date()
        _DEFAULT_RANGE = (_DS_START, _DS_END)

        # Apply pending reset BEFORE the widget is instantiated — Streamlit
        # forbids writing a widget's session-state key after creation.
        if st.session_state.get("_reset_date_pending"):
            st.session_state["date_range_input"] = _DEFAULT_RANGE
            st.session_state["_reset_date_pending"] = False

        date_range = st.date_input(
            "Date Range",
            value=_DEFAULT_RANGE,
            min_value=_DS_START,
            max_value=_DS_END,
            help="Full dataset: Feb 2010 – Oct 2012. Narrow the range to zoom the chart.",
            key="date_range_input",
        )

        # Normalize: date_input returns () / (start,) / (start, end) depending
        # on how many dates the user has clicked so far.
        if isinstance(date_range, tuple):
            if len(date_range) == 0:
                date_range = _DEFAULT_RANGE
            elif len(date_range) == 1:
                st.caption(
                    "Select an end date — showing to end of dataset for now."
                )
                date_range = (date_range[0], _DS_END)
            elif date_range[0] > date_range[1]:
                st.warning(
                    "Start date must be before end date. Showing full range."
                )
                date_range = _DEFAULT_RANGE

        if date_range != _DEFAULT_RANGE:
            if st.button(
                "Reset date range",
                key="reset_date_range",
                use_container_width=True,
                type="primary",
                help="Reset to full dataset: Feb 2010 – Oct 2012",
            ):
                st.session_state["_reset_date_pending"] = True
                st.rerun()

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
        if st.button("Refresh Data", use_container_width=True, type="primary", help="Clear all caches and reload data from disk"):
            st.cache_data.clear()
            st.rerun()

    return {
        "store_sel":      store_sel,
        "dept_sel":       dept_sel,
        "date_range":     date_range,
        "primary_model":  primary_model,
        "compare_models": compare_models,
    }

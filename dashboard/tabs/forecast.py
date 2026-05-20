from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import chart_forecast_with_models
from dashboard.config import (
    _MODEL_RESULT_KEYS,
    _TRAIN_TEST_SPLIT,
    BORDER,
    DARK_TEXT,
    DANGER,
    MID_TEXT,
    MODEL_PRED_COLS,
    PAGE_BG,
    SUCCESS,
    TEST_END,
    TEST_START,
)
from dashboard.data_loader import _load_results
from dashboard.helpers import (
    _apply_filters,
    _chart_card,
    _fmt_date,
    _fmt_millions,
    _fmt_thousands,
    _section,
    build_forecast_table,
)


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

    if in_test_period:
        col = primary_col_name
        has_data = not forecast_table.empty and col in forecast_table.columns
        if has_data:
            total_pred = forecast_table[col].sum()
            weekly_avg = forecast_table[col].mean()
            best_idx = forecast_table[col].idxmax()
            best_val = forecast_table.loc[best_idx, col]
            best_week_str = _fmt_date(forecast_table.loc[best_idx, "Date"])
        else:
            total_pred = weekly_avg = best_val = 0.0
            best_week_str = "N/A"
        wmape_str = "N/A"
        results = _load_results()
        if results:
            result_key = _MODEL_RESULT_KEYS.get(primary_model)
            if result_key and result_key in results:
                wmape_val = results[result_key].get("MAPE")
                if wmape_val is not None:
                    wmape_str = f"{wmape_val:.1f}%"
        kpi_total_label = "Total Forecasted Sales"
        kpi_total_delta = primary_model
        kpi_avg_label   = "Weekly Average Forecast"
        kpi_best_label  = "Best Forecast Week"
        kpi_acc_delta   = "test period"
    else:
        _hist = _apply_filters(actuals_train, filters)
        if not _hist.empty and "actual" in _hist.columns:
            _hist_by_date = _hist.groupby("Date")["actual"].sum().sort_index()
            total_pred = float(_hist_by_date.sum())
            weekly_avg = float(_hist_by_date.mean())
            _best_idx = _hist_by_date.idxmax()
            best_val = float(_hist_by_date[_best_idx])
            best_week_str = _fmt_date(_best_idx)
        else:
            total_pred = weekly_avg = best_val = 0.0
            best_week_str = "N/A"
        wmape_str = "N/A"
        kpi_total_label = "Total Actual Sales"
        kpi_total_delta = "historical actuals"
        kpi_avg_label   = "Weekly Average Sales"
        kpi_best_label  = "Best Sales Week"
        kpi_acc_delta   = "outside forecast range"

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{kpi_total_label}</div>
            <div class="kpi-value">{_fmt_millions(total_pred)}</div>
            <div class="kpi-delta">{kpi_total_delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{kpi_avg_label}</div>
            <div class="kpi-value">{_fmt_thousands(weekly_avg)}</div>
            <div class="kpi-delta">per week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{kpi_best_label}</div>
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
            <div class="kpi-delta">{kpi_acc_delta}</div>
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

    # When the selected range has no overlap with the test period, hide forecast
    # lines so only the historical actuals are visible in the zoomed window.
    _chart_models = model_series if in_test_period else {}
    _chart_test   = test_actual_series if in_test_period else None

    if not _chart_models and _chart_test is None and train_weekly is None:
        st.warning(
            "No data matches the current filters. "
            "Try widening the date range or selecting a different store."
        )
    else:
        _forecast_fig = chart_forecast_with_models(
            train_weekly, _chart_test, _chart_models, chart_title
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
        if not in_test_period:
            st.info(
                "The selected date range is in the training period "
                "(before Apr 6, 2012). Forecast model lines are only available "
                "for the test period (Apr–Oct 2012)."
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

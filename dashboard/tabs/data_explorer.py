from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import chart_monthly_heatmap, chart_sales_by_type
from dashboard.data_loader import load_data
from dashboard.helpers import _chart_card, _section


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

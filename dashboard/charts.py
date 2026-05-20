from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.config import (
    MODEL_COLORS,
    PLOTLY_TMPL,
    PRIMARY,
    TEAL,
    NAVY,
    BLUE_SOFT,
    DARK_TEXT,
    MUTED,
    BORDER,
    CARD_BG,
    _TRAIN_TEST_SPLIT,
    _MODEL_SHORT_NAMES,
    STORE_TYPE_COLORS,
    DANGER,
)

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

def chart_grouped_bar_comparison(results: dict) -> go.Figure:  # noqa: E302
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

def chart_fold_mape(cv_df: pd.DataFrame) -> go.Figure:  # noqa: E302
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

def chart_ensemble_weights(weights: dict) -> go.Figure:  # noqa: E302
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

def chart_sales_by_type(df: pd.DataFrame) -> go.Figure:  # noqa: E302
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

def chart_monthly_heatmap(df: pd.DataFrame) -> go.Figure:  # noqa: E302
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

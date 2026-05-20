from __future__ import annotations

from dashboard.config import (
    PAGE_BG, NAVY, TEAL, TEAL_SOFT, DARK_TEXT, MID_TEXT,
    MUTED, BORDER, CARD_BG, SUCCESS, DANGER,
)


def get_css() -> str:
    return f"""
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
    """

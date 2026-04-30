"""Streamlit dashboard for the TSX signal scanner.

Run with:    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

import stock_signals as ss

st.set_page_config(page_title="Stock Signals", layout="wide", page_icon="📈")


# Global CSS — make tabs larger, bolder, with clearer active state
st.markdown(
    """
    <style>
    /* Extra breathing room between major page sections */
    .stApp .main .block-container > div > div > div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 14px;
    }

    /* Edge / Windows 11-style soft warm-grey theme on top of Streamlit base */
    .stApp { background: #202124 !important; }

    /* Bordered containers (popups) styled as soft rounded cards */
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: 12px !important;
        margin-bottom: 12px !important;
        padding: 16px 22px !important;
        background: #2d2e31 !important;
        border: 1px solid #3a3b3e !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    /* Buttons — rounded pills with subtle hover */
    .stApp button[kind="primary"],
    .stApp .stButton > button {
        background: #2d2e31 !important;
        color: #f0f0f0 !important;
        border: 1px solid #3a3b3e !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stApp .stButton > button:hover {
        background: #3a3b3e !important;
        border-color: #4a4b4e !important;
    }
    .stApp button[kind="primary"] {
        background: #60a5fa !important;
        color: #0e1117 !important;
        border-color: #60a5fa !important;
    }
    .stApp button[kind="primary"]:hover {
        background: #93c5fd !important;
    }

    /* Inputs, selectboxes, text areas — match card style */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="textarea"] textarea,
    .stApp .stTextInput > div > div > input,
    .stApp .stTextArea > div > div > textarea {
        background: #2d2e31 !important;
        border: 1px solid #3a3b3e !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }

    /* Sidebar */
    .stApp [data-testid="stSidebar"] {
        background: #1a1b1d !important;
        border-right: 1px solid #2a2b2e;
    }

    /* Metric cards */
    .stApp [data-testid="stMetric"] {
        background: transparent;
        padding: 4px 0;
    }

    /* Dataframe styling */
    .stApp [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Headings — slightly warmer white */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #f0f0f0;
    }
    /* Slightly larger fonts inside the popup */
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stMarkdownContainer"] {
        font-size: 1rem !important;
    }
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stCaptionContainer"] {
        font-size: 0.92rem !important;
    }
    /* TIGHTER vertical spacing between rows inside the popup */
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stHorizontalBlock"] {
        margin-bottom: 4px !important;
    }
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stVerticalBlock"] > div {
        margin-bottom: 2px;
    }
    /* Plotly chart's outer wrapper — kill default spacing */
    .stApp [data-testid="stVerticalBlockBorderWrapper"]
        .element-container:has(.js-plotly-plot) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Space between the watchlist bar rows */
    .stApp [data-testid="stHorizontalBlock"] {
        margin-bottom: 8px;
    }

    /* Macro caption rows: a small gap between Canadian and US lines */
    .stApp .element-container > div[data-testid="stMarkdownContainer"] {
        margin-bottom: 4px;
    }

    /* Top headline title needs extra room before content */
    .stApp .main h1:first-child,
    .stApp .main [data-testid="stHeading"]:first-child {
        margin-bottom: 24px !important;
    }

    /* High-contrast text selection — bright yellow on black */
    ::selection {
        background: #fbbf24 !important;
        color: #0e1117 !important;
    }
    ::-moz-selection {
        background: #fbbf24 !important;
        color: #0e1117 !important;
    }

    /* Comic Sans everywhere — but preserve icon fonts so glyphs render */
    html, body, .stApp, [class*="st-"], [class*="css-"],
    button, input, textarea, select, code, pre {
        font-family: "Comic Sans MS", "Comic Sans", cursive, sans-serif !important;
    }
    /* Material Icons / Symbols must keep their original font so the icon
       glyphs (▼, ▶, etc.) render — otherwise their names appear as literal
       text like "arrow_drop_down" or "expand_more". */
    [class*="material-icons"],
    [class*="material-symbols"],
    [class*="MuiSvgIcon"],
    .material-icons, .material-icons-outlined, .material-icons-round,
    .material-symbols-outlined, .material-symbols-rounded,
    span[data-testid*="Icon"], span[data-testid*="icon"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons Outlined", "Material Icons" !important;
    }

    /* Tab list container — adds spacing and a subtle bottom border */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #374151 !important;
        padding-bottom: 0;
        margin-bottom: 16px;
    }

    /* Individual tab buttons */
    button[data-baseweb="tab"] {
        height: auto !important;
        padding: 12px 24px !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        transition: background 0.15s ease, color 0.15s ease !important;
    }

    /* Hover state */
    button[data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.04) !important;
        color: #f9fafb !important;
    }

    /* Active tab — bigger, bolder, highlighted background */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(239, 68, 68, 0.10) !important;
        color: #fca5a5 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    /* The red underline on the active tab (Streamlit's default) */
    div[data-baseweb="tab-highlight"] {
        height: 3px !important;
        border-radius: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------- watchlist persistence via URL query param ---------
# Reads `?wl=AAPL,RY.TO,...` from the URL on first load, syncs back on every run.
# This means:
#  - Browser refresh keeps your custom watchlist
#  - You can bookmark a URL with your specific tickers
#  - Sharing the URL shares the watchlist (handy for phone)

def _init_watchlist_from_url():
    if "watchlist_input" not in st.session_state:
        wl = st.query_params.get("wl")
        if wl:
            # Normalize: strip spaces, uppercase
            parts = [p.strip().upper() for p in wl.split(",") if p.strip()]
            st.session_state.watchlist_input = ", ".join(parts)
        else:
            st.session_state.watchlist_input = ", ".join(ss.DEFAULT_WATCHLIST)


def _sync_watchlist_to_url():
    current = st.session_state.get("watchlist_input", "")
    parts = [p.strip().upper() for p in current.split(",") if p.strip()]
    if parts:
        # URL-compact form: comma-separated, no spaces
        st.query_params["wl"] = ",".join(parts)
    elif "wl" in st.query_params:
        del st.query_params["wl"]


_init_watchlist_from_url()


# --------- caching wrappers ---------

@st.cache_data(ttl=900, show_spinner=False)
def cached_macro_ca() -> dict:
    return {
        "USD/CAD": ss.boc_valet("FXUSDCAD"),
        "BoC Rate": ss.boc_valet("V39079"),
        "CA 10Y": ss.boc_valet("BD.CDN.10YR.DQ.YLD"),
        "WTI Crude": ss.yf_spot("CL=F"),
        "Gold": ss.yf_spot("GC=F"),
    }


@st.cache_data(ttl=900, show_spinner=False)
def cached_macro_us() -> dict:
    return {
        "DXY": ss.yf_spot("DX-Y.NYB"),
        "US 10Y": ss.yf_spot("^TNX"),
        "VIX": ss.yf_spot("^VIX"),
        "WTI Crude": ss.yf_spot("CL=F"),
        "Gold": ss.yf_spot("GC=F"),
    }


def _format_macro_value(label: str, val: float | None) -> str:
    if val is None:
        return "—"
    if label == "USD/CAD":
        return f"{val:.4f}"
    if label == "Gold":
        return f"${val:,.0f}"
    if "Crude" in label:
        return f"${val:.2f}"
    if label == "DXY" or label == "VIX":
        return f"{val:.2f}"
    return f"{val:.2f}%"


def render_macro_row(macro: dict, header: str | None = None) -> None:
    """Compact single-line macro display — much smaller than st.metric tiles."""
    parts = []
    for label, val in macro.items():
        v = _format_macro_value(label, val)
        parts.append(
            f'<span style="color:#9ca3af;">{label}</span> '
            f'<b style="color:#e5e7eb;">{v}</b>'
        )
    prefix = (
        f'<span style="color:#9ca3af; font-weight:600; margin-right:10px;">'
        f'{header}</span>'
        if header else ""
    )
    st.markdown(
        f'<div style="font-size:0.85rem; padding:2px 0; line-height:1.6;">'
        f'{prefix}{" &nbsp;·&nbsp; ".join(parts)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _inject_double_click_fullscreen():
    """JS shim: double-click any Plotly chart -> browser fullscreen.
    Press Esc to exit (browser default behavior)."""
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            const pdoc = window.parent.document;
            function attach() {
                const charts = pdoc.querySelectorAll('.js-plotly-plot');
                let added = 0;
                charts.forEach(chart => {
                    if (chart.dataset.fsAttached === '1') return;
                    chart.dataset.fsAttached = '1';
                    chart.addEventListener('dblclick', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        if (pdoc.fullscreenElement) {
                            pdoc.exitFullscreen();
                        } else if (chart.requestFullscreen) {
                            chart.style.background = '#0e1117';
                            chart.requestFullscreen().catch(() => {});
                        }
                    }, true);
                    added++;
                });
                return added;
            }
            // Retry with backoff so we catch charts that render after this script runs
            [100, 300, 600, 1000, 1500, 2500, 4000].forEach(
                d => setTimeout(attach, d)
            );
        })();
        </script>
        """,
        height=0,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def cached_metrics(ticker: str) -> dict:
    """Per-ticker fundamentals cache. 24h TTL — fundamentals update quarterly,
    and Yahoo rate-limits aggressively from cloud data centers."""
    return ss.yf_metrics(ticker)


@st.cache_data(ttl=600, show_spinner=False)
def cached_quotes(tickers: tuple) -> dict:
    """Watchlist tile prices. 10 min TTL — refreshed often enough to feel live
    during market hours but not so often we burn API quota."""
    return ss.fetch_watchlist_quotes(list(tickers))


@st.cache_data(ttl=300, show_spinner=False)
def cached_search(query: str) -> list:
    """Ticker search via Finnhub. 5 min TTL — short queries change often."""
    return ss.finnhub_search(query)


def _add_search_result(symbol: str):
    """on_click handler for search-result Add buttons."""
    try:
        sym = ss.normalize_ticker(symbol)
    except SystemExit:
        return
    current = st.session_state.get(
        "watchlist_input", ", ".join(ss.DEFAULT_WATCHLIST)
    )
    parts = [p.strip() for p in current.split(",") if p.strip()]
    if not any(p.upper() == sym for p in parts):
        parts.append(sym)
        st.session_state.watchlist_input = ", ".join(parts)
        st.session_state["_add_msg"] = f"✅ Added {sym}"
    else:
        st.session_state["_add_msg"] = f"ℹ️ {sym} already in watchlist"
    st.session_state["search_query"] = ""


def _on_tile_click(ticker: str):
    st.session_state["selected_tile"] = ticker


def render_watchlist_bar(tickers: tuple) -> None:
    if not tickers:
        return
    quotes = cached_quotes(tickers)

    # CSS to make tile buttons more compact + look like cards
    st.markdown(
        "<style>"
        "div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] "
        "  div.stButton > button {"
        "    width: 100%;"
        "    background: #1f2937;"
        "    border: 1px solid #374151;"
        "    color: #e5e7eb;"
        "    font-weight: 600;"
        "    font-size: 0.75rem;"
        "    padding: 4px 6px;"
        "    line-height: 1.2;"
        "}"
        "div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] "
        "  div.stButton > button:hover {"
        "    border-color: #6b7280;"
        "    background: #374151;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    cols_per_row = 8
    for row_start in range(0, len(tickers), cols_per_row):
        row_tickers = tickers[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, t in enumerate(row_tickers):
            with cols[i]:
                cols[i].button(
                    t, key=f"tile_btn_{t}",
                    on_click=_on_tile_click, args=(t,),
                    use_container_width=True,
                )
                q = quotes.get(t)
                if not q:
                    st.markdown(
                        '<div style="text-align:center; color:#6b7280; '
                        'font-size:0.75rem; line-height:1.1;">—<br>—</div>',
                        unsafe_allow_html=True,
                    )
                    continue
                chg = q["change_pct"]
                color = "#16a34a" if chg >= 0 else "#dc2626"
                arrow = "▲" if chg >= 0 else "▼"
                sign = "+" if chg >= 0 else ""
                st.markdown(
                    f'<div style="text-align:center; line-height:1.15;">'
                    f'<span style="font-size:0.85rem;">${q["price"]:.2f}</span>'
                    f'<br>'
                    f'<span style="font-size:0.7rem; color:{color}; font-weight:600;">'
                    f'{arrow} {sign}{chg:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


@st.dialog("📊 Quick Analysis", width="large")
def show_quick_analysis_dialog(ticker: str):
    """Modal popup with chart, signal, and key metrics for a ticker.
    Used by the Screener tab when a row is clicked."""
    interval = st.session_state.get("_interval", "1d")
    adx_filter = st.session_state.get("_adx_filter", False)
    stop_loss_pct = st.session_state.get("_stop_loss_pct")

    # Strategy + lookback selectors at top of dialog (default = Bollinger)
    sel_l, sel_r = st.columns([3, 2])
    _strategy_keys = list(ss.STRATEGY_LABELS.keys())
    strategy = sel_l.selectbox(
        "Strategy",
        options=_strategy_keys,
        format_func=lambda k: ss.STRATEGY_LABELS[k],
        index=_strategy_keys.index(ss.DEFAULT_STRATEGY_KEY),
        key=f"dlg_strategy_{ticker}",
    )
    _period_options = ["6mo", "1y", "2y", "5y"]
    _sidebar_period = st.session_state.get("_period", "2y")
    period = sel_r.selectbox(
        "Lookback",
        _period_options,
        index=(_period_options.index(_sidebar_period)
               if _sidebar_period in _period_options else 2),
        key=f"dlg_period_{ticker}",
    )

    try:
        norm_ticker = ss.normalize_ticker(ticker)
    except SystemExit as e:
        st.error(str(e))
        return

    with st.spinner(f"Loading {norm_ticker}…"):
        df, stats = cached_single(norm_ticker, period, interval,
                                  strategy, adx_filter, stop_loss_pct)
    if df is None:
        st.error(f"No data for {norm_ticker}.")
        return

    last = df.iloc[-1]
    if bool(last["BUY"]):
        sig = '<span style="color:#16a34a; font-weight:700;">🟢 BUY</span>'
    elif bool(last["SELL"]):
        sig = '<span style="color:#dc2626; font-weight:700;">🔴 SELL</span>'
    else:
        sig = f'<span style="color:#9ca3af;">⚪ HOLD ({int(last["SCORE"]):+d})</span>'
    st.markdown(
        f'<div style="font-size:0.85rem; color:#9ca3af; padding:4px 0;">'
        f'{sig} &nbsp;·&nbsp; '
        f'<b style="color:#e5e7eb;">${float(last["Close"]):.2f}</b> &nbsp;·&nbsp; '
        f'Strat <b style="color:#e5e7eb;">{stats["total_return"]:+.1%}</b> &nbsp;·&nbsp; '
        f'B&H <b style="color:#e5e7eb;">{stats["buy_hold"]:+.1%}</b> &nbsp;·&nbsp; '
        f'DD <b style="color:#e5e7eb;">{stats.get("max_drawdown", 0):.1%}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    fig = ss.build_chart_plotly(df, norm_ticker, stats, compact=True)
    st.plotly_chart(fig, use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "scrollZoom": True,
                        "doubleClick": False,
                        "modeBarButtonsToRemove": [
                            "select2d", "lasso2d",
                        ],
                    })

    metrics = cached_metrics(norm_ticker)
    if metrics:
        bits = []
        if metrics.get("pe") is not None:
            bits.append(f"P/E <b>{metrics['pe']:.1f}</b>")
        if metrics.get("yield_pct") is not None:
            bits.append(f"Yield <b>{metrics['yield_pct']:.2f}%</b>")
        if metrics.get("upside_pct") is not None:
            bits.append(f"Upside <b>{metrics['upside_pct']:+.1f}%</b>")
        if metrics.get("earn_days") is not None:
            bits.append(f"Earn in <b>{metrics['earn_days']}d</b>")
        if bits:
            st.markdown(
                '<div style="font-size:0.8rem; color:#9ca3af;">'
                + " &nbsp;·&nbsp; ".join(bits)
                + "</div>",
                unsafe_allow_html=True,
            )

    st.caption("📰 News → Single Ticker tab")


def render_quick_analysis():
    """Inline analysis panel shown when a watchlist tile is clicked."""
    selected = st.session_state.get("selected_tile")
    if not selected:
        return
    interval = st.session_state.get("_interval", "1d")
    adx_filter = st.session_state.get("_adx_filter", False)
    stop_loss_pct = st.session_state.get("_stop_loss_pct")

    expanded = st.session_state.get(f"_qv_expanded_{selected}", False)

    # Inline popup uses near-full page width for maximum chart room
    _l, popup_col, _r = st.columns([0.05, 12, 0.05])
    with popup_col, st.container(border=True):
        # Header row: title + strategy + lookback + expand + close
        h1, h2, h3, h_ex, h4 = st.columns([2, 2, 1.3, 0.9, 1])
        h1.markdown(f"#### 🎯 {selected}")
        _strategy_keys = list(ss.STRATEGY_LABELS.keys())
        strategy = h2.selectbox(
            "Strategy",
            options=_strategy_keys,
            format_func=lambda k: ss.STRATEGY_LABELS[k],
            index=_strategy_keys.index(ss.DEFAULT_STRATEGY_KEY),
            key=f"qv_strategy_{selected}",
            label_visibility="collapsed",
        )
        _period_options = ["6mo", "1y", "2y", "5y"]
        _sidebar_period = st.session_state.get("_period", "2y")
        period = h3.selectbox(
            "Lookback",
            _period_options,
            index=(_period_options.index(_sidebar_period)
                   if _sidebar_period in _period_options else 2),
            key=f"qv_period_{selected}",
            label_visibility="collapsed",
        )
        expand_label = "↩ Collapse" if expanded else "🔍 Expand"
        if h_ex.button(expand_label, key=f"qv_expand_{selected}",
                       use_container_width=True,
                       help="Toggle large chart view"):
            st.session_state[f"_qv_expanded_{selected}"] = not expanded
            st.rerun()
        if h4.button("✖ Close", key="close_quick_view",
                     use_container_width=True):
            st.session_state.pop("selected_tile", None)
            st.session_state.pop(f"_qv_expanded_{selected}", None)
            st.rerun()

        try:
            ticker = ss.normalize_ticker(selected)
        except SystemExit as e:
            st.error(str(e))
            return

        with st.spinner(f"Loading {ticker}…"):
            df, stats = cached_single(ticker, period, interval,
                                      strategy, adx_filter, stop_loss_pct)
        if df is None:
            st.error(f"No data for {ticker}.")
            return

        last = df.iloc[-1]

        # Single condensed stats line above the chart (skipped in expanded mode)
        if not expanded:
            if bool(last["BUY"]):
                sig_html = '<span style="color:#16a34a; font-weight:700;">🟢 BUY</span>'
            elif bool(last["SELL"]):
                sig_html = '<span style="color:#dc2626; font-weight:700;">🔴 SELL</span>'
            else:
                sig_html = f'<span style="color:#9ca3af;">⚪ HOLD ({int(last["SCORE"]):+d})</span>'

            st.markdown(
                f'<div style="font-size:1rem; color:#9ca3af; padding:8px 0; line-height:1.7;">'
                f'{sig_html} &nbsp;·&nbsp; '
                f'Last <b style="color:#e5e7eb;">${float(last["Close"]):.2f}</b> &nbsp;·&nbsp; '
                f'Strategy <b style="color:#e5e7eb;">{stats["total_return"]:+.1%}</b> &nbsp;·&nbsp; '
                f'B&H <b style="color:#e5e7eb;">{stats["buy_hold"]:+.1%}</b> &nbsp;·&nbsp; '
                f'Max DD <b style="color:#e5e7eb;">{stats.get("max_drawdown", 0):.1%}</b> &nbsp;·&nbsp; '
                f'Win <b style="color:#e5e7eb;">{stats["win_rate"]:.0%}</b> &nbsp;·&nbsp; '
                f'<b style="color:#e5e7eb;">{stats["trades"]}</b> trades'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Build chart — non-compact (taller) version when expanded
        fig = ss.build_chart_plotly(df, ticker, stats, compact=not expanded)
        st.plotly_chart(fig, use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "scrollZoom": True,
                            "doubleClick": "autosize",
                            "modeBarButtonsToRemove": [
                                "select2d", "lasso2d",
                            ],
                        })
        if not expanded:
            st.caption(
                "💡 **Drag** to pan · **scroll** to zoom · **double-click** to auto-fit Y · "
                "🔍 **Expand** for max-size chart · "
                "📰 news + fundamentals → **Single Ticker** tab"
            )


def plt_close_cleanup(fig):
    """Close the matplotlib figure to free memory between reruns."""
    try:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(tickers: tuple, period: str, interval: str,
                strategy: str, adx_filter: bool,
                stop_loss_pct: float | None) -> list:
    return ss.scan(list(tickers), period, interval,
                   strategy=strategy, adx_filter=adx_filter,
                   stop_loss_pct=stop_loss_pct,
                   metrics_fn=cached_metrics)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_single(ticker: str, period: str, interval: str,
                  strategy: str, adx_filter: bool,
                  stop_loss_pct: float | None):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = ss.compute_indicators(df)
    df = ss.generate_signals(df, strategy=strategy, adx_filter=adx_filter)
    stats = ss.backtest(df, stop_loss_pct=stop_loss_pct)
    return df, stats


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker: str, days: int = 7) -> list:
    return ss.finnhub_news(ticker, days=days)


# --------- header / macro ---------

st.title("📈 Stock Signal Dashboard")

# Watchlist ticker bar — uses session state from the Scan tab's text area,
# falls back to default on first load.
_wl_str = st.session_state.get(
    "watchlist_input", ", ".join(ss.DEFAULT_WATCHLIST)
)
_wl_normalized = []
for _raw in _wl_str.split(","):
    _raw = _raw.strip()
    if not _raw:
        continue
    try:
        _wl_normalized.append(ss.normalize_tsx_ticker(_raw))
    except SystemExit:
        continue
render_watchlist_bar(tuple(_wl_normalized[:30]))
render_quick_analysis()

# Macro context — view selected from sidebar
_macro_view = st.session_state.get("macro_view", "🇨🇦 Canada")
if _macro_view == "🇨🇦 Canada":
    render_macro_row(cached_macro_ca())
elif _macro_view == "🇺🇸 US":
    render_macro_row(cached_macro_us())
else:  # Both
    render_macro_row(cached_macro_ca(), header="🇨🇦 Canadian macro")
    render_macro_row(cached_macro_us(), header="🇺🇸 US macro")

st.divider()


# --------- sidebar ---------

def _add_to_watchlist():
    """on_click handler — add the typed ticker to the watchlist text area."""
    new_t = st.session_state.get("add_ticker_input", "").strip().upper()
    if not new_t:
        return
    try:
        new_t = ss.normalize_ticker(new_t)
    except SystemExit:
        st.session_state["_add_msg"] = f"⚠️ Invalid ticker: {new_t}"
        return
    current = st.session_state.get(
        "watchlist_input", ", ".join(ss.DEFAULT_WATCHLIST)
    )
    parts = [p.strip() for p in current.split(",") if p.strip()]
    if any(p.upper() == new_t for p in parts):
        st.session_state["_add_msg"] = f"ℹ️ {new_t} already in watchlist"
    else:
        parts.append(new_t)
        st.session_state.watchlist_input = ", ".join(parts)
        st.session_state["_add_msg"] = f"✅ Added {new_t}"
    st.session_state.add_ticker_input = ""


def _remove_from_watchlist():
    """Remove the selected ticker from the watchlist."""
    target = st.session_state.get("remove_ticker_select", "")
    if not target:
        return
    current = st.session_state.get("watchlist_input", "")
    parts = [p.strip() for p in current.split(",") if p.strip()]
    parts = [p for p in parts if p.upper() != target.upper()]
    st.session_state.watchlist_input = ", ".join(parts)
    st.session_state["_add_msg"] = f"🗑️ Removed {target}"


with st.sidebar:
    st.header("Watchlist")

    add_col1, add_col2 = st.columns([3, 1])
    add_col1.text_input(
        "Add ticker",
        placeholder="AAPL, RY.TO, BRK.B…",
        key="add_ticker_input",
        label_visibility="collapsed",
        on_change=_add_to_watchlist,
    )
    add_col2.button("➕ Add", on_click=_add_to_watchlist,
                    use_container_width=True)

    # Remove dropdown — populated from current watchlist
    _current_wl = st.session_state.get(
        "watchlist_input", ", ".join(ss.DEFAULT_WATCHLIST)
    )
    _wl_list = [p.strip() for p in _current_wl.split(",") if p.strip()]
    if _wl_list:
        rm_col1, rm_col2 = st.columns([3, 1])
        rm_col1.selectbox(
            "Remove ticker",
            options=[""] + _wl_list,
            key="remove_ticker_select",
            label_visibility="collapsed",
        )
        rm_col2.button("🗑️ Remove", on_click=_remove_from_watchlist,
                       use_container_width=True)

    if "_add_msg" in st.session_state:
        st.caption(st.session_state["_add_msg"])

    st.divider()

    st.header("Settings")
    period = st.selectbox("Lookback period", ["6mo", "1y", "2y", "5y", "10y"],
                          index=2)
    interval = st.selectbox("Bar interval", ["1d", "1wk"], index=0)
    st.radio(
        "Macro view",
        options=["🇨🇦 Canada", "🇺🇸 US", "Both"],
        index=2,
        key="macro_view",
        horizontal=True,
    )

    st.subheader("Strategy")
    strategy = st.selectbox(
        "Signal strategy",
        options=list(ss.STRATEGY_LABELS.keys()),
        format_func=lambda k: ss.STRATEGY_LABELS[k],
        index=0,
    )
    adx_filter = st.checkbox(
        "ADX trend filter (>25)",
        value=False,
        help="Suppresses signals in choppy/range-bound markets. "
             "Improves quality, reduces frequency.",
    )

    st.subheader("Risk")
    stop_choice = st.selectbox(
        "Stop loss",
        options=["None", "5%", "7%", "10%", "15%"],
        index=0,
        help="Exit if cumulative drawdown from entry hits this level.",
    )
    stop_loss_pct = None if stop_choice == "None" else int(stop_choice.rstrip("%")) / 100

    st.divider()
    if st.button("🔄 Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        f"Cached 15 min (scan/macro/news), 1 hr (single). "
        f"Last loaded: {datetime.now().strftime('%H:%M:%S')}"
    )
    st.divider()
    st.caption(
        "Free APIs: yfinance (data + .info), Bank of Canada Valet (macro). "
        "Finnhub key required for analyst & news columns."
    )

# Persist sidebar settings so the quick-view panel (rendered earlier) can read them
st.session_state["_period"] = period
st.session_state["_interval"] = interval
st.session_state["_strategy"] = strategy
st.session_state["_adx_filter"] = adx_filter
st.session_state["_stop_loss_pct"] = stop_loss_pct


# --------- tabs ---------

tab_scan, tab_single, tab_screener, tab_news, tab_help = st.tabs(
    ["📊 Scan", "🔍 Single Ticker", "🎯 Screener", "📰 News", "ℹ️ Help"]
)


# === Scan tab ===
with tab_scan:
    st.subheader("Watchlist Scan")

    default_str = ", ".join(ss.DEFAULT_WATCHLIST)

    # --- Ticker search & add ---
    search_query = st.text_input(
        "🔍 Search tickers (by symbol or company name)",
        placeholder="e.g. AAPL, royal bank, tesla, RY.TO",
        key="search_query",
    )
    if search_query and len(search_query) >= 2:
        with st.spinner("Searching…"):
            results = cached_search(search_query)
        if results:
            st.caption(f"Top {min(8, len(results))} matches — click to add:")
            res_cols = st.columns(2)
            for i, r in enumerate(results[:8]):
                with res_cols[i % 2]:
                    desc = r["description"][:35] + ("…" if len(r["description"]) > 35 else "")
                    st.button(
                        f"➕ **{r['symbol']}** — {desc}",
                        key=f"add_search_{r['symbol']}",
                        on_click=_add_search_result,
                        args=(r["symbol"],),
                        use_container_width=True,
                    )
        else:
            st.caption("No matches. Tip: Finnhub free-tier coverage is best for US + cross-listed names.")

    # --- Bulk edit (hidden by default) ---
    with st.expander("Bulk edit watchlist (paste/clear all)", expanded=False):
        st.text_area(
            "Comma-separated tickers — bare = US, .TO = TSX, .V = TSXV",
            default_str, height=68, key="watchlist_input",
            label_visibility="collapsed",
        )

    watchlist_str = st.session_state.get("watchlist_input", default_str)
    tickers = tuple(t.strip() for t in watchlist_str.split(",") if t.strip())

    with st.spinner(f"Scanning {len(tickers)} tickers…"):
        rows = cached_scan(tickers, period, interval,
                           strategy, adx_filter, stop_loss_pct)

    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]

    if ok:
        action_label = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}
        df_view = pd.DataFrame([
            {
                "Ticker": r["ticker"],
                "Action": action_label.get(r["action"], r["action"]),
                "Score": r["score"],
                "Close": r["close"],
                "RSI": r["rsi"],
                "P/E": r.get("pe"),
                "Yield %": r.get("yield_pct"),
                "Beta": r.get("beta"),
                "Upside %": r.get("upside_pct"),
                "Earn Days": r.get("earn_days"),
                "Buys": r["rec"][0] if r.get("rec") else None,
                "Holds": r["rec"][1] if r.get("rec") else None,
                "Sells": r["rec"][2] if r.get("rec") else None,
                "Trades": r["trades"],
                "Stops": r.get("stops", 0),
                "Win %": r["win_rate"] * 100,
                "Strat %": r["strat"] * 100,
                "Max DD %": r.get("max_dd", 0) * 100,
                "B&H %": r["bh"] * 100,
                "ADX": r.get("adx"),
            }
            for r in ok
        ])
        rank = {"🟢 BUY": 0, "🔴 SELL": 1, "⚪ HOLD": 2}
        df_view = df_view.sort_values(
            by=["Action", "Score"],
            key=lambda c: c.map(rank) if c.name == "Action" else -c,
        )

        st.caption("💡 **Click any ticker** to open the chart in a popup.")

        # Column widths — Ticker, Action, Score, Close, RSI, P/E, Yield, Beta,
        # Upside, Earn, B/H/S, Trades, Win, Strat, MaxDD, B&H, ADX
        col_widths = [1.0, 0.9, 0.6, 0.9, 0.6, 0.7, 0.7, 0.6,
                      0.7, 0.6, 1.0, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6]
        headers = ["Ticker", "Action", "Score", "Close", "RSI", "P/E",
                   "Yield", "Beta", "Up%", "Earn", "B/H/S", "Trd",
                   "Win%", "Strat%", "MaxDD%", "B&H%", "ADX"]
        h = st.columns(col_widths)
        for col, label in zip(h, headers):
            col.markdown(
                f'<span style="font-size:0.85rem; color:#9ca3af; '
                f'font-weight:600;">{label}</span>',
                unsafe_allow_html=True,
            )

        def _fmt(val, kind="num", default="—"):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            if kind == "money":
                return f"${val:.2f}"
            if kind == "pct":
                return f"{val:.2f}"
            if kind == "spct":
                return f"{val:+.1f}"
            if kind == "int":
                return f"{int(val)}"
            if kind == "f1":
                return f"{val:.1f}"
            if kind == "f2":
                return f"{val:.2f}"
            return str(val)

        for r in ok:
            cols = st.columns(col_widths)
            # Ticker — clickable button
            if cols[0].button(r["ticker"], key=f"scan_btn_{r['ticker']}",
                              use_container_width=True):
                show_quick_analysis_dialog(r["ticker"])
            cols[1].markdown(r["action"])
            cols[2].markdown(f"{int(r['score']):+d}")
            cols[3].markdown(_fmt(r.get("close"), "money"))
            cols[4].markdown(_fmt(r.get("rsi"), "f1"))
            cols[5].markdown(_fmt(r.get("pe"), "f1"))
            cols[6].markdown(_fmt(r.get("yield_pct"), "pct"))
            cols[7].markdown(_fmt(r.get("beta"), "f2"))
            up = r.get("upside_pct")
            cols[8].markdown(_fmt(up, "spct") if up is not None else "—")
            ed = r.get("earn_days")
            cols[9].markdown(f"{int(ed)}d" if ed is not None else "—")
            rec = r.get("rec")
            cols[10].markdown(f"{rec[0]}/{rec[1]}/{rec[2]}" if rec else "—")
            cols[11].markdown(_fmt(r.get("trades"), "int"))
            cols[12].markdown(_fmt(r["win_rate"] * 100, "f1"))
            cols[13].markdown(_fmt(r["strat"] * 100, "spct"))
            cols[14].markdown(_fmt(r.get("max_dd", 0) * 100, "f1"))
            cols[15].markdown(_fmt(r["bh"] * 100, "spct"))
            cols[16].markdown(_fmt(r.get("adx"), "f1"))

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY signals",
                  sum(1 for r in ok if r["action"] == "BUY"))
        c2.metric("🔴 SELL signals",
                  sum(1 for r in ok if r["action"] == "SELL"))
        c3.metric("Tickers analyzed", len(ok))
    else:
        st.warning("No tickers analyzed.")

    if bad:
        with st.expander(f"Skipped ({len(bad)})"):
            for r in bad:
                st.text(f"{r['ticker']}: {r['error']}")


# === Single ticker tab ===
with tab_single:
    st.subheader("Single Ticker Analysis")

    raw = st.text_input(
        "Ticker", value="", placeholder="e.g. AAPL, RY.TO, BRK.B",
        key="single_ticker",
    )
    if raw:
        try:
            ticker = ss.normalize_tsx_ticker(raw)
        except SystemExit as e:
            st.error(str(e))
        else:
            with st.spinner(f"Loading {ticker}…"):
                df, stats = cached_single(ticker, period, interval,
                                          strategy, adx_filter, stop_loss_pct)

            if df is None:
                st.error(f"No data for {ticker}.")
            else:
                last = df.iloc[-1]
                if bool(last["BUY"]):
                    st.success(f"🟢 BUY signal today ({df.index[-1].date()})")
                elif bool(last["SELL"]):
                    st.error(f"🔴 SELL signal today ({df.index[-1].date()})")
                else:
                    st.info(f"⚪ HOLD — score {int(last['SCORE']):+d}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Trades", stats["trades"])
                c2.metric("Win Rate", f"{stats['win_rate']:.0%}")
                c3.metric("Strategy", f"{stats['total_return']:+.1%}")
                c4.metric("Buy & Hold", f"{stats['buy_hold']:+.1%}",
                          delta=f"{(stats['total_return']-stats['buy_hold'])*100:+.1f}%")
                c5.metric("Max DD", f"{stats.get('max_drawdown', 0):.1%}",
                          help="Worst peak-to-trough drawdown of the strategy")
                if stats.get("stops_hit", 0) > 0:
                    st.caption(f"⚠️ {stats['stops_hit']} of {stats['trades']} "
                               f"trades exited via stop-loss")

                metrics = ss.yf_metrics(ticker)
                if metrics:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("P/E", f"{metrics.get('pe', 0):.1f}"
                              if metrics.get("pe") else "—")
                    c2.metric("Yield",
                              f"{metrics.get('yield_pct', 0):.2f}%"
                              if metrics.get("yield_pct") else "—")
                    c3.metric("Beta", f"{metrics.get('beta', 0):.2f}"
                              if metrics.get("beta") else "—")
                    c4.metric("Analyst Upside",
                              f"{metrics.get('upside_pct', 0):+.1f}%"
                              if metrics.get("upside_pct") else "—")
                    c5.metric("Earnings In",
                              f"{metrics.get('earn_days')}d"
                              if metrics.get("earn_days") is not None else "—")

                fig = ss.build_chart_plotly(df, ticker, stats)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": True,
                                        "scrollZoom": True})


# === Screener tab ===
with tab_screener:
    st.subheader("Multi-strategy Buy Screener")
    st.caption(
        "Find stocks where **both** Bollinger Mean Reversion (price at lower band) "
        "and **RSI oversold** are firing — a confluence of two independent oversold signals."
    )

    sc_col1, sc_col2 = st.columns([1, 1])
    universe_choice = sc_col1.radio(
        "Universe",
        options=[
            "S&P 100 (~100)",
            "S&P 500 (~500)",
            "TSX 60 (~60)",
            "TSX Composite (~250)",
            "Popular ETFs (~80)",
            "All US + TSX + ETFs (~850)",
            "Custom watchlist",
        ],
        index=0,
        key="screener_universe",
    )
    lookback_days = sc_col2.select_slider(
        "Bollinger BUY lookback window",
        options=[5, 10, 22, 45, 66],
        value=22,
        format_func=lambda d: f"{d}d (~{d//5}wk)" if d < 22 else
                              ("1 month" if d == 22 else
                               "2 months" if d == 45 else
                               "3 months"),
        key="screener_lookback",
    )
    rsi_thresh = sc_col2.slider(
        "RSI threshold (≤ to qualify as oversold)",
        min_value=20, max_value=50, value=35, step=1,
        key="screener_rsi_thresh",
    )

    bb_col, rsi_col, dip_col = sc_col1.columns(3)
    require_bb = bb_col.checkbox(
        "Bollinger BUY", value=True, key="screener_require_bb",
        help="Bollinger lower-band touch in the lookback window",
    )
    require_rsi = rsi_col.checkbox(
        "RSI oversold", value=True, key="screener_require_rsi",
        help="Current RSI ≤ threshold",
    )
    require_dip = dip_col.checkbox(
        "Recent dip", value=False, key="screener_require_dip",
        help="Price has dropped at least the threshold % over the past N days",
    )

    dip_a, dip_b = sc_col2.columns(2)
    dip_window = dip_a.slider(
        "Dip window (days)", 2, 10, 4, key="screener_dip_window",
    )
    dip_threshold = dip_b.slider(
        "Dip threshold % (≤ to qualify)",
        -15, -1, -3, key="screener_dip_threshold",
    )

    @st.cache_data(ttl=86400 * 7, show_spinner=False)
    def _cached_sp500() -> list:
        return ss.get_sp500()

    @st.cache_data(ttl=86400 * 7, show_spinner=False)
    def _cached_tsx_composite() -> list:
        return ss.get_tsx_composite()

    if universe_choice.startswith("S&P 100"):
        universe = ss.UNIVERSE_SP100
    elif universe_choice.startswith("S&P 500"):
        universe = _cached_sp500()
    elif universe_choice.startswith("TSX 60"):
        universe = ss.UNIVERSE_TSX60
    elif universe_choice.startswith("TSX Composite"):
        universe = _cached_tsx_composite()
    elif universe_choice.startswith("Popular ETFs"):
        universe = ss.UNIVERSE_POPULAR_ETFS
    elif universe_choice.startswith("All US"):
        universe = list(dict.fromkeys(
            list(_cached_sp500()) + list(_cached_tsx_composite())
            + list(ss.UNIVERSE_POPULAR_ETFS)
        ))
    else:
        universe = list(tickers)

    n_batches = (len(universe) + 99) // 100
    eta_seconds = n_batches * 5
    st.caption(
        f"Will scan **{len(universe)}** tickers in {n_batches} batched API call"
        f"{'s' if n_batches != 1 else ''}. ETA ~{eta_seconds}s."
    )

    run_col, clear_col = st.columns([3, 1])
    if run_col.button("🎯 Run screener", type="primary",
                      disabled=not (require_bb or require_rsi or require_dip),
                      use_container_width=True):
        progress = st.progress(0.0, text=f"Scanning {len(universe)} tickers…")

        def _progress_cb(frac: float, hits: int):
            progress.progress(
                frac,
                text=f"Scanning… {int(frac*100)}% — {hits} match{'es' if hits != 1 else ''} so far",
            )

        matches = ss.screen_buy_signals(
            universe,
            rsi_threshold=rsi_thresh,
            lookback_bars=lookback_days,
            require_bollinger=require_bb,
            require_rsi=require_rsi,
            require_dip=require_dip,
            dip_window=dip_window,
            dip_threshold_pct=float(dip_threshold),
            progress_callback=_progress_cb,
        )
        progress.empty()
        # Persist results so they survive across reruns triggered by row clicks
        st.session_state["_screener_matches"] = matches
    if clear_col.button("Clear results", use_container_width=True,
                        disabled="_screener_matches" not in st.session_state):
        st.session_state.pop("_screener_matches", None)
        st.rerun()

    matches = st.session_state.get("_screener_matches")
    if matches is None:
        pass  # nothing to show
    elif not matches:
        st.info(
            "No matches. Try a wider lookback window, higher RSI threshold, "
            "or untick one filter."
        )
    else:
        today_matches = [m for m in matches if m.get("bb_buy_age") == 0]
        earlier_matches = [m for m in matches if m.get("bb_buy_age") and m["bb_buy_age"] > 0]

        summary_parts = [f"**{len(matches)}** total"]
        if today_matches:
            summary_parts.append(f"🎯 **{len(today_matches)} TODAY**")
        if earlier_matches:
            summary_parts.append(f"{len(earlier_matches)} within window")
        st.success(" · ".join(summary_parts))

        def _fmt_age(age):
            if age is None:
                return "—"
            if age == 0:
                return "today"
            if age == 1:
                return "1d ago"
            return f"{age}d ago"

        st.caption("💡 **Click any ticker** to open the chart in a popup.")

        # Header row — added Dip% column
        col_widths = [1.2, 1.0, 0.7, 1.2, 1.4, 1.0, 0.8, 0.8]
        h = st.columns(col_widths)
        for col, label in zip(h, ["Ticker", "Price", "RSI", "vs BB Lower",
                                  "BB BUY Date", "Age", "Dip%", "RSI OS"]):
            col.markdown(f"**{label}**")

        for m in matches:
            cols = st.columns(col_widths)
            if cols[0].button(m["ticker"], key=f"sc_view_{m['ticker']}",
                              use_container_width=True):
                show_quick_analysis_dialog(m["ticker"])
            cols[1].markdown(f"${m['price']:.2f}")
            cols[2].markdown(f"{m['rsi']:.1f}")
            bb_color = "#dc2626" if m["bb_distance_pct"] < 0 else "#9ca3af"
            cols[3].markdown(
                f'<span style="color:{bb_color}">{m["bb_distance_pct"]:+.2f}%</span>',
                unsafe_allow_html=True,
            )
            cols[4].markdown(m.get("bb_buy_date") or "—")
            age_label = _fmt_age(m.get("bb_buy_age"))
            age_color = "#16a34a" if m.get("bb_buy_age") == 0 else "#e5e7eb"
            cols[5].markdown(
                f'<span style="color:{age_color}">{age_label}</span>',
                unsafe_allow_html=True,
            )
            dip = m.get("dip_pct")
            if dip is None:
                cols[6].markdown("—")
            else:
                dip_color = "#dc2626" if dip < 0 else "#16a34a"
                cols[6].markdown(
                    f'<span style="color:{dip_color}">{dip:+.2f}%</span>',
                    unsafe_allow_html=True,
                )
            cols[7].markdown("✓" if m["rsi_oversold"] else "·")


# === News tab ===
with tab_news:
    st.subheader("Recent News")

    if not ss.FINNHUB_API_KEY:
        st.warning("Set FINNHUB_API_KEY in .env to enable news.")
    else:
        col1, col2 = st.columns([3, 1])
        news_raw = col1.text_input("Ticker for news", "AAPL",
                                   key="news_ticker",
                                   help="Try US tickers — Finnhub TSX news coverage is sparse")
        days = col2.number_input("Days back", 1, 30, 7)

        if news_raw:
            try:
                # Allow US tickers in news view (no TSX validation)
                t = news_raw.strip().upper()
                with st.spinner(f"Loading news for {t}…"):
                    articles = cached_news(t, days=days)

                if not articles:
                    st.info(f"No news returned for {t} in the last {days} days.")
                else:
                    st.success(f"Found {len(articles)} articles")
                    for art in articles[:30]:
                        try:
                            ts = datetime.fromtimestamp(art.get("datetime", 0))
                            date_str = ts.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError, OSError):
                            date_str = "?"
                        with st.container(border=True):
                            st.caption(
                                f"📅 {date_str}  |  📰 {art.get('source', '')}"
                            )
                            st.markdown(f"**{art.get('headline', '')}**")
                            if art.get("summary"):
                                st.write(art["summary"])
                            if art.get("url"):
                                st.markdown(f"[Read more →]({art['url']})")
            except Exception as e:
                st.error(f"Error: {e}")


# === Help tab ===
with tab_help:
    st.markdown("""
### Quick guide

**Scan tab**: full watchlist with signals + fundamentals + analyst data.
- Edit the ticker list (top text area) to scan custom names
- BUYs sort first, then SELLs, then HOLDs
- Click any column header to re-sort

**Single Ticker tab**: drill into one ticker — full chart, backtest, fundamentals.
- Bare tickers (e.g. `AAPL`) are US listings
- `.TO` / `.V` / `.CN` are TSX / TSXV / CSE
- Class shares: type `BRK.B` (auto-converted to Yahoo's `BRK-B`)
- Period & interval are set in the sidebar

**News tab**: Finnhub headlines. Coverage is best for US-listed names.

### Signal logic

Each ticker gets a score from -3 to +3 based on:
- **+1** RSI(14) crosses above 30 (oversold bounce)
- **+1** MACD line crosses above signal line
- **+1** SMA50 crosses above SMA200 (golden cross)
- **-1** RSI crosses below 70 (overbought)
- **-1** MACD bearish cross
- **-1** SMA death cross

Triggers within a 5-bar window are summed. **Score ≥ +2 fires BUY, ≤ -2 fires SELL.**

### Macro context

Top bar shows live macro indicators that drive TSX performance:
- USD/CAD — affects exporters, financials
- BoC overnight rate — bank margins, REITs, growth stocks
- 10Y bond yield — long-duration assets
- WTI crude — energy sector (~17% of TSX)
- Gold — mining sector (~10% of TSX)

### Data sources

| Data | Source | Cost |
|------|--------|------|
| OHLC, P/E, yield, beta, target, earnings | yfinance | Free, no key |
| Analyst recommendations | Finnhub | Free w/ key, US + cross-listed TSX |
| News headlines | Finnhub | Free w/ key, US-heavy coverage |
| Macro (FX, rates) | Bank of Canada Valet | Free, no key |
| Macro (oil, gold) | yfinance futures | Free, no key |

This tool is a **screener**, not a trading system. Use signals as
"investigate further" — never as auto-buy/sell.
""")


# Mirror the current watchlist into the URL so refreshes (and bookmarks /
# share links) preserve the user's tickers. Runs at the very end so it
# captures any changes made during this script run.
_sync_watchlist_to_url()

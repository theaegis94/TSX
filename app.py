"""Streamlit dashboard for the TSX signal scanner.

Run with:    streamlit run app.py
"""

import json
import os
import pathlib
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

import stock_signals as ss

st.set_page_config(
    page_title="Stock Signals",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)


# --------- auto-refresh on session start / new day ---------
# Streamlit's st.cache_data persists across browser refreshes (server-side).
# That meant a page refresh showed yesterday's prices when the user came back
# the next morning. Two triggers force-clear the cache to keep data fresh:
#   1) First run of a fresh Streamlit session (typical browser refresh in
#      Streamlit Cloud creates a new session token → state resets → we clear).
#   2) Calendar day has rolled over since the last clear (catches the case
#      where session state survived overnight).
# A manual "🔄 Refresh data" button (rendered below) lets the user force a
# clear at any time.
def _maybe_clear_stale_cache() -> None:
    today_key = datetime.now().strftime("%Y-%m-%d")
    last_day = st.session_state.get("_cache_day")
    first_run = "_cache_initialized" not in st.session_state
    if first_run or last_day != today_key:
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.session_state["_cache_initialized"] = True
        st.session_state["_cache_day"] = today_key
        st.session_state["_cache_cleared_at"] = datetime.now()


_maybe_clear_stale_cache()


# Global CSS — make tabs larger, bolder, with clearer active state
st.markdown(
    """
    <style>
    /* Compact spacing between major page sections — was 14px which caused
       a big empty gap under the watchlist bar before the macro row. */
    .stApp .main .block-container > div > div > div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 4px;
    }

    /* st.divider() defaults to ~16px top + ~16px bottom margin — tighten. */
    .stApp hr {
        margin: 6px 0 !important;
    }

    /* Make the Streamlit header's background match the page so the
       Share/star/edit/GitHub Cloud icons blend in but the sidebar
       toggle remains accessible. */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Trim the empty padding above the title now that the header is short */
    .stApp .main .block-container,
    .stApp [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
    }

    /* Lighter Edge / Windows 11-style theme */
    .stApp { background: #3a3b3e !important; }

    /* Bordered containers (popups) styled as soft rounded cards */
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: 12px !important;
        margin-bottom: 12px !important;
        padding: 16px 22px !important;
        background: #4a4b4e !important;
        border: 1px solid #5a5b5e !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    /* Buttons — rounded pills with subtle hover */
    .stApp button[kind="primary"],
    .stApp .stButton > button {
        background: #4a4b4e !important;
        color: #f0f0f0 !important;
        border: 1px solid #5a5b5e !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stApp .stButton > button:hover {
        background: #5a5b5e !important;
        border-color: #6a6b6e !important;
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
        background: #4a4b4e !important;
        border: 1px solid #5a5b5e !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }

    /* Sidebar — slightly darker than page for separation */
    .stApp [data-testid="stSidebar"] {
        background: #33343a !important;
        border-right: 1px solid #44454a;
    }

    /* Buttons inside horizontal column groups: equal height + small
       font + allow wrapping to 2 lines so labels stay readable when
       there are many columns. */
    .stApp [data-testid="stHorizontalBlock"] [data-testid="stColumn"]
        .stButton button {
        min-height: 56px !important;
        font-size: 0.73rem !important;
        padding: 4px 6px !important;
        white-space: normal;
        line-height: 1.2;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        word-break: keep-all;
        hyphens: none;
    }
    /* Inner <p> wrap that Streamlit creates inside the button */
    .stApp [data-testid="stHorizontalBlock"] [data-testid="stColumn"]
        .stButton button p {
        font-size: 0.73rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        white-space: normal !important;
        text-align: center !important;
        word-break: keep-all;
    }

    /* Metric cards — compact */
    .stApp [data-testid="stMetric"] {
        background: transparent;
        padding: 2px 0;
    }
    .stApp [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #9ca3af !important;
    }
    .stApp [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        line-height: 1.3 !important;
        font-weight: 600 !important;
    }
    .stApp [data-testid="stMetricValue"] > div {
        font-size: 1.05rem !important;
    }
    .stApp [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
        padding-top: 0 !important;
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

    /* Space between the watchlist bar rows + vertically center column contents
       so ticker buttons align with adjacent text cells. */
    .stApp [data-testid="stHorizontalBlock"] {
        margin-bottom: 8px;
        align-items: center !important;
    }
    /* Make column children fill the column so vertical centering works */
    .stApp [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
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

    /* Fade-in animation for UI elements as they appear (fast, ~180ms) */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    /* Major section blocks — popup containers, charts, alerts */
    .stApp [data-testid="stVerticalBlockBorderWrapper"],
    .stApp [data-testid="stAlert"],
    .stApp .stPlotlyChart,
    .stApp [data-testid="stDataFrame"],
    .stApp [role="dialog"] {
        animation: fadeInUp 0.18s ease-out;
    }
    /* Lighter fade for general content blocks */
    .stApp .element-container {
        animation: fadeIn 0.15s ease-out;
    }
    /* Tab content swap */
    .stApp [data-baseweb="tab-panel"] {
        animation: fadeIn 0.12s ease-out;
    }
    /* Smooth color/transform transitions on hover for interactive elements */
    .stApp button,
    .stApp [data-baseweb="select"],
    .stApp .stSelectbox,
    .stApp [data-testid="stMetric"] {
        transition: background 0.15s ease, border-color 0.15s ease,
                    transform 0.15s ease;
    }

    /* Thicker horizontal dividers (st.divider) */
    .stApp hr,
    .stApp [data-testid="stHorizontalRule"],
    .stApp [data-testid="stHeading"] + hr {
        border: none !important;
        height: 2px !important;
        background: #6a6b6e !important;
        margin: 12px 0 !important;
        opacity: 1 !important;
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
        margin-bottom: 6px;
    }

    /* Individual tab buttons */
    button[data-baseweb="tab"] {
        height: auto !important;
        padding: 4px 24px !important;
        font-size: 1.640625rem !important;
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
        font-size: 1.71875rem !important;
    }


    /* The red underline on the active tab (Streamlit's default) */
    div[data-baseweb="tab-highlight"] {
        height: 3px !important;
        border-radius: 2px !important;
    }

    /* === Nested tabs (sub-tabs inside a tab) — smaller than top-level === */
    /* These are tabs inside another tab's content — typically used for
       sub-navigation within a section like the Screener tab. */
    [data-baseweb="tab-panel"] [data-baseweb="tab-list"]
        button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        padding: 4px 16px !important;
    }
    [data-baseweb="tab-panel"] [data-baseweb="tab-list"]
        button[data-baseweb="tab"][aria-selected="true"] {
        font-size: 1.0rem !important;
        background: rgba(96, 165, 250, 0.12) !important;
        color: #93c5fd !important;
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

# --- Multi-list watchlist support ---
# All saved lists live in watchlists.json: {name: [tickers]}. One list is
# "active" at any time — its tickers populate `watchlist_input`, which all
# downstream features (snapshot, tile bar, Custom Patterns, alerts) read.
WATCHLISTS_PATH = pathlib.Path("watchlists.json")


def _load_watchlists() -> dict:
    """Load all saved watchlists. Returns {name: [tickers]} dict."""
    if not WATCHLISTS_PATH.exists():
        return {"Default": list(ss.DEFAULT_WATCHLIST)}
    try:
        data = json.loads(WATCHLISTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            # Normalize: ensure each value is a list of strings
            return {
                k: [str(t).strip().upper() for t in v if str(t).strip()]
                for k, v in data.items()
                if isinstance(v, list)
            }
    except Exception:
        pass
    return {"Default": list(ss.DEFAULT_WATCHLIST)}


def _save_watchlists(d: dict) -> None:
    try:
        WATCHLISTS_PATH.write_text(
            json.dumps(d, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# --------- per-ticker target prices ---------
# Stored in target_prices.json as {TICKER: float}. Independent of watchlist
# membership — a target survives even if you remove the ticker, and is shared
# across all named watchlists (a $TSLA target of $300 means $300 regardless
# of which list TSLA is in).
TARGET_PRICES_PATH = pathlib.Path("target_prices.json")


def _load_target_prices() -> dict:
    if not TARGET_PRICES_PATH.exists():
        return {}
    try:
        data = json.loads(TARGET_PRICES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if fv > 0:
                        out[str(k).upper()] = fv
                except (TypeError, ValueError):
                    continue
            return out
    except Exception:
        pass
    return {}


def _save_target_prices(d: dict) -> None:
    try:
        TARGET_PRICES_PATH.write_text(
            json.dumps(d, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _set_target_price(ticker: str, price: float | None) -> None:
    """Persist a target price for a ticker. Pass None or 0 to clear."""
    targets = st.session_state.setdefault(
        "_target_prices", _load_target_prices()
    )
    t = ticker.strip().upper()
    if not t:
        return
    if price is None or price <= 0:
        targets.pop(t, None)
    else:
        targets[t] = float(price)
    _save_target_prices(targets)


def _set_active_watchlist(name: str) -> None:
    """Switch active list. Updates session state + URL."""
    all_lists = st.session_state.get("_all_watchlists", {})
    if name not in all_lists:
        return
    st.session_state["_active_watchlist"] = name
    st.session_state["watchlist_input"] = ", ".join(all_lists[name])
    st.session_state["_wl_from_url"] = True
    st.query_params["list"] = name
    _sync_watchlist_to_url()


def _init_watchlist_from_url():
    if "watchlist_input" not in st.session_state:
        # Load all named watchlists from disk
        all_lists = _load_watchlists()
        st.session_state["_all_watchlists"] = all_lists

        # Pick active list: URL ?list= first, then first list, then "Default"
        names = list(all_lists.keys())
        requested = st.query_params.get("list", "")
        active_name = (
            requested if requested in all_lists
            else (names[0] if names else "Default")
        )
        st.session_state["_active_watchlist"] = active_name

        # Tickers for the active list: prefer URL ?wl= over saved tickers
        # (handy when sharing — pasted URL should override saved state)
        wl = st.query_params.get("wl")
        if wl:
            parts = [p.strip().upper() for p in wl.split(",") if p.strip()]
            st.session_state["watchlist_input"] = ", ".join(parts)
            st.session_state["_wl_from_url"] = True
            # Persist URL tickers into the active list
            all_lists[active_name] = parts
            _save_watchlists(all_lists)
        else:
            st.session_state["watchlist_input"] = ", ".join(
                all_lists.get(active_name, ss.DEFAULT_WATCHLIST)
            )
            st.session_state["_wl_from_url"] = bool(
                all_lists.get(active_name)
            )


def _sync_watchlist_to_url():
    """Write the watchlist to URL ?wl=, ?list=NAME, and also persist to
    watchlists.json under the active list's name.
    """
    current = st.session_state.get("watchlist_input", "")
    parts = [p.strip().upper() for p in current.split(",") if p.strip()]
    is_user_owned = st.session_state.get("_wl_from_url", False)
    if parts and is_user_owned:
        st.query_params["wl"] = ",".join(parts)
    elif "wl" in st.query_params and not parts:
        del st.query_params["wl"]

    # Sync active list name to URL
    active = st.session_state.get("_active_watchlist", "Default")
    st.query_params["list"] = active

    # Persist active list's tickers back into watchlists.json
    all_lists = st.session_state.get("_all_watchlists")
    if all_lists is not None and is_user_owned:
        all_lists[active] = parts
        _save_watchlists(all_lists)
        st.session_state["_all_watchlists"] = all_lists


def _on_bulk_edit_watchlist():
    """Bulk-edit text area on_change → claim ownership + sync URL."""
    st.session_state["_wl_from_url"] = True
    _sync_watchlist_to_url()


_init_watchlist_from_url()


def _consume_open_param():
    """Read ?open=TICKER and ?from_tab=News from URL, set state, clean URL."""
    if "open" in st.query_params:
        t = st.query_params["open"]
        if t:
            st.session_state["selected_tile"] = t.upper()
        del st.query_params["open"]
    if "from_tab" in st.query_params:
        ft = st.query_params["from_tab"]
        if ft:
            st.session_state["__active_tab"] = ft
            # Re-click the tab for 2 reruns: popup-open render + popup-close render
            st.session_state["__redirect_pending"] = 2
        del st.query_params["from_tab"]


_consume_open_param()


def _chip_href(ticker: str, from_tab: str = "News") -> str:
    """URL preserving current query params + open=ticker + from_tab."""
    qp = dict(st.query_params)
    qp["open"] = ticker
    qp["from_tab"] = from_tab
    parts = [f"{k}={v}" for k, v in qp.items()]
    return "?" + "&".join(parts)


def _restore_active_tab():
    """JS-click the originating tab after a rerun caused by a chip click."""
    target = st.session_state.get("__active_tab")
    pending = st.session_state.get("__redirect_pending", 0)
    if not target or pending <= 0:
        st.session_state.pop("__active_tab", None)
        st.session_state.pop("__redirect_pending", None)
        return
    import streamlit.components.v1 as components
    components.html(
        f"""<script>
        (function() {{
            const target = "{target}";
            const ttd = window.parent.document;
            const click = () => {{
                const tabs = ttd.querySelectorAll(
                    'button[data-baseweb="tab"]'
                );
                let found = null;
                tabs.forEach(t => {{
                    if (t.innerText.indexOf(target) !== -1) found = t;
                }});
                if (found && found.getAttribute('aria-selected') !== 'true') {{
                    found.click();
                }}
            }};
            [60, 200, 500, 1000].forEach(d => setTimeout(click, d));
        }})();
        </script>""",
        height=0,
    )
    st.session_state["__redirect_pending"] = pending - 1


def _inject_tab_persistence():
    """Persist the active tab in localStorage so it survives reruns.

    Streamlit's st.tabs() resets to the first tab on every rerun. This JS:
      1. Reads localStorage on every render → clicks that tab if set
      2. Attaches click listeners to all tabs → saves the clicked tab name

    The chip-click flow (`__active_tab` + `__redirect_pending`) takes
    precedence over localStorage so popups still return the user to the
    originating tab.
    """
    import streamlit.components.v1 as components
    components.html(
        """<script>
        (function() {
            const ttd = window.parent.document;
            const KEY = 'streamlit_active_tab';

            function findTab(name) {
                const tabs = ttd.querySelectorAll(
                    'button[data-baseweb="tab"]'
                );
                let found = null;
                tabs.forEach(t => {
                    if (t.innerText.indexOf(name) !== -1) found = t;
                });
                return found;
            }

            function restore() {
                const saved = window.parent.localStorage.getItem(KEY);
                if (!saved) return;
                const tab = findTab(saved);
                if (tab && tab.getAttribute('aria-selected') !== 'true') {
                    tab.click();
                }
            }

            function attachClickSavers() {
                const tabs = ttd.querySelectorAll(
                    'button[data-baseweb="tab"]'
                );
                tabs.forEach(t => {
                    if (t.dataset.tabPersistAttached === '1') return;
                    t.dataset.tabPersistAttached = '1';
                    t.addEventListener('click', function() {
                        try {
                            window.parent.localStorage.setItem(
                                KEY, t.innerText
                            );
                        } catch (e) {}
                    });
                });
            }

            // Restore + attach listeners on every component injection
            [60, 200, 500, 1000].forEach(d => setTimeout(() => {
                attachClickSavers();
                restore();
            }, d));
        })();
        </script>""",
        height=0,
    )


def _is_dark_theme() -> bool:
    """Detect Streamlit's current theme so chart line colors can adapt."""
    try:
        return (st.get_option("theme.base") or "dark") != "light"
    except Exception:
        return True


# --------- caching wrappers ---------

@st.cache_data(ttl=180, show_spinner=False)
def cached_macro_ca() -> dict:
    return {
        "USD/CAD": ss.boc_valet("FXUSDCAD"),
        "BoC Rate": ss.boc_valet("V39079"),
        "CA 10Y": ss.boc_valet("BD.CDN.10YR.DQ.YLD"),
        "WTI Crude": ss.yf_spot("CL=F"),
        "Gold": ss.yf_spot("GC=F"),
    }


@st.cache_data(ttl=180, show_spinner=False)
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


def _inject_auto_rescale_y():
    """Auto-rescale Y axes to fit visible X window — on initial load AND on zoom/pan."""
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            const pdoc = window.parent.document;

            function rescale(chart) {
                const layout = chart.layout || {};
                let xMin, xMax;
                const ax = layout.xaxis || {};
                if (ax.range) {
                    xMin = new Date(ax.range[0]).getTime();
                    xMax = new Date(ax.range[1]).getTime();
                } else {
                    // Fall back to data extent
                    const allX = [];
                    chart.data.forEach(tr => {
                        if (tr.x) tr.x.forEach(v => {
                            const t = new Date(v).getTime();
                            if (!isNaN(t)) allX.push(t);
                        });
                    });
                    if (!allX.length) return;
                    xMin = Math.min.apply(null, allX);
                    xMax = Math.max.apply(null, allX);
                }

                // Only rescale the Price panel (yaxis); leave RSI (yaxis2)
                // and MACD (yaxis3) at their fixed initial ranges.
                const yByAxis = {};
                chart.data.forEach(trace => {
                    if (!trace.x || !trace.y) return;
                    const yref = trace.yaxis || 'y';
                    if (yref !== 'y') return;  // skip y2, y3
                    if (!yByAxis[yref]) yByAxis[yref] = [];
                    for (let i = 0; i < trace.x.length; i++) {
                        const t = new Date(trace.x[i]).getTime();
                        if (t < xMin || t > xMax) continue;
                        const v = trace.y[i];
                        if (v == null || !isFinite(v)) continue;
                        yByAxis[yref].push(v);
                    }
                });

                const updates = {};
                Object.keys(yByAxis).forEach(yref => {
                    const vals = yByAxis[yref];
                    if (!vals.length) return;
                    const lo = Math.min.apply(null, vals);
                    const hi = Math.max.apply(null, vals);
                    const range = hi - lo;
                    // Tight fit: 5% padding above and below the visible range
                    // so the chart fills the available y-space.
                    const pad = range > 0 ? range * 0.05 : Math.abs(hi) * 0.02;
                    // Floor at -1 (invisible — tick0=0 hides the negative band)
                    // for a slight visual buffer at the bottom.
                    const yLo = Math.max(-1, lo - pad);
                    const yHi = hi + pad;
                    const axName = yref === 'y'
                        ? 'yaxis' : 'yaxis' + yref.slice(1);
                    updates[axName + '.range'] = [yLo, yHi];
                    updates[axName + '.autorange'] = false;
                    updates[axName + '.minallowed'] = -1;
                });
                if (Object.keys(updates).length) {
                    try {
                        window.parent.Plotly.relayout(chart, updates);
                    } catch (e) {}
                }
            }

            function attach() {
                const charts = pdoc.querySelectorAll('.js-plotly-plot');
                charts.forEach(chart => {
                    if (chart.dataset.autoYAttached === '1') return;
                    // Plotly attaches `.on` to the chart div once initialized.
                    // If it's not there yet, skip this round; the next
                    // setTimeout retry will attempt again.
                    if (typeof chart.on !== 'function') return;
                    if (!chart.data || !chart.data.length) return;
                    chart.dataset.autoYAttached = '1';

                    // Initial rescale once chart has data
                    setTimeout(() => rescale(chart), 50);

                    // Re-rescale on every zoom/pan (final state)
                    chart.on('plotly_relayout', function(ev) {
                        // Clamp Price y-axis (yaxis) bottom to -1 — block any
                        // user pan/zoom attempt that drops below -$1.
                        const yLo = ev['yaxis.range[0]'];
                        const yHi = ev['yaxis.range[1]'];
                        const yRng = ev['yaxis.range'];
                        let needClamp = false;
                        let newRange = null;
                        if (yLo !== undefined && yLo < -1) {
                            needClamp = true;
                            newRange = [-1, yHi !== undefined ? yHi
                                : (chart.layout.yaxis.range
                                   ? chart.layout.yaxis.range[1] : 1)];
                        } else if (yRng && yRng[0] < -1) {
                            needClamp = true;
                            newRange = [-1, yRng[1]];
                        }
                        if (needClamp) {
                            try {
                                window.parent.Plotly.relayout(chart, {
                                    'yaxis.range': newRange,
                                    'yaxis.autorange': false,
                                });
                            } catch (e) {}
                            return;
                        }

                        const hasXChange =
                            ev['xaxis.range[0]'] !== undefined ||
                            ev['xaxis.autorange'] !== undefined ||
                            ev['xaxis.range'] !== undefined;
                        if (!hasXChange) return;
                        setTimeout(() => rescale(chart), 0);
                    });

                    // Live rescale DURING drag (rAF-throttled for performance)
                    let liveTicking = false;
                    chart.on('plotly_relayouting', function(ev) {
                        const hasXChange =
                            ev['xaxis.range[0]'] !== undefined ||
                            ev['xaxis.range'] !== undefined;
                        if (!hasXChange || liveTicking) return;
                        liveTicking = true;
                        window.requestAnimationFrame(() => {
                            rescale(chart);
                            liveTicking = false;
                        });
                    });
                });
            }
            [100, 300, 600, 1200, 2500, 4000].forEach(d => setTimeout(attach, d));
        })();
        </script>
        """,
        height=0,
    )


def _inject_scroll_to_pan():
    """Mouse interactions:
       - Wheel: zoom X axis (cursor-anchored). Up=in, down=out.
       - Middle-mouse-button drag: pan X axis.
    """
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            const pdoc = window.parent.document;

            function attachZoom(chart) {
                if (chart.dataset.scrollZoomAttached === '1') return;
                chart.dataset.scrollZoomAttached = '1';

                // ============ Wheel = zoom on X (cursor-anchored) ============
                const wheelHandler = function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    let layout = chart.layout || {};
                    let ax = layout.xaxis || {};
                    // Fallback: read from DOM data if layout not populated
                    let xMin, xMax;
                    if (ax.range) {
                        xMin = new Date(ax.range[0]).getTime();
                        xMax = new Date(ax.range[1]).getTime();
                    } else {
                        return;
                    }
                    const span = xMax - xMin;
                    if (!isFinite(span) || span <= 0) return;

                    const zoomFactor = e.deltaY > 0 ? 1.10 : 0.90;
                    const newSpan = span * zoomFactor;

                    const rect = chart.getBoundingClientRect();
                    let frac = (e.clientX - rect.left) / rect.width;
                    if (!isFinite(frac)) frac = 0.5;
                    frac = Math.max(0, Math.min(1, frac));
                    const cursorTime = xMin + span * frac;

                    let newMin = cursorTime - newSpan * frac;
                    let newMax = cursorTime + newSpan * (1 - frac);

                    const minA = ax.minallowed
                        ? new Date(ax.minallowed).getTime() : null;
                    const maxA = ax.maxallowed
                        ? new Date(ax.maxallowed).getTime() : null;
                    if (minA !== null && newMin < minA) newMin = minA;
                    if (maxA !== null && newMax > maxA) newMax = maxA;
                    if (newMax - newMin < 1000) return;

                    try {
                        window.parent.Plotly.relayout(chart, {
                            'xaxis.range': [
                                new Date(newMin), new Date(newMax)
                            ],
                        });
                    } catch (err) {}
                };
                // Attach in BOTH capture and bubble phase to maximize chances
                // of catching the event before/after Plotly's own handlers.
                chart.addEventListener('wheel', wheelHandler,
                    { passive: false, capture: true });

                // ============ Middle-click drag = pan on X axis ============
                // Use CAPTURE phase so we get the event before Plotly's zoom
                // dragmode handlers bubble it. Also use pointer events for
                // unified mouse/pen/touch handling.
                let panState = null;
                let panTicking = false;
                let lastMoveX = 0;

                const mdHandler = function(e) {
                    if (e.button !== 1) return;  // middle button only
                    e.preventDefault();
                    e.stopPropagation();
                    const layout = chart.layout || {};
                    const ax = layout.xaxis || {};
                    if (!ax.range) return;
                    panState = {
                        startX: e.clientX,
                        xMinStart: new Date(ax.range[0]).getTime(),
                        xMaxStart: new Date(ax.range[1]).getTime(),
                        rect: chart.getBoundingClientRect(),
                    };
                    lastMoveX = e.clientX;
                    chart.style.cursor = 'grabbing';
                };
                // capture: true → run before Plotly's own listeners
                chart.addEventListener('mousedown', mdHandler, true);
                chart.addEventListener('pointerdown', mdHandler, true);

                const mmHandler = function(e) {
                    if (!panState) return;
                    lastMoveX = e.clientX;
                    if (panTicking) return;
                    panTicking = true;
                    window.requestAnimationFrame(() => {
                        if (!panState) { panTicking = false; return; }
                        const ps = panState;
                        const dx = lastMoveX - ps.startX;
                        const span = ps.xMaxStart - ps.xMinStart;
                        const delta = -(dx / ps.rect.width) * span;
                        let newMin = ps.xMinStart + delta;
                        let newMax = ps.xMaxStart + delta;

                        const ax = (chart.layout || {}).xaxis || {};
                        const minA = ax.minallowed
                            ? new Date(ax.minallowed).getTime() : null;
                        const maxA = ax.maxallowed
                            ? new Date(ax.maxallowed).getTime() : null;
                        if (minA !== null && newMin < minA) {
                            newMax += (minA - newMin);
                            newMin = minA;
                        }
                        if (maxA !== null && newMax > maxA) {
                            newMin -= (newMax - maxA);
                            newMax = maxA;
                        }
                        try {
                            window.parent.Plotly.relayout(chart, {
                                'xaxis.range': [
                                    new Date(newMin), new Date(newMax)
                                ],
                            });
                        } catch (err) {}
                        panTicking = false;
                    });
                };
                pdoc.addEventListener('mousemove', mmHandler, true);

                const muHandler = function(e) {
                    if (panState && (e.button === 1 || e.button === undefined)) {
                        panState = null;
                        chart.style.cursor = '';
                    }
                };
                pdoc.addEventListener('mouseup', muHandler, true);
                pdoc.addEventListener('pointerup', muHandler, true);

                // Suppress browser autoscroll on middle click over the chart
                chart.addEventListener('auxclick', function(e) {
                    if (e.button === 1) e.preventDefault();
                }, true);
            }

            function attach() {
                pdoc.querySelectorAll('.js-plotly-plot').forEach(attachZoom);
            }
            [100, 300, 600, 1200, 2500, 4000].forEach(d => setTimeout(attach, d));
        })();
        </script>
        """,
        height=0,
    )


def _inject_price_tick_format():
    """Rewrite price y-axis tick labels:
       single-digit values keep $x.xx, $10+ drops decimals → $42, $123."""
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            const pdoc = window.parent.document;
            function format(text) {
                // Match $4.50, $42.00, $123.45, etc.
                const m = text.match(/^\\$(-?\\d+(?:\\.\\d+)?)$/);
                if (!m) return text;
                const v = parseFloat(m[1]);
                if (Math.abs(v) >= 10) {
                    return '$' + Math.round(v).toLocaleString();
                }
                return '$' + v.toFixed(2);
            }
            function rewrite() {
                pdoc.querySelectorAll('.js-plotly-plot').forEach(chart => {
                    // The first y-axis tick layer is the Price panel
                    const yaxes = chart.querySelectorAll('g.yaxislayer-above .ytick text');
                    yaxes.forEach(t => {
                        const orig = t.textContent;
                        const fmt = format(orig);
                        if (fmt !== orig) t.textContent = fmt;
                    });
                });
            }
            // Run repeatedly so re-renders (zoom, pan) get reformatted
            setInterval(rewrite, 200);
        })();
        </script>
        """,
        height=0,
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


@st.cache_data(ttl=120, show_spinner=False)
def cached_quotes(tickers: tuple) -> dict:
    """Watchlist tile prices. 2 min TTL — refreshes often enough to feel live
    during market hours without burning excessive API quota."""
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

    # Manual refresh row — small button on the right that clears just the
    # watchlist's quote cache (not all caches) and reruns. Useful when the
    # user wants to force a fresh price pull without touching the bigger
    # screener/scan caches.
    _bar_l, _bar_r = st.columns([5, 1])
    with _bar_r:
        if st.button("🔄 Refresh prices",
                     key="wl_refresh_btn",
                     use_container_width=True,
                     help="Force a fresh fetch of all watchlist prices"):
            try:
                cached_quotes.clear()
            except Exception:
                # Fallback: nuke everything if the targeted clear fails
                st.cache_data.clear()
            st.session_state["_cache_cleared_at"] = datetime.now()
            st.rerun()

    quotes = cached_quotes(tickers)
    # Hydrate target-price map into session_state once per run
    targets = st.session_state.setdefault(
        "_target_prices", _load_target_prices()
    )

    # CSS to make tile buttons compact and tighten vertical spacing.
    # The `.wl-tile-anchor` marker (injected inside each tile below) is used
    # purely for CSS scoping — it has no visual effect — so these spacing
    # rules apply ONLY to watchlist tiles, not to other column layouts in
    # the app (Custom Patterns, Screener, etc).
    st.markdown(
        "<style>"
        ".wl-tile-anchor { display: none; }"
        # Ticker button styling
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
        # Compact number-input for the target-price box — shrink padding,
        # hide the +/- spinner buttons, smaller font so it fits in the tile.
        "div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] "
        "  div[data-testid='stNumberInput'] input {"
        "    font-size: 0.72rem !important;"
        "    padding: 2px 4px !important;"
        "    height: 22px !important;"
        "    text-align: center;"
        "    background: transparent !important;"
        "    color: #e5e7eb !important;"
        "    border: 1px solid #5a5b5e !important;"
        "    border-radius: 6px !important;"
        "}"
        "div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] "
        "  div[data-testid='stNumberInput'] input:focus {"
        "    border-color: #9ca3af !important;"
        "}"
        "div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] "
        "  div[data-testid='stNumberInput'] button {"
        "    display: none !important;"
        "}"
        # --- Tight vertical spacing INSIDE each watchlist tile column ---
        # Scoped by `:has(.wl-tile-anchor)` so only watchlist columns shrink;
        # other column-based layouts keep their normal spacing.
        "div[data-testid='stHorizontalBlock'] > div:has(.wl-tile-anchor) "
        "  div[data-testid='stVerticalBlock'] {"
        "    gap: 2px !important;"
        "}"
        "div[data-testid='stHorizontalBlock'] > div:has(.wl-tile-anchor) "
        "  div[data-testid='stMarkdownContainer'] p {"
        "    margin: 0 !important;"
        "}"
        "div[data-testid='stHorizontalBlock'] > div:has(.wl-tile-anchor) "
        "  div[data-testid='stNumberInput'] {"
        "    margin: 0 !important;"
        "}"
        "div[data-testid='stHorizontalBlock'] > div:has(.wl-tile-anchor) "
        "  div[data-testid='stElementContainer'] {"
        "    margin-bottom: 0 !important;"
        "}"
        # Tighten the horizontal row containing tiles (no top/bottom margin)
        # but give a generous `gap` between columns so tiles aren't packed
        # shoulder-to-shoulder.
        "div[data-testid='stHorizontalBlock']:has(.wl-tile-anchor) {"
        "    margin-bottom: 0 !important;"
        "    margin-top: 0 !important;"
        "    gap: 16px !important;"
        "}"
        # Tighten the divider <hr> between rows AND its wrapper container.
        # Streamlit wraps every element in stElementContainer + stMarkdown
        # which add their own margins — those margins were the big gap the
        # user saw highlighted between rows. Zero them out so only the thin
        # 1px line shows.
        ".wl-row-divider {"
        "    margin: 0 !important;"
        "    border: 0;"
        "    border-top: 1px solid #4a4b4e;"
        "}"
        "div[data-testid='stElementContainer']:has(.wl-row-divider) {"
        "    margin: 0 !important;"
        "    padding: 0 !important;"
        "}"
        "div[data-testid='stElementContainer']:has(.wl-row-divider) "
        "  div[data-testid='stMarkdownContainer'] {"
        "    margin: 0 !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    # Top border of the watchlist (sits above the first row of tiles)
    st.markdown(
        '<hr class="wl-row-divider">',
        unsafe_allow_html=True,
    )

    cols_per_row = 8
    total_rows = (len(tickers) + cols_per_row - 1) // cols_per_row
    for row_idx, row_start in enumerate(
        range(0, len(tickers), cols_per_row)
    ):
        row_tickers = tickers[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, t in enumerate(row_tickers):
            with cols[i]:
                # Hidden anchor — CSS uses `:has(.wl-tile-anchor)` to scope
                # the tight-spacing rules to watchlist tiles only.
                st.markdown(
                    '<div class="wl-tile-anchor"></div>',
                    unsafe_allow_html=True,
                )
                # 1. Ticker button (top, full width)
                cols[i].button(
                    t, key=f"tile_btn_{t}",
                    on_click=_on_tile_click, args=(t,),
                    use_container_width=True,
                )

                # 2. Below the ticker, a 2-column inner row:
                #      [ Target $ input ]  [ Price / day-change % ]
                # Target on the left, price + percent stacked on the right.
                inner_l, inner_r = st.columns([1, 1], gap="small")

                with inner_l:
                    t_upper = t.upper()
                    cur_tgt = targets.get(t_upper)
                    new_tgt = st.number_input(
                        f"Target for {t}",
                        min_value=0.0,
                        value=float(cur_tgt) if cur_tgt else None,
                        step=0.01,
                        format="%.2f",
                        key=f"tgt_{t}",
                        label_visibility="collapsed",
                        placeholder="🎯 $",
                    )
                    # Persist on change
                    if (new_tgt or 0) != (cur_tgt or 0):
                        _set_target_price(t, new_tgt)

                with inner_r:
                    q = quotes.get(t)
                    if not q:
                        st.markdown(
                            '<div style="text-align:center; color:#6b7280; '
                            'font-size:0.75rem; line-height:1.1;">—<br>—</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        chg = q["change_pct"]
                        color = "#16a34a" if chg >= 0 else "#dc2626"
                        arrow = "▲" if chg >= 0 else "▼"
                        sign = "+" if chg >= 0 else ""
                        st.markdown(
                            f'<div style="text-align:center; line-height:1.15;">'
                            f'<span style="font-size:0.9rem; font-weight:700; color:#f0f0f0;">'
                            f'${q["price"]:.2f}</span>'
                            f'<br>'
                            f'<span style="font-size:0.72rem; color:{color}; font-weight:600;">'
                            f'{arrow}{sign}{chg:.2f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # 3. 5-day mini-history (below the target/price row, full
                # width of the tile). Each closing price is color-coded
                # green/red vs the prior day. Day labels (Mon/Tue/etc.)
                # above each price.
                q = quotes.get(t)
                if q and q.get("closes_5d"):
                    history = q["closes_5d"]
                    # Pick a price-format string that fits the magnitude:
                    #   penny stocks → 3 decimals
                    #   single-dollar → 2 decimals
                    #   above $100 → 1 decimal (saves horizontal space)
                    max_px = max(item[1] for item in history)
                    if max_px < 1:
                        fmt = "{:.3f}"
                    elif max_px < 100:
                        fmt = "{:.2f}"
                    else:
                        fmt = "{:.1f}"
                    # Each item is (day_label, price, direction) for new
                    # cache entries, or (day_label, price) for old ones
                    # still living in st.cache_data from before this commit
                    # (cache TTL=120s, so they'll roll off shortly). Handle
                    # both shapes; for legacy 2-tuples we recompute direction
                    # inline against the prior item.
                    cells = []
                    prev_px = None
                    for item in history:
                        if len(item) == 3:
                            day_lbl, px, direction = item
                        elif len(item) == 2:
                            day_lbl, px = item
                            if prev_px is None:
                                direction = "flat"
                            elif px > prev_px:
                                direction = "up"
                            elif px < prev_px:
                                direction = "down"
                            else:
                                direction = "flat"
                        else:
                            continue
                        prev_px = px
                        cell_color = {
                            "up": "#16a34a",
                            "down": "#dc2626",
                            "flat": "#9ca3af",
                        }.get(direction, "#9ca3af")
                        cells.append(
                            f'<div style="flex:1; text-align:center; '
                            f'line-height:1.15; padding:1px 0;">'
                            f'<div style="font-size:0.65rem; '
                            f'color:#9ca3af; font-weight:500;">{day_lbl}</div>'
                            f'<div style="font-size:0.78rem; '
                            f'color:{cell_color}; font-weight:700;">'
                            f'{fmt.format(px)}</div>'
                            f'</div>'
                        )
                    st.markdown(
                        '<div style="display:flex; gap:2px; '
                        'margin-top:6px; padding:4px 2px; '
                        'border-top:1px dashed #4a4b4e;">'
                        + "".join(cells) + "</div>",
                        unsafe_allow_html=True,
                    )

        # Horizontal divider between every row (including after the last
        # row — that one acts as the bottom border of the watchlist).
        # Uses the .wl-row-divider class so its margin is tight, not the
        # browser-default 16px.
        st.markdown(
            '<hr class="wl-row-divider">',
            unsafe_allow_html=True,
        )


def _open_dialog_for(ticker: str):
    """Set the sticky dialog flag for a ticker and rerun."""
    st.session_state["_open_dialog_ticker"] = ticker
    st.rerun()


@st.dialog("📊 Quick Analysis", width="large")
def show_quick_analysis_dialog(ticker: str):
    """Modal popup with chart, signal, and key metrics for a ticker.
    Used by the Screener tab when a row is clicked."""
    interval = st.session_state.get("_interval", "1d")
    adx_filter = st.session_state.get("_adx_filter", False)
    stop_loss_pct = st.session_state.get("_stop_loss_pct")

    # Strategy + lookback selectors
    sel_l, sel_r = st.columns([3, 2])
    _strategy_keys = list(ss.STRATEGY_LABELS.keys())
    strategy = sel_l.selectbox(
        "Strategy",
        options=_strategy_keys,
        format_func=lambda k: ss.STRATEGY_LABELS[k],
        index=_strategy_keys.index(ss.DEFAULT_STRATEGY_KEY),
        key=f"dlg_strategy_{ticker}",
    )
    _period_options = ["6mo", "1y", "2y", "5y", "10y", "max"]
    _saved_period = st.session_state.get("_period", "max")
    period = sel_r.selectbox(
        "Lookback",
        _period_options,
        index=(_period_options.index(_saved_period)
               if _saved_period in _period_options
               else _period_options.index("max")),
        key=f"dlg_period_{ticker}",
    )
    dlg_indicators = st.multiselect(
        "Indicators",
        options=list(ss.INDICATOR_LABELS.keys()),
        default=list(ss.DEFAULT_INDICATORS),
        format_func=lambda k: ss.INDICATOR_LABELS[k],
        key=f"dlg_indicators_{ticker}",
    )

    # --- Add to watchlist (uses same handler as the sidebar add) ---
    _wl_now = st.session_state.get(
        "watchlist_input", ", ".join(ss.DEFAULT_WATCHLIST)
    )
    _wl_set = {p.strip().upper() for p in _wl_now.split(",") if p.strip()}
    _t_upper = ticker.strip().upper()
    if _t_upper in _wl_set:
        st.success(f"✅ {_t_upper} is in your watchlist")
    else:
        if st.button(f"➕ Add {_t_upper} to watchlist",
                     key=f"dlg_add_wl_{ticker}",
                     use_container_width=True):
            _add_ticker_to_watchlist(ticker)
            st.rerun()

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

    st.markdown(
        f'<div style="margin-bottom:6px;">'
        f'<span style="font-size:1.4rem; font-weight:700; color:#f0f0f0;">'
        f'{norm_ticker}</span> '
        f'<span style="font-size:0.9rem; color:#9ca3af;">'
        f'&nbsp;·&nbsp; {stats["trades"]} trades '
        f'&nbsp;·&nbsp; {stats["win_rate"]:.0%} win'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig = ss.build_chart_plotly(df, norm_ticker, stats, compact=True,
                                indicators=dlg_indicators,
                                theme_dark=_is_dark_theme())
    st.plotly_chart(fig, use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "scrollZoom": False,
                        "doubleClick": False,
                        "modeBarButtonsToRemove": [
                            "select2d", "lasso2d",
                        ],
                    })
    _inject_double_click_fullscreen()
    _inject_auto_rescale_y()
    _inject_scroll_to_pan()
    _inject_price_tick_format()
    st.caption("💡 **Double-click chart for fullscreen** · Esc to exit")

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


def _cached_anomaly(df):
    """Compute anomaly score for a ticker df, cached on the latest bar's
    timestamp + bar count so we don't retrain IsolationForest on every rerun.

    Defined at module top-level so render_quick_analysis (which runs early in
    the script) can use it. Also reused later by the Custom Patterns tab.
    """
    if df is None or df.empty:
        return None
    sig = (str(df.index[-1]), len(df))
    cache = st.session_state.setdefault("__anomaly_cache", {})
    if sig in cache:
        return cache[sig]
    result = ss.compute_anomaly_score(df)
    cache[sig] = result
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def cached_sentiment(ticker: str):
    """Cache Finnhub news-sentiment for 30 minutes."""
    return ss.finnhub_sentiment(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_stocktwits(ticker: str):
    """Cache StockTwits sentiment for 15 minutes."""
    return ss.stocktwits_sentiment(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_market_regime():
    """Cache market regime classification for 30 minutes."""
    return ss.compute_market_regime()


def _cached_vol_outlook(ticker: str, df):
    """Cache volume outlook per (ticker, last bar) — uses news + earnings
    APIs so we don't recompute on every rerun."""
    if df is None or df.empty or not ticker:
        return None
    sig = (ticker.upper(), str(df.index[-1]), len(df))
    cache = st.session_state.setdefault("__vol_outlook_cache", {})
    if sig in cache:
        return cache[sig]
    result = ss.compute_volume_outlook(ticker, df)
    cache[sig] = result
    return result


def render_quick_analysis():
    """Inline analysis panel shown when a watchlist tile is clicked."""
    selected = st.session_state.get("selected_tile")
    if not selected:
        return
    interval = st.session_state.get("_interval", "1d")
    adx_filter = st.session_state.get("_adx_filter", False)
    stop_loss_pct = st.session_state.get("_stop_loss_pct")

    # Inline popup uses near-full page width for maximum chart room
    _l, popup_col, _r = st.columns([0.05, 12, 0.05])
    with popup_col, st.container(border=True):
        # Header row: title + strategy + lookback + close
        h1, h2, h3, h4 = st.columns([3, 2, 1.5, 1])
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
        _period_options = ["6mo", "1y", "2y", "5y", "10y", "max"]
        _saved_period = st.session_state.get("_period", "max")
        period = h3.selectbox(
            "Lookback",
            _period_options,
            index=(_period_options.index(_saved_period)
                   if _saved_period in _period_options
                   else _period_options.index("max")),
            key=f"qv_period_{selected}",
            label_visibility="collapsed",
        )
        if h4.button(":red[**✖**] Close", key="close_quick_view",
                     use_container_width=True):
            st.session_state.pop("selected_tile", None)
            st.rerun()

        # Indicator multiselect (full row below header)
        indicators = st.multiselect(
            "Indicators",
            options=list(ss.INDICATOR_LABELS.keys()),
            default=list(ss.DEFAULT_INDICATORS),
            format_func=lambda k: ss.INDICATOR_LABELS[k],
            key=f"qv_indicators_{selected}",
            label_visibility="collapsed",
        )

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

        # Single condensed stats line above the chart
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

        # === Info panel: company + price context + technicals + analyst ===
        prof = cached_company_profile(ticker)
        rec = cached_recommendation(ticker)
        anom_data = _cached_anomaly(df)

        # Header line: company name + sector
        if prof.get("name"):
            sector_bits = []
            if prof.get("sector"): sector_bits.append(prof["sector"])
            if prof.get("industry"): sector_bits.append(prof["industry"])
            if prof.get("country"): sector_bits.append(prof["country"])
            sector_str = " · ".join(sector_bits) if sector_bits else ""
            st.markdown(
                f"<div style='padding:4px 0 8px;'>"
                f"<span style='font-size:1.05rem; color:#e5e7eb; "
                f"font-weight:600;'>{prof['name']}</span>"
                + (f"<br><span style='font-size:0.8rem; color:#9ca3af;'>"
                   f"{sector_str}</span>" if sector_str else "")
                + "</div>",
                unsafe_allow_html=True,
            )

        # === Single-line dense info bar (horizontally scrolls if too narrow) ===
        cur = float(last["Close"])
        wk_hi = prof.get("week52_high"); wk_lo = prof.get("week52_low")
        avg_vol = prof.get("avg_vol")
        today_vol = float(last.get("Volume", 0)) if "Volume" in last else 0
        try:
            metrics_y = ss.yf_metrics(ticker)
            earn_days = metrics_y.get("earn_days")
        except Exception:
            earn_days = None
        rsi_v = (float(last["RSI"])
                 if "RSI" in last and pd.notna(last["RSI"]) else None)
        mh = (float(last.get("MACD_HIST", 0))
              if "MACD_HIST" in last and pd.notna(last.get("MACD_HIST"))
              else None)
        pctb = None
        if {"BB_LOWER", "BB_UPPER"}.issubset(df.columns):
            bb_rng = float(last["BB_UPPER"]) - float(last["BB_LOWER"])
            if bb_rng > 0:
                pctb = (cur - float(last["BB_LOWER"])) / bb_rng
        anom_pctile = anom_data.get("pctile") if anom_data else None
        pe = prof.get("pe") or prof.get("pe_forward")
        yld = prof.get("yield_pct")
        if yld is not None:
            yld = yld * 100 if yld < 1 else yld
        beta = prof.get("beta")

        # Build each chip
        def chip(label, value, color="#e5e7eb", subtle=None):
            sub_html = (f' <span style="color:#9ca3af; '
                        f'font-size:0.7rem;">{subtle}</span>'
                        if subtle else "")
            return (
                f'<span style="display:inline-flex; flex-direction:column; '
                f'padding:4px 10px; margin-right:6px; border-right:'
                f'1px solid #4a4b4e;">'
                f'<span style="font-size:0.65rem; color:#9ca3af; '
                f'text-transform:uppercase; letter-spacing:0.4px;">'
                f'{label}</span>'
                f'<span style="font-size:0.95rem; color:{color}; '
                f'font-weight:600;">{value}{sub_html}</span></span>'
            )

        chips: list[str] = []
        # Market cap
        chips.append(chip("M.Cap",
                          _fmt_compact_num(prof.get("market_cap"))))
        # 52w range
        if wk_hi and wk_lo and wk_hi > wk_lo:
            pct_in_range = (cur - wk_lo) / (wk_hi - wk_lo) * 100
            chips.append(chip(
                "52w",
                f"${wk_lo:.2f}–${wk_hi:.2f}",
                subtle=f"{pct_in_range:.0f}%",
            ))
        else:
            chips.append(chip("52w", "—"))
        # Volume
        if avg_vol:
            ratio = today_vol / avg_vol if avg_vol else 1
            vc = "#22c55e" if ratio > 1 else "#9ca3af"
            chips.append(chip("Vol",
                              _fmt_compact_num(today_vol),
                              color=vc,
                              subtle=f"{ratio:.2f}×"))
        else:
            chips.append(chip("Vol", _fmt_compact_num(today_vol)))
        # Next earnings
        chips.append(chip("Earn",
                          f"{earn_days}d" if earn_days is not None else "—"))
        # RSI
        if rsi_v is not None:
            rc = ("#22c55e" if rsi_v < 30
                  else "#ef4444" if rsi_v > 70 else "#e5e7eb")
            tag = ("OS" if rsi_v < 30 else "OB" if rsi_v > 70 else "")
            chips.append(chip("RSI", f"{rsi_v:.1f}", color=rc,
                              subtle=tag if tag else None))
        else:
            chips.append(chip("RSI", "—"))
        # MACD hist
        if mh is not None:
            mc = "#22c55e" if mh > 0 else "#ef4444"
            chips.append(chip("MACD", f"{mh:+.3f}", color=mc))
        else:
            chips.append(chip("MACD", "—"))
        # BB %B
        if pctb is not None:
            bc = ("#22c55e" if pctb < 0.2
                  else "#ef4444" if pctb > 0.8 else "#e5e7eb")
            chips.append(chip("BB%B", f"{pctb:.2f}", color=bc))
        else:
            chips.append(chip("BB%B", "—"))
        # Anomaly
        if anom_pctile is not None:
            ac = ("#ef4444" if anom_pctile < 10
                  else "#fbbf24" if anom_pctile < 25 else "#9ca3af")
            chips.append(chip("🤖 Anom", f"{anom_pctile:.0f}%ile", color=ac))
        else:
            chips.append(chip("🤖 Anom", "—"))
        # Volume outlook (next-week volume forecast)
        vol_out = _cached_vol_outlook(ticker, df)
        if vol_out:
            vc = {"high": "#ef4444", "elevated": "#fbbf24",
                  "normal": "#9ca3af", "quiet": "#60a5fa"}.get(
                vol_out["label"], "#9ca3af")
            chips.append(chip("🔮 VolOut",
                              f"{vol_out['score']:.0f}",
                              color=vc,
                              subtle=vol_out["label"]))
        else:
            chips.append(chip("🔮 VolOut", "—"))
        # StockTwits retail sentiment
        st_data = cached_stocktwits(ticker)
        if st_data:
            blp = st_data.get("bullish_pct")
            buzz = st_data.get("msg_count_24h", 0)
            if blp is not None:
                stc = ("#22c55e" if blp >= 0.6
                       else "#ef4444" if blp <= 0.4 else "#9ca3af")
                chips.append(chip("💬 ST",
                                  f"{blp*100:.0f}%🐂",
                                  color=stc,
                                  subtle=f"{buzz} msg/24h"))
            else:
                chips.append(chip("💬 ST",
                                  f"{buzz} msg",
                                  subtle="no tags"))
        else:
            chips.append(chip("💬 ST", "—"))
        # P/E, Yield, Beta
        chips.append(chip("P/E", f"{pe:.1f}" if pe else "—"))
        chips.append(chip("Yield", f"{yld:.2f}%" if yld is not None else "—"))
        chips.append(chip("Beta", f"{beta:.2f}" if beta else "—"))
        # Analysts
        if rec:
            b, h, s = rec
            total = b + h + s
            if total > 0:
                buy_pct = b / total * 100
                ac2 = ("#22c55e" if b > s
                       else "#ef4444" if s > b else "#9ca3af")
                chips.append(chip(
                    "Anlst",
                    f"{b}B·{h}H·{s}S",
                    color=ac2,
                    subtle=f"{buy_pct:.0f}%B",
                ))
            else:
                chips.append(chip("Anlst", "—"))
        else:
            chips.append(chip("Anlst", "—"))

        st.markdown(
            '<div style="display:flex; overflow-x:auto; '
            'white-space:nowrap; padding:8px 4px; '
            'background:rgba(96,165,250,0.03); border-radius:8px; '
            'border:1px solid #4a4b4e; margin-bottom:8px;">'
            + "".join(chips)
            + '</div>',
            unsafe_allow_html=True,
        )


        # Build chart
        fig = ss.build_chart_plotly(df, ticker, stats, compact=True,
                                    indicators=indicators,
                                    theme_dark=_is_dark_theme())
        st.plotly_chart(fig, use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "scrollZoom": False,
                            "doubleClick": False,
                            "modeBarButtonsToRemove": [
                                "select2d", "lasso2d",
                            ],
                        })
        _inject_double_click_fullscreen()
        _inject_auto_rescale_y()
        _inject_scroll_to_pan()
        _inject_price_tick_format()
        st.caption(
            "💡 **Left-drag** pans · **scroll** zooms · "
            "switch to box-zoom via the modebar · "
            "**double-click chart for fullscreen** (Esc to exit) · "
            "📰 news + fundamentals → **Single Ticker** tab"
        )

        # --- Tabs: ticker-specific news + business summary ---
        # Default 30-day window; user can widen via dropdown inside the tab.
        days_key = f"qv_news_days_{selected}"
        news_days = st.session_state.get(days_key, 30)
        news, news_source = cached_news_combined(ticker, days=news_days)
        news_label = (f"📰 News ({len(news)})" if news else "📰 News")
        about_label = "ℹ️ About"
        info_tabs = st.tabs([news_label, about_label])

        with info_tabs[0]:
            # Days-back selector + count summary
            ctl_l, ctl_r = st.columns([1, 4])
            picked_days = ctl_l.selectbox(
                "Window",
                options=[7, 14, 30, 60, 90],
                index=[7, 14, 30, 60, 90].index(news_days)
                       if news_days in [7, 14, 30, 60, 90] else 2,
                key=days_key,
                format_func=lambda d: f"{d} days",
                label_visibility="collapsed",
            )
            source_badge = {
                "finnhub": "<span style='color:#9ca3af; "
                           "font-size:0.75rem;'>via Finnhub</span>",
                "yahoo":   "<span style='color:#a855f7; "
                           "font-size:0.75rem;'>via Yahoo Finance "
                           "(Finnhub had nothing)</span>",
                "none":    "",
            }.get(news_source, "")
            if news:
                ctl_r.markdown(
                    f"<div style='padding-top:6px;'>"
                    f"Showing <b>{len(news)} articles</b> for "
                    f"<b>{ticker}</b> in the last {picked_days} days "
                    f"&nbsp;·&nbsp; {source_badge}</div>",
                    unsafe_allow_html=True,
                )
            else:
                ctl_r.caption(
                    f"_No news for {ticker} in the last {picked_days} days "
                    "from either Finnhub or Yahoo Finance. Try a wider "
                    "window._"
                )
            if news:
                # Time formatting: relative for recent, date for older
                now_ts = datetime.now()

                def _rel_time(seconds_ago: float) -> str:
                    if seconds_ago < 60:
                        return "just now"
                    if seconds_ago < 3600:
                        return f"{int(seconds_ago/60)}m ago"
                    if seconds_ago < 86400:
                        return f"{int(seconds_ago/3600)}h ago"
                    if seconds_ago < 86400 * 2:
                        return "yesterday"
                    if seconds_ago < 86400 * 7:
                        return f"{int(seconds_ago/86400)}d ago"
                    return None  # fall back to date

                # Color-code sources for variety
                def _src_color(src: str) -> str:
                    palette = ["#60a5fa", "#a78bfa", "#f472b6",
                               "#fbbf24", "#34d399", "#fb923c"]
                    return palette[abs(hash(src)) % len(palette)] if src else "#9ca3af"

                for art in news[:100]:
                    raw_ts = art.get("datetime", 0)
                    try:
                        dt = datetime.fromtimestamp(raw_ts)
                        sec_ago = (now_ts - dt).total_seconds()
                        rel = _rel_time(sec_ago)
                        when = rel if rel else dt.strftime("%b %d, %Y")
                        full_when = dt.strftime("%b %d, %Y · %H:%M")
                    except (ValueError, TypeError, OSError):
                        when = "?"
                        full_when = ""

                    src = (art.get("source") or "").strip()
                    head = (art.get("headline") or "").strip()
                    summary = (art.get("summary") or "").strip()
                    url = art.get("url", "#")
                    img = art.get("image", "")
                    src_color = _src_color(src)

                    img_html = ""
                    if img and img.startswith("http"):
                        img_html = (
                            f'<div style="flex:0 0 110px; height:80px; '
                            f'border-radius:8px; overflow:hidden; '
                            f'background:#3a3b3e;">'
                            f'<img src="{img}" style="width:100%; '
                            f'height:100%; object-fit:cover;" '
                            f'onerror="this.style.display=\'none\'"/></div>'
                        )

                    summary_html = ""
                    if summary:
                        s = (summary[:220] + "…") if len(summary) > 220 else summary
                        summary_html = (
                            f'<div style="font-size:0.82rem; '
                            f'color:#9ca3af; line-height:1.5; '
                            f'margin-top:4px;">{s}</div>'
                        )

                    # Single-line HTML so Streamlit's markdown parser
                    # doesn't mistake indented lines for a code block.
                    card = (
                        f'<a href="{url}" target="_blank" '
                        f'style="text-decoration:none; color:inherit;">'
                        f'<div style="display:flex; gap:14px; '
                        f'padding:12px 14px; margin-bottom:10px; '
                        f'border-radius:10px; '
                        f'background:rgba(96,165,250,0.04); '
                        f'border:1px solid #4a4b4e; '
                        f'transition:all 0.15s ease; cursor:pointer;" '
                        f'onmouseover="this.style.borderColor=\'#60a5fa\';'
                        f'this.style.background=\'rgba(96,165,250,0.10)\';" '
                        f'onmouseout="this.style.borderColor=\'#4a4b4e\';'
                        f'this.style.background=\'rgba(96,165,250,0.04)\';">'
                        f'{img_html}'
                        f'<div style="flex:1; min-width:0;">'
                        f'<div style="display:flex; align-items:center; '
                        f'gap:8px; margin-bottom:6px; font-size:0.72rem;">'
                        f'<span style="background:{src_color}; '
                        f'color:#0a0a0a; padding:2px 8px; border-radius:6px; '
                        f'font-weight:700; text-transform:uppercase; '
                        f'letter-spacing:0.3px;">{src}</span>'
                        f'<span style="color:#9ca3af;" '
                        f'title="{full_when}">⏱ {when}</span>'
                        f'</div>'
                        f'<div style="font-size:0.95rem; color:#f0f0f0; '
                        f'font-weight:600; line-height:1.4;">{head}</div>'
                        f'{summary_html}'
                        f'</div>'
                        f'</div>'
                        f'</a>'
                    )
                    st.markdown(card, unsafe_allow_html=True)

        with info_tabs[1]:
            if prof.get("summary"):
                st.write(prof["summary"])
            else:
                st.caption("_No business summary available._")
            if prof.get("website"):
                st.markdown(f"🔗 [Website]({prof['website']})")


def plt_close_cleanup(fig):
    """Close the matplotlib figure to free memory between reruns."""
    try:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def cached_scan(tickers: tuple, period: str, interval: str,
                strategy: str, adx_filter: bool,
                stop_loss_pct: float | None) -> list:
    return ss.scan(list(tickers), period, interval,
                   strategy=strategy, adx_filter=adx_filter,
                   stop_loss_pct=stop_loss_pct,
                   metrics_fn=cached_metrics)


@st.cache_data(ttl=300, show_spinner=False)
def cached_top_movers(tickers: tuple, window_days: int) -> list:
    """Fetch batched OHLC for all tickers, compute window-day return.
    Returns list of dicts: {Ticker, Price, Return %, Avg Daily %}.
    """
    if not tickers:
        return []
    # Fetch enough days for the window + buffer for weekends/holidays
    period = f"{max(window_days * 3 + 14, 45)}d"
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    rows: list[dict] = []
    for batch in batches:
        try:
            df = yf.download(
                " ".join(batch), period=period, interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
                threads=True,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t not in df.columns.get_level_values(0):
                        continue
                    td = df[t]["Close"].dropna()
                else:
                    td = df["Close"].dropna()
                if len(td) < window_days + 1:
                    continue
                entry = float(td.iloc[-window_days - 1])
                exit_close = float(td.iloc[-1])
                if entry <= 0 or not pd.notna(entry) or not pd.notna(exit_close):
                    continue
                ret_pct = (exit_close - entry) / entry * 100.0
                # Per-day avg return (geometric)
                daily_avg = ((exit_close / entry) ** (1 / window_days)
                             - 1) * 100
                rows.append({
                    "Ticker": t,
                    "Price": round(exit_close, 2),
                    "Return %": round(ret_pct, 2),
                    "Avg Daily %": round(daily_avg, 2),
                })
            except Exception:
                continue
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def cached_inception_trend(tickers: tuple) -> list:
    """For each ticker, fit a linear regression on log(price) vs time over
    the FULL price history (period='max') and return trend-quality metrics.

    A "forever uptrend" ticker has:
      • positive slope (compounding annual growth rate > 0)
      • high R² (price stayed close to its trendline — few big detours)
      • limited max drawdown (the uptrend wasn't a 90%-crash-and-back ride)

    Returns list of dicts:
      {Ticker, Price, Years, Total Return %, CAGR %, R², Max DD %, Score}
    where Score = CAGR × R² × (1 - |max_dd|/200), so bigger = steadier uptrend.
    """
    import numpy as np
    if not tickers:
        return []
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    rows: list[dict] = []
    for batch in batches:
        try:
            df = yf.download(
                " ".join(batch), period="max", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
                threads=True,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t not in df.columns.get_level_values(0):
                        continue
                    td = df[t]["Close"].dropna()
                else:
                    td = df["Close"].dropna()
                # Need at least ~6 months of history to compute a meaningful
                # long-term trend; under that, the regression is noise.
                if len(td) < 126:
                    continue
                first_price = float(td.iloc[0])
                last_price = float(td.iloc[-1])
                if first_price <= 0 or last_price <= 0:
                    continue
                days = (td.index[-1] - td.index[0]).days
                years = max(days / 365.25, 0.5)
                total_return = (last_price / first_price - 1) * 100
                # CAGR — geometric annualized return
                cagr = (((last_price / first_price) ** (1 / years)) - 1) * 100
                # Linear regression on log(price) vs bar-index. Log makes
                # exponential growth (like compounding) look linear, so R²
                # reflects "consistency of compounding" not raw shape.
                log_p = np.log(td.values.astype(float))
                x = np.arange(len(log_p), dtype=float)
                if np.std(x) == 0 or np.std(log_p) == 0:
                    continue
                slope, intercept = np.polyfit(x, log_p, 1)
                yhat = slope * x + intercept
                ss_res = float(np.sum((log_p - yhat) ** 2))
                ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                # Skip pure noise / flat-line / declining names — we only
                # want uptrends.
                if slope <= 0 or cagr <= 0:
                    continue
                # Max drawdown over the whole history
                cummax = np.maximum.accumulate(td.values)
                drawdown = (td.values - cummax) / cummax
                max_dd = float(drawdown.min()) * 100
                # Composite score — penalize big drawdowns and reward both
                # high CAGR and high R².
                score = cagr * max(r2, 0) * (1 - min(abs(max_dd) / 200, 0.95))
                # Current RSI(14) — useful for finding compounders that are
                # currently oversold (dip-buying setup) or overbought (riding
                # momentum). Falls back to None if too few bars.
                try:
                    rsi_now = float(ss.rsi(td, 14).iloc[-1])
                    if not np.isfinite(rsi_now):
                        rsi_now = None
                except Exception:
                    rsi_now = None
                rows.append({
                    "Ticker": t,
                    "Price": round(last_price, 2),
                    "Years": round(years, 1),
                    "Total Return %": round(total_return, 1),
                    "CAGR %": round(cagr, 1),
                    "R²": round(r2, 3),
                    "Max DD %": round(max_dd, 1),
                    "RSI": (round(rsi_now, 1)
                            if rsi_now is not None else None),
                    "Score": round(score, 2),
                })
            except Exception:
                continue
    return rows


@st.cache_data(ttl=600, show_spinner=False)
def cached_rally_scan(tickers: tuple) -> list:
    """Scan for tickers in a 'rally setup' — oversold-but-turning,
    volume building, momentum curling up. Composite Rally Score (0–100)
    combines six signals:

      1. RSI in the sweet spot (30–55) and rising
      2. MACD histogram positive or curling up from negative
      3. Volume expansion (recent 5d avg vs prior 30d avg)
      4. Price above 200-SMA (long-term uptrend intact)
      5. Price below 20-SMA but inside Bollinger band (room to run up)
      6. Bollinger Band squeeze (low volatility = potential breakout)

    Each component scored 0–100 then averaged for the final score.
    """
    import numpy as np
    if not tickers:
        return []
    # Need ~1 year of bars for 200-SMA + 30d volume baseline + BB-width MA
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    rows: list[dict] = []
    for batch in batches:
        try:
            df = yf.download(
                " ".join(batch), period="1y", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
                threads=True,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t not in df.columns.get_level_values(0):
                        continue
                    sub = df[t].dropna(subset=["Close"])
                else:
                    sub = df.dropna(subset=["Close"])
                if len(sub) < 60:
                    continue
                closes = sub["Close"]
                volumes = sub.get("Volume", pd.Series(dtype=float))
                price = float(closes.iloc[-1])
                if price <= 0:
                    continue

                # --- RSI signal ---
                rsi_series = ss.rsi(closes, 14)
                rsi_now = float(rsi_series.iloc[-1])
                rsi_5_ago = float(rsi_series.iloc[-6]) if len(
                    rsi_series) >= 6 else rsi_now
                rsi_change = rsi_now - rsi_5_ago
                # Sweet spot 30-55 + rising. Score peaks at RSI 40 rising +5
                if 30 <= rsi_now <= 55:
                    rsi_score = 100 - abs(rsi_now - 42) * 3
                elif rsi_now < 30:
                    # Deep oversold — could rally hard, but riskier
                    rsi_score = 60 + (30 - rsi_now) * 1.5
                else:
                    # Above 55 — already running, less "setup" potential
                    rsi_score = max(0, 100 - (rsi_now - 55) * 4)
                # Boost for rising RSI (momentum confirmation)
                rsi_score += min(15, max(-15, rsi_change * 2))
                rsi_score = max(0, min(100, rsi_score))

                # --- MACD signal ---
                macd_line, signal_line, hist = ss.macd(closes)
                macd_hist_now = float(hist.iloc[-1])
                macd_hist_prev = float(hist.iloc[-6]) if len(
                    hist) >= 6 else macd_hist_now
                hist_change = macd_hist_now - macd_hist_prev
                # Score: histogram positive AND rising is ideal
                if macd_hist_now > 0 and hist_change > 0:
                    macd_score = 80 + min(20, hist_change * 100)
                elif macd_hist_now < 0 and hist_change > 0:
                    # Turning up from negative — early-stage reversal
                    macd_score = 50 + min(30, hist_change * 100)
                elif macd_hist_now > 0 and hist_change < 0:
                    # Positive but rolling over
                    macd_score = 30
                else:
                    # Negative and falling
                    macd_score = max(0, 20 + hist_change * 100)
                macd_score = max(0, min(100, macd_score))

                # --- Volume signal ---
                if len(volumes) >= 35 and volumes.tail(35).sum() > 0:
                    recent_vol = float(volumes.tail(5).mean())
                    baseline_vol = float(volumes.iloc[-35:-5].mean())
                    if baseline_vol > 0:
                        vol_ratio = recent_vol / baseline_vol
                    else:
                        vol_ratio = 1.0
                else:
                    vol_ratio = 1.0
                # Score: 1.0 = neutral (50), 2.0+ = strong accumulation (100)
                vol_score = max(0, min(100, (vol_ratio - 0.5) * 60))

                # --- SMA trend signals ---
                sma20 = float(closes.rolling(20).mean().iloc[-1])
                sma200 = (float(closes.rolling(200).mean().iloc[-1])
                          if len(closes) >= 200 else None)
                above_200 = (sma200 is not None and price > sma200)
                # Score: in uptrend (above 200) AND pulled back near 20-SMA
                if sma200 is None:
                    sma_score = 50  # Not enough history — neutral
                elif above_200:
                    # In uptrend; reward pullbacks (price ≤ 20-SMA gets full)
                    if price <= sma20:
                        sma_score = 100
                    else:
                        # Already above 20-SMA — still OK, fading
                        sma_score = max(40, 100 - (price / sma20 - 1) * 500)
                else:
                    # Below 200 = not in confirmed uptrend
                    sma_score = 20

                # --- Bollinger Band squeeze signal ---
                bb_mid, bb_up, bb_lo = ss.bollinger(closes, 20, 2.0)
                bb_width = (bb_up - bb_lo) / bb_mid
                bb_width_now = float(bb_width.iloc[-1])
                bb_width_avg = float(bb_width.tail(60).mean()) if len(
                    bb_width) >= 60 else bb_width_now
                if bb_width_avg > 0:
                    squeeze_ratio = bb_width_now / bb_width_avg
                else:
                    squeeze_ratio = 1.0
                # Tight bands (low ratio) = squeeze = high score
                squeeze_score = max(0, min(100, (1.5 - squeeze_ratio) * 100))

                # --- Composite Rally Score ---
                rally_score = (
                    rsi_score * 0.25 +
                    macd_score * 0.20 +
                    vol_score * 0.15 +
                    sma_score * 0.20 +
                    squeeze_score * 0.20
                )

                rows.append({
                    "Ticker": t,
                    "Price": round(price, 2),
                    "Rally Score": round(rally_score, 1),
                    "RSI": round(rsi_now, 1),
                    "RSI Δ5d": round(rsi_change, 1),
                    "MACD Hist": round(macd_hist_now, 3),
                    "Vol Ratio": round(vol_ratio, 2),
                    "vs 20-SMA %": round((price / sma20 - 1) * 100, 1),
                    "vs 200-SMA %": (
                        round((price / sma200 - 1) * 100, 1)
                        if sma200 else None),
                    "BB Squeeze": round(squeeze_ratio, 2),
                    "Above 200": bool(above_200),
                })
            except Exception:
                continue
    return rows


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


@st.cache_data(ttl=900, show_spinner=False)
def cached_yf_news(ticker: str) -> list:
    """Yahoo Finance news (free, no key) — better TSX coverage than Finnhub."""
    return ss.yf_news(ticker)


def cached_news_combined(ticker: str, days: int = 30) -> tuple[list, str]:
    """Finnhub first; if empty, fall back to Yahoo. Returns (articles, source).
    `source` is "finnhub", "yahoo", or "none"."""
    fh = cached_news(ticker, days=days)
    if fh:
        return fh, "finnhub"
    yf_items = cached_yf_news(ticker)
    # Filter to the requested days window
    if yf_items:
        cutoff = datetime.now().timestamp() - (days * 86400)
        yf_items = [x for x in yf_items
                    if (x.get("datetime", 0) or 0) >= cutoff]
        if yf_items:
            return yf_items, "yahoo"
    return [], "none"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_profile(ticker: str) -> dict:
    return ss.yf_company_profile(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_recommendation(ticker: str):
    return ss.finnhub_recommendation(ticker)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r, g, b' string (for rgba CSS)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"
    except ValueError:
        return "156, 163, 175"  # fallback gray


def _fmt_compact_num(v: float | None) -> str:
    """Format big numbers compactly: 1234567 → 1.23M, 1234567890 → 1.23B."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (ValueError, TypeError):
        return "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e12: return f"{sign}{av/1e12:.2f}T"
    if av >= 1e9:  return f"{sign}{av/1e9:.2f}B"
    if av >= 1e6:  return f"{sign}{av/1e6:.2f}M"
    if av >= 1e3:  return f"{sign}{av/1e3:.1f}K"
    return f"{sign}{av:.2f}"


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


def _render_save_url_banner():
    """Show a 'bookmark this URL' prompt + copy button when the watchlist
    has been customized. Reliable cross-browser fallback to localStorage."""
    if not st.session_state.get("_wl_from_url", False):
        return
    parts = [p.strip().upper() for p in
             st.session_state.get("watchlist_input", "").split(",")
             if p.strip()]
    if not parts:
        return
    wl_param = ",".join(parts)
    # Inject JS that builds the absolute save-URL from the current page +
    # provides a copy-to-clipboard button.
    import streamlit.components.v1 as components
    js_param = json.dumps(wl_param)
    components.html(
        f"""<div style='font-family:"Comic Sans MS","Comic Sans",cursive;
             padding:6px 10px; background:#1f2937; color:#e5e7eb;
             border-radius:6px; font-size:0.78rem; margin:4px 0;
             display:flex; align-items:center; gap:8px;'>
            <span>💾 <b>Save your watchlist:</b>
              <span style='color:#9ca3af;'>bookmark this URL or click copy →</span>
            </span>
            <button id='wl-copy-btn'
                style='background:#3b82f6; color:#fff; border:none;
                       padding:3px 12px; border-radius:5px; cursor:pointer;
                       font-family:inherit; font-size:0.78rem;'>
              📋 Copy save URL
            </button>
            <span id='wl-copy-status' style='color:#22c55e;
                 font-size:0.75rem;'></span>
          </div>
          <script>
          (function() {{
            const btn = document.getElementById('wl-copy-btn');
            const status = document.getElementById('wl-copy-status');
            if (!btn) return;
            btn.addEventListener('click', () => {{
              try {{
                const u = new URL(window.parent.location.href);
                u.searchParams.set('wl', {js_param});
                navigator.clipboard.writeText(u.toString()).then(() => {{
                  status.textContent = '✓ copied!';
                  setTimeout(() => status.textContent = '', 2500);
                }});
              }} catch (e) {{
                status.textContent = 'copy failed';
              }}
            }});
          }})();
          </script>""",
        height=46,
    )


_render_save_url_banner()
render_quick_analysis()

# Top-right "data refreshed at" chip — shows the current page-render time in
# Eastern Time (TSX / NYSE / NASDAQ clock) so it matches the market clock the
# user is watching, not the Streamlit Cloud server's UTC clock. A small
# "(data Xm old)" suffix shows how long since the cache was actually cleared,
# so the user can tell whether the prices shown are freshly pulled or cached.
try:
    from zoneinfo import ZoneInfo
    _et_now = datetime.now(ZoneInfo("America/New_York"))
except Exception:
    # zoneinfo not available — fall back to fixed UTC-4 (EDT). Will drift by
    # 1 hr during EST (Nov–Mar) but better than showing UTC.
    from datetime import timezone, timedelta
    _et_now = datetime.now(timezone(timedelta(hours=-4)))
_now_str = _et_now.strftime("%H:%M:%S")
_now_date = _et_now.strftime("%b %d")
_now_tz = _et_now.strftime("%Z") or "ET"
_cleared = st.session_state.get("_cache_cleared_at")
_age_str = ""
if _cleared:
    # _cleared is timezone-naive (from datetime.now() on server). Compute age
    # using server-local time on both sides to avoid tz-mixing errors.
    _age_seconds = int((datetime.now() - _cleared).total_seconds())
    if _age_seconds < 60:
        _age_str = f"(data {_age_seconds}s old)"
    elif _age_seconds < 3600:
        _age_str = f"(data {_age_seconds // 60}m old)"
    else:
        _age_str = f"(data {_age_seconds // 3600}h old)"
st.markdown(
    f'<div style="text-align:right; font-size:0.78rem; color:#9ca3af; '
    f'margin-top:-4px; margin-bottom:4px;">'
    f'🕒 Tickers refreshed: '
    f'<b style="color:#e5e7eb;">{_now_str} {_now_tz}</b>'
    f'<span style="color:#6b7280;"> &nbsp;·&nbsp; {_now_date}</span>'
    f'<span style="color:#6b7280;"> &nbsp;{_age_str}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

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

def _add_ticker_to_watchlist(new_t: str) -> None:
    """Shared logic — add a normalized ticker to the watchlist.

    Auto-appends `-USD` to bare crypto bases (BTC → BTC-USD, ETH → ETH-USD)
    so users don't need to know yfinance's crypto naming convention.
    """
    new_t = (new_t or "").strip().upper()
    if not new_t:
        return
    # Auto-suffix bare crypto bases to yfinance format
    if "-" not in new_t and "." not in new_t and new_t in CRYPTO_BASES:
        new_t = f"{new_t}-USD"
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
        # User has explicitly modified — claim ownership so URL/localStorage
        # mirror this watchlist instead of treating it as default.
        st.session_state["_wl_from_url"] = True
        _sync_watchlist_to_url()


def _add_to_watchlist():
    """on_click handler — add the typed ticker to the watchlist text area."""
    _add_ticker_to_watchlist(st.session_state.get("add_ticker_input", ""))
    st.session_state.add_ticker_input = ""


def _add_from_dropdown():
    """on_change handler — add the dropdown selection."""
    selected = st.session_state.get("add_dropdown_select", "")
    if selected:
        _add_ticker_to_watchlist(selected)
    st.session_state.add_dropdown_select = ""


def _view_from_search_dropdown():
    """on_change handler — open chart popup for selected ticker."""
    selected = st.session_state.get("sidebar_view_select", "")
    if selected:
        st.session_state["selected_tile"] = selected
    st.session_state.sidebar_view_select = ""


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
    st.session_state["_wl_from_url"] = True
    _sync_watchlist_to_url()


@st.cache_data(ttl=86400, show_spinner=False)
def _all_tickers_for_dropdown() -> list:
    """Combined sorted list of S&P 500 + TSX Composite + popular ETFs + crypto."""
    parts = []
    try:
        parts.extend(ss.get_sp500())
    except Exception:
        pass
    try:
        parts.extend(ss.get_tsx_composite())
    except Exception:
        pass
    parts.extend(ss.UNIVERSE_POPULAR_ETFS)
    # Crypto: include both `BTC-USD` (yfinance form) and `BTC` (bare) so
    # users searching for "BTC" find a hit; the add-handler auto-appends
    # `-USD` to bare crypto bases.
    parts.extend(ss.UNIVERSE_CRYPTO)
    return sorted(set(parts))


# Set of known crypto bases (without -USD suffix) — used to auto-append
# the -USD suffix when a user types just "BTC".
CRYPTO_BASES: set[str] = {
    t.split("-")[0].upper() for t in ss.UNIVERSE_CRYPTO if "-" in t
}


def _build_ai_context() -> str:
    """Compact context the AI can ground on: watchlist, last prices, strategy."""
    parts = []
    wl = st.session_state.get("watchlist_input", "")
    if wl:
        parts.append(f"Watchlist tickers: {wl}")
    quotes = st.session_state.get("_wl_quotes_for_ai")
    if quotes:
        snap = ", ".join(
            f"{t} ${q['price']:.2f} ({q['change_pct']:+.2f}%)"
            for t, q in quotes.items() if q.get("price") is not None
        )
        if snap:
            parts.append(f"Latest quotes: {snap}")
    strat = st.session_state.get("_strategy")
    if strat:
        parts.append(f"Active strategy: {strat}")
    saved = st.session_state.get("saved_rules") or {}
    if saved:
        parts.append(f"Saved rule sets: {', '.join(saved.keys())}")
    return "\n".join(parts) if parts else "(no extra context)"


def _anthropic_key() -> str:
    val = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if val:
        return val
    try:
        return str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except Exception:
        return ""


def _call_claude(history: list[dict], user_msg: str) -> str:
    api_key = _anthropic_key()
    if not api_key:
        return ("⚠️ Set `ANTHROPIC_API_KEY` in `.env` "
                "(or Streamlit Cloud secrets) to enable the chat.")
    try:
        import anthropic  # lazy import so missing pkg doesn't break the app
    except ImportError:
        return ("⚠️ `anthropic` package not installed. "
                "Run `pip install anthropic` (it's in requirements.txt).")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        ctx = _build_ai_context()
        system_prompt = (
            "You are a stock-analysis assistant inside a Streamlit dashboard. "
            "Be factual, concise, and refuse to give investment advice or "
            "specific buy/sell recommendations. When uncertain, say so. "
            "Use the user's watchlist and current quotes when relevant.\n\n"
            f"User context:\n{ctx}"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=history + [{"role": "user", "content": user_msg}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"⚠️ AI request failed: {e}"


# Pre-fetch quotes once (cached) so the AI context has live data on first ask
if _wl_normalized:
    st.session_state["_wl_quotes_for_ai"] = cached_quotes(
        tuple(_wl_normalized[:20])
    ) or {}


with st.sidebar:
    with st.expander("💬 Talk to me", expanded=False):
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []

        # Show last few turns (compact)
        for msg in st.session_state.ai_chat_history[-8:]:
            who = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f"<div style='font-size:0.85rem; padding:4px 0;'>"
                f"<b>{who}</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )

        with st.form(key="ai_chat_form", clear_on_submit=True):
            user_q = st.text_area(
                "Ask anything",
                key="ai_chat_input",
                height=70,
                placeholder="e.g. Which of my watchlist tickers look oversold?",
                label_visibility="collapsed",
            )
            ask_c, clear_c = st.columns([3, 1])
            asked = ask_c.form_submit_button(
                "Send", use_container_width=True, type="primary"
            )
            cleared = clear_c.form_submit_button(
                "Clear", use_container_width=True
            )

        if asked and user_q.strip():
            with st.spinner("Thinking…"):
                reply = _call_claude(
                    st.session_state.ai_chat_history, user_q.strip()
                )
            st.session_state.ai_chat_history.append(
                {"role": "user", "content": user_q.strip()}
            )
            st.session_state.ai_chat_history.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()
        if cleared:
            st.session_state.ai_chat_history = []
            st.rerun()


with st.sidebar:
    st.header("Watchlist")

    # --- Active list selector (categories) ---
    _all_lists = st.session_state.get("_all_watchlists", {})
    if not _all_lists:
        _all_lists = _load_watchlists()
        st.session_state["_all_watchlists"] = _all_lists
    _list_names = list(_all_lists.keys()) or ["Default"]
    _active_name = st.session_state.get("_active_watchlist", _list_names[0])
    if _active_name not in _list_names:
        _active_name = _list_names[0]
        st.session_state["_active_watchlist"] = _active_name

    sel_idx = _list_names.index(_active_name)
    picked = st.selectbox(
        "Active list",
        options=_list_names,
        index=sel_idx,
        key="_watchlist_selector",
        format_func=lambda n: f"📋 {n}  ({len(_all_lists.get(n, []))})",
        help="Switch which list of tickers is currently active. "
             "All app features (snapshot, charts, screeners) use this list.",
    )
    if picked != _active_name:
        _set_active_watchlist(picked)
        st.rerun()

    # --- Manage lists (rename, create, delete, duplicate) ---
    with st.expander("⚙️ Manage lists", expanded=False):
        # Create new list
        new_list_name = st.text_input(
            "New list name",
            placeholder="e.g. Crypto, Holdings, Penny Stocks",
            key="_new_list_input",
        )
        mc1, mc2 = st.columns(2)
        if mc1.button("➕ Create empty", key="_create_list_btn",
                      use_container_width=True,
                      disabled=not new_list_name.strip()):
            n = new_list_name.strip()
            if n in _all_lists:
                st.warning(f"'{n}' already exists.")
            else:
                _all_lists[n] = []
                _save_watchlists(_all_lists)
                st.session_state["_all_watchlists"] = _all_lists
                _set_active_watchlist(n)
                st.rerun()
        if mc2.button("📋 Duplicate active", key="_dup_list_btn",
                      use_container_width=True,
                      disabled=not new_list_name.strip()):
            n = new_list_name.strip()
            if n in _all_lists:
                st.warning(f"'{n}' already exists.")
            else:
                _all_lists[n] = list(_all_lists.get(_active_name, []))
                _save_watchlists(_all_lists)
                st.session_state["_all_watchlists"] = _all_lists
                _set_active_watchlist(n)
                st.rerun()

        st.divider()

        # Rename active
        rn_name = st.text_input(
            f"Rename '{_active_name}' to:",
            value=_active_name,
            key="_rename_list_input",
        )
        if st.button("✏️ Rename", key="_rename_list_btn",
                     use_container_width=True,
                     disabled=(rn_name.strip() == _active_name
                               or not rn_name.strip())):
            new_n = rn_name.strip()
            if new_n in _all_lists:
                st.warning(f"'{new_n}' already exists.")
            else:
                _all_lists[new_n] = _all_lists.pop(_active_name)
                _save_watchlists(_all_lists)
                st.session_state["_all_watchlists"] = _all_lists
                _set_active_watchlist(new_n)
                st.rerun()

        st.divider()

        # Delete active (only if more than 1 list)
        if len(_all_lists) > 1:
            if st.button(f"🗑️ Delete '{_active_name}'",
                         key="_delete_list_btn",
                         use_container_width=True,
                         type="secondary",
                         help="Permanently deletes this list. Cannot undo."):
                del _all_lists[_active_name]
                _save_watchlists(_all_lists)
                st.session_state["_all_watchlists"] = _all_lists
                # Switch to first remaining list
                next_name = list(_all_lists.keys())[0]
                _set_active_watchlist(next_name)
                st.rerun()
        else:
            st.caption(
                "_Need at least one list — create another before deleting._"
            )

    st.divider()

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

    # Searchable dropdown — type to filter ~800 tickers from S&P 500 + TSX + ETFs.
    # Selecting one opens the chart popup (does NOT add to watchlist).
    st.selectbox(
        "🔍 Search & view ticker",
        options=[""] + _all_tickers_for_dropdown(),
        key="sidebar_view_select",
        on_change=_view_from_search_dropdown,
        help="Type any letters to filter; click a ticker to open its chart popup",
    )

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

    # Quick snapshot — live price + day change for each watchlist ticker
    st.subheader("Snapshot")
    if _wl_normalized:
        snapshot_quotes = cached_quotes(tuple(_wl_normalized[:20]))
        # Stash quotes for the AI sidebar chat so it can ground answers
        st.session_state["_wl_quotes_for_ai"] = snapshot_quotes or {}
        if snapshot_quotes:
            for t in _wl_normalized[:20]:
                q = snapshot_quotes.get(t)
                if not q:
                    continue
                chg = q["change_pct"]
                color = "#16a34a" if chg >= 0 else "#dc2626"
                arrow = "▲" if chg >= 0 else "▼"
                sign = "+" if chg >= 0 else ""
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'padding:4px 6px; margin:2px 0; border-radius:6px; '
                    f'background:#3a3b3e; font-size:0.85rem;">'
                    f'<span style="font-weight:600;">{t}</span>'
                    f'<span>${q["price"]:.2f} '
                    f'<span style="color:{color};">{arrow} {sign}{chg:.2f}%</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Loading prices…")
    else:
        st.caption("No tickers in watchlist.")

    # --- Recent news from watchlist ---
    if ss.FINNHUB_API_KEY and _wl_normalized:
        st.divider()
        with st.expander("📰 Recent news", expanded=False):
            news_items = []
            for t in _wl_normalized[:20]:
                arts = cached_news(t, days=3)
                for a in arts[:3]:
                    a = dict(a)
                    a["_ticker"] = t
                    news_items.append(a)
            news_items.sort(
                key=lambda x: x.get("datetime", 0) or 0, reverse=True
            )
            if news_items:
                for art in news_items[:15]:
                    try:
                        ts = datetime.fromtimestamp(art.get("datetime", 0))
                        when = ts.strftime("%b %d %H:%M")
                    except (ValueError, TypeError, OSError):
                        when = "?"
                    headline = (art.get("headline") or "")[:120]
                    url = art.get("url") or "#"
                    st.markdown(
                        f"<div style='font-size:0.8rem; padding:4px 0; "
                        f"border-bottom:1px solid #4a4b4e;'>"
                        f"<b style='color:#60a5fa;'>{art['_ticker']}</b> "
                        f"<span style='color:#9ca3af;'>· {when}</span><br>"
                        f"<a href='{url}' target='_blank' "
                        f"style='color:#e5e7eb; text-decoration:none;'>"
                        f"{headline}</a></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("_No recent news for watchlist tickers._")

    st.divider()

    # Sensible defaults — Settings UI removed. Tweak per-chart from the popup.
    period = "max"
    interval = "1d"
    st.session_state.setdefault("macro_view", "Both")
    strategy = ss.DEFAULT_STRATEGY_KEY
    adx_filter = False
    stop_loss_pct = None

    if st.button("🔄 Refresh data now"):
        st.cache_data.clear()
        st.session_state["_cache_cleared_at"] = datetime.now()
        st.rerun()
    _cleared = st.session_state.get("_cache_cleared_at")
    _cleared_str = _cleared.strftime("%H:%M:%S") if _cleared else "—"
    st.caption(
        f"Auto-refreshes on every new session and at midnight. "
        f"TTLs: macro/quotes ~3 min, scans ~5 min.\n\n"
        f"Data last refreshed: **{_cleared_str}**"
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

(tab_scan, tab_single, tab_screener, tab_patterns, tab_news,
 tab_help) = st.tabs(
    ["📊 Watchlist", "🔍 Single Ticker", "🎯 Screener",
     "🧩 Custom Patterns", "📰 News", "ℹ️ Help"]
)
# After the popup closes, restore the tab the user was on (if any)
_restore_active_tab()
# Persist tab clicks across reruns via localStorage so any rerun
# (widget interaction, range button click, etc.) doesn't reset to first tab.
_inject_tab_persistence()


# === Scan tab ===
with tab_scan:
    # === 📅 Last Session Movers — quick daily-glance section ===
    with st.expander(
        "📅 Last Session's Top Movers (TSX & more) — click to expand",
        expanded=False,
    ):
        st.caption(
            "Biggest gainers and losers from the most recent completed "
            "trading day. Updated every 15 min. Click any ticker to open "
            "the chart popup."
        )
        lsm_col1, lsm_col2 = st.columns([3, 1])
        lsm_universe_label = lsm_col1.selectbox(
            "Universe",
            options=[
                "TSX Composite (~250) — fast",
                "TSX 60 (~60) — fastest",
                "Entire TSX (~1500) — slower",
                "Entire TSX Venture (~1500) — slower",
                "S&P 100 (~100)",
                "S&P 500 (~500)",
                "Popular ETFs (~80)",
                "My watchlist",
            ],
            index=0,
            key="lsm_universe_label",
            label_visibility="collapsed",
        )
        lsm_run = lsm_col2.button(
            "🔍 Find movers", key="lsm_run",
            use_container_width=True, type="primary",
        )

        if lsm_run:
            universe_map = {
                "TSX Composite (~250) — fast": ("tsx_composite",
                                                ss.get_tsx_composite),
                "TSX 60 (~60) — fastest": ("tsx60",
                                            lambda: list(ss.UNIVERSE_TSX60)),
                "Entire TSX (~1500) — slower": ("tsx_full",
                                                lambda: ss.get_full_tsx_listing("tsx")),
                "Entire TSX Venture (~1500) — slower": ("tsxv",
                                                          lambda: ss.get_full_tsx_listing("tsxv")),
                "S&P 100 (~100)": ("sp100",
                                    lambda: list(ss.UNIVERSE_SP100)),
                "S&P 500 (~500)": ("sp500", ss.get_sp500),
                "Popular ETFs (~80)": ("etfs",
                                        lambda: list(ss.UNIVERSE_POPULAR_ETFS)),
                "My watchlist": ("wl",
                                  lambda: [
                                      t.strip().upper() for t in
                                      st.session_state.get(
                                          "watchlist_input", ""
                                      ).split(",") if t.strip()
                                  ]),
            }
            _, fetch_fn = universe_map.get(
                lsm_universe_label,
                ("tsx_composite", ss.get_tsx_composite),
            )
            with st.spinner(f"Loading universe…"):
                try:
                    tickers = fetch_fn()
                except Exception:
                    tickers = []
            if not tickers:
                st.warning("Universe is empty.")
            else:
                with st.spinner(
                    f"Scanning {len(tickers)} tickers for "
                    "last-session moves…"
                ):
                    rows = cached_top_movers(tuple(tickers), 1)
                st.session_state["lsm_results"] = {
                    "rows": rows,
                    "universe": lsm_universe_label,
                    "scanned": len(tickers),
                }

        lsm_res = st.session_state.get("lsm_results")
        if lsm_res and lsm_res["rows"]:
            df_lsm = pd.DataFrame(lsm_res["rows"])
            up = df_lsm.sort_values("Return %", ascending=False).head(10)
            down = df_lsm.sort_values("Return %").head(10)
            st.caption(
                f"📊 Scanned **{lsm_res['scanned']}** tickers in **"
                f"{lsm_res['universe']}** · "
                f"{len(lsm_res['rows'])} had valid data · "
                f"showing top 10 each direction"
            )

            def _quick_chips(d, color):
                chips = []
                for _, r in d.iterrows():
                    t = r["Ticker"]
                    ret = r["Return %"]
                    href = _chip_href(t, from_tab="Watchlist")
                    chips.append(
                        f"<a href='{href}' target='_self' "
                        f"style='background:{color}; color:#fff; "
                        "padding:4px 10px; border-radius:8px; "
                        "font-size:0.8rem; font-weight:700; "
                        "margin:3px; text-decoration:none; "
                        "display:inline-block;' "
                        f"title='Open {t}'>"
                        f"{t} {ret:+.1f}%</a>"
                    )
                st.markdown(
                    "<div style='padding:8px; border-radius:8px; "
                    f"background:rgba({_hex_to_rgb(color)},0.05); "
                    f"border:1px solid rgba({_hex_to_rgb(color)},0.25); "
                    "margin-bottom:8px;'>"
                    + "".join(chips) + "</div>",
                    unsafe_allow_html=True,
                )

            up_col, down_col = st.columns(2)
            with up_col:
                st.markdown(
                    "<b style='color:#22c55e;'>📈 Top 10 Gainers</b>",
                    unsafe_allow_html=True,
                )
                _quick_chips(up, "#16a34a")
            with down_col:
                st.markdown(
                    "<b style='color:#ef4444;'>📉 Top 10 Decliners</b>",
                    unsafe_allow_html=True,
                )
                _quick_chips(down, "#dc2626")

            st.caption(
                "💡 For longer windows (5d / 20d) + news context for "
                "each mover, see the full **Top Movers** section in "
                "the **Screener** tab."
            )
        elif lsm_res and not lsm_res["rows"]:
            st.info("No data returned — try a smaller universe.")

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
    with st.expander("📝 Edit / reorder watchlist (paste, clear, rearrange)",
                     expanded=False):
        st.text_area(
            "Comma-separated tickers — bare = US, .TO = TSX, .V = TSXV",
            default_str, height=68, key="watchlist_input",
            label_visibility="collapsed",
            # Sync to URL immediately on edit so reopening the tab
            # restores the latest watchlist
            on_change=_on_bulk_edit_watchlist,
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

                # Compact one-line stats — replaces the bulky st.metric grids
                stats_bits = [
                    f'<b style="color:#e5e7eb;">{stats["trades"]}</b> trades',
                    f'Win <b style="color:#e5e7eb;">{stats["win_rate"]:.0%}</b>',
                    f'Strat <b style="color:#e5e7eb;">{stats["total_return"]:+.1%}</b>',
                    f'B&H <b style="color:#e5e7eb;">{stats["buy_hold"]:+.1%}</b>',
                    f'Max DD <b style="color:#e5e7eb;">{stats.get("max_drawdown", 0):.1%}</b>',
                ]
                if stats.get("stops_hit", 0) > 0:
                    stats_bits.append(
                        f'<b style="color:#f59e0b;">{stats["stops_hit"]} stops</b>'
                    )

                metrics = ss.yf_metrics(ticker)
                if metrics:
                    if metrics.get("pe") is not None:
                        stats_bits.append(
                            f'P/E <b style="color:#e5e7eb;">{metrics["pe"]:.1f}</b>')
                    if metrics.get("yield_pct") is not None:
                        stats_bits.append(
                            f'Yield <b style="color:#e5e7eb;">{metrics["yield_pct"]:.2f}%</b>')
                    if metrics.get("beta") is not None:
                        stats_bits.append(
                            f'Beta <b style="color:#e5e7eb;">{metrics["beta"]:.2f}</b>')
                    if metrics.get("upside_pct") is not None:
                        stats_bits.append(
                            f'Upside <b style="color:#e5e7eb;">{metrics["upside_pct"]:+.1f}%</b>')
                    if metrics.get("earn_days") is not None:
                        stats_bits.append(
                            f'Earn <b style="color:#e5e7eb;">{metrics["earn_days"]}d</b>')

                st.markdown(
                    '<div style="font-size:0.9rem; color:#9ca3af; '
                    'padding:6px 0; line-height:1.7;">'
                    + ' &nbsp;·&nbsp; '.join(stats_bits)
                    + '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:1.4rem; font-weight:700; '
                    f'color:#f0f0f0;">{ticker}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                fig = ss.build_chart_plotly(df, ticker, stats,
                                            theme_dark=_is_dark_theme())
                st.plotly_chart(fig, use_container_width=True,
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "scrollZoom": False,
                                    "doubleClick": False,
                                    "modeBarButtonsToRemove": [
                                        "select2d", "lasso2d",
                                    ],
                                })
                _inject_double_click_fullscreen()
                _inject_auto_rescale_y()
                _inject_scroll_to_pan()
                _inject_price_tick_format()
                st.caption("💡 **Double-click chart for fullscreen** · Esc to exit")


# === Screener tab ===
with tab_screener:
    (sc_tab1, sc_tab2, sc_tab3, sc_tab4, sc_tab5,
     sc_tab6, sc_tab7) = st.tabs([
        "🎯 BUY Screener",
        "📊 Strategy Leaderboard",
        "🎯 Winning Tickers",
        "🏆 Top Movers",
        "🪨 TSXV Mining Catalysts",
        "📈 Forever Uptrend",
        "🚀 Rally Setup",
    ])

    with sc_tab1:
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
                "Entire US (~7000, no OTC)",
                "TSX 60 (~60)",
                "TSX Composite (~250)",
                "Entire TSX (~1500)",
                "Entire TSX Venture (~1500)",
                "Popular ETFs (~80)",
                "All US + TSX + ETFs (~850)",
                "Custom watchlist",
            ],
            index=0,
            key="screener_universe",
        )
        # Industry filter — overrides universe if picked
        industry_filter = sc_col1.selectbox(
            "🏷️ Industry filter (optional, overrides universe)",
            options=["(none — use universe above)"]
                    + [f"{lbl} ({len(tk)})"
                       for lbl, tk in ss.INDUSTRY_UNIVERSES.items()],
            index=0,
            key="screener_industry",
            help="Pick an industry to scan only those tickers. Overrides "
                 "the universe selection above.",
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

        @st.cache_data(ttl=86400, show_spinner=False)
        def _cached_full_tsx(market: str) -> list:
            return ss.get_full_tsx_listing(market)

        @st.cache_data(ttl=86400, show_spinner=False)
        def _cached_full_us() -> list:
            return ss.get_full_us_listing()

        # Industry filter takes priority over universe radio
        industry_chosen = None
        if industry_filter and not industry_filter.startswith("(none"):
            for ind_lbl, ind_tk in ss.INDUSTRY_UNIVERSES.items():
                if industry_filter.startswith(ind_lbl):
                    industry_chosen = ind_lbl
                    universe = list(ind_tk)
                    break

        if industry_chosen is None:
            if universe_choice.startswith("S&P 100"):
                universe = ss.UNIVERSE_SP100
            elif universe_choice.startswith("S&P 500"):
                universe = _cached_sp500()
            elif universe_choice.startswith("Entire US"):
                universe = _cached_full_us()
            elif universe_choice.startswith("TSX 60"):
                universe = ss.UNIVERSE_TSX60
            elif universe_choice.startswith("TSX Composite"):
                universe = _cached_tsx_composite()
            elif universe_choice.startswith("Entire TSX Venture"):
                universe = _cached_full_tsx("tsxv")
            elif universe_choice.startswith("Entire TSX"):
                universe = _cached_full_tsx("tsx")
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

            # Cap the results in a scrollable container so they don't dominate
            # the page. Wrapped in an expander so user can fully collapse to
            # see other screener sections below.
            with st.expander(
                f"📋 Results — {len(matches)} matches (click to collapse)",
                expanded=True,
            ):
                st.caption("💡 **Click any ticker** to open the chart in a popup.")

                # Sort controls
                SORT_KEYS = {
                    "Ticker (A→Z)":     ("ticker", False),
                    "Ticker (Z→A)":     ("ticker", True),
                    "Price ↑":          ("price", False),
                    "Price ↓":          ("price", True),
                    "RSI ↑ (oversold first)":  ("rsi", False),
                    "RSI ↓":            ("rsi", True),
                    "vs BB Lower ↑ (deepest below first)":
                        ("bb_distance_pct", False),
                    "vs BB Lower ↓":    ("bb_distance_pct", True),
                    "BB BUY Date — newest first": ("bb_buy_age", False),
                    "BB BUY Date — oldest first": ("bb_buy_age", True),
                    "Dip% ↑ (worst dip first)": ("dip_pct", False),
                    "Dip% ↓":           ("dip_pct", True),
                    "RSI OS — yes first":  ("rsi_oversold", True),
                    "RSI OS — no first":   ("rsi_oversold", False),
                }
                sort_col_l, sort_col_r = st.columns([2, 5])
                sort_choice = sort_col_l.selectbox(
                    "Sort",
                    options=list(SORT_KEYS.keys()),
                    index=8,  # default: BB BUY Date — newest first
                    key="screener_sort",
                    label_visibility="collapsed",
                )
                sort_field, sort_desc = SORT_KEYS[sort_choice]
                sort_col_r.caption(
                    f"Sorted by **{sort_choice}** · {len(matches)} rows"
                )

                def _sort_key(m: dict):
                    v = m.get(sort_field)
                    # None values sort last regardless of direction
                    if v is None:
                        return (1, 0)
                    # For string fields (ticker), use lowercase for stable sort
                    if isinstance(v, str):
                        return (0, v.lower())
                    return (0, v)

                matches_sorted = sorted(
                    matches, key=_sort_key, reverse=sort_desc
                )

                # Header row — added Dip% column
                col_widths = [1.2, 1.0, 0.7, 1.2, 1.4, 1.0, 0.8, 0.8]
                h = st.columns(col_widths)
                for col, label in zip(h, ["Ticker", "Price", "RSI", "vs BB Lower",
                                          "BB BUY Date", "Age", "Dip%", "RSI OS"]):
                    col.markdown(f"**{label}**")

                for m in matches_sorted:
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
    with sc_tab2:
        st.divider()
        st.subheader("📊 Strategy Performance Leaderboard")
        st.caption(
            "Backtest **every available strategy** against a universe of "
            "tickers, then rank by aggregate win rate / return. Lets you see "
            "which strategies have an edge right now — and which don't."
        )

        lb_col1, lb_col2 = st.columns([2, 1])
        lb_options = [
            "Custom watchlist",
            "TSX 60 (~60)",
            "Popular ETFs (~80)",
            "S&P 100 (~100)",
            "S&P 500 (~500 — slow)",
        ] + [
            f"{lbl} ({len(tk)})"
            for lbl, tk in ss.INDUSTRY_UNIVERSES.items()
        ]
        lb_universe = lb_col1.selectbox(
            "Universe (smaller = faster)",
            options=lb_options,
            index=0,
            key="lb_universe",
        )
        lb_min_trades = lb_col2.number_input(
            "Min trades to qualify",
            min_value=0, max_value=50, value=3, step=1,
            help="Filter out strategies that barely traded. 3 = each ticker "
                 "must have ≥3 trades to count.",
            key="lb_min_trades",
        )

        if st.button("🏁 Run leaderboard", key="lb_run", type="primary"):
            # Resolve universe
            with st.spinner("Loading universe…"):
                if lb_universe == "Custom watchlist":
                    tickers = [t.strip().upper() for t in
                               st.session_state.get("watchlist_input", "")
                                    .split(",") if t.strip()]
                elif lb_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                elif lb_universe == "Popular ETFs (~80)":
                    tickers = list(ss.UNIVERSE_POPULAR_ETFS)
                elif lb_universe == "S&P 100 (~100)":
                    tickers = list(ss.UNIVERSE_SP100)
                elif lb_universe == "S&P 500 (~500 — slow)":
                    tickers = ss.get_sp500()
                else:
                    # Try industry universes
                    tickers = []
                    for lbl, tk in ss.INDUSTRY_UNIVERSES.items():
                        if lb_universe.startswith(lbl):
                            tickers = list(tk)
                            break

            if not tickers:
                st.warning("Universe is empty.")
            else:
                n_strats = len(ss.STRATEGY_LABELS)
                leaderboard = []
                progress = st.progress(0.0)
                status = st.empty()
                for idx, (strat_key, strat_label) in enumerate(
                    ss.STRATEGY_LABELS.items()
                ):
                    status.caption(
                        f"Backtesting **{strat_label}** "
                        f"({idx + 1}/{n_strats})…"
                    )
                    try:
                        rows = cached_scan(
                            tuple(tickers), period, interval,
                            strat_key, adx_filter, stop_loss_pct,
                        )
                    except Exception:
                        rows = []
                    progress.progress((idx + 1) / n_strats)
                    if not rows:
                        continue
                    # Filter to tickers that actually traded
                    qualified = [r for r in rows
                                 if r.get("trades", 0) >= lb_min_trades]
                    if not qualified:
                        continue
                    wins = [r["win_rate"] for r in qualified
                            if r.get("win_rate") is not None]
                    rets = [r["total_return"] for r in qualified
                            if r.get("total_return") is not None]
                    bhs = [r["buy_hold"] for r in qualified
                           if r.get("buy_hold") is not None]
                    dds = [r["max_drawdown"] for r in qualified
                           if r.get("max_drawdown") is not None]
                    trades = sum(r.get("trades", 0) for r in qualified)
                    profitable = sum(
                        1 for r in qualified
                        if (r.get("total_return") or 0) > 0
                    )
                    alpha = [
                        (r.get("total_return") or 0) - (r.get("buy_hold") or 0)
                        for r in qualified
                    ]
                    leaderboard.append({
                        "Strategy": strat_label,
                        "_key": strat_key,
                        "Avg Win %":
                            round(sum(wins) / len(wins) * 100, 1)
                            if wins else None,
                        "Avg Return %":
                            round(sum(rets) / len(rets) * 100, 2)
                            if rets else None,
                        "Avg B&H %":
                            round(sum(bhs) / len(bhs) * 100, 2)
                            if bhs else None,
                        "Avg α (vs B&H) %":
                            round(sum(alpha) / len(alpha) * 100, 2)
                            if alpha else None,
                        "Avg Max DD %":
                            round(sum(dds) / len(dds) * 100, 2)
                            if dds else None,
                        "Total Trades": int(trades),
                        "Profitable / Total":
                            f"{profitable}/{len(qualified)}",
                        "Profitable %":
                            round(profitable / len(qualified) * 100, 1),
                        "Sample (tickers)": len(qualified),
                    })
                progress.empty()
                status.empty()

                if not leaderboard:
                    st.info("No strategy produced trades — try lowering "
                            "the minimum-trades threshold or pick a "
                            "different universe.")
                else:
                    st.session_state["lb_results"] = leaderboard
                    st.session_state["lb_universe_used"] = lb_universe

        # Render last results (survive popup/rerun)
        lb_data = st.session_state.get("lb_results")
        if lb_data:
            st.caption(
                f"Last run · universe: **"
                f"{st.session_state.get('lb_universe_used', '')}** · "
                f"{len(lb_data)} strategies ranked"
            )
            df_lb = pd.DataFrame(lb_data).drop(columns=["_key"])
            # Sort by alpha by default (strategy excess return over buy-hold)
            if "Avg α (vs B&H) %" in df_lb.columns:
                df_lb = df_lb.sort_values(
                    "Avg α (vs B&H) %", ascending=False, na_position="last"
                )
            st.dataframe(
                df_lb,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Win %": st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                    "Avg Return %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Avg B&H %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Avg α (vs B&H) %": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="Avg strategy return minus buy-and-hold. "
                             "Positive = strategy beats passive holding.",
                    ),
                    "Avg Max DD %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Profitable %": st.column_config.NumberColumn(
                        format="%.0f%%",
                        help="% of tickers where this strategy made money.",
                    ),
                },
            )

            # Honest takeaway based on top result
            try:
                top = df_lb.iloc[0]
                alpha = top.get("Avg α (vs B&H) %")
                win = top.get("Avg Win %")
                if alpha is not None and alpha > 5:
                    st.success(
                        f"**Top performer**: {top['Strategy']} — "
                        f"+{alpha:.1f}% alpha vs buy-hold with "
                        f"{win:.0f}% avg win rate."
                    )
                elif alpha is not None and alpha > 0:
                    st.info(
                        f"**Top performer**: {top['Strategy']} — "
                        f"{alpha:+.1f}% alpha. Modest edge."
                    )
                else:
                    st.warning(
                        f"**Top performer**: {top['Strategy']} — "
                        f"{alpha:+.1f}% alpha. **No strategy beat buy-and-"
                        f"hold in this universe**. Passive may be the "
                        f"honest call here."
                    )
            except (KeyError, IndexError, TypeError):
                pass
    with sc_tab3:
        st.divider()
        st.subheader("🎯 Winning Tickers (per-strategy)")
        st.caption(
            "For one strategy, find tickers where it has historically worked "
            "best. Ranks by win rate (default) or by total return / alpha. "
            "**Past performance doesn't predict future** — but it filters "
            "out tickers where the pattern just doesn't fit the security."
        )

        wt_col1, wt_col2 = st.columns([1, 1])
        wt_strat = wt_col1.selectbox(
            "Strategy",
            options=list(ss.STRATEGY_LABELS.keys()),
            format_func=lambda k: ss.STRATEGY_LABELS[k],
            index=list(ss.STRATEGY_LABELS.keys()).index(
                ss.DEFAULT_STRATEGY_KEY
            ),
            key="wt_strat",
        )
        wt_options = [
            "TSX Composite (~250)",
            "TSX 60 (~60)",
            "Entire TSX (~1500) — slow",
            "Entire TSX Venture (~1500) — slow",
            "S&P 100 (~100)",
            "S&P 500 (~500) — slow",
            "Popular ETFs (~80)",
            "My watchlist",
        ] + [
            f"{lbl} ({len(tk)})"
            for lbl, tk in ss.INDUSTRY_UNIVERSES.items()
        ]
        wt_universe = wt_col2.selectbox(
            "Universe",
            options=wt_options,
            index=0,
            key="wt_universe",
        )

        wt_col3, wt_col4, wt_col5 = st.columns([1, 1, 1])
        wt_sort_by = wt_col3.selectbox(
            "Sort by",
            options=["Win %", "Strategy %", "α (vs B&H) %", "Profit factor"],
            index=0,
            key="wt_sort_by",
        )
        wt_min_trades = wt_col4.number_input(
            "Min trades",
            min_value=1, max_value=100, value=5, step=1,
            help="Tickers with fewer trades are filtered (stats too noisy).",
            key="wt_min_trades",
        )
        wt_top_n = wt_col5.number_input(
            "Show top N",
            min_value=10, max_value=500, value=50, step=10,
            key="wt_top_n",
        )

        if st.button("🎯 Find winning tickers", key="wt_run",
                     type="primary"):
            with st.spinner("Loading universe…"):
                if wt_universe == "My watchlist":
                    tickers = [t.strip().upper() for t in
                               st.session_state.get("watchlist_input", "")
                                    .split(",") if t.strip()]
                elif wt_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                elif wt_universe == "TSX Composite (~250)":
                    tickers = ss.get_tsx_composite()
                elif wt_universe == "Entire TSX (~1500) — slow":
                    tickers = ss.get_full_tsx_listing("tsx")
                elif wt_universe == "Entire TSX Venture (~1500) — slow":
                    tickers = ss.get_full_tsx_listing("tsxv")
                elif wt_universe == "S&P 100 (~100)":
                    tickers = list(ss.UNIVERSE_SP100)
                elif wt_universe == "S&P 500 (~500) — slow":
                    tickers = ss.get_sp500()
                elif wt_universe == "Popular ETFs (~80)":
                    tickers = list(ss.UNIVERSE_POPULAR_ETFS)
                else:
                    # Try industry universes
                    tickers = []
                    for lbl, tk in ss.INDUSTRY_UNIVERSES.items():
                        if wt_universe.startswith(lbl):
                            tickers = list(tk)
                            break

            if not tickers:
                st.warning("Universe is empty.")
            else:
                with st.spinner(
                    f"Backtesting {ss.STRATEGY_LABELS[wt_strat]} on "
                    f"{len(tickers)} tickers…"
                ):
                    try:
                        rows = cached_scan(
                            tuple(tickers), period, interval,
                            wt_strat, adx_filter, stop_loss_pct,
                        )
                    except Exception:
                        rows = []
                if not rows:
                    st.warning("No backtest results. Try a different "
                               "universe.")
                else:
                    # Augment with profit factor + alpha
                    augmented = []
                    for r in rows:
                        if r.get("trades", 0) < int(wt_min_trades):
                            continue
                        win = r.get("win_rate", 0) or 0
                        ret = r.get("total_return", 0) or 0
                        bh = r.get("buy_hold", 0) or 0
                        dd = r.get("max_drawdown", 0) or 0
                        alpha = ret - bh
                        # Profit factor proxy: total_return positive vs max_dd
                        pf_proxy = (
                            abs(ret) / abs(dd) if dd != 0 else
                            (10.0 if ret > 0 else 0.0)
                        )
                        augmented.append({
                            "Ticker": r["ticker"],
                            "Win %": round(win * 100, 1),
                            "Strategy %": round(ret * 100, 2),
                            "B&H %": round(bh * 100, 2),
                            "α (vs B&H) %": round(alpha * 100, 2),
                            "Max DD %": round(dd * 100, 2),
                            "Profit factor": round(pf_proxy, 2),
                            "Trades": int(r.get("trades", 0) or 0),
                            "Close": round(r.get("close", 0) or 0, 2),
                        })
                    st.session_state["wt_results"] = {
                        "rows": augmented,
                        "strategy_label": ss.STRATEGY_LABELS[wt_strat],
                        "universe": wt_universe,
                        "sort_by": wt_sort_by,
                        "top_n": int(wt_top_n),
                        "scanned": len(tickers),
                    }

        wt_res = st.session_state.get("wt_results")
        if wt_res:
            rows = wt_res["rows"]
            if not rows:
                st.info(
                    f"No tickers passed the min-trades filter. Lower "
                    f"'Min trades' or pick a stricter strategy that fires "
                    "more often."
                )
            else:
                df_wt = pd.DataFrame(rows)
                sort_col = wt_res["sort_by"]
                if sort_col in df_wt.columns:
                    df_wt = df_wt.sort_values(
                        sort_col, ascending=False, na_position="last"
                    )
                df_wt = df_wt.head(wt_res["top_n"]).reset_index(drop=True)
                df_wt.insert(0, "Rank", range(1, len(df_wt) + 1))

                st.caption(
                    f"Strategy: **{wt_res['strategy_label']}** · "
                    f"Universe: **{wt_res['universe']}** · "
                    f"Scanned **{wt_res['scanned']}** tickers · "
                    f"{len(rows)} passed filter · "
                    f"showing top **{len(df_wt)}** by {sort_col}"
                )

                # Top tickers as clickable chips (top 30 by win rate)
                top_30 = df_wt.head(30)
                chips = []
                for _, r in top_30.iterrows():
                    t = r["Ticker"]
                    wr = r["Win %"]
                    # Color-code: green for high win %, gradient down
                    if wr >= 65:
                        color = "#16a34a"
                    elif wr >= 55:
                        color = "#65a30d"
                    elif wr >= 45:
                        color = "#a16207"
                    else:
                        color = "#9ca3af"
                    href = _chip_href(t, from_tab="Screener")
                    chips.append(
                        f"<a href='{href}' target='_self' style='"
                        f"background:{color}; color:#fff; "
                        "padding:3px 9px; border-radius:8px; "
                        "font-size:0.78rem; font-weight:700; "
                        "margin:3px; text-decoration:none; "
                        "display:inline-block;' "
                        f"title='Open {t} (Win {wr}%)'>"
                        f"{t} {wr:.0f}%</a>"
                    )
                st.markdown(
                    "<div style='padding:6px; margin-bottom:10px; "
                    "border-radius:8px; background:rgba(34,197,94,0.03); "
                    "border:1px solid rgba(34,197,94,0.2);'>"
                    "<b style='color:#9ca3af; margin-right:6px; "
                    "font-size:0.85rem;'>🎯 Top 30 by win rate "
                    "(click to open chart):</b>"
                    + "".join(chips) + "</div>",
                    unsafe_allow_html=True,
                )

                # Sortable detail table
                st.dataframe(
                    df_wt, use_container_width=True, hide_index=True,
                    column_config={
                        "Win %": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="% of trades that were winners.",
                        ),
                        "Strategy %": st.column_config.NumberColumn(
                            format="%+.2f%%"
                        ),
                        "B&H %": st.column_config.NumberColumn(
                            format="%+.2f%%"
                        ),
                        "α (vs B&H) %": st.column_config.NumberColumn(
                            format="%+.2f%%",
                            help="Strategy return minus buy-and-hold.",
                        ),
                        "Max DD %": st.column_config.NumberColumn(
                            format="%.2f%%"
                        ),
                        "Close": st.column_config.NumberColumn(
                            format="$%.2f"
                        ),
                    },
                )

                # Honest takeaway
                top_row = df_wt.iloc[0] if not df_wt.empty else None
                if top_row is not None:
                    wr = top_row["Win %"]
                    alpha = top_row["α (vs B&H) %"]
                    if wr >= 65 and alpha > 5:
                        st.success(
                            f"**Strong fit**: {top_row['Ticker']} has "
                            f"{wr:.0f}% win rate + {alpha:+.1f}% alpha "
                            f"using {wt_res['strategy_label']}. Real edge "
                            "on this name historically."
                        )
                    elif wr >= 55 and alpha > 0:
                        st.info(
                            f"**Modest fit**: {top_row['Ticker']} has "
                            f"{wr:.0f}% win rate, {alpha:+.1f}% alpha. "
                            "Better than random but not exceptional."
                        )
                    else:
                        st.warning(
                            f"**Weak fit**: even the best ticker has "
                            f"{wr:.0f}% win rate. This strategy may not "
                            f"work well on **{wt_res['universe']}** in "
                            "general — try a different strategy."
                        )
    with sc_tab4:
        st.divider()
        st.subheader("🏆 Top Movers")
        st.caption(
            "Find the biggest gainers (and losers) over a chosen window. "
            "Useful for spotting what's running in a sector — but mind that "
            "biggest gainers often mean-revert."
        )

        tm_col1, tm_col2, tm_col3 = st.columns([2, 1, 1])
        tm_options = [
            "Entire TSX (~1500)",
            "TSX Composite (~250)",
            "TSX 60 (~60)",
            "Entire TSX Venture (~1500)",
            "S&P 100 (~100)",
            "S&P 500 (~500 — slow)",
            "Entire US (~7000, very slow)",
            "Popular ETFs (~80)",
            "Custom watchlist",
        ] + [
            f"{lbl} ({len(tk)})"
            for lbl, tk in ss.INDUSTRY_UNIVERSES.items()
        ]
        tm_universe = tm_col1.selectbox(
            "Universe",
            options=tm_options,
            index=0,
            key="tm_universe",
        )
        tm_window = tm_col2.selectbox(
            "Window",
            options=[1, 5, 10, 20, 60],
            index=1,
            format_func=lambda d: f"{d} day" if d == 1 else f"{d} days",
            key="tm_window",
            help="Trading days, not calendar days.",
        )
        tm_topn = tm_col3.number_input(
            "Show top N",
            min_value=10, max_value=200, value=50, step=10,
            key="tm_topn",
        )

        if st.button("🏆 Find top movers", key="tm_run", type="primary"):
            with st.spinner("Loading universe…"):
                if tm_universe == "Custom watchlist":
                    tickers = [t.strip().upper() for t in
                               st.session_state.get("watchlist_input", "")
                                    .split(",") if t.strip()]
                elif tm_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                elif tm_universe == "TSX Composite (~250)":
                    tickers = ss.get_tsx_composite()
                elif tm_universe == "Entire TSX (~1500)":
                    tickers = ss.get_full_tsx_listing("tsx")
                elif tm_universe == "Entire TSX Venture (~1500)":
                    tickers = ss.get_full_tsx_listing("tsxv")
                elif tm_universe == "S&P 100 (~100)":
                    tickers = list(ss.UNIVERSE_SP100)
                elif tm_universe == "S&P 500 (~500 — slow)":
                    tickers = ss.get_sp500()
                elif tm_universe == "Entire US (~7000, very slow)":
                    tickers = ss.get_full_us_listing()
                elif tm_universe == "Popular ETFs (~80)":
                    tickers = list(ss.UNIVERSE_POPULAR_ETFS)
                else:
                    # Try industry universes
                    tickers = []
                    for lbl, tk in ss.INDUSTRY_UNIVERSES.items():
                        if tm_universe.startswith(lbl):
                            tickers = list(tk)
                            break

            if not tickers:
                st.warning("Universe is empty.")
            else:
                with st.spinner(
                    f"Scanning {len(tickers)} tickers over {tm_window}d…"
                ):
                    rows = cached_top_movers(tuple(tickers), int(tm_window))
                st.session_state["tm_results"] = {
                    "rows": rows,
                    "window": int(tm_window),
                    "universe": tm_universe,
                    "scanned": len(tickers),
                    "topn": int(tm_topn),
                }

        # Render results
        tm_res = st.session_state.get("tm_results")
        if tm_res:
            rows = tm_res["rows"]
            window = tm_res["window"]
            topn = tm_res["topn"]
            if not rows:
                st.info("No data returned — try a smaller universe.")
            else:
                df_tm = pd.DataFrame(rows)
                up = df_tm.sort_values("Return %", ascending=False).head(topn)
                down = df_tm.sort_values("Return %").head(topn)

                st.caption(
                    f"Scanned **{tm_res['scanned']} tickers** in "
                    f"**{tm_res['universe']}** over **{window} days**. "
                    f"{len(rows)} returned valid data."
                )

                up_tab, down_tab = st.tabs([
                    f"📈 Top {len(up)} Gainers",
                    f"📉 Top {len(down)} Decliners",
                ])

                def _render_movers(d, color: str):
                    # Clickable ticker chips
                    chips = []
                    for _, r in d.iterrows():
                        t = r["Ticker"]
                        ret = r["Return %"]
                        href = _chip_href(t, from_tab="Screener")
                        chips.append(
                            f"<a href='{href}' target='_self' "
                            f"style='background:{color}; color:#fff; "
                            "padding:3px 9px; border-radius:8px; "
                            "font-size:0.78rem; font-weight:700; "
                            "margin:3px; text-decoration:none; "
                            "display:inline-block;' "
                            f"title='Open {t} chart'>{t} "
                            f"{ret:+.1f}%</a>"
                        )
                    st.markdown(
                        "<div style='max-height:200px; overflow-y:auto; "
                        f"padding:6px; border-radius:8px; "
                        f"background:rgba({_hex_to_rgb(color)},0.04); "
                        f"border:1px solid rgba({_hex_to_rgb(color)},0.2); "
                        "margin-bottom:8px;'>"
                        + "".join(chips) + "</div>",
                        unsafe_allow_html=True,
                    )

                    # === What triggered the move? ===
                    # Fetch news for top 10 from the same window
                    st.markdown(
                        "<div style='font-size:0.85rem; color:#9ca3af; "
                        "margin-top:8px; margin-bottom:4px;'>"
                        "<b>🗞️ What triggered the move?</b> (news from the "
                        f"last {window + 2} days for top 10)</div>",
                        unsafe_allow_html=True,
                    )
                    news_lookback = window + 2
                    for _, r in d.head(10).iterrows():
                        t = r["Ticker"]
                        ret = r["Return %"]
                        href = _chip_href(t, from_tab="Screener")
                        # Fetch news (Finnhub → Yahoo fallback)
                        try:
                            news, src_kind = cached_news_combined(
                                t, days=news_lookback
                            )
                        except Exception:
                            news, src_kind = [], "none"
                        # Pick top 2 most relevant headlines (newest first)
                        top_news = news[:2] if news else []
                        news_html_parts = []
                        if top_news:
                            for art in top_news:
                                try:
                                    ts = datetime.fromtimestamp(
                                        art.get("datetime", 0)
                                    )
                                    when = ts.strftime("%b %d")
                                except (ValueError, TypeError, OSError):
                                    when = "?"
                                head = (art.get("headline") or "")[:140]
                                url = art.get("url", "#")
                                src_name = art.get("source", "")
                                news_html_parts.append(
                                    f"<div style='font-size:0.78rem; "
                                    f"color:#e5e7eb; padding:3px 0; "
                                    f"line-height:1.4;'>"
                                    f"<span style='color:#9ca3af; "
                                    f"min-width:48px; "
                                    f"display:inline-block;'>{when}</span>"
                                    f"<span style='color:#60a5fa; "
                                    f"font-size:0.7rem; "
                                    f"margin-right:6px;'>{src_name}</span>"
                                    f"<a href='{url}' target='_blank' "
                                    f"style='color:#e5e7eb; "
                                    f"text-decoration:none;'>{head}</a></div>"
                                )
                            news_block = "".join(news_html_parts)
                        else:
                            news_block = (
                                "<div style='font-size:0.78rem; "
                                "color:#9ca3af; padding:3px 0; "
                                "font-style:italic;'>"
                                "No news found in window — could be a "
                                "technical/sector move or coverage gap"
                                "</div>"
                            )
                        st.markdown(
                            f"<div style='padding:8px 12px; margin:6px 0; "
                            f"border-radius:8px; border-left:3px solid "
                            f"{color}; "
                            f"background:rgba({_hex_to_rgb(color)},0.03);'>"
                            f"<div style='display:flex; "
                            f"align-items:center; gap:10px; "
                            f"margin-bottom:4px;'>"
                            f"<a href='{href}' target='_self' "
                            f"style='background:{color}; color:#fff; "
                            "padding:3px 10px; border-radius:6px; "
                            "font-size:0.8rem; font-weight:700; "
                            "text-decoration:none;'>"
                            f"{t} {ret:+.1f}%</a>"
                            f"<span style='color:#9ca3af; "
                            f"font-size:0.75rem;'>"
                            f"${r['Price']:.2f}</span>"
                            f"</div>"
                            + news_block + "</div>",
                            unsafe_allow_html=True,
                        )

                    # Sortable detail table (all top N)
                    st.markdown(
                        "<div style='font-size:0.85rem; color:#9ca3af; "
                        "margin-top:10px;'><b>Full list</b> "
                        f"(top {len(d)})</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        d, use_container_width=True, hide_index=True,
                        column_config={
                            "Price": st.column_config.NumberColumn(
                                format="$%.2f"
                            ),
                            "Return %": st.column_config.NumberColumn(
                                format="%+.2f%%"
                            ),
                            "Avg Daily %": st.column_config.NumberColumn(
                                format="%+.2f%%",
                                help=("Geometric mean daily return over "
                                      "the window."),
                            ),
                        },
                    )

                with up_tab:
                    _render_movers(up, "#16a34a")
                with down_tab:
                    _render_movers(down, "#dc2626")
    with sc_tab5:
        st.divider()
        st.subheader("🪨 TSXV Mining Catalysts")
        st.caption(
            "Find TSXV stocks that moved + classify the news that drove it. "
            "Covers drill results, resource estimates, M&A, permits, "
            "feasibility studies, insider buying, financings. **NOT** a buy "
            "list — many catalysts are already priced in by the time you see "
            "the headline."
        )

        mc_col1, mc_col2, mc_col3 = st.columns([2, 1, 1])
        mc_universe = mc_col1.selectbox(
            "Universe",
            options=[
                "TSX Venture (~1500)",
                "Entire TSX (~1500)",
                "TSX Composite (~250)",
                "TSX 60 (~60)",
            ],
            index=0,
            key="mc_universe",
            help="TSX Venture has the most mining juniors. The mining catalysts "
                 "screener works on other Canadian universes too.",
        )
        mc_window = mc_col2.selectbox(
            "Window",
            options=[1, 3, 5, 10, 20],
            index=2,
            format_func=lambda d: f"{d} day" if d == 1 else f"{d} days",
            key="mc_window",
        )
        mc_min_move = mc_col3.number_input(
            "Min |% move|",
            min_value=0.0, max_value=200.0, value=5.0, step=1.0,
            key="mc_min_move",
            help="Only show stocks that moved at least this much (filters noise).",
        )

        if st.button("🪨 Find catalysts", key="mc_run", type="primary"):
            with st.spinner("Loading universe…"):
                if mc_universe == "TSX Venture (~1500)":
                    tickers = ss.get_full_tsx_listing("tsxv")
                elif mc_universe == "Entire TSX (~1500)":
                    tickers = ss.get_full_tsx_listing("tsx")
                elif mc_universe == "TSX Composite (~250)":
                    tickers = ss.get_tsx_composite()
                elif mc_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                else:
                    tickers = []

            if not tickers:
                st.warning("Universe is empty.")
            else:
                # Step 1: find movers in the window
                with st.spinner(
                    f"Step 1/2 · Scanning {len(tickers)} tickers for movers…"
                ):
                    rows = cached_top_movers(tuple(tickers), int(mc_window))
                # Step 2: filter to those that moved enough
                min_pct = float(mc_min_move)
                big_movers = [
                    r for r in rows
                    if abs(r.get("Return %", 0)) >= min_pct
                ]
                # Cap the news-fetch step so we don't hit rate limits
                big_movers.sort(key=lambda r: abs(r["Return %"]), reverse=True)
                max_news_fetches = 50
                to_classify = big_movers[:max_news_fetches]
                extra_movers = big_movers[max_news_fetches:]

                with st.spinner(
                    f"Step 2/2 · Classifying news for {len(to_classify)} "
                    "top movers…"
                ):
                    classified = []
                    progress = st.progress(0.0)
                    for i, r in enumerate(to_classify):
                        t = r["Ticker"]
                        try:
                            news, _src = cached_news_combined(
                                t, days=int(mc_window) + 3
                            )
                        except Exception:
                            news = []
                        catalyst = None
                        matched_article = None
                        for art in news[:10]:
                            cat = ss.classify_mining_news(art)
                            if cat:
                                catalyst = cat
                                matched_article = art
                                break
                        classified.append({
                            **r,
                            "catalyst": catalyst,
                            "article": matched_article,
                        })
                        progress.progress((i + 1) / len(to_classify))
                    progress.empty()

                st.session_state["mc_results"] = {
                    "classified": classified,
                    "extra": extra_movers,
                    "universe": mc_universe,
                    "window": int(mc_window),
                    "min_move": min_pct,
                    "scanned": len(tickers),
                }

        mc_res = st.session_state.get("mc_results")
        if mc_res:
            classified = mc_res["classified"]
            extras = mc_res.get("extra", [])
            window = mc_res["window"]

            if not classified:
                st.info(
                    f"No tickers moved ≥{mc_res['min_move']:.0f}% over "
                    f"{window} days. Try lowering the threshold."
                )
            else:
                st.caption(
                    f"Scanned **{mc_res['scanned']}** tickers in **"
                    f"{mc_res['universe']}** · {len(classified)} moved "
                    f"≥{mc_res['min_move']:.0f}% in {window}d "
                    + (f"(+{len(extras)} more not classified due to rate "
                       f"limits)" if extras else "")
                )

                # Group by catalyst
                with_catalyst = [c for c in classified if c["catalyst"]]
                without_catalyst = [c for c in classified if not c["catalyst"]]

                if with_catalyst:
                    st.markdown(
                        f"<div style='color:#22c55e; font-weight:700; "
                        f"margin-top:10px;'>🎯 Movers WITH identified "
                        f"catalysts: {len(with_catalyst)}</div>",
                        unsafe_allow_html=True,
                    )
                    # Sort by abs return descending
                    with_catalyst.sort(
                        key=lambda c: abs(c["Return %"]), reverse=True
                    )
                    for c in with_catalyst:
                        art = c.get("article") or {}
                        try:
                            ts = datetime.fromtimestamp(
                                art.get("datetime", 0)
                            )
                            when = ts.strftime("%b %d")
                        except (ValueError, TypeError, OSError):
                            when = "?"
                        head = (art.get("headline") or "")[:160]
                        url = art.get("url", "#")
                        src = art.get("source", "")
                        href = _chip_href(c["Ticker"], from_tab="Screener")
                        ret = c["Return %"]
                        ret_color = "#16a34a" if ret > 0 else "#dc2626"
                        st.markdown(
                            f"<div style='padding:10px 14px; margin:8px 0; "
                            f"border-radius:8px; border-left:4px solid "
                            f"#22c55e; background:rgba(34,197,94,0.04);'>"
                            f"<div style='display:flex; align-items:center; "
                            f"gap:10px; margin-bottom:6px; flex-wrap:wrap;'>"
                            f"<a href='{href}' target='_self' style='"
                            f"background:{ret_color}; color:#fff; "
                            f"padding:3px 10px; border-radius:6px; "
                            f"font-size:0.85rem; font-weight:700; "
                            f"text-decoration:none;'>"
                            f"{c['Ticker']} {ret:+.1f}%</a>"
                            f"<span style='background:#1e3a8a; color:#fff; "
                            f"padding:3px 10px; border-radius:6px; "
                            f"font-size:0.78rem; font-weight:700;'>"
                            f"{c['catalyst']}</span>"
                            f"<span style='color:#9ca3af; "
                            f"font-size:0.75rem;'>"
                            f"${c['Price']:.2f} · {window}d {ret:+.1f}%</span>"
                            f"</div>"
                            f"<div style='font-size:0.82rem; color:#e5e7eb; "
                            f"line-height:1.4;'>"
                            f"<a href='{url}' target='_blank' style='"
                            f"color:#e5e7eb; text-decoration:none;'>"
                            f"<span style='color:#9ca3af;'>{when} · "
                            f"{src}</span> — {head}</a></div></div>",
                            unsafe_allow_html=True,
                        )

                if without_catalyst:
                    with st.expander(
                        f"⚪ Movers WITHOUT identified catalyst: "
                        f"{len(without_catalyst)} (often technical or "
                        f"sector-driven moves; no public news tagged)",
                        expanded=False,
                    ):
                        chips = []
                        for c in without_catalyst[:30]:
                            t = c["Ticker"]
                            ret = c["Return %"]
                            color = "#16a34a" if ret > 0 else "#dc2626"
                            href = _chip_href(t, from_tab="Screener")
                            chips.append(
                                f"<a href='{href}' target='_self' style='"
                                f"background:{color}; color:#fff; "
                                f"padding:3px 9px; border-radius:8px; "
                                f"font-size:0.78rem; font-weight:700; "
                                f"margin:3px; text-decoration:none; "
                                f"display:inline-block;' "
                                f"title='Open {t}'>"
                                f"{t} {ret:+.1f}%</a>"
                            )
                        st.markdown(
                            "<div>" + "".join(chips) + "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "These moved but no public news matched the "
                            "catalyst keywords. Could be: TMX press release "
                            "not picked up by Finnhub/Yahoo, technical/sector "
                            "move, low-coverage ticker, or genuine momentum "
                            "without news."
                        )

    with sc_tab6:
        st.divider()
        st.subheader("📈 Forever Uptrend")
        st.caption(
            "Find tickers whose **entire price history** trends up. Fits a "
            "log-linear regression on every bar since the company started "
            "trading and ranks names with high CAGR, high R² (consistent "
            "compounding — not a one-time pump), and limited drawdowns."
        )

        ft_col1, ft_col2 = st.columns([2, 1])
        ft_options = [
            "S&P 100 (~100)",
            "S&P 500 (~500 — slow)",
            "TSX 60 (~60)",
            "TSX Composite (~250)",
            "Entire TSX (~1500, slow)",
            "Popular ETFs (~80)",
            "Custom watchlist",
        ] + [
            f"{lbl} ({len(tk)})"
            for lbl, tk in ss.INDUSTRY_UNIVERSES.items()
        ]
        ft_universe = ft_col1.selectbox(
            "Universe",
            options=ft_options,
            index=0,
            key="ft_universe",
        )
        ft_topn = ft_col2.number_input(
            "Show top N",
            min_value=10, max_value=200, value=50, step=10,
            key="ft_topn",
        )

        # Quality thresholds — sliders so the user can tune "what counts as
        # a real uptrend" vs. a marginal one.
        ft_col3, ft_col4, ft_col5 = st.columns([1, 1, 1])
        ft_min_cagr = ft_col3.slider(
            "Min CAGR %",
            min_value=0, max_value=50, value=10, step=1,
            key="ft_min_cagr",
            help="Compounding annual return floor. 10% ≈ S&P average.",
        )
        ft_min_r2 = ft_col4.slider(
            "Min R²",
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            key="ft_min_r2",
            help="How tightly the price hugs its trendline (1.0 = perfect "
                 "exponential growth, 0.5 = noisy uptrend).",
        )
        ft_min_years = ft_col5.slider(
            "Min years of history",
            min_value=1, max_value=20, value=3, step=1,
            key="ft_min_years",
            help="Skip recent IPOs — uptrends mean little with <3 yrs data.",
        )

        # Current-RSI filter — drag both handles to pick the RSI band you
        # want the long-term compounder to be in RIGHT NOW. Common uses:
        #   • 0–35   → forever-uptrenders currently oversold (dip-buy)
        #   • 30–70  → "normal" RSI — middle of the trend
        #   • 70–100 → already running hot, momentum chase
        # Default 0–100 = no filter.
        ft_rsi_range = st.slider(
            "Current RSI(14) filter",
            min_value=0, max_value=100, value=(0, 100), step=1,
            key="ft_rsi_range",
            help="Filter by today's RSI value. Set 0–35 to find forever-"
                 "uptrend names currently in an oversold dip; 70–100 to "
                 "see ones already running.",
        )

        if st.button("📈 Find forever uptrenders",
                     key="ft_run", type="primary"):
            with st.spinner("Loading universe…"):
                if ft_universe == "Custom watchlist":
                    tickers = [t.strip().upper() for t in
                               st.session_state.get("watchlist_input", "")
                                    .split(",") if t.strip()]
                elif ft_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                elif ft_universe == "TSX Composite (~250)":
                    tickers = ss.get_tsx_composite()
                elif ft_universe == "Entire TSX (~1500, slow)":
                    tickers = ss.get_full_tsx_listing("tsx")
                elif ft_universe == "S&P 100 (~100)":
                    tickers = list(ss.UNIVERSE_SP100)
                elif ft_universe == "S&P 500 (~500 — slow)":
                    tickers = ss.get_sp500()
                elif ft_universe == "Popular ETFs (~80)":
                    tickers = list(ss.UNIVERSE_POPULAR_ETFS)
                else:
                    # Industry universes
                    tickers = []
                    for lbl, tk in ss.INDUSTRY_UNIVERSES.items():
                        if ft_universe.startswith(lbl):
                            tickers = list(tk)
                            break

            if not tickers:
                st.warning("Universe is empty.")
            else:
                with st.spinner(
                    f"Fetching full history for {len(tickers)} tickers… "
                    "(this may take a few minutes for big universes)"
                ):
                    rows = cached_inception_trend(tuple(tickers))
                st.session_state["ft_results"] = {
                    "rows": rows,
                    "universe": ft_universe,
                    "scanned": len(tickers),
                }

        # Render results
        ft_res = st.session_state.get("ft_results")
        if ft_res:
            rows = ft_res["rows"]
            if not rows:
                st.info("No tickers passed the long-term-uptrend filter.")
            else:
                # Apply user filters. RSI range is only enforced when the
                # user has narrowed it from the default 0–100 (so we don't
                # accidentally drop names whose RSI we couldn't compute).
                rsi_lo, rsi_hi = ft_rsi_range
                rsi_active = (rsi_lo > 0) or (rsi_hi < 100)
                filtered = []
                for r in rows:
                    if r["CAGR %"] < ft_min_cagr:
                        continue
                    if r["R²"] < ft_min_r2:
                        continue
                    if r["Years"] < ft_min_years:
                        continue
                    if rsi_active:
                        rsi_val = r.get("RSI")
                        if rsi_val is None or not (
                            rsi_lo <= rsi_val <= rsi_hi
                        ):
                            continue
                    filtered.append(r)
                # Sort by composite score (CAGR × R² × drawdown penalty)
                filtered.sort(key=lambda r: r["Score"], reverse=True)
                filtered = filtered[:int(ft_topn)]

                _rsi_note = (
                    f", RSI {rsi_lo}–{rsi_hi}" if rsi_active else ""
                )
                st.caption(
                    f"Scanned **{ft_res['scanned']} tickers** in "
                    f"**{ft_res['universe']}**. "
                    f"{len(rows)} returned valid data, "
                    f"**{len(filtered)} passed** filters "
                    f"(CAGR ≥ {ft_min_cagr}%, R² ≥ {ft_min_r2:.2f}, "
                    f"≥ {ft_min_years}y history{_rsi_note})."
                )

                if not filtered:
                    st.warning(
                        "No tickers passed the current filters. Try lowering "
                        "Min CAGR or Min R²."
                    )
                else:
                    df_ft = pd.DataFrame(filtered)
                    # Ticker chips for quick visual scan
                    chips = []
                    for r in filtered[:30]:
                        t = r["Ticker"]
                        cagr = r["CAGR %"]
                        href = _chip_href(t, from_tab="Screener")
                        chips.append(
                            f"<a href='{href}' target='_self' "
                            f"style='background:#16a34a; color:#fff; "
                            "padding:3px 9px; border-radius:8px; "
                            "font-size:0.78rem; font-weight:700; "
                            "margin:3px; text-decoration:none; "
                            "display:inline-block;' "
                            f"title='Open {t} chart'>{t} "
                            f"{cagr:+.1f}%/yr</a>"
                        )
                    st.markdown(
                        "<div style='padding:8px; border-radius:8px; "
                        "background:rgba(22,163,74,0.06); "
                        "border:1px solid rgba(22,163,74,0.25); "
                        "margin-bottom:10px;'>"
                        + "".join(chips) + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Full table — sortable in Streamlit
                    st.dataframe(
                        df_ft,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Price": st.column_config.NumberColumn(
                                format="$%.2f"),
                            "Total Return %": st.column_config.NumberColumn(
                                format="%+.1f%%"),
                            "CAGR %": st.column_config.NumberColumn(
                                format="%+.1f%%"),
                            "Max DD %": st.column_config.NumberColumn(
                                format="%.1f%%"),
                            "R²": st.column_config.NumberColumn(
                                format="%.3f"),
                            "Years": st.column_config.NumberColumn(
                                format="%.1f"),
                            "RSI": st.column_config.NumberColumn(
                                format="%.1f",
                                help="Current RSI(14). <30 oversold, "
                                     ">70 overbought.",
                            ),
                            "Score": st.column_config.NumberColumn(
                                format="%.2f",
                                help="CAGR × R² × drawdown penalty — "
                                     "higher = steadier compounder.",
                            ),
                        },
                    )
                    st.caption(
                        "**How to read this:** *CAGR* = compounding "
                        "annual return. *R²* near 1.0 means the price "
                        "stayed glued to its trendline (a steady "
                        "compounder); R² near 0.5 means choppy. "
                        "*Max DD* is the worst peak-to-trough drop in "
                        "the entire history — context for whether "
                        "you could have held through it. *Score* "
                        "combines all three."
                    )

    with sc_tab7:
        st.divider()
        st.subheader("🚀 Rally Setup")
        st.caption(
            "Find tickers in a **rally setup** — oversold-but-turning, "
            "volume building, momentum curling up. Composite Rally Score "
            "combines RSI position + slope, MACD histogram trend, volume "
            "expansion, position vs 20/200 SMA, and Bollinger Band squeeze."
        )

        rs_col1, rs_col2 = st.columns([2, 1])
        rs_options = [
            "S&P 100 (~100)",
            "S&P 500 (~500 — slow)",
            "TSX 60 (~60)",
            "TSX Composite (~250)",
            "Entire TSX (~1500, slow)",
            "Entire TSX Venture (~1500, slow)",
            "Popular ETFs (~80)",
            "Custom watchlist",
        ] + [
            f"{lbl} ({len(tk)})"
            for lbl, tk in ss.INDUSTRY_UNIVERSES.items()
        ]
        rs_universe = rs_col1.selectbox(
            "Universe",
            options=rs_options,
            index=0,
            key="rs_universe",
        )
        rs_topn = rs_col2.number_input(
            "Show top N",
            min_value=10, max_value=200, value=30, step=10,
            key="rs_topn",
        )

        # Filter sliders
        rs_col3, rs_col4 = st.columns([1, 1])
        rs_min_score = rs_col3.slider(
            "Min Rally Score",
            min_value=0, max_value=100, value=60, step=5,
            key="rs_min_score",
            help="Composite 0–100 score. 60+ is a meaningful setup; "
                 "75+ is a strong confluence.",
        )
        rs_rsi_range = rs_col4.slider(
            "RSI(14) range",
            min_value=0, max_value=100, value=(25, 60), step=1,
            key="rs_rsi_range",
            help="Rally setups typically have RSI 30–55 — oversold but "
                 "not in free-fall. Drag to widen or tighten.",
        )

        rs_col5, rs_col6, rs_col7 = st.columns([1, 1, 1])
        rs_min_vol = rs_col5.slider(
            "Min volume ratio",
            min_value=0.5, max_value=3.0, value=1.0, step=0.1,
            key="rs_min_vol",
            help="Recent 5-day avg volume vs prior 30-day avg. "
                 ">1.0 = volume building (institutional accumulation).",
        )
        rs_above_200 = rs_col6.checkbox(
            "Require above 200-SMA",
            value=True,
            key="rs_above_200",
            help="Only show tickers still in their long-term uptrend.",
        )
        rs_squeeze = rs_col7.checkbox(
            "Require BB squeeze",
            value=False,
            key="rs_squeeze",
            help="Only show tickers with Bollinger Band width below "
                 "average (compressed volatility = potential breakout).",
        )

        if st.button("🚀 Find rally setups",
                     key="rs_run", type="primary"):
            with st.spinner("Loading universe…"):
                if rs_universe == "Custom watchlist":
                    tickers = [t.strip().upper() for t in
                               st.session_state.get("watchlist_input", "")
                                    .split(",") if t.strip()]
                elif rs_universe == "TSX 60 (~60)":
                    tickers = list(ss.UNIVERSE_TSX60)
                elif rs_universe == "TSX Composite (~250)":
                    tickers = ss.get_tsx_composite()
                elif rs_universe == "Entire TSX (~1500, slow)":
                    tickers = ss.get_full_tsx_listing("tsx")
                elif rs_universe == "Entire TSX Venture (~1500, slow)":
                    tickers = ss.get_full_tsx_listing("tsxv")
                elif rs_universe == "S&P 100 (~100)":
                    tickers = list(ss.UNIVERSE_SP100)
                elif rs_universe == "S&P 500 (~500 — slow)":
                    tickers = ss.get_sp500()
                elif rs_universe == "Popular ETFs (~80)":
                    tickers = list(ss.UNIVERSE_POPULAR_ETFS)
                else:
                    tickers = []
                    for lbl, tk in ss.INDUSTRY_UNIVERSES.items():
                        if rs_universe.startswith(lbl):
                            tickers = list(tk)
                            break

            if not tickers:
                st.warning("Universe is empty.")
            else:
                with st.spinner(
                    f"Computing rally signals for {len(tickers)} "
                    "tickers…"
                ):
                    rows = cached_rally_scan(tuple(tickers))
                st.session_state["rs_results"] = {
                    "rows": rows,
                    "universe": rs_universe,
                    "scanned": len(tickers),
                }

        # Render results
        rs_res = st.session_state.get("rs_results")
        if rs_res:
            rows = rs_res["rows"]
            if not rows:
                st.info("No tickers returned valid data.")
            else:
                rsi_lo, rsi_hi = rs_rsi_range
                filtered = []
                for r in rows:
                    if r["Rally Score"] < rs_min_score:
                        continue
                    if not (rsi_lo <= r["RSI"] <= rsi_hi):
                        continue
                    if r["Vol Ratio"] < rs_min_vol:
                        continue
                    if rs_above_200 and not r["Above 200"]:
                        continue
                    if rs_squeeze and r["BB Squeeze"] >= 1.0:
                        continue
                    filtered.append(r)

                filtered.sort(key=lambda r: r["Rally Score"], reverse=True)
                filtered = filtered[:int(rs_topn)]

                st.caption(
                    f"Scanned **{rs_res['scanned']} tickers** in "
                    f"**{rs_res['universe']}**. "
                    f"{len(rows)} returned data, "
                    f"**{len(filtered)} passed** the rally filter."
                )

                if not filtered:
                    st.warning(
                        "No setups found. Try lowering Min Rally Score "
                        "or widening the RSI range."
                    )
                else:
                    # Rally chip cloud — color intensity scales with score
                    chips = []
                    for r in filtered[:30]:
                        t = r["Ticker"]
                        sc = r["Rally Score"]
                        rsi_v = r["RSI"]
                        href = _chip_href(t, from_tab="Screener")
                        # Score 60-100 → green intensity 30-100%
                        intensity = max(0.3, min(1.0, (sc - 50) / 50))
                        chips.append(
                            f"<a href='{href}' target='_self' "
                            f"style='background:rgba(34,197,94,{intensity}); "
                            "color:#fff; padding:3px 9px; "
                            "border-radius:8px; font-size:0.78rem; "
                            "font-weight:700; margin:3px; "
                            "text-decoration:none; display:inline-block;' "
                            f"title='Rally Score {sc:.0f} · RSI {rsi_v:.0f}'>"
                            f"{t} ({sc:.0f})</a>"
                        )
                    st.markdown(
                        "<div style='padding:8px; border-radius:8px; "
                        "background:rgba(34,197,94,0.06); "
                        "border:1px solid rgba(34,197,94,0.25); "
                        "margin-bottom:10px;'>"
                        + "".join(chips) + "</div>",
                        unsafe_allow_html=True,
                    )

                    df_rs = pd.DataFrame(filtered)
                    st.dataframe(
                        df_rs,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Price": st.column_config.NumberColumn(
                                format="$%.2f"),
                            "Rally Score": st.column_config.NumberColumn(
                                format="%.1f",
                                help="0–100 composite score. Higher = "
                                     "stronger rally setup.",
                            ),
                            "RSI": st.column_config.NumberColumn(
                                format="%.1f"),
                            "RSI Δ5d": st.column_config.NumberColumn(
                                format="%+.1f",
                                help="RSI change over the last 5 bars. "
                                     "Positive = momentum building.",
                            ),
                            "MACD Hist": st.column_config.NumberColumn(
                                format="%+.3f",
                                help="MACD histogram value. Positive + "
                                     "rising = bullish momentum.",
                            ),
                            "Vol Ratio": st.column_config.NumberColumn(
                                format="%.2fx",
                                help=">1.0 = recent volume above 30-day "
                                     "baseline (accumulation).",
                            ),
                            "vs 20-SMA %": st.column_config.NumberColumn(
                                format="%+.1f%%",
                                help="Price vs 20-day moving average. "
                                     "Negative = pulled back below MA.",
                            ),
                            "vs 200-SMA %": st.column_config.NumberColumn(
                                format="%+.1f%%",
                                help="Price vs 200-day MA. Positive = "
                                     "long-term uptrend intact.",
                            ),
                            "BB Squeeze": st.column_config.NumberColumn(
                                format="%.2f",
                                help="BB width / 60-day avg width. "
                                     "<1.0 = compressed = potential "
                                     "breakout.",
                            ),
                            "Above 200": st.column_config.CheckboxColumn(),
                        },
                    )
                    st.caption(
                        "**How to read this:** A high *Rally Score* "
                        "means several signals align. Look for RSI "
                        "30–50 with positive Δ5d (turning up), MACD "
                        "histogram positive or curling up, Vol Ratio "
                        ">1.2 (volume building), Above 200 = ✓, and "
                        "BB Squeeze <1.0 (coiled). The strongest "
                        "setups have ALL of these together."
                    )


# === Custom Patterns tab ===
RULE_INDICATORS = {
    "Close":           "Close price ($)",
    "Volume":          "Volume",
    "RSI":             "RSI(14) — standard",
    "RSI7":            "RSI(7) — short-term swing",
    "RSI5":            "RSI(5) — fast / very short",
    "MACD":            "MACD line",
    "MACD_SIGNAL":     "MACD signal",
    "MACD_HIST":       "MACD histogram",
    "SMA5":            "SMA(5)",
    "SMA20":           "SMA(20)",
    "SMA50":           "SMA(50)",
    "SMA200":          "SMA(200)",
    "BB_LOWER":        "Bollinger lower",
    "BB_MID":          "Bollinger middle",
    "BB_UPPER":        "Bollinger upper",
    "ADX":             "ADX(14)",
    "DAILY_CHG_PCT":   "Daily change %",
    "DIST_SMA5_PCT":   "Distance from SMA5 (%)",
    "DIST_SMA20_PCT":  "Distance from SMA20 (%)",
    "DIST_SMA50_PCT":  "Distance from SMA50 (%)",
    "DIST_SMA200_PCT": "Distance from SMA200 (%)",
    "BB_PCT_B":          "Bollinger %B (0=lower band, 1=upper)",
    "BB_DIST_LOWER_PCT": "Distance to BB lower (%) [neg = below band]",
    "BB_DIST_UPPER_PCT": "Distance to BB upper (%) [neg = above band]",
    "BB_BANDWIDTH_PCT":  "Bollinger bandwidth (%) [low = squeeze]",
    "ANOMALY_SCORE":     "🤖 Anomaly score (IF) [more neg = anomalous]",
    "ANOMALY_PCTILE":    "🤖 Anomaly percentile (0=most anomalous)",
    "CONVICTION":        "🎯 Conviction score (-100 bear ↔ +100 bull)",
    "MFI":               "🌊 Money Flow Index (0-100, vol-weighted RSI)",
    "CMF":               "💧 Chaikin Money Flow (-1 to +1, accumulation)",
    "NEWS_SENT":         "📰 News sentiment (0=bear, 1=bull) [latest]",
    "NEWS_BUZZ":         "📣 News buzz (0=quiet, 1+=above avg) [latest]",
    "VOL_OUTLOOK":       "🔮 Volume outlook (0-100, higher=more vol coming)",
    "VOL_RATIO":         "📊 Volume ratio (today / 20d avg)",
    "ST_BULLISH":        "💬 StockTwits bullish % (0-1, retail sentiment)",
    "ST_BUZZ":           "💬 StockTwits buzz (msgs in last 24h)",
}
RULE_OPS = ["<", "<=", ">", ">=", "between"]

# Sensible default thresholds per indicator: (low, high)
# Low used for `<`/`<=`, high used for `>`/`>=`, both for `between`.
RULE_DEFAULTS: dict[str, tuple[float, float]] = {
    "Close":           (10.0, 100.0),
    "Volume":          (1_000_000.0, 10_000_000.0),
    "RSI":             (30.0, 70.0),
    "RSI7":            (25.0, 75.0),  # tighter band for shorter period
    "RSI5":            (20.0, 80.0),  # extreme band for very short period
    "MACD":            (-1.0, 1.0),
    "MACD_SIGNAL":     (-1.0, 1.0),
    "MACD_HIST":       (-0.5, 0.5),
    "SMA5":            (10.0, 100.0),
    "SMA20":           (10.0, 100.0),
    "SMA50":           (10.0, 100.0),
    "SMA200":          (10.0, 100.0),
    "BB_LOWER":        (10.0, 100.0),
    "BB_MID":          (10.0, 100.0),
    "BB_UPPER":        (10.0, 100.0),
    "ADX":             (20.0, 25.0),
    "DAILY_CHG_PCT":   (-5.0, 5.0),
    "DIST_SMA5_PCT":   (-3.0, 3.0),
    "DIST_SMA20_PCT":  (-5.0, 5.0),
    "DIST_SMA50_PCT":  (-7.0, 7.0),
    "DIST_SMA200_PCT": (-10.0, 10.0),
    "BB_PCT_B":          (0.0, 1.0),
    "BB_DIST_LOWER_PCT": (0.0, 5.0),   # < 0 = below lower band; > 5 = far above
    "BB_DIST_UPPER_PCT": (0.0, 5.0),   # < 0 = above upper band; > 5 = far below
    "BB_BANDWIDTH_PCT":  (5.0, 15.0),  # < 5 = squeeze; > 15 = volatile
    "ANOMALY_SCORE":     (-0.2, 0.0),  # < -0.2 ≈ anomalous; near 0 = normal
    "ANOMALY_PCTILE":    (5.0, 95.0),  # < 5 = bottom 5% (most anomalous)
    "CONVICTION":        (-30.0, 50.0),  # < -30 = avoid; > 50 = strong buy
    "MFI":               (20.0, 80.0),   # < 20 = oversold; > 80 = overbought
    "CMF":               (-0.05, 0.05),  # < -0.05 dist; > 0.05 accum
    "NEWS_SENT":         (0.4, 0.6),     # < 0.4 bearish news; > 0.6 bullish
    "NEWS_BUZZ":         (0.5, 1.5),     # > 1 = above-avg news activity
    "VOL_OUTLOOK":       (30.0, 70.0),   # < 30 quiet; > 70 elevated
    "VOL_RATIO":         (0.5, 2.0),     # > 2 = 2× avg volume
    "ST_BULLISH":        (0.4, 0.65),    # < 0.4 retail bearish; > 0.65 bullish
    "ST_BUZZ":           (5.0, 30.0),    # > 30 msgs/24h = loud
}


# Short value-meaning hints per indicator, shown above the value input in
# the rule editor to help users pick sensible thresholds.
RULE_VALUE_HINTS: dict[str, str] = {
    "Close":           "price in $ — depends on stock",
    "Volume":          "shares traded — depends on stock",
    "RSI":             "0-100 · < 30 oversold · > 70 overbought",
    "RSI7":            "0-100 · < 25 oversold · > 75 overbought (faster)",
    "RSI5":            "0-100 · < 20 oversold · > 80 overbought (fastest)",
    "MACD":            "raw value · positive bullish · negative bearish",
    "MACD_SIGNAL":     "raw value · MACD smoothed",
    "MACD_HIST":       "positive = bullish momentum · negative = bearish",
    "SMA5":            "$ value of 5-day moving average",
    "SMA20":           "$ value of 20-day moving average",
    "SMA50":           "$ value of 50-day moving average",
    "SMA200":          "$ value of 200-day moving average",
    "BB_LOWER":        "$ value of Bollinger lower band",
    "BB_MID":          "$ value of Bollinger middle band",
    "BB_UPPER":        "$ value of Bollinger upper band",
    "ADX":             "0-100 · < 20 weak trend · > 25 strong trend",
    "DAILY_CHG_PCT":   "% change today · ±2% = average · ±5% = big move",
    "DIST_SMA5_PCT":   "% above/below SMA5 · 0 = on the line",
    "DIST_SMA20_PCT":  "% above/below SMA20 · 0 = on the line",
    "DIST_SMA50_PCT":  "% above/below SMA50 · 0 = on the line",
    "DIST_SMA200_PCT": "% above/below SMA200 · > 0 = uptrend, < 0 = downtrend",
    "BB_PCT_B":        "0 = at lower band · 1 = at upper · < 0 below · > 1 above",
    "BB_DIST_LOWER_PCT": "% from BB lower · negative = below band",
    "BB_DIST_UPPER_PCT": "% from BB upper · negative = above band",
    "BB_BANDWIDTH_PCT":  "BB width as % of middle · < 5 = squeeze",
    "ANOMALY_SCORE":     "-0.5 to 0 · more negative = more anomalous",
    "ANOMALY_PCTILE":    "0-100 · < 10 = anomalous (bottom 10% of history)",
    "CONVICTION":        "-100 to +100 · > 50 strong bull · < -30 bear",
    "MFI":               "0-100 · < 20 oversold · > 80 overbought (vol-weighted)",
    "CMF":               "-1 to +1 · > 0.05 accumulation · < -0.05 distribution",
    "NEWS_SENT":         "0-1 · < 0.4 bearish · > 0.6 bullish news",
    "NEWS_BUZZ":         "0+ · 1 = average · > 2 = 2× normal coverage",
    "VOL_OUTLOOK":       "0-100 · > 60 elevated volume expected today",
    "VOL_RATIO":         "today vol ÷ 20d avg · > 1.5 above avg · > 3 spike",
    "ST_BULLISH":        "0-1 · > 0.65 retail bullish · < 0.4 retail bearish",
    "ST_BUZZ":           "msgs/24h · > 10 active · > 30 loud",
}


# Pre-built presets removed — Quick Presets row is now populated from any
# saved rule set the user has tagged "📌 Pinned" (see the Saved rule sets
# section). The dict below stays as the empty default for backward compat.
RULE_PRESETS: dict[str, list[dict]] = {}


# Hardcoded presets archive (kept as code for reference / future revival)
_RULE_PRESETS_ARCHIVE: dict[str, list[dict]] = {
    "🎯 Strong Buy": [
        {"left": "CONVICTION", "op": ">", "a": 50.0, "b": None,
         "date": None},
    ],
    "🟢 Bounce buy": [
        {"left": "RSI", "op": "<", "a": 35.0, "b": None, "date": None},
        {"left": "BB_PCT_B", "op": "<", "a": 0.2, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
        {"left": "MACD_HIST", "op": ">", "a": -0.5, "b": None,
         "date": None},
        {"left": "CONVICTION", "op": ">", "a": 30.0, "b": None,
         "date": None},
    ],
    "🌊 Vol-swing buy": [
        {"left": "MFI", "op": "<", "a": 25.0, "b": None, "date": None},
        {"left": "BB_PCT_B", "op": "<", "a": 0.25, "b": None, "date": None},
        {"left": "CMF", "op": ">", "a": 0.0, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
    ],
    "📰 News BUY": [
        {"left": "CONVICTION", "op": ">", "a": 30.0, "b": None,
         "date": None},
        {"left": "NEWS_SENT", "op": ">", "a": 0.55, "b": None,
         "date": None},
        {"left": "NEWS_BUZZ", "op": ">", "a": 0.5, "b": None,
         "date": None},
    ],
    "📰 News warning": [
        {"left": "CONVICTION", "op": ">", "a": 20.0, "b": None,
         "date": None},
        {"left": "NEWS_SENT", "op": "<", "a": 0.4, "b": None,
         "date": None},
    ],
    "⚠️ Wait (falling)": [
        {"left": "BB_PCT_B", "op": "<", "a": 0.3, "b": None, "date": None},
        {"left": "MACD_HIST", "op": "<", "a": 0.0, "b": None, "date": None},
        {"left": "DIST_SMA50_PCT", "op": "<", "a": 0.0, "b": None,
         "date": None},
        {"left": "RSI", "op": ">", "a": 30.0, "b": None, "date": None},
    ],
    "🚀 Momentum": [
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
        {"left": "DIST_SMA50_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
        {"left": "MACD_HIST", "op": ">", "a": 0.0, "b": None, "date": None},
        {"left": "ADX", "op": ">", "a": 25.0, "b": None, "date": None},
        {"left": "RSI", "op": "between", "a": 50.0, "b": 70.0, "date": None},
    ],
    "🔴 Strong Sell": [
        {"left": "CONVICTION", "op": "<", "a": -30.0, "b": None,
         "date": None},
    ],
    "💥 Squeeze": [
        {"left": "BB_BANDWIDTH_PCT", "op": "<", "a": 5.0, "b": None,
         "date": None},
    ],
    "🔮 Vol incoming": [
        # High volume outlook + bullish bias = potential breakout setup
        {"left": "VOL_OUTLOOK", "op": ">", "a": 60.0, "b": None,
         "date": None},
        {"left": "CONVICTION", "op": ">", "a": 20.0, "b": None,
         "date": None},
    ],
    "💬 Retail bullish": [
        # Retail crowd is bullish AND there's actual chatter (not stale)
        # AND technicals don't disagree
        {"left": "ST_BULLISH", "op": ">", "a": 0.65, "b": None,
         "date": None},
        {"left": "ST_BUZZ", "op": ">", "a": 10.0, "b": None,
         "date": None},
        {"left": "CONVICTION", "op": ">", "a": -10.0, "b": None,
         "date": None},
    ],
    "🆎 Retail vs technicals (contrarian)": [
        # Retail very bullish but technicals say overbought — classic
        # contrarian fade signal (retail tops are real)
        {"left": "ST_BULLISH", "op": ">", "a": 0.75, "b": None,
         "date": None},
        {"left": "RSI", "op": ">", "a": 70.0, "b": None, "date": None},
    ],
    "📚 Connors-style": [
        # Larry Connors' canonical short-term mean-reversion strategy:
        # RSI(2)<10 + price>SMA200. Adapted with RSI(5)<15. Works in
        # established uptrends; volume confirmation reduces false signals.
        {"left": "RSI5", "op": "<", "a": 15.0, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
        {"left": "CMF", "op": ">", "a": 0.0, "b": None, "date": None},
    ],
    "📚 Strong bounce": [
        # Volume-confirmed extreme oversold in uptrend.
        # MFI<20 (selling exhausted) + BB%B<0.15 (price extended below
        # band) + CMF>0 (institutional accumulation) + above SMA200.
        # The historical edge in this combo is materially better than
        # any single indicator alone.
        {"left": "MFI", "op": "<", "a": 20.0, "b": None, "date": None},
        {"left": "BB_PCT_B", "op": "<", "a": 0.15, "b": None, "date": None},
        {"left": "CMF", "op": ">", "a": 0.0, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
    ],
    "📚 News drift": [
        # Post-earnings drift proxy: high news buzz + bullish news
        # sentiment + bullish technicals + volume incoming. PEAD is
        # the strongest documented anomaly in finance — beats keep
        # drifting up for weeks.
        {"left": "NEWS_BUZZ", "op": ">", "a": 1.5, "b": None,
         "date": None},
        {"left": "NEWS_SENT", "op": ">", "a": 0.55, "b": None,
         "date": None},
        {"left": "CONVICTION", "op": ">", "a": 30.0, "b": None,
         "date": None},
        {"left": "VOL_OUTLOOK", "op": ">", "a": 50.0, "b": None,
         "date": None},
    ],
    "📚 Connors deep OS": [
        # Stricter Connors variant: RSI(5)<10 (deep oversold) + 5%+
        # above SMA200 (strong uptrend, not falling knife) + CMF>0.05
        # (real accumulation, not just neutral). Historically ~62-65%
        # 1-3 day win rate per Connors backtests.
        {"left": "RSI5", "op": "<", "a": 10.0, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 5.0, "b": None,
         "date": None},
        {"left": "CMF", "op": ">", "a": 0.05, "b": None, "date": None},
    ],
    "🕵️ Smart-money accumulation": [
        # The "something is coming" signal: high news buzz (story
        # developing) + above-average volume (institutional flow) +
        # but price hasn't moved much yet (not chasing — getting in
        # before the move). Classic "smart money before the news"
        # pattern. Verify edge with the validator.
        {"left": "NEWS_BUZZ", "op": ">", "a": 2.0, "b": None,
         "date": None},
        {"left": "VOL_RATIO", "op": ">", "a": 1.5, "b": None,
         "date": None},
        {"left": "DAILY_CHG_PCT", "op": "between", "a": -1.5, "b": 1.5,
         "date": None},
    ],
    "🔋 Oversold + Vol Surge": [
        # Low-price reversal candidate: oversold RSI + volume surge +
        # accumulation (CMF>0) + in long-term uptrend (filters falling
        # knives). The "stock dropped to a low and smart money is
        # picking it up" pattern.
        {"left": "RSI", "op": "<", "a": 30.0, "b": None, "date": None},
        {"left": "VOL_RATIO", "op": ">", "a": 1.5, "b": None,
         "date": None},
        {"left": "CMF", "op": ">", "a": 0.0, "b": None, "date": None},
        {"left": "DIST_SMA200_PCT", "op": ">", "a": 0.0, "b": None,
         "date": None},
    ],
}


def _rule_default_a(indicator: str, op: str) -> float:
    lo, hi = RULE_DEFAULTS.get(indicator, (0.0, 0.0))
    if op in (">", ">="):
        return hi
    return lo


def _rule_default_b(indicator: str) -> float:
    _, hi = RULE_DEFAULTS.get(indicator, (0.0, 0.0))
    return hi


def _bar_at(df, date_str: str | None):
    """Return the dataframe row at-or-before date_str, or the last row."""
    if df is None or df.empty:
        return None, None
    if not date_str:
        return df.iloc[-1], df  # latest bar; df for prev-bar lookup
    try:
        target = pd.Timestamp(date_str)
        if target.tzinfo is None and df.index.tz is not None:
            target = target.tz_localize(df.index.tz)
        sub = df[df.index <= target]
        if sub.empty:
            return None, None
        return sub.iloc[-1], sub
    except Exception:
        return None, None


def _last_value(df, key: str, date_str: str | None = None,
                ticker: str | None = None):
    """Get the value of a named indicator at a specific date (or latest bar).

    `ticker` is required for news-sentiment indicators (NEWS_SENT/NEWS_BUZZ)
    since those come from a per-ticker API rather than the price dataframe.
    """
    bar, sub = _bar_at(df, date_str)
    if bar is None:
        return None
    if key in df.columns:
        v = bar[key]
    elif key == "DAILY_CHG_PCT" and sub is not None and len(sub) >= 2:
        prev = sub.iloc[-2]["Close"]
        v = (bar["Close"] - prev) / prev * 100 if prev else None
    elif key == "DIST_SMA5_PCT" and "SMA5" in df.columns:
        v = (bar["Close"] - bar["SMA5"]) / bar["SMA5"] * 100 \
            if bar["SMA5"] else None
    elif key == "DIST_SMA20_PCT" and "SMA20" in df.columns:
        v = (bar["Close"] - bar["SMA20"]) / bar["SMA20"] * 100 \
            if bar["SMA20"] else None
    elif key == "DIST_SMA50_PCT" and "SMA50" in df.columns:
        v = (bar["Close"] - bar["SMA50"]) / bar["SMA50"] * 100 \
            if bar["SMA50"] else None
    elif key == "DIST_SMA200_PCT" and "SMA200" in df.columns:
        v = (bar["Close"] - bar["SMA200"]) / bar["SMA200"] * 100 \
            if bar["SMA200"] else None
    elif key == "BB_PCT_B" and {"BB_LOWER", "BB_UPPER"}.issubset(df.columns):
        rng = bar["BB_UPPER"] - bar["BB_LOWER"]
        v = (bar["Close"] - bar["BB_LOWER"]) / rng if rng else None
    elif key == "BB_DIST_LOWER_PCT" and "BB_LOWER" in df.columns:
        v = ((bar["Close"] - bar["BB_LOWER"]) / bar["BB_LOWER"] * 100
             if bar["BB_LOWER"] else None)
    elif key == "BB_DIST_UPPER_PCT" and "BB_UPPER" in df.columns:
        v = ((bar["BB_UPPER"] - bar["Close"]) / bar["BB_UPPER"] * 100
             if bar["BB_UPPER"] else None)
    elif key == "BB_BANDWIDTH_PCT" and {
        "BB_LOWER", "BB_UPPER", "BB_MID"
    }.issubset(df.columns):
        v = ((bar["BB_UPPER"] - bar["BB_LOWER"]) / bar["BB_MID"] * 100
             if bar["BB_MID"] else None)
    elif key in ("ANOMALY_SCORE", "ANOMALY_PCTILE"):
        # Anomaly detection always uses the LATEST bar (training requires
        # historical context). For date-anchored rules this evaluates the
        # current state, not a back-in-time snapshot.
        result = _cached_anomaly(df)
        if result is None:
            return None
        v = result["score"] if key == "ANOMALY_SCORE" else result["pctile"]
    elif key in ("ST_BULLISH", "ST_BUZZ"):
        if not ticker:
            return None
        st_data = cached_stocktwits(ticker)
        if not st_data:
            return None
        if key == "ST_BULLISH":
            v = st_data.get("bullish_pct")
            if v is None:
                return None
        else:  # ST_BUZZ
            v = st_data.get("msg_count_24h")
    elif key == "VOL_OUTLOOK":
        if not ticker:
            return None
        out = _cached_vol_outlook(ticker, df)
        if not out:
            return None
        v = out.get("score")
    elif key in ("NEWS_SENT", "NEWS_BUZZ"):
        # News sentiment is a current snapshot per ticker (Finnhub API).
        # Date-anchored rules can't fetch historical sentiment.
        if not ticker:
            return None
        sent = cached_sentiment(ticker)
        if not sent:
            return None
        if key == "NEWS_SENT":
            try:
                v = float(sent.get("sentiment", {}).get("bullishPercent"))
            except (TypeError, ValueError):
                return None
        else:  # NEWS_BUZZ
            try:
                v = float(sent.get("buzz", {}).get("buzz"))
            except (TypeError, ValueError):
                return None
    else:
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _series_for_indicator(df, key: str):
    """Return a pandas Series of an indicator's value across all bars.

    Used by historical-validation: lets us compute a boolean mask for
    every bar where a rule was true, instead of just the latest bar.
    Returns None for indicators that can't be computed historically
    (e.g., NEWS_SENT/NEWS_BUZZ are current-snapshot only).
    """
    if df is None or df.empty:
        return None
    if key in df.columns:
        return df[key]
    close = df["Close"]
    if key == "DAILY_CHG_PCT":
        return close.pct_change() * 100
    if key == "DIST_SMA5_PCT" and "SMA5" in df.columns:
        return (close - df["SMA5"]) / df["SMA5"].replace(0, float("nan")) * 100
    if key == "DIST_SMA20_PCT" and "SMA20" in df.columns:
        return (close - df["SMA20"]) / df["SMA20"].replace(0, float("nan")) * 100
    if key == "DIST_SMA50_PCT" and "SMA50" in df.columns:
        return (close - df["SMA50"]) / df["SMA50"].replace(0, float("nan")) * 100
    if key == "DIST_SMA200_PCT" and "SMA200" in df.columns:
        return (close - df["SMA200"]) / df["SMA200"].replace(0, float("nan")) * 100
    if key == "BB_PCT_B" and {"BB_LOWER", "BB_UPPER"}.issubset(df.columns):
        rng = (df["BB_UPPER"] - df["BB_LOWER"]).replace(0, float("nan"))
        return (close - df["BB_LOWER"]) / rng
    if key == "BB_DIST_LOWER_PCT" and "BB_LOWER" in df.columns:
        return ((close - df["BB_LOWER"])
                / df["BB_LOWER"].replace(0, float("nan")) * 100)
    if key == "BB_DIST_UPPER_PCT" and "BB_UPPER" in df.columns:
        return ((df["BB_UPPER"] - close)
                / df["BB_UPPER"].replace(0, float("nan")) * 100)
    if key == "BB_BANDWIDTH_PCT" and {
        "BB_LOWER", "BB_UPPER", "BB_MID"
    }.issubset(df.columns):
        return ((df["BB_UPPER"] - df["BB_LOWER"])
                / df["BB_MID"].replace(0, float("nan")) * 100)
    if key in ("ANOMALY_SCORE", "ANOMALY_PCTILE"):
        anom = ss.compute_anomaly_per_bar(df)
        if anom is None or anom.empty:
            return None
        # Reindex to df's index, NaN where missing
        return (anom["score"] if key == "ANOMALY_SCORE"
                else anom["pctile"]).reindex(df.index)
    # NEWS_SENT/NEWS_BUZZ are current-snapshot only — no historical series.
    return None


def _rule_mask(df, rule: dict):
    """Boolean Series: True at every bar where rule passes (False/NaN else).
    Returns None if the indicator can't be computed historically."""
    s = _series_for_indicator(df, rule["left"])
    if s is None:
        return None
    op = rule["op"]
    a = rule.get("a")
    b = rule.get("b")
    if a is None:
        return None
    if op == "<":      return s < a
    if op == "<=":     return s <= a
    if op == ">":      return s > a
    if op == ">=":     return s >= a
    if op == "between" and b is not None:
        lo, hi = (a, b) if a <= b else (b, a)
        return (s >= lo) & (s <= hi)
    return None


def _historical_match_returns(df, rules, fwd_days: int = 5):
    """For all bars where ALL non-date rules passed, compute the forward
    return over `fwd_days` trading days plus max drawdown during the hold.

    Returns list of dicts with keys: date, ret_pct, max_dd_pct.
    Skips: rules with explicit dates, rules using NEWS_* (no history).
    """
    if df is None or df.empty:
        return []
    masks = []
    for r in rules:
        if r.get("date"):
            continue  # date-anchored rules don't apply historically
        m = _rule_mask(df, r)
        if m is None:
            continue
        masks.append(m.fillna(False))
    if not masks:
        return []
    from functools import reduce
    combined = reduce(lambda x, y: x & y, masks)
    if not combined.any():
        return []

    closes = df["Close"]
    out = []
    for ts in combined[combined].index:
        try:
            i = closes.index.get_loc(ts)
        except KeyError:
            continue
        if i + fwd_days >= len(closes):
            continue
        entry = float(closes.iloc[i])
        exit_close = float(closes.iloc[i + fwd_days])
        if entry <= 0 or not isfinite_(entry):
            continue
        ret_pct = (exit_close - entry) / entry * 100.0
        # Max drawdown during the hold (peak-to-trough on Close)
        window = closes.iloc[i:i + fwd_days + 1]
        peak = window.cummax()
        dd = (window - peak) / peak * 100.0
        max_dd_pct = float(dd.min())
        out.append({
            "date": ts, "ret_pct": ret_pct,
            "max_dd_pct": max_dd_pct,
        })
    return out


def isfinite_(x):
    return x == x and x not in (float("inf"), float("-inf"))


def _eval_rule(df, rule: dict, ticker: str | None = None) -> bool | None:
    """Evaluate one rule. If rule has a 'date', evaluate against that bar;
    otherwise against the latest bar. Pass `ticker` for indicators that
    need API lookups (news sentiment). Returns None if data missing."""
    left = _last_value(df, rule["left"], rule.get("date"), ticker=ticker)
    if left is None:
        return None
    op = rule["op"]
    a = rule.get("a")
    b = rule.get("b")
    if a is None:
        return None
    if op == "<":      return left < a
    if op == "<=":     return left <= a
    if op == ">":      return left > a
    if op == ">=":     return left >= a
    if op == "between" and b is not None:
        lo, hi = (a, b) if a <= b else (b, a)
        return lo <= left <= hi
    return None


SAVED_RULES_PATH = pathlib.Path("saved_rules.json")


def _load_saved_rules() -> dict:
    if not SAVED_RULES_PATH.exists():
        return {}
    try:
        return json.loads(SAVED_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _persist_saved_rules(d: dict) -> None:
    SAVED_RULES_PATH.write_text(
        json.dumps(d, indent=2), encoding="utf-8"
    )


# Active-rules persistence: encode rules into URL `?rules=` so reopening the
# browser tab restores exactly what the user was editing.
def _rules_to_url(rules: list[dict]) -> str:
    import base64
    raw = json.dumps(rules, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _rules_from_url(s: str) -> list[dict] | None:
    import base64
    try:
        pad = "=" * (-len(s) % 4)
        raw = base64.urlsafe_b64decode((s + pad).encode("ascii"))
        out = json.loads(raw.decode("utf-8"))
        return out if isinstance(out, list) else None
    except Exception:
        return None


def _init_active_rules_from_url():
    if "custom_rules" in st.session_state:
        return
    encoded = st.query_params.get("rules")
    if encoded:
        loaded = _rules_from_url(encoded)
        if loaded:
            st.session_state.custom_rules = loaded
            return
    st.session_state.custom_rules = [
        {"left": "RSI", "op": "<", "a": 30.0, "b": None, "date": None},
    ]


def _sync_active_rules_to_url():
    rules = st.session_state.get("custom_rules") or []
    if rules:
        st.query_params["rules"] = _rules_to_url(rules)
    elif "rules" in st.query_params:
        del st.query_params["rules"]


with tab_patterns:
    (pt_sub1, pt_sub2) = st.tabs([
        "🛠️ Build & Run",
        "📊 Deep Analysis",
    ])

    with pt_sub1:
        st.subheader("Custom Watchlist Screener")
        st.caption(
            "Pick a preset, or build your own indicator rules. "
            "Tickers matching **all** rules are returned."
        )

        _init_active_rules_from_url()
        if "saved_rules" not in st.session_state:
            st.session_state.saved_rules = _load_saved_rules()

        saved = st.session_state.saved_rules

        # --- 🌐 Market regime banner ---
        regime = cached_market_regime()
        if regime and regime.get("regime") != "unknown":
            vix_str = (f"VIX <b>{regime['vix']:.1f}</b>"
                       if regime.get("vix") is not None else "VIX —")
            spy_str = ("above" if regime.get("spy_above_sma200") else "below")
            adx_str = f"ADX <b>{regime.get('spy_adx', 0):.0f}</b>"
            ret_str = (f"20d <b>{regime.get('spy_ret_20d', 0):+.1f}%</b>"
                       if "spy_ret_20d" in regime else "")
            suit_chips = "".join(
                f"<span style='background:#16a34a; color:#fff; "
                f"padding:2px 8px; border-radius:6px; font-size:0.72rem; "
                f"font-weight:700; margin-right:4px;'>{p}</span>"
                for p in regime.get("suitable", [])
            )
            st.markdown(
                f"<div style='padding:10px 14px; border-radius:10px; "
                f"background:rgba(96,165,250,0.06); "
                f"border:1px solid #4a4b4e; margin-bottom:10px;'>"
                f"<div style='font-size:0.95rem; color:#e5e7eb; "
                f"font-weight:700; margin-bottom:4px;'>"
                f"{regime['emoji']} Market regime: {regime['label']}</div>"
                f"<div style='font-size:0.75rem; color:#9ca3af; "
                f"margin-bottom:8px;'>"
                f"SPY {spy_str} SMA200 · {ret_str} · {adx_str} · {vix_str}"
                f"</div>"
                f"<div style='font-size:0.72rem; color:#9ca3af; "
                f"margin-bottom:4px;'>Presets that historically work in "
                f"this regime:</div><div>{suit_chips}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # --- ⚡ Quick presets — your own pinned saved rule sets ---
        st.markdown(
            "**⚡ Quick presets** — your pinned saved rules. "
            "📌 a saved rule below to add it here."
        )
        # Collect pinned saved rules
        pinned = []
        for name, entry in saved.items():
            if isinstance(entry, dict) and entry.get("pinned"):
                pinned.append((name, entry.get("rules", [])))
        if pinned:
            preset_cols = st.columns(len(pinned))
            for col, (pname, prules) in zip(preset_cols, pinned):
                if col.button(pname, key=f"preset_{pname}",
                              use_container_width=True,
                              help=(f"Click to load {pname} "
                                    f"({len(prules)} rules)")):
                    st.session_state.custom_rules = [
                        dict(r) for r in prules
                    ]
                    for r in st.session_state.custom_rules:
                        r.pop("_keyspec", None)
                    st.rerun()
        else:
            st.caption(
                "_No pinned rules yet. Build a rule set below, save it, "
                "then click 📌 to pin it here for one-click access._"
            )

        # --- 💾 Saved rule sets — collapsed by default ---
        saved_count = len(saved)
        saved_label = (f"💾 Saved rule sets ({saved_count})"
                       if saved_count else "💾 Save / load rule sets")
        with st.expander(saved_label, expanded=False):
            if saved:
                st.caption(
                    "🔔 Tag a rule set to receive **daily email alerts** when "
                    "it matches across the TSX + TSXV at market open."
                )
                for name in list(saved.keys()):
                    # Saved rule sets used to be plain lists; now support
                    # dict form {"rules": [...], "alert": bool}
                    entry = saved[name]
                    if isinstance(entry, list):
                        rules_list = entry
                        is_alert = False
                        is_pinned = False
                    else:
                        rules_list = entry.get("rules", [])
                        is_alert = bool(entry.get("alert", False))
                        is_pinned = bool(entry.get("pinned", False))

                    sc1, sc2, sc3, sc4, sc5 = st.columns(
                        [3.5, 1.0, 1.0, 1.0, 0.5]
                    )
                    badges = ""
                    if is_pinned:
                        badges += (
                            " &nbsp;·&nbsp; <span style='color:#fbbf24; "
                            "font-size:0.78rem; font-weight:700;'>📌 PINNED"
                            "</span>"
                        )
                    if is_alert:
                        badges += (
                            " &nbsp;·&nbsp; <span style='color:#22c55e; "
                            "font-size:0.78rem; font-weight:700;'>🔔 ALERT"
                            "</span>"
                        )
                    sc1.markdown(
                        f"**{name}** &nbsp;·&nbsp; "
                        f"<span style='color:#9ca3af'>"
                        f"{len(rules_list)} rule(s)</span>" + badges,
                        unsafe_allow_html=True,
                    )
                    if sc2.button("📂 Load", key=f"saved_load_{name}",
                                  use_container_width=True):
                        st.session_state.custom_rules = [
                            dict(r) for r in rules_list
                        ]
                        for r in st.session_state.custom_rules:
                            r.pop("_keyspec", None)
                        st.rerun()
                    pin_label = "📍 Unpin" if is_pinned else "📌 Pin"
                    if sc3.button(pin_label, key=f"saved_pin_{name}",
                                  use_container_width=True,
                                  help=("Remove from Quick Presets bar"
                                        if is_pinned else
                                        "Add to Quick Presets bar for "
                                        "one-click access")):
                        saved[name] = {
                            "rules": rules_list,
                            "alert": is_alert,
                            "pinned": not is_pinned,
                        }
                        _persist_saved_rules(saved)
                        st.rerun()
                    toggle_label = "🔕 Mute" if is_alert else "🔔 Alert"
                    if sc4.button(toggle_label, key=f"saved_alert_{name}",
                                  use_container_width=True,
                                  help=("Stop daily email alerts for this rule"
                                        if is_alert else
                                        "Get a daily email when this rule "
                                        "matches in the TSX/TSXV scan")):
                        saved[name] = {
                            "rules": rules_list,
                            "alert": not is_alert,
                            "pinned": is_pinned,
                        }
                        _persist_saved_rules(saved)
                        st.rerun()
                    if sc5.button("🗑️", key=f"saved_del_{name}",
                                  help=f"Delete '{name}'"):
                        del saved[name]
                        _persist_saved_rules(saved)
                        st.rerun()
                st.divider()
            save_c1, save_c2 = st.columns([4, 1.5])
            new_name = save_c1.text_input(
                "Save current rules as…",
                key="saved_new_name",
                placeholder="e.g. Oversold mean reversion",
                label_visibility="collapsed",
            )
            if save_c2.button("💾 Save", key="saved_save_btn",
                              use_container_width=True):
                nm = (new_name or "").strip()
                if not nm:
                    st.warning("Give the rule set a name first.")
                elif not st.session_state.custom_rules:
                    st.warning("No rules to save.")
                else:
                    # Preserve existing alert + pinned flags when overwriting
                    existing_alert = False
                    existing_pinned = False
                    existing = saved.get(nm)
                    if isinstance(existing, dict):
                        existing_alert = bool(existing.get("alert", False))
                        existing_pinned = bool(existing.get("pinned", False))
                    saved[nm] = {
                        "rules": [dict(r) for r in st.session_state.custom_rules],
                        "alert": existing_alert,
                        "pinned": existing_pinned,
                    }
                    _persist_saved_rules(saved)
                    st.success(f"Saved “{nm}”.")
                    st.rerun()

        st.divider()

        rules = st.session_state.custom_rules

        # --- Rule editor (simplified) ---
        st.markdown("**🛠️ Rules**")
        st.caption(
            "Indicator + comparison + value. Click 📅 to lock a rule to a "
            "specific historical date instead of the latest bar."
        )
        today = datetime.now().date()
        for i, rule in enumerate(rules):
            # Inline hint showing what the value means for the chosen
            # indicator. Helps pick sensible thresholds without trial+error.
            hint = RULE_VALUE_HINTS.get(rule["left"], "")
            if hint:
                # Stretch a small caption across the value column area
                # (indicator + op + value = 3+1.4+2 weights). Use HTML so
                # we can right-align it under the value box.
                st.markdown(
                    f"<div style='font-size:0.7rem; color:#9ca3af; "
                    f"margin:0 0 -4px 0; padding-left:53%;'>"
                    f"💡 {hint}</div>",
                    unsafe_allow_html=True,
                )
            # Layout: indicator | op | value (+ optional upper) | date | delete
            is_between = rule["op"] == "between"
            if is_between:
                cols = st.columns([3, 1.4, 1.5, 1.5, 0.6, 0.6])
                c_left, c_op, c_a, c_b, c_date, c_del = cols
            else:
                cols = st.columns([3, 1.4, 2, 0.6, 0.6])
                c_left, c_op, c_a, c_date, c_del = cols
                c_b = None

            rule["left"] = c_left.selectbox(
                "Indicator",
                options=list(RULE_INDICATORS.keys()),
                format_func=lambda k: RULE_INDICATORS[k],
                index=list(RULE_INDICATORS.keys()).index(rule["left"]),
                key=f"rule_left_{i}",
                label_visibility="collapsed",
            )
            rule["op"] = c_op.selectbox(
                "Op", options=RULE_OPS,
                index=RULE_OPS.index(rule["op"]),
                key=f"rule_op_{i}",
                label_visibility="collapsed",
            )

            # Auto-default value when indicator/op changes
            keyspec = f"{rule['left']}_{rule['op']}"
            if rule.get("_keyspec") != keyspec:
                rule["a"] = _rule_default_a(rule["left"], rule["op"])
                if rule["op"] == "between":
                    rule["b"] = _rule_default_b(rule["left"])
                rule["_keyspec"] = keyspec

            rule["a"] = c_a.number_input(
                "Value",
                value=float(rule.get("a") or 0.0),
                key=f"rule_a_{i}_{keyspec}",
                label_visibility="collapsed",
                format="%.4f",
            )
            if is_between and c_b is not None:
                rule["b"] = c_b.number_input(
                    "Upper",
                    value=float(rule.get("b") or 0.0),
                    key=f"rule_b_{i}_{keyspec}",
                    label_visibility="collapsed",
                    format="%.4f",
                )
            elif not is_between:
                rule["b"] = None

            # Date as a popover — clean, compact, no checkbox needed
            has_date = bool(rule.get("date"))
            date_icon = "📅✓" if has_date else "📅"
            with c_date.popover(date_icon,
                                help="Evaluate this rule on a specific date "
                                     "(defaults to latest bar)"):
                existing_date = rule.get("date")
                try:
                    init_date = (datetime.strptime(existing_date, "%Y-%m-%d").date()
                                 if existing_date else today)
                except ValueError:
                    init_date = today
                use_date = st.checkbox(
                    "Use specific date",
                    value=has_date,
                    key=f"rule_usedate_{i}",
                )
                if use_date:
                    picked = st.date_input(
                        "Date",
                        value=init_date,
                        max_value=today,
                        key=f"rule_date_{i}",
                    )
                    rule["date"] = picked.strftime("%Y-%m-%d")
                    st.caption(f"_Rule fires only on bar at/before {rule['date']}_")
                else:
                    rule["date"] = None
                    st.caption("_Rule evaluates the latest bar_")

            if c_del.button("🗑️", key=f"rule_del_{i}",
                            help="Remove this rule"):
                rules.pop(i)
                st.rerun()

        add_c, _ = st.columns([2, 8])
        if add_c.button("➕ Add rule", key="rule_add"):
            rules.append({"left": "Close", "op": ">", "a": 0.0,
                          "b": None, "date": None})
            st.rerun()

        # Persist active rules to URL so the next browser visit restores them
        _sync_active_rules_to_url()

        st.divider()

        # --- Run section ---
        wl_tickers = [t.strip().upper() for t in
                      st.session_state.get("watchlist_input", "").split(",")
                      if t.strip()]
        st.markdown("##### Run against")

        universe_options = {
            "watchlist": f"📋 Watchlist ({len(wl_tickers)} tickers)",
            "popular_etfs": f"🧺 Popular ETFs ({len(ss.UNIVERSE_POPULAR_ETFS)})",
            "sp100": f"🇺🇸 S&P 100 ({len(ss.UNIVERSE_SP100)})",
            "sp500": "🇺🇸 S&P 500 (~500 — slower)",
            "tsx60": f"🍁 TSX 60 ({len(ss.UNIVERSE_TSX60)})",
            "tsx_full": "🍁 Full TSX (~1500 — much slower)",
            "us_full": "🇺🇸 Full US (~10000 — very slow, no OTC)",
            "crypto": f"₿ Crypto ({len(ss.UNIVERSE_CRYPTO)} major coins)",
        }
        # Industry filters — add curated sector lists
        for ind_label, ind_tickers in ss.INDUSTRY_UNIVERSES.items():
            universe_options[f"industry:{ind_label}"] = (
                f"{ind_label} ({len(ind_tickers)})"
            )
        uni_col, lim_col = st.columns([3, 1])
        universe_key = uni_col.selectbox(
            "Universe",
            options=list(universe_options.keys()),
            format_func=lambda k: universe_options[k],
            index=0,
            key="patterns_universe",
            label_visibility="collapsed",
        )
        max_tickers = lim_col.number_input(
            "Max tickers",
            min_value=10, max_value=10000, value=10000, step=100,
            key="patterns_max_tickers",
            help="Cap how many tickers are scanned. Default = max (no cap). "
                 "Lower it to speed up scans on Full TSX / Full US.",
            label_visibility="collapsed",
        )

        run_c, clear_c = st.columns([3, 1])
        run_btn = run_c.button(
            "🔍 Evaluate", key="rules_run", type="primary",
            use_container_width=True,
        )
        if clear_c.button(
            "🧹 Clear results", key="rules_clear",
            use_container_width=True,
            disabled=not st.session_state.get("patterns_last_results"),
        ):
            st.session_state.pop("patterns_last_results", None)
            st.rerun()

        if run_btn:
            if not rules:
                st.warning("Add at least one rule.")
            else:
                # Resolve the universe to a ticker list
                with st.spinner("Loading universe…"):
                    if universe_key == "watchlist":
                        target_tickers = wl_tickers
                    elif universe_key == "popular_etfs":
                        target_tickers = ss.UNIVERSE_POPULAR_ETFS
                    elif universe_key == "sp100":
                        target_tickers = ss.UNIVERSE_SP100
                    elif universe_key == "sp500":
                        target_tickers = ss.get_sp500()
                    elif universe_key == "tsx60":
                        target_tickers = ss.UNIVERSE_TSX60
                    elif universe_key == "tsx_full":
                        target_tickers = ss.get_full_tsx_listing()
                    elif universe_key == "us_full":
                        target_tickers = ss.get_full_us_listing()
                    elif universe_key == "crypto":
                        target_tickers = list(ss.UNIVERSE_CRYPTO)
                    elif universe_key.startswith("industry:"):
                        ind_label = universe_key.removeprefix("industry:")
                        target_tickers = list(
                            ss.INDUSTRY_UNIVERSES.get(ind_label, [])
                        )
                    else:
                        target_tickers = wl_tickers

                target_tickers = (target_tickers or [])[:int(max_tickers)]
                if not target_tickers:
                    st.warning("Universe is empty.")
                else:
                    matches = []
                    details = []
                    hist_returns = []  # forward-return validation across history
                    fwd_days = 5  # 5 trading days = ~1 week
                    progress = st.progress(0.0)
                    status = st.empty()
                    for idx, t in enumerate(target_tickers):
                        try:
                            norm = ss.normalize_ticker(t)
                        except SystemExit:
                            continue
                        try:
                            df, _ = cached_single(
                                norm, period, interval, strategy,
                                adx_filter, stop_loss_pct,
                            )
                        except Exception:
                            df = None
                        if df is None:
                            progress.progress((idx + 1) / len(target_tickers))
                            continue
                        row = {"Ticker": t}
                        rule_results = [_eval_rule(df, r, ticker=t) for r in rules]
                        row["Matches"] = (
                            all(r is True for r in rule_results)
                            if rule_results else False
                        )
                        for k in ["Close", "RSI", "MACD_HIST", "DAILY_CHG_PCT"]:
                            v = _last_value(df, k)
                            row[k] = round(v, 4) if v is not None else None
                        details.append(row)
                        if row["Matches"]:
                            matches.append(t)
                        # Historical validation: forward-return where rules
                        # passed in this ticker's history
                        try:
                            h = _historical_match_returns(df, rules,
                                                           fwd_days=fwd_days)
                            for entry in h:
                                entry["ticker"] = t
                                hist_returns.append(entry)
                        except Exception:
                            pass
                        progress.progress((idx + 1) / len(target_tickers))
                        if (idx + 1) % 20 == 0 or idx == len(target_tickers) - 1:
                            status.caption(
                                f"Scanned {idx + 1}/{len(target_tickers)} · "
                                f"{len(matches)} matches · "
                                f"{len(hist_returns)} historical so far"
                            )
                    progress.empty()
                    status.empty()

                    # Cache results so they survive popup-related reruns
                    st.session_state["patterns_last_results"] = {
                        "matches": matches,
                        "details": details,
                        "scanned": len(target_tickers),
                        "universe": universe_options.get(universe_key, ""),
                        "hist_returns": hist_returns,
                        "fwd_days": fwd_days,
                    }

        # Render last results (if any) — survives popup open/close reruns
        last = st.session_state.get("patterns_last_results")
        if last:
            matches = last["matches"]
            details = last["details"]
            scanned = last["scanned"]
            st.caption(
                f"Last evaluation · scanned **{scanned}** tickers in "
                f"**{last.get('universe', '')}**"
            )
            if matches:
                st.success(
                    f"✅ {len(matches)} match{'es' if len(matches) > 1 else ''} "
                    f"out of {scanned} scanned"
                )
                # Render all matched-ticker chips (no cap). For long lists, the
                # chips wrap naturally and scroll vertically with a max-height.
                MAX_CHIPS = 300
                shown = matches[:MAX_CHIPS]
                chips = []
                for t in shown:
                    href = _chip_href(t, from_tab="Custom Patterns")
                    chips.append(
                        f"<a href='{href}' target='_self' "
                        "style='background:#16a34a; color:#fff; "
                        "padding:3px 10px; border-radius:8px; "
                        "font-size:0.85rem; font-weight:700; "
                        "margin:3px; text-decoration:none; "
                        "display:inline-block;' "
                        f"title='Open {t} chart'>📊 {t}</a>"
                    )
                extra = len(matches) - len(shown)
                extra_html = (
                    f"<span style='color:#9ca3af; font-size:0.8rem; "
                    f"margin-left:6px;'>+{extra} more (see Details table)"
                    f"</span>" if extra > 0 else ""
                )
                # Cap container height so 100s of chips don't dominate the page
                max_h = 200 if len(shown) > 30 else "auto"
                st.markdown(
                    f"<div style='max-height:{max_h}px; overflow-y:auto; "
                    f"padding:6px; border-radius:8px; "
                    f"background:rgba(34,197,94,0.04); "
                    f"border:1px solid rgba(34,197,94,0.2);'>"
                    "<b style='color:#9ca3af; margin-right:6px; "
                    "font-size:0.85rem;'>📊 Tickers (click to open chart):</b>"
                    + "".join(chips) + extra_html + "</div>"
                    if isinstance(max_h, int) else
                    "<div style='line-height:2.2;'>"
                    "<b style='color:#9ca3af; margin-right:6px;'>"
                    "📊 Tickers (click to open chart):</b>"
                    + "".join(chips) + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No matches in {scanned} tickers scanned.")

            if details:
                st.markdown("##### Details")
                st.dataframe(
                    pd.DataFrame(details),
                    use_container_width=True,
                    hide_index=True,
                )


    with pt_sub2:
        last = st.session_state.get("patterns_last_results")
        if not last:
            st.info(
                "📭 Run an evaluation in **🛠️ Build & Run** "
                "first to see historical edge analysis here."
            )
        else:
            rules = st.session_state.get("custom_rules", [])
            # === Historical edge: forward-return validation ===
            hist_returns = last.get("hist_returns") or []
            fwd_days = last.get("fwd_days", 5)
            if hist_returns:
                st.markdown(f"##### 📈 Historical edge (next {fwd_days}d returns)")
                st.caption(
                    "Across this ticker history: every bar where ALL rules "
                    "would have passed, then look forward "
                    f"{fwd_days} trading days. **NOT a prediction** — "
                    "just empirical pattern measurement."
                )
                rets = pd.Series([h["ret_pct"] for h in hist_returns])
                dds = pd.Series([h["max_dd_pct"] for h in hist_returns])
                n = len(rets)
                win_rate = (rets > 0).mean() * 100
                avg_ret = rets.mean()

                # Compute profit factor early so verdict can use it
                wins = rets[rets > 0]
                losses = rets[rets <= 0]
                sum_wins = wins.sum() if len(wins) else 0.0
                sum_losses = abs(losses.sum()) if len(losses) else 0.0
                pf_early = (sum_wins / sum_losses
                            if sum_losses > 0 else float("inf"))

                # === Verdict box: explicit KEEP / REFINE / KILL ===
                verdict_emoji = "❓"
                verdict_color = "#9ca3af"
                verdict_label = "INCONCLUSIVE"
                verdict_reason = ""

                if n < 30:
                    verdict_emoji = "📉"
                    verdict_color = "#9ca3af"
                    verdict_label = "INCONCLUSIVE — not enough data"
                    verdict_reason = (
                        f"Only {n} historical matches. Stats are noisy below 30. "
                        "Widen rules or scan a bigger universe (S&P 500, "
                        "full TSX) to get a meaningful sample."
                    )
                elif (win_rate >= 55 and pf_early >= 1.3
                      and avg_ret > 0.5):
                    verdict_emoji = "✅"
                    verdict_color = "#22c55e"
                    verdict_label = "KEEP — passes 55% threshold"
                    verdict_reason = (
                        f"Win rate {win_rate:.1f}% (≥55%), profit factor "
                        f"{pf_early:.2f}, avg return {avg_ret:+.2f}%. "
                        "Real edge in this sample. Still expect 30-50% "
                        "degradation in live trading from costs/slippage."
                    )
                elif win_rate < 50 or pf_early < 1.0:
                    verdict_emoji = "❌"
                    verdict_color = "#ef4444"
                    verdict_label = "KILL — no edge"
                    verdict_reason = (
                        f"Win rate {win_rate:.1f}% and profit factor "
                        f"{pf_early:.2f}. These rules didn't help "
                        "historically — refine or scrap. Common fixes: add "
                        "a trend filter (above SMA200), require volume "
                        "confirmation (CMF>0), tighten thresholds."
                    )
                elif pf_early >= 1.0 and avg_ret > 0:
                    verdict_emoji = "⚠️"
                    verdict_color = "#fbbf24"
                    verdict_label = "REFINE — marginal edge"
                    verdict_reason = (
                        f"Win rate {win_rate:.1f}%, profit factor "
                        f"{pf_early:.2f}, avg {avg_ret:+.2f}%. Better than "
                        "random but probably won't survive transaction "
                        "costs. Try combining with another factor "
                        "(volume, news sentiment) or stricter thresholds."
                    )
                else:
                    verdict_emoji = "❌"
                    verdict_color = "#ef4444"
                    verdict_label = "KILL — negative expectancy"
                    verdict_reason = (
                        f"Win rate {win_rate:.1f}%, avg return {avg_ret:+.2f}%. "
                        "Losing money historically. Scrap and rebuild."
                    )

                st.markdown(
                    f"<div style='padding:14px 16px; border-radius:10px; "
                    f"margin:10px 0; "
                    f"background:rgba({_hex_to_rgb(verdict_color)}, 0.12); "
                    f"border-left:4px solid {verdict_color};'>"
                    f"<div style='font-size:1.05rem; "
                    f"color:{verdict_color}; font-weight:700; "
                    f"margin-bottom:4px;'>"
                    f"{verdict_emoji} {verdict_label}</div>"
                    f"<div style='font-size:0.85rem; color:#e5e7eb; "
                    f"line-height:1.5;'>{verdict_reason}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                med_ret = rets.median()
                avg_dd = dds.mean()
                best = rets.max()
                worst = rets.min()
                std_ret = rets.std()

                stats_cols = st.columns(6)
                stats_cols[0].metric("Sample", f"{n}")
                stats_cols[1].metric(
                    "Win rate", f"{win_rate:.1f}%",
                    delta=("edge" if win_rate > 55 else
                           "weak" if win_rate > 50 else "no edge"),
                    delta_color=("normal" if win_rate > 55 else "off"
                                 if win_rate > 50 else "inverse"),
                )
                stats_cols[2].metric(
                    "Avg return", f"{avg_ret:+.2f}%",
                    delta_color="normal" if avg_ret > 0 else "inverse",
                )
                stats_cols[3].metric("Median", f"{med_ret:+.2f}%")
                stats_cols[4].metric(
                    "Avg max DD", f"{avg_dd:.2f}%",
                    delta_color="off",
                )
                stats_cols[5].metric("σ", f"{std_ret:.2f}%")

                # Asymmetric-payoff metrics
                wins = rets[rets > 0]
                losses = rets[rets <= 0]
                avg_win = wins.mean() if len(wins) else 0.0
                avg_loss = losses.mean() if len(losses) else 0.0
                sum_wins = wins.sum() if len(wins) else 0.0
                sum_losses = abs(losses.sum()) if len(losses) else 0.0
                profit_factor = (
                    sum_wins / sum_losses if sum_losses > 0 else float("inf")
                )
                rr = (avg_win / abs(avg_loss)
                      if avg_loss < 0 else float("inf"))
                extra_cols = st.columns(3)
                extra_cols[0].metric(
                    "Avg win", f"{avg_win:+.2f}%",
                    delta=f"{len(wins)} bars", delta_color="off",
                )
                extra_cols[1].metric(
                    "Avg loss", f"{avg_loss:+.2f}%",
                    delta=f"{len(losses)} bars", delta_color="off",
                )
                extra_cols[2].metric(
                    "Profit factor",
                    f"{profit_factor:.2f}" if profit_factor != float("inf")
                    else "∞",
                    delta=f"R:R {rr:.2f}" if rr != float("inf") else "—",
                    delta_color=("normal" if profit_factor > 1.3
                                 else "off" if profit_factor > 1.0
                                 else "inverse"),
                )

                # Distribution histogram
                try:
                    import plotly.graph_objects as go
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Histogram(
                        x=rets, nbinsx=40,
                        marker_color="#60a5fa", opacity=0.85,
                    ))
                    fig_h.add_vline(x=0, line_color="#e5e7eb",
                                    line_width=1, opacity=0.6)
                    fig_h.add_vline(x=avg_ret, line_color="#22c55e",
                                    line_dash="dash", line_width=1.5,
                                    annotation_text=f"avg {avg_ret:+.1f}%",
                                    annotation_position="top right")
                    fig_h.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#4a4b4e",
                        plot_bgcolor="#4a4b4e",
                        height=260,
                        margin=dict(l=40, r=20, t=20, b=40),
                        xaxis_title=f"{fwd_days}-day forward return %",
                        yaxis_title="count",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                except Exception:
                    pass

                # Honest takeaway — based primarily on profit factor + avg return,
                # not just win rate. Trend-following systems often have <50% win
                # rate but positive expectancy from asymmetric R:R.
                if n < 30:
                    st.warning(
                        f"📉 **Sample too small** ({n} matches) — stats are "
                        "noisy. Widen the rules or scan a bigger universe."
                    )
                elif profit_factor >= 1.5 and avg_ret > 0.5:
                    st.success(
                        f"✅ **Real edge**: profit factor "
                        f"{profit_factor:.2f}, avg {avg_ret:+.2f}% over "
                        f"{n} matches. Asymmetric payoff (R:R {rr:.2f}) — "
                        f"{win_rate:.0f}% win rate is fine here."
                        if win_rate < 55 else
                        f"✅ **Real edge**: {win_rate:.0f}% win rate + "
                        f"{avg_ret:+.2f}% avg over {n} matches "
                        f"(profit factor {profit_factor:.2f})."
                    )
                elif profit_factor >= 1.15 and avg_ret > 0:
                    st.info(
                        f"⚠️ **Marginal edge**: profit factor "
                        f"{profit_factor:.2f}, avg {avg_ret:+.2f}%, "
                        f"win rate {win_rate:.0f}%. "
                        "Real but small — might not survive transaction "
                        "costs / slippage / taxes."
                    )
                elif avg_ret > 0 and profit_factor >= 1.0:
                    st.info(
                        f"💡 **Asymmetric profile**: {win_rate:.0f}% win rate "
                        f"but {avg_ret:+.2f}% avg (avg win {avg_win:+.1f}% vs "
                        f"avg loss {avg_loss:+.1f}%). Wins are bigger than "
                        f"losses, but profit factor {profit_factor:.2f} is "
                        "barely above 1 — fragile, easy to lose to costs."
                    )
                else:
                    st.error(
                        f"❌ **No edge**: {win_rate:.0f}% win rate, "
                        f"{avg_ret:+.2f}% avg, profit factor "
                        f"{profit_factor:.2f}. This rule set lost money "
                        "historically — refine the rules."
                    )

                # === Per-ticker breakdown — which tickers does this rule
                #     actually work best on? ===
                st.markdown("##### 🎯 Per-ticker win rate breakdown")
                st.caption(
                    "Same rule set, grouped by ticker. Shows which specific "
                    "names respond best to your pattern. Even if the "
                    "aggregate win rate is mediocre, some tickers may have "
                    "real edge — others may have none."
                )
                per_ticker: dict[str, list[float]] = {}
                for h in hist_returns:
                    tk = h.get("ticker")
                    if not tk:
                        continue
                    per_ticker.setdefault(tk, []).append(h["ret_pct"])
                # Build per-ticker stats with min sample size
                min_sample = 3
                per_ticker_rows = []
                for tk, ret_list in per_ticker.items():
                    if len(ret_list) < min_sample:
                        continue
                    s = pd.Series(ret_list)
                    wins_t = (s > 0).sum()
                    per_ticker_rows.append({
                        "Ticker": tk,
                        "Matches": int(len(ret_list)),
                        "Win %": round(wins_t / len(ret_list) * 100, 1),
                        "Avg Return %": round(float(s.mean()), 2),
                        "Median %": round(float(s.median()), 2),
                        "Best %": round(float(s.max()), 2),
                        "Worst %": round(float(s.min()), 2),
                    })
                if per_ticker_rows:
                    df_pt = pd.DataFrame(per_ticker_rows)
                    df_pt = df_pt.sort_values(
                        "Win %", ascending=False, na_position="last"
                    ).reset_index(drop=True)
                    df_pt.insert(0, "Rank", range(1, len(df_pt) + 1))

                    # Top-N chips
                    top_chips = []
                    for _, r in df_pt.head(30).iterrows():
                        tk = r["Ticker"]
                        wr = r["Win %"]
                        if wr >= 65:
                            color = "#16a34a"
                        elif wr >= 55:
                            color = "#65a30d"
                        elif wr >= 45:
                            color = "#a16207"
                        else:
                            color = "#9ca3af"
                        href = _chip_href(tk, from_tab="Custom Patterns")
                        top_chips.append(
                            f"<a href='{href}' target='_self' style='"
                            f"background:{color}; color:#fff; "
                            f"padding:3px 9px; border-radius:8px; "
                            f"font-size:0.78rem; font-weight:700; "
                            f"margin:3px; text-decoration:none; "
                            f"display:inline-block;' "
                            f"title='{tk} · Win {wr}% · "
                            f"{r['Matches']} historical matches'>"
                            f"{tk} {wr:.0f}%</a>"
                        )
                    st.markdown(
                        "<div style='padding:6px; margin-bottom:10px; "
                        "border-radius:8px; "
                        "background:rgba(34,197,94,0.03); "
                        "border:1px solid rgba(34,197,94,0.2);'>"
                        "<b style='color:#9ca3af; margin-right:6px; "
                        "font-size:0.85rem;'>Top 30 tickers (click for "
                        "chart):</b>"
                        + "".join(top_chips) + "</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        df_pt, use_container_width=True, hide_index=True,
                        column_config={
                            "Win %": st.column_config.NumberColumn(
                                format="%.1f%%"
                            ),
                            "Avg Return %": st.column_config.NumberColumn(
                                format="%+.2f%%"
                            ),
                            "Median %": st.column_config.NumberColumn(
                                format="%+.2f%%"
                            ),
                            "Best %": st.column_config.NumberColumn(
                                format="%+.2f%%"
                            ),
                            "Worst %": st.column_config.NumberColumn(
                                format="%+.2f%%"
                            ),
                        },
                    )
                    # Quick takeaway based on top result
                    top = df_pt.iloc[0]
                    if top["Win %"] >= 70 and top["Matches"] >= 5:
                        st.success(
                            f"🏆 **{top['Ticker']}** is a strong fit: "
                            f"{top['Win %']}% win rate over "
                            f"{top['Matches']} matches, "
                            f"{top['Avg Return %']:+.2f}% avg return. "
                            "Consider focusing this rule on this ticker."
                        )
                    elif top["Win %"] >= 60:
                        st.info(
                            f"**{top['Ticker']}** is the best fit "
                            f"({top['Win %']}%, n={top['Matches']}), but "
                            "no ticker is a clear standout."
                        )
                else:
                    st.caption(
                        f"_No ticker had ≥{min_sample} historical matches. "
                        "Try a bigger universe or looser rules to get "
                        "per-ticker stats._"
                    )
            elif last.get("matches") is not None:
                # We ran an evaluation but got 0 historical matches
                # (e.g., all rules date-anchored or NEWS_*-only)
                note = ""
                if any(r.get("date") for r in rules):
                    note = (
                        " (rules with specific dates can't be validated "
                        "historically — drop the date pin to enable validation)."
                    )
                elif any(r.get("left") in ("NEWS_SENT", "NEWS_BUZZ")
                         for r in rules):
                    note = (
                        " (news-sentiment indicators only have current values, "
                        "not historical — they're skipped in validation)."
                    )
                st.caption(
                    "📈 Historical edge: 0 matches across history" + note
                )



# === News tab ===
@st.cache_data(ttl=900, show_spinner=False)
def cached_general_news(category: str = "general") -> list:
    return ss.finnhub_general_news(category=category)


def _aggregate_ticker_news(tickers: list[str], days: int = 3,
                           per_ticker: int = 4) -> list[dict]:
    """Pull news for each ticker, tag with ticker, sort newest first."""
    out = []
    for t in tickers:
        for art in cached_news(t, days=days)[:per_ticker]:
            a = dict(art)
            a["_ticker"] = t
            out.append(a)
    out.sort(key=lambda x: x.get("datetime", 0) or 0, reverse=True)
    return out


def _affected_tickers(art: dict, watchlist: list[str],
                      curated: list[str]) -> tuple[list[str], list[str]]:
    """Return (in_watchlist_hits, other_curated_hits) for an article.

    Sources:
      1. The "_ticker" we tagged when aggregating per-ticker news.
      2. Finnhub's "related" field (comma-separated tickers).
      3. Keyword scan of headline + summary for watchlist + curated tickers.
    """
    candidates = set()
    src = art.get("_ticker")
    if src:
        candidates.add(src.upper())
    rel = art.get("related") or ""
    for t in rel.split(","):
        t = t.strip().upper()
        if t:
            candidates.add(t)
    text = (art.get("headline", "") + " " + art.get("summary", "")).upper()
    # Cheap word-boundary scan: bare ticker (no .TO) appears as a token.
    import re
    for t in set(watchlist + curated):
        bare = t.split(".")[0]
        if len(bare) < 2:
            continue
        if re.search(rf"\b{re.escape(bare)}\b", text):
            candidates.add(t.upper())
    wl_set = {w.upper() for w in watchlist}
    in_wl = sorted(c for c in candidates if c in wl_set)
    other = sorted(c for c in candidates if c not in wl_set)
    return in_wl, other


def _render_news_card(art: dict, show_ticker: bool = True,
                      watchlist: list[str] | None = None,
                      curated: list[str] | None = None,
                      key_prefix: str = "") -> None:
    try:
        ts = datetime.fromtimestamp(art.get("datetime", 0))
        date_str = ts.strftime("%b %d %H:%M")
    except (ValueError, TypeError, OSError):
        date_str = "?"
    headline = (art.get("headline") or "").strip()
    summary = (art.get("summary") or "").strip()
    src = art.get("source", "")
    url = art.get("url") or "#"
    in_wl, other = _affected_tickers(
        art, watchlist or [], curated or []
    )
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:0.78rem; color:#9ca3af;'>"
            f"📅 {date_str} &nbsp;·&nbsp; 📰 {src}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{headline}**")
        if summary:
            st.caption(summary[:280] + ("…" if len(summary) > 280 else ""))
        # Affected-ticker chips — clickable links styled as pills
        all_affected = in_wl + other
        if all_affected:
            chips = []
            for t in in_wl[:8]:
                href = _chip_href(t)
                chips.append(
                    f"<a href='{href}' target='_self' "
                    "style='background:#16a34a; color:#fff; "
                    "padding:2px 9px; border-radius:8px; "
                    "font-size:0.78rem; font-weight:700; "
                    "margin-right:5px; text-decoration:none; "
                    "display:inline-block;'"
                    f">★ {t}</a>"
                )
            for t in other[:10]:
                href = _chip_href(t)
                chips.append(
                    f"<a href='{href}' target='_self' "
                    "style='background:#374151; color:#f0f0f0; "
                    "padding:2px 9px; border-radius:8px; "
                    "font-size:0.78rem; font-weight:700; "
                    "margin-right:5px; text-decoration:none; "
                    "display:inline-block;'"
                    f">{t}</a>"
                )
            st.markdown(
                "<div style='margin-top:6px; line-height:1.9;'>"
                "<span style='font-size:0.78rem; color:#9ca3af; "
                "margin-right:6px;'>📊 <b>Affects:</b></span>"
                + "".join(chips) +
                "</div>",
                unsafe_allow_html=True,
            )
        if art.get("url"):
            st.markdown(f"[Read more →]({url})")


@st.cache_data(ttl=900, show_spinner=False)
def cached_insider(ticker: str, days: int = 90) -> list:
    return ss.finnhub_insider_transactions(ticker, days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_earnings(days_ahead: int = 30, symbol: str | None = None) -> list:
    return ss.finnhub_earnings_calendar(days_ahead=days_ahead, symbol=symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ipo(days_ahead: int = 30) -> list:
    return ss.finnhub_ipo_calendar(days_ahead=days_ahead)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_etf_holdings(ticker: str) -> list:
    return ss.finnhub_etf_holdings(ticker)


with tab_news:
    if not ss.FINNHUB_API_KEY:
        st.warning("Set FINNHUB_API_KEY in .env to enable news.")
    else:
        # --- MAIN SECTION: Insider Transactions + IPO Calendar ---
        main_l, main_r = st.columns(2)

        with main_l:
            st.subheader("👤 Insider Transactions")
            st.caption("Form 4 filings (last 90 days) for watchlist tickers.")
            insider_rows = []
            for t in _wl_normalized[:20]:
                txns = cached_insider(t, days=90)
                buys = sum(1 for x in txns if (x.get("change") or 0) > 0)
                sells = sum(1 for x in txns if (x.get("change") or 0) < 0)
                if buys or sells:
                    insider_rows.append({
                        "Ticker": t, "Buys": buys, "Sells": sells,
                        "Net": buys - sells, "Total": buys + sells,
                    })
            if insider_rows:
                insider_df = pd.DataFrame(insider_rows).sort_values(
                    "Net", ascending=False
                )
                st.dataframe(insider_df, use_container_width=True,
                             hide_index=True)
            else:
                st.info("No recent insider activity for watchlist tickers.")

        with main_r:
            st.subheader("🚀 Upcoming IPOs")
            st.caption("Next 30 days.")
            ipos = cached_ipo(days_ahead=30)
            if ipos:
                ipo_df = pd.DataFrame([{
                    "Date": x.get("date"),
                    "Symbol": x.get("symbol"),
                    "Name": x.get("name"),
                    "Exchange": x.get("exchange"),
                    "Price range": x.get("priceRange"),
                } for x in ipos[:50]])
                st.dataframe(ipo_df, use_container_width=True,
                             hide_index=True)
            else:
                st.info("No upcoming IPOs.")

        st.divider()

        # --- SUBSECTION: All market data (collapsed by default) ---
        with st.expander("📊 All market data (earnings · insider · ETF holdings · IPOs)",
                         expanded=False):

            # Earnings calendar (next 30 days for watchlist)
            st.markdown("##### 📅 Earnings calendar — watchlist (30 days)")
            wl_earn_rows = []
            for t in _wl_normalized[:20]:
                er = cached_earnings(days_ahead=30, symbol=t)
                for e in er:
                    wl_earn_rows.append({
                        "Date": e.get("date"),
                        "Ticker": t,
                        "EPS est.": e.get("epsEstimate"),
                        "Rev est.": e.get("revenueEstimate"),
                        "Hour": e.get("hour"),
                    })
            if wl_earn_rows:
                st.dataframe(
                    pd.DataFrame(wl_earn_rows).sort_values("Date"),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("_No upcoming earnings for watchlist tickers._")

            st.divider()

            # Detailed insider per-ticker
            st.markdown("##### 👤 Insider transactions — detailed")
            ins_pick = st.selectbox(
                "Ticker", options=_wl_normalized,
                key="insider_detail_ticker",
            )
            if ins_pick:
                txns = cached_insider(ins_pick, days=90)
                if txns:
                    df_ins = pd.DataFrame([{
                        "Date": x.get("transactionDate"),
                        "Name": x.get("name"),
                        "Δ Shares": x.get("change"),
                        "Price": x.get("transactionPrice"),
                        "Code": x.get("transactionCode"),
                    } for x in txns[:50]])
                    st.dataframe(df_ins, use_container_width=True,
                                 hide_index=True)
                else:
                    st.caption("_No transactions in last 90 days._")

            st.divider()

            # ETF holdings lookup
            st.markdown("##### 🧺 ETF holdings lookup")
            etf_pick = st.text_input(
                "ETF ticker", placeholder="e.g. SPY, HOD.TO, XEQT.TO",
                key="etf_holdings_ticker",
            )
            if etf_pick:
                holdings = cached_etf_holdings(etf_pick.strip().upper())
                if holdings:
                    df_h = pd.DataFrame([{
                        "Symbol": h.get("symbol"),
                        "Name": h.get("name"),
                        "% of fund": h.get("percent"),
                        "Shares": h.get("share"),
                    } for h in holdings[:50]])
                    st.dataframe(df_h, use_container_width=True,
                                 hide_index=True)
                else:
                    st.caption(
                        "_No holdings data — Finnhub free tier covers "
                        "a limited set of US ETFs._"
                    )

            st.divider()

            # Full IPO calendar (longer horizon)
            st.markdown("##### 🚀 IPO calendar — 90 days")
            ipos_long = cached_ipo(days_ahead=90)
            if ipos_long:
                st.dataframe(
                    pd.DataFrame([{
                        "Date": x.get("date"),
                        "Symbol": x.get("symbol"),
                        "Name": x.get("name"),
                        "Exchange": x.get("exchange"),
                        "Price range": x.get("priceRange"),
                        "Shares (M)": x.get("numberOfShares"),
                    } for x in ipos_long]).sort_values("Date"),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("_No upcoming IPOs in next 90 days._")

        st.divider()
        st.subheader("📰 Market News")

        # --- Main: TSX (left) + AI/US (right) ---
        main_l, main_r = st.columns(2)

        all_curated = list(set(
            ss.MAJOR_TSX_FOR_NEWS + ss.MAJOR_AI_US_FOR_NEWS
            + ss.OIL_GAS_FOR_NEWS
        ))

        with main_l:
            st.markdown("##### 🛢️ Oil & Gas")
            oil_main = _aggregate_ticker_news(
                ss.OIL_GAS_FOR_NEWS, days=3, per_ticker=2
            )
            if oil_main:
                for art in oil_main[:12]:
                    _render_news_card(
                        art, show_ticker=True,
                        watchlist=_wl_normalized, curated=all_curated,
                        key_prefix="main_oil",
                    )
            else:
                st.caption("_No oil/gas news in the last 3 days._")

        with main_r:
            st.markdown("##### 🤖 AI / US Tech")
            ai_news = _aggregate_ticker_news(
                ss.MAJOR_AI_US_FOR_NEWS, days=3, per_ticker=2
            )
            if ai_news:
                for art in ai_news[:12]:
                    _render_news_card(
                        art, show_ticker=True,
                        watchlist=_wl_normalized, curated=all_curated,
                        key_prefix="main_ai",
                    )
            else:
                st.caption("_No AI/US Tech news in the last 3 days._")

        st.divider()

        # --- Subtabs: All Canadian / All US market-affecting news ---
        st.markdown("##### All market-affecting news")
        sub_ca, sub_us, sub_tsx, sub_search = st.tabs(
            ["🍁 Canadian", "🇺🇸 US", "🍁 TSX & Canada",
             "🔍 Search by ticker"]
        )

        with sub_ca:
            ca_all = _aggregate_ticker_news(
                ss.MAJOR_TSX_FOR_NEWS, days=7, per_ticker=8
            )
            if ca_all:
                st.caption(
                    f"{len(ca_all)} headlines from {len(ss.MAJOR_TSX_FOR_NEWS)} "
                    "major TSX names · last 7 days"
                )
                for art in ca_all[:60]:
                    _render_news_card(
                        art, show_ticker=True,
                        watchlist=_wl_normalized, curated=all_curated,
                        key_prefix="sub_ca",
                    )
            else:
                st.info("No Canadian market news in the last 7 days.")

        with sub_us:
            general = cached_general_news(category="general")
            us_curated = _aggregate_ticker_news(
                ss.MAJOR_AI_US_FOR_NEWS, days=7, per_ticker=4
            )
            # Merge general + curated, dedupe by url, sort newest first
            seen = set()
            combined = []
            for art in (general or [])[:60] + us_curated:
                u = art.get("url")
                if not u or u in seen:
                    continue
                seen.add(u)
                combined.append(art)
            combined.sort(
                key=lambda x: x.get("datetime", 0) or 0, reverse=True
            )
            if combined:
                st.caption(
                    f"{len(combined)} headlines · Finnhub general feed + "
                    f"{len(ss.MAJOR_AI_US_FOR_NEWS)} major US tickers"
                )
                for art in combined[:80]:
                    _render_news_card(
                        art, show_ticker=bool(art.get("_ticker")),
                        watchlist=_wl_normalized, curated=all_curated,
                        key_prefix="sub_us",
                    )
            else:
                st.info("No US market news available.")

        with sub_tsx:
            tsx_all = _aggregate_ticker_news(
                ss.MAJOR_TSX_FOR_NEWS, days=7, per_ticker=4
            )
            if tsx_all:
                st.caption(
                    f"{len(tsx_all)} headlines from "
                    f"{len(ss.MAJOR_TSX_FOR_NEWS)} major TSX names "
                    "(banks, energy, telecom, rail) · last 7 days."
                )
                for art in tsx_all[:80]:
                    _render_news_card(
                        art, show_ticker=True,
                        watchlist=_wl_normalized, curated=all_curated,
                        key_prefix="sub_tsx",
                    )
            else:
                st.info("No TSX news in the last 7 days.")

        with sub_search:
            col1, col2 = st.columns([3, 1])
            news_raw = col1.text_input("Ticker for news", "AAPL",
                                       key="news_ticker",
                                       help="Try US tickers — Finnhub TSX news coverage is sparse")
            days = col2.number_input("Days back", 1, 30, 7)

            if news_raw:
                try:
                    t = news_raw.strip().upper()
                    with st.spinner(f"Loading news for {t}…"):
                        articles = cached_news(t, days=days)
                    if not articles:
                        st.info(f"No news returned for {t} in the last {days} days.")
                    else:
                        st.success(f"Found {len(articles)} articles")
                        for art in articles[:30]:
                            _render_news_card(
                                art, show_ticker=False,
                                watchlist=_wl_normalized, curated=all_curated,
                                key_prefix="sub_search",
                            )
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


# Browser-side localStorage backup. Survives Streamlit Cloud rebuilds and
# situations where the URL ?wl= gets stripped (e.g., navigating to bare URL).
# Logic on each page load:
#   1. Mirror current ?wl= → localStorage (write-through)
#   2. If ?wl= is missing AND localStorage has a value, redirect once with
#      ?wl=... appended. The sessionStorage flag prevents redirect loops.
def _inject_watchlist_localstorage():
    import streamlit.components.v1 as components
    # Only mirror to localStorage when the watchlist is USER-OWNED
    # (i.e., set explicitly via URL, add/remove buttons, or bulk edit).
    # Default-fallback values must NEVER overwrite the user's saved list.
    is_user_owned = st.session_state.get("_wl_from_url", False)
    current_wl = st.session_state.get("watchlist_input", "")
    parts = [p.strip().upper() for p in current_wl.split(",") if p.strip()]
    current_wl_str = ",".join(parts) if (parts and is_user_owned) else ""
    js_value = json.dumps(current_wl_str)
    components.html(
        f"""<script>
        (function() {{
            const win = window.parent;
            const url = new URL(win.location.href);
            const KEY = 'streamlit_watchlist';
            const FLAG = 'wl_restored_once';
            const current = {js_value};
            if (current) {{
                // Write-through: keep localStorage in sync with this session
                try {{ win.localStorage.setItem(KEY, current); }} catch (e) {{}}
            }}
            // If URL is missing the param, try restoring from localStorage
            if (!url.searchParams.get('wl')) {{
                let stored = '';
                try {{ stored = win.localStorage.getItem(KEY) || ''; }} catch (e) {{}}
                if (stored && !win.sessionStorage.getItem(FLAG)) {{
                    win.sessionStorage.setItem(FLAG, '1');
                    url.searchParams.set('wl', stored);
                    win.location.replace(url.toString());
                }}
            }} else {{
                // We have a URL value — clear the redirect flag for next time
                try {{ win.sessionStorage.removeItem(FLAG); }} catch (e) {{}}
            }}
        }})();
        </script>""",
        height=0,
    )


_inject_watchlist_localstorage()

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


# Global CSS — make tabs larger, bolder, with clearer active state
st.markdown(
    """
    <style>
    /* Extra breathing room between major page sections */
    .stApp .main .block-container > div > div > div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 14px;
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


def _is_dark_theme() -> bool:
    """Detect Streamlit's current theme so chart line colors can adapt."""
    try:
        return (st.get_option("theme.base") or "dark") != "light"
    except Exception:
        return True


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
                    f'<div style="text-align:center; line-height:1.25;">'
                    f'<span style="font-size:1.05rem; font-weight:600;">${q["price"]:.2f}</span>'
                    f'<br>'
                    f'<span style="font-size:0.9rem; color:{color}; font-weight:600;">'
                    f'{arrow} {sign}{chg:.2f}%</span>'
                    f'</div>',
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

def _add_ticker_to_watchlist(new_t: str) -> None:
    """Shared logic — add a normalized ticker to the watchlist."""
    new_t = (new_t or "").strip().upper()
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


@st.cache_data(ttl=86400, show_spinner=False)
def _all_tickers_for_dropdown() -> list:
    """Combined sorted list of S&P 500 + TSX Composite + popular ETFs."""
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
    return sorted(set(parts))


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

(tab_scan, tab_single, tab_screener, tab_patterns, tab_news,
 tab_help) = st.tabs(
    ["📊 Stocks/ETFs", "🔍 Single Ticker", "🎯 Screener",
     "🧩 Custom Patterns", "📰 News", "ℹ️ Help"]
)
# After the popup closes, restore the tab the user was on (if any)
_restore_active_tab()


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
    with st.expander("📝 Edit / reorder watchlist (paste, clear, rearrange)",
                     expanded=False):
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


# === Custom Patterns tab ===
RULE_INDICATORS = {
    "Close":           "Close price ($)",
    "Volume":          "Volume",
    "RSI":             "RSI(14)",
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
    "BB_PCT_B":        "Bollinger %B (0–1)",
}
RULE_OPS = ["<", "<=", ">", ">=", "between"]


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


def _last_value(df, key: str, date_str: str | None = None):
    """Get the value of a named indicator at a specific date (or latest bar)."""
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
    else:
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _eval_rule(df, rule: dict) -> bool | None:
    """Evaluate one rule. If rule has a 'date', evaluate against that bar;
    otherwise against the latest bar. Returns None if data missing."""
    left = _last_value(df, rule["left"], rule.get("date"))
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


with tab_patterns:
    st.subheader("Custom Watchlist Screener")
    st.caption(
        "Build a set of indicator rules. Tickers in your watchlist matching "
        "**all** rules (AND) on the latest bar are listed below. "
        "Save named rule sets to reload later or share via the JSON file."
    )

    if "custom_rules" not in st.session_state:
        st.session_state.custom_rules = [
            {"left": "RSI", "op": "<", "a": 30.0, "b": None},
        ]
    if "saved_rules" not in st.session_state:
        st.session_state.saved_rules = _load_saved_rules()

    saved = st.session_state.saved_rules

    # --- Saved rules ---
    st.markdown("##### Saved rule sets")
    if saved:
        for name in list(saved.keys()):
            sc1, sc2, sc3, sc4 = st.columns([4, 2, 1, 1])
            sc1.markdown(
                f"**{name}** &nbsp;·&nbsp; "
                f"<span style='color:#9ca3af'>"
                f"{len(saved[name])} rule(s)</span>",
                unsafe_allow_html=True,
            )
            if sc2.button("📂 Load", key=f"saved_load_{name}",
                          use_container_width=True):
                st.session_state.custom_rules = [
                    dict(r) for r in saved[name]
                ]
                st.rerun()
            if sc3.button("🗑️", key=f"saved_del_{name}",
                          help=f"Delete '{name}'"):
                del saved[name]
                _persist_saved_rules(saved)
                st.rerun()
            sc4.markdown("")
    else:
        st.caption("_No saved rule sets yet._")

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
            saved[nm] = [dict(r) for r in st.session_state.custom_rules]
            _persist_saved_rules(saved)
            st.success(f"Saved “{nm}”.")
            st.rerun()

    st.divider()

    rules = st.session_state.custom_rules

    # --- Rule editor ---
    st.markdown("##### Rules")
    st.caption(
        "Each rule can target the **latest bar** (default) or a **specific date** "
        "in history. Multiple rules with different dates let you screen for "
        "patterns like *RSI < 30 on Mar 26* AND *RSI > 70 on Apr 10*."
    )
    today = datetime.now().date()
    for i, rule in enumerate(rules):
        cols = st.columns([2.5, 1.5, 1.8, 1.8, 1.4, 2.2, 0.6])
        c_left, c_op, c_a, c_b, c_use_date, c_date, c_del = cols
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
        rule["a"] = c_a.number_input(
            "Value",
            value=float(rule.get("a") or 0.0),
            key=f"rule_a_{i}",
            label_visibility="collapsed",
            format="%.4f",
        )
        if rule["op"] == "between":
            rule["b"] = c_b.number_input(
                "Upper",
                value=float(rule.get("b") or 0.0),
                key=f"rule_b_{i}",
                label_visibility="collapsed",
                format="%.4f",
            )
        else:
            c_b.markdown("&nbsp;", unsafe_allow_html=True)
            rule["b"] = None

        use_date = c_use_date.checkbox(
            "On date",
            value=bool(rule.get("date")),
            key=f"rule_usedate_{i}",
            help="Evaluate against a specific historical date instead of the latest bar",
        )
        if use_date:
            existing_date = rule.get("date")
            try:
                init_date = (datetime.strptime(existing_date, "%Y-%m-%d").date()
                             if existing_date else today)
            except ValueError:
                init_date = today
            picked = c_date.date_input(
                "Date",
                value=init_date,
                max_value=today,
                key=f"rule_date_{i}",
                label_visibility="collapsed",
            )
            rule["date"] = picked.strftime("%Y-%m-%d")
        else:
            c_date.markdown("&nbsp;", unsafe_allow_html=True)
            rule["date"] = None

        if c_del.button("🗑️", key=f"rule_del_{i}",
                        help="Remove this rule"):
            rules.pop(i)
            st.rerun()

    add_c, _ = st.columns([2, 8])
    if add_c.button("➕ Add rule", key="rule_add"):
        rules.append({"left": "Close", "op": ">", "a": 0.0,
                      "b": None, "date": None})
        st.rerun()

    st.divider()

    # --- Run section ---
    wl_tickers = [t.strip().upper() for t in
                  st.session_state.get("watchlist_input", "").split(",")
                  if t.strip()]
    st.markdown(f"##### Run against watchlist · {len(wl_tickers)} tickers")

    run_btn = st.button("🔍 Evaluate", key="rules_run", type="primary")
    if run_btn:
        if not rules:
            st.warning("Add at least one rule.")
        elif not wl_tickers:
            st.warning("Watchlist is empty.")
        else:
            matches = []
            details = []  # everything (matched and not), for transparency
            progress = st.progress(0.0)
            for idx, t in enumerate(wl_tickers):
                try:
                    norm = ss.normalize_ticker(t)
                except SystemExit:
                    continue
                df, _ = cached_single(norm, period, interval, strategy,
                                      adx_filter, stop_loss_pct)
                row = {"Ticker": t}
                rule_results = [_eval_rule(df, r) for r in rules]
                row["Matches"] = (
                    all(r is True for r in rule_results)
                    if rule_results else False
                )
                # Snapshot key values
                for k in ["Close", "RSI", "MACD_HIST", "DAILY_CHG_PCT"]:
                    v = _last_value(df, k)
                    row[k] = round(v, 4) if v is not None else None
                details.append(row)
                if row["Matches"]:
                    matches.append(t)
                progress.progress((idx + 1) / len(wl_tickers))
            progress.empty()

            if matches:
                st.success(
                    f"✅ {len(matches)} match — {', '.join(matches)}"
                )
            else:
                st.info("No tickers in your watchlist match all rules.")

            st.markdown("##### Details")
            st.dataframe(
                pd.DataFrame(details),
                use_container_width=True,
                hide_index=True,
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

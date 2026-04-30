"""Streamlit dashboard for the TSX signal scanner.

Run with:    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

import stock_signals as ss

st.set_page_config(page_title="Stock Signals", layout="wide", page_icon="📈")


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
    # Rates / yields
    return f"{val:.2f}%"


def render_macro_row(macro: dict, header: str | None = None) -> None:
    if header:
        st.caption(header)
    cols = st.columns(len(macro))
    for col, (label, val) in zip(cols, macro.items()):
        col.metric(label, _format_macro_value(label, val))


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


def render_quick_analysis():
    """Inline analysis panel shown when a watchlist tile is clicked."""
    selected = st.session_state.get("selected_tile")
    if not selected:
        return
    period = st.session_state.get("_period", "2y")
    interval = st.session_state.get("_interval", "1d")
    strategy = st.session_state.get("_strategy", "trend")
    adx_filter = st.session_state.get("_adx_filter", False)
    stop_loss_pct = st.session_state.get("_stop_loss_pct")

    with st.container(border=True):
        header_col, close_col = st.columns([5, 1])
        header_col.markdown(f"### 🎯 Quick view: **{selected}**")
        if close_col.button("✖ Close", key="close_quick_view",
                            use_container_width=True):
            st.session_state.pop("selected_tile", None)
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
        c4.metric("Buy & Hold", f"{stats['buy_hold']:+.1%}")
        c5.metric("Max DD", f"{stats.get('max_drawdown', 0):.1%}")

        fig = ss.build_chart(df, ticker, stats)
        st.pyplot(fig, use_container_width=True)
        st.caption("For news, analyst data, and fundamentals — open the **Single Ticker** tab below.")


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

        st.dataframe(
            df_view,
            use_container_width=True,
            height=720,
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%+d"),
                "Close": st.column_config.NumberColumn(format="$%.2f"),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "P/E": st.column_config.NumberColumn(format="%.1f"),
                "Yield %": st.column_config.NumberColumn(format="%.2f"),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
                "Upside %": st.column_config.NumberColumn(format="%+.1f"),
                "Earn Days": st.column_config.NumberColumn(format="%d"),
                "Buys": st.column_config.NumberColumn(format="%d"),
                "Holds": st.column_config.NumberColumn(format="%d"),
                "Sells": st.column_config.NumberColumn(format="%d"),
                "Trades": st.column_config.NumberColumn(format="%d"),
                "Stops": st.column_config.NumberColumn(format="%d"),
                "Win %": st.column_config.NumberColumn(format="%.0f"),
                "Strat %": st.column_config.NumberColumn(format="%+.1f"),
                "Max DD %": st.column_config.NumberColumn(
                    format="%.1f",
                    help="Worst peak-to-trough drawdown of the strategy",
                ),
                "B&H %": st.column_config.NumberColumn(format="%+.1f"),
                "ADX": st.column_config.NumberColumn(
                    format="%.1f",
                    help="Trend strength: <20 weak, >25 strong, >40 very strong",
                ),
            },
        )

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

    raw = st.text_input("Ticker", "RY", key="single_ticker")
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

                fig = ss.build_chart(df, ticker, stats)
                st.pyplot(fig, use_container_width=True)


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
        options=["S&P 100 (~100)", "TSX 60 (~60)", "Custom watchlist"],
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

    bb_col, rsi_col = sc_col1.columns(2)
    require_bb = bb_col.checkbox(
        "Bollinger BUY", value=True, key="screener_require_bb",
        help=f"Bollinger lower-band touch in the lookback window",
    )
    require_rsi = rsi_col.checkbox(
        "RSI oversold", value=True, key="screener_require_rsi",
        help="Current RSI ≤ threshold",
    )

    if universe_choice.startswith("S&P"):
        universe = ss.UNIVERSE_SP100
    elif universe_choice.startswith("TSX"):
        universe = ss.UNIVERSE_TSX60
    else:
        universe = list(tickers)

    st.caption(f"Will scan **{len(universe)}** tickers (one batched API call).")

    if st.button("🎯 Run screener", type="primary",
                 disabled=not (require_bb or require_rsi)):
        with st.spinner(f"Scanning {len(universe)} tickers…"):
            matches = ss.screen_buy_signals(
                universe,
                rsi_threshold=rsi_thresh,
                lookback_bars=lookback_days,
                require_bollinger=require_bb,
                require_rsi=require_rsi,
            )

        if not matches:
            st.info(
                "No matches. Try a wider lookback window, higher RSI threshold, "
                "or untick one filter."
            )
        else:
            # Split into "today" and "earlier"
            today_matches = [m for m in matches if m.get("bb_buy_age") == 0]
            earlier_matches = [m for m in matches if m.get("bb_buy_age") and m["bb_buy_age"] > 0]
            no_bb_matches = [m for m in matches if m.get("bb_buy_age") is None]

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

            df_m = pd.DataFrame([{
                "Ticker": m["ticker"],
                "Price": m["price"],
                "RSI": m["rsi"],
                "vs BB Lower": m["bb_distance_pct"],
                "BB BUY Date": m.get("bb_buy_date") or "—",
                "BB BUY Age": _fmt_age(m.get("bb_buy_age")),
                "RSI Oversold": "✓" if m["rsi_oversold"] else "·",
            } for m in matches])
            st.dataframe(
                df_m,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                    "vs BB Lower": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="Price relative to Bollinger lower band. "
                             "Negative = below lower band (oversold).",
                    ),
                    "BB BUY Date": st.column_config.TextColumn(
                        help="Date of most recent Bollinger lower-band touch in the lookback window",
                    ),
                    "BB BUY Age": st.column_config.TextColumn(
                        help="Trading days since the BUY signal — 'today' = fired on the latest bar",
                    ),
                },
            )
            st.caption(
                "Sorted by most recent BB BUY first. Add any to your watchlist via "
                "the Scan tab search box, then click the tile to see full analysis."
            )


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

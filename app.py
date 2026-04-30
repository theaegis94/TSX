"""Streamlit dashboard for the TSX signal scanner.

Run with:    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

import stock_signals as ss

st.set_page_config(page_title="Stock Signals", layout="wide", page_icon="📈")


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


def render_watchlist_bar(tickers: tuple) -> None:
    if not tickers:
        return
    quotes = cached_quotes(tickers)
    tiles_html = []
    for t in tickers:
        q = quotes.get(t)
        if not q:
            tiles_html.append(
                f'<div class="ticker-tile">'
                f'<div class="tt-code">{t}</div>'
                f'<div class="tt-price">—</div>'
                f'<div class="tt-change">—</div>'
                f'</div>'
            )
            continue
        chg = q["change_pct"]
        color = "#16a34a" if chg >= 0 else "#dc2626"
        arrow = "▲" if chg >= 0 else "▼"
        sign = "+" if chg >= 0 else ""
        tiles_html.append(
            f'<div class="ticker-tile">'
            f'<div class="tt-code">{t}</div>'
            f'<div class="tt-price">${q["price"]:.2f}</div>'
            f'<div class="tt-change" style="color:{color}">'
            f'{arrow} {sign}{chg:.2f}%</div>'
            f'</div>'
        )

    html = (
        "<style>"
        ".ticker-bar {"
        "  display: flex;"
        "  overflow-x: auto;"
        "  gap: 6px;"
        "  padding: 4px 0 12px 0;"
        "  margin-bottom: 4px;"
        "  -webkit-overflow-scrolling: touch;"
        "}"
        ".ticker-bar::-webkit-scrollbar { height: 6px; }"
        ".ticker-bar::-webkit-scrollbar-thumb {"
        "  background: #4b5563; border-radius: 3px;"
        "}"
        ".ticker-tile {"
        "  background: #1f2937;"
        "  padding: 6px 10px;"
        "  border-radius: 6px;"
        "  min-width: 105px;"
        "  flex-shrink: 0;"
        "  border: 1px solid #374151;"
        "}"
        ".tt-code {"
        "  font-size: 0.7rem;"
        "  color: #9ca3af;"
        "  font-weight: 600;"
        "  letter-spacing: 0.5px;"
        "}"
        ".tt-price {"
        "  font-size: 1rem;"
        "  color: #e5e7eb;"
        "  font-weight: 500;"
        "}"
        ".tt-change {"
        "  font-size: 0.75rem;"
        "  font-weight: 600;"
        "}"
        "</style>"
        '<div class="ticker-bar">' + "".join(tiles_html) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


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


# --------- tabs ---------

tab_scan, tab_single, tab_news, tab_help = st.tabs(
    ["📊 Scan", "🔍 Single Ticker", "📰 News", "ℹ️ Help"]
)


# === Scan tab ===
with tab_scan:
    st.subheader("Watchlist Scan")

    default_str = ", ".join(ss.DEFAULT_WATCHLIST)
    watchlist_str = st.text_area(
        "Tickers (comma-separated; bare = US, .TO = TSX, .V = TSXV)",
        default_str, height=80, key="watchlist_input",
    )
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

"""Streamlit dashboard for the TSX signal scanner.

Run with:    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

import stock_signals as ss

st.set_page_config(page_title="TSX Signals", layout="wide", page_icon="📈")


# --------- caching wrappers ---------

@st.cache_data(ttl=900, show_spinner=False)
def cached_macro() -> dict:
    return {
        "USD/CAD": ss.boc_valet("FXUSDCAD"),
        "BoC Rate": ss.boc_valet("V39079"),
        "10Y Yield": ss.boc_valet("BD.CDN.10YR.DQ.YLD"),
        "WTI Crude": ss.yf_spot("CL=F"),
        "Gold": ss.yf_spot("GC=F"),
    }


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(tickers: tuple, period: str, interval: str) -> list:
    return ss.scan(list(tickers), period, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_single(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = ss.compute_indicators(df)
    df = ss.generate_signals(df)
    stats = ss.backtest(df)
    return df, stats


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker: str, days: int = 7) -> list:
    return ss.finnhub_news(ticker, days=days)


# --------- header / macro ---------

st.title("📈 TSX Signal Dashboard")

macro = cached_macro()
cols = st.columns(5)
for col, (label, val) in zip(cols, macro.items()):
    if val is None:
        col.metric(label, "—")
    elif label.startswith("USD"):
        col.metric(label, f"{val:.4f}")
    elif label == "Gold":
        col.metric(label, f"${val:,.0f}")
    elif "Crude" in label:
        col.metric(label, f"${val:.2f}")
    else:
        col.metric(label, f"{val:.2f}%")

st.divider()


# --------- sidebar ---------

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Lookback period", ["6mo", "1y", "2y", "5y", "10y"],
                          index=2)
    interval = st.selectbox("Bar interval", ["1d", "1wk"], index=0)
    st.divider()
    if st.button("🔄 Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        f"Cached for 15 min (scan/macro/news), 1 hr (single ticker). "
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
        "Tickers (comma-separated; bare tickers auto-suffix to .TO)",
        default_str, height=80, key="watchlist_input",
    )
    tickers = tuple(t.strip() for t in watchlist_str.split(",") if t.strip())

    with st.spinner(f"Scanning {len(tickers)} tickers…"):
        rows = cached_scan(tickers, period, interval)

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
                "Win %": r["win_rate"] * 100,
                "Strat %": r["strat"] * 100,
                "B&H %": r["bh"] * 100,
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
                "Win %": st.column_config.NumberColumn(format="%.0f"),
                "Strat %": st.column_config.NumberColumn(format="%+.1f"),
                "B&H %": st.column_config.NumberColumn(format="%+.1f"),
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
                df, stats = cached_single(ticker, period, interval)

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

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", stats["trades"])
                c2.metric("Win Rate", f"{stats['win_rate']:.0%}")
                c3.metric("Strategy", f"{stats['total_return']:+.1%}")
                c4.metric("Buy & Hold", f"{stats['buy_hold']:+.1%}",
                          delta=f"{(stats['total_return']-stats['buy_hold'])*100:+.1f}%")

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
- Bare tickers auto-suffix to `.TO`
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

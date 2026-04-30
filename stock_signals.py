"""
Stock & ETF buy/sell signal indicator (TSX + US + international).

Pulls OHLC data with yfinance, computes RSI + MACD + SMA(50/200),
generates buy/sell signals where multiple indicators agree, plots
them on a chart, and runs a basic long-only backtest.

Ticker conventions:
    Bare tickers (AAPL, SPY) are treated as US listings.
    .TO / .V / .CN suffixes are TSX / TSXV / CSE.
    Other suffixes (.L London, .HK Hong Kong, .DE Germany, ...) are
    passed through to Yahoo as-is.
    Class shares are auto-converted: BRK.B -> BRK-B.

Usage:
    python stock_signals.py                  # defaults to XIC.TO, 2y
    python stock_signals.py AAPL             # US ticker
    python stock_signals.py RY.TO 5y         # TSX, 5y backtest
    python stock_signals.py SHOP.TO 5y 1d    # + interval
    python stock_signals.py BRK.B            # class share -> BRK-B
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


def load_env(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.
    Existing env vars take precedence; unquoted values supported."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_env()


def _read_finnhub_key() -> str:
    """Read FINNHUB_API_KEY from env, then fall back to st.secrets (cloud deploy)."""
    val = os.environ.get("FINNHUB_API_KEY", "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get("FINNHUB_API_KEY", "")
    except (ImportError, FileNotFoundError, KeyError, AttributeError):
        return ""


FINNHUB_API_KEY = _read_finnhub_key()


def yf_metrics(ticker: str) -> dict:
    """P/E, yield, beta, analyst upside, days to earnings — from yfinance .info.

    Free, no API key, full TSX coverage including ETFs.
    Returns dict with optional keys: pe, yield_pct, beta, upside_pct, earn_days.
    Returns {} on any error (rate limit, network, parse) so the scan continues.
    """
    out: dict = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        # Broad catch: yfinance has many custom exception types (YFRateLimitError,
        # YFTickerMissingError, etc.) plus network/parse errors. We never want
        # an info fetch to break the scan.
        return out

    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe is not None:
        try:
            out["pe"] = float(pe)
        except (ValueError, TypeError):
            pass

    div_yield = info.get("dividendYield")
    if div_yield is not None:
        try:
            out["yield_pct"] = float(div_yield)
        except (ValueError, TypeError):
            pass

    beta = info.get("beta")
    if beta is not None:
        try:
            out["beta"] = float(beta)
        except (ValueError, TypeError):
            pass

    target = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and current:
        try:
            t, c = float(target), float(current)
            if c > 0:
                out["upside_pct"] = (t - c) / c * 100
        except (ValueError, TypeError):
            pass

    ed_raw = info.get("earningsDate") or info.get("earningsTimestamp")
    if ed_raw:
        try:
            ts = ed_raw[0] if isinstance(ed_raw, (list, tuple)) and ed_raw else ed_raw
            ed = datetime.fromtimestamp(int(ts)).date()
            today = datetime.now().date()
            if ed >= today:
                out["earn_days"] = (ed - today).days
        except (ValueError, TypeError, OSError, IndexError):
            pass

    return out


def boc_valet(series: str) -> float | None:
    """Latest observation from a Bank of Canada Valet series. Free, no key."""
    try:
        r = requests.get(
            f"https://www.bankofcanada.ca/valet/observations/{series}/json",
            params={"recent": 1},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        if not obs:
            return None
        for key, val in obs[-1].items():
            if key == "d" or not isinstance(val, dict):
                continue
            try:
                return float(val.get("v"))
            except (ValueError, TypeError):
                pass
        return None
    except (requests.RequestException, ValueError):
        return None


def _quote_from_df(df, t: str) -> dict | None:
    try:
        if isinstance(df.columns, pd.MultiIndex) and t in df.columns.get_level_values(0):
            tdf = df[t]
        elif "Close" in df.columns:
            tdf = df
        else:
            return None
        closes = tdf["Close"].dropna()
        if len(closes) < 2:
            return None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        return {
            "price": last,
            "prev": prev,
            "change_pct": (last - prev) / prev * 100 if prev else 0.0,
        }
    except (KeyError, AttributeError, ValueError, IndexError):
        return None


def screen_buy_signals(tickers: list[str], rsi_threshold: float = 35.0,
                       lookback_bars: int = 22,
                       require_bollinger: bool = True,
                       require_rsi: bool = True,
                       batch_size: int = 100,
                       progress_callback=None) -> list[dict]:
    """Find tickers with confluence buy signals: Bollinger lower-band BUY + RSI oversold.

    rsi_threshold: RSI must be at or below this value (default 35 — relaxed oversold).
    lookback_bars: a Bollinger BUY anywhere in the last N trading days counts
                   (default 22 ≈ 1 calendar month).
    require_bollinger / require_rsi: toggle each filter independently.

    Returns list of dicts including:
      ticker, price, rsi, bb_lower, bb_distance_pct,
      bollinger_buy (bool, was there a BUY in the lookback window?),
      rsi_oversold (bool, current RSI <= threshold),
      bb_buy_date (date of the most recent BUY in the window, or None),
      bb_buy_age (trading-days ago, or None).
    Sorted by most recent BB BUY first, then RSI ascending.
    """
    if not tickers or not (require_bollinger or require_rsi):
        return []

    # Deduplicate and chunk into batches (Yahoo handles ~100 tickers per call well)
    unique = list(dict.fromkeys(tickers))
    batches = [unique[i:i + batch_size] for i in range(0, len(unique), batch_size)]

    matches: list[dict] = []
    for batch_idx, batch in enumerate(batches):
        try:
            df = yf.download(
                " ".join(batch),
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception:
            df = None

        if df is not None and not df.empty:
            matches.extend(_screen_batch(
                df, batch, rsi_threshold, lookback_bars,
                require_bollinger, require_rsi,
            ))

        if progress_callback is not None:
            progress_callback((batch_idx + 1) / len(batches), len(matches))

    matches.sort(key=lambda r: (
        r["bb_buy_age"] if r["bb_buy_age"] is not None else 9999,
        r["rsi"],
    ))
    return matches


def _screen_batch(df, tickers, rsi_threshold, lookback_bars,
                  require_bollinger, require_rsi) -> list[dict]:
    """Process one batch of yf.download output into match rows."""
    matches: list[dict] = []
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex) and t in df.columns.get_level_values(0):
                tdf = df[t].copy()
            elif "Close" in df.columns:
                tdf = df.copy()
            else:
                continue
            if len(tdf.dropna(subset=["Close"])) < 30:
                continue
            tdf = compute_indicators(tdf)
            if "BB_LOWER" not in tdf.columns or "RSI" not in tdf.columns:
                continue
            bb_sig = _strategy_bollinger(tdf)
            window = bb_sig.iloc[-lookback_bars:]
            buy_indices = window.index[window["BUY"]]
            bollinger_buy = len(buy_indices) > 0
            bb_buy_date = buy_indices[-1] if bollinger_buy else None
            bb_buy_age = (
                int((tdf.index[-1] - bb_buy_date).days)
                if bb_buy_date is not None else None
            )

            last = tdf.iloc[-1]
            if pd.isna(last["RSI"]) or pd.isna(last["Close"]) or pd.isna(last["BB_LOWER"]):
                continue
            rsi_val = float(last["RSI"])
            close_val = float(last["Close"])
            bb_lo = float(last["BB_LOWER"])
            rsi_oversold = rsi_val <= rsi_threshold

            if require_bollinger and not bollinger_buy:
                continue
            if require_rsi and not rsi_oversold:
                continue

            matches.append({
                "ticker": t,
                "price": close_val,
                "rsi": rsi_val,
                "bb_lower": bb_lo,
                "bb_distance_pct": (close_val - bb_lo) / bb_lo * 100 if bb_lo else 0.0,
                "bollinger_buy": bollinger_buy,
                "rsi_oversold": rsi_oversold,
                "bb_buy_date": bb_buy_date.date().isoformat() if bb_buy_date is not None else None,
                "bb_buy_age": bb_buy_age,
            })
        except (KeyError, AttributeError, ValueError, IndexError, TypeError):
            continue
    return matches


def fetch_watchlist_quotes(tickers: list[str]) -> dict:
    """Latest price + day change for a list of tickers, in ONE batched yf.download
    call (rate-limit-friendly). Returns {ticker: {price, prev, change_pct}}.

    Bare tickers that return no data are retried with .TO (TSX-only names like HOD).
    """
    if not tickers:
        return {}
    try:
        df = yf.download(
            " ".join(tickers),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return {}

    out: dict = {}
    missing_bare: list[str] = []
    if df is not None and not df.empty:
        for t in tickers:
            q = _quote_from_df(df, t)
            if q is not None:
                out[t] = q
            elif "." not in t:
                # Bare ticker with no data — likely a TSX-only name (e.g. HOD)
                missing_bare.append(t)

    # Retry missing bare tickers with .TO suffix
    if missing_bare:
        try:
            df2 = yf.download(
                " ".join(f"{t}.TO" for t in missing_bare),
                period="5d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
        except Exception:
            df2 = None
        if df2 is not None and not df2.empty:
            for t in missing_bare:
                q = _quote_from_df(df2, f"{t}.TO")
                if q is not None:
                    out[t] = q  # store under original key so the UI matches
    return out


def yf_spot(symbol: str) -> float | None:
    try:
        df = yf.download(symbol, period="2d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def print_macro_header() -> None:
    """One-line macro context: USD/CAD, BoC overnight rate, 10y yield, WTI, gold."""
    cad_usd = boc_valet("FXUSDCAD")
    boc_rate = boc_valet("V39079")
    yield_10y = boc_valet("BD.CDN.10YR.DQ.YLD")
    wti = yf_spot("CL=F")
    gold = yf_spot("GC=F")

    parts = []
    if cad_usd is not None:
        parts.append(f"USD/CAD {cad_usd:.4f}")
    if boc_rate is not None:
        parts.append(f"BoC {boc_rate:.2f}%")
    if yield_10y is not None:
        parts.append(f"10Y {yield_10y:.2f}%")
    if wti is not None:
        parts.append(f"WTI ${wti:.2f}")
    if gold is not None:
        parts.append(f"Gold ${gold:.0f}")
    if parts:
        print("Macro: " + "  |  ".join(parts) + "\n")


def _finnhub_sym(ticker: str) -> str:
    """Strip TSX suffixes — Finnhub free tier needs bare ticker (e.g. 'RY' not 'RY.TO')."""
    for suffix in (".TO", ".V", ".CN"):
        if ticker.upper().endswith(suffix):
            return ticker[: -len(suffix)].upper()
    return ticker.upper()


def finnhub_search(query: str, limit: int = 15) -> list[dict]:
    """Search tickers by symbol or company name. Returns list of
    {symbol, display_symbol, description, type}."""
    if not FINNHUB_API_KEY or not query.strip():
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/search",
            params={"q": query.strip(), "token": FINNHUB_API_KEY},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json() or {}
        seen: set = set()
        out: list[dict] = []
        for row in (data.get("result") or []):
            sym = (row.get("displaySymbol") or row.get("symbol") or "").strip().upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append({
                "symbol": sym,
                "description": row.get("description", "").strip(),
                "type": row.get("type", "").strip(),
            })
            if len(out) >= limit:
                break
        return out
    except (requests.RequestException, ValueError):
        return []


def finnhub_news(ticker: str, days: int = 7) -> list[dict]:
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().date()
        frm = (today - timedelta(days=days)).isoformat()
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": _finnhub_sym(ticker), "from": frm,
                    "to": today.isoformat(), "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return r.json() or []
    except (requests.RequestException, ValueError):
        return []


def finnhub_sentiment(ticker: str) -> dict | None:
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news-sentiment",
            params={"symbol": _finnhub_sym(ticker), "token": FINNHUB_API_KEY},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return r.json() or None
    except (requests.RequestException, ValueError):
        return None


def print_news(ticker: str, limit: int = 15) -> None:
    sent = finnhub_sentiment(ticker)
    news = finnhub_news(ticker, days=7)

    print(f"\n=== {ticker} — last 7 days ===")
    if sent:
        buzz = sent.get("buzz") or {}
        if buzz:
            print(
                f"Buzz: {buzz.get('buzz', 0):.2f}  |  "
                f"Articles last week: {buzz.get('articlesInLastWeek', 0)}  |  "
                f"Weekly avg: {buzz.get('weeklyAverage', 0):.2f}"
            )
        score = sent.get("companyNewsScore")
        bull = sent.get("bullishPercent")
        sector = sent.get("sectorAverageBullishPercent")
        if score is not None:
            line = f"News score: {score:.2f}"
            if bull is not None:
                line += f"  |  bullish % {bull*100:.0f}"
            if sector is not None:
                line += f"  vs sector {sector*100:.0f}"
            print(line)

    if not news:
        print("\nNo recent news.")
        return

    print(f"\n{len(news)} articles found, showing latest {min(limit, len(news))}:\n")
    for art in news[:limit]:
        try:
            ts = datetime.fromtimestamp(art.get("datetime", 0))
            date_str = ts.strftime("%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            date_str = "?"
        source = (art.get("source", "") or "")[:18]
        headline = art.get("headline", "")
        url = art.get("url", "")
        print(f"  [{date_str}] {source:<18}  {headline}")
        if url:
            print(f"             {url}")


def finnhub_recommendation(ticker: str) -> tuple[int, int, int] | None:
    """(buys, holds, sells) from the most recent analyst rec snapshot, or None.

    Buys = strongBuy + buy; sells = strongSell + sell.
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": _finnhub_sym(ticker), "token": FINNHUB_API_KEY},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json() or []
        if not data:
            return None
        latest = data[0]
        b = int(latest.get("strongBuy", 0)) + int(latest.get("buy", 0))
        h = int(latest.get("hold", 0))
        s = int(latest.get("strongSell", 0)) + int(latest.get("sell", 0))
        if b + h + s == 0:
            return None
        return (b, h, s)
    except (requests.RequestException, ValueError):
        return None


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — values >25 indicate a trending market."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = series.rolling(period).mean()
    dev = series.rolling(period).std()
    return mid - std * dev, mid, mid + std * dev


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["RSI"] = rsi(out["Close"])
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["BB_LOWER"], out["BB_MID"], out["BB_UPPER"] = bollinger(out["Close"])
    out["DC_HIGH"] = out["High"].rolling(20).max()
    out["DC_LOW"] = out["Low"].rolling(20).min()
    if {"High", "Low", "Close"}.issubset(out.columns):
        out["ADX"] = adx(out)
    return out


# === Strategy registry ===

STRATEGY_LABELS = {
    "trend": "Trend (RSI+MACD+SMA, default)",
    "bollinger": "Bollinger Mean Reversion",
    "donchian": "Donchian 20d Breakout",
    "sma200_dip": "SMA200 Dip Buy",
}


def _strategy_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Original confirmation-based trend strategy."""
    out = df.copy()
    window = 5

    rsi_up = ((out["RSI"].shift(1) < 30) & (out["RSI"] >= 30)).astype(int)
    rsi_dn = ((out["RSI"].shift(1) > 70) & (out["RSI"] <= 70)).astype(int)

    macd_up = (
        (out["MACD"].shift(1) < out["MACD_SIGNAL"].shift(1))
        & (out["MACD"] >= out["MACD_SIGNAL"])
    ).astype(int)
    macd_dn = (
        (out["MACD"].shift(1) > out["MACD_SIGNAL"].shift(1))
        & (out["MACD"] <= out["MACD_SIGNAL"])
    ).astype(int)

    sma_up = (
        (out["SMA50"].shift(1) < out["SMA200"].shift(1))
        & (out["SMA50"] >= out["SMA200"])
    ).astype(int)
    sma_dn = (
        (out["SMA50"].shift(1) > out["SMA200"].shift(1))
        & (out["SMA50"] <= out["SMA200"])
    ).astype(int)

    bull = (rsi_up.rolling(window).max().fillna(0)
            + macd_up.rolling(window).max().fillna(0)
            + sma_up.rolling(window).max().fillna(0))
    bear = (rsi_dn.rolling(window).max().fillna(0)
            + macd_dn.rolling(window).max().fillna(0)
            + sma_dn.rolling(window).max().fillna(0))

    out["SCORE"] = bull - bear
    out["BUY"] = (out["SCORE"] >= 2) & (out["SCORE"].shift(1) < 2)
    out["SELL"] = (out["SCORE"] <= -2) & (out["SCORE"].shift(1) > -2)
    return out


def _strategy_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reversion: buy at lower band touch, sell at upper band touch."""
    out = df.copy()
    out["BUY"] = (
        (out["Close"].shift(1) > out["BB_LOWER"].shift(1))
        & (out["Close"] <= out["BB_LOWER"])
    )
    out["SELL"] = (
        (out["Close"].shift(1) < out["BB_UPPER"].shift(1))
        & (out["Close"] >= out["BB_UPPER"])
    )
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_donchian(df: pd.DataFrame) -> pd.DataFrame:
    """Breakout: buy on 20d high break, sell on 20d low break (compares to prior day's channel)."""
    out = df.copy()
    out["BUY"] = (
        (out["Close"] > out["DC_HIGH"].shift(1))
        & (out["Close"].shift(1) <= out["DC_HIGH"].shift(2))
    )
    out["SELL"] = (
        (out["Close"] < out["DC_LOW"].shift(1))
        & (out["Close"].shift(1) >= out["DC_LOW"].shift(2))
    )
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_sma200_dip(df: pd.DataFrame, dip: float = 0.05) -> pd.DataFrame:
    """In a long-term uptrend (rising SMA200), buy 5%+ dips below SMA200, sell on SMA50 recovery."""
    out = df.copy()
    in_uptrend = out["SMA200"] > out["SMA200"].shift(20)
    dip_threshold = out["SMA200"] * (1 - dip)
    out["BUY"] = (
        in_uptrend
        & (out["Close"] < dip_threshold)
        & (out["Close"].shift(1) >= dip_threshold.shift(1))
    )
    out["SELL"] = (
        (out["Close"] >= out["SMA50"])
        & (out["Close"].shift(1) < out["SMA50"].shift(1))
    )
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


_STRATEGIES = {
    "trend": _strategy_trend,
    "bollinger": _strategy_bollinger,
    "donchian": _strategy_donchian,
    "sma200_dip": _strategy_sma200_dip,
}


def generate_signals(df: pd.DataFrame, strategy: str = "trend",
                     adx_filter: bool = False,
                     adx_threshold: float = 25.0) -> pd.DataFrame:
    """Generate BUY/SELL/SCORE columns for the chosen strategy.

    If adx_filter=True, signals are zeroed out where ADX < adx_threshold
    (suppresses signals in choppy/range-bound markets)."""
    fn = _STRATEGIES.get(strategy, _strategy_trend)
    out = fn(df)
    if adx_filter and "ADX" in out.columns:
        weak = out["ADX"] < adx_threshold
        out.loc[weak, "BUY"] = False
        out.loc[weak, "SELL"] = False
    return out


def backtest(df: pd.DataFrame, stop_loss_pct: float | None = None) -> dict:
    """Long-only: enter on BUY close, exit on SELL close (or stop). Cash otherwise.

    stop_loss_pct: e.g. 0.07 = exit if drawdown from entry hits -7%. None disables.
    Returns trades, win_rate, total_return, buy_hold, max_drawdown, stops_hit.
    """
    bh = (float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)
          if len(df) >= 2 else 0.0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    position = 0
    entry = 0.0
    entry_ts = None
    trades = []

    for ts, row in df.iterrows():
        price = float(row["Close"])

        # Mark-to-market equity for drawdown tracking
        if position == 1 and entry > 0:
            cur_equity = equity * (price / entry)
        else:
            cur_equity = equity
        peak = max(peak, cur_equity)
        if peak > 0:
            dd = (cur_equity - peak) / peak
            if dd < max_dd:
                max_dd = dd

        if position == 0 and bool(row["BUY"]):
            position = 1
            entry = price
            entry_ts = ts
            continue

        if position == 1:
            stopped = False
            if stop_loss_pct is not None:
                stop_price = entry * (1 - stop_loss_pct)
                if price <= stop_price:
                    stopped = True

            if stopped or bool(row["SELL"]):
                ret = (price - entry) / entry
                trades.append({"entry": entry_ts, "exit": ts,
                               "return": ret, "stopped": stopped})
                equity *= (1 + ret)
                position = 0

    if position == 1:
        last_price = float(df["Close"].iloc[-1])
        ret = (last_price - entry) / entry
        trades.append({"entry": entry_ts, "exit": df.index[-1],
                       "return": ret, "open": True, "stopped": False})
        equity *= (1 + ret)

    stops_hit = sum(1 for t in trades if t.get("stopped"))

    if not trades:
        return {"trades": 0, "win_rate": 0.0, "total_return": 0.0,
                "buy_hold": bh, "max_drawdown": max_dd, "stops_hit": 0}

    rets = np.array([t["return"] for t in trades])
    total = float(equity - 1)
    win_rate = float((rets > 0).mean())
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_return": total,
        "buy_hold": bh,
        "max_drawdown": max_dd,
        "stops_hit": stops_hit,
        "detail": trades,
    }


def build_chart(df: pd.DataFrame, ticker: str, stats: dict, compact: bool = False):
    """Build the 3-panel matplotlib figure (price/RSI/MACD) and return it.
    Caller is responsible for showing/saving/closing.

    compact=True returns a smaller figure suitable for embedded popups.
    """
    figsize = (10, 5.5) if compact else (14, 10)
    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax_price.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.2)
    ax_price.plot(df.index, df["SMA50"], label="SMA50", color="tab:blue", linewidth=1)
    ax_price.plot(df.index, df["SMA200"], label="SMA200", color="tab:orange", linewidth=1)

    buys = df[df["BUY"]]
    sells = df[df["SELL"]]
    ax_price.scatter(buys.index, buys["Close"], marker="^", s=120,
                     color="green", label="BUY", zorder=5, edgecolors="black")
    ax_price.scatter(sells.index, sells["Close"], marker="v", s=120,
                     color="red", label="SELL", zorder=5, edgecolors="black")

    ax_price.set_title(
        f"{ticker} — signals: {stats['trades']} trades | "
        f"win rate {stats['win_rate']:.0%} | "
        f"strategy {stats['total_return']:+.1%} vs buy&hold {stats['buy_hold']:+.1%}"
    )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_rsi.plot(df.index, df["RSI"], color="purple", linewidth=1)
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.grid(alpha=0.3)

    ax_macd.plot(df.index, df["MACD"], label="MACD", color="tab:blue", linewidth=1)
    ax_macd.plot(df.index, df["MACD_SIGNAL"], label="Signal",
                 color="tab:orange", linewidth=1)
    colors = ["green" if v >= 0 else "red" for v in df["MACD_HIST"].fillna(0)]
    ax_macd.bar(df.index, df["MACD_HIST"], color=colors, alpha=0.4, width=1.0)
    ax_macd.axhline(0, color="black", linewidth=0.5)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")
    ax_macd.grid(alpha=0.3)

    ax_macd.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot(df: pd.DataFrame, ticker: str, stats: dict, show: bool = True) -> None:
    fig = build_chart(df, ticker, stats)
    out_path = f"{ticker}_signals.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved chart: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


_EXCHANGE_SUFFIXES = {
    # North America
    "TO", "V", "CN", "NE",
    # Europe
    "L", "PA", "DE", "MI", "AS", "BR", "MC", "ST", "SW", "VI", "WA", "IS", "F",
    # Asia-Pacific
    "HK", "T", "SS", "SZ", "KS", "KQ", "NS", "BO", "AX", "NZ", "SI", "BK",
    # Latin America / Africa / Middle East
    "MX", "SA", "BA", "SN", "JO", "TA", "CR", "IR",
    # Other
    "TSX",  # rewritten to .TO upstream
}


def normalize_ticker(raw: str) -> str:
    """Normalize a ticker symbol for Yahoo Finance.

    - .TSX suffix is rewritten to .TO.
    - .TO / .V / .CN are TSX/TSXV/CSE — accepted as-is.
    - Bare tickers (e.g. AAPL, MSFT) are treated as US listings.
    - Known exchange suffixes (.L London, .HK Hong Kong, etc.) pass through.
    - Class-share dots (BRK.B) become hyphens for Yahoo: BRK-B.
    """
    t = raw.strip().upper()
    if not t:
        raise SystemExit("Empty ticker.")
    if t.endswith(".TSX"):
        return t[:-4] + ".TO"
    if "." in t:
        prefix, _, last = t.rpartition(".")
        if last in _EXCHANGE_SUFFIXES:
            return t
        # Class share — Yahoo expects a hyphen (BRK.B -> BRK-B).
        return f"{prefix}-{last}"
    return t


# Back-compat alias (older code paths called the TSX-only name).
normalize_tsx_ticker = normalize_ticker


# --- Universe lists for the screener ---
# Larger lists fetched from Wikipedia (cached for a week). Hardcoded fallbacks
# in case Wikipedia is unreachable from the deploy environment.

_SP500_FALLBACK_NOTE = "(fetch from Wikipedia failed — using a smaller fallback list)"


_WIKI_UA = "Mozilla/5.0 (compatible; SignalDashboard/1.0)"


def _wiki_tables(url: str) -> list:
    """Fetch a Wikipedia page with a proper UA, then parse all tables."""
    import io
    resp = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_sp500() -> list[str]:
    """Live S&P 500 tickers from Wikipedia. Yahoo-format (BRK.B -> BRK-B)."""
    try:
        tables = _wiki_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0]
        symbols = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
        symbols = [s for s in symbols if s and s.replace("-", "").isalnum()]
        return symbols
    except Exception:
        return UNIVERSE_SP100


def get_tsx_composite() -> list[str]:
    """Live S&P/TSX Composite tickers from Wikipedia, suffixed .TO."""
    try:
        tables = _wiki_tables(
            "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        )
        # The constituent table column may be named "Symbol" or "Ticker"
        for t in tables:
            cols_lc = [str(c).strip().lower() for c in t.columns]
            sym_col = None
            for cand in ("symbol", "ticker"):
                if cand in cols_lc:
                    sym_col = t.columns[cols_lc.index(cand)]
                    break
            if sym_col is None:
                continue
            symbols = t[sym_col].fillna("").astype(str).str.upper().str.strip().tolist()
            cleaned = []
            for s in symbols:
                if not isinstance(s, str) or not s or s in ("NAN", "NONE"):
                    continue
                if s.endswith((".A", ".B", ".U", ".UN")):
                    s = s.replace(".", "-")
                if len(s) > 12 or "—" in s:
                    continue
                if not s.endswith(".TO") and "." not in s:
                    s = f"{s}.TO"
                cleaned.append(s)
            if len(cleaned) > 100:  # plausible composite size
                return cleaned
        raise ValueError("no constituent table found")
    except Exception:
        return UNIVERSE_TSX60


# Curated popular ETFs from US + Canadian markets
UNIVERSE_POPULAR_ETFS = [
    # US broad market
    "SPY", "VOO", "IVV", "VTI", "ITOT", "SPLG",
    # US sectors
    "QQQ", "QQQM", "DIA", "IWM", "MDY", "VTV", "VUG", "IWF", "IWD",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    # US fixed income
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "MUB", "TIP",
    # US international / EM
    "VXUS", "EFA", "VEA", "EEM", "VWO", "IEMG",
    # US commodities / alts
    "GLD", "SLV", "GDX", "USO", "UNG", "DBC",
    # US thematic
    "ARKK", "ARKG", "SOXX", "SMH", "XBI", "IBB", "XOP", "TAN", "ICLN", "JETS", "KWEB",
    # CA broad market
    "XIC.TO", "XIU.TO", "VCN.TO", "ZCN.TO", "XEQT.TO", "VEQT.TO", "HEQT.TO",
    # CA US-exposure
    "VFV.TO", "ZSP.TO", "HXS.TO", "XSP.TO", "VSP.TO",
    # CA NASDAQ
    "QQC.TO", "ZQQ.TO", "HXQ.TO",
    # CA sector / thematic
    "XEG.TO", "XFN.TO", "XGD.TO", "XIT.TO", "XMA.TO",
    # CA fixed income
    "XBB.TO", "ZAG.TO", "VAB.TO", "XSB.TO",
    # CA dividend
    "VDY.TO", "XEI.TO", "CDZ.TO", "ZDV.TO",
    # CA inverse / leveraged
    "HOD.TO", "HOU.TO", "HXD.TO", "HXU.TO",
]


# Predefined universes for the screener
UNIVERSE_SP100 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK-B",
    "LLY", "AVGO", "JPM", "V", "WMT", "XOM", "MA", "UNH", "JNJ", "PG", "ORCL",
    "HD", "COST", "ABBV", "BAC", "MRK", "CVX", "KO", "PEP", "ADBE", "NFLX",
    "CRM", "TMO", "AMD", "ACN", "MCD", "LIN", "DIS", "ABT", "WFC", "CSCO",
    "TXN", "DHR", "VZ", "INTU", "AMGN", "CAT", "PM", "PFE", "IBM", "GE",
    "AXP", "QCOM", "ISRG", "NOW", "RTX", "BX", "GS", "T", "NEE", "MS",
    "UBER", "LOW", "BKNG", "SPGI", "UNP", "BA", "C", "ELV", "TJX", "PGR",
    "BLK", "MDT", "GILD", "SYK", "VRTX", "MMC", "ADP", "PLD", "DE", "BSX",
    "ETN", "LMT", "MDLZ", "SCHW", "AMT", "ADI", "REGN", "FI", "MO", "PANW",
    "INTC", "BMY", "SO", "DUK", "CB", "CL", "EOG", "TGT", "USB", "MU",
]

UNIVERSE_TSX60 = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",  # banks
    "SHOP.TO", "CSU.TO", "OTEX.TO", "CGI.TO",  # tech
    "ENB.TO", "TRP.TO", "SU.TO", "CNQ.TO", "IMO.TO", "CVE.TO", "TOU.TO",
    "PPL.TO", "ARX.TO", "TPZ.TO",  # energy
    "BCE.TO", "T.TO", "RCI-B.TO",  # telecom
    "CNR.TO", "CP.TO",  # rail
    "FTS.TO", "EMA.TO", "ALA.TO", "H.TO", "AQN.TO",  # utilities
    "NTR.TO", "ABX.TO", "AEM.TO", "TECK-B.TO", "FNV.TO", "WPM.TO",  # materials
    "WCN.TO", "GIB-A.TO",  # services
    "ATD.TO", "L.TO", "MRU.TO", "WN.TO", "DOL.TO",  # consumer
    "SLF.TO", "MFC.TO", "GWO.TO", "POW.TO", "IFC.TO", "FFH.TO",  # insurance
    "BAM.TO", "BN.TO", "BIP-UN.TO", "BEP-UN.TO",  # alt asset mgrs
    "MG.TO", "CCL-B.TO", "WSP.TO", "STN.TO",  # industrial
    "QSR.TO", "GIL.TO",  # consumer cyc
    "REI-UN.TO", "CAR-UN.TO", "SRU-UN.TO",  # REITs
]


DEFAULT_WATCHLIST = [
    # Broad CA ETFs
    "XIC.TO", "XIU.TO", "XEQT.TO", "VFV.TO", "ZSP.TO", "HXT.TO",
    # CA banks
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    # CA tech / energy / telecom / rail / utilities
    "SHOP.TO", "CSU.TO", "ENB.TO", "TRP.TO", "SU.TO", "CNQ.TO",
    "BCE.TO", "T.TO", "CNR.TO", "CP.TO", "FTS.TO",
    # US benchmarks
    "SPY", "QQQ", "DIA",
    # US mega-caps for context
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
]


def scan(tickers: list[str], period: str, interval: str,
         strategy: str = "trend", adx_filter: bool = False,
         stop_loss_pct: float | None = None,
         metrics_fn=None) -> list[dict]:
    """metrics_fn: optional override for fundamentals lookup (defaults to yf_metrics).
    Allows the Streamlit layer to inject its own cached version."""
    metrics_fn = metrics_fn or yf_metrics
    rows = []
    for raw in tickers:
        try:
            ticker = normalize_tsx_ticker(raw)
        except SystemExit as e:
            rows.append({"ticker": raw, "error": str(e)})
            continue

        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 200:
            rows.append({
                "ticker": ticker,
                "error": "no data" if df.empty else f"only {len(df)} bars (<200)",
            })
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = compute_indicators(df)
        df = generate_signals(df, strategy=strategy, adx_filter=adx_filter)
        stats = backtest(df, stop_loss_pct=stop_loss_pct)

        last = df.iloc[-1]
        score = float(last["SCORE"])
        if bool(last["BUY"]):
            action = "BUY"
        elif bool(last["SELL"]):
            action = "SELL"
        else:
            action = "HOLD"

        rows.append({
            "ticker": ticker,
            "close": float(last["Close"]),
            "rsi": float(last["RSI"]),
            "adx": float(last["ADX"]) if "ADX" in df.columns and pd.notna(last.get("ADX")) else None,
            "score": score,
            "action": action,
            "trades": stats["trades"],
            "win_rate": stats["win_rate"],
            "strat": stats["total_return"],
            "bh": stats["buy_hold"],
            "max_dd": stats["max_drawdown"],
            "stops": stats.get("stops_hit", 0),
            "rec": finnhub_recommendation(ticker),
            **metrics_fn(ticker),
        })
    return rows


def print_scan_table(rows: list[dict], color: bool = False) -> None:
    rank = {"BUY": 0, "SELL": 1, "HOLD": 2}
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    ok.sort(key=lambda r: (rank.get(r["action"], 9), -r["score"]))

    show_rec = bool(FINNHUB_API_KEY)

    header = (
        f"{'TICKER':<10}{'CLOSE':>10}{'RSI':>7}{'SCORE':>7}"
        f"{'ACTION':>8}{'TRD':>5}{'WIN%':>7}{'STRAT':>9}{'B&H':>9}"
        f"{'P/E':>7}{'YLD%':>7}{'BETA':>6}{'UP%':>7}{'EARN':>7}"
    )
    if show_rec:
        header += f"{'B/H/S':>11}"
    print(header)
    print("-" * len(header))
    for r in ok:
        action = f"{r['action']:>8}"
        if color and r["action"] == "BUY":
            action = f"\033[92m{action}\033[0m"
        elif color and r["action"] == "SELL":
            action = f"\033[91m{action}\033[0m"
        line = (
            f"{r['ticker']:<10}{r['close']:>10.2f}{r['rsi']:>7.1f}"
            f"{r['score']:>+7.0f}{action}{r['trades']:>5}"
            f"{r['win_rate']*100:>6.0f}%{r['strat']*100:>+8.1f}%"
            f"{r['bh']*100:>+8.1f}%"
        )
        pe = r.get("pe")
        line += f"{pe:>7.1f}" if pe is not None else f"{'-':>7}"
        yld = r.get("yield_pct")
        line += f"{yld:>6.2f}%" if yld is not None else f"{'-':>7}"
        beta = r.get("beta")
        line += f"{beta:>6.2f}" if beta is not None else f"{'-':>6}"
        up = r.get("upside_pct")
        line += f"{up:>+6.1f}%" if up is not None else f"{'-':>7}"
        ed = r.get("earn_days")
        line += f"{(str(ed) + 'd') if ed is not None else '-':>7}"
        if show_rec:
            rec = r.get("rec")
            rec_str = f"{rec[0]}/{rec[1]}/{rec[2]}" if rec else "—"
            line += f"{rec_str:>11}"
        print(line)

    if bad:
        print()
        print("Skipped:")
        for r in bad:
            print(f"  {r['ticker']:<10}  {r['error']}")


def run_scan(tickers_arg: str | None, period: str, interval: str,
             color: bool = False) -> None:
    if tickers_arg:
        tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]
    else:
        tickers = DEFAULT_WATCHLIST
    print_macro_header()
    print(f"Scanning {len(tickers)} TSX tickers ({period}, {interval})…\n")
    rows = scan(tickers, period, interval)
    print_scan_table(rows, color=color)


def debug_keys() -> None:
    """Probe FMP and Finnhub with universal + TSX-specific calls."""
    def probe(label: str, url: str, params: dict) -> None:
        print(f"\n--- {label} ---")
        print(f"URL: {url}")
        try:
            r = requests.get(url, params=params, timeout=10)
        except requests.RequestException as e:
            print(f"  network error: {e}")
            return
        print(f"  HTTP {r.status_code}  ({len(r.content)} bytes)")
        body = r.text[:300].replace("\n", " ").replace("\r", "")
        print(f"  body[:300]: {body}")
        if r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f"  [OK] parsed as list, {len(data)} items")
                elif isinstance(data, dict):
                    keys = list(data.keys())[:8]
                    print(f"  [OK] parsed as dict, keys: {keys}")
            except ValueError:
                print("  [FAIL] body wasn't JSON")
        elif r.status_code == 401:
            print("  [FAIL] key rejected -- verify the value, email "
                  "confirmation, and that you copied the right key")
        elif r.status_code == 403:
            print("  [FAIL] forbidden -- endpoint may require a paid tier")
        elif r.status_code == 429:
            print("  [FAIL] rate-limited -- wait a minute and retry")

    def inspect(name: str, key: str) -> None:
        print(f"\n{name}:")
        print(f"  str len:    {len(key)}")
        print(f"  byte len:   {len(key.encode('utf-8'))}")
        non_print = [(i, c, hex(ord(c))) for i, c in enumerate(key)
                     if not c.isprintable()]
        if non_print:
            print(f"  WARNING: non-printable chars: {non_print}")
        else:
            print("  no non-printable chars")
        non_ascii = [(i, c, hex(ord(c))) for i, c in enumerate(key)
                     if ord(c) > 127]
        if non_ascii:
            print(f"  WARNING: non-ASCII chars: {non_ascii}")
        else:
            print("  all ASCII")
        all_alnum = all(c.isalnum() for c in key)
        print(f"  all alphanumeric: {all_alnum}")
        if key:
            head = key[:4]
            tail = key[-4:]
            head_hex = " ".join(f"{ord(c):02x}" for c in head)
            tail_hex = " ".join(f"{ord(c):02x}" for c in tail)
            print(f"  first 4 chars: {head!r}  hex: {head_hex}")
            print(f"  last 4 chars:  {tail!r}  hex: {tail_hex}")

    print("=" * 60)
    print("API KEY DIAGNOSTICS")
    print("=" * 60)
    print(f"Finnhub key present: {bool(FINNHUB_API_KEY)}")
    inspect("FINNHUB_API_KEY", FINNHUB_API_KEY)

    env_path = Path(".env")
    if env_path.exists():
        raw = env_path.read_bytes()
        has_bom = raw.startswith(b"\xef\xbb\xbf")
        has_crlf = b"\r\n" in raw
        print(f"\n.env file: {len(raw)} bytes  "
              f"BOM={'YES (problem!)' if has_bom else 'no'}  "
              f"CRLF={'yes' if has_crlf else 'no'}")

    if FINNHUB_API_KEY:
        probe(
            "Finnhub basic test (AAPL quote)",
            "https://finnhub.io/api/v1/quote",
            {"symbol": "AAPL", "token": FINNHUB_API_KEY},
        )
        probe(
            "Finnhub TSX test (RY recommendation, bare ticker)",
            "https://finnhub.io/api/v1/stock/recommendation",
            {"symbol": "RY", "token": FINNHUB_API_KEY},
        )
        probe(
            "Finnhub TSX news (RY last 7d, bare ticker)",
            "https://finnhub.io/api/v1/company-news",
            {
                "symbol": "RY",
                "from": (datetime.now().date() - timedelta(days=7)).isoformat(),
                "to": datetime.now().date().isoformat(),
                "token": FINNHUB_API_KEY,
            },
        )

    print("\n" + "=" * 60)
    print("Reading guide:")
    print("  HTTP 200 + parsed JSON  -> key works for that endpoint")
    print("  HTTP 401                -> key invalid (most common: typo or "
          "unverified email)")
    print("  HTTP 403                -> key valid but endpoint requires paid tier")
    print("  HTTP 200 + empty list   -> key works but TSX coverage missing for "
          "that ticker")


def main():
    if "--debug-keys" in sys.argv:
        debug_keys()
        return

    if "--news" in sys.argv:
        idx = sys.argv.index("--news")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("Usage: python stock_signals.py --news <TICKER>")
        if not FINNHUB_API_KEY:
            raise SystemExit(
                "FINNHUB_API_KEY not set in .env. "
                "Get a free key at https://finnhub.io"
            )
        ticker = normalize_tsx_ticker(sys.argv[idx + 1])
        print_news(ticker)
        return

    if "--scan" in sys.argv:
        idx = sys.argv.index("--scan")
        tickers_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if tickers_arg and tickers_arg.startswith("--"):
            tickers_arg = None
        period_args = [a for a in sys.argv[1:]
                       if not a.startswith("--") and a != tickers_arg]
        period = period_args[0] if len(period_args) > 0 else "2y"
        interval = period_args[1] if len(period_args) > 1 else "1d"
        run_scan(tickers_arg, period, interval, color="--color" in sys.argv)
        return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--no-show" not in sys.argv
    raw = args[0] if len(args) > 0 else "XIC.TO"
    period = args[1] if len(args) > 1 else "2y"
    interval = args[2] if len(args) > 2 else "1d"

    ticker = normalize_tsx_ticker(raw)
    if ticker != raw.upper():
        print(f"Normalized {raw} -> {ticker}")

    print(f"Downloading {ticker} ({period}, {interval})…")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        print(
            f"No data for {ticker}. Verify it is listed on TSX/TSXV/CSE "
            f"(check at https://finance.yahoo.com/quote/{ticker})."
        )
        sys.exit(1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_indicators(df)
    df = generate_signals(df)
    stats = backtest(df)

    print(f"\n=== {ticker} signal summary ===")
    print(f"Trades:        {stats['trades']}")
    print(f"Win rate:      {stats['win_rate']:.1%}")
    print(f"Strategy ret:  {stats['total_return']:+.2%}")
    print(f"Buy & hold:    {stats['buy_hold']:+.2%}")

    last = df.iloc[-1]
    if bool(last["BUY"]):
        action = "BUY signal today"
    elif bool(last["SELL"]):
        action = "SELL signal today"
    else:
        score = float(last["SCORE"])
        action = f"HOLD (score {score:+.0f})"
    print(f"Latest bar:    {df.index[-1].date()} — {action}")

    plot(df, ticker, stats, show=show)


if __name__ == "__main__":
    main()

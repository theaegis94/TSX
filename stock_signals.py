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
                       require_dip: bool = False,
                       dip_window: int = 4,
                       dip_threshold_pct: float = -3.0,
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
    if not tickers or not (require_bollinger or require_rsi or require_dip):
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
                require_dip, dip_window, dip_threshold_pct,
            ))

        if progress_callback is not None:
            progress_callback((batch_idx + 1) / len(batches), len(matches))

    matches.sort(key=lambda r: (
        r["bb_buy_age"] if r["bb_buy_age"] is not None else 9999,
        r["rsi"],
    ))
    return matches


def _screen_batch(df, tickers, rsi_threshold, lookback_bars,
                  require_bollinger, require_rsi,
                  require_dip=False, dip_window=4,
                  dip_threshold_pct=-3.0) -> list[dict]:
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

            # N-bar dip: % change from N bars ago to now (negative = dip)
            dip_pct = None
            if len(tdf) > dip_window:
                ref_close = float(tdf["Close"].iloc[-dip_window - 1])
                if ref_close > 0:
                    dip_pct = (close_val - ref_close) / ref_close * 100
            dip_qualifies = (
                dip_pct is not None and dip_pct <= dip_threshold_pct
            )

            if require_bollinger and not bollinger_buy:
                continue
            if require_rsi and not rsi_oversold:
                continue
            if require_dip and not dip_qualifies:
                continue

            matches.append({
                "ticker": t,
                "price": close_val,
                "rsi": rsi_val,
                "bb_lower": bb_lo,
                "bb_distance_pct": (close_val - bb_lo) / bb_lo * 100 if bb_lo else 0.0,
                "bollinger_buy": bollinger_buy,
                "rsi_oversold": rsi_oversold,
                "dip_pct": dip_pct,
                "dip_qualifies": dip_qualifies,
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


def get_full_us_listing() -> list[str]:
    """All US exchange-listed common stocks + ETFs via Finnhub. EXCLUDES OTC.
    Returns Yahoo-format symbols (mostly bare tickers like 'AAPL', 'BRK-B').
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/symbol",
            params={"exchange": "US", "token": FINNHUB_API_KEY},
            timeout=30,
        )
        if r.status_code != 200:
            return []
        data = r.json() or []
        # Real exchange MICs only — exclude OTC markets (OOTC, PINX, OTCM, etc.)
        keep_mics = {
            "XNAS", "XNGS", "XNCM", "XNMS",   # NASDAQ tiers
            "XNYS", "XASE", "ARCX",            # NYSE / NYSE American / Arca
            "BATS", "BATY", "EDGA", "EDGX",    # CBOE / BATS family
            "IEXG",                            # IEX
        }
        keep_types = {"Common Stock", "ETP", "ADR", "REIT"}
        out: list[str] = []
        for s in data:
            mic = s.get("mic") or ""
            stype = s.get("type") or ""
            if mic not in keep_mics:
                continue
            if stype not in keep_types:
                continue
            sym = (s.get("displaySymbol") or s.get("symbol") or "").upper().strip()
            if not sym:
                continue
            # Drop warrants, rights, units, preferreds
            if any(x in sym for x in ("-WT", "-W", "-RT", "-R", "WS", "+")):
                continue
            sym = sym.replace(".", "-")
            out.append(sym)
        return sorted(set(out))
    except (requests.RequestException, ValueError):
        return []


def get_full_tsx_listing(market: str = "tsx") -> list[str]:
    """All listed companies on TSX (or TSXV) via TMX Group's directory.
    market='tsx' for the main board (~1000 names), 'tsxv' for Venture (~1700).
    Returns Yahoo-format tickers (e.g. 'RY.TO', 'GRT-UN.TO').
    """
    suffix = ".TO" if market == "tsx" else ".V"
    try:
        r = requests.get(
            f"https://www.tsx.com/json/company-directory/search/{market}/.*",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        results = data.get("results") or []
        out: list[str] = []
        for row in results:
            sym = (row.get("symbol") or "").upper().strip()
            if not sym:
                continue
            # TMX uses dots for class shares + unit suffixes (RDS.A, IGBT.UN);
            # Yahoo expects all of these as hyphens (RDS-A.TO, IGBT-UN.TO).
            sym = sym.replace(".", "-")
            out.append(f"{sym}{suffix}")
        return out
    except (requests.RequestException, ValueError):
        return []


def finnhub_exchange_symbols(exchange: str) -> list[str]:
    """All listed symbols on a Finnhub-supported exchange.
    'TO' = TSX, 'V' = TSX Venture, 'US' = NYSE+NASDAQ, etc.
    Returns Yahoo-format tickers (e.g. RY.TO).
    Free Finnhub endpoint — but the full list is huge so cache aggressively.
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/symbol",
            params={"exchange": exchange, "token": FINNHUB_API_KEY},
            timeout=30,
        )
        if r.status_code != 200:
            return []
        data = r.json() or []
        out: list[str] = []
        for s in data:
            sym = (s.get("displaySymbol") or s.get("symbol") or "").upper().strip()
            if not sym:
                continue
            # Drop warrants, rights, noise
            up = sym.upper()
            if any(x in up for x in (".WT", ".W", ".R", ".RT", ".UN.", "WS")):
                # be conservative — keep .UN (REIT units) but drop generic .W warrants
                if up.endswith(".UN"):
                    pass
                else:
                    continue
            out.append(sym)
        return out
    except (requests.RequestException, ValueError):
        return []


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


def finnhub_general_news(category: str = "general") -> list[dict]:
    """Broad market news (Finnhub categories: general, forex, crypto, merger)."""
    if not FINNHUB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": category, "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return r.json() or []
    except (requests.RequestException, ValueError):
        return []


# Curated lists for the "main news" section
MAJOR_TSX_FOR_NEWS = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "ENB.TO", "TRP.TO", "SHOP.TO", "CSU.TO", "BCE.TO", "T.TO",
    "CNR.TO", "CP.TO", "SU.TO", "CNQ.TO",
]
MAJOR_AI_US_FOR_NEWS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AAPL", "TSLA",
    "AMD", "AVGO", "PLTR", "ORCL", "CRM", "SMCI", "MU",
]
OIL_GAS_FOR_NEWS = [
    # US oil majors / E&P
    "XOM", "CVX", "COP", "OXY", "EOG", "PXD", "FANG", "DVN",
    "HES", "MRO", "PSX", "MPC", "VLO", "SLB",
    # US gas / midstream
    "WMB", "KMI", "OKE", "LNG",
    # Canadian oil / gas
    "SU.TO", "CNQ.TO", "ENB.TO", "TRP.TO", "IMO.TO", "CVE.TO",
    "ARX.TO", "TOU.TO", "MEG.TO", "PEY.TO",
    # Energy / commodity ETFs
    "XLE", "USO", "UNG", "BNO",
    "HOD.TO", "HOU.TO", "HND.TO", "HNU.TO",
]


def finnhub_insider_transactions(ticker: str, days: int = 90) -> list[dict]:
    """Recent insider buys/sells (Form 4 via Finnhub). Newest first."""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().date()
        frm = (today - timedelta(days=days)).isoformat()
        r = requests.get(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": _finnhub_sym(ticker), "from": frm,
                    "to": today.isoformat(), "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = (r.json() or {}).get("data", []) or []
        # Sort newest first
        data.sort(key=lambda x: x.get("transactionDate", ""), reverse=True)
        return data
    except (requests.RequestException, ValueError):
        return []


def finnhub_earnings_calendar(days_ahead: int = 30,
                              symbol: str | None = None) -> list[dict]:
    """Upcoming earnings reports in the next N days."""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().date()
        to = (today + timedelta(days=days_ahead)).isoformat()
        params = {"from": today.isoformat(), "to": to,
                  "token": FINNHUB_API_KEY}
        if symbol:
            params["symbol"] = _finnhub_sym(symbol)
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params=params, timeout=15,
        )
        if r.status_code != 200:
            return []
        return (r.json() or {}).get("earningsCalendar", []) or []
    except (requests.RequestException, ValueError):
        return []


def finnhub_ipo_calendar(days_ahead: int = 30) -> list[dict]:
    """Upcoming IPOs in the next N days."""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().date()
        to = (today + timedelta(days=days_ahead)).isoformat()
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/ipo",
            params={"from": today.isoformat(), "to": to,
                    "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return (r.json() or {}).get("ipoCalendar", []) or []
    except (requests.RequestException, ValueError):
        return []


def finnhub_etf_holdings(ticker: str) -> list[dict]:
    """Holdings inside an ETF (top constituents)."""
    if not FINNHUB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/etf/holdings",
            params={"symbol": _finnhub_sym(ticker),
                    "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = (r.json() or {}).get("holdings", []) or []
        # Sort by percent descending
        data.sort(key=lambda x: x.get("percent", 0) or 0, reverse=True)
        return data
    except (requests.RequestException, ValueError):
        return []


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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — used by Keltner, Supertrend, Parabolic SAR."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
               slow_k: int = 3):
    """Slow Stochastic Oscillator. Returns (%K, %D)."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    range_ = (high_max - low_min).replace(0, np.nan)
    fast_k = 100 * (df["Close"] - low_min) / range_
    slow_k_line = fast_k.rolling(slow_k).mean()
    slow_d = slow_k_line.rolling(d_period).mean()
    return slow_k_line, slow_d


def parabolic_sar(df: pd.DataFrame, af_init: float = 0.02,
                  af_max: float = 0.2, af_step: float = 0.02) -> pd.Series:
    """Parabolic SAR — trailing-stop indicator. Returns the SAR series."""
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    sar = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index)
    bull = True
    af = af_init
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = min(cur_sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < cur_sar:
                bull = False
                cur_sar = ep
                ep = low[i]
                af = af_init
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = max(cur_sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > cur_sar:
                bull = True
                cur_sar = ep
                ep = high[i]
                af = af_init
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar[i] = cur_sar
    return pd.Series(sar, index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend — ATR-based trailing stop. Returns (supertrend, direction).
    direction: +1 = uptrend (price > supertrend), -1 = downtrend.
    """
    a = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * a
    lower_band = hl2 - multiplier * a

    n = len(df)
    st_line = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)
    if n < 2:
        return st_line, direction

    direction.iloc[0] = 1
    st_line.iloc[0] = lower_band.iloc[0]
    for i in range(1, n):
        if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            st_line.iloc[i] = st_line.iloc[i - 1]
            continue
        if direction.iloc[i - 1] == 1:
            new_lb = max(lower_band.iloc[i], st_line.iloc[i - 1])
            if df["Close"].iloc[i] < new_lb:
                direction.iloc[i] = -1
                st_line.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                st_line.iloc[i] = new_lb
        else:
            new_ub = min(upper_band.iloc[i], st_line.iloc[i - 1])
            if df["Close"].iloc[i] > new_ub:
                direction.iloc[i] = 1
                st_line.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                st_line.iloc[i] = new_ub
    return st_line, direction


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou: int = 52):
    """Ichimoku Cloud — returns (tenkan, kijun, senkou_a, senkou_b, chikou)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tk = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kj = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    sa = ((tk + kj) / 2).shift(kijun)
    sb = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return tk, kj, sa, sb, chikou


def fib_levels(df: pd.DataFrame) -> dict:
    """Fibonacci retracement levels for the period (uses period high/low)."""
    high = float(df["High"].max())
    low = float(df["Low"].min())
    rng = high - low
    return {
        "0.0%": high,
        "23.6%": high - rng * 0.236,
        "38.2%": high - rng * 0.382,
        "50.0%": high - rng * 0.5,
        "61.8%": high - rng * 0.618,
        "78.6%": high - rng * 0.786,
        "100.0%": low,
    }


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
    out["SMA5"] = out["Close"].rolling(5).mean()
    out["SMA20"] = out["Close"].rolling(20).mean()
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
    "trend": "Trend (RSI + MACD + SMA)",
    "bollinger": "Bollinger Mean Reversion",
    "donchian": "Donchian 20d Breakout",
    "sma200_dip": "SMA200 Dip Buy",
    "rsi": "RSI Strategy (oversold/overbought)",
    "macd": "MACD Strategy (signal cross)",
    "momentum": "Momentum (10-day rate of change)",
    "stochastic": "Stochastic Slow",
    "keltner": "Keltner Channels",
    "supertrend": "Supertrend",
    "psar": "Parabolic SAR",
    "ma_cross": "MA Cross (20/50)",
    "inside_bar": "Inside Bar Breakout",
    "outside_bar": "Outside Bar Reversal",
    "outside_bar_breakout": "Outside Bar Breakout",
    "candlestick": "Candlestick Patterns (Hammer/Engulfing/Star/...)",
    "double_topbot": "Double Top / Double Bottom",
}

DEFAULT_STRATEGY_KEY = "bollinger"


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


def _strategy_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Buy on RSI cross up through 30 (oversold reversal); sell on cross down through 70."""
    out = df.copy()
    out["BUY"] = (out["RSI"].shift(1) < 30) & (out["RSI"] >= 30)
    out["SELL"] = (out["RSI"].shift(1) > 70) & (out["RSI"] <= 70)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line crossing its signal line."""
    out = df.copy()
    out["BUY"] = (
        (out["MACD"].shift(1) < out["MACD_SIGNAL"].shift(1))
        & (out["MACD"] >= out["MACD_SIGNAL"])
    )
    out["SELL"] = (
        (out["MACD"].shift(1) > out["MACD_SIGNAL"].shift(1))
        & (out["MACD"] <= out["MACD_SIGNAL"])
    )
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_momentum(df: pd.DataFrame, period: int = 10,
                       buy_threshold: float = 5.0,
                       sell_threshold: float = -5.0) -> pd.DataFrame:
    """N-day rate of change crosses thresholds."""
    out = df.copy()
    roc = (out["Close"] / out["Close"].shift(period) - 1) * 100
    out["BUY"] = (roc.shift(1) < buy_threshold) & (roc >= buy_threshold)
    out["SELL"] = (roc.shift(1) > sell_threshold) & (roc <= sell_threshold)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic %K crosses %D in oversold/overbought zones."""
    out = df.copy()
    if {"High", "Low", "Close"}.issubset(out.columns):
        k, d = stochastic(out)
    else:
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    cross_up = (k.shift(1) < d.shift(1)) & (k >= d)
    cross_dn = (k.shift(1) > d.shift(1)) & (k <= d)
    out["BUY"] = cross_up & (k < 30)
    out["SELL"] = cross_dn & (k > 70)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_keltner(df: pd.DataFrame, period: int = 20,
                      atr_mult: float = 2.0) -> pd.DataFrame:
    """Keltner channels: EMA ± (ATR × multiplier). Mean-reversion at bands."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    mid = out["Close"].ewm(span=period, adjust=False).mean()
    a = atr(out, period)
    upper = mid + atr_mult * a
    lower = mid - atr_mult * a
    out["BUY"] = (out["Close"].shift(1) > lower.shift(1)) & (out["Close"] <= lower)
    out["SELL"] = (out["Close"].shift(1) < upper.shift(1)) & (out["Close"] >= upper)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """Buy when supertrend flips to uptrend (-1 -> +1); sell on flip to downtrend."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    _, direction = supertrend(out)
    out["BUY"] = (direction.shift(1) == -1) & (direction == 1)
    out["SELL"] = (direction.shift(1) == 1) & (direction == -1)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_psar(df: pd.DataFrame) -> pd.DataFrame:
    """Buy when close crosses above PSAR; sell when close crosses below."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    psar = parabolic_sar(out)
    above = out["Close"] > psar
    out["BUY"] = (~above.shift(1).fillna(False)) & above
    out["SELL"] = above.shift(1).fillna(False) & (~above)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_ma_cross(df: pd.DataFrame, fast: int = 20,
                       slow: int = 50) -> pd.DataFrame:
    """Fast SMA crosses slow SMA — classic golden/death cross variant."""
    out = df.copy()
    sma_fast = out["Close"].rolling(fast).mean()
    sma_slow = out["Close"].rolling(slow).mean()
    out["BUY"] = (sma_fast.shift(1) < sma_slow.shift(1)) & (sma_fast >= sma_slow)
    out["SELL"] = (sma_fast.shift(1) > sma_slow.shift(1)) & (sma_fast <= sma_slow)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Inside bar = today's high < yesterday's high AND today's low > yesterday's low.
    Buy when price breaks above the inside-bar's high; sell on break below."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    inside = (out["High"] < out["High"].shift(1)) & (out["Low"] > out["Low"].shift(1))
    inside_high = out["High"].where(inside)
    inside_low = out["Low"].where(inside)
    last_high = inside_high.ffill().shift(1)
    last_low = inside_low.ffill().shift(1)
    out["BUY"] = out["Close"] > last_high
    out["SELL"] = out["Close"] < last_low
    # Only fire once per breakout — when first crossing
    out["BUY"] = out["BUY"] & ~out["BUY"].shift(1).fillna(False)
    out["SELL"] = out["SELL"] & ~out["SELL"].shift(1).fillna(False)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _detect_candlestick_patterns(df: pd.DataFrame):
    """Detect ~12 reliable candlestick patterns.
    Returns (bullish_mask, bearish_mask) — boolean Series.
    """
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        empty = pd.Series(False, index=df.index)
        return empty, empty

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    full = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    bear = o > c

    # Single-candle: small body relative to range
    small_body = body < (full * 0.30)

    # Hammer: small body at top, long lower shadow (>=2x body), uptrend not required for screen
    hammer = small_body & (lower > body * 2) & (upper < body)
    # Inverted Hammer: small body at bottom, long upper shadow
    inv_hammer = small_body & (upper > body * 2) & (lower < body)
    # Shooting Star: like inverted hammer but bearish context (after up move)
    prior_up = c.shift(1) > c.shift(5)
    shooting_star = small_body & (upper > body * 2) & (lower < body) & prior_up

    # Doji: open ≈ close
    doji = body < (full * 0.10)
    dragonfly_doji = doji & (lower > full * 0.6) & (upper < full * 0.1)
    gravestone_doji = doji & (upper > full * 0.6) & (lower < full * 0.1)

    # Engulfing
    prev_body = (c.shift(1) - o.shift(1)).abs()
    bullish_engulfing = (
        bear.shift(1) & bull
        & (o < c.shift(1)) & (c > o.shift(1))
        & (body > prev_body)
    )
    bearish_engulfing = (
        bull.shift(1) & bear
        & (o > c.shift(1)) & (c < o.shift(1))
        & (body > prev_body)
    )

    # Harami (opposite of engulfing — current is inside prior body)
    bullish_harami = (
        bear.shift(1) & bull
        & (o > c.shift(1)) & (c < o.shift(1))
        & (body < prev_body)
    )
    bearish_harami = (
        bull.shift(1) & bear
        & (o < c.shift(1)) & (c > o.shift(1))
        & (body < prev_body)
    )

    # Three White Soldiers / Three Black Crows
    three_white_soldiers = (
        bull.shift(2) & bull.shift(1) & bull
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & (o > o.shift(1)) & (o.shift(1) > o.shift(2))
    )
    three_black_crows = (
        bear.shift(2) & bear.shift(1) & bear
        & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        & (o < o.shift(1)) & (o.shift(1) < o.shift(2))
    )

    # Morning Star (bull) / Evening Star (bear) — 3-candle reversal
    star_body_small = body.shift(1) < (full.shift(1) * 0.30)
    morning_star = (
        bear.shift(2) & star_body_small & bull
        & (c > (o.shift(2) + c.shift(2)) / 2)
    )
    evening_star = (
        bull.shift(2) & star_body_small & bear
        & (c < (o.shift(2) + c.shift(2)) / 2)
    )

    # Piercing / Dark Cloud Cover
    piercing = (
        bear.shift(1) & bull
        & (o < l.shift(1))
        & (c > (o.shift(1) + c.shift(1)) / 2)
        & (c < o.shift(1))
    )
    dark_cloud = (
        bull.shift(1) & bear
        & (o > h.shift(1))
        & (c < (o.shift(1) + c.shift(1)) / 2)
        & (c > o.shift(1))
    )

    bullish = (
        hammer.fillna(False) | inv_hammer.fillna(False)
        | dragonfly_doji.fillna(False)
        | bullish_engulfing.fillna(False) | bullish_harami.fillna(False)
        | three_white_soldiers.fillna(False)
        | morning_star.fillna(False) | piercing.fillna(False)
    )
    bearish = (
        shooting_star.fillna(False) | gravestone_doji.fillna(False)
        | bearish_engulfing.fillna(False) | bearish_harami.fillna(False)
        | three_black_crows.fillna(False)
        | evening_star.fillna(False) | dark_cloud.fillna(False)
    )
    return bullish.astype(bool), bearish.astype(bool)


def _strategy_candlestick(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate of ~12 candlestick patterns. BUY = any bullish, SELL = any bearish."""
    out = df.copy()
    bull, bear = _detect_candlestick_patterns(out)
    out["BUY"] = bull
    out["SELL"] = bear
    out["SCORE"] = bull.astype(int) * 2 - bear.astype(int) * 2
    return out


def _strategy_double_topbot(df: pd.DataFrame, window: int = 30,
                            tolerance: float = 0.03) -> pd.DataFrame:
    """Naive Double Top / Double Bottom detector.
    Looks for two local extrema within `window` bars at similar price levels
    (within `tolerance` percent). Fires SELL on second top, BUY on second bottom.
    """
    out = df.copy()
    close = out["Close"]

    # Rolling local maxima/minima — index where this bar equals window's max/min
    half = max(2, window // 4)
    is_peak = (close == close.rolling(2 * half + 1, center=True).max())
    is_trough = (close == close.rolling(2 * half + 1, center=True).min())

    buy = pd.Series(False, index=out.index)
    sell = pd.Series(False, index=out.index)

    # For each bar, look back `window` bars and see if there's another peak/trough nearby
    peak_idx = np.where(is_peak.fillna(False))[0]
    trough_idx = np.where(is_trough.fillna(False))[0]

    for i in range(1, len(peak_idx)):
        cur, prev = peak_idx[i], peak_idx[i - 1]
        if cur - prev > window or cur - prev < 5:
            continue
        if abs(close.iloc[cur] - close.iloc[prev]) / close.iloc[prev] <= tolerance:
            sell.iloc[cur] = True

    for i in range(1, len(trough_idx)):
        cur, prev = trough_idx[i], trough_idx[i - 1]
        if cur - prev > window or cur - prev < 5:
            continue
        if abs(close.iloc[cur] - close.iloc[prev]) / close.iloc[prev] <= tolerance:
            buy.iloc[cur] = True

    out["BUY"] = buy
    out["SELL"] = sell
    out["SCORE"] = buy.astype(int) * 2 - sell.astype(int) * 2
    return out


def _strategy_outside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Outside bar = today's high > yesterday's high AND today's low < yesterday's low.
    Buy on outside bar that closes above prior close (bullish reversal);
    sell on outside bar that closes below prior close (bearish reversal)."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    outside = (out["High"] > out["High"].shift(1)) & (out["Low"] < out["Low"].shift(1))
    out["BUY"] = outside & (out["Close"] > out["Close"].shift(1))
    out["SELL"] = outside & (out["Close"] < out["Close"].shift(1))
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


def _strategy_outside_bar_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Outside-bar breakout: identify outside bars (today's range engulfs
    yesterday's). After one forms, BUY when a later close breaks above the
    outside bar's high, SELL when a close breaks below its low.
    Different from `outside_bar` (the reversal version) — this version waits
    for confirmation after the engulfing pattern."""
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns):
        out["BUY"] = False
        out["SELL"] = False
        out["SCORE"] = 0
        return out
    outside = (
        (out["High"] > out["High"].shift(1))
        & (out["Low"] < out["Low"].shift(1))
    )
    # The most recent outside bar's high/low — held until the next outside bar
    last_outside_high = out["High"].where(outside).ffill().shift(1)
    last_outside_low = out["Low"].where(outside).ffill().shift(1)

    breakout_up = out["Close"] > last_outside_high
    breakout_dn = out["Close"] < last_outside_low

    # Fire only on the first bar of each breakout (avoid repeats while held)
    out["BUY"] = breakout_up & ~breakout_up.shift(1).fillna(False)
    out["SELL"] = breakout_dn & ~breakout_dn.shift(1).fillna(False)
    out["SCORE"] = out["BUY"].astype(int) * 2 - out["SELL"].astype(int) * 2
    return out


_STRATEGIES = {
    "trend": _strategy_trend,
    "bollinger": _strategy_bollinger,
    "donchian": _strategy_donchian,
    "sma200_dip": _strategy_sma200_dip,
    "rsi": _strategy_rsi,
    "macd": _strategy_macd,
    "momentum": _strategy_momentum,
    "stochastic": _strategy_stochastic,
    "keltner": _strategy_keltner,
    "supertrend": _strategy_supertrend,
    "psar": _strategy_psar,
    "ma_cross": _strategy_ma_cross,
    "inside_bar": _strategy_inside_bar,
    "outside_bar": _strategy_outside_bar,
    "outside_bar_breakout": _strategy_outside_bar_breakout,
    "candlestick": _strategy_candlestick,
    "double_topbot": _strategy_double_topbot,
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
    if compact:
        figsize = (12, 6)
        dpi = 150
        title_fs, label_fs, tick_fs, legend_fs = 12, 11, 10, 10
        marker_size, lw = 120, 1.1
    else:
        figsize = (14, 10)
        dpi = 120
        title_fs, label_fs, tick_fs, legend_fs = 14, 13, 12, 12
        marker_size, lw = 140, 1.2

    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=figsize, sharex=True, dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax_price.plot(df.index, df["Close"], label="Close", color="black", linewidth=lw)
    ax_price.plot(df.index, df["SMA50"], label="SMA50", color="tab:blue", linewidth=lw * 0.85)
    ax_price.plot(df.index, df["SMA200"], label="SMA200", color="tab:orange", linewidth=lw * 0.85)

    buys = df[df["BUY"]]
    sells = df[df["SELL"]]
    ax_price.scatter(buys.index, buys["Close"], marker="^", s=marker_size,
                     color="green", label="BUY", zorder=5, edgecolors="black",
                     linewidths=0.6)
    ax_price.scatter(sells.index, sells["Close"], marker="v", s=marker_size,
                     color="red", label="SELL", zorder=5, edgecolors="black",
                     linewidths=0.6)

    if compact:
        title = f"{ticker} ({stats['trades']} trades, {stats['win_rate']:.0%} win)"
    else:
        title = (
            f"{ticker} — signals: {stats['trades']} trades | "
            f"win rate {stats['win_rate']:.0%} | "
            f"strategy {stats['total_return']:+.1%} vs buy&hold {stats['buy_hold']:+.1%}"
        )
    ax_price.set_title(title, fontsize=title_fs)
    ax_price.set_ylabel("Price", fontsize=label_fs)
    ax_price.legend(loc="upper left", fontsize=legend_fs, ncol=2 if compact else 1,
                    framealpha=0.85)
    ax_price.grid(alpha=0.3)
    ax_price.tick_params(axis="both", labelsize=tick_fs)

    ax_rsi.plot(df.index, df["RSI"], color="purple", linewidth=lw * 0.85)
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5, linewidth=0.5)
    ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5, linewidth=0.5)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)", fontsize=label_fs)
    ax_rsi.grid(alpha=0.3)
    ax_rsi.tick_params(axis="both", labelsize=tick_fs)

    ax_macd.plot(df.index, df["MACD"], label="MACD", color="tab:blue", linewidth=lw * 0.85)
    ax_macd.plot(df.index, df["MACD_SIGNAL"], label="Signal",
                 color="tab:orange", linewidth=lw * 0.85)
    colors = ["green" if v >= 0 else "red" for v in df["MACD_HIST"].fillna(0)]
    ax_macd.bar(df.index, df["MACD_HIST"], color=colors, alpha=0.4, width=1.0)
    ax_macd.axhline(0, color="black", linewidth=0.5)
    ax_macd.set_ylabel("MACD", fontsize=label_fs)
    ax_macd.legend(loc="upper left", fontsize=legend_fs, ncol=2 if compact else 1,
                   framealpha=0.85)
    ax_macd.grid(alpha=0.3)
    ax_macd.tick_params(axis="both", labelsize=tick_fs)

    ax_macd.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


INDICATOR_LABELS = {
    "sma50": "SMA 50 (Simple Moving Average, 50 bars)",
    "sma200": "SMA 200 (Simple Moving Average, 200 bars)",
    "ema20": "EMA 20 (Exponential Moving Average, 20 bars)",
    "ema50": "EMA 50 (Exponential Moving Average, 50 bars)",
    "bollinger": "Bollinger Bands (mean ± 2σ volatility envelope)",
    "ichimoku": "Ichimoku Cloud (Tenkan / Kijun / Senkou A & B)",
    "fibonacci": "Fibonacci levels (key retracement ratios)",
}
DEFAULT_INDICATORS = ["sma50", "sma200"]


def build_chart_plotly(df: pd.DataFrame, ticker: str, stats: dict,
                       compact: bool = False,
                       indicators: list[str] | None = None,
                       theme_dark: bool = True):
    """Interactive Plotly version of the price/RSI/MACD chart.
    Supports mousewheel zoom, click+drag pan, hover tooltips, and range buttons.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.60, 0.20, 0.20],
    )

    # --- Price panel — single Close line, color follows the theme ---
    close_color = "#ffffff" if theme_dark else "#000000"
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close",
        line=dict(color=close_color, width=0.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Overlay indicators — render only the user-selected ones
    if indicators is None:
        indicators = list(DEFAULT_INDICATORS)
    indicators_set = set(indicators)

    if "sma50" in indicators_set:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA50"], mode="lines", name="SMA50",
            line=dict(color="#3b82f6", width=0.5),
            hovertemplate="SMA50: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "sma200" in indicators_set:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA200"], mode="lines", name="SMA200",
            line=dict(color="#f59e0b", width=0.5),
            hovertemplate="SMA200: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "ema20" in indicators_set:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].ewm(span=20, adjust=False).mean(),
            mode="lines", name="EMA20",
            line=dict(color="#10b981", width=0.5),
            hovertemplate="EMA20: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "ema50" in indicators_set:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].ewm(span=50, adjust=False).mean(),
            mode="lines", name="EMA50",
            line=dict(color="#a855f7", width=0.5),
            hovertemplate="EMA50: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "bollinger" in indicators_set and "BB_LOWER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"], mode="lines", name="BB Upper",
            line=dict(color="#94a3b8", width=0.5, dash="dot"),
            hovertemplate="BB Upper: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"], mode="lines", name="BB Lower",
            line=dict(color="#94a3b8", width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
            hovertemplate="BB Lower: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "ichimoku" in indicators_set and {"High", "Low"}.issubset(df.columns):
        tk, kj, sa, sb, _chikou = ichimoku(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=tk, mode="lines", name="Tenkan",
            line=dict(color="#3b82f6", width=0.5),
            hovertemplate="Tenkan: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=kj, mode="lines", name="Kijun",
            line=dict(color="#ef4444", width=0.5),
            hovertemplate="Kijun: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        # Cloud: SA and SB as filled area
        fig.add_trace(go.Scatter(
            x=df.index, y=sa, mode="lines", name="Senkou A",
            line=dict(color="#22c55e", width=0.6),
            hovertemplate="Senkou A: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=sb, mode="lines", name="Senkou B",
            line=dict(color="#dc2626", width=0.6),
            fill="tonexty", fillcolor="rgba(34,197,94,0.10)",
            hovertemplate="Senkou B: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if "fibonacci" in indicators_set and {"High", "Low"}.issubset(df.columns):
        levels = fib_levels(df)
        fib_colors = {
            "0.0%": "#9ca3af", "23.6%": "#fbbf24", "38.2%": "#f59e0b",
            "50.0%": "#a855f7", "61.8%": "#3b82f6",
            "78.6%": "#10b981", "100.0%": "#9ca3af",
        }
        x_start, x_end = df.index[0], df.index[-1]
        for label, value in levels.items():
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[value, value],
                mode="lines", name=f"Fib {label}",
                line=dict(color=fib_colors.get(label, "#9ca3af"),
                          width=0.6, dash="dash"),
                hovertemplate=f"Fib {label}: ${value:.2f}<extra></extra>",
            ), row=1, col=1)

    buys = df[df["BUY"]]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"], mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", color="#16a34a", size=14,
                        line=dict(color="black", width=1)),
            hovertemplate="<b>BUY</b><br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    sells = df[df["SELL"]]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"], mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", color="#dc2626", size=14,
                        line=dict(color="black", width=1)),
            hovertemplate="<b>SELL</b><br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # --- RSI panel ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], mode="lines", name="RSI",
        line=dict(color="#c084fc", width=1.5),
        showlegend=False,
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  line_width=0.6, opacity=0.6, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  line_width=0.6, opacity=0.6, row=2, col=1)

    # --- MACD panel ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], mode="lines", name="MACD",
        line=dict(color="#60a5fa", width=1.5),
        showlegend=False,
        hovertemplate="MACD: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal",
        line=dict(color="#fbbf24", width=1.5),
        showlegend=False,
        hovertemplate="Signal: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    hist = df["MACD_HIST"].fillna(0)
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="Hist",
        marker_color=colors, opacity=0.5,
        showlegend=False,
        hovertemplate="Hist: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="#6b7280", line_width=0.4, row=3, col=1)

    # --- Layout ---
    if compact:
        height = 1200
    else:
        height = 1400

    # Hide Candlestick's default rangeslider (we have rangeselector buttons instead)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Chart title is rendered as Streamlit markdown above the chart instead
    # of inside the figure — avoids collisions with the range-selector buttons.
    fig.update_layout(
        height=height,
        # Wider left margin to fit the rotated panel labels clear of the
        # widest y-axis numbers (e.g. "100" on the RSI panel).
        margin=dict(l=90, r=60, t=60, b=100),
        hovermode="x unified",
        # Legend BELOW the chart (under the bottom MACD panel) — never
        # competes with the modebar (top-right) or range buttons (top-left).
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.06,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        ),
        template="plotly_dark",
        plot_bgcolor="#4a4b4e",
        paper_bgcolor="#4a4b4e",
        bargap=0,
        # "pan" lets users drag to scroll horizontally without drawing a
        # 2D zoom box. Scroll wheel still zooms in/out.
        dragmode="pan",
        font=dict(
            family='"Comic Sans MS", "Comic Sans", cursive',
            color="#e5e7eb",
            size=14,
        ),
    )
    # Y-axis titles — Plotly handles positioning automatically
    title_font = dict(size=15, color="#9ca3af",
                      family='"Comic Sans MS", "Comic Sans", cursive')
    fig.update_yaxes(
        title=dict(text="<b>Price</b>", standoff=10, font=title_font),
        row=1, col=1, gridcolor="#5a5b5e", zerolinecolor="#5a5b5e",
        autorange=True,
        tickprefix="$",
        tickformat=".2f",
    )
    fig.update_yaxes(
        title=dict(text="<b>RSI</b>", standoff=10, font=title_font),
        range=[0, 100], row=2, col=1, gridcolor="#5a5b5e",
    )
    fig.update_yaxes(
        title=dict(text="<b>MACD</b>", standoff=10, font=title_font),
        row=3, col=1, gridcolor="#5a5b5e", zerolinecolor="#5a5b5e",
        autorange=True,
    )

    # Explicitly set the visible x-range AND lock pan/zoom bounds to the
    # data range so users can't drag off into empty space.
    # x_end anchored exactly to the last data point so the rangeselector
    # buttons (1m, 3m, …) always land on real data.
    if len(df) >= 2:
        x_start = df.index[0]
        x_end = df.index[-1]
        for r in (1, 2, 3):
            fig.update_xaxes(
                range=[x_start, x_end],
                minallowed=x_start,
                maxallowed=x_end,
                row=r, col=1, gridcolor="#5a5b5e",
            )

        # Price panel — linear scale, floor strictly at $0.
        # minallowed=0 is a hard pan/zoom floor — user can't scroll below.
        # Top is 10% above the highest line in the data window.
        price_max = float(df["Close"].max())
        if price_max > 0:
            fig.update_yaxes(
                autorange=False,
                rangemode="nonnegative",
                minallowed=0,
                range=[0, price_max * 1.10],
                row=1, col=1,
            )

        # RSI bounded 0..100 — natural range
        fig.update_yaxes(autorange=True, range=[0, 100], row=2, col=1)

        # MACD: data range with 5% padding (was 30% — too much)
        macd_vals = pd.concat([df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"]]).dropna()
        if not macd_vals.empty:
            mlo, mhi = float(macd_vals.min()), float(macd_vals.max())
            if mhi > mlo:
                pad = (mhi - mlo) * 0.05
                fig.update_yaxes(
                    autorange=True,
                    range=[mlo - pad, mhi + pad],
                    row=3, col=1,
                )

    # Range buttons above the chart, top-left (legend sits at top-right)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1f2937",
            activecolor="#3b82f6",
            bordercolor="#374151",
            borderwidth=1,
            font=dict(color="#e5e7eb", size=12),
            x=0,
            xanchor="left",
            y=1.05,
            yanchor="bottom",
        ),
        row=1, col=1,
    )

    # Distinct, thicker colored border per panel + visible vertical & horizontal
    # grid lines so each panel is clearly framed and time alignment is obvious.
    panel_borders = {
        1: "#60a5fa",  # blue   → Price
        2: "#a855f7",  # purple → RSI
        3: "#f59e0b",  # orange → MACD
    }
    for r, color in panel_borders.items():
        fig.update_xaxes(
            showgrid=True, gridcolor="#6a6b6e", gridwidth=0.5,
            showline=True, linecolor=color, linewidth=1.2,
            mirror=True,
            showticklabels=True,  # show month labels on every panel
            row=r, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#6a6b6e", gridwidth=0.5,
            showline=True, linecolor=color, linewidth=1.2,
            mirror=True, row=r, col=1,
        )

    # Subtle matching tint on each panel background (so the area inside is
    # slightly different too, not just the border).
    panel_tints = {
        1: "rgba(96,165,250,0.04)",
        2: "rgba(168,85,247,0.06)",
        3: "rgba(245,158,11,0.05)",
    }
    for r, tint in panel_tints.items():
        fig.add_shape(
            type="rect",
            xref="x domain", yref="y domain",
            x0=0, x1=1, y0=0, y1=1,
            fillcolor=tint, line_width=0, layer="below",
            row=r, col=1,
        )

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
    "TSLA", "HOD.TO", "HOU.TO", "HNU.TO", "HND.TO", "NOW",
]

# Original broad-CA/US default kept for reference/seed
_LEGACY_BROAD_DEFAULT = [
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

"""Feature computation for the paper-trading agent.

Strategies consume a dict of features computed once per cycle. Keeping
the feature build separate from the strategy logic means we can:
  - cache the data fetch (yfinance call is the slow part)
  - reuse the same features across many strategies
  - add new features (EIA inventory, weather, etc.) without touching
    individual strategies

Right now the feature set is purely price-based on CL=F (WTI crude),
NG=F (Henry Hub natgas), DXY (dollar index), OVX (oil volatility),
and the 4 ETFs themselves. Week 2+ will add EIA + NOAA features here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

import stock_signals as ss

from . import eia
from . import cftc
from . import weather

# Underlying commodities + macro
SYMBOLS = {
    "wti":  "CL=F",       # WTI crude oil futures
    "ng":   "NG=F",       # Henry Hub natural gas futures
    "brent": "BZ=F",      # Brent crude (lead signal for WTI)
    "dxy":  "DX-Y.NYB",   # US dollar index
    "ovx":  "^OVX",       # CBOE oil volatility index
    "xle":  "XLE",        # US energy equity ETF (lead/lag)
    "vix":  "^VIX",       # Equity vol (risk-on/off)
}


def _safe_float(x) -> float | None:
    try:
        f = float(x)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _series_features(close: pd.Series) -> dict[str, float | None]:
    """Extract price + technical features from a single close series."""
    if close is None or close.empty or len(close) < 20:
        return {}
    out: dict[str, float | None] = {
        "close": _safe_float(close.iloc[-1]),
    }
    # Returns over multiple horizons
    for n in (1, 5, 20):
        if len(close) > n:
            ret = (close.iloc[-1] / close.iloc[-n - 1] - 1) * 100
            out[f"ret_{n}d_pct"] = _safe_float(ret)
        else:
            out[f"ret_{n}d_pct"] = None
    # RSI
    try:
        rsi_series = ss.rsi(close, 14)
        out["rsi"] = _safe_float(rsi_series.iloc[-1])
        # 5-bar RSI change (momentum of momentum)
        if len(rsi_series) >= 6:
            out["rsi_change_5d"] = _safe_float(
                rsi_series.iloc[-1] - rsi_series.iloc[-6]
            )
        else:
            out["rsi_change_5d"] = None
    except Exception:
        out["rsi"] = None
        out["rsi_change_5d"] = None
    # MACD
    try:
        macd_line, signal_line, hist = ss.macd(close)
        out["macd"] = _safe_float(macd_line.iloc[-1])
        out["macd_signal"] = _safe_float(signal_line.iloc[-1])
        out["macd_hist"] = _safe_float(hist.iloc[-1])
        # Cross detection: True if line crossed signal between t-1 and t
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
            cur_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
            out["macd_cross_bull"] = bool(prev_diff <= 0 < cur_diff)
            out["macd_cross_bear"] = bool(prev_diff >= 0 > cur_diff)
        else:
            out["macd_cross_bull"] = False
            out["macd_cross_bear"] = False
    except Exception:
        out["macd"] = None
        out["macd_signal"] = None
        out["macd_hist"] = None
        out["macd_cross_bull"] = False
        out["macd_cross_bear"] = False
    # Bollinger position (distance to bands, normalized)
    try:
        mid, up, lo = ss.bollinger(close, 20, 2.0)
        cur = float(close.iloc[-1])
        u = float(up.iloc[-1])
        l = float(lo.iloc[-1])
        # 0 = at lower band, 1 = at upper band
        out["bb_position"] = (
            _safe_float((cur - l) / (u - l)) if u > l else None
        )
    except Exception:
        out["bb_position"] = None
    # 200-day SMA + bear-regime detection (iter 44 — too slow)
    try:
        if len(close) >= 200:
            sma200 = close.rolling(200).mean()
            sma200_now = float(sma200.iloc[-1])
            sma200_60d_ago = (
                float(sma200.iloc[-61]) if len(sma200) >= 61 else sma200_now
            )
            cur = float(close.iloc[-1])
            out["sma200"] = _safe_float(sma200_now)
            out["sma200_slope_60d"] = _safe_float(
                (sma200_now - sma200_60d_ago) / sma200_60d_ago * 100
            )
            out["bear_regime"] = (
                cur < sma200_now and sma200_now < sma200_60d_ago
            )
        else:
            out["sma200"] = None
            out["sma200_slope_60d"] = None
            out["bear_regime"] = False
    except Exception:
        out["sma200"] = None
        out["sma200_slope_60d"] = None
        out["bear_regime"] = False
    # ITER 45: faster regime detector — 30-day momentum. Sharp drops
    # catch trending bear markets BEFORE the 200d SMA rolls over.
    try:
        if len(close) > 30:
            ret_30d = (close.iloc[-1] / close.iloc[-31] - 1) * 100
            out["ret_30d_pct"] = _safe_float(ret_30d)
            out["fast_bear"] = ret_30d < -10.0
        else:
            out["ret_30d_pct"] = None
            out["fast_bear"] = False
    except Exception:
        out["ret_30d_pct"] = None
        out["fast_bear"] = False
    return out


def fetch_features() -> dict[str, Any]:
    """Pull latest data for all underlying symbols, compute features.
    Returns a flat dict keyed by `<symbol>_<feature>` plus metadata.
    """
    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "as_of_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Need ≥200 bars for the SMA200 regime filter (iter 44). Bump to 1y.
    syms = list(SYMBOLS.values())
    try:
        df = yf.download(
            " ".join(syms),
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        out["fetch_error"] = str(e)
        return out
    if df is None or df.empty:
        out["fetch_error"] = "empty download"
        return out

    for short, full_sym in SYMBOLS.items():
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if full_sym not in df.columns.get_level_values(0):
                    continue
                close = df[full_sym]["Close"].dropna()
            else:
                close = df["Close"].dropna()
            sub = _series_features(close)
            for k, v in sub.items():
                out[f"{short}_{k}"] = v
        except Exception:
            continue

    # Calendar features
    now = datetime.now(timezone.utc)
    out["day_of_week"] = now.weekday()  # 0=Mon ... 6=Sun
    out["is_eia_oil_day"] = (now.weekday() == 2)   # Wednesday
    out["is_eia_gas_day"] = (now.weekday() == 3)   # Thursday
    out["month"] = now.month
    out["is_winter"] = now.month in (11, 12, 1, 2, 3)
    out["is_summer"] = now.month in (6, 7, 8)

    # Brent-WTI spread (positive means Brent premium — typical)
    if "brent_close" in out and "wti_close" in out:
        b = out.get("brent_close")
        w = out.get("wti_close")
        if b is not None and w is not None:
            out["brent_wti_spread"] = b - w

    # --- EIA inventory features (live) ---
    try:
        out.update(eia.oil_inventory_features())
    except Exception:
        pass
    try:
        out.update(eia.natgas_storage_features())
    except Exception:
        pass

    # --- CFTC speculator positioning (live) ---
    try:
        cot_df = cftc.fetch_cot_data()
        out.update(cftc.compute_cftc_features(cot_df))
    except Exception:
        pass

    # --- Weather (live) — natgas demand driver ---
    try:
        wx_df = weather.fetch_weather_history()
        out.update(weather.compute_weather_features(wx_df))
    except Exception:
        pass

    return out


def fetch_latest_etf_prices() -> dict[str, float]:
    """Quick spot-price fetch for the 4 paper-tradable ETFs."""
    from .storage import TICKERS
    try:
        df = yf.download(
            " ".join(TICKERS),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for t in TICKERS:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if t not in df.columns.get_level_values(0):
                    continue
                close = df[t]["Close"].dropna()
            else:
                close = df["Close"].dropna()
            if len(close) >= 1:
                out[t] = float(close.iloc[-1])
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Backtest support — features as-of a past date, with no future leakage.
# ---------------------------------------------------------------------------

def precompute_feature_history(years_back: int = 5) -> pd.DataFrame:
    """One-shot fetch of every symbol's full daily history for the
    backtest window. Returns a MultiIndex DataFrame so callers can
    slice by (symbol, date) without further network calls."""
    syms = list(SYMBOLS.values())
    try:
        df = yf.download(
            " ".join(syms),
            period=f"{years_back + 1}y",  # +1 yr buffer for lookback windows
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def precompute_etf_history(years_back: int = 5) -> pd.DataFrame:
    """Same idea, for the 4 paper-tradable ETFs."""
    from .storage import TICKERS
    try:
        df = yf.download(
            " ".join(TICKERS),
            period=f"{years_back + 1}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def features_as_of(
    precomputed_df: pd.DataFrame,
    as_of_date,
    eia_oil_df: pd.DataFrame | None = None,
    eia_gas_df: pd.DataFrame | None = None,
    cftc_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute features as they would have looked at the close of
    `as_of_date`. CRITICAL: slices the series to bars AT or BEFORE
    as_of_date so we have zero future leakage.

    `eia_oil_df` / `eia_gas_df` are pre-fetched EIA inventory series
    (passed in once for the whole backtest so we don't re-hit the
    API every day). Pass None to skip EIA features.
    """
    as_of_ts = pd.Timestamp(as_of_date)
    out: dict[str, Any] = {
        "as_of": as_of_ts.strftime("%Y-%m-%d"),
    }
    if precomputed_df is None or precomputed_df.empty:
        return out

    for short, full_sym in SYMBOLS.items():
        try:
            if isinstance(precomputed_df.columns, pd.MultiIndex):
                if full_sym not in precomputed_df.columns.get_level_values(0):
                    continue
                close = precomputed_df[full_sym]["Close"].dropna()
            else:
                close = precomputed_df["Close"].dropna()
            # Slice to <= as_of_date — that's where the leakage guard lives.
            # Normalize tz so the comparison works whether the index is
            # tz-aware (yfinance auto_adjust returns UTC-naive) or naive.
            if getattr(close.index, "tz", None) is not None:
                cutoff = as_of_ts.tz_localize(close.index.tz) if as_of_ts.tz is None else as_of_ts
            else:
                cutoff = as_of_ts.tz_localize(None) if as_of_ts.tz else as_of_ts
            close = close[close.index <= cutoff]
            if len(close) < 20:
                continue
            sub = _series_features(close)
            for k, v in sub.items():
                out[f"{short}_{k}"] = v
        except Exception:
            continue

    # Calendar features derived from as_of_date — not from "now"
    out["day_of_week"] = int(as_of_ts.dayofweek)
    out["is_eia_oil_day"] = (as_of_ts.dayofweek == 2)
    out["is_eia_gas_day"] = (as_of_ts.dayofweek == 3)
    out["month"] = int(as_of_ts.month)
    out["is_winter"] = as_of_ts.month in (11, 12, 1, 2, 3)
    out["is_summer"] = as_of_ts.month in (6, 7, 8)

    # Brent-WTI spread
    if "brent_close" in out and "wti_close" in out:
        b = out.get("brent_close")
        w = out.get("wti_close")
        if b is not None and w is not None:
            out["brent_wti_spread"] = b - w

    # --- EIA inventory features for backtest ---
    # The eia module enforces the publication-lag filter so historical
    # decisions never see data published after the decision date.
    if eia_oil_df is not None and not eia_oil_df.empty:
        try:
            out.update(eia.oil_inventory_features(
                as_of_date=as_of_ts, precomputed=eia_oil_df,
            ))
        except Exception:
            pass
    if eia_gas_df is not None and not eia_gas_df.empty:
        try:
            out.update(eia.natgas_storage_features(
                as_of_date=as_of_ts, precomputed=eia_gas_df,
            ))
        except Exception:
            pass

    # --- CFTC speculator positioning (backtest path) ---
    if cftc_df is not None and not cftc_df.empty:
        try:
            out.update(cftc.compute_cftc_features(
                cftc_df, as_of_date=as_of_ts,
            ))
        except Exception:
            pass

    # --- Weather (backtest path) ---
    if weather_df is not None and not weather_df.empty:
        try:
            out.update(weather.compute_weather_features(
                weather_df, as_of_date=as_of_ts,
            ))
        except Exception:
            pass

    return out


def etf_close_on(
    etf_df: pd.DataFrame,
    ticker: str,
    on_date,
) -> float | None:
    """Get the ETF's close on `on_date` (exact match — None if the ETF
    didn't trade that day, e.g. holiday or pre-IPO)."""
    if etf_df is None or etf_df.empty:
        return None
    ts = pd.Timestamp(on_date)
    try:
        if isinstance(etf_df.columns, pd.MultiIndex):
            if ticker not in etf_df.columns.get_level_values(0):
                return None
            close = etf_df[ticker]["Close"].dropna()
        else:
            close = etf_df["Close"].dropna()
        # Normalize tz on the comparison
        if getattr(close.index, "tz", None) is not None:
            target = ts.tz_localize(close.index.tz) if ts.tz is None else ts
        else:
            target = ts.tz_localize(None) if ts.tz else ts
        # Try exact-date match; tolerate timestamp-shaped dates.
        mask = close.index.normalize() == target.normalize()
        if mask.any():
            return float(close[mask].iloc[0])
    except Exception:
        return None
    return None


def trading_days_in_window(etf_df: pd.DataFrame) -> list:
    """Return the sorted list of trading-day timestamps available in
    `etf_df` — the union across all 4 ETFs so we don't miss any day
    one ticker traded but another didn't."""
    if etf_df is None or etf_df.empty:
        return []
    try:
        if isinstance(etf_df.columns, pd.MultiIndex):
            # Use the index from any first-level slice — all share the same index
            return sorted(etf_df.index.normalize().unique())
        return sorted(etf_df.index.normalize().unique())
    except Exception:
        return []

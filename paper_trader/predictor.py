"""Next-day bullish-opening predictor.

Goal: rank ETFs by how likely they are to gap UP at tomorrow's open.
Used by the agent's 3:30 PM ET buy slot.

Approach: a transparent composite momentum score (no ML), so the
user can audit it. Four sub-scores, each normalized 0-1:

  1. close_position    — where today's close sits in the day's range
                         (close-near-high = strong continuation signal)
  2. recent_momentum   — 5-day return sign + magnitude
  3. rsi_zone          — RSI 50-70 is the "trend continuing" sweet
                         spot; <40 or >75 is penalized
  4. volume_surge      — today's volume vs 20-day avg; surges often
                         precede continuation

Final score = weighted average. Top entries = most likely to open up.

This is a heuristic, not a backtested predictor. Documented honestly
so the user can read the picks with appropriate skepticism.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import UNIVERSE

LOGGER = logging.getLogger("paper_trader.predictor")

# Sub-score weights — sum to 1.0
WEIGHTS = {
    "close_position":  0.35,
    "recent_momentum": 0.30,
    "rsi_zone":        0.20,
    "volume_surge":    0.15,
}


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Last RSI value, period-14 default."""
    if len(series) < period + 1:
        return float("nan")
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def _score_ticker(ticker: str) -> dict[str, Any] | None:
    """Compute the bullish-opening score for a single ticker.

    Pulls 60 trading days so we have RSI + 20-day volume mean.
    Returns None if data is missing or too short.
    """
    try:
        df = yf.download(ticker, period="3mo", interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 25:
            return None
    except Exception as e:
        LOGGER.debug(f"predictor fetch {ticker} failed: {e}")
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    today_close = float(close.iloc[-1])
    today_high  = float(high.iloc[-1])
    today_low   = float(low.iloc[-1])

    # --- 1. close position in today's range ---
    rng = today_high - today_low
    close_pos = ((today_close - today_low) / rng) if rng > 0 else 0.5
    close_pos = float(np.clip(close_pos, 0.0, 1.0))

    # --- 2. 5-day return ---
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
    # Map -5%..+5% → 0..1, clipped
    momentum = float(np.clip((ret_5d + 0.05) / 0.10, 0.0, 1.0))

    # --- 3. RSI zone (50-70 = good, 40-50 = decent, <40 or >75 = bad) ---
    rsi14 = _rsi(close, 14)
    if np.isnan(rsi14):
        rsi_zone = 0.5
    elif 55 <= rsi14 <= 70:
        rsi_zone = 1.0  # ideal continuation zone
    elif 50 <= rsi14 < 55 or 70 < rsi14 <= 75:
        rsi_zone = 0.7
    elif 40 <= rsi14 < 50:
        rsi_zone = 0.4
    elif 75 < rsi14:
        rsi_zone = 0.2  # overbought — risk of pullback
    else:
        rsi_zone = 0.1  # oversold — momentum is weak

    # --- 4. volume vs 20-day average ---
    vol_ma = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float("nan")
    if vol_ma and vol_ma > 0 and not np.isnan(vol_ma):
        vol_ratio = float(volume.iloc[-1]) / vol_ma
    else:
        vol_ratio = 1.0
    # Map 0.5x..2.0x → 0..1
    vol_surge = float(np.clip((vol_ratio - 0.5) / 1.5, 0.0, 1.0))

    score = (
        WEIGHTS["close_position"]  * close_pos
        + WEIGHTS["recent_momentum"] * momentum
        + WEIGHTS["rsi_zone"]        * rsi_zone
        + WEIGHTS["volume_surge"]    * vol_surge
    )

    return {
        "ticker": ticker,
        "score": score,
        "close": today_close,
        "ret_5d_pct": ret_5d * 100,
        "rsi_14": rsi14,
        "close_pos": close_pos,
        "vol_ratio": vol_ratio,
    }


def rank_next_day_bullish(top_k: int = 10) -> list[dict[str, Any]]:
    """Score every ticker in the universe and return top_k by composite
    bullish-opening score (highest first)."""
    rows = [r for r in (_score_ticker(t) for t in UNIVERSE) if r]
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]

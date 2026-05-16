"""Live buy/sell recommendation engine for HOU/HOD/HNU/HND.

For each pair (oil and natgas), checks the current feature snapshot
and emits a recommendation. Since HOU/HOD are exact inverses (and
HNU/HND likewise), only one side of each pair can succeed on a given
day — so a bull signal on oil means BUY HOU + SELL HOD, while a bear
signal means BUY HOD + SELL HOU.

Validation status (from the 10-year backtest sweep):
  - Bull oil signals: VALIDATED (PF 1.55-5.25 across 4 windows)
  - Bear oil signals: EXPERIMENTAL (not backtested as a system —
    individual signal logic is symmetric to bull signals, but the
    bear ETF (HOD) has structurally faster decay so edge is unclear)
  - Natgas signals: EXPERIMENTAL (the natgas validation test failed
    catastrophically — PF 0.52 on 10y, account wiped out)

The UI labels each recommendation with its validation status so the
user can size confidence appropriately.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import features as feat_mod


# Conviction thresholds — same scale as the production agent uses
WEAK = 0.50
MODERATE = 0.65
STRONG = 0.80


def _check_bull_signals(features: dict, prefix: str) -> list[dict]:
    """Run the validated bull strategies and return all firing signals.

    `prefix` is "wti" for oil or "ng" for natgas — selects which
    underlying's features to read.
    """
    out: list[dict] = []
    rsi = features.get(f"{prefix}_rsi")
    bb_pos = features.get(f"{prefix}_bb_position")
    ret_1 = features.get(f"{prefix}_ret_1d_pct")
    ret_20 = features.get(f"{prefix}_ret_20d_pct")

    # Sharp dip in uptrend (oil_sharp_dip)
    if ret_1 is not None and ret_20 is not None:
        if ret_1 < -4.0 and ret_20 > 0:
            out.append({
                "name": "sharp_dip_strong",
                "conviction": 0.70,
                "label": f"Sharp drop ({ret_1:.1f}%) inside uptrend "
                         f"(20d {ret_20:+.1f}%)",
            })
        elif ret_1 < -2.0 and ret_20 > 2:
            out.append({
                "name": "sharp_dip",
                "conviction": 0.55,
                "label": f"Moderate drop ({ret_1:.1f}%) inside uptrend "
                         f"(20d {ret_20:+.1f}%)",
            })
        elif ret_1 < -1.5 and ret_20 > 4:
            out.append({
                "name": "sharp_dip_light",
                "conviction": 0.50,
                "label": f"Mild dip ({ret_1:.1f}%) in strong uptrend "
                         f"(20d {ret_20:+.1f}%)",
            })

    # Pullback in uptrend (oil_pullback_uptrend)
    if ret_20 is not None and rsi is not None:
        if ret_20 > 3.0 and 40 <= rsi <= 50:
            out.append({
                "name": "pullback_uptrend",
                "conviction": 0.55,
                "label": f"Healthy pullback: 20d {ret_20:+.1f}%, "
                         f"RSI {rsi:.1f} (40-50 zone)",
            })
        elif ret_20 > 5.0 and 35 <= rsi <= 55:
            out.append({
                "name": "pullback_uptrend_strong",
                "conviction": 0.50,
                "label": f"Strong uptrend pullback: 20d {ret_20:+.1f}%, "
                         f"RSI {rsi:.1f}",
            })

    # BB oversold (oil_bb_oversold) — exclusive: only when RSI not also
    # oversold (per the iter 31 design)
    if bb_pos is not None and rsi is not None and rsi >= 45:
        if bb_pos < 0.10:
            out.append({
                "name": "bb_oversold_strong",
                "conviction": 0.70,
                "label": f"BB position {bb_pos:.2f} (deep lower band) "
                         f"with RSI {rsi:.1f} mid-range",
            })
        elif bb_pos < 0.20:
            out.append({
                "name": "bb_oversold",
                "conviction": 0.55,
                "label": f"BB position {bb_pos:.2f} (lower band) "
                         f"with RSI {rsi:.1f} mid-range",
            })
    return out


def _check_bear_signals(features: dict, prefix: str) -> list[dict]:
    """Mirror of bull signals — fires when conditions reverse.
    EXPERIMENTAL: these aren't backtested as a system.
    """
    out: list[dict] = []
    rsi = features.get(f"{prefix}_rsi")
    bb_pos = features.get(f"{prefix}_bb_position")
    ret_1 = features.get(f"{prefix}_ret_1d_pct")
    ret_20 = features.get(f"{prefix}_ret_20d_pct")

    # Sharp rally in downtrend (bear mirror of sharp_dip)
    if ret_1 is not None and ret_20 is not None:
        if ret_1 > 4.0 and ret_20 < 0:
            out.append({
                "name": "sharp_rally_strong",
                "conviction": 0.70,
                "label": f"Sharp rally ({ret_1:+.1f}%) inside downtrend "
                         f"(20d {ret_20:.1f}%)",
            })
        elif ret_1 > 2.0 and ret_20 < -2:
            out.append({
                "name": "sharp_rally",
                "conviction": 0.55,
                "label": f"Moderate rally ({ret_1:+.1f}%) inside "
                         f"downtrend (20d {ret_20:.1f}%)",
            })

    # Bounce in downtrend (bear mirror of pullback)
    if ret_20 is not None and rsi is not None:
        if ret_20 < -3.0 and 50 <= rsi <= 60:
            out.append({
                "name": "bounce_downtrend",
                "conviction": 0.55,
                "label": f"Dead-cat bounce: 20d {ret_20:.1f}%, "
                         f"RSI {rsi:.1f} (50-60 zone)",
            })

    # BB overbought (bear mirror)
    if bb_pos is not None and rsi is not None and rsi <= 55:
        if bb_pos > 0.90:
            out.append({
                "name": "bb_overbought_strong",
                "conviction": 0.70,
                "label": f"BB position {bb_pos:.2f} (deep upper band) "
                         f"with RSI {rsi:.1f} mid-range",
            })
        elif bb_pos > 0.80:
            out.append({
                "name": "bb_overbought",
                "conviction": 0.55,
                "label": f"BB position {bb_pos:.2f} (upper band) "
                         f"with RSI {rsi:.1f} mid-range",
            })
    return out


def _pair_recommendation(
    features: dict,
    prefix: str,
    bull_ticker: str,
    bear_ticker: str,
    bull_validated: bool,
    bear_validated: bool,
) -> dict:
    """Compute the full recommendation for one pair."""
    # Regime check — same logic the live agent uses
    bear_regime = features.get(f"{prefix}_bear_regime", False)
    fast_bear = features.get(f"{prefix}_fast_bear", False)
    in_bear_regime = bool(bear_regime or fast_bear)

    # Get all firing signals (bull-side blocked by regime filter)
    bull_signals = (
        [] if in_bear_regime else _check_bull_signals(features, prefix)
    )
    bear_signals = _check_bear_signals(features, prefix)

    # Apply CFTC + OVX boosts on bull side (same as production agent)
    ovx_close = features.get("ovx_close")
    ovx_boost = 0.0
    if ovx_close is not None:
        if ovx_close > 60:
            ovx_boost = +0.10
        elif ovx_close > 40:
            ovx_boost = +0.05
    cftc_boost = 0.0
    days_old = features.get("cftc_days_old")
    if days_old is not None and days_old <= 14:
        if features.get("cftc_spec_extreme_short"):
            cftc_boost = +0.05
    for s in bull_signals:
        s["conviction"] = min(0.90, s["conviction"] + ovx_boost + cftc_boost)

    bull_top = max(bull_signals, key=lambda s: s["conviction"]) if bull_signals else None
    bear_top = max(bear_signals, key=lambda s: s["conviction"]) if bear_signals else None

    bull_conv = bull_top["conviction"] if bull_top else 0.0
    bear_conv = bear_top["conviction"] if bear_top else 0.0

    # Decide which side wins. If neither has a strong signal, fall
    # back to trend direction (5-day return) so we ALWAYS pick a side
    # — never "neutral". Low conviction (0.30) signals the fallback.
    direction = None
    fallback_reason = None

    if bull_conv >= WEAK and bull_conv > bear_conv:
        direction = "bull"
    elif bear_conv >= WEAK and bear_conv > bull_conv:
        direction = "bear"
    else:
        # Trend tiebreaker — short-term momentum
        ret_5 = features.get(f"{prefix}_ret_5d_pct")
        ret_1 = features.get(f"{prefix}_ret_1d_pct")
        # Prefer 5d, fall back to 1d, fall back to neutral-bull
        trend = ret_5 if ret_5 is not None else (ret_1 if ret_1 is not None else 0.0)
        if trend >= 0:
            direction = "bull"
            bull_conv = max(bull_conv, 0.30)
            fallback_reason = f"5d trend {trend:+.1f}% (bull fallback)"
        else:
            direction = "bear"
            bear_conv = max(bear_conv, 0.30)
            fallback_reason = f"5d trend {trend:+.1f}% (bear fallback)"
        # In bear regime, force bear bias even if trend tiebreaker says bull —
        # the agent itself wouldn't trade bull here.
        if in_bear_regime and direction == "bull":
            direction = "bear"
            bull_conv = 0.0
            bear_conv = max(bear_conv, 0.30)
            fallback_reason = "bear regime active (bear fallback)"

    # Per-ETF recommendation — always one BUY and one SELL
    if direction == "bull":
        bull_action = "BUY"
        bear_action = "SELL"
        primary_conv = bull_conv
    else:  # bear
        bull_action = "SELL"
        bear_action = "BUY"
        primary_conv = bear_conv

    return {
        "prefix": prefix,
        "direction": direction,
        "primary_conviction": primary_conv,
        "in_bear_regime": in_bear_regime,
        "fallback_reason": fallback_reason,
        "regime_reason": (
            "fast bear (30d return < -10%)" if features.get(f"{prefix}_fast_bear")
            else "bear regime (200d SMA declining)" if features.get(f"{prefix}_bear_regime")
            else None
        ),
        "bull": {
            "ticker": bull_ticker,
            "action": bull_action,
            "conviction": bull_conv,
            "validated": bull_validated,
            "signals": bull_signals,
            "top_signal": bull_top,
        },
        "bear": {
            "ticker": bear_ticker,
            "action": bear_action,
            "conviction": bear_conv,
            "validated": bear_validated,
            "signals": bear_signals,
            "top_signal": bear_top,
        },
    }


def compute_recommendations() -> dict:
    """Compute current recommendations for both pairs.

    Returns a dict with:
      as_of, oil (pair_rec), gas (pair_rec), features (raw snapshot)
    """
    features = feat_mod.fetch_features()
    return {
        "as_of": features.get("as_of_ts", datetime.now(timezone.utc).isoformat()),
        "oil": _pair_recommendation(
            features, "wti", "HOU.TO", "HOD.TO",
            bull_validated=True,    # +453% 10y test
            bear_validated=False,   # not backtested
        ),
        "gas": _pair_recommendation(
            features, "ng", "HNU.TO", "HND.TO",
            bull_validated=False,   # natgas 10y test failed
            bear_validated=False,
        ),
        "features": features,
    }

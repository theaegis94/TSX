"""Strategies that emit (ticker, conviction) signals.

Each strategy is a small class with a `.signal(features)` method. The
agent runs every enabled strategy and picks the highest-conviction
signal across all of them to act on.

Conviction is on [0, 1]. We use:
   0.0      = no signal (don't trade)
   0.3–0.5  = weak / speculative
   0.5–0.7  = solid setup
   0.7–1.0  = strong confluence

A strategy's signal is for the underlying (oil vs gas, up vs down).
The agent translates that into the right ETF (HOU/HOD/HNU/HND).

Week 1 ships 3 deliberately simple strategies. They're a baseline — we
expect mediocre performance and will improve once EIA and weather data
are wired up. The point of week 1 is proving the infra works.
"""
from __future__ import annotations

from typing import Any


class Strategy:
    """Base class. Subclasses set `name` + `description` and implement
    `signal(features) -> (ticker, conviction)`.

    Return ("HOU.TO" | "HOD.TO" | "HNU.TO" | "HND.TO", float) when a
    signal fires, or (None, 0.0) when there's nothing to do.
    """
    name: str = "abstract"
    description: str = ""

    def signal(
        self, features: dict[str, Any]
    ) -> tuple[str | None, float]:
        raise NotImplementedError


class OilRsiReversion(Strategy):
    """When WTI RSI is extreme, bet on mean reversion of the underlying.
    Oversold WTI (RSI<30) → long HOU (2x bull oil).
    Overbought WTI (RSI>70) → long HOD (2x bear oil).
    """
    name = "oil_rsi_reversion"
    description = (
        "Buy 2x-bull oil (HOU) when WTI RSI(14) < 30; buy 2x-bear oil "
        "(HOD) when RSI > 70. Bets on RSI mean reversion in the "
        "underlying."
    )

    def signal(self, features):
        rsi = features.get("wti_rsi")
        if rsi is None:
            return (None, 0.0)
        if rsi < 25:
            # Deep oversold — strong conviction
            return ("HOU.TO", 0.80)
        if rsi < 30:
            return ("HOU.TO", 0.60)
        if rsi > 75:
            return ("HOD.TO", 0.80)
        if rsi > 70:
            return ("HOD.TO", 0.60)
        return (None, 0.0)


class NatgasMacdCross(Strategy):
    """Natgas MACD crossover. Bullish cross → HNU. Bearish cross → HND.
    Confirms with histogram direction so we don't fire on flat noise.
    """
    name = "natgas_macd_cross"
    description = (
        "Buy 2x-bull natgas (HNU) on bullish MACD cross on NG=F; "
        "buy 2x-bear (HND) on bearish cross. Histogram must agree."
    )

    def signal(self, features):
        bull = features.get("ng_macd_cross_bull", False)
        bear = features.get("ng_macd_cross_bear", False)
        hist = features.get("ng_macd_hist")
        if hist is None:
            return (None, 0.0)
        if bull and hist > 0:
            return ("HNU.TO", 0.65)
        if bear and hist < 0:
            return ("HND.TO", 0.65)
        return (None, 0.0)


class OilInventoryDrawdown(Strategy):
    """When weekly EIA oil inventory drew much more than the recent
    4-week trailing trend, supply is tighter than expected — bullish
    crude → buy HOU (2x bull oil).

    Only fires within 2 days of the EIA release (Wed AM ET) so we
    don't act on stale data.
    """
    name = "oil_inv_drawdown"
    description = (
        "Buy HOU when crude oil inventory drew far below the 4-week "
        "trailing average change (bullish supply surprise). Fires "
        "only within 2 days of the EIA Wed release."
    )

    def signal(self, features):
        surprise = features.get("oil_inv_surprise")
        days_old = features.get("oil_inv_days_since_report")
        if surprise is None or days_old is None or days_old > 2:
            return (None, 0.0)
        # surprise is in thousand barrels. >5,000 = ~5M bbl move
        # Negative surprise = bigger draw than expected = bullish
        if surprise < -5000:
            return ("HOU.TO", 0.80)
        if surprise < -2500:
            return ("HOU.TO", 0.65)
        return (None, 0.0)


class OilInventoryBuild(Strategy):
    """Mirror of OilInventoryDrawdown: bigger-than-expected build =
    surplus → bearish crude → buy HOD (2x bear oil)."""
    name = "oil_inv_build"
    description = (
        "Buy HOD when crude oil inventory built far above the 4-week "
        "trailing average change (bearish supply surprise). Fires "
        "only within 2 days of the EIA Wed release."
    )

    def signal(self, features):
        surprise = features.get("oil_inv_surprise")
        days_old = features.get("oil_inv_days_since_report")
        if surprise is None or days_old is None or days_old > 2:
            return (None, 0.0)
        if surprise > 5000:
            return ("HOD.TO", 0.80)
        if surprise > 2500:
            return ("HOD.TO", 0.65)
        return (None, 0.0)


class NatgasStorageDrawdown(Strategy):
    """When EIA gas storage drew far more than the 4-week trailing
    trend, supply tighter than expected → bullish natgas → buy HNU.

    Fires within 2 days of the EIA Thu release.
    """
    name = "gas_storage_drawdown"
    description = (
        "Buy HNU when natgas storage drew far below the 4-week "
        "trailing average change. Fires only within 2 days of the "
        "EIA Thu release."
    )

    def signal(self, features):
        surprise = features.get("gas_stor_surprise")
        days_old = features.get("gas_stor_days_since_report")
        if surprise is None or days_old is None or days_old > 2:
            return (None, 0.0)
        # surprise is in Bcf
        if surprise < -50:
            return ("HNU.TO", 0.80)
        if surprise < -25:
            return ("HNU.TO", 0.65)
        return (None, 0.0)


class NatgasStorageBuild(Strategy):
    """Mirror: bigger-than-expected build = surplus → bearish → HND."""
    name = "gas_storage_build"
    description = (
        "Buy HND when natgas storage built far above the 4-week "
        "trailing average change. Fires only within 2 days of the "
        "EIA Thu release."
    )

    def signal(self, features):
        surprise = features.get("gas_stor_surprise")
        days_old = features.get("gas_stor_days_since_report")
        if surprise is None or days_old is None or days_old > 2:
            return (None, 0.0)
        if surprise > 50:
            return ("HND.TO", 0.80)
        if surprise > 25:
            return ("HND.TO", 0.65)
        return (None, 0.0)


class DxyOilInverse(Strategy):
    """Dollar weakness with oil holding up = oil bullish. Strong DXY
    drop (>0.5%) on a day WTI was flat or up → buy HOU next bar.
    Mirror: strong DXY rally + WTI weak → buy HOD.
    """
    name = "dxy_oil_inverse"
    description = (
        "Buy 2x-bull oil (HOU) when USD dropped >0.5% and WTI was "
        "non-negative — dollar weakness is a tailwind for oil. "
        "Mirror logic for HOD on dollar strength + oil weakness."
    )

    def signal(self, features):
        dxy_ret = features.get("dxy_ret_1d_pct")
        wti_ret = features.get("wti_ret_1d_pct")
        if dxy_ret is None or wti_ret is None:
            return (None, 0.0)
        if dxy_ret < -0.5 and wti_ret >= 0:
            return ("HOU.TO", 0.55)
        if dxy_ret > 0.5 and wti_ret <= 0:
            return ("HOD.TO", 0.55)
        return (None, 0.0)


# Registry of all strategies the agent should run on startup. Order
# doesn't matter — the agent picks the highest-conviction signal across
# all of them.
ALL_STRATEGIES: list[Strategy] = [
    # Price-pattern baselines (the 5-year backtest showed these are
    # likely too weak on 2x leveraged ETFs — keeping them for now as
    # a control / comparison group).
    OilRsiReversion(),
    NatgasMacdCross(),
    DxyOilInverse(),
    # EIA-driven catalyst strategies — the real bet.
    # Inventory surprise vs 4-week trailing trend has historically
    # been THE biggest mover for these commodities.
    OilInventoryDrawdown(),
    OilInventoryBuild(),
    NatgasStorageDrawdown(),
    NatgasStorageBuild(),
]


def run_all_strategies(
    features: dict[str, Any],
    enabled_names: set[str] | None = None,
) -> list[dict]:
    """Run every (optionally filtered) strategy and return a list of
    raw signals — even no-signal results — so we can log them.

    Returns a list of dicts:
      {"strategy": name, "ticker": str|None, "conviction": float}
    """
    out: list[dict] = []
    for strat in ALL_STRATEGIES:
        if enabled_names is not None and strat.name not in enabled_names:
            continue
        try:
            ticker, conv = strat.signal(features)
        except Exception:
            ticker, conv = (None, 0.0)
        out.append({
            "strategy": strat.name,
            "ticker": ticker,
            "conviction": float(conv),
        })
    return out


def best_signal(signals: list[dict]) -> dict | None:
    """Pick the single highest-conviction signal across all strategies.
    Returns None if no strategy fired (all conviction 0)."""
    fired = [s for s in signals if s.get("ticker") and s.get("conviction", 0) > 0]
    if not fired:
        return None
    return max(fired, key=lambda s: s["conviction"])

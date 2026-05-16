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

# === Strategy mode toggle ===
# Iteration result: the 5-year backtest showed bear-ETF positions
# (HOD/HND) have systematically lower win rates and worse PF than the
# bull-ETF positions (HOU/HNU). Bear ETFs decay faster than bull ETFs
# because shorting + daily rebalancing costs more than going long.
# When BULL_ONLY is True, every strategy that would have emitted a HOD
# or HND signal returns None instead.
BULL_ONLY = True
BEAR_TICKERS = {"HOD.TO", "HND.TO"}


def _bull_only_filter(sig: tuple[str | None, float]) -> tuple[str | None, float]:
    """Strip bear-ETF signals when BULL_ONLY is enabled."""
    if not BULL_ONLY:
        return sig
    ticker, conv = sig
    if ticker in BEAR_TICKERS:
        return (None, 0.0)
    return sig


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


class OilBollingerOversold(Strategy):
    """Buy HOU when WTI's price is touching or below the lower
    Bollinger band — a different oversold signal than RSI that fires
    on different days. The hypothesis is that combining two
    uncorrelated mean-reversion signals lets us capture more setups
    without lowering signal quality.
    """
    name = "oil_bb_oversold"
    description = (
        "Buy HOU when WTI Bollinger position < 0.10 (touching lower "
        "band) — a deep mean-reversion entry on the bull side only."
    )

    def signal(self, features):
        bb_pos = features.get("wti_bb_position")
        rsi = features.get("wti_rsi")
        if bb_pos is None:
            return (None, 0.0)
        # Iter 31: only fire when RSI ISN'T also oversold — otherwise
        # we'd just be a weaker duplicate of OilRsiReversion. By firing
        # exclusively on "BB oversold + RSI mid-range" days, we catch a
        # genuinely different setup (sharp single-day drop without
        # cumulative weakness).
        if rsi is not None and rsi < 45:
            return (None, 0.0)
        if bb_pos < 0.10:
            return ("HOU.TO", 0.70)
        if bb_pos < 0.20:
            return ("HOU.TO", 0.55)
        return (None, 0.0)


class OilRsiReversion(Strategy):
    """Broader RSI-based oil entries. Was conservative (<30 only);
    now graded thresholds from <25 down to <40 to fire more often.
    """
    name = "oil_rsi_reversion"
    description = (
        "Buy HOU on WTI RSI mean reversion, multi-tier thresholds."
    )

    def signal(self, features):
        rsi = features.get("wti_rsi")
        if rsi is None:
            return (None, 0.0)
        if rsi < 25:
            return ("HOU.TO", 0.80)
        if rsi < 30:
            return ("HOU.TO", 0.65)
        if rsi < 35:
            return ("HOU.TO", 0.55)
        if rsi < 40:
            return ("HOU.TO", 0.50)
        # Mirror for short side (bull_only filter strips these)
        if rsi > 70:
            return ("HOD.TO", 0.60)
        return (None, 0.0)


class OilMacdMomentum(Strategy):
    """Buy HOU only on actual MACD bullish cross — much rarer signal
    than 'hist > 0' but should carry real momentum information.
    Iteration-11 lesson: the lax hist>0 branch fired 299 times at
    PF 0.68 (catastrophic) — removed."""
    name = "oil_macd_momentum"
    description = "Buy HOU on confirmed WTI MACD bullish cross."

    def signal(self, features):
        bull = features.get("wti_macd_cross_bull", False)
        hist = features.get("wti_macd_hist")
        if hist is None:
            return (None, 0.0)
        if bull and hist > 0:
            return ("HOU.TO", 0.65)
        return (None, 0.0)


class OilSharpDip(Strategy):
    """Buy HOU on a sharp 1-day drop (>2%) when the broader trend is
    still up. 'Buy the panic flush' — different timing than RSI or BB
    oversold, fires on single-day events rather than cumulative."""
    name = "oil_sharp_dip"
    description = (
        "Buy HOU when WTI dropped >2% in one day AND 20d return is "
        "still positive — buying a panic dip inside an uptrend."
    )

    def signal(self, features):
        ret_1 = features.get("wti_ret_1d_pct")
        ret_20 = features.get("wti_ret_20d_pct")
        if ret_1 is None or ret_20 is None:
            return (None, 0.0)
        if ret_1 < -4.0 and ret_20 > 0:
            return ("HOU.TO", 0.70)
        if ret_1 < -2.0 and ret_20 > 2:
            return ("HOU.TO", 0.55)
        # Iteration 14: loosen to capture more dips
        if ret_1 < -1.5 and ret_20 > 4:
            return ("HOU.TO", 0.50)
        return (None, 0.0)


class NatgasPullbackInUptrend(Strategy):
    """Natgas mirror of OilPullbackInUptrend — buy HNU on NG=F pullback
    inside a confirmed uptrend."""
    name = "natgas_pullback_uptrend"
    description = (
        "Buy HNU when NG=F 20d return is positive AND RSI is in a "
        "40-50 pullback zone."
    )

    def signal(self, features):
        ret_20 = features.get("ng_ret_20d_pct")
        rsi = features.get("ng_rsi")
        if ret_20 is None or rsi is None:
            return (None, 0.0)
        if ret_20 > 3.0 and 40 <= rsi <= 50:
            return ("HNU.TO", 0.55)
        if ret_20 > 5.0 and 35 <= rsi <= 55:
            return ("HNU.TO", 0.50)
        return (None, 0.0)


class NatgasSharpDip(Strategy):
    """Natgas mirror of OilSharpDip — buy HNU on sharp 1-day drop while
    20d trend remains up."""
    name = "natgas_sharp_dip"
    description = (
        "Buy HNU when NG=F dropped >2% in one day AND 20d return is "
        "still positive — buying a panic dip inside a natgas uptrend."
    )

    def signal(self, features):
        ret_1 = features.get("ng_ret_1d_pct")
        ret_20 = features.get("ng_ret_20d_pct")
        if ret_1 is None or ret_20 is None:
            return (None, 0.0)
        if ret_1 < -4.0 and ret_20 > 0:
            return ("HNU.TO", 0.70)
        if ret_1 < -2.0 and ret_20 > 2:
            return ("HNU.TO", 0.55)
        if ret_1 < -1.5 and ret_20 > 4:
            return ("HNU.TO", 0.50)
        return (None, 0.0)


class OilDxyTailwind(Strategy):
    """Buy HOU when DXY weakens substantially over 5 days — dollar
    weakness historically lifts dollar-priced commodities. Fires on
    macro days unrelated to oil's own price action."""
    name = "oil_dxy_tailwind"
    description = (
        "Buy HOU when DXY 5d return < -1.5% (sustained dollar "
        "weakness — tailwind for oil-priced-in-USD)."
    )

    def signal(self, features):
        dxy_ret = features.get("dxy_ret_5d_pct")
        if dxy_ret is None:
            return (None, 0.0)
        if dxy_ret < -2.5:
            return ("HOU.TO", 0.65)
        if dxy_ret < -1.5:
            return ("HOU.TO", 0.55)
        return (None, 0.0)


class OilEarlyBounce(Strategy):
    """Catch the earliest sign of a reversal: 3-day return is negative
    but today's return is positive — the 'first green day after a
    selloff' pattern."""
    name = "oil_early_bounce"
    description = (
        "Buy HOU when WTI 5d return < -3% but today's return > 0 — "
        "first green day after a selloff."
    )

    def signal(self, features):
        ret_1 = features.get("wti_ret_1d_pct")
        ret_5 = features.get("wti_ret_5d_pct")
        if ret_1 is None or ret_5 is None:
            return (None, 0.0)
        if ret_5 < -5.0 and ret_1 > 0.5:
            return ("HOU.TO", 0.65)
        if ret_5 < -3.0 and ret_1 > 0.5:
            return ("HOU.TO", 0.55)
        return (None, 0.0)


class OilTrendContinuation(Strategy):
    """Trend-follow: WTI 20d return strongly positive AND RSI in
    mid-range (40-60) — buying the trend, not the bottom. Fires on
    different days than mean-reversion signals."""
    name = "oil_trend_continuation"
    description = (
        "Buy HOU when WTI 20d return is strongly positive (>8%) "
        "and RSI is mid-range — trend-following entry."
    )

    def signal(self, features):
        ret_20 = features.get("wti_ret_20d_pct")
        rsi = features.get("wti_rsi")
        if ret_20 is None or rsi is None:
            return (None, 0.0)
        if ret_20 > 12 and 45 <= rsi <= 60:
            return ("HOU.TO", 0.60)
        if ret_20 > 8 and 45 <= rsi <= 60:
            return ("HOU.TO", 0.50)
        return (None, 0.0)


class OilPullbackInUptrend(Strategy):
    """Buy the dip: when 20-day return is positive (uptrend) AND RSI
    is in mid-range pullback territory (40-50). This is "buy weakness
    in strength" — different from RSI extreme oversold."""
    name = "oil_pullback_uptrend"
    description = (
        "Buy HOU when WTI 20d return is positive AND RSI is in a "
        "40-50 pullback zone (buy-the-dip in established uptrend)."
    )

    def signal(self, features):
        ret_20 = features.get("wti_ret_20d_pct")
        rsi = features.get("wti_rsi")
        if ret_20 is None or rsi is None:
            return (None, 0.0)
        if ret_20 > 3.0 and 40 <= rsi <= 50:
            return ("HOU.TO", 0.55)
        if ret_20 > 5.0 and 35 <= rsi <= 55:
            return ("HOU.TO", 0.50)
        return (None, 0.0)


class NatgasRsiReversion(Strategy):
    """ITER 27: Iteration 9 tested natgas RSI with broad thresholds —
    PF was 0.79 across 130 trades (loser). Hypothesis: only the
    DEEPEST oversold has any edge. Tightening to RSI<25 only, with
    high conviction to attract the position-sizing boost."""
    name = "natgas_rsi_deep"
    description = "Buy HNU only on natgas RSI<25 (deepest oversold)."

    def signal(self, features):
        rsi = features.get("ng_rsi")
        if rsi is None:
            return (None, 0.0)
        if rsi < 20:
            return ("HNU.TO", 0.80)
        if rsi < 25:
            return ("HNU.TO", 0.70)
        return (None, 0.0)


class NatgasBollingerOversold(Strategy):
    """Natgas BB position oversold (analog of OilBollingerOversold)."""
    name = "natgas_bb_oversold"
    description = "Buy HNU when NG=F Bollinger position is low."

    def signal(self, features):
        bb_pos = features.get("ng_bb_position")
        if bb_pos is None:
            return (None, 0.0)
        if bb_pos < 0.05:
            return ("HNU.TO", 0.70)
        if bb_pos < 0.15:
            return ("HNU.TO", 0.55)
        if bb_pos < 0.25:
            return ("HNU.TO", 0.50)
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
    # Iter 46 production config: oil-only edge, regime-filtered.
    # Natgas tested separately (PF 0.52 on sharp dip — broken on
    # natgas pair, kept in code for future research).
    OilBollingerOversold(),
    OilPullbackInUptrend(),
    OilSharpDip(),
    # NatgasPullbackInUptrend(),  # PF 1.33 alone (decent but not used)
    # NatgasSharpDip(),  # PF 0.52 on 10y — broken
    # NatgasBollingerOversold(),  # never fires due to RSI competition
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
    # ITER 45: faster regime detector. Skip if EITHER condition fires:
    #   - bear_regime (200d SMA filter — slow but durable)
    #   - fast_bear (30-day return < -10% — catches crashes early)
    bear_regime = features.get("wti_bear_regime", False)
    fast_bear = features.get("wti_fast_bear", False)
    if bear_regime or fast_bear:
        return [{
            "strategy": s.name, "ticker": None, "conviction": 0.0,
        } for s in ALL_STRATEGIES
            if enabled_names is None or s.name in enabled_names]

    # CFTC conviction modifier — boost-only (iter 36 reduce hurt us)
    cftc_boost = 0.0
    days_old = features.get("cftc_days_old")
    if days_old is not None and days_old <= 14:
        if features.get("cftc_spec_extreme_short"):
            cftc_boost = +0.05

    # Iter 39: high OVX = mean reversion pays best. Moderate boost.
    # (iter 43 tried bigger boosts — pushed result into fantasy zone.)
    ovx = features.get("ovx_close")
    ovx_boost = 0.0
    if ovx is not None:
        if ovx > 60:
            ovx_boost = +0.10
        elif ovx > 40:
            ovx_boost = +0.05

    for strat in ALL_STRATEGIES:
        if enabled_names is not None and strat.name not in enabled_names:
            continue
        try:
            ticker, conv = strat.signal(features)
        except Exception:
            ticker, conv = (None, 0.0)
        # Apply the bull-only filter (no-op when BULL_ONLY is False)
        ticker, conv = _bull_only_filter((ticker, conv))
        # Combined conviction boosts for bull-side signals
        if ticker in ("HOU.TO", "HNU.TO"):
            conv = max(0.0, min(0.90, conv + cftc_boost + ovx_boost))
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

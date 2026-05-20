"""Universe of commodity-themed Canadian ETFs the paper trader picks from.

Two flavors:

  Pure commodity (futures-backed or bullion-backed):
    HOU.TO / HOD.TO   — WTI crude oil 2x bull / bear
    HNU.TO / HND.TO   — Henry Hub natgas 2x bull / bear
    CGL.TO            — gold bullion (CAD-hedged)
    MNT.TO            — Royal Canadian Mint physical gold trust

  Commodity-correlated equity (single-resource sector ETFs):
    XEG.TO            — iShares Canadian energy producers
    ZEO.TO            — BMO equal-weight oil & gas
    XGD.TO            — iShares S&P/TSX gold miners
    ZJG.TO            — BMO junior gold miners
    XMA.TO            — iShares Canadian materials

11 tickers total. Concentrated by design so the morning intraday
top-mover pick + the 3:30pm bullish-opening pick are always
commodity-themed.

Trade-off note: HOU/HOD and HNU/HND are inverse pairs. The two-slot
agent can in theory go long HOU intraday AND long HOD overnight on
the same day if momentum + bullish-opening score on opposite sides.
That's accidentally a partial hedge, not necessarily bad — but worth
being aware of when reading the trade log.
"""
from __future__ import annotations

UNIVERSE: list[str] = [
    # --- Pure commodity (futures + bullion) ---
    "HOU.TO", "HOD.TO",     # WTI 2x bull / bear
    "HNU.TO", "HND.TO",     # Natgas 2x bull / bear
    "CGL.TO",               # Gold bullion (CAD-hedged)
    "MNT.TO",               # Royal Cdn Mint gold trust
    # --- Commodity-correlated equity sectors ---
    "XEG.TO",               # Canadian energy producers
    "ZEO.TO",               # BMO equal-weight oil & gas
    "XGD.TO",               # Gold miners
    "ZJG.TO",               # Junior gold miners
    "XMA.TO",               # Canadian materials
]

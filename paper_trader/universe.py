"""Universe of strict commodity Canadian ETFs the paper trader picks from.

"Commodity ETF" in the strict sense = the fund holds either:
  (a) commodity futures contracts (oil, natgas), or
  (b) physical bullion (gold)

NOT equity ETFs of commodity-producing companies. That distinction
matters because equity ETFs trade like stocks (company-specific
risk, earnings, dividends) whereas futures/bullion ETFs are pure
commodity exposure.

Six tickers across three underlying commodities:

  WTI crude oil:
    HOU.TO   2x daily long WTI
    HOD.TO   2x daily short WTI

  Henry Hub natural gas:
    HNU.TO   2x daily long natgas
    HND.TO   2x daily short natgas

  Gold bullion:
    CGL.TO   iShares gold bullion (CAD-hedged)
    MNT.TO   Royal Canadian Mint physical gold trust

Notes on the small universe:
  - On any given day the agent's morning + evening picks will likely
    surface different sides of these 3 underlyings (oil / gas / gold).
  - HOU/HOD and HNU/HND are inverse pairs. The predictor naturally
    picks the bull side when momentum is up — but the two-slot agent
    CAN technically go long the bull on one slot and the bear on the
    other (partial hedge). Watch the trade log if that happens.
  - Leveraged ETFs have daily-reset decay over time. The agent's
    intraday + overnight horizons are short enough that decay is
    minimal per trade, but compounded over many trades it adds up.
"""
from __future__ import annotations

UNIVERSE: list[str] = [
    # WTI crude oil 2x bull/bear
    "HOU.TO", "HOD.TO",
    # Henry Hub natural gas 2x bull/bear
    "HNU.TO", "HND.TO",
    # Gold bullion
    "CGL.TO",
    "MNT.TO",
]

"""Canadian-ETF paper trader.

Two-slot rotation:
  Intraday  : BUY 10:00 AM ET → SELL 3:45 PM ET same day
  Overnight : BUY 3:30 PM ET → SELL 9:55 AM ET next day

Buy picks are sourced from:
  intraday   : top % gainer-from-open across ~111 Canadian ETFs
  overnight  : composite next-day-bullish-opening score

25% of equity per buy. Auto-retroactive execution: page loads
backfill missed scheduled trades using actual historical prices.
"""
from .universe import UNIVERSE
from .movers import compute_top_movers
from .predictor import rank_next_day_bullish
from .agent import (
    tick,
    next_scheduled_event,
    get_portfolio_value,
    SCHEDULE,
    ALLOCATION_PCT,
)
from .storage import (
    init_db,
    get_cash,
    get_initial_capital,
    get_all_positions,
    get_trade_history,
    reset_all,
    DEFAULT_INITIAL_CAPITAL,
)
from .backtest import run_backtest, run_backtest_long

__all__ = [
    "UNIVERSE",
    "compute_top_movers",
    "rank_next_day_bullish",
    "tick",
    "next_scheduled_event",
    "get_portfolio_value",
    "SCHEDULE",
    "ALLOCATION_PCT",
    "init_db",
    "get_cash",
    "get_initial_capital",
    "get_all_positions",
    "get_trade_history",
    "reset_all",
    "DEFAULT_INITIAL_CAPITAL",
    "run_backtest",
    "run_backtest_long",
]

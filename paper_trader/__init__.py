"""Paper-trading agent for HOU/HOD/HNU/HND.

This is a self-contained background daemon plus a Streamlit dashboard tab.

Architecture:
  - storage.py   — two SQLite DBs (paper_trader.db is resettable; the
                   strategies_state.db persists across resets)
  - features.py  — daily feature computation (RSI/MACD on CL=F, NG=F, DXY)
  - strategies.py — 3 starting strategies that consume features and emit
                   (ticker, conviction) signals
  - exits.py     — exit logic (±5%, signal-flip, 5-day timeout)
  - agent.py     — main loop, runs forever, polls every 5 min during
                   market hours and every 30 min off-hours

Public re-exports below let the Streamlit UI talk to the agent's state
without importing every submodule.
"""
from .storage import (
    DEFAULT_INITIAL_CAPITAL,
    get_balance,
    get_open_position,
    get_trade_history,
    get_equity_curve,
    get_strategy_stats,
    get_heartbeat,
    get_signal_log,
    is_agent_paused,
    set_agent_paused,
    reset_capital,
    init_databases,
    count_sim_trades,
    clear_sim_trades,
    compute_sim_stats,
    TICKERS,
)
from .simulate import run_backtest
from .recommend import compute_recommendations

__all__ = [
    "DEFAULT_INITIAL_CAPITAL",
    "get_balance",
    "get_open_position",
    "get_trade_history",
    "get_equity_curve",
    "get_strategy_stats",
    "get_heartbeat",
    "get_signal_log",
    "is_agent_paused",
    "set_agent_paused",
    "reset_capital",
    "init_databases",
    "count_sim_trades",
    "clear_sim_trades",
    "compute_sim_stats",
    "run_backtest",
    "compute_recommendations",
    "TICKERS",
]

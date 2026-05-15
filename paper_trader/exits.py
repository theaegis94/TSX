"""Exit-signal logic.

Four reasons to exit a position:
  1. Stop loss (-5% from entry price)
  2. Take profit (+5% from entry price)
  3. Signal flip (a strategy now recommends a different ETF in the
     opposite direction of the current position)
  4. 5-day max hold (hard timeout regardless of P&L)

The agent calls `check_exit(position, current_price, current_signals)`
on every market-hours cycle. Returns an exit reason string if the
position should be closed, or None if it should continue holding.
"""
from __future__ import annotations

from datetime import datetime, timezone

# All thresholds in one place so they're easy to tune
STOP_LOSS_PCT = -5.0
TAKE_PROFIT_PCT = 5.0
MAX_HOLD_DAYS = 5

# Pair definitions — used to detect "opposite direction" for signal-flip exits
PAIR_OPPOSITE = {
    "HOU.TO": "HOD.TO",
    "HOD.TO": "HOU.TO",
    "HNU.TO": "HND.TO",
    "HND.TO": "HNU.TO",
}


def _parse_iso(s: str) -> datetime:
    """Parse an ISO datetime, treating naive strings as UTC."""
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Pandas-style 'YYYY-MM-DD HH:MM:SS+00:00' should be ISO-compatible.
        # If something weird gets through, default to now to avoid crash.
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_held(position: dict) -> float:
    entry = _parse_iso(position["entry_date"])
    now = datetime.now(timezone.utc)
    return (now - entry).total_seconds() / 86400.0


def pnl_pct(position: dict, current_price: float) -> float:
    """Mark-to-market P&L percent vs entry (excludes commission for
    speed — exit logic uses gross P&L for decisions; the trade-log
    P&L is net of commission)."""
    entry = float(position["entry_price"])
    if entry <= 0:
        return 0.0
    return (current_price - entry) / entry * 100.0


def check_exit(
    position: dict,
    current_price: float,
    best_signal: dict | None,
) -> str | None:
    """Return one of {"stop_loss", "take_profit", "signal_flip",
    "timeout", None}. None means hold.
    """
    if not position:
        return None

    pct = pnl_pct(position, current_price)

    # 1. Stop loss
    if pct <= STOP_LOSS_PCT:
        return "stop_loss"

    # 2. Take profit
    if pct >= TAKE_PROFIT_PCT:
        return "take_profit"

    # 3. Signal flip — a current strategy is firing in the opposite
    #    direction with non-trivial conviction
    if best_signal and best_signal.get("ticker"):
        opp = PAIR_OPPOSITE.get(position["ticker"])
        if (best_signal["ticker"] == opp
                and best_signal.get("conviction", 0) >= 0.6):
            return "signal_flip"

    # 4. Hard timeout
    if days_held(position) >= MAX_HOLD_DAYS:
        return "timeout"

    return None

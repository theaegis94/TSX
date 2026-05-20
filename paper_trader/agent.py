"""The scheduled paper-trading agent.

Trades on a fixed daily schedule (Eastern Time):

  10:00 AM  BUY  intraday  — top % gainer from today's open
  3:30  PM  BUY  overnight — top next-day-bullish-opening pick
  3:45  PM  SELL intraday   (held ~5h 45m intraday)
  9:55  AM  SELL overnight  (held ~18h, captures opening gap)

Each BUY uses 25% of CURRENT total equity (cash + open positions
marked-to-market).

Execution is auto-retroactive: when `tick()` is called (every page
load), the agent looks at the last action timestamp, enumerates all
scheduled events that should have fired since then, and executes
each one at the actual historical price for that timestamp. This
means the agent's history is "complete" even if the Streamlit page
was only opened once that day.

Skips weekends and US/CA market holidays implicitly by relying on
yfinance to return no data for those days.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

from . import storage
from .movers import compute_top_mover_at, get_price_at
from .predictor import rank_next_day_bullish

LOGGER = logging.getLogger("paper_trader.agent")
ET = ZoneInfo("America/Toronto")  # handles EST/EDT transitions

# Schedule definition: (hour, minute, action, slot)
SCHEDULE = [
    (9, 55,  "SELL", "overnight"),  # sell yesterday's overnight pick
    (10, 0,  "BUY",  "intraday"),   # buy today's intraday pick
    (15, 30, "BUY",  "overnight"),  # buy today's overnight pick
    (15, 45, "SELL", "intraday"),   # sell intraday pick
]
ALLOCATION_PCT = 0.25  # 25% of equity per buy


def _now_et() -> datetime:
    return datetime.now(ET)


def _enumerate_events(from_ts: datetime, to_ts: datetime) -> list[dict[str, Any]]:
    """Generate every scheduled event in (from_ts, to_ts] in chronological
    order. Skips weekends (Saturday=5, Sunday=6)."""
    events: list[dict[str, Any]] = []
    if to_ts <= from_ts:
        return events
    cursor = from_ts.replace(microsecond=0)
    while cursor.date() <= to_ts.date():
        if cursor.weekday() < 5:  # Mon-Fri
            for hour, minute, action, slot in SCHEDULE:
                event_ts = cursor.replace(hour=hour, minute=minute, second=0)
                if event_ts > from_ts and event_ts <= to_ts:
                    events.append({
                        "ts": event_ts,
                        "action": action,
                        "slot": slot,
                    })
        cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    return sorted(events, key=lambda e: e["ts"])


def _mark_to_market(ts: datetime) -> float:
    """Total equity = cash + sum of (shares * current_price) for open
    positions, valued AT `ts`."""
    cash = storage.get_cash()
    total = cash
    for slot, pos in storage.get_all_positions().items():
        px = get_price_at(pos["ticker"], ts)
        if px is None:
            px = float(pos["entry_price"])  # fallback to entry
        total += float(pos["shares"]) * px
    return total


def _execute_buy(slot: str, event_ts: datetime) -> dict[str, Any] | None:
    """Pick a ticker for `slot` at `event_ts`, execute the buy, log it."""
    existing = storage.get_position(slot)
    if existing is not None:
        LOGGER.info(f"[{event_ts}] {slot} buy skipped — slot already filled")
        return None

    # Pick ticker
    if slot == "intraday":
        movers = compute_top_mover_at(event_ts, top_k=5)
        if not movers:
            LOGGER.warning(f"[{event_ts}] no intraday candidates")
            return None
        pick = movers[0]["ticker"]
        rationale = (
            f"Top intraday gainer (+{movers[0]['change_pct']:.2f}% from open)"
        )
    else:  # overnight
        # Note: we use TODAY's predictor ranking. For a true historical
        # replay we'd snapshot the rank at 3:30pm on the event_ts date;
        # this is a small known limitation we surface in the UI.
        picks = rank_next_day_bullish(top_k=5)
        if not picks:
            LOGGER.warning(f"[{event_ts}] no overnight candidates")
            return None
        pick = picks[0]["ticker"]
        rationale = (
            f"Top bullish-opening score {picks[0]['score']:.2f} "
            f"(close-pos {picks[0]['close_pos']:.2f}, "
            f"5d {picks[0]['ret_5d_pct']:+.1f}%, "
            f"RSI {picks[0]['rsi_14']:.0f})"
        )

    # Get the actual execution price at event_ts
    px = get_price_at(pick, event_ts)
    if px is None or px <= 0:
        LOGGER.warning(f"[{event_ts}] no price for {pick}, skipping buy")
        return None

    # Size the position
    equity_at_ts = _mark_to_market(event_ts)
    notional = equity_at_ts * ALLOCATION_PCT
    cash = storage.get_cash()
    if notional > cash:
        notional = cash * 0.99  # leave a sliver for rounding
    if notional < 50:
        LOGGER.warning(f"[{event_ts}] insufficient cash ({cash:.2f})")
        return None

    shares = notional / px
    storage.set_cash(cash - notional)
    storage.open_position(
        slot=slot, ticker=pick, shares=shares, price=px,
        ts=event_ts.isoformat(), notional=notional,
    )
    LOGGER.info(
        f"[{event_ts}] BUY {slot} {pick} @ {px:.4f} x {shares:.4f} "
        f"= ${notional:.2f}   {rationale}"
    )
    return {
        "ticker": pick, "shares": shares, "price": px,
        "notional": notional, "rationale": rationale,
    }


def _execute_sell(slot: str, event_ts: datetime) -> dict[str, Any] | None:
    """Close the position in `slot` at the price at `event_ts`."""
    pos = storage.get_position(slot)
    if pos is None:
        return None
    px = get_price_at(pos["ticker"], event_ts)
    if px is None or px <= 0:
        LOGGER.warning(
            f"[{event_ts}] no price for {pos['ticker']} on close, "
            f"using entry as fallback"
        )
        px = float(pos["entry_price"])
    summary = storage.close_position(slot, sell_price=px, ts=event_ts.isoformat())
    if summary:
        storage.set_cash(storage.get_cash() + summary["proceeds"])
        LOGGER.info(
            f"[{event_ts}] SELL {slot} {summary['ticker']} @ {px:.4f} "
            f"PnL ${summary['pnl']:+.2f} ({summary['pnl_pct']:+.2f}%)"
        )
    return summary


def tick() -> dict[str, Any]:
    """Run the agent forward to NOW. Executes any scheduled events
    that fell between the last action timestamp and now, in order.

    Returns a summary dict with:
      events_executed : count of fills
      executions      : list of {ts, action, slot, ticker, ...}
    """
    storage.init_db()
    now = _now_et()
    last_action_str = storage.get_state("last_action_ts")
    if last_action_str:
        last_action = datetime.fromisoformat(last_action_str)
        if last_action.tzinfo is None:
            last_action = last_action.replace(tzinfo=ET)
    else:
        # First run — start the agent fresh from now. We deliberately
        # do NOT backfill historical events on first init because:
        #   (a) it'd be ~100 ETF data fetches per event = minutes per
        #       firing, which makes the page load unusable, and
        #   (b) a paper trader should clearly demarcate when the user
        #       started it; pretending to have traded yesterday is
        #       misleading.
        storage.set_state("last_action_ts", now.isoformat())
        return {
            "now": now.isoformat(),
            "events_seen": 0, "events_executed": 0,
            "executions": [], "first_run": True,
        }

    # Cap retroactive backfill to 36h. Within that window the data
    # fetches stay tractable (≤ 8 events × 1-min bar lookups). Beyond
    # 36h, just resume from "now" — we don't try to invent multi-day
    # history.
    if (now - last_action).total_seconds() > 36 * 3600:
        LOGGER.warning(
            f"More than 36h since last tick ({last_action}); "
            f"skipping backfill and resuming from now."
        )
        storage.set_state("last_action_ts", now.isoformat())
        return {
            "now": now.isoformat(),
            "events_seen": 0, "events_executed": 0,
            "executions": [], "backfill_skipped": True,
        }

    events = _enumerate_events(last_action, now)
    executions: list[dict[str, Any]] = []
    for evt in events:
        ts = evt["ts"]
        if evt["action"] == "BUY":
            result = _execute_buy(evt["slot"], ts)
        else:
            result = _execute_sell(evt["slot"], ts)
        if result:
            executions.append({
                "ts": ts.isoformat(),
                "action": evt["action"],
                "slot": evt["slot"],
                **result,
            })

    storage.set_state("last_action_ts", now.isoformat())
    return {
        "now": now.isoformat(),
        "events_seen": len(events),
        "events_executed": len(executions),
        "executions": executions,
    }


def next_scheduled_event(from_ts: datetime | None = None) -> dict[str, Any] | None:
    """Return the next upcoming scheduled event from `from_ts` (default
    now), so the UI can display 'next action: SELL intraday at 3:45 PM'."""
    from_ts = from_ts or _now_et()
    # Look forward up to 3 days to handle weekends
    horizon = from_ts + timedelta(days=3)
    upcoming = _enumerate_events(from_ts, horizon)
    if not upcoming:
        return None
    evt = upcoming[0]
    return {
        "ts": evt["ts"].isoformat(),
        "action": evt["action"],
        "slot": evt["slot"],
        "seconds_until": int((evt["ts"] - from_ts).total_seconds()),
    }


def get_portfolio_value() -> dict[str, Any]:
    """Live snapshot: cash + open positions marked-to-market with
    fetched current prices."""
    storage.init_db()
    cash = storage.get_cash()
    initial = storage.get_initial_capital()
    positions = storage.get_all_positions()
    now = _now_et()
    pos_values = {}
    open_mtm = 0.0
    for slot, pos in positions.items():
        px = get_price_at(pos["ticker"], now)
        if px is None:
            px = float(pos["entry_price"])
        mtm = float(pos["shares"]) * px
        pnl = mtm - float(pos["cost_basis"])
        pnl_pct = (px - float(pos["entry_price"])) / float(pos["entry_price"]) * 100
        pos_values[slot] = {
            **pos,
            "current_price": px,
            "mtm": mtm,
            "open_pnl": pnl,
            "open_pnl_pct": pnl_pct,
        }
        open_mtm += mtm
    total_equity = cash + open_mtm
    total_pnl = total_equity - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0.0
    return {
        "cash": cash,
        "open_mtm": open_mtm,
        "total_equity": total_equity,
        "initial_capital": initial,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "positions": pos_values,
    }

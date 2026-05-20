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
ALLOCATION_PCT = 0.25  # 25% baseline (fixed-sizing fallback only)

# Conviction-based sizing: scale position size 15% to 35% based on the
# strength of the signal. Linear scale validated by parameter sweep:
# 3-year backtest +121.5% (vs fixed 25%'s +110.5%) on the commodity
# universe. Identical win rate; the extra return comes from putting
# more capital on the strongest setups.
SIZING_RANGE = (0.15, 0.35)
INTRADAY_GAP_FOR_MIN = 1.5   # at this gap%, size = 0.15 (the floor)
INTRADAY_GAP_FOR_MAX = 5.0   # at this gap%, size = 0.35 (the cap)
OVERNIGHT_SCORE_FOR_MIN = 0.70  # at this score, size = 0.15
OVERNIGHT_SCORE_FOR_MAX = 0.95  # at this score, size = 0.35


def size_intraday(gap_pct: float) -> float:
    """Linear scale from 15% to 35% over [1.5%, 5.0%] gap range."""
    lo, hi = SIZING_RANGE
    x = (gap_pct - INTRADAY_GAP_FOR_MIN) / (INTRADAY_GAP_FOR_MAX - INTRADAY_GAP_FOR_MIN)
    x = max(0.0, min(1.0, x))
    return lo + x * (hi - lo)


def size_overnight(score: float) -> float:
    """Linear scale from 15% to 35% over [0.70, 0.95] score range."""
    lo, hi = SIZING_RANGE
    x = (score - OVERNIGHT_SCORE_FOR_MIN) / (OVERNIGHT_SCORE_FOR_MAX - OVERNIGHT_SCORE_FOR_MIN)
    x = max(0.0, min(1.0, x))
    return lo + x * (hi - lo)


# Underlying commodity each ETF tracks — used for "underlying sympathy"
ETF_TO_UNDERLYING = {
    "HOU.TO": ("CL=F", "WTI"),
    "HOD.TO": ("CL=F", "WTI"),
    "HNU.TO": ("NG=F", "Natgas"),
    "HND.TO": ("NG=F", "Natgas"),
    "CGL.TO": ("GC=F", "Gold"),
    "MNT.TO": ("GC=F", "Gold"),
}


def evaluate_intraday_pick(mover: dict, equity: float = 10_000.0) -> dict:
    """Decorate an intraday top-mover dict with the full trader context:
    trend alignment, whether all filters pass, allocation%, dollar size,
    stop/target prices. Used by the dashboard to show 'what would
    actually fire'."""
    out = dict(mover)
    ticker = mover["ticker"]
    gap = mover.get("change_pct", 0.0)

    # Trend alignment check (same as live agent)
    trend_ok = _trend_aligned(ticker) if FILTERS["require_trend_alignment"] else True
    # All filters pass?
    gap_ok = gap >= FILTERS["min_intraday_pct"]
    would_fire = gap_ok and trend_ok

    alloc = size_intraday(gap) if would_fire else 0.0
    notional = equity * alloc
    entry_px = float(mover.get("current", 0.0))
    stop_px = entry_px * (1 + FILTERS["stop_loss_pct"]) if entry_px else 0
    target_px = entry_px * (1 + FILTERS["take_profit_pct"]) if entry_px else 0

    out.update({
        "trend_ok": trend_ok,
        "gap_ok": gap_ok,
        "would_fire": would_fire,
        "allocation_pct": alloc,
        "notional": notional,
        "entry_px": entry_px,
        "stop_px": stop_px,
        "target_px": target_px,
        "underlying": ETF_TO_UNDERLYING.get(ticker, ("?", "?"))[1],
        "reject_reason": (
            "" if would_fire
            else (f"gap +{gap:.2f}% < {FILTERS['min_intraday_pct']:.1f}%" if not gap_ok
                  else "trend mismatch")
        ),
    })
    return out


def evaluate_overnight_pick(pick: dict, equity: float = 10_000.0) -> dict:
    """Same idea for overnight bullish picks."""
    out = dict(pick)
    ticker = pick["ticker"]
    score = pick.get("score", 0.0)

    trend_ok = _trend_aligned(ticker) if FILTERS["require_trend_alignment"] else True
    score_ok = score >= FILTERS["min_overnight_score"]
    would_fire = score_ok and trend_ok

    alloc = size_overnight(score) if would_fire else 0.0
    notional = equity * alloc
    entry_px = float(pick.get("close", 0.0))
    stop_px = entry_px * (1 + FILTERS["stop_loss_pct"]) if entry_px else 0
    target_px = entry_px * (1 + FILTERS["take_profit_pct"]) if entry_px else 0

    out.update({
        "trend_ok": trend_ok,
        "score_ok": score_ok,
        "would_fire": would_fire,
        "allocation_pct": alloc,
        "notional": notional,
        "entry_px": entry_px,
        "stop_px": stop_px,
        "target_px": target_px,
        "underlying": ETF_TO_UNDERLYING.get(ticker, ("?", "?"))[1],
        "reject_reason": (
            "" if would_fire
            else (f"score {score:.2f} < {FILTERS['min_overnight_score']:.2f}" if not score_ok
                  else "trend mismatch")
        ),
    })
    return out


def get_underlying_today() -> dict:
    """Today's intraday move on the 3 underlying commodities. Used in
    the dashboard's commodity-context strip."""
    import yfinance as yf
    out = {}
    for sym, label in [("CL=F", "WTI"), ("NG=F", "Natgas"), ("GC=F", "Gold"),
                        ("DX-Y.NYB", "DXY"), ("^VIX", "VIX")]:
        try:
            df = yf.download(sym, period="1d", interval="5m",
                             auto_adjust=False, progress=False)
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 2:
                # Fall back to daily
                df = yf.download(sym, period="5d", interval="1d",
                                 auto_adjust=False, progress=False)
                if hasattr(df.columns, "get_level_values"):
                    df.columns = df.columns.get_level_values(0)
                if df.empty:
                    continue
                op = float(df["Open"].iloc[-1])
                cp = float(df["Close"].iloc[-1])
            else:
                op = float(df["Open"].iloc[0])
                cp = float(df["Close"].iloc[-1])
            if op > 0:
                out[label] = {
                    "value": cp,
                    "open": op,
                    "change_pct": (cp - op) / op * 100,
                }
        except Exception:
            continue
    return out

# ============================================================
# Strategy filters — added to address the -5.76% backtest result.
# Each filter is a "veto" — if conditions aren't met, the buy is
# skipped and that slot sits in cash for the day. Goal: avoid the
# "HND when natgas rallying" type of bad pick that destroyed the
# overnight P&L in the 30d backtest.
# ============================================================
FILTERS = {
    # Intraday top mover must be moving at least this much from open.
    "min_intraday_pct": 1.5,

    # Overnight composite score must clear this threshold (0-1).
    "min_overnight_score": 0.70,

    # Trend alignment: for the bull/bear inverse pairs (HOU/HOD,
    # HNU/HND), the underlying's 20-day SMA slope must agree with
    # the direction of the pick.
    "require_trend_alignment": True,
    "trend_slope_threshold_pct": 1.0,

    # --- v5: Catalyst-aware risk management ---
    # Hard stop-loss as a percent of entry price (negative = loss).
    # Triggered when the daily Low (intraday) or next-day Low
    # (overnight) breaches the threshold. -3% on a 2x ETF caps
    # exposure to ~1.5% adverse underlying move.
    "stop_loss_pct": -0.03,

    # Take-profit threshold — sell early if daily High clears this.
    # Locks in outlier wins instead of giving them back by close.
    "take_profit_pct": 0.05,

    # Cross-asset confirmation (DXY for oil, 10y yield for gold).
    # DEFAULT OFF based on backtest. Tested separately on 3 years of
    # commodity data: cross-asset alignment vetoed 32 trades but the
    # vetoed trades were positive in expectation. The USD-oil inverse
    # relationship has weakened in recent years. Keeping the hook for
    # future experimentation but off by default.
    "require_cross_asset_alignment": False,
    "dxy_5d_threshold_pct": 0.5,
    "yield_5d_threshold_bps": 5.0,
}

# Map of inverse-pair ETFs to their underlying for trend lookup
PAIR_UNDERLYING = {
    "HOU.TO": ("CL=F", "bull"),   # WTI
    "HOD.TO": ("CL=F", "bear"),
    "HNU.TO": ("NG=F", "bull"),   # natgas
    "HND.TO": ("NG=F", "bear"),
}


def _trend_aligned(ticker: str) -> bool:
    """Check 20-day SMA slope of the underlying commodity. Returns
    True if the slope direction matches the ETF's bull/bear side,
    or if this ETF isn't an inverse-pair member (always pass)."""
    if ticker not in PAIR_UNDERLYING:
        return True
    underlying, side = PAIR_UNDERLYING[ticker]
    try:
        import yfinance as yf
        df = yf.download(underlying, period="2mo", interval="1d",
                         auto_adjust=False, progress=False)
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 30:
            return True  # not enough data → fail open (allow trade)
        close = df["Close"].dropna()
        sma20 = close.rolling(20).mean()
        if len(sma20.dropna()) < 11:
            return True
        slope_pct = float((sma20.iloc[-1] - sma20.iloc[-11]) / sma20.iloc[-11] * 100)
        thresh = FILTERS["trend_slope_threshold_pct"]
        if side == "bull":
            return slope_pct > thresh
        else:  # bear
            return slope_pct < -thresh
    except Exception as e:
        LOGGER.warning(f"trend check failed for {ticker}: {e}")
        return True  # fail open


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

    # Pick ticker — apply filters to skip weak setups
    if slot == "intraday":
        movers = compute_top_mover_at(event_ts, top_k=5)
        if not movers:
            LOGGER.warning(f"[{event_ts}] no intraday candidates")
            return None
        top = movers[0]
        # Filter 1: minimum momentum
        if top["change_pct"] < FILTERS["min_intraday_pct"]:
            LOGGER.info(
                f"[{event_ts}] intraday skipped — top mover only "
                f"+{top['change_pct']:.2f}% (need ≥{FILTERS['min_intraday_pct']:.1f}%)"
            )
            return None
        pick = top["ticker"]
        # Filter 2: trend alignment for inverse pairs
        if FILTERS["require_trend_alignment"] and not _trend_aligned(pick):
            LOGGER.info(
                f"[{event_ts}] intraday skipped — {pick} fails trend alignment"
            )
            return None
        rationale = (
            f"Top intraday gainer (+{top['change_pct']:.2f}% from open)"
        )
        # Conviction signal for sizing — the gap %
        conviction_alloc = size_intraday(float(top["change_pct"]))
    else:  # overnight
        picks = rank_next_day_bullish(top_k=5)
        if not picks:
            LOGGER.warning(f"[{event_ts}] no overnight candidates")
            return None
        top = picks[0]
        # Filter 1: minimum bullish score
        if top["score"] < FILTERS["min_overnight_score"]:
            LOGGER.info(
                f"[{event_ts}] overnight skipped — top score "
                f"{top['score']:.2f} (need ≥{FILTERS['min_overnight_score']:.2f})"
            )
            return None
        pick = top["ticker"]
        # Filter 2: trend alignment for inverse pairs
        if FILTERS["require_trend_alignment"] and not _trend_aligned(pick):
            LOGGER.info(
                f"[{event_ts}] overnight skipped — {pick} fails trend alignment"
            )
            return None
        rationale = (
            f"Top bullish-opening score {top['score']:.2f} "
            f"(close-pos {top['close_pos']:.2f}, "
            f"5d {top['ret_5d_pct']:+.1f}%, "
            f"RSI {top['rsi_14']:.0f})"
        )
        # Conviction signal for sizing — the composite score
        conviction_alloc = size_overnight(float(top["score"]))

    # Get the actual execution price at event_ts
    px = get_price_at(pick, event_ts)
    if px is None or px <= 0:
        LOGGER.warning(f"[{event_ts}] no price for {pick}, skipping buy")
        return None

    # Size the position — conviction-based (15% to 35%)
    equity_at_ts = _mark_to_market(event_ts)
    notional = equity_at_ts * conviction_alloc
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
        f"= ${notional:.2f} ({conviction_alloc*100:.0f}% alloc)   {rationale}"
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

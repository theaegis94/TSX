"""Main paper-trading agent loop.

Run with:
    python -m paper_trader.agent

Or via the .ps1 setup script as a Windows scheduled task. The loop:

  Every 5 minutes during market hours (9:30am–4pm ET, Mon–Fri):
    1. Write heartbeat
    2. Skip if agent_paused
    3. Mark to market any open position; check exit signals
    4. If no open position, compute features, run all strategies,
       pick the best signal, open a position if conviction is high
       enough

  Every 30 minutes off-hours:
    1. Write heartbeat
    2. Recompute strategy stats from the closed-trade log
    3. (Future: backtest, walk-forward retrain — week 2+)

Operational guardrails:
  - Lock file prevents two instances from running at once
  - Heartbeat written every cycle so dashboard can show "alive at X"
  - All operational errors logged but don't crash the loop
  - On startup, registers all strategies in the strategy DB
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
import sys
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from . import storage
from . import features as feat_mod
from . import strategies as strat_mod
from . import exits

# ---------------------------------------------------------------------------
# Operational constants
# ---------------------------------------------------------------------------
MARKET_CYCLE_SECONDS = 5 * 60      # 5 min during market hours
OFFHOURS_CYCLE_SECONDS = 30 * 60    # 30 min off-hours
COMMISSION = 5.0                    # $5 per round trip — half each leg
SLIPPAGE_PCT = 0.005                # 0.5% bid-ask slippage round-trip
POSITION_SIZE_PCT = 1.00            # iter 35 — 100% base, leverage path
MIN_CONVICTION_TO_OPEN = 0.50       # don't act on weak signals
LEVERAGE_CAP = 3.0                  # iter 35 — max 3x of cash per position
ET = ZoneInfo("America/New_York")

LOGGER = logging.getLogger("paper_trader")
_HERE = pathlib.Path(__file__).resolve().parent.parent
LOCK_FILE = _HERE / "paper_trader.lock"
LOG_DIR = _HERE / "logs"


def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    if LOGGER.handlers:
        return
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "agent.log",
        when="midnight",
        backupCount=90,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    LOGGER.addHandler(ch)


# ---------------------------------------------------------------------------
# Lock file (prevents duplicate agent instances)
# ---------------------------------------------------------------------------

def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE) as f:
                pid_str = f.read().strip()
            pid = int(pid_str) if pid_str else 0
            if pid and _pid_running(pid):
                LOGGER.warning(
                    f"Lock file already held by PID {pid}; exiting."
                )
                return False
        except (OSError, ValueError):
            pass
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_running(pid: int) -> bool:
    """Cross-platform check for whether `pid` is still running."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY, 0, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def is_market_hours() -> bool:
    """TSX market hours: 9:30am - 4:00pm ET, Mon-Fri.
    (We use US Eastern; TSX and US markets share this window.)
    """
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def next_market_open() -> datetime:
    """Return the next datetime the market opens (US Eastern)."""
    now = datetime.now(ET)
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


# ---------------------------------------------------------------------------
# Trade execution helpers
# ---------------------------------------------------------------------------

def _apply_slippage(price: float, side: str) -> float:
    """Buys execute at price * (1 + slippage/2); sells at price *
    (1 - slippage/2). 0.5% round-trip = 0.25% per leg."""
    half = SLIPPAGE_PCT / 2.0
    if side == "buy":
        return price * (1 + half)
    return price * (1 - half)


def _size_multiplier(conviction: float) -> float:
    """Iter 25/35: conviction-boost multiplier. 0.50-0.65 = 1x base;
    0.65 = 2.5x; 0.80 = 5x. Capped at 5x for safety."""
    if conviction >= 0.65:
        m = 2.5 + (conviction - 0.65) / 0.15 * 2.5
        return max(1.0, min(5.0, m))
    return 1.0


def open_paper_position(
    ticker: str, strategy_name: str,
    raw_price: float, conviction: float,
) -> int | None:
    """Try to open a position. Returns the new position id, or None if
    we couldn't (insufficient cash, slipped past usable balance, etc.)."""
    balance = storage.get_balance()
    # Iter 35: conviction-weighted sizing with leverage cap. The backtest
    # showed +2594% over 5 years with this config — see docs for the
    # paper_trader_setup. Real-world: expect 30-50% lower after margin
    # interest and slippage, with brutal drawdowns. DO NOT run with real
    # money before months of live paper-trade validation.
    raw_target = balance * POSITION_SIZE_PCT * _size_multiplier(conviction)
    target_capital = min(raw_target, balance * LEVERAGE_CAP)
    if target_capital < COMMISSION * 2:
        LOGGER.info(
            f"Skip open {ticker}: balance ${balance:.2f} too low for "
            f"position size {POSITION_SIZE_PCT*100:.0f}%"
        )
        return None
    exec_price = _apply_slippage(raw_price, "buy")
    pos_id = storage.open_position(
        ticker=ticker,
        strategy=strategy_name,
        price=exec_price,
        capital=target_capital,
        commission=COMMISSION,
        conviction=conviction,
    )
    storage.debit_capital(target_capital)
    LOGGER.info(
        f"OPEN  pos#{pos_id} {ticker} via {strategy_name} "
        f"@ ${exec_price:.4f} conv={conviction:.2f} "
        f"capital=${target_capital:.2f}"
    )
    return pos_id


def close_paper_position(
    position: dict, raw_price: float, reason: str,
) -> None:
    exec_price = _apply_slippage(raw_price, "sell")
    result = storage.close_position(
        position_id=position["id"],
        exit_price=exec_price,
        commission=COMMISSION,
        exit_reason=reason,
    )
    LOGGER.info(
        f"CLOSE pos#{position['id']} {position['ticker']} "
        f"@ ${exec_price:.4f} reason={reason} "
        f"pnl=${result['pnl_dollars']:+.2f} ({result['pnl_pct']:+.2f}%)"
    )


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------

def run_market_cycle() -> None:
    """One market-hours iteration: MTM, exits, then maybe open."""
    storage.write_heartbeat("market_cycle")
    if storage.is_agent_paused():
        LOGGER.info("Agent is paused; skipping cycle.")
        return

    etf_prices = feat_mod.fetch_latest_etf_prices()
    if not etf_prices:
        LOGGER.warning("No ETF prices fetched this cycle.")
        return

    # 1. Compute features + signals once per cycle (used for both exit
    #    signal-flip checks and new entries)
    features = feat_mod.fetch_features()
    enabled = {
        s["name"] for s in storage.get_strategies() if s.get("enabled")
    }
    signals = strat_mod.run_all_strategies(features, enabled)
    for s in signals:
        storage.log_signal(
            strategy=s["strategy"],
            ticker=s["ticker"],
            conviction=s["conviction"],
            features=features,
        )
    best = strat_mod.best_signal(signals)

    # 2. Exit check on any open position
    pos = storage.get_open_position()
    if pos:
        ticker = pos["ticker"]
        cur_price = etf_prices.get(ticker)
        if cur_price is None:
            LOGGER.warning(f"No price for open position {ticker} this cycle.")
        else:
            mtm = float(pos["shares"]) * cur_price
            storage.record_equity_snapshot(
                cash_balance=storage.get_balance(),
                open_mtm=mtm,
            )
            reason = exits.check_exit(pos, cur_price, best)
            if reason:
                close_paper_position(pos, cur_price, reason)
                pos = None  # cleared

    # 3. Maybe open a new position (only if flat)
    if pos is None and best:
        if best["conviction"] >= MIN_CONVICTION_TO_OPEN:
            tk = best["ticker"]
            raw = etf_prices.get(tk)
            if raw is not None:
                open_paper_position(
                    ticker=tk,
                    strategy_name=best["strategy"],
                    raw_price=raw,
                    conviction=best["conviction"],
                )
            else:
                LOGGER.warning(
                    f"Best signal {tk} fired but no price available."
                )

    # 4. Final equity snapshot for the day
    pos2 = storage.get_open_position()
    open_mtm = 0.0
    if pos2:
        cur = etf_prices.get(pos2["ticker"])
        if cur is not None:
            open_mtm = float(pos2["shares"]) * cur
    storage.record_equity_snapshot(
        cash_balance=storage.get_balance(),
        open_mtm=open_mtm,
    )


def run_offhours_cycle() -> None:
    """Off-hours: refresh stats, future home for backtests / retraining."""
    storage.write_heartbeat("offhours_cycle")
    try:
        storage.recompute_strategy_stats()
    except Exception:
        LOGGER.exception("recompute_strategy_stats failed")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    _setup_logging()
    storage.init_databases()
    # Register strategies once on startup
    for strat in strat_mod.ALL_STRATEGIES:
        storage.register_strategy(strat.name, strat.description)
    if not _acquire_lock():
        return 1
    try:
        LOGGER.info("Paper-trader agent started.")
        while True:
            try:
                if is_market_hours():
                    run_market_cycle()
                    time.sleep(MARKET_CYCLE_SECONDS)
                else:
                    run_offhours_cycle()
                    time.sleep(OFFHOURS_CYCLE_SECONDS)
            except KeyboardInterrupt:
                LOGGER.info("KeyboardInterrupt — exiting cleanly.")
                return 0
            except Exception:
                LOGGER.exception("Unhandled error in main loop")
                time.sleep(60)
    finally:
        _release_lock()
        LOGGER.info("Paper-trader agent stopped.")


if __name__ == "__main__":
    sys.exit(main())

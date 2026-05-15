"""SQLite storage for the paper trader.

Two databases, on purpose, so a 'reset capital' button can wipe the
financial ledger without erasing what the agent has learned:

  paper_trader.db        — resettable
    config               — current balance, initial capital, pause flag,
                            last reset timestamp
    positions            — currently-open paper position (max 1 row with
                            status='open' at any time; old rows kept for
                            history with status='closed')
    trades               — full trade log (one row per closed position)
    equity_curve         — daily balance + open MTM snapshot
    heartbeat            — single-row table: last time the agent was alive
    signals              — every signal emitted by every strategy each
                            day (resettable so the log mirrors the trades)

  strategies_state.db    — persistent across resets
    strategies           — strategy registry: name, enabled flag,
                            capital weight, description
    strategy_stats       — daily snapshot of per-strategy lifetime stats
                            (win rate, profit factor, expectancy)

All datetime values are stored as ISO strings in UTC. Convert to display
timezone (US/Eastern) only in the UI layer.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
from datetime import datetime, timezone
from typing import Any

# All database files live next to the script for easy backup / move
_HERE = pathlib.Path(__file__).resolve().parent.parent
PAPER_DB = _HERE / "paper_trader.db"
STRATEGIES_DB = _HERE / "strategies_state.db"

DEFAULT_INITIAL_CAPITAL = 10_000.0

# The 4 ETFs the agent is allowed to trade. Pair structure:
#   oil:  HOU.TO (2x bull)  /  HOD.TO (2x bear)
#   gas:  HNU.TO (2x bull)  /  HND.TO (2x bear)
TICKERS = ["HOU.TO", "HOD.TO", "HNU.TO", "HND.TO"]
PAIR_OIL = ("HOU.TO", "HOD.TO")
PAIR_GAS = ("HNU.TO", "HND.TO")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker               TEXT NOT NULL,
    strategy             TEXT NOT NULL,
    entry_date           TEXT NOT NULL,
    entry_price          REAL NOT NULL,
    shares               REAL NOT NULL,
    capital_used         REAL NOT NULL,   -- shares * price + commission
    commission_paid      REAL NOT NULL,
    conviction           REAL NOT NULL,
    status               TEXT NOT NULL DEFAULT 'open',
    exit_date            TEXT,
    exit_price           REAL,
    exit_reason          TEXT,
    pnl_dollars          REAL,
    pnl_pct              REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_entry_date ON positions(entry_date);

-- NB: the `is_sim` column is added by _migrate_paper_db() rather than
-- declared here so the same code-path works for both fresh DBs and DBs
-- created by earlier versions of this app (CREATE TABLE IF NOT EXISTS
-- won't add columns to an existing table).
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id     INTEGER REFERENCES positions(id),
    ticker          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    entry_date      TEXT NOT NULL,
    exit_date       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    shares          REAL NOT NULL,
    pnl_dollars     REAL NOT NULL,
    pnl_pct         REAL NOT NULL,
    exit_reason     TEXT NOT NULL,
    direction_ok    INTEGER NOT NULL    -- 1 if pnl>0, 0 otherwise
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_exit_date ON trades(exit_date);

CREATE TABLE IF NOT EXISTS equity_curve (
    as_of_date           TEXT PRIMARY KEY,
    cash_balance         REAL NOT NULL,
    open_position_mtm    REAL NOT NULL DEFAULT 0,
    total_equity         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeat (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    last_alive    TEXT NOT NULL,
    last_action   TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date    TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    ticker        TEXT,
    conviction    REAL,
    features_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(as_of_date);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
"""

STRATEGIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    name            TEXT PRIMARY KEY,
    description     TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1,
    capital_weight  REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_stats (
    strategy             TEXT NOT NULL,
    as_of_date           TEXT NOT NULL,
    lifetime_trades      INTEGER NOT NULL DEFAULT 0,
    lifetime_wins        INTEGER NOT NULL DEFAULT 0,
    lifetime_pnl         REAL NOT NULL DEFAULT 0,
    win_rate             REAL NOT NULL DEFAULT 0,
    profit_factor        REAL NOT NULL DEFAULT 0,
    expectancy           REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (strategy, as_of_date)
);
"""


def _connect(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_paper_db(c: sqlite3.Connection) -> None:
    """Idempotent migrations. Runs AFTER the base schema's
    CREATE TABLE IF NOT EXISTS, so the trades table is guaranteed to
    exist by the time we reach here — we only need to add columns that
    weren't in older schema revisions.
    """
    cols = {row["name"] for row in c.execute("PRAGMA table_info(trades)")}
    if not cols:
        # Defensive: table somehow missing. Shouldn't happen because
        # executescript runs first, but if it does we just skip — the
        # next init_databases call will fix it.
        return
    if "is_sim" not in cols:
        c.execute(
            "ALTER TABLE trades "
            "ADD COLUMN is_sim INTEGER NOT NULL DEFAULT 0"
        )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_is_sim ON trades(is_sim)"
    )


def init_databases() -> None:
    """Create tables if missing; safe to call repeatedly."""
    with _connect(PAPER_DB) as c:
        c.executescript(PAPER_SCHEMA)
        _migrate_paper_db(c)
        # Seed config defaults
        defaults = [
            ("initial_capital", str(DEFAULT_INITIAL_CAPITAL)),
            ("current_balance", str(DEFAULT_INITIAL_CAPITAL)),
            ("agent_paused", "0"),
            ("last_reset_at", _utcnow_iso()),
        ]
        for k, v in defaults:
            c.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                (k, v),
            )
        c.commit()
    with _connect(STRATEGIES_DB) as c:
        c.executescript(STRATEGIES_SCHEMA)
        c.commit()


# ---------------------------------------------------------------------------
# Config / pause / reset
# ---------------------------------------------------------------------------

def _get_config(key: str, default: str | None = None) -> str | None:
    with _connect(PAPER_DB) as c:
        row = c.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def _set_config(key: str, value: str) -> None:
    with _connect(PAPER_DB) as c:
        c.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        c.commit()


def get_balance() -> float:
    """Return current cash balance (does NOT include open MTM)."""
    return float(_get_config("current_balance",
                              str(DEFAULT_INITIAL_CAPITAL)))


def set_balance(new_balance: float) -> None:
    _set_config("current_balance", f"{new_balance:.4f}")


def is_agent_paused() -> bool:
    return _get_config("agent_paused", "0") == "1"


def set_agent_paused(paused: bool) -> None:
    _set_config("agent_paused", "1" if paused else "0")


def reset_capital(amount: float) -> None:
    """Wipe positions, trades, equity curve, signals — strategy state
    survives. Resets balance to `amount`.
    """
    with _connect(PAPER_DB) as c:
        c.execute("DELETE FROM positions")
        c.execute("DELETE FROM trades")
        c.execute("DELETE FROM equity_curve")
        c.execute("DELETE FROM signals")
        c.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("initial_capital", f"{amount:.4f}"),
        )
        c.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("current_balance", f"{amount:.4f}"),
        )
        c.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("last_reset_at", _utcnow_iso()),
        )
        c.commit()


def get_initial_capital() -> float:
    return float(_get_config("initial_capital",
                              str(DEFAULT_INITIAL_CAPITAL)))


def get_last_reset_at() -> str:
    return _get_config("last_reset_at", _utcnow_iso())


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def get_open_position() -> dict | None:
    """Return the (at most one) currently-open position, or None."""
    with _connect(PAPER_DB) as c:
        row = c.execute(
            "SELECT * FROM positions WHERE status = 'open' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def open_position(
    ticker: str,
    strategy: str,
    price: float,
    capital: float,
    commission: float,
    conviction: float,
    entry_date: str | None = None,
) -> int:
    """Open a new paper position. Returns the new position id.

    `capital` is the gross dollars to allocate (e.g. 25% of balance).
    Shares are computed as (capital - commission) / price.
    """
    if entry_date is None:
        entry_date = _utcnow_iso()
    shares = max(0.0, (capital - commission) / price)
    with _connect(PAPER_DB) as c:
        cur = c.execute(
            "INSERT INTO positions "
            "(ticker, strategy, entry_date, entry_price, shares, "
            " capital_used, commission_paid, conviction, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')",
            (ticker, strategy, entry_date, price, shares,
             capital, commission, conviction),
        )
        c.commit()
        return int(cur.lastrowid)


def close_position(
    position_id: int,
    exit_price: float,
    commission: float,
    exit_reason: str,
    exit_date: str | None = None,
) -> dict:
    """Close an open position, log a trade row, return the closed position
    as a dict with computed P&L."""
    if exit_date is None:
        exit_date = _utcnow_iso()
    with _connect(PAPER_DB) as c:
        row = c.execute(
            "SELECT * FROM positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Position {position_id} not found")
        if row["status"] != "open":
            raise ValueError(f"Position {position_id} not open")
        shares = float(row["shares"])
        entry_price = float(row["entry_price"])
        gross_proceeds = shares * exit_price
        net_proceeds = gross_proceeds - commission
        pnl_dollars = net_proceeds - float(row["capital_used"])
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        direction_ok = 1 if pnl_dollars > 0 else 0

        c.execute(
            "UPDATE positions SET "
            "status='closed', exit_date=?, exit_price=?, "
            "exit_reason=?, pnl_dollars=?, pnl_pct=? "
            "WHERE id=?",
            (exit_date, exit_price, exit_reason,
             pnl_dollars, pnl_pct, position_id),
        )
        c.execute(
            "INSERT INTO trades "
            "(position_id, ticker, strategy, entry_date, exit_date, "
            " entry_price, exit_price, shares, pnl_dollars, pnl_pct, "
            " exit_reason, direction_ok) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (position_id, row["ticker"], row["strategy"],
             row["entry_date"], exit_date,
             entry_price, exit_price, shares,
             pnl_dollars, pnl_pct, exit_reason, direction_ok),
        )
        # Credit net proceeds back to cash
        cur_bal = float(_get_config("current_balance",
                                      str(DEFAULT_INITIAL_CAPITAL)))
        new_bal = cur_bal + net_proceeds
        c.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("current_balance", f"{new_bal:.4f}"),
        )
        c.commit()
        return {
            "position_id": position_id,
            "ticker": row["ticker"],
            "strategy": row["strategy"],
            "entry_date": row["entry_date"],
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
        }


def debit_capital(amount: float) -> None:
    """Subtract `amount` from current cash balance (used when opening
    a position — the open_position helper does NOT auto-debit so the
    caller controls when cash leaves the ledger)."""
    cur = get_balance()
    set_balance(cur - amount)


def get_trade_history(
    limit: int = 200,
    include_sim: bool = True,
    only_sim: bool = False,
) -> list[dict]:
    """Return trades ordered newest-first.

    By default returns BOTH live and sim trades. Set include_sim=False
    to filter out sim trades, or only_sim=True for the inverse.
    """
    with _connect(PAPER_DB) as c:
        if only_sim:
            sql = "SELECT * FROM trades WHERE is_sim=1 ORDER BY exit_date DESC LIMIT ?"
        elif include_sim:
            sql = "SELECT * FROM trades ORDER BY exit_date DESC LIMIT ?"
        else:
            sql = "SELECT * FROM trades WHERE is_sim=0 ORDER BY exit_date DESC LIMIT ?"
        rows = c.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def record_sim_trade(
    ticker: str,
    strategy: str,
    entry_date: str,
    exit_date: str,
    entry_price: float,
    exit_price: float,
    shares: float,
    pnl_dollars: float,
    pnl_pct: float,
    exit_reason: str,
) -> int:
    """Insert a simulated trade row directly. Does NOT touch the live
    cash balance or positions table — sim trades are purely a learning
    record. Returns the new trade id.
    """
    direction_ok = 1 if pnl_dollars > 0 else 0
    with _connect(PAPER_DB) as c:
        cur = c.execute(
            "INSERT INTO trades "
            "(position_id, ticker, strategy, entry_date, exit_date, "
            " entry_price, exit_price, shares, pnl_dollars, pnl_pct, "
            " exit_reason, direction_ok, is_sim) "
            "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (ticker, strategy, entry_date, exit_date,
             entry_price, exit_price, shares,
             pnl_dollars, pnl_pct, exit_reason, direction_ok),
        )
        c.commit()
        return int(cur.lastrowid)


def clear_sim_trades() -> int:
    """Delete every sim trade (is_sim=1). Returns rows deleted.
    Used to re-run a backtest from scratch without polluting prior data.
    """
    with _connect(PAPER_DB) as c:
        cur = c.execute("DELETE FROM trades WHERE is_sim=1")
        c.commit()
        return int(cur.rowcount or 0)


def count_sim_trades() -> int:
    with _connect(PAPER_DB) as c:
        row = c.execute(
            "SELECT COUNT(*) as n FROM trades WHERE is_sim=1"
        ).fetchone()
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def record_equity_snapshot(
    cash_balance: float,
    open_mtm: float,
    as_of_date: str | None = None,
) -> None:
    """Upsert today's row in the equity curve."""
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = cash_balance + open_mtm
    with _connect(PAPER_DB) as c:
        c.execute(
            "INSERT INTO equity_curve "
            "(as_of_date, cash_balance, open_position_mtm, total_equity) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(as_of_date) DO UPDATE SET "
            "  cash_balance=excluded.cash_balance, "
            "  open_position_mtm=excluded.open_position_mtm, "
            "  total_equity=excluded.total_equity",
            (as_of_date, cash_balance, open_mtm, total),
        )
        c.commit()


def get_equity_curve() -> list[dict]:
    with _connect(PAPER_DB) as c:
        rows = c.execute(
            "SELECT * FROM equity_curve ORDER BY as_of_date ASC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def write_heartbeat(action: str | None = None) -> None:
    with _connect(PAPER_DB) as c:
        c.execute(
            "INSERT INTO heartbeat (id, last_alive, last_action) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  last_alive=excluded.last_alive, "
            "  last_action=excluded.last_action",
            (_utcnow_iso(), action),
        )
        c.commit()


def get_heartbeat() -> dict | None:
    with _connect(PAPER_DB) as c:
        row = c.execute(
            "SELECT * FROM heartbeat WHERE id = 1"
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Signal log
# ---------------------------------------------------------------------------

def log_signal(
    strategy: str,
    ticker: str | None,
    conviction: float | None,
    features: dict | None = None,
    as_of_date: str | None = None,
) -> None:
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _connect(PAPER_DB) as c:
        c.execute(
            "INSERT INTO signals "
            "(as_of_date, strategy, ticker, conviction, features_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (as_of_date, strategy, ticker, conviction,
             json.dumps(features) if features else None),
        )
        c.commit()


def get_signal_log(limit: int = 200) -> list[dict]:
    with _connect(PAPER_DB) as c:
        rows = c.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Strategy registry + stats
# ---------------------------------------------------------------------------

def register_strategy(name: str, description: str = "") -> None:
    """Idempotent — inserts a strategy row if it doesn't exist."""
    with _connect(STRATEGIES_DB) as c:
        c.execute(
            "INSERT OR IGNORE INTO strategies "
            "(name, description, enabled, capital_weight, created_at) "
            "VALUES (?, ?, 1, 1.0, ?)",
            (name, description, _utcnow_iso()),
        )
        c.commit()


def set_strategy_enabled(name: str, enabled: bool) -> None:
    with _connect(STRATEGIES_DB) as c:
        c.execute(
            "UPDATE strategies SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        c.commit()


def get_strategies() -> list[dict]:
    with _connect(STRATEGIES_DB) as c:
        rows = c.execute(
            "SELECT * FROM strategies ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def recompute_strategy_stats(
    include_sim: bool = False,
) -> dict[str, dict[str, Any]]:
    """Walk every closed trade and compute per-strategy stats.
    Writes a snapshot row to strategy_stats (one per day per strategy).
    Returns {strategy_name: stats_dict}.

    By default this aggregates LIVE trades only — sim trades have
    their own stats endpoint (compute_sim_stats) so the dashboard can
    show both side by side without confusing the two.
    """
    with _connect(PAPER_DB) as c:
        if include_sim:
            sql = (
                "SELECT strategy, pnl_dollars, direction_ok FROM trades"
            )
        else:
            sql = (
                "SELECT strategy, pnl_dollars, direction_ok "
                "FROM trades WHERE is_sim=0"
            )
        rows = c.execute(sql).fetchall()

    stats: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["strategy"]
        st = stats.setdefault(s, {
            "trades": 0,
            "wins": 0,
            "gross_pnl": 0.0,
            "gross_wins": 0.0,
            "gross_losses": 0.0,
        })
        st["trades"] += 1
        pnl = float(r["pnl_dollars"])
        st["gross_pnl"] += pnl
        if r["direction_ok"]:
            st["wins"] += 1
            st["gross_wins"] += pnl
        else:
            st["gross_losses"] += abs(pnl)

    # Compute derived metrics
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: dict[str, dict[str, Any]] = {}
    for name, st in stats.items():
        trades = st["trades"]
        wins = st["wins"]
        wr = wins / trades if trades else 0.0
        pf = (st["gross_wins"] / st["gross_losses"]
              if st["gross_losses"] > 0 else 0.0)
        exp = st["gross_pnl"] / trades if trades else 0.0
        rec = {
            "strategy": name,
            "as_of_date": today,
            "lifetime_trades": trades,
            "lifetime_wins": wins,
            "lifetime_pnl": st["gross_pnl"],
            "win_rate": wr,
            "profit_factor": pf,
            "expectancy": exp,
        }
        out[name] = rec
        with _connect(STRATEGIES_DB) as c:
            c.execute(
                "INSERT INTO strategy_stats "
                "(strategy, as_of_date, lifetime_trades, lifetime_wins, "
                " lifetime_pnl, win_rate, profit_factor, expectancy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(strategy, as_of_date) DO UPDATE SET "
                "  lifetime_trades=excluded.lifetime_trades, "
                "  lifetime_wins=excluded.lifetime_wins, "
                "  lifetime_pnl=excluded.lifetime_pnl, "
                "  win_rate=excluded.win_rate, "
                "  profit_factor=excluded.profit_factor, "
                "  expectancy=excluded.expectancy",
                (rec["strategy"], rec["as_of_date"],
                 rec["lifetime_trades"], rec["lifetime_wins"],
                 rec["lifetime_pnl"], rec["win_rate"],
                 rec["profit_factor"], rec["expectancy"]),
            )
            c.commit()
    return out


def compute_sim_stats() -> dict[str, dict[str, Any]]:
    """Per-strategy aggregates from SIM trades only. Doesn't write to
    strategy_stats — returns a fresh in-memory dict for display.
    """
    with _connect(PAPER_DB) as c:
        rows = c.execute(
            "SELECT strategy, pnl_dollars, direction_ok "
            "FROM trades WHERE is_sim=1"
        ).fetchall()
    stats: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["strategy"]
        st = stats.setdefault(s, {
            "trades": 0,
            "wins": 0,
            "gross_pnl": 0.0,
            "gross_wins": 0.0,
            "gross_losses": 0.0,
        })
        st["trades"] += 1
        pnl = float(r["pnl_dollars"])
        st["gross_pnl"] += pnl
        if r["direction_ok"]:
            st["wins"] += 1
            st["gross_wins"] += pnl
        else:
            st["gross_losses"] += abs(pnl)
    out: dict[str, dict[str, Any]] = {}
    for name, st in stats.items():
        trades = st["trades"]
        wins = st["wins"]
        wr = wins / trades if trades else 0.0
        pf = (st["gross_wins"] / st["gross_losses"]
              if st["gross_losses"] > 0 else 0.0)
        exp = st["gross_pnl"] / trades if trades else 0.0
        out[name] = {
            "strategy": name,
            "sim_trades": trades,
            "sim_wins": wins,
            "sim_pnl": st["gross_pnl"],
            "sim_win_rate": wr,
            "sim_profit_factor": pf,
            "sim_expectancy": exp,
        }
    return out


def get_strategy_stats() -> list[dict]:
    """Get the latest snapshot for each strategy."""
    with _connect(STRATEGIES_DB) as c:
        rows = c.execute(
            "SELECT s.name, s.description, s.enabled, s.capital_weight, "
            "       st.lifetime_trades, st.lifetime_wins, st.lifetime_pnl, "
            "       st.win_rate, st.profit_factor, st.expectancy, "
            "       st.as_of_date "
            "FROM strategies s "
            "LEFT JOIN ( "
            "  SELECT strategy, MAX(as_of_date) AS max_date "
            "  FROM strategy_stats GROUP BY strategy"
            ") latest ON latest.strategy = s.name "
            "LEFT JOIN strategy_stats st "
            "  ON st.strategy = s.name AND st.as_of_date = latest.max_date "
            "ORDER BY s.name"
        ).fetchall()
    return [dict(r) for r in rows]

"""SQLite state layer for the Canadian-ETF paper trader.

Three small tables:
  agent_state  : key/value (cash balance, last_action_ts, initial_capital)
  positions    : current open position per slot (intraday / overnight)
  trades       : full immutable log of every fill (buy + sell)

Everything is keyed by 'slot' so the two-slot rotation is enforced
at the schema level — at most one row per slot in `positions`.
"""
from __future__ import annotations

import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Any

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "ca_etf_trader.db"
DEFAULT_INITIAL_CAPITAL = 10_000.0


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create tables on first run. Idempotent."""
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS agent_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            slot         TEXT PRIMARY KEY,        -- 'intraday' or 'overnight'
            ticker       TEXT NOT NULL,
            shares       REAL NOT NULL,
            entry_price  REAL NOT NULL,
            entry_ts     TEXT NOT NULL,
            cost_basis   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            slot      TEXT NOT NULL,
            ticker    TEXT NOT NULL,
            side      TEXT NOT NULL,             -- 'BUY' or 'SELL'
            shares    REAL NOT NULL,
            price     REAL NOT NULL,
            notional  REAL NOT NULL,
            ts        TEXT NOT NULL,
            pnl       REAL,                       -- only set on SELL
            note      TEXT
        );
        """)
        # Seed initial capital + cash if first run
        cur = c.execute("SELECT 1 FROM agent_state WHERE key='cash'")
        if cur.fetchone() is None:
            c.execute(
                "INSERT INTO agent_state (key, value) VALUES (?, ?)",
                ("cash", str(DEFAULT_INITIAL_CAPITAL)),
            )
            c.execute(
                "INSERT INTO agent_state (key, value) VALUES (?, ?)",
                ("initial_capital", str(DEFAULT_INITIAL_CAPITAL)),
            )


def get_cash() -> float:
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM agent_state WHERE key='cash'"
        ).fetchone()
        return float(row["value"]) if row else 0.0


def set_cash(amount: float) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO agent_state (key, value) VALUES ('cash', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(amount),),
        )


def get_initial_capital() -> float:
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM agent_state WHERE key='initial_capital'"
        ).fetchone()
        return float(row["value"]) if row else DEFAULT_INITIAL_CAPITAL


def get_state(key: str) -> str | None:
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM agent_state WHERE key=?", (key,),
        ).fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO agent_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_position(slot: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM positions WHERE slot=?", (slot,),
        ).fetchone()
        return dict(row) if row else None


def get_all_positions() -> dict[str, dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM positions").fetchall()
        return {r["slot"]: dict(r) for r in rows}


def open_position(slot: str, ticker: str, shares: float, price: float,
                  ts: str, notional: float) -> None:
    """Open a new position in a slot. Slot must be empty."""
    with _conn() as c:
        c.execute(
            "INSERT INTO positions "
            "(slot, ticker, shares, entry_price, entry_ts, cost_basis) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (slot, ticker, shares, price, ts, notional),
        )
        c.execute(
            "INSERT INTO trades "
            "(slot, ticker, side, shares, price, notional, ts) "
            "VALUES (?, ?, 'BUY', ?, ?, ?, ?)",
            (slot, ticker, shares, price, notional, ts),
        )


def close_position(slot: str, sell_price: float, ts: str) -> dict[str, Any] | None:
    """Close the position in a slot at sell_price, log the trade, return
    a summary dict {ticker, pnl, pnl_pct, proceeds}. Returns None if no
    position was open."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM positions WHERE slot=?", (slot,),
        ).fetchone()
        if row is None:
            return None
        shares = float(row["shares"])
        entry_price = float(row["entry_price"])
        cost_basis = float(row["cost_basis"])
        proceeds = shares * sell_price
        pnl = proceeds - cost_basis
        pnl_pct = (sell_price - entry_price) / entry_price * 100
        c.execute(
            "INSERT INTO trades "
            "(slot, ticker, side, shares, price, notional, ts, pnl) "
            "VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)",
            (slot, row["ticker"], shares, sell_price, proceeds, ts, pnl),
        )
        c.execute("DELETE FROM positions WHERE slot=?", (slot,))
        return {
            "ticker": row["ticker"],
            "shares": shares,
            "entry_price": entry_price,
            "sell_price": sell_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "proceeds": proceeds,
        }


def get_trade_history(limit: int = 200) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def reset_all(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> None:
    """Wipe ALL paper-trader state. Used when user clicks 'Reset'."""
    with _conn() as c:
        c.executescript(
            "DELETE FROM positions; "
            "DELETE FROM trades; "
            "DELETE FROM agent_state;"
        )
        c.execute(
            "INSERT INTO agent_state (key, value) VALUES ('cash', ?), "
            "('initial_capital', ?)",
            (str(initial_capital), str(initial_capital)),
        )

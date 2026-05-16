"""CFTC Commitment of Traders (COT) data — free weekly speculator positioning.

Published every Friday at 3:30 PM ET, reflecting Tuesday-of-the-week
positions held by different trader categories on NYMEX Light Sweet
Crude Oil futures. Free public data via CFTC Socrata API.

The edge:
  When non-commercial (speculative) net positioning hits extremes,
  positioning unwinds tend to follow. Academic studies (Hamilton/Wu,
  Sanders/Irwin) show modest but persistent edge from positioning
  extremes — especially the SHORT extreme, which precedes squeezes.

In our system: use as a CONVICTION BOOSTER on bull-side signals.
When specs are extremely net-short (bottom 10th percentile of
trailing 52 weeks), boost any HOU.TO signal's conviction by +0.05.
This pushes more trades into the size-multiplier band and amplifies
the strongest setups when positioning + price agree.

Doesn't try to be a standalone strategy — that overfits with so few
data points (one row per week, ~260 over 5 years).
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger("paper_trader.cftc")

# Cache to disk so we don't hammer the API
_HERE = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _HERE / ".eia_cache"   # reuse the same cache dir
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "cftc_crude_oil.parquet"

# NYMEX Light Sweet Crude Oil — the standard WTI futures contract
CFTC_CRUDE_OIL_CODE = "067651"

# Legacy Futures-Only report (simpler than disaggregated)
CFTC_API = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"


def _read_cache(max_age_hours: int = 24) -> pd.DataFrame | None:
    if not _CACHE_FILE.exists():
        return None
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        _CACHE_FILE.stat().st_mtime, tz=timezone.utc,
    )
    if age > timedelta(hours=max_age_hours):
        return None
    try:
        return pd.read_parquet(_CACHE_FILE)
    except Exception:
        return None


def _write_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_CACHE_FILE, index=False)
    except Exception:
        pass


def fetch_cot_data(limit: int = 2000) -> pd.DataFrame:
    """Fetch crude oil COT history. Returns DataFrame sorted ascending
    by report date with columns:
      period, noncomm_long, noncomm_short, comm_long, comm_short
    """
    cached = _read_cache()
    if cached is not None and not cached.empty:
        return cached

    params = {
        "$where": f"cftc_contract_market_code='{CFTC_CRUDE_OIL_CODE}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(limit),
    }
    try:
        resp = requests.get(CFTC_API, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        LOGGER.warning(f"CFTC fetch failed: {e}")
        return pd.DataFrame()
    if not items:
        LOGGER.warning("CFTC returned 0 rows")
        return pd.DataFrame()

    rows = []
    for d in items:
        try:
            date_str = d.get("report_date_as_yyyy_mm_dd")
            if not date_str:
                continue
            rows.append({
                "period": pd.Timestamp(date_str),
                "noncomm_long": float(d.get("noncomm_positions_long_all", 0) or 0),
                "noncomm_short": float(d.get("noncomm_positions_short_all", 0) or 0),
                "comm_long": float(d.get("comm_positions_long_all", 0) or 0),
                "comm_short": float(d.get("comm_positions_short_all", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    _write_cache(df)
    LOGGER.info(f"CFTC: fetched {len(df)} weekly rows")
    return df


def compute_cftc_features(
    df: pd.DataFrame,
    as_of_date=None,
) -> dict[str, Any]:
    """Compute speculator positioning features. Walk-forward-safe:
    only uses reports whose publication date (Friday after the
    Tuesday report date) is <= as_of_date.

    Returns:
      cftc_spec_net          — non-commercial long - short
      cftc_spec_net_pctile   — percentile over trailing 52 weeks
      cftc_spec_extreme_long — pctile >= 90 (bearish for crude)
      cftc_spec_extreme_short — pctile <= 10 (bullish — squeeze setup)
      cftc_days_old          — days since the report was published
    """
    out: dict[str, Any] = {
        "cftc_spec_net": None,
        "cftc_spec_net_pctile": None,
        "cftc_spec_extreme_long": False,
        "cftc_spec_extreme_short": False,
        "cftc_days_old": None,
    }
    if df is None or df.empty or len(df) < 60:
        return out

    if as_of_date is None:
        as_of_ts = pd.Timestamp(datetime.now(timezone.utc))
    else:
        as_of_ts = pd.Timestamp(as_of_date)
        if as_of_ts.tz is None:
            as_of_ts = as_of_ts.tz_localize("UTC")

    # Make the period column tz-aware
    work = df.copy()
    if work["period"].dt.tz is None:
        work["period"] = work["period"].dt.tz_localize("UTC")
    # CFTC publishes Friday for Tuesday data — 3-day lag
    work["available_at"] = work["period"] + pd.Timedelta(days=3)
    visible = work[work["available_at"] <= as_of_ts].copy()
    if len(visible) < 52:
        return out

    visible["spec_net"] = visible["noncomm_long"] - visible["noncomm_short"]
    last = visible.iloc[-1]
    spec_net = float(last["spec_net"])
    recent_52 = visible["spec_net"].tail(52)
    pctile = (recent_52 <= spec_net).sum() / len(recent_52) * 100
    days_old = (as_of_ts - last["available_at"]).days

    out["cftc_spec_net"] = round(spec_net, 0)
    out["cftc_spec_net_pctile"] = round(pctile, 1)
    out["cftc_spec_extreme_long"] = pctile >= 90
    out["cftc_spec_extreme_short"] = pctile <= 10
    out["cftc_days_old"] = int(days_old)
    return out

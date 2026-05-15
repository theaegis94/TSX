"""EIA Open Data API client — weekly oil + natgas inventory data.

Why this matters:
  - The EIA Weekly Petroleum Status Report (Wed 10:30 AM ET) is the
    single biggest catalyst for WTI crude price moves. Surprise vs.
    consensus regularly causes ±3-5% same-day moves.
  - The EIA Weekly Natural Gas Storage Report (Thu 10:30 AM ET) is
    the equivalent for natgas — even bigger relative moves because
    storage is more inelastic.
  - Both are FREE via the EIA Open Data API.

Setup:
  1. Register a free key: https://www.eia.gov/opendata/register.php
  2. Set the env var (one-time):
       Windows PowerShell:  $env:EIA_API_KEY = "your_key_here"
       Or persistent:       setx EIA_API_KEY "your_key_here"
       Or .streamlit/secrets.toml:  EIA_API_KEY = "your_key_here"
  3. Restart the Streamlit app + the paper-trader agent.

Without a key: every fetch returns an empty DataFrame and inventory
features stay None — the related strategies will simply not fire.
The non-inventory strategies (oil_rsi_reversion etc.) keep working.

"Surprise" feature definition:
  surprise = this_week_change - mean(prior_4_week_changes)
  Positive = bigger build than recent trend (bearish for that commodity)
  Negative = bigger draw than recent trend (bullish)

We expose a `days_since_report` field so strategies can refuse to act
on stale data — typically you only want to trade on the report day or
the next.
"""
from __future__ import annotations

import logging
import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger("paper_trader.eia")

# Cache the raw EIA responses to disk so backtest reruns don't re-fetch
_HERE = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _HERE / ".eia_cache"
_CACHE_DIR.mkdir(exist_ok=True)

# Use the legacy v1 series API — much simpler than v2's facet system
# and the series IDs are stable and well-documented. (v1 is "deprecated"
# but EIA has kept it working for years and the failure mode if they
# ever turn it off is graceful — same as having no key.)
EIA_V1_BASE = "https://api.eia.gov/series/"

# US ending crude oil stocks (commercial, excl. SPR), weekly, thousand bbl
OIL_STOCKS_SERIES = "PET.WCRSTUS1.W"
# Lower-48 working gas in underground storage, weekly, billion cubic feet
NATGAS_STORAGE_SERIES = "NG.NW2_EPG0_SWO_R48_BCF.W"

_API_KEY_WARNED = False


def _get_api_key() -> str | None:
    """Get the EIA key from env var, falling back to Streamlit secrets
    if Streamlit is loaded. Returns None if not set."""
    global _API_KEY_WARNED
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st  # type: ignore
            key = st.secrets.get("EIA_API_KEY", "")
        except Exception:
            pass
    if not key:
        if not _API_KEY_WARNED:
            LOGGER.warning(
                "EIA_API_KEY not set — inventory features will be "
                "unavailable. Register a free key at "
                "https://www.eia.gov/opendata/register.php"
            )
            _API_KEY_WARNED = True
        return None
    return key


def _fetch_series(series_id: str) -> pd.DataFrame:
    """Hit the EIA v1 series API and return a 2-column DataFrame
    [period (datetime), value (float)] sorted ascending. Empty if
    no key or API failure.

    v1 response shape:
      {
        "series": [{
          "series_id": "PET.WCRSTUS1.W",
          "data": [["20231201", 462123], ["20231124", 458200], ...],
          ...
        }]
      }
    """
    key = _get_api_key()
    if not key:
        return pd.DataFrame()
    url = f"{EIA_V1_BASE}?api_key={key}&series_id={series_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        LOGGER.warning(f"EIA fetch failed for {series_id}: {e}")
        return pd.DataFrame()
    series_list = (data or {}).get("series") or []
    if not series_list:
        # Try to surface the actual API error if there is one
        err = (data or {}).get("data", {}).get("error") or (data or {}).get("error")
        msg = f" ({err})" if err else ""
        LOGGER.warning(f"EIA returned no series for {series_id}{msg}")
        return pd.DataFrame()
    data_rows = series_list[0].get("data") or []
    if not data_rows:
        LOGGER.warning(f"EIA returned 0 rows for {series_id}")
        return pd.DataFrame()
    rows = []
    for row in data_rows:
        try:
            date_str, value = row[0], row[1]
            if value is None:
                continue
            rows.append({
                "period": pd.Timestamp(date_str),
                "value": float(value),
            })
        except (TypeError, ValueError, IndexError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def _cache_path(name: str) -> pathlib.Path:
    return _CACHE_DIR / f"{name}.parquet"


def _read_cache(name: str, max_age_hours: int = 6) -> pd.DataFrame | None:
    """Read a cached parquet if it exists and is fresh enough."""
    p = _cache_path(name)
    if not p.exists():
        return None
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        p.stat().st_mtime, tz=timezone.utc
    )
    if age > timedelta(hours=max_age_hours):
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(name), index=False)
    except Exception:
        # parquet engine missing? fall back to CSV cache silently
        try:
            df.to_csv(_cache_path(name).with_suffix(".csv"), index=False)
        except Exception:
            pass


def fetch_oil_stocks() -> pd.DataFrame:
    """US crude oil stocks (commercial, ex-SPR), weekly, thousand bbl.
    Cached for 6 hours to avoid hammering the API."""
    cached = _read_cache("oil_stocks")
    if cached is not None and not cached.empty:
        return cached
    df = _fetch_series(OIL_STOCKS_SERIES)
    if not df.empty:
        _write_cache("oil_stocks", df)
    return df


def fetch_natgas_storage() -> pd.DataFrame:
    """Lower-48 working gas in underground storage, weekly, Bcf.
    Cached for 6 hours."""
    cached = _read_cache("natgas_storage")
    if cached is not None and not cached.empty:
        return cached
    df = _fetch_series(NATGAS_STORAGE_SERIES)
    if not df.empty:
        _write_cache("natgas_storage", df)
    return df


def _compute_surprise_features(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
    prefix: str = "inv",
    publication_lag_days: int = 5,
) -> dict[str, Any]:
    """Given a weekly inventory series, compute the latest surprise
    relative to the prior 4-week trailing average change.

    `as_of_date` is the date we're computing features FOR. For live
    use this is "today"; for backtest it's the historical decision date.
    Only reports with `period + publication_lag_days <= as_of_date`
    are considered "known" — protects against future leakage.

    `publication_lag_days` = 5 because EIA publishes a report dated
    Friday of week N on the following Wed (oil) or Thu (gas) — about
    5-6 calendar days after the period end.
    """
    out: dict[str, Any] = {
        f"{prefix}_change": None,
        f"{prefix}_surprise": None,
        f"{prefix}_days_since_report": None,
    }
    if df is None or df.empty or len(df) < 6:
        return out

    if as_of_date is None:
        cutoff = pd.Timestamp(datetime.now(timezone.utc))
    else:
        cutoff = pd.Timestamp(as_of_date)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize("UTC")

    # Make the data's period column tz-aware in UTC for consistent compare
    series = df.copy()
    series["period"] = pd.to_datetime(series["period"]).dt.tz_localize(
        "UTC", nonexistent="shift_forward", ambiguous="NaT"
    ) if series["period"].dt.tz is None else series["period"]

    # The data IS public starting `period + publication_lag_days`.
    series["available_at"] = series["period"] + pd.Timedelta(
        days=publication_lag_days
    )
    visible = series[series["available_at"] <= cutoff].copy()
    if len(visible) < 6:
        return out

    # Weekly changes
    visible["change"] = visible["value"].diff()
    last = visible.iloc[-1]
    prior_4 = visible["change"].iloc[-5:-1]  # 4 prior weeks
    avg_prior = float(prior_4.mean()) if len(prior_4) > 0 else 0.0
    change_now = float(last["change"]) if pd.notna(last["change"]) else None
    if change_now is None:
        return out
    surprise = change_now - avg_prior
    days_since = (cutoff - last["available_at"]).days

    out[f"{prefix}_change"] = round(change_now, 1)
    out[f"{prefix}_surprise"] = round(surprise, 1)
    out[f"{prefix}_days_since_report"] = int(days_since)
    return out


def oil_inventory_features(
    as_of_date=None,
    precomputed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute oil inventory features for live or backtest.
    Prefix: oil_inv. Unit: thousand barrels."""
    df = precomputed if precomputed is not None else fetch_oil_stocks()
    return _compute_surprise_features(df, as_of_date, prefix="oil_inv")


def natgas_storage_features(
    as_of_date=None,
    precomputed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute natgas storage features. Prefix: gas_stor. Unit: Bcf."""
    df = precomputed if precomputed is not None else fetch_natgas_storage()
    return _compute_surprise_features(df, as_of_date, prefix="gas_stor")

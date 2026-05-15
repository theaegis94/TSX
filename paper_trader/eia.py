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

# EIA v2 API. (v1 was retired — returns 404 as of mid-2025.)
EIA_V2_BASE = "https://api.eia.gov/v2"

# Per-endpoint config. v2's facet names differ between commodities,
# so we keep them in a structured dict instead of one URL string.
# For each endpoint we also store a list of fallback facet sets to try
# if the first one returns 0 rows (defensive against EIA facet renames).
OIL_STOCKS_ENDPOINT = {
    "name": "oil_stocks",
    "path": "petroleum/stoc/wstk/data/",
    "facet_attempts": [
        # Primary: U.S. ending stocks of crude oil excluding SPR
        {"duoarea": "NUS", "product": "EPC0", "process": "SAX"},
        # Fallback 1: drop the process filter (returns multiple series;
        # we'll filter to commercial-stocks-like values client-side)
        {"duoarea": "NUS", "product": "EPC0"},
        # Fallback 2: just crude oil products, US-wide
        {"product": "EPC0"},
    ],
    # Client-side hint: prefer rows whose series-description hints at
    # commercial crude oil ending stocks
    "name_hints": [
        "commercial crude oil",
        "ending stocks excluding spr",
        "ending stocks of crude oil",
    ],
}

NATGAS_STORAGE_ENDPOINT = {
    "name": "natgas_storage",
    "path": "natural-gas/stor/wkly/data/",
    "facet_attempts": [
        # Primary: Lower 48 working gas in underground storage
        {"duoarea": "R48", "process": "SAW"},
        # Fallback 1: working gas, alternative facet code
        {"duoarea": "NUS", "process": "SAW"},
        # Fallback 2: Lower 48 only
        {"duoarea": "R48"},
        # Fallback 3: minimal — fetch and filter client-side
        {},
    ],
    "name_hints": [
        "working gas in underground storage",
        "lower 48",
    ],
}

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


def _build_v2_url(path: str, facets: dict, key: str) -> str:
    """Construct a v2 data URL with facets correctly URL-encoded."""
    parts = [
        f"api_key={key}",
        "frequency=weekly",
        "data[0]=value",
        "sort[0][column]=period",
        "sort[0][direction]=desc",
        "offset=0",
        "length=5000",
    ]
    for k, v in facets.items():
        parts.append(f"facets[{k}][]={v}")
    return f"{EIA_V2_BASE}/{path}?" + "&".join(parts)


def _fetch_v2_single(url: str) -> tuple[list, str | None]:
    """Single v2 GET. Returns (data_rows, err_str). err_str is None on
    success, otherwise a short human-readable error for logging."""
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        return [], f"network error: {e}"
    if resp.status_code == 404:
        return [], f"404 — bad endpoint path: {url.split('?')[0]}"
    try:
        resp.raise_for_status()
    except Exception as e:
        return [], f"HTTP {resp.status_code}: {e}"
    try:
        data = resp.json()
    except Exception:
        return [], "non-JSON response"
    items = (data or {}).get("response", {}).get("data") or []
    if not items:
        err = (data or {}).get("response", {}).get("error")
        return [], (f"0 rows ({err})" if err else "0 rows")
    return items, None


def _rows_to_df(items: list, name_hints: list[str]) -> pd.DataFrame:
    """Convert raw EIA items to a clean [period, value] DataFrame.
    If multiple distinct series are present, prefer the one whose
    `series-description` matches one of `name_hints` (case-insensitive).
    """
    if not items:
        return pd.DataFrame()
    # Detect if multiple series were returned (each row has a
    # series-description); if so, prefer the best-matching one.
    descriptions = {it.get("series-description") for it in items
                    if it.get("series-description")}
    chosen_desc = None
    if len(descriptions) > 1 and name_hints:
        lower_hints = [h.lower() for h in name_hints]
        for desc in descriptions:
            d_low = (desc or "").lower()
            if any(h in d_low for h in lower_hints):
                chosen_desc = desc
                break
    rows = []
    for it in items:
        if chosen_desc and it.get("series-description") != chosen_desc:
            continue
        try:
            period = it.get("period")
            value = it.get("value")
            if period is None or value is None:
                continue
            rows.append({
                "period": pd.Timestamp(period),
                "value": float(value),
            })
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def _fetch_endpoint(endpoint_config: dict) -> pd.DataFrame:
    """Try each facet attempt in order, return first non-empty result.
    Logs detailed diagnostics on failure so we can debug facet names."""
    key = _get_api_key()
    if not key:
        return pd.DataFrame()
    name = endpoint_config["name"]
    path = endpoint_config["path"]
    name_hints = endpoint_config.get("name_hints", [])
    for i, facets in enumerate(endpoint_config["facet_attempts"]):
        url = _build_v2_url(path, facets, key)
        items, err = _fetch_v2_single(url)
        if err:
            facet_label = (
                ", ".join(f"{k}={v}" for k, v in facets.items())
                or "(no facets)"
            )
            LOGGER.warning(
                f"EIA {name} attempt {i + 1} [{facet_label}] failed: {err}"
            )
            continue
        df = _rows_to_df(items, name_hints)
        if not df.empty:
            LOGGER.info(
                f"EIA {name}: fetched {len(df)} weekly rows "
                f"using facets {facets or '(none)'}"
            )
            return df
    LOGGER.error(
        f"EIA {name}: all facet attempts returned empty. "
        f"Inventory features for this commodity will stay None."
    )
    return pd.DataFrame()


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
    df = _fetch_endpoint(OIL_STOCKS_ENDPOINT)
    if not df.empty:
        _write_cache("oil_stocks", df)
    return df


def fetch_natgas_storage() -> pd.DataFrame:
    """Lower-48 working gas in underground storage, weekly, Bcf.
    Cached for 6 hours."""
    cached = _read_cache("natgas_storage")
    if cached is not None and not cached.empty:
        return cached
    df = _fetch_endpoint(NATGAS_STORAGE_ENDPOINT)
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

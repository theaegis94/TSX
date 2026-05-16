"""Weather data for natural-gas demand prediction.

The thesis: natgas demand is dominated by:
  - WINTER heating (HDD = heating degree days)
  - SUMMER cooling (CDD = cooling degree days, drives power generation)

When actual + forecasted weather deviates from seasonal normal,
natgas prices respond. Cold-snap forecasts → HNU rallies; mild-winter
forecasts → HND rallies.

This is one of the few fundamental signals with persistent edge
even AFTER public release, because:
  1. Weather forecasts have meaningful error → news keeps coming
  2. Updates arrive multiple times per day from NOAA/ECMWF
  3. Demand sensitivity is non-linear (extreme cold = much more gas)

Data source: Open-Meteo archive API
  - Free, no API key
  - Daily data back to 1940
  - Documented at https://open-meteo.com/

We track 6 major US metros weighted by natgas residential demand:
  NYC, Chicago, Boston, Atlanta, Houston, Los Angeles
This gives us a national HDD/CDD proxy that matches what natgas
traders care about (EIA's gas-weighted HDD uses similar weighting).
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger("paper_trader.weather")

_HERE = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _HERE / ".eia_cache"  # reuse cache dir
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "weather_hdd_cdd.parquet"

# Population-weighted cities. Weights chosen to approximate EIA's
# gas-weighted HDD index (NE and Midwest dominate winter; SE and SW
# dominate summer). Sum of weights = 1.0.
CITIES = [
    # (name, lat, lon, weight)
    ("New York",    40.7128, -74.0060, 0.25),
    ("Chicago",     41.8781, -87.6298, 0.20),
    ("Boston",      42.3601, -71.0589, 0.10),
    ("Atlanta",     33.7490, -84.3880, 0.10),
    ("Houston",     29.7604, -95.3698, 0.20),
    ("Los Angeles", 34.0522, -118.2437, 0.15),
]

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Base temperature for HDD/CDD computation (industry standard: 65°F)
HDD_BASE_F = 65.0


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


def _fetch_city(lat: float, lon: float,
                start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily mean temperature for one city."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        LOGGER.warning(f"Open-Meteo fetch failed ({lat},{lon}): {e}")
        return pd.DataFrame()
    daily = (data or {}).get("daily") or {}
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not times or not temps or len(times) != len(temps):
        return pd.DataFrame()
    rows = []
    for t, temp in zip(times, temps):
        if temp is None:
            continue
        try:
            rows.append({"date": pd.Timestamp(t), "temp_f": float(temp)})
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_weather_history(years_back: int = 12) -> pd.DataFrame:
    """Fetch + cache population-weighted daily HDD/CDD for the past
    N years. Returns DataFrame with columns:
      date, hdd, cdd, temp_avg_f, normal_hdd, normal_cdd,
      hdd_anomaly, cdd_anomaly
    """
    cached = _read_cache(max_age_hours=24)
    if cached is not None and not cached.empty:
        return cached

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=years_back * 366 + 30)
    end_str = end_date.strftime("%Y-%m-%d")
    start_str = start_date.strftime("%Y-%m-%d")
    LOGGER.info(
        f"Fetching weather data {start_str} to {end_str} "
        f"across {len(CITIES)} cities…"
    )

    # Per-city dataframes, indexed by date
    per_city = {}
    for name, lat, lon, weight in CITIES:
        df = _fetch_city(lat, lon, start_str, end_str)
        if df.empty:
            LOGGER.warning(f"No data for {name}; skipping")
            continue
        df = df.set_index("date")
        per_city[name] = (df, weight)
        LOGGER.info(f"  {name}: {len(df)} days")

    if not per_city:
        LOGGER.error("All city fetches failed")
        return pd.DataFrame()

    # Build a single DataFrame aligned on date with population-weighted
    # average temp, then compute HDD/CDD from that.
    aligned = pd.DataFrame()
    total_weight = sum(w for _, w in per_city.values())
    for name, (df, weight) in per_city.items():
        aligned[name] = df["temp_f"] * (weight / total_weight)
    weighted_temp = aligned.sum(axis=1, min_count=1)
    out = pd.DataFrame({
        "date": weighted_temp.index,
        "temp_avg_f": weighted_temp.values,
    }).dropna()
    out["hdd"] = (HDD_BASE_F - out["temp_avg_f"]).clip(lower=0)
    out["cdd"] = (out["temp_avg_f"] - HDD_BASE_F).clip(lower=0)

    # Compute seasonal normals (day-of-year average across all years)
    out["day_of_year"] = pd.to_datetime(out["date"]).dt.dayofyear
    normals_hdd = out.groupby("day_of_year")["hdd"].mean()
    normals_cdd = out.groupby("day_of_year")["cdd"].mean()
    out["normal_hdd"] = out["day_of_year"].map(normals_hdd)
    out["normal_cdd"] = out["day_of_year"].map(normals_cdd)
    out["hdd_anomaly"] = out["hdd"] - out["normal_hdd"]
    out["cdd_anomaly"] = out["cdd"] - out["normal_cdd"]
    out = out.drop(columns=["day_of_year"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])

    _write_cache(out)
    LOGGER.info(f"Weather: cached {len(out)} daily rows")
    return out


def compute_weather_change_features(
    df: pd.DataFrame,
    as_of_date=None,
) -> dict[str, Any]:
    """Compute weather TRANSITION features (no lookahead).

    The hypothesis: traders react slowly to weather changes. If today's
    7-day HDD just spiked from last week's level, the move hasn't been
    fully priced. We test this using ONLY past data — strict no-lookahead.

    Returns:
      weather_hdd_change_7v7 — current 7d HDD vs prior 7d HDD (% change)
      weather_cdd_change_7v7 — same for cooling
      weather_cold_front     — bool: HDD up >50% over 7 days
      weather_warm_front     — bool: HDD down >50% over 7 days
    """
    out: dict[str, Any] = {
        "weather_hdd_change_7v7": None,
        "weather_cdd_change_7v7": None,
        "weather_cold_front": False,
        "weather_warm_front": False,
    }
    if df is None or df.empty:
        return out
    if as_of_date is None:
        as_of = pd.Timestamp(datetime.now(timezone.utc).date())
    else:
        as_of = pd.Timestamp(as_of_date)
        if as_of.tz is not None:
            as_of = as_of.tz_localize(None)
    cutoff = as_of - pd.Timedelta(days=1)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    visible = work[work["date"] <= cutoff].sort_values("date")
    if len(visible) < 14:
        return out

    # Current 7 days vs prior 7 days
    cur7 = visible.tail(7)
    prior7 = visible.tail(14).head(7)
    cur_hdd = float(cur7["hdd"].sum())
    prior_hdd = float(prior7["hdd"].sum())
    cur_cdd = float(cur7["cdd"].sum())
    prior_cdd = float(prior7["cdd"].sum())

    if prior_hdd > 10:  # only meaningful if we had heating demand before
        change = (cur_hdd - prior_hdd) / prior_hdd * 100
        out["weather_hdd_change_7v7"] = round(change, 1)
        out["weather_cold_front"] = change > 50  # 50% spike in HDD
        out["weather_warm_front"] = change < -50
    if prior_cdd > 10:
        change_c = (cur_cdd - prior_cdd) / prior_cdd * 100
        out["weather_cdd_change_7v7"] = round(change_c, 1)
    return out


def compute_weather_oracle_features(
    df: pd.DataFrame,
    as_of_date=None,
    lookahead_days: int = 7,
) -> dict[str, Any]:
    """⚠️ ORACLE / LOOKAHEAD BIAS ⚠️

    Uses ACTUAL FUTURE weather data as a feature. This is intentional
    — it's an upper-bound test: 'if we had a perfect 7-day forecast,
    would it help our natgas strategies?' If yes, real (imperfect)
    forecasts might add edge. If no, no forecast will save natgas.

    DO NOT USE THIS IN LIVE TRADING. Only valid for research.

    Returns:
      weather_oracle_hdd_future7 — sum of HDD over next 7 days
      weather_oracle_hdd_anomaly — same, vs seasonal normal
      weather_oracle_cold_coming — bool: future HDD >+30% vs normal
      weather_oracle_warm_coming — bool: future HDD <-30% vs normal
    """
    out: dict[str, Any] = {
        "weather_oracle_hdd_future7": None,
        "weather_oracle_hdd_anomaly": None,
        "weather_oracle_cold_coming": False,
        "weather_oracle_warm_coming": False,
    }
    if df is None or df.empty:
        return out
    if as_of_date is None:
        as_of = pd.Timestamp(datetime.now(timezone.utc).date())
    else:
        as_of = pd.Timestamp(as_of_date)
        if as_of.tz is not None:
            as_of = as_of.tz_localize(None)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    # LOOKAHEAD: grab days STRICTLY AFTER as_of, up to lookahead_days
    future = work[
        (work["date"] > as_of)
        & (work["date"] <= as_of + pd.Timedelta(days=lookahead_days))
    ]
    if len(future) < lookahead_days:
        return out
    future_hdd = float(future["hdd"].sum())
    normal_hdd = float(future["normal_hdd"].sum())
    out["weather_oracle_hdd_future7"] = round(future_hdd, 1)
    out["weather_oracle_hdd_anomaly"] = round(future_hdd - normal_hdd, 1)
    if normal_hdd > 5:
        anomaly_pct = (future_hdd - normal_hdd) / normal_hdd * 100
        out["weather_oracle_cold_coming"] = anomaly_pct > 30
        out["weather_oracle_warm_coming"] = anomaly_pct < -30
    return out


def compute_weather_features(
    df: pd.DataFrame,
    as_of_date=None,
) -> dict[str, Any]:
    """Compute weather features as of `as_of_date`. Walk-forward-safe:
    only uses data dated <= as_of_date (no forecast lookahead).

    For a true prediction agent, we'd also fetch the 6-10 day FORECAST
    from NOAA CPC — leaving that as a future enhancement. Here we use
    a 7-day TRAILING anomaly which still tells us "is this week
    unusually cold/hot" — a strong signal on its own.

    Returns:
      weather_hdd_7d         — 7-day trailing HDD sum
      weather_hdd_anomaly_7d — 7-day HDD vs seasonal normal
      weather_cdd_7d         — 7-day trailing CDD
      weather_cdd_anomaly_7d — 7-day CDD vs seasonal normal
      weather_cold_snap      — bool: HDD anomaly > +20% of normal
      weather_warm_anomaly   — bool: HDD anomaly < -20% in winter
      weather_heat_dome      — bool: CDD anomaly > +30% in summer
    """
    out: dict[str, Any] = {
        "weather_hdd_7d": None,
        "weather_hdd_anomaly_7d": None,
        "weather_cdd_7d": None,
        "weather_cdd_anomaly_7d": None,
        "weather_cold_snap": False,
        "weather_warm_anomaly": False,
        "weather_heat_dome": False,
    }
    if df is None or df.empty:
        return out

    if as_of_date is None:
        as_of = pd.Timestamp(datetime.now(timezone.utc).date())
    else:
        as_of = pd.Timestamp(as_of_date)
        if as_of.tz is not None:
            as_of = as_of.tz_localize(None)
    # Slice to data up to as_of (weather data has ~1-2 day publication lag)
    cutoff = as_of - pd.Timedelta(days=1)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    visible = work[work["date"] <= cutoff].sort_values("date")
    if len(visible) < 14:
        return out

    last7 = visible.tail(7)
    hdd_7 = float(last7["hdd"].sum())
    cdd_7 = float(last7["cdd"].sum())
    normal_hdd_7 = float(last7["normal_hdd"].sum())
    normal_cdd_7 = float(last7["normal_cdd"].sum())

    out["weather_hdd_7d"] = round(hdd_7, 1)
    out["weather_cdd_7d"] = round(cdd_7, 1)
    out["weather_hdd_anomaly_7d"] = round(hdd_7 - normal_hdd_7, 1)
    out["weather_cdd_anomaly_7d"] = round(cdd_7 - normal_cdd_7, 1)

    # Boolean regimes (used as conviction boosters by strategies)
    if normal_hdd_7 > 5:  # only meaningful when there's heating demand
        anomaly_pct = (hdd_7 - normal_hdd_7) / normal_hdd_7 * 100
        out["weather_cold_snap"] = anomaly_pct > 20
        out["weather_warm_anomaly"] = anomaly_pct < -20
    if normal_cdd_7 > 5:  # only meaningful when there's cooling demand
        cdd_anomaly_pct = (cdd_7 - normal_cdd_7) / normal_cdd_7 * 100
        out["weather_heat_dome"] = cdd_anomaly_pct > 30
    return out

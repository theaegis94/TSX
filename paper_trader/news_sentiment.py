"""News-sentiment layer for oil/natgas using Claude as the analyst.

The ML predictor in predictor.py is blind to news — it only sees price
patterns. This module fetches recent energy headlines from Finnhub and
asks Claude Haiku to score them on a -1 to +1 scale for next-day WTI
and Henry Hub natgas direction.

Output (compute_news_sentiment):
  {
    "as_of": "2026-05-18T18:30:00",
    "oil": {
      "score": -0.62,
      "direction": "bearish",
      "reasoning": "<short explanation>",
      "headlines_analyzed": [...],
    },
    "gas": {...},
    "n_headlines": 18,
  }

Honest limitations (printed to UI):
  - Backtest is not meaningfully possible (Finnhub free tier ~1y history)
  - News sentiment moves AFTER headlines drop — not predictive on short
    horizons unless you trade fast
  - Mixed-news days get averaged toward zero
  - Costs ~$0.005/day (Claude Haiku pricing)
  - Graceful fail: if FINNHUB_API_KEY or ANTHROPIC_API_KEY is missing,
    the module returns None and the dashboard panel simply doesn't show

This is meant to be USED ALONGSIDE the ML predictor, not in place of it.
On days where the ML is in FALLBACK tier (essentially ~50% noise) and
news sentiment is strong, the news layer is the more informative signal.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

LOGGER = logging.getLogger("paper_trader.news_sentiment")

# Keywords used to filter Finnhub's general news feed down to oil/gas relevant
# stories. Wide net by design — we'd rather over-include and let Claude
# filter than miss relevant headlines.
OIL_KEYWORDS = {
    "oil", "crude", "wti", "brent", "opec", "saudi", "iran", "venezuela",
    "hormuz", "tanker", "refinery", "sanctions", "drilling", "rig count",
    "shale", "petroleum", "barrel", "gasoline", "energy", "permian",
    "exxon", "chevron", "saudi aramco", "russia oil", "spr",
}
GAS_KEYWORDS = {
    "natural gas", "natgas", "lng", "henry hub", "henry-hub", "pipeline",
    "shale gas", "storage report", "winter heating", "cold front",
    "polar vortex", "heating demand", "cooling demand", "freeze",
    "hurricane gulf", "permian gas",
}
# Anything that could move both / macro
MACRO_KEYWORDS = {
    "fed rate", "dollar index", "dxy", "recession", "inflation",
    "geopolitical", "ukraine", "middle east", "war",
}


def _anthropic_key() -> str | None:
    """Try env var, then Streamlit secrets, then dotenv."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    return None


def _finnhub_key() -> str | None:
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        key = st.secrets.get("FINNHUB_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    return None


def fetch_energy_news(hours_back: int = 36, limit: int = 40) -> list[dict]:
    """Pull Finnhub general news, filter to oil/gas/macro relevance.

    Returns list of dicts: {datetime (unix), headline, summary, source, url}.
    Newest first, deduplicated by headline.
    """
    key = _finnhub_key()
    if not key:
        LOGGER.warning("FINNHUB_API_KEY not set; cannot fetch news.")
        return []
    cutoff_ts = int(
        (datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp()
    )
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": key},
            timeout=20,
        )
        if r.status_code != 200:
            LOGGER.warning(f"Finnhub returned HTTP {r.status_code}")
            return []
        items = r.json() or []
    except Exception as e:
        LOGGER.warning(f"Finnhub fetch failed: {e}")
        return []

    relevant = []
    seen_headlines = set()
    all_kw = OIL_KEYWORDS | GAS_KEYWORDS | MACRO_KEYWORDS
    for it in items:
        try:
            ts = int(it.get("datetime") or 0)
            if ts < cutoff_ts:
                continue
            head = (it.get("headline") or "").strip()
            summary = (it.get("summary") or "").strip()
            if not head or head in seen_headlines:
                continue
            text = f"{head} {summary}".lower()
            if not any(kw in text for kw in all_kw):
                continue
            seen_headlines.add(head)
            relevant.append({
                "datetime": ts,
                "headline": head,
                "summary": summary[:300],
                "source": it.get("source", ""),
                "url": it.get("url", ""),
            })
        except (TypeError, ValueError):
            continue
    # Sort newest first, limit
    relevant.sort(key=lambda x: x["datetime"], reverse=True)
    return relevant[:limit]


def score_with_claude(headlines: list[dict]) -> dict[str, Any] | None:
    """Send headlines to Claude Haiku, get structured sentiment scores.

    Returns:
      {
        "oil":  {"score": float, "direction": str, "reasoning": str},
        "gas":  {"score": float, "direction": str, "reasoning": str},
        "n":    int (headlines analyzed),
      }
    """
    if not headlines:
        return None
    key = _anthropic_key()
    if not key:
        LOGGER.warning("ANTHROPIC_API_KEY not set; cannot score sentiment.")
        return None
    try:
        import anthropic
    except ImportError:
        LOGGER.error("anthropic package not installed")
        return None

    formatted = "\n".join(
        f"- ({h.get('source','?')}) {h['headline']}"
        + (f" — {h['summary'][:200]}" if h.get("summary") else "")
        for h in headlines
    )

    prompt = (
        "You are an energy-markets analyst. Given the following recent "
        "news headlines (newest first), score the implications for "
        "NEXT-DAY price direction of two assets:\n"
        "  1. WTI crude oil\n"
        "  2. Henry Hub natural gas\n\n"
        "Each score is from -1.0 (strongly bearish — expect price down) "
        "to +1.0 (strongly bullish — expect price up). Use 0.0 for net "
        "neutral or mixed signals.\n\n"
        "Consider: supply (production, sanctions, OPEC, rig count, "
        "outages), demand (driving season, weather, industrial output, "
        "China imports, LNG exports), and macro (USD, recession risk, "
        "war/geopolitical).\n\n"
        "Respond with ONLY a JSON object, no markdown, no commentary "
        "outside the JSON. Format:\n"
        "{\n"
        '  "oil":  {"score": <float>, "reasoning": "<1-2 sentence summary>"},\n'
        '  "gas":  {"score": <float>, "reasoning": "<1-2 sentence summary>"}\n'
        "}\n\n"
        f"Headlines ({len(headlines)} total):\n{formatted}"
    )

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Claude sometimes wraps JSON in ```json fences; strip them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")
        data = json.loads(text)
    except json.JSONDecodeError as e:
        LOGGER.warning(f"Claude returned non-JSON: {e}")
        return None
    except Exception as e:
        LOGGER.warning(f"Claude API call failed: {e}")
        return None

    # Normalize + add direction labels
    out = {"n": len(headlines)}
    for k in ("oil", "gas"):
        block = data.get(k, {})
        try:
            score = float(block.get("score", 0))
            score = max(-1.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        reasoning = str(block.get("reasoning", ""))[:500]
        if score >= 0.5:
            direction = "strongly bullish"
        elif score >= 0.15:
            direction = "bullish"
        elif score >= -0.15:
            direction = "neutral / mixed"
        elif score >= -0.5:
            direction = "bearish"
        else:
            direction = "strongly bearish"
        out[k] = {
            "score": round(score, 2),
            "direction": direction,
            "reasoning": reasoning,
        }
    return out


def compute_news_sentiment(
    hours_back: int = 36,
    max_headlines: int = 30,
) -> dict[str, Any] | None:
    """End-to-end: fetch news + score with Claude. Returns None if any
    upstream piece (key, API) is unavailable so the caller can hide the
    panel gracefully.
    """
    headlines = fetch_energy_news(
        hours_back=hours_back, limit=max_headlines,
    )
    if not headlines:
        return None
    scores = score_with_claude(headlines)
    if scores is None:
        return None
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "oil": scores.get("oil"),
        "gas": scores.get("gas"),
        "n_headlines": scores.get("n", len(headlines)),
        "headlines": headlines,
    }

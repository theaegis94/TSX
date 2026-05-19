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
    """Try env var, then Streamlit secrets. Robust against st.secrets
    returning non-string types."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if isinstance(key, str) and key.strip():
        return key.strip()
    try:
        import streamlit as st  # type: ignore
        raw = st.secrets.get("ANTHROPIC_API_KEY", "")
        key = str(raw).strip()
        if key:
            return key
    except Exception:
        pass
    return None


def _finnhub_key() -> str | None:
    """Reuse stock_signals' resolved key — that one is loaded at import
    time and already handles env + .env + st.secrets correctly."""
    try:
        import stock_signals as ss
        if ss.FINNHUB_API_KEY and isinstance(ss.FINNHUB_API_KEY, str):
            return ss.FINNHUB_API_KEY.strip() or None
    except Exception:
        pass
    # Direct fallbacks
    key = os.environ.get("FINNHUB_API_KEY", "")
    if isinstance(key, str) and key.strip():
        return key.strip()
    try:
        import streamlit as st  # type: ignore
        raw = st.secrets.get("FINNHUB_API_KEY", "")
        key = str(raw).strip()
        if key:
            return key
    except Exception:
        pass
    return None


def _fetch_finnhub_general(key: str) -> list[dict]:
    """Raw fetch of Finnhub general news. Returns items or empty list."""
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": key},
            timeout=20,
        )
        if r.status_code != 200:
            LOGGER.warning(f"Finnhub /news returned HTTP {r.status_code}")
            return []
        return r.json() or []
    except Exception as e:
        LOGGER.warning(f"Finnhub fetch failed: {e}")
        return []


def _fetch_finnhub_company(key: str, symbol: str) -> list[dict]:
    """Per-ticker news (fills in oil/gas specific stories Finnhub buckets
    by company)."""
    from_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": from_date, "to": to_date, "token": key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        return r.json() or []
    except Exception:
        return []


# Tickers we pull company-specific news for to augment the general feed.
# These move with oil/gas fundamentals and provide direct industry signal.
ENERGY_TICKERS_FOR_NEWS = [
    "XOM", "CVX", "COP", "OXY", "USO",   # US oil majors + WTI ETF
    "BNO", "UNG",                          # Brent ETF + natgas ETF
    "SU.TO", "CNQ.TO", "ENB.TO", "TRP.TO", # CA energy
]


def fetch_energy_news(hours_back: int = 36, limit: int = 40,
                      include_diagnostics: bool = False) -> list[dict] | tuple:
    """Pull Finnhub general news + per-ticker oil/gas news, filter to
    oil/gas/macro relevance.

    If include_diagnostics=True, returns (headlines, diag_dict) so
    callers can show counts/reasons in the UI.
    """
    key = _finnhub_key()
    if not key:
        LOGGER.warning("FINNHUB_API_KEY not set; cannot fetch news.")
        return ([], {"error": "no_finnhub_key"}) if include_diagnostics else []
    cutoff_ts = int(
        (datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp()
    )

    # 1) General news (broad market) — large list
    general_items = _fetch_finnhub_general(key)

    # 2) Per-ticker news for energy companies (more focused signal)
    ticker_items = []
    for sym in ENERGY_TICKERS_FOR_NEWS:
        ticker_items.extend(_fetch_finnhub_company(key, sym))

    all_items = general_items + ticker_items
    diag = {
        "general_count": len(general_items),
        "ticker_count": len(ticker_items),
        "raw_total": len(all_items),
    }

    relevant = []
    seen_headlines = set()
    all_kw = OIL_KEYWORDS | GAS_KEYWORDS | MACRO_KEYWORDS
    in_window = 0
    keyword_matched = 0
    for it in all_items:
        try:
            ts = int(it.get("datetime") or 0)
            if ts < cutoff_ts:
                continue
            in_window += 1
            head = (it.get("headline") or "").strip()
            summary = (it.get("summary") or "").strip()
            if not head or head in seen_headlines:
                continue
            text = f"{head} {summary}".lower()
            if not any(kw in text for kw in all_kw):
                continue
            keyword_matched += 1
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

    diag["in_36h_window"] = in_window
    diag["keyword_matched"] = keyword_matched

    # FALLBACK: if keyword filter is too narrow today, take the most
    # recent ticker-specific items unfiltered (already energy-tagged)
    if len(relevant) < 5:
        LOGGER.info(
            f"Only {len(relevant)} keyword-matched headlines — adding "
            f"top {30-len(relevant)} energy-ticker items as fallback"
        )
        for it in ticker_items:
            try:
                ts = int(it.get("datetime") or 0)
                if ts < cutoff_ts:
                    continue
                head = (it.get("headline") or "").strip()
                if not head or head in seen_headlines:
                    continue
                seen_headlines.add(head)
                relevant.append({
                    "datetime": ts,
                    "headline": head,
                    "summary": (it.get("summary") or "")[:300],
                    "source": it.get("source", ""),
                    "url": it.get("url", ""),
                })
                if len(relevant) >= 25:
                    break
            except (TypeError, ValueError):
                continue
        diag["fallback_used"] = True
        diag["after_fallback"] = len(relevant)
    else:
        diag["fallback_used"] = False

    relevant.sort(key=lambda x: x["datetime"], reverse=True)
    relevant = relevant[:limit]
    diag["returned"] = len(relevant)
    return (relevant, diag) if include_diagnostics else relevant


# Models to try in order. If the first one isn't accessible on the
# account, we fall back. claude-haiku-4-5 is the newest; the 3-5 variant
# is the long-tenured one almost every account can hit.
CLAUDE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
]


def score_with_claude(headlines: list[dict]) -> dict[str, Any] | None:
    """Send headlines to Claude, get structured sentiment scores.

    Returns:
      success: {"oil": {...}, "gas": {...}, "n": int}
      failure: {"_error": "<reason>", "_detail": "<api/parse msg>"}
        — caller distinguishes via presence of "_error" key.
    """
    if not headlines:
        return {"_error": "no_headlines", "_detail": "empty headline list"}
    key = _anthropic_key()
    if not key:
        LOGGER.warning("ANTHROPIC_API_KEY not set; cannot score sentiment.")
        return {"_error": "no_anthropic_key", "_detail": "ANTHROPIC_API_KEY not resolved"}
    try:
        import anthropic
    except ImportError:
        LOGGER.error("anthropic package not installed")
        return {"_error": "no_anthropic_package", "_detail": "pip install anthropic"}

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

    client = anthropic.Anthropic(api_key=key)
    last_err: str = ""
    text = ""
    model_used: str = ""
    for model in CLAUDE_MODELS:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            model_used = model
            last_err = ""
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            LOGGER.warning(f"Claude API call failed on {model}: {last_err}")
            continue

    if not text:
        return {"_error": "api_call_failed", "_detail": last_err or "all models failed"}

    # Parse JSON. Claude sometimes wraps in ```json fences or includes a
    # short prose preamble. Try strict first, then extract a JSON object.
    data = None
    parse_err = ""
    candidates = [text]
    if text.startswith("```"):
        inner = text.split("```")[1]
        if inner.startswith("json"):
            inner = inner[4:]
        candidates.append(inner.strip("` \n"))
    # Last-ditch: pull the first {...} block out of the response
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            parse_err = ""
            break
        except json.JSONDecodeError as e:
            parse_err = f"{e}"
            continue

    if data is None:
        snippet = text[:200].replace("\n", " ")
        LOGGER.warning(f"Claude returned non-JSON (model={model_used}): {snippet}")
        return {
            "_error": "json_parse_failed",
            "_detail": f"{parse_err} | snippet: {snippet}",
        }

    # Normalize + add direction labels
    out = {"n": len(headlines), "_model_used": model_used}
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
    """End-to-end: fetch news + score with Claude. Returns None on
    fatal upstream failure; partial diagnostics included in dict on
    soft failures (e.g. headlines fetched but Claude errored).
    """
    headlines, diag = fetch_energy_news(
        hours_back=hours_back, limit=max_headlines,
        include_diagnostics=True,
    )
    if not headlines:
        # Hard fail: nothing to score. Caller can show diagnostics.
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "error": "no_headlines",
            "diagnostics": diag,
            "oil": None,
            "gas": None,
            "n_headlines": 0,
            "headlines": [],
        }
    scores = score_with_claude(headlines)
    if scores is None or scores.get("_error"):
        # Propagate the specific Claude failure into diagnostics so
        # the UI can show api_call_failed vs json_parse_failed etc.
        diag = dict(diag)
        if scores:
            diag["claude_error"] = scores.get("_error", "unknown")
            diag["claude_detail"] = scores.get("_detail", "")
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "error": "claude_failed",
            "diagnostics": diag,
            "oil": None,
            "gas": None,
            "n_headlines": len(headlines),
            "headlines": headlines,
        }
    diag = dict(diag)
    diag["claude_model_used"] = scores.get("_model_used", "")
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "oil": scores.get("oil"),
        "gas": scores.get("gas"),
        "n_headlines": scores.get("n", len(headlines)),
        "headlines": headlines,
        "diagnostics": diag,
    }

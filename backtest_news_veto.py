"""News-veto backtest: does layering Claude sentiment over the ML
predictor's calibrated buckets improve risk-adjusted returns?

For each trading day in the past ~9 months:
  1. Compute the ML signal (prob_up) using the same model as live.
  2. Determine the variant-D trade plan (allocation, stop-loss).
  3. Fetch the prior 36h of energy news from Finnhub.
  4. Score with Claude Haiku -> oil_sentiment (-1 to +1).
  5. Apply news-veto rule: if proposing HOU and oil_sentiment < threshold,
     hold cash instead.
  6. Simulate next-day return as before.

Results saved to news_backtest_cache.json (resumable on failure).
Comparison printed at the end vs plain variant D.

Cost: ~$0.005 per day x ~180 days = ~$0.90 in Claude credits.
Runtime: ~10-15 minutes total.
"""
from __future__ import annotations

import os
import sys
import json
import time
import pathlib
from datetime import datetime, timezone, timedelta

# Force-load .env (system may have an empty ANTHROPIC_API_KEY blocking load_env)
for _line in open(".env").read().splitlines():
    _line = _line.strip()
    if _line and "=" in _line and not _line.startswith("#"):
        _k, _, _v = _line.partition("=")
        os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from paper_trader.predictor import _fetch_history, _build_feature_matrix
from paper_trader.news_sentiment import (
    OIL_KEYWORDS, GAS_KEYWORDS, MACRO_KEYWORDS,
    ENERGY_TICKERS_FOR_NEWS, CLAUDE_MODELS,
)

CACHE_PATH = pathlib.Path("news_backtest_cache.json")
FINNHUB_KEY = os.environ["FINNHUB_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]


def fetch_news_for_date(target_date: pd.Timestamp) -> list[dict]:
    """Fetch energy-related news from 36h before target_date through
    target_date open. Returns list of {headline, summary, source}."""
    # Window: 36 hours ending at target_date 13:30 UTC (~9:30am ET market open)
    end_dt = target_date.tz_localize(None).normalize() + timedelta(hours=13, minutes=30)
    start_dt = end_dt - timedelta(hours=36)
    from_str = start_dt.strftime("%Y-%m-%d")
    to_str = end_dt.strftime("%Y-%m-%d")

    items = []
    for sym in ENERGY_TICKERS_FOR_NEWS:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": sym, "from": from_str, "to": to_str,
                        "token": FINNHUB_KEY},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            for it in r.json() or []:
                ts = int(it.get("datetime") or 0)
                if start_dt.timestamp() <= ts <= end_dt.timestamp():
                    items.append({
                        "ts": ts,
                        "headline": (it.get("headline") or "").strip(),
                        "summary": (it.get("summary") or "").strip()[:300],
                        "source": it.get("source", ""),
                    })
        except Exception:
            continue

    # Dedup by headline
    seen = set()
    filtered = []
    all_kw = OIL_KEYWORDS | GAS_KEYWORDS | MACRO_KEYWORDS
    for it in items:
        h = it["headline"]
        if not h or h in seen:
            continue
        text = f"{h} {it['summary']}".lower()
        if not any(kw in text for kw in all_kw):
            # Keep ticker-specific items even without keyword match
            pass
        seen.add(h)
        filtered.append(it)
    # Newest first, cap at 30
    filtered.sort(key=lambda x: x["ts"], reverse=True)
    return filtered[:30]


def score_news(headlines: list[dict]) -> dict | None:
    """Single Claude call. Returns {oil, gas, reasoning} or None on failure."""
    if not headlines:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    formatted = "\n".join(
        f"- ({h.get('source','?')}) {h['headline']}"
        + (f" — {h['summary'][:200]}" if h.get('summary') else "")
        for h in headlines
    )
    prompt = (
        "You are an energy-markets analyst. Given the following recent "
        "news headlines (newest first), score the implications for "
        "NEXT-DAY price direction of:\n"
        "  1. WTI crude oil\n"
        "  2. Henry Hub natural gas\n\n"
        "Score from -1.0 (strongly bearish) to +1.0 (strongly bullish). "
        "Use 0.0 for neutral/mixed.\n\n"
        "Respond with ONLY a JSON object, no markdown:\n"
        '{"oil": {"score": <float>, "reasoning": "<1 sentence>"},\n'
        ' "gas": {"score": <float>, "reasoning": "<1 sentence>"}}\n\n'
        f"Headlines ({len(headlines)}):\n{formatted}"
    )
    for model in CLAUDE_MODELS:
        try:
            resp = client.messages.create(
                model=model, max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                inner = text.split("```")[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                text = inner.strip("` \n")
            # Last-ditch: pull first {...}
            if not text.startswith("{") and "{" in text:
                s = text.find("{"); e = text.rfind("}")
                text = text[s:e+1]
            return json.loads(text)
        except Exception as e:
            print(f"    Claude {model} failed: {type(e).__name__}: {e}", flush=True)
            continue
    return None


def main():
    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        print(f"Loaded {len(cache)} cached daily scores")

    # Build the ML model on pre-cutoff data (same as the dashboard)
    print("Fetching market data + training model…", flush=True)
    data = _fetch_history(years=12)
    df = _build_feature_matrix(data, "wti")
    feat_cols = [c for c in df.columns
                 if c not in ("px", "next_ret", "label")
                 and not c.endswith("_px")]
    if df.index.tz is not None:
        df = df.copy(); df.index = df.index.tz_localize(None)

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=9)
    train_mask = df.index < cutoff
    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, random_state=42,
    )
    clf.fit(df.loc[train_mask, feat_cols].values, df.loc[train_mask, "label"].values)
    print(f"Trained on {train_mask.sum()} samples, testing on {(df.index >= cutoff).sum()} days", flush=True)

    # Iterate test days, fetch + score news (with cache)
    test_dates = df.loc[df.index >= cutoff].index.tolist()
    print(f"Backtesting {len(test_dates)} trading days from {test_dates[0].date()} to {test_dates[-1].date()}", flush=True)

    new_calls = 0
    for i, date in enumerate(test_dates):
        key = date.strftime("%Y-%m-%d")
        # Only skip days with successful scores; retry transient failures
        # (claude_failed = credits ran out, fetch_failed = network blip)
        cached = cache.get(key, {})
        if "oil" in cached:
            continue
        if cached.get("error") == "no_news":
            continue  # genuinely no news — no point retrying
        try:
            heads = fetch_news_for_date(date)
        except Exception as e:
            print(f"  [{key}] fetch failed: {e}", flush=True)
            cache[key] = {"error": "fetch_failed", "n_heads": 0}
            continue
        if not heads:
            cache[key] = {"error": "no_news", "n_heads": 0}
            print(f"  [{i+1:>3}/{len(test_dates)}] {key}: no news", flush=True)
            continue
        scores = score_news(heads)
        if not scores:
            cache[key] = {"error": "claude_failed", "n_heads": len(heads)}
            print(f"  [{i+1:>3}/{len(test_dates)}] {key}: claude failed ({len(heads)} heads)", flush=True)
        else:
            try:
                oil = float(scores.get("oil", {}).get("score", 0))
                gas = float(scores.get("gas", {}).get("score", 0))
            except Exception:
                oil, gas = 0.0, 0.0
            cache[key] = {"oil": oil, "gas": gas, "n_heads": len(heads)}
            print(f"  [{i+1:>3}/{len(test_dates)}] {key}: oil={oil:+.2f} gas={gas:+.2f} (n={len(heads)})", flush=True)
        new_calls += 1
        # Save every 10 calls
        if new_calls % 10 == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
            print(f"    cache saved ({len(cache)} entries)", flush=True)
        # Light rate-limit pacing
        time.sleep(0.5)

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"Cache saved: {len(cache)} entries total ({new_calls} new)", flush=True)

    # ============ Simulate ============
    wti = data["wti"].copy()
    if wti.index.tz is not None: wti.index = wti.index.tz_localize(None)
    wti_close = wti["Close"]; wti_low = wti["Low"]; wti_high = wti["High"]
    wti_ret  = wti_close.pct_change().shift(-1)
    wti_drop = (wti_low.shift(-1)  - wti_close) / wti_close
    wti_rally = (wti_high.shift(-1) - wti_close) / wti_close

    LEVERAGE = 2.0
    TIER_ALLOC = {"HOU_very_strong":0.25, "HOU_strong":0.35, "HOU_weak":0.10, "HOU_fallback":0.05}
    TIER_STOP  = {"HOU_very_strong":-0.10, "HOU_strong":-0.06, "HOU_weak":-0.05, "HOU_fallback":-0.05}

    def run_sim(label, veto_threshold=None):
        equity, peak, mdd = 10000.0, 10000.0, 0.0
        trades, vetoed, stops, wins = 0, 0, 0, 0
        for date in test_dates:
            row = df.loc[date]
            x = row[feat_cols].values.reshape(1, -1)
            prob_up = float(clf.predict_proba(x)[0, 1])
            if prob_up >= 0.70:    pick='HOU'; tier='very_strong'
            elif 0.55<=prob_up<0.60: pick='HOU'; tier='strong'
            elif 0.60<=prob_up<0.70: pick='HOU'; tier='weak'
            elif prob_up>=0.50:    pick='HOU'; tier='fallback'
            else: continue  # skip HOD signals (variant D)
            tier_key = f'{pick}_{tier}'
            alloc = TIER_ALLOC.get(tier_key, 0.0)
            if alloc <= 0: continue

            # News veto
            if veto_threshold is not None:
                cached = cache.get(date.strftime("%Y-%m-%d"), {})
                oil_sent = cached.get("oil")
                if oil_sent is not None and oil_sent < veto_threshold:
                    vetoed += 1
                    continue

            try:
                r = wti_ret.asof(date); drop = wti_drop.asof(date); rally = wti_rally.asof(date)
            except Exception: continue
            if pd.isna(r): continue

            worst_intra = LEVERAGE * drop
            close_to_close = LEVERAGE * r
            stop = TIER_STOP[tier_key]
            etf_ret = stop if worst_intra <= stop else close_to_close
            if worst_intra <= stop: stops += 1

            port_ret = alloc * etf_ret
            equity *= (1 + port_ret)
            if port_ret > 0: wins += 1
            trades += 1
            if equity > peak: peak = equity
            dd = (equity - peak)/peak
            if dd < mdd: mdd = dd

        print(f"{label:<40}  end={equity:>10,.2f}  ret={(equity/10000-1)*100:+6.2f}%  DD={mdd*100:>5.2f}%  trades={trades}  vetoed={vetoed}  stops={stops}")

    print()
    print("=" * 90)
    print("RESULTS:")
    print("=" * 90)
    run_sim("Variant D (no news veto)")
    run_sim("Variant D + news veto (oil < -0.5)", veto_threshold=-0.5)
    run_sim("Variant D + news veto (oil < -0.3)", veto_threshold=-0.3)
    run_sim("Variant D + news veto (oil < -0.15)", veto_threshold=-0.15)
    run_sim("Variant D + news veto (oil < 0)",   veto_threshold=0.0)


if __name__ == "__main__":
    main()

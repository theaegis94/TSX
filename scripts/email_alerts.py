"""Daily email alerts runner.

Designed to run on a cron schedule (GitHub Actions). Scans the configured
universe and emails the user:
  1. Top tickers ranked by "upside potential" (CONVICTION + VOL_OUTLOOK +
     bullish news sentiment composite)
  2. Tickers matching any saved rule set tagged "alert-on" in
     `saved_rules.json` (a rule set is alert-tagged if its entry contains
     `"alert": true`; for backward compat, all sets in `enabled_rules` of
     `alerts_config.json` are also alerted)

Setup
-----
Required environment variables:
  SMTP_USER         — Gmail address (sender)
  SMTP_PASS         — Gmail App Password (NOT your account password)
  ALERT_TO          — Recipient email (defaults to SMTP_USER)
  FINNHUB_API_KEY   — for news/sentiment (optional but recommended)
  ANTHROPIC_API_KEY — unused here, but env consistency

Optional:
  SMTP_HOST         — defaults to smtp.gmail.com
  SMTP_PORT         — defaults to 587
  ALERTS_UNIVERSE   — defaults to "tsx_and_tsxv"
                      Options: tsx_full, tsxv, tsx_and_tsxv, tsx_composite,
                      tsx60, sp100, sp500, watchlist
  ALERTS_MIN_VOL    — minimum 20-day avg volume to qualify (default 50000;
                      filters out illiquid penny stocks on TSXV)
  ALERTS_TOP_N      — how many top potential picks to email (default 15)

Gmail App Password setup
------------------------
1. Enable 2FA on your Google account if not already
2. Go to https://myaccount.google.com/apppasswords
3. Select "Mail" → "Other (Custom name)" → name it "StockSignals"
4. Copy the 16-character password (no spaces) into SMTP_PASS
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensure UTF-8 stdout so emoji in logs don't crash on Windows (cp1252).
# GitHub Actions Ubuntu is already UTF-8; this is a no-op there.
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Add the parent directory to sys.path so we can import stock_signals
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_signals as ss  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
ALERT_TO = os.environ.get("ALERT_TO", SMTP_USER).strip()
ALERTS_UNIVERSE = os.environ.get("ALERTS_UNIVERSE", "tsx_and_tsxv").lower()
ALERTS_MIN_VOL = int(os.environ.get("ALERTS_MIN_VOL", "50000"))
ALERTS_TOP_N = int(os.environ.get("ALERTS_TOP_N", "15"))

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVED_RULES_PATH = REPO_ROOT / "saved_rules.json"
ALERTS_CONFIG_PATH = REPO_ROOT / "alerts_config.json"


# ---------------------------------------------------------------------------
# Universe resolution
# ---------------------------------------------------------------------------

def resolve_universe(name: str) -> list[str]:
    """Map a universe name to a ticker list."""
    name = (name or "").lower().strip()
    try:
        if name == "tsx_full":
            return ss.get_full_tsx_listing("tsx")
        if name == "tsxv":
            return ss.get_full_tsx_listing("tsxv")
        if name == "tsx_and_tsxv":
            return (ss.get_full_tsx_listing("tsx")
                    + ss.get_full_tsx_listing("tsxv"))
        if name == "tsx_composite":
            return ss.get_tsx_composite()
        if name == "tsx60":
            return list(ss.UNIVERSE_TSX60)
        if name == "sp100":
            return list(ss.UNIVERSE_SP100)
        if name == "sp500":
            return ss.get_sp500()
        if name == "watchlist":
            return list(ss.DEFAULT_WATCHLIST)
    except Exception as e:
        log(f"⚠️ Could not resolve universe '{name}': {e}")
    return list(ss.UNIVERSE_TSX60)


# ---------------------------------------------------------------------------
# Rule evaluation (mirrors app.py logic but standalone)
# ---------------------------------------------------------------------------

def get_indicator_value(df: pd.DataFrame, key: str,
                       ticker: str | None = None) -> float | None:
    """Latest-bar value of any RULE_INDICATOR (subset of app.py's
    _last_value, adapted to live without Streamlit cache)."""
    if df is None or df.empty:
        return None
    last = df.iloc[-1]

    if key in df.columns:
        v = last.get(key)
    elif key == "DAILY_CHG_PCT" and len(df) >= 2:
        prev = df.iloc[-2]["Close"]
        v = (last["Close"] - prev) / prev * 100 if prev else None
    elif key == "DIST_SMA5_PCT" and "SMA5" in df.columns:
        v = ((last["Close"] - last["SMA5"]) / last["SMA5"] * 100
             if last["SMA5"] else None)
    elif key == "DIST_SMA20_PCT" and "SMA20" in df.columns:
        v = ((last["Close"] - last["SMA20"]) / last["SMA20"] * 100
             if last["SMA20"] else None)
    elif key == "DIST_SMA50_PCT" and "SMA50" in df.columns:
        v = ((last["Close"] - last["SMA50"]) / last["SMA50"] * 100
             if last["SMA50"] else None)
    elif key == "DIST_SMA200_PCT" and "SMA200" in df.columns:
        v = ((last["Close"] - last["SMA200"]) / last["SMA200"] * 100
             if last["SMA200"] else None)
    elif key == "BB_PCT_B" and {"BB_LOWER", "BB_UPPER"}.issubset(df.columns):
        rng = last["BB_UPPER"] - last["BB_LOWER"]
        v = (last["Close"] - last["BB_LOWER"]) / rng if rng else None
    elif key == "BB_DIST_LOWER_PCT" and "BB_LOWER" in df.columns:
        v = (((last["Close"] - last["BB_LOWER"]) / last["BB_LOWER"] * 100)
             if last["BB_LOWER"] else None)
    elif key == "BB_DIST_UPPER_PCT" and "BB_UPPER" in df.columns:
        v = (((last["BB_UPPER"] - last["Close"]) / last["BB_UPPER"] * 100)
             if last["BB_UPPER"] else None)
    elif key == "BB_BANDWIDTH_PCT" and {
            "BB_LOWER", "BB_UPPER", "BB_MID"}.issubset(df.columns):
        v = (((last["BB_UPPER"] - last["BB_LOWER"]) / last["BB_MID"] * 100)
             if last["BB_MID"] else None)
    elif key in ("ANOMALY_SCORE", "ANOMALY_PCTILE"):
        result = ss.compute_anomaly_score(df)
        if not result:
            return None
        v = result["score"] if key == "ANOMALY_SCORE" else result["pctile"]
    elif key == "VOL_OUTLOOK" and ticker:
        out = ss.compute_volume_outlook(ticker, df)
        if not out:
            return None
        v = out.get("score")
    elif key in ("NEWS_SENT", "NEWS_BUZZ") and ticker:
        sent = ss.finnhub_sentiment(ticker)
        if not sent:
            return None
        if key == "NEWS_SENT":
            v = (sent.get("sentiment") or {}).get("bullishPercent")
        else:
            v = (sent.get("buzz") or {}).get("buzz")
    elif key in ("ST_BULLISH", "ST_BUZZ") and ticker:
        st_data = ss.stocktwits_sentiment(ticker)
        if not st_data:
            return None
        v = (st_data.get("bullish_pct") if key == "ST_BULLISH"
             else st_data.get("msg_count_24h"))
    else:
        return None
    try:
        f = float(v) if v is not None else None
        return f if f is not None and f == f else None
    except (TypeError, ValueError):
        return None


def eval_rule(df, rule: dict, ticker: str | None = None) -> bool | None:
    left = get_indicator_value(df, rule.get("left", ""), ticker=ticker)
    if left is None:
        return None
    op = rule.get("op")
    a = rule.get("a")
    b = rule.get("b")
    if a is None:
        return None
    if op == "<":      return left < a
    if op == "<=":     return left <= a
    if op == ">":      return left > a
    if op == ">=":     return left >= a
    if op == "between" and b is not None:
        lo, hi = (a, b) if a <= b else (b, a)
        return lo <= left <= hi
    return None


# ---------------------------------------------------------------------------
# Upside-potential scoring
# ---------------------------------------------------------------------------

def upside_score_fast(df: pd.DataFrame, ticker: str) -> dict | None:
    """FAST pass: cheap composite (no API calls, no anomaly training).
    Only uses indicators already in `df`. Used to filter to top candidates
    before expensive enrichment runs.
    """
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    score = 0.0
    parts: dict = {}

    # Conviction (already computed in compute_indicators)
    if "CONVICTION" in df.columns:
        c = float(last.get("CONVICTION", 0) or 0)
        parts["conviction"] = round(c, 1)
        score += c

    # SMA200 trend bonus
    if "SMA200" in df.columns and pd.notna(last.get("SMA200")):
        in_uptrend = float(last["Close"]) > float(last["SMA200"])
        score += 10 if in_uptrend else -10
        parts["uptrend"] = in_uptrend

    # MFI/CMF (volume confirmation) — cheap, in df
    if "CMF" in df.columns and pd.notna(last.get("CMF")):
        c = float(last["CMF"])
        score += c * 30  # CMF is -1 to +1; multiply for impact
        parts["cmf"] = round(c, 3)

    return {
        "ticker": ticker,
        "score": round(score, 1),
        "parts": parts,
        "close": float(last["Close"]) if pd.notna(last["Close"]) else None,
        "rsi": (round(float(last["RSI"]), 1)
                if "RSI" in df.columns and pd.notna(last["RSI"]) else None),
    }


def enrich_score(base: dict, df: pd.DataFrame, ticker: str) -> dict:
    """SLOW pass: add VOL_OUTLOOK + news sentiment to a fast-pass result.
    Only run for top candidates (cuts API calls 95%+ for big universes)."""
    parts = dict(base.get("parts") or {})
    score = base.get("score", 0.0)

    # Volume outlook (hits Finnhub + yfinance for earnings)
    try:
        vo = ss.compute_volume_outlook(ticker, df)
        if vo:
            score += vo.get("score", 0) * 0.5
            parts["vol_outlook"] = vo.get("score")
    except Exception:
        pass

    # News sentiment (Finnhub)
    try:
        sent = ss.finnhub_sentiment(ticker)
        if sent:
            bull = (sent.get("sentiment") or {}).get("bullishPercent")
            if bull is not None:
                score += (float(bull) - 0.5) * 20
                parts["news_sent"] = round(float(bull), 2)
    except Exception:
        pass

    return {**base, "score": round(score, 1), "parts": parts}


# ---------------------------------------------------------------------------
# Ticker fetching (with caching to avoid re-downloading within one run)
# ---------------------------------------------------------------------------

_DF_CACHE: dict[str, pd.DataFrame | None] = {}


def fetch_with_indicators(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    """Download + compute indicators for one ticker. Cached per-process."""
    if ticker in _DF_CACHE:
        return _DF_CACHE[ticker]
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        if df is None or df.empty or len(df) < 50:
            _DF_CACHE[ticker] = None
            return None
        # Liquidity filter: skip illiquid tickers
        if "Volume" in df.columns:
            avg_vol = float(df["Volume"].tail(20).mean())
            if avg_vol < ALERTS_MIN_VOL:
                _DF_CACHE[ticker] = None
                return None
        df = ss.compute_indicators(df)
        _DF_CACHE[ticker] = df
        return df
    except Exception:
        _DF_CACHE[ticker] = None
        return None


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def load_saved_rules() -> dict:
    if not SAVED_RULES_PATH.exists():
        return {}
    try:
        data = json.loads(SAVED_RULES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_alerts_config() -> dict:
    if not ALERTS_CONFIG_PATH.exists():
        return {"enabled_rules": [], "universe": ALERTS_UNIVERSE}
    try:
        return json.loads(ALERTS_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled_rules": [], "universe": ALERTS_UNIVERSE}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_scan() -> dict:
    """Returns dict with: top_picks (list), rule_matches (dict), universe_name,
    scanned (int), filtered (int)."""
    universe = resolve_universe(ALERTS_UNIVERSE)
    log(f"Universe '{ALERTS_UNIVERSE}': {len(universe)} tickers raw")

    saved = load_saved_rules()
    config = load_alerts_config()
    enabled_rule_names = set(config.get("enabled_rules", []))
    # Also allow inline "alert": true flag in saved_rules
    for name, rule_set in saved.items():
        if isinstance(rule_set, dict) and rule_set.get("alert"):
            enabled_rule_names.add(name)

    candidates: list[dict] = []  # fast-pass results (no API calls)
    rule_matches: dict[str, list[str]] = {n: [] for n in enabled_rule_names}
    scanned = 0
    filtered = 0

    # === PASS 1: cheap scoring + rule matching across full universe ===
    log("Pass 1: fast scoring (no API calls)…")
    for i, t in enumerate(universe):
        if i % 200 == 0:
            log(f"  Pass 1 progress {i}/{len(universe)}…")
        try:
            t_norm = ss.normalize_ticker(t)
        except SystemExit:
            continue
        df = fetch_with_indicators(t_norm)
        if df is None:
            filtered += 1
            continue
        scanned += 1

        s = upside_score_fast(df, t_norm)
        if s and s["score"] > 0:
            candidates.append((s, t_norm))

        for rname in enabled_rule_names:
            rules = saved.get(rname)
            if isinstance(rules, dict):
                rules = rules.get("rules", [])
            if not rules:
                continue
            results = [eval_rule(df, r, ticker=t_norm) for r in rules]
            if results and all(r is True for r in results):
                rule_matches[rname].append(t_norm)

    # Sort by fast-pass score, take top 2× target for enrichment
    candidates.sort(key=lambda x: x[0]["score"], reverse=True)
    n_to_enrich = min(ALERTS_TOP_N * 2, len(candidates))
    log(f"Pass 2: enriching top {n_to_enrich} candidates with news + "
        "volume-outlook…")

    # === PASS 2: enrich top candidates with API-backed signals ===
    top_picks: list[dict] = []
    for i, (base, t_norm) in enumerate(candidates[:n_to_enrich]):
        if i % 10 == 0:
            log(f"  Pass 2 progress {i}/{n_to_enrich}…")
        df = _DF_CACHE.get(t_norm)  # already cached from pass 1
        if df is None:
            continue
        top_picks.append(enrich_score(base, df, t_norm))

    top_picks.sort(key=lambda x: x["score"], reverse=True)
    log(f"Done. Scanned {scanned}, filtered {filtered}, "
        f"top_picks {len(top_picks)}, rule matches "
        f"{sum(len(v) for v in rule_matches.values())}")
    return {
        "top_picks": top_picks[:ALERTS_TOP_N],
        "rule_matches": rule_matches,
        "universe_name": ALERTS_UNIVERSE,
        "scanned": scanned,
        "filtered": filtered,
    }


# ---------------------------------------------------------------------------
# Email composition
# ---------------------------------------------------------------------------

def render_email_html(result: dict) -> str:
    """Build the HTML email body."""
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    top = result["top_picks"]
    rules = result["rule_matches"]

    style_header = (
        'style="font-family: -apple-system, BlinkMacSystemFont, '
        '\'Segoe UI\', sans-serif; background:#1e1e1e; color:#e5e7eb;"'
    )
    style_card = (
        'style="background:#2a2b2e; padding:14px 18px; border-radius:10px; '
        'margin:12px 0; border-left:4px solid #60a5fa;"'
    )

    # Top picks table
    if top:
        rows_html = ""
        for i, p in enumerate(top, 1):
            score_color = ("#22c55e" if p["score"] > 60
                           else "#fbbf24" if p["score"] > 30 else "#9ca3af")
            parts_str = " · ".join(
                f"{k}: {v}" for k, v in (p.get("parts") or {}).items()
            )
            rows_html += (
                f"<tr><td style='padding:8px 12px;'>{i}</td>"
                f"<td style='padding:8px 12px; font-weight:700;'>"
                f"<a href='https://finance.yahoo.com/quote/{p['ticker']}' "
                f"style='color:#60a5fa; text-decoration:none;'>"
                f"{p['ticker']}</a></td>"
                f"<td style='padding:8px 12px; color:{score_color}; "
                f"font-weight:700;'>{p['score']:+.1f}</td>"
                f"<td style='padding:8px 12px;'>"
                f"${p['close']:.2f}</td>"
                f"<td style='padding:8px 12px;'>"
                f"{p['rsi'] if p['rsi'] is not None else '—'}</td>"
                f"<td style='padding:8px 12px; font-size:11px; "
                f"color:#9ca3af;'>{parts_str}</td></tr>"
            )
        top_html = (
            f"<div {style_card}>"
            "<h3 style='margin:0 0 8px; color:#60a5fa;'>"
            f"🎯 Top {len(top)} Upside Candidates</h3>"
            "<p style='margin:0 0 10px; color:#9ca3af; font-size:13px;'>"
            "Ranked by composite score (CONVICTION + VOL_OUTLOOK + "
            "trend + news sentiment). Higher = stronger setup. "
            "<b>NOT a prediction</b> — these are tickers where multiple "
            "indicators currently align bullishly.</p>"
            "<table style='width:100%; border-collapse:collapse; "
            "font-size:13px;'><thead><tr style='background:#3a3b3e;'>"
            "<th style='padding:8px 12px; text-align:left;'>#</th>"
            "<th style='padding:8px 12px; text-align:left;'>Ticker</th>"
            "<th style='padding:8px 12px; text-align:left;'>Score</th>"
            "<th style='padding:8px 12px; text-align:left;'>Close</th>"
            "<th style='padding:8px 12px; text-align:left;'>RSI</th>"
            "<th style='padding:8px 12px; text-align:left;'>Components</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
        )
    else:
        top_html = (
            f"<div {style_card}><h3 style='margin:0; color:#9ca3af;'>"
            "No high-conviction setups today</h3>"
            "<p style='margin:6px 0 0; color:#9ca3af; font-size:13px;'>"
            "No tickers had a positive composite score. Often happens "
            "in broad downturns or low-volatility days.</p></div>"
        )

    # Rule matches
    rules_html = ""
    if rules:
        for rname, tickers in rules.items():
            if not tickers:
                continue
            tickers_html = ", ".join(
                f"<a href='https://finance.yahoo.com/quote/{t}' "
                f"style='color:#22c55e; text-decoration:none; "
                f"font-weight:700;'>{t}</a>"
                for t in tickers[:50]
            )
            extra = (f" <span style='color:#9ca3af;'>+{len(tickers)-50} more"
                     f"</span>" if len(tickers) > 50 else "")
            rules_html += (
                f"<div {style_card.replace('#60a5fa','#22c55e')}>"
                f"<h3 style='margin:0 0 8px; color:#22c55e;'>"
                f"🎯 {rname} — {len(tickers)} matches</h3>"
                f"<div style='font-size:14px; line-height:1.8;'>"
                f"{tickers_html}{extra}</div></div>"
            )

    if not rules_html and any(rules.values()):
        rules_html = ""
    if not rules_html and rules:
        rules_html = (
            f"<div {style_card.replace('#60a5fa','#9ca3af')}>"
            "<p style='margin:0; color:#9ca3af;'>"
            "No matches for your alert-enabled rule sets today.</p></div>"
        )

    body = f"""<html><body {style_header}>
<div style="max-width:800px; margin:0 auto; padding:20px;">
  <h1 style="color:#60a5fa; margin:0;">📊 Daily Stock Alerts</h1>
  <p style="color:#9ca3af; margin:4px 0 20px;">{date_str} · scanned
    {result['scanned']} tickers in {result['universe_name']}
    ({result['filtered']} filtered for low liquidity)</p>
  {top_html}
  {rules_html}
  <hr style="border:none; border-top:1px solid #4a4b4e; margin:24px 0;">
  <p style="color:#9ca3af; font-size:11px; line-height:1.5;">
    ⚠️ Not investment advice. Indicators measure historical patterns,
    not future prices. Multi-factor "high conviction" scores have ~55-60%
    historical hit rate at best. Do your own analysis.
  </p>
</div></body></html>"""
    return body


def send_email(html_body: str) -> None:
    if not (SMTP_USER and SMTP_PASS):
        log("❌ SMTP_USER or SMTP_PASS missing — skipping send")
        return
    if not ALERT_TO:
        log("❌ ALERT_TO missing — skipping send")
        return

    date_str = datetime.now().strftime("%b %d")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📊 Stock Signals — {date_str}"
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_TO
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, ALERT_TO.split(","), msg.as_string())
        log(f"✅ Email sent to {ALERT_TO}")
    except Exception as e:
        log(f"❌ Email send failed: {e}")
        traceback.print_exc()


def main() -> int:
    log("=== Daily Alerts Run ===")
    log(f"Universe: {ALERTS_UNIVERSE}, MinVol: {ALERTS_MIN_VOL}, "
        f"TopN: {ALERTS_TOP_N}")
    try:
        result = run_scan()
        html = render_email_html(result)
        send_email(html)
        return 0
    except Exception as e:
        log(f"❌ Run failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

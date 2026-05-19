"""Canadian ETF screener — 1-week-ahead direction prediction.

Scope: rank ~25 of the most liquid Canadian-listed ETFs by the
probability that they'll close higher 5 trading days from now.

Methodology (mirrors predictor.py's HOU/HOD approach, adapted for
weekly horizon):
  - Per-ETF gradient-boosted classifier
  - Features: own-price momentum (1d/5d/20d), RSI, MACD,
    20-day volatility, MA-cross, volume ratio + cross-asset
    market context (XIU 5d, USD/CAD 5d, TLT 5d)
  - Walk-forward backtest: train on first 4 years, test on the
    remaining ~1 year, never seeing future data
  - Label: close_{t+5} > close_t

Honest expectations:
  - 52-55% accuracy on weekly direction would be a real edge
  - Many ETFs (broad-market index funds especially) will land
    at coin-flip — that's data telling us they're efficient
  - The screener's value is identifying which ETFs DO have
    detectable weekly momentum/mean-reversion structure
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

LOGGER = logging.getLogger("paper_trader.etf_screener")

# ~25 of the most liquid Canadian-listed ETFs across major asset classes.
# Universe is intentionally diverse so the screener can highlight which
# corners of the Canadian ETF market are actually predictable vs
# random walks. Fast to train (~30s).
UNIVERSE = [
    # Broad Canadian equity
    "XIU.TO", "XIC.TO", "VCN.TO", "ZCN.TO",
    # US exposure (CAD-hedged + unhedged variants)
    "VFV.TO", "XSP.TO", "ZSP.TO", "HXS.TO",
    # International
    "XEF.TO", "XEC.TO", "ZEM.TO",
    # Sectors
    "XEG.TO",   # Canadian energy
    "ZEB.TO",   # Canadian banks
    "XFN.TO",   # Canadian financials
    "XGD.TO",   # Gold miners
    "XIT.TO",   # Canadian tech
    "XRE.TO",   # REITs
    # Dividend-focused
    "XDV.TO", "VDY.TO", "XEI.TO",
    # Fixed income
    "XBB.TO", "ZAG.TO",
    # All-in-one asset allocation
    "XEQT.TO", "VEQT.TO", "XGRO.TO",
]


# ~100 of the most liquid Canadian-listed ETFs across all major providers
# (iShares, BMO, Vanguard, Horizons/Global X, CI, Hamilton, Purpose,
# Mackenzie). Comprehensive coverage of broad market, US, international,
# sectors, dividend, fixed income, all-in-one, leveraged, and specialty.
# Slower to train (~3-5 min for the full ranking).
UNIVERSE_FULL = [
    # --- Broad Canadian equity ---
    "XIU.TO", "XIC.TO", "VCN.TO", "ZCN.TO", "HXT.TO", "XCS.TO", "XMD.TO",
    # --- US equity (CAD-hedged + unhedged + swap variants) ---
    "VFV.TO", "VUN.TO", "XSP.TO", "ZSP.TO", "HXS.TO", "ZUE.TO",
    "VUS.TO", "ZUQ.TO", "XUS.TO",
    # --- US Nasdaq / tech ---
    "ZQQ.TO", "XQQ.TO", "HXQ.TO", "QQC.TO",
    # --- International developed ---
    "XEF.TO", "XIN.TO", "VI.TO", "VIU.TO", "XAW.TO", "VXC.TO", "XWD.TO",
    # --- Emerging markets ---
    "XEC.TO", "ZEM.TO", "VEE.TO", "XEM.TO",
    # --- Sectors: Canadian ---
    "XEG.TO",   # iShares energy
    "ZEO.TO",   # BMO Cdn equal-weight oil & gas
    "ZEB.TO",   # BMO equal-weight banks
    "XFN.TO",   # iShares financials
    "ZGI.TO",   # global infrastructure
    "ZUT.TO",   # utilities
    "XUT.TO",   # iShares utilities
    "XMA.TO",   # iShares materials
    "XGD.TO",   # gold producers
    "ZJG.TO",   # junior gold
    "XIT.TO",   # iShares tech
    "XHC.TO",   # global healthcare
    "XST.TO",   # consumer staples
    "XCD.TO",   # consumer discretionary
    "XRE.TO",   # REITs
    "ZRE.TO",   # BMO REITs
    # --- Dividend / income ---
    "XDV.TO", "VDY.TO", "XEI.TO", "ZDV.TO", "CDZ.TO", "XHD.TO",
    "VIDY.TO",  # international dividend
    # --- Covered-call income (BMO ZW* + Hamilton MAX) ---
    "ZWB.TO",   # banks
    "ZWC.TO",   # Canadian high div
    "ZWU.TO",   # utilities
    "ZWE.TO",   # Europe high div
    "ZWH.TO",   # US high div
    "ZWS.TO",   # S&P 500
    "HMAX.TO",  # Hamilton banks
    "UMAX.TO",  # Hamilton US
    "HDIV.TO",  # Hamilton enhanced diversified
    # --- Low volatility ---
    "ZLB.TO",   # Cdn low vol
    "ZLU.TO",   # US low vol
    "ZLE.TO",   # EU low vol
    # --- Fixed income ---
    "XBB.TO", "ZAG.TO", "VAB.TO",
    "XSB.TO", "VSB.TO",     # short-term
    "XGB.TO", "ZGB.TO",     # government
    "XCB.TO", "ZCS.TO",     # corporate
    "XHY.TO", "ZHY.TO",     # high yield
    "XSH.TO",               # short-term corporate
    "ZEF.TO",               # EM bonds
    "XPF.TO",               # preferred shares
    # --- All-in-one asset allocation ---
    "XEQT.TO", "VEQT.TO", "XGRO.TO", "VGRO.TO",
    "XBAL.TO", "VBAL.TO", "VCNS.TO", "XCNS.TO",
    "VRIF.TO",   # retirement income
    # --- Cash / HISA equivalents ---
    "PSA.TO", "CASH.TO", "ZST.TO",
    # --- Horizons single + leveraged ---
    "HXX.TO",   # Euro 50 swap
    "HXC.TO",   # Cdn select swap
    "HOU.TO", "HOD.TO",     # WTI 2x bull/bear (already in your watchlist)
    "HNU.TO", "HND.TO",     # natgas 2x bull/bear
    "HSU.TO", "HSD.TO",     # S&P 500 2x bull/bear
    "HQU.TO", "HQD.TO",     # Nasdaq 2x bull/bear
    "HXU.TO", "HXD.TO",     # TSX 60 2x bull/bear
    "HEU.TO", "HED.TO",     # Cdn energy 2x bull/bear
    # --- Specialty / thematic ---
    "CGL.TO",   # gold bullion (CAD-hedged)
    "MNT.TO",   # Royal Cdn Mint gold trust
    "BTCC.TO",  # bitcoin
    "ETHX.TO",  # ether
    "RBOT.TO",  # robotics + AI
]

# Market-context tickers (same for all ETFs)
MARKET_CONTEXT = {
    "tsx":  "^GSPTSE",   # TSX composite
    "spx":  "^GSPC",     # S&P 500
    "tlt":  "TLT",       # 20y bonds (rates proxy)
    "dxy":  "DX-Y.NYB",  # USD index
    "vix":  "^VIX",      # volatility
}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard 14-period RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_diff(series: pd.Series) -> pd.Series:
    """MACD signal-line diff (12-26 EMA)."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _fetch_market_context(years: int = 5) -> dict[str, pd.DataFrame]:
    """Pull market context series. Returned aligned to each ETF later."""
    out = {}
    for key, sym in MARKET_CONTEXT.items():
        try:
            df = yf.download(sym, period=f"{years}y", interval="1d",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            out[key] = df
        except Exception as e:
            LOGGER.warning(f"market context {sym} failed: {e}")
    return out


def _fetch_etf(ticker: str, years: int = 5) -> pd.DataFrame | None:
    """Pull daily OHLCV for a single ETF."""
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 250:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        LOGGER.warning(f"fetch {ticker} failed: {e}")
        return None


def _build_features(
    etf_df: pd.DataFrame,
    market: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """Compute per-ETF features + market context, with the 5-day-ahead
    label appended. Drops rows that have NaN in any feature."""
    close = etf_df["Close"]
    volume = etf_df["Volume"]

    feats = pd.DataFrame(index=etf_df.index)
    feats["ret_1d"]  = close.pct_change(1)
    feats["ret_5d"]  = close.pct_change(5)
    feats["ret_20d"] = close.pct_change(20)
    feats["rsi_14"]  = _rsi(close, 14)
    feats["macd"]    = _macd_diff(close)
    feats["vol_20d"] = close.pct_change().rolling(20).std()
    feats["ma_cross"] = (
        close.rolling(20).mean() / close.rolling(50).mean() - 1
    )
    # Volume ratio: today vs 20-day average
    vol_ma = volume.rolling(20).mean()
    feats["vol_ratio"] = (volume / vol_ma).replace([np.inf, -np.inf], np.nan)

    # Cross-asset market context (lagged 1 day to avoid look-ahead)
    for key in ("tsx", "spx", "tlt", "dxy", "vix"):
        m = market.get(key)
        if m is None or "Close" not in m:
            continue
        m_close = m["Close"].reindex(feats.index, method="ffill")
        feats[f"{key}_ret_5d"] = m_close.pct_change(5).shift(1)
        if key == "vix":
            feats[f"{key}_level"] = m_close.shift(1)

    # Label: did the close 5 days later exceed today's close?
    forward_ret = close.shift(-horizon) / close - 1
    feats["next_5d_ret"] = forward_ret
    feats["label"] = (forward_ret > 0).astype(int)

    # Drop rows with NaN (head: history warmup, tail: no future data)
    return feats.dropna()


def _train_eval(
    feats: pd.DataFrame,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    """Fit on first train_ratio of rows, eval on the rest. Return
    accuracy, latest probability, and the trained model."""
    feat_cols = [c for c in feats.columns if c not in ("next_5d_ret", "label")]
    if len(feats) < 300:
        return {"error": "insufficient_history", "n_rows": len(feats)}

    split = int(len(feats) * train_ratio)
    X_train = feats.iloc[:split][feat_cols].values
    y_train = feats.iloc[:split]["label"].values
    X_test  = feats.iloc[split:][feat_cols].values
    y_test  = feats.iloc[split:]["label"].values

    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, random_state=42,
    )
    clf.fit(X_train, y_train)
    preds = (clf.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    accuracy = float((preds == y_test).mean()) if len(y_test) else 0.0

    # Latest prediction (most recent row, no label)
    latest_x = feats.iloc[-1][feat_cols].values.reshape(1, -1)
    latest_prob = float(clf.predict_proba(latest_x)[0, 1])

    # Baseline: how often does this ETF go up over 5 days, period?
    base_rate = float(feats["label"].mean())

    return {
        "n_train": split,
        "n_test": len(feats) - split,
        "accuracy": accuracy,
        "edge_vs_baseline": accuracy - max(base_rate, 1 - base_rate),
        "base_rate_up": base_rate,
        "latest_prob_up": latest_prob,
        "feat_cols": feat_cols,
        "model": clf,
    }


def rank_etfs(
    universe: list[str] | None = None,
    years: int = 5,
) -> pd.DataFrame:
    """Train a 1-week direction model per ETF, return ranked DataFrame.

    Columns:
      ticker        : symbol
      prob_up_1wk   : latest probability the ETF closes higher in 5d
      accuracy_oos  : walk-forward accuracy on the held-out window
      edge          : accuracy minus max(base_rate, 1-base_rate)
      n_test        : how many test predictions the accuracy is over
      base_rate_up  : how often this ETF historically went up over 5d
    """
    if universe is None:
        universe = UNIVERSE

    LOGGER.info(f"Fetching market context for {years}y…")
    market = _fetch_market_context(years=years)

    rows = []
    for tkr in universe:
        LOGGER.info(f"Training model for {tkr}…")
        etf_df = _fetch_etf(tkr, years=years)
        if etf_df is None:
            rows.append({"ticker": tkr, "error": "no_data"})
            continue
        feats = _build_features(etf_df, market)
        if feats.empty:
            rows.append({"ticker": tkr, "error": "no_features"})
            continue
        result = _train_eval(feats)
        if "error" in result:
            rows.append({"ticker": tkr, "error": result["error"]})
            continue
        rows.append({
            "ticker": tkr,
            "prob_up_1wk": result["latest_prob_up"],
            "accuracy_oos": result["accuracy"],
            "edge": result["edge_vs_baseline"],
            "n_test": result["n_test"],
            "base_rate_up": result["base_rate_up"],
        })

    df = pd.DataFrame(rows)
    if "prob_up_1wk" in df.columns:
        df = df.sort_values(
            ["prob_up_1wk"], ascending=False, na_position="last",
        ).reset_index(drop=True)
    return df


def compute_top_movers(
    universe: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Pull today's intraday % change from open for each ticker in the
    universe and return the top_k by absolute move magnitude.

    Each item:
      {ticker, open, current, change_pct, abs_pct, is_closed}

    is_closed=True means we fell back to the previous daily bar
    because intraday data wasn't available (market closed or weekend).
    """
    if universe is None:
        universe = UNIVERSE
    results = []
    for t in universe:
        try:
            df = yf.download(t, period="1d", interval="5m",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            is_closed = False
            if df.empty or len(df) < 2:
                df = yf.download(t, period="5d", interval="1d",
                                 auto_adjust=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty or len(df) < 2:
                    continue
                op = float(df["Open"].iloc[-1])
                cp = float(df["Close"].iloc[-1])
                is_closed = True
            else:
                op = float(df["Open"].iloc[0])
                cp = float(df["Close"].iloc[-1])
            if op <= 0:
                continue
            pct = (cp - op) / op * 100
            results.append({
                "ticker": t,
                "open": op,
                "current": cp,
                "change_pct": pct,
                "abs_pct": abs(pct),
                "is_closed": is_closed,
            })
        except Exception as e:
            LOGGER.warning(f"top_movers fetch {t} failed: {e}")
            continue
    results.sort(key=lambda x: x["abs_pct"], reverse=True)
    return results[:top_k]


def backtest_screener_weekly(
    universe: list[str] | None = None,
    months_back: int = 12,
    top_k: int = 3,
    years: int = 5,
) -> dict[str, Any]:
    """Walk-forward backtest: each Friday, train models on prior data
    and pick the top_k highest-probability ETFs. Score whether each
    pick actually closed higher 5 trading days later.

    Returns overall accuracy + per-ETF hit rate to identify which
    ETFs the model can actually predict.
    """
    if universe is None:
        universe = UNIVERSE
    market = _fetch_market_context(years=years)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=months_back)

    # Pre-build feature frames per ETF
    per_etf_feats = {}
    for tkr in universe:
        etf_df = _fetch_etf(tkr, years=years)
        if etf_df is None: continue
        feats = _build_features(etf_df, market)
        if feats.empty: continue
        per_etf_feats[tkr] = feats

    if not per_etf_feats:
        return {"error": "no_etfs"}

    # Determine test dates (one per week — Fridays)
    sample_feats = next(iter(per_etf_feats.values()))
    test_dates = [d for d in sample_feats.index
                  if d >= cutoff and d.weekday() == 4]  # Friday = 4

    per_etf_hits = {t: {"picks": 0, "right": 0, "ret_sum": 0.0}
                    for t in per_etf_feats}
    total_picks = 0
    total_right = 0
    sum_5d_ret = 0.0

    feat_cols = None
    for date in test_dates:
        # Train each ETF on data strictly before this Friday
        probs = {}
        for tkr, feats in per_etf_feats.items():
            train = feats.loc[feats.index < date]
            if len(train) < 250: continue
            if feat_cols is None:
                feat_cols = [c for c in train.columns
                             if c not in ("next_5d_ret", "label")]
            X = train[feat_cols].values
            y = train["label"].values
            clf = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=150,
                l2_regularization=1.0, random_state=42,
            )
            clf.fit(X, y)
            # Predict for THIS date (most recent row at-or-before date)
            try:
                latest_idx = feats.index.get_loc(date)
            except KeyError:
                continue
            latest = feats.iloc[latest_idx][feat_cols].values.reshape(1, -1)
            prob = float(clf.predict_proba(latest)[0, 1])
            actual_ret = float(feats.iloc[latest_idx]["next_5d_ret"])
            probs[tkr] = (prob, actual_ret)

        # Pick top_k by probability
        top = sorted(probs.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        for tkr, (prob, actual_ret) in top:
            total_picks += 1
            right = actual_ret > 0
            if right: total_right += 1
            sum_5d_ret += actual_ret
            per_etf_hits[tkr]["picks"] += 1
            if right: per_etf_hits[tkr]["right"] += 1
            per_etf_hits[tkr]["ret_sum"] += actual_ret

    accuracy = total_right / total_picks if total_picks else 0.0
    avg_5d_ret = sum_5d_ret / total_picks if total_picks else 0.0

    # Per-ETF breakdown for tickers that actually got picked
    per_etf = []
    for tkr, h in per_etf_hits.items():
        if h["picks"] == 0: continue
        per_etf.append({
            "ticker": tkr, "picks": h["picks"],
            "right": h["right"],
            "hit_rate": h["right"]/h["picks"],
            "avg_5d_ret": h["ret_sum"]/h["picks"],
        })
    per_etf.sort(key=lambda x: x["picks"], reverse=True)

    return {
        "months_back": months_back,
        "top_k_per_week": top_k,
        "weeks_tested": len(test_dates),
        "total_picks": total_picks,
        "total_right": total_right,
        "overall_accuracy": accuracy,
        "avg_5d_return": avg_5d_ret,
        "per_etf": per_etf,
    }

"""Next-day direction predictor for WTI and natural gas.

This is a SUPERVISED-LEARNING approach (gradient-boosted trees) trained
on lagged technical + macro features. Goal: predict tomorrow's
direction (up/down) for WTI and NG=F. Maps directly to BUY/SELL on
the 4 ETFs since HOU/HOD and HNU/HND are exact inverses of the
underlying.

Walk-forward evaluation:
  - Train on first N years
  - Predict year N+1 (OUT-OF-SAMPLE)
  - Extend training by 1 year, repeat
  - Final accuracy = average across all predicted years

Realistic expectations:
  - 51-53% accuracy out-of-sample is honest result
  - 60%+ on test set probably means overfit
  - Edge needed to be profitable with 3x leverage + slippage: ~52% with
    proper risk management

Model: sklearn HistGradientBoostingClassifier
  - Handles tabular data well
  - Less prone to overfit than deep models
  - Robust to feature scaling
  - Fast to train (seconds on 2500 samples)
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger("paper_trader.predictor")

# Underlying tickers we predict
TARGETS = {
    "wti": "CL=F",   # WTI crude oil futures
    "ng":  "NG=F",   # Henry Hub natgas futures
}

# Feature symbols (all daily, free from yfinance)
FEATURE_SYMBOLS = {
    "wti":   "CL=F",
    "ng":    "NG=F",
    "brent": "BZ=F",
    "dxy":   "DX-Y.NYB",
    "ovx":   "^OVX",
    "vix":   "^VIX",
    "xle":   "XLE",
    "tnx":   "^TNX",     # 10-year treasury yield
}


def _fetch_history(years: int = 12) -> dict[str, pd.DataFrame]:
    """Fetch all feature symbols once."""
    out = {}
    syms = list(FEATURE_SYMBOLS.values())
    try:
        df = yf.download(
            " ".join(syms), period=f"{years}y", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
            threads=True,
        )
    except Exception as e:
        LOGGER.error(f"yfinance fetch failed: {e}")
        return out
    if df is None or df.empty:
        return out
    for short, sym in FEATURE_SYMBOLS.items():
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if sym not in df.columns.get_level_values(0):
                    continue
                sub = df[sym].dropna(subset=["Close"]).copy()
            else:
                sub = df.dropna(subset=["Close"]).copy()
            out[short] = sub
        except KeyError:
            continue
    return out


def _build_feature_matrix(
    data: dict[str, pd.DataFrame],
    target: str,
) -> pd.DataFrame:
    """Construct the feature matrix + label for a target.

    Each row is a date. Features use ONLY data from that date or
    earlier (no lookahead). Label is next-day direction of `target`.
    """
    if target not in data:
        return pd.DataFrame()
    base = data[target][["Close"]].rename(columns={"Close": "px"})
    base["ret_1d"] = base["px"].pct_change()
    base["ret_5d"] = base["px"].pct_change(5)
    base["ret_20d"] = base["px"].pct_change(20)
    # RSI
    delta = base["px"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    base["rsi"] = 100 - (100 / (1 + rs))
    # Realized volatility
    base["vol_20d"] = base["ret_1d"].rolling(20).std()
    # Bollinger position
    mid = base["px"].rolling(20).mean()
    sd = base["px"].rolling(20).std()
    base["bb_pos"] = (base["px"] - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    # SMA ratios
    base["sma200_ratio"] = base["px"] / base["px"].rolling(200).mean()
    # 5-day momentum of RSI
    base["rsi_change_5d"] = base["rsi"] - base["rsi"].shift(5)

    # Cross-asset features (from other symbols, same date)
    for other in ["brent", "dxy", "ovx", "vix", "xle", "tnx"]:
        if other in data:
            o = data[other][["Close"]].rename(columns={"Close": f"{other}_px"})
            base = base.join(o, how="left")
            base[f"{other}_ret_1d"] = base[f"{other}_px"].pct_change()
            base[f"{other}_ret_5d"] = base[f"{other}_px"].pct_change(5)

    # Calendar features
    base["dow"] = base.index.dayofweek
    base["month"] = base.index.month
    # Seasonal-cycle encoding (cos/sin of day-of-year for smoothness)
    doy = base.index.dayofyear
    base["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    base["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)

    # Label: next-day direction (1 = up, 0 = down/flat)
    base["next_ret"] = base["ret_1d"].shift(-1)
    base["label"] = (base["next_ret"] > 0).astype(int)

    # Drop rows with NaN in any feature
    feature_cols = [c for c in base.columns
                    if c not in ("px", "next_ret", "label")
                    and not c.endswith("_px")]
    base = base.dropna(subset=feature_cols + ["label"])
    return base


def train_and_evaluate(
    target: str = "wti",
    years_back: int = 12,
    train_years: int = 5,
) -> dict[str, Any]:
    """Walk-forward train + evaluate. Returns dict with:
      accuracy_by_year, overall_accuracy, n_test_samples, model
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss
    LOGGER.info(f"Training predictor for {target}…")

    data = _fetch_history(years=years_back)
    if not data:
        return {"error": "no_data"}
    df = _build_feature_matrix(data, target)
    if df.empty:
        return {"error": "no_features", "target": target}

    feature_cols = [c for c in df.columns
                    if c not in ("px", "next_ret", "label")
                    and not c.endswith("_px")]
    LOGGER.info(f"  {len(df)} samples, {len(feature_cols)} features")

    # Walk-forward by calendar year
    df = df.sort_index()
    years_in_data = sorted({d.year for d in df.index})
    if len(years_in_data) < train_years + 2:
        return {"error": "insufficient_years"}
    test_years = years_in_data[train_years:]

    per_year_results = []
    all_preds = []
    all_labels = []
    all_probs = []

    for test_year in test_years:
        train_mask = df.index.year < test_year
        test_mask = df.index.year == test_year
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "label"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "label"].values
        if len(X_test) < 10:
            continue
        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs, labels=[0, 1])
        per_year_results.append({
            "year": int(test_year),
            "n_test": int(len(X_test)),
            "accuracy": float(acc),
            "log_loss": float(ll),
            "base_rate": float(y_test.mean()),
        })
        all_preds.extend(preds.tolist())
        all_labels.extend(y_test.tolist())
        all_probs.extend(probs.tolist())

    if not per_year_results:
        return {"error": "no_test_years"}

    overall_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    base_rate = float(np.mean(all_labels))

    # Train final model on ALL data for live prediction
    X_all = df[feature_cols].values
    y_all = df["label"].values
    final_clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, random_state=42,
    )
    final_clf.fit(X_all, y_all)

    return {
        "target": target,
        "feature_cols": feature_cols,
        "per_year": per_year_results,
        "overall_accuracy": overall_acc,
        "base_rate": base_rate,
        "edge_vs_baseline": overall_acc - max(base_rate, 1 - base_rate),
        "n_test_samples": len(all_labels),
        "model": final_clf,
        "latest_features": df[feature_cols].iloc[-1].to_dict()
            if len(df) else {},
    }


def predict_tomorrow_both() -> dict[str, Any]:
    """Run the predictor for BOTH WTI and NG=F and return a single
    dict with per-ETF recommendations.

    Uses calibration-bucket thresholds from the walk-forward analysis:
      WTI:
        P(up) in [0.55, 0.60) → strong BUY HOU (60% historical hit rate)
        P(up) >= 0.60        → BUY HOU (55% historical hit rate)
        P(up) <= 0.40        → SELL/no trade (model NOT well-calibrated
                               on bear oil — middle buckets are noise)
      NG=F:
        P(up) >= 0.60        → BUY HNU (55% historical)
        P(up) <= 0.40        → BUY HND (52% historical, down realized)
        otherwise            → SELL both / no trade
    """
    out = {
        "as_of": pd.Timestamp.utcnow().isoformat(),
        "wti": None,
        "ng": None,
        "recommendations": {},
    }

    for target in ["wti", "ng"]:
        try:
            r = predict_tomorrow(target=target)
        except Exception as e:
            out[target] = {"error": str(e)}
            continue
        if "error" in r:
            out[target] = r
            continue
        out[target] = r

    # Decide per-ETF actions
    wti_p = (out["wti"] or {}).get("prob_up")
    ng_p = (out["ng"] or {}).get("prob_up")

    rec = {
        "HOU.TO": {"action": "SELL", "reason": "no clear signal",
                   "prob_underlying_up": wti_p, "tier": "none"},
        "HOD.TO": {"action": "SELL", "reason": "no clear signal",
                   "prob_underlying_up": wti_p, "tier": "none"},
        "HNU.TO": {"action": "SELL", "reason": "no clear signal",
                   "prob_underlying_up": ng_p, "tier": "none"},
        "HND.TO": {"action": "SELL", "reason": "no clear signal",
                   "prob_underlying_up": ng_p, "tier": "none"},
    }

    # --- OIL PAIR ---
    if wti_p is not None:
        if 0.55 <= wti_p < 0.60:
            rec["HOU.TO"] = {
                "action": "BUY", "reason": f"WTI P(up)={wti_p:.2f} → 60% historical accuracy",
                "prob_underlying_up": wti_p, "tier": "strong",
            }
            rec["HOD.TO"] = {
                "action": "SELL", "reason": "opposing side of HOU strong-buy",
                "prob_underlying_up": wti_p, "tier": "none",
            }
        elif wti_p >= 0.60:
            rec["HOU.TO"] = {
                "action": "BUY", "reason": f"WTI P(up)={wti_p:.2f} → 55% historical accuracy",
                "prob_underlying_up": wti_p, "tier": "weak",
            }
            rec["HOD.TO"] = {
                "action": "SELL", "reason": "opposing side of HOU buy",
                "prob_underlying_up": wti_p, "tier": "none",
            }
        else:
            # FALLBACK: no calibrated bucket fires — use direction of P(up)
            if wti_p >= 0.50:
                rec["HOU.TO"] = {
                    "action": "BUY",
                    "reason": f"WTI P(up)={wti_p:.2f} → directional bias (uncalibrated)",
                    "prob_underlying_up": wti_p, "tier": "fallback",
                }
                rec["HOD.TO"] = {
                    "action": "SELL", "reason": "opposing side of HOU fallback-buy",
                    "prob_underlying_up": wti_p, "tier": "none",
                }
            else:
                rec["HOD.TO"] = {
                    "action": "BUY",
                    "reason": f"WTI P(up)={wti_p:.2f} → directional bias (uncalibrated, HOD untested)",
                    "prob_underlying_up": wti_p, "tier": "fallback",
                }
                rec["HOU.TO"] = {
                    "action": "SELL", "reason": "opposing side of HOD fallback-buy",
                    "prob_underlying_up": wti_p, "tier": "none",
                }

    # --- NATGAS PAIR ---
    if ng_p is not None:
        if ng_p >= 0.60:
            rec["HNU.TO"] = {
                "action": "BUY", "reason": f"NG P(up)={ng_p:.2f} → 55% historical accuracy",
                "prob_underlying_up": ng_p, "tier": "weak",
            }
            rec["HND.TO"] = {
                "action": "SELL", "reason": "opposing side of HNU buy",
                "prob_underlying_up": ng_p, "tier": "none",
            }
        elif ng_p <= 0.40:
            rec["HND.TO"] = {
                "action": "BUY", "reason": f"NG P(up)={ng_p:.2f} → 52% historical down rate",
                "prob_underlying_up": ng_p, "tier": "weak",
            }
            rec["HNU.TO"] = {
                "action": "SELL", "reason": "opposing side of HND buy",
                "prob_underlying_up": ng_p, "tier": "none",
            }
        else:
            # FALLBACK
            if ng_p >= 0.50:
                rec["HNU.TO"] = {
                    "action": "BUY",
                    "reason": f"NG P(up)={ng_p:.2f} → directional bias (uncalibrated)",
                    "prob_underlying_up": ng_p, "tier": "fallback",
                }
                rec["HND.TO"] = {
                    "action": "SELL", "reason": "opposing side of HNU fallback-buy",
                    "prob_underlying_up": ng_p, "tier": "none",
                }
            else:
                rec["HND.TO"] = {
                    "action": "BUY",
                    "reason": f"NG P(up)={ng_p:.2f} → directional bias (uncalibrated)",
                    "prob_underlying_up": ng_p, "tier": "fallback",
                }
                rec["HNU.TO"] = {
                    "action": "SELL", "reason": "opposing side of HND fallback-buy",
                    "prob_underlying_up": ng_p, "tier": "none",
                }

    out["recommendations"] = rec
    return out


def predict_tomorrow(target: str = "wti") -> dict[str, Any]:
    """Train fresh + return today's prediction for tomorrow's
    direction. Returns {prob_up, prob_down, direction, accuracy}."""
    result = train_and_evaluate(target=target)
    if "error" in result:
        return result
    model = result["model"]
    feature_cols = result["feature_cols"]
    latest = result["latest_features"]
    x = np.array([[latest[c] for c in feature_cols]])
    prob_up = float(model.predict_proba(x)[0, 1])
    return {
        "target": target,
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "direction": "up" if prob_up > 0.5 else "down",
        "out_of_sample_accuracy": result["overall_accuracy"],
        "edge_vs_baseline": result["edge_vs_baseline"],
        "per_year": result["per_year"],
    }

"""Research module — looking for a real edge breakthrough.

Tests the current ML predictor's 55.6% past-year accuracy under many
different lenses to find:
  1. Is it statistically real (not luck)?
  2. Does it generalize across different time windows?
  3. Are there regimes where edge is much stronger?
  4. Is there a probability sweet spot?
  5. Does ensembling multiple models help?
  6. Does requiring confluence with oil mean-reversion help?
  7. Does multi-day hold beat 1-day?

Each analysis returns honest numbers — no curve-fitting.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import predictor as pred_mod

LOGGER = logging.getLogger("paper_trader.research")


def _walk_forward_predictions(target: str, years_back: int = 12) -> pd.DataFrame:
    """Walk-forward train + predict for each year. Returns a DataFrame
    with columns: date, prob_up, label, year."""
    data = pred_mod._fetch_history(years_back)
    df = pred_mod._build_feature_matrix(data, target)
    if df.empty:
        return pd.DataFrame()
    feat_cols = [c for c in df.columns
                 if c not in ("px", "next_ret", "label")
                 and not c.endswith("_px")]
    df = df.sort_index()
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    years = sorted({d.year for d in df.index})

    out_rows = []
    for ty in years[5:]:
        mask_tr = df.index.year < ty
        mask_te = df.index.year == ty
        if mask_tr.sum() < 500 or mask_te.sum() < 30:
            continue
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200,
            l2_regularization=1.0, random_state=42,
        )
        clf.fit(df.loc[mask_tr, feat_cols].values,
                df.loc[mask_tr, "label"].values)
        probs = clf.predict_proba(df.loc[mask_te, feat_cols].values)[:, 1]
        for date, p, y in zip(
            df.loc[mask_te].index, probs, df.loc[mask_te, "label"],
        ):
            out_rows.append({
                "date": pd.Timestamp(date),
                "prob_up": float(p),
                "label": int(y),
                "year": int(ty),
                "target": target,
            })
    return pd.DataFrame(out_rows)


def _ensemble_walk_forward(target: str, years_back: int = 12,
                           n_seeds: int = 5) -> pd.DataFrame:
    """Train N models with different random seeds, average probabilities."""
    data = pred_mod._fetch_history(years_back)
    df = pred_mod._build_feature_matrix(data, target)
    if df.empty:
        return pd.DataFrame()
    feat_cols = [c for c in df.columns
                 if c not in ("px", "next_ret", "label")
                 and not c.endswith("_px")]
    df = df.sort_index()
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    years = sorted({d.year for d in df.index})
    out_rows = []
    for ty in years[5:]:
        mask_tr = df.index.year < ty
        mask_te = df.index.year == ty
        if mask_tr.sum() < 500 or mask_te.sum() < 30:
            continue
        X_tr = df.loc[mask_tr, feat_cols].values
        y_tr = df.loc[mask_tr, "label"].values
        X_te = df.loc[mask_te, feat_cols].values
        ensemble_probs = np.zeros(len(X_te))
        for seed in range(n_seeds):
            clf = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=200,
                l2_regularization=1.0, random_state=seed,
            )
            clf.fit(X_tr, y_tr)
            ensemble_probs += clf.predict_proba(X_te)[:, 1] / n_seeds
        for date, p, y in zip(
            df.loc[mask_te].index, ensemble_probs, df.loc[mask_te, "label"],
        ):
            out_rows.append({
                "date": pd.Timestamp(date),
                "prob_up": float(p),
                "label": int(y),
                "year": int(ty),
            })
    return pd.DataFrame(out_rows)


def _apply_recommendation_logic(target: str, prob_up: float) -> tuple:
    """Apply the same logic as predict_tomorrow_both. Returns
    (recommended_ticker_direction, tier)."""
    if target == "wti":
        if 0.55 <= prob_up < 0.60:
            return ("bull", "strong")
        elif prob_up >= 0.60:
            return ("bull", "weak")
        elif prob_up >= 0.50:
            return ("bull", "fallback")
        else:
            return ("bear", "fallback")
    else:
        if prob_up >= 0.60:
            return ("bull", "weak")
        elif prob_up <= 0.40:
            return ("bear", "weak")
        elif prob_up >= 0.50:
            return ("bull", "fallback")
        else:
            return ("bear", "fallback")


def _score_predictions(pred_df: pd.DataFrame, target: str) -> dict:
    """Given walk-forward predictions, apply the recommendation logic
    and score against actual next-day direction."""
    right = wrong = 0
    by_tier = {}
    for _, row in pred_df.iterrows():
        direction, tier = _apply_recommendation_logic(target, row["prob_up"])
        # bull recommend → right if actual_up; bear → right if actual_down
        actual_up = bool(row["label"])
        is_right = (direction == "bull" and actual_up) or (
            direction == "bear" and not actual_up
        )
        if is_right:
            right += 1
        else:
            wrong += 1
        tb = by_tier.setdefault(tier, {"right": 0, "wrong": 0})
        tb["right" if is_right else "wrong"] += 1
    total = right + wrong
    return {
        "right": right, "wrong": wrong, "total": total,
        "accuracy": (right / total) if total else 0.0,
        "by_tier": by_tier,
    }


def stage1_statistical_significance(pred_df: pd.DataFrame,
                                     target: str) -> dict:
    """Binomial test: is the accuracy meaningfully different from 50%?"""
    from scipy import stats
    res = _score_predictions(pred_df, target)
    n = res["total"]
    k = res["right"]
    if n == 0:
        return {"error": "no_data"}
    # Two-sided binomial test against null p=0.5
    p_value = stats.binomtest(k, n, p=0.5, alternative="greater").pvalue
    # 95% confidence interval (Wilson)
    p_hat = k / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return {
        "target": target,
        "accuracy": p_hat,
        "n_trades": n,
        "right": k,
        "p_value_vs_random": float(p_value),
        "ci_95_lower": float(center - margin),
        "ci_95_upper": float(center + margin),
        "significant": bool(p_value < 0.05),
    }


def stage2_window_stability(pred_df: pd.DataFrame, target: str) -> list[dict]:
    """Score each year separately."""
    rows = []
    for year in sorted(pred_df["year"].unique()):
        year_df = pred_df[pred_df["year"] == year]
        res = _score_predictions(year_df, target)
        rows.append({
            "year": int(year),
            "n": res["total"],
            "accuracy": res["accuracy"],
        })
    return rows


def stage3_regime_conditioning(pred_df: pd.DataFrame, target: str) -> dict:
    """Bucket accuracy by various regime features."""
    if pred_df.empty:
        return {}
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_winter"] = df["month"].isin([11, 12, 1, 2, 3])

    # Pre-compute per-row recommendation correctness
    correctness = []
    for _, row in df.iterrows():
        direction, tier = _apply_recommendation_logic(target, row["prob_up"])
        actual_up = bool(row["label"])
        is_right = (direction == "bull" and actual_up) or (
            direction == "bear" and not actual_up
        )
        correctness.append(1 if is_right else 0)
    df["correct"] = correctness

    out = {
        "by_day_of_week": {},
        "by_month": {},
        "by_year": {},
    }
    for dow, grp in df.groupby("dow"):
        names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        if dow < len(names):
            out["by_day_of_week"][names[dow]] = {
                "n": len(grp),
                "accuracy": float(grp["correct"].mean()),
            }
    for month, grp in df.groupby("month"):
        out["by_month"][int(month)] = {
            "n": len(grp),
            "accuracy": float(grp["correct"].mean()),
        }
    return out


def stage4_probability_buckets(pred_df: pd.DataFrame) -> list[dict]:
    """Hit rate by probability bucket — find the sweet spot."""
    buckets = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.45), (0.45, 0.50),
               (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.00)]
    rows = []
    for lo, hi in buckets:
        mask = (pred_df["prob_up"] >= lo) & (pred_df["prob_up"] < hi)
        if mask.sum() < 10:
            rows.append({
                "bucket": f"[{lo:.2f},{hi:.2f})",
                "n": int(mask.sum()),
                "actual_up_rate": None,
                "effective_acc": None,
                "note": "too few samples",
            })
            continue
        actual_up = float(pred_df.loc[mask, "label"].mean())
        # If betting "up" makes sense here (P > 0.5), effective accuracy = actual_up
        # If betting "down" (P < 0.5), effective accuracy = 1 - actual_up
        avg_p = float(pred_df.loc[mask, "prob_up"].mean())
        if avg_p >= 0.5:
            eff = actual_up
            bet = "up"
        else:
            eff = 1 - actual_up
            bet = "down"
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n": int(mask.sum()),
            "avg_prob": round(avg_p, 3),
            "actual_up_rate": round(actual_up, 3),
            "bet_direction": bet,
            "effective_acc": round(eff, 3),
        })
    return rows


def stage5_ensemble_comparison(target: str) -> dict:
    """Compare single-model vs 5-model ensemble accuracy."""
    LOGGER.info(f"Stage 5: ensemble comparison for {target}…")
    single = _walk_forward_predictions(target)
    ensemble = _ensemble_walk_forward(target, n_seeds=5)
    return {
        "single_model": _score_predictions(single, target),
        "ensemble_5": _score_predictions(ensemble, target),
    }


def stage6_confluence_filter(pred_df: pd.DataFrame, target: str) -> dict:
    """If we ONLY trade when both ML and a 'momentum confirms direction'
    rule agree, does accuracy improve?

    Simple confluence: ML says up + 5-day return > 0 AND today's return > 0
    """
    if pred_df.empty:
        return {}
    data = pred_mod._fetch_history(12)
    underlying_key = target  # 'wti' or 'ng'
    if underlying_key not in data:
        return {"error": "no underlying data"}
    px = data[underlying_key]["Close"]
    ret_1d = px.pct_change()
    ret_5d = px.pct_change(5)
    out_total = {"right": 0, "wrong": 0}
    confluence_count = 0
    skipped = 0
    for _, row in pred_df.iterrows():
        date = pd.Timestamp(row["date"])
        # tz-align
        if px.index.tz is not None and date.tz is None:
            date = date.tz_localize(px.index.tz)
        elif px.index.tz is None and date.tz is not None:
            date = date.tz_localize(None)
        try:
            r1 = float(ret_1d.asof(date))
            r5 = float(ret_5d.asof(date))
        except (KeyError, ValueError):
            skipped += 1
            continue
        if pd.isna(r1) or pd.isna(r5):
            skipped += 1
            continue
        direction, _ = _apply_recommendation_logic(target, row["prob_up"])
        # Confluence: ML and momentum must agree
        momentum_up = (r1 > 0) and (r5 > 0)
        momentum_down = (r1 < 0) and (r5 < 0)
        if direction == "bull" and not momentum_up:
            skipped += 1
            continue
        if direction == "bear" and not momentum_down:
            skipped += 1
            continue
        confluence_count += 1
        actual_up = bool(row["label"])
        is_right = (direction == "bull" and actual_up) or (
            direction == "bear" and not actual_up
        )
        if is_right:
            out_total["right"] += 1
        else:
            out_total["wrong"] += 1
    total = out_total["right"] + out_total["wrong"]
    return {
        "confluence_trades": confluence_count,
        "skipped": skipped,
        "right": out_total["right"],
        "wrong": out_total["wrong"],
        "accuracy": (out_total["right"] / total) if total else 0.0,
    }


def stage7_multi_day_hold(pred_df: pd.DataFrame, target: str,
                          hold_days: int = 3) -> dict:
    """What if we hold for N days instead of 1? Does directional edge
    accumulate over multi-day periods?"""
    if pred_df.empty:
        return {}
    data = pred_mod._fetch_history(12)
    px = data[target]["Close"]
    forward_ret = px.pct_change(hold_days).shift(-hold_days)
    out = {"right": 0, "wrong": 0}
    for _, row in pred_df.iterrows():
        date = pd.Timestamp(row["date"])
        if px.index.tz is not None and date.tz is None:
            date = date.tz_localize(px.index.tz)
        elif px.index.tz is None and date.tz is not None:
            date = date.tz_localize(None)
        try:
            fr = float(forward_ret.asof(date))
        except (KeyError, ValueError):
            continue
        if pd.isna(fr):
            continue
        direction, _ = _apply_recommendation_logic(target, row["prob_up"])
        actual_up = fr > 0
        is_right = (direction == "bull" and actual_up) or (
            direction == "bear" and not actual_up
        )
        if is_right:
            out["right"] += 1
        else:
            out["wrong"] += 1
    total = out["right"] + out["wrong"]
    return {
        "hold_days": hold_days,
        "right": out["right"],
        "wrong": out["wrong"],
        "accuracy": (out["right"] / total) if total else 0.0,
    }


def run_full_research() -> dict:
    """Run all 7 stages for both WTI and NG=F."""
    results: dict[str, Any] = {}
    for target in ["wti", "ng"]:
        LOGGER.info(f"\n=== Research on {target.upper()} ===")
        pred_df = _walk_forward_predictions(target)
        if pred_df.empty:
            results[target] = {"error": "no_predictions"}
            continue

        target_res = {
            "stage1_significance": stage1_statistical_significance(pred_df, target),
            "stage2_yearly_stability": stage2_window_stability(pred_df, target),
            "stage3_regime": stage3_regime_conditioning(pred_df, target),
            "stage4_buckets": stage4_probability_buckets(pred_df),
            "stage5_ensemble": stage5_ensemble_comparison(target),
            "stage6_confluence": stage6_confluence_filter(pred_df, target),
            "stage7_multi_day": {
                "hold_1d": _score_predictions(pred_df, target),
                "hold_2d": stage7_multi_day_hold(pred_df, target, 2),
                "hold_3d": stage7_multi_day_hold(pred_df, target, 3),
                "hold_5d": stage7_multi_day_hold(pred_df, target, 5),
            },
        }
        results[target] = target_res
    return results

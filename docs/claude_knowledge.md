# Stock Signal Dashboard — Reference Knowledge

This document is a reference for the AI assistant. Upload to a claude.ai Project's **Knowledge** section so the assistant can ground answers without the user re-explaining definitions every time.

---

## 1. Watchlist tickers

### TSLA — Tesla, Inc.
- US large-cap, NASDAQ. Sector: Automotive / Energy / AI.
- High beta, news-driven. Earnings, delivery numbers, Musk announcements all move it.

### NOW — ServiceNow, Inc.
- US large-cap, NYSE. Sector: Enterprise SaaS (workflow automation).
- Lower volatility than TSLA. Big quarterly earnings reactions.

### HOD.TO — Horizons BetaPro Crude Oil Inverse Leveraged Daily Bear ETF
- Canadian ETF on TSX. Issuer: Global X (formerly Horizons).
- Inverse 2x exposure to front-month WTI crude oil futures, **reset daily**.
- **Decay risk**: in choppy markets, daily resetting causes erosion regardless of direction. Not a long-term hold.

### HOU.TO — Horizons BetaPro Crude Oil Leveraged Daily Bull ETF
- 2x bull exposure to WTI crude oil futures, daily reset.
- Same decay risk as HOD.TO. Pairs trade against HOD.

### HND.TO — Horizons BetaPro Natural Gas Inverse Leveraged Daily Bear ETF
- 2x inverse exposure to front-month natural gas futures, daily reset.

### HNU.TO — Horizons BetaPro Natural Gas Leveraged Daily Bull ETF
- 2x bull exposure to natural gas futures, daily reset.

### Important context for HOD/HOU/HND/HNU.TO
- These are designed for **intraday or very short-term tactical trading**, not multi-day holds.
- "Volatility decay" means a leveraged ETF can lose value even when the underlying ends flat after large daily swings.
- Spot oil/gas direction is a necessary but not sufficient predictor — daily path matters.

---

## 2. Indicator formulas + interpretation

### RSI (Relative Strength Index, 14-period)
- **Formula**: RSI = 100 − [100 / (1 + RS)], where RS = avg gain / avg loss over 14 bars (Wilder's smoothing).
- **Interpretation**:
  - < 30: oversold (mean-reversion bias to upside)
  - > 70: overbought (mean-reversion bias to downside)
  - 30–70: no clear signal
- **Caveat**: in strong trends, RSI can stay overbought/oversold for weeks. Not a reliable reversal trigger alone.

### MACD (12, 26, 9)
- **Formula**: MACD line = EMA(12) − EMA(26). Signal line = EMA(9) of MACD line. Histogram = MACD − Signal.
- **Interpretation**:
  - MACD crosses above signal → bullish momentum shift
  - MACD crosses below signal → bearish momentum shift
  - Histogram growing positive → momentum accelerating up
  - Zero line cross → trend change confirmation

### Bollinger Bands (20-period, 2 standard deviations)
- **Formula**: Middle = SMA(20). Upper = SMA(20) + 2σ. Lower = SMA(20) − 2σ.
- **%B** = (Close − Lower) / (Upper − Lower). 0 = touch lower, 1 = touch upper.
- **Bandwidth** = (Upper − Lower) / Middle. Low bandwidth = volatility squeeze (often precedes large move).
- **Interpretation**:
  - Touch of lower band ≠ buy signal alone; mean reversion only works in ranging markets
  - Sustained walks along upper band = strong uptrend, NOT overbought

### Simple Moving Averages
- **SMA(5)** — short-term, captures last week's momentum
- **SMA(20)** — ~1 month average, trend filter
- **SMA(50)** — medium-term trend
- **SMA(200)** — long-term trend, "bull/bear market line"
- **Golden cross**: SMA(50) crosses above SMA(200) → long-term bullish
- **Death cross**: SMA(50) crosses below SMA(200) → long-term bearish

### ADX (Average Directional Index, 14-period)
- **Interpretation**:
  - < 20: weak/no trend (range-bound)
  - 20–25: emerging trend
  - > 25: strong trend (direction independent — ADX doesn't say up or down)
  - > 40: very strong trend, possibly extended

### Donchian Channels (20-period)
- High = max(High, 20). Low = min(Low, 20).
- Breakout above 20-day high or below 20-day low. Used by trend-following systems (e.g., Turtles).

### ATR (Average True Range, 14-period)
- Measures volatility, not direction. Useful for sizing stops (e.g., stop at 2× ATR).

### Stochastic (%K=14, %D=3, slow=3)
- Similar to RSI but bounded by recent high/low range.
- < 20 oversold, > 80 overbought. Cross of %K above/below %D = trigger.

### Parabolic SAR
- Plots dots above (downtrend) or below (uptrend) price. Flip = trend change signal.

### Supertrend (period=10, multiplier=3)
- ATR-based trailing stop indicator. Above price = bearish, below price = bullish.

### Keltner Channels
- EMA(20) center ± multiplier × ATR(20). Similar to Bollinger but ATR-based.

### Ichimoku
- Complex 5-line system. Cloud (Senkou A/B) acts as dynamic support/resistance.

---

## 3. Strategies in the dashboard

| Strategy | Logic | Best in |
|---|---|---|
| Trend (RSI + MACD + SMA) | RSI > 50 + MACD bullish + Close > SMA50 | Trending markets |
| Bollinger Mean Reversion | Buy at lower band, sell at upper | Range-bound |
| Donchian Breakout | Buy 20-day high break, sell 20-day low break | Trending |
| SMA200 Dip Buy | Buy 5%+ dip below SMA200 in long-term uptrend | Bull markets |
| RSI Strategy | Buy < 30, sell > 70 | Range-bound |
| MACD Cross | Buy on bullish cross, sell on bearish cross | Trending |
| Momentum | 10-day rate of change > 0 | Trending |
| Stochastic Slow | %K crosses %D below 20 / above 80 | Range-bound |
| Keltner | Bands as dynamic S/R | All conditions |
| Supertrend | Flip-based trend follow | Trending |
| Parabolic SAR | Dot flips | Trending |
| MA Cross (20/50) | Fast above slow = buy, below = sell | Trending |
| Inside Bar Breakout | Inside bar then break of range | Volatility expansion |
| Outside Bar Reversal | Outside bar reverses prior trend | Reversals |
| Outside Bar Breakout | Outside bar breaks high/low after | Continuation |
| Candlestick Patterns | Hammer, Engulfing, Star, Doji combos | Reversals |
| Double Top / Bottom | M-pattern reversal | Reversals |

---

## 4. How to interpret backtest stats shown

Every chart shows: Strategy %, B&H %, Max DD %, Win %, Trades.

- **Strategy %** = total return of executing the buy/sell signals over the period
- **B&H %** = simple buy-and-hold return for the same period (benchmark)
- **Strategy beating B&H** = signal generates alpha vs. passive holding
- **Strategy losing to B&H in a strong uptrend is normal** — most strategies underperform in raging bulls because they sit in cash during pullbacks
- **Max DD** = peak-to-trough drawdown. Lower is better for risk-adjusted thinking
- **Win %** alone is misleading — a 30% win rate with 4:1 risk/reward beats a 70% win rate with 1:3 risk/reward

---

## 5. Hard rules for the assistant

- **Never** give specific investment advice or buy/sell recommendations.
- **Never** state predictions about price direction with confidence.
- Always frame technical setups as **statistical tendencies**, not certainties.
- If asked "what should I buy?", redirect to "I can describe what the indicators are saying about ticker X, but the trade decision is yours."
- For leveraged ETF questions (HOD/HOU/HND/HNU.TO), always remind the user about volatility decay if they suggest holding past intraday.
- For 10%+ short-term move requests, note honestly that no public technical pattern reliably predicts moves of that magnitude in a week.

# Paper Trader — Setup

A background agent that paper-trades HOU.TO / HOD.TO / HNU.TO / HND.TO
based on signals from WTI crude, Henry Hub natgas, and DXY. Runs 24/7
on your machine; the Streamlit app's **🤖 Paper Trader** tab is a
read-only dashboard.

## Quick start

### 0. (Optional but recommended) Get an EIA API key

The inventory-based strategies (`oil_inv_drawdown`, `oil_inv_build`,
`gas_storage_drawdown`, `gas_storage_build`) need the free EIA Open
Data API. Without a key they simply don't fire — the other
strategies still work. With a key, the backtest and live agent both
benefit from real catalyst data.

1. Register here (takes one minute): https://www.eia.gov/opendata/register.php
2. Set the env var in PowerShell:
   ```powershell
   setx EIA_API_KEY "your_key_here"
   ```
   The `setx` form persists across reboots. Restart PowerShell + the
   agent + Streamlit after running this.
3. Alternatively, add to `.streamlit/secrets.toml`:
   ```
   EIA_API_KEY = "your_key_here"
   ```

### 1. Initialize the databases

The first time you click the **🤖 Paper Trader** tab in the Streamlit
app, it creates `paper_trader.db` and `strategies_state.db` in the
project root. Or do it manually:

```bash
python -c "import paper_trader; paper_trader.init_databases()"
```

### 2. Test the agent in the foreground

Before automating, make sure the agent starts cleanly:

```bash
python -m paper_trader.agent
```

You should see lines like:

```
2026-05-14 09:42:13 [INFO] Paper-trader agent started.
2026-05-14 09:42:14 [INFO] OPEN  pos#1 HOU.TO via oil_rsi_reversion @ ...
```

Stop with Ctrl-C. If it ran successfully, move on to step 3.

### 3. Register as a Windows scheduled task

Open PowerShell **as Administrator**, cd to the project folder, and
run:

```powershell
./scripts/setup_agent_task.ps1
```

This registers a task named **PaperTraderAgent** that:

- Starts when you log into Windows
- Restarts automatically if it crashes (3 attempts, 5 min apart)
- Has no visible window (uses `pythonw.exe`)
- Persists across reboots
- Runs on battery

Useful PowerShell commands afterwards:

```powershell
Start-ScheduledTask   -TaskName PaperTraderAgent
Stop-ScheduledTask    -TaskName PaperTraderAgent
Get-ScheduledTask     -TaskName PaperTraderAgent | Format-List
Unregister-ScheduledTask -TaskName PaperTraderAgent -Confirm:$false
```

### 4. Verify it's running

- Open the **🤖 Paper Trader** tab. The top-of-page health chip should
  show **🟢 Alive** with a fresh heartbeat timestamp.
- Check `logs/agent.log` — should append a new line every 5 min during
  market hours, every 30 min off-hours.

## How it works

### The 4 ETFs

All four are Horizons 2x leveraged ETFs on the TSX, organized as two
pairs:

| Pair | Bull | Bear |
|------|------|------|
| Oil (WTI) | HOU.TO | HOD.TO |
| Natgas (Henry Hub) | HNU.TO | HND.TO |

At any moment the agent holds **at most one position** among the four.
A signal is a (ticker, conviction) pair; the highest-conviction signal
across all enabled strategies wins.

### Trade rules

- **Position size**: 25% of current cash per trade
- **Commission**: $5 per leg (simulated; never debited from a real
  brokerage)
- **Slippage**: 0.5% round trip (0.25% per leg) — these ETFs have
  wide spreads
- **Exits** (whichever fires first):
  - Stop loss: −5%
  - Take profit: +5%
  - Signal flip: a strategy now recommends the opposite-direction ETF
    in the same pair with conviction ≥ 0.6
  - Hard timeout: 5 trading days

### Starting strategies (week 1)

| Name | Logic |
|------|-------|
| `oil_rsi_reversion` | RSI(14) on CL=F. < 30 → HOU. > 70 → HOD. |
| `natgas_macd_cross` | MACD cross on NG=F (with hist confirming). |
| `dxy_oil_inverse` | DXY drops >0.5% + WTI ≥ 0 → HOU; mirror → HOD. |

These are deliberate baselines. Real edge comes in weeks 2-4 once EIA
inventory + weather features are added.

### "Training" — bandit, not RL

Every off-hours cycle, the agent recomputes per-strategy lifetime
stats from the closed-trade log:

- Win rate
- Profit factor (gross wins ÷ gross losses)
- Expectancy (avg $ per trade)
- Total P&L

Strategies are stored in a separate database (`strategies_state.db`)
that is **not wiped** when you reset capital. So the agent's memory
of which strategies work survives any number of capital resets.

Future versions (week 3+) will adjust `capital_weight` toward
profitable strategies — winners get bigger positions, losers shrink.

## Reset semantics

The **Reset capital** button in the dashboard wipes:

- ✅ Positions (open + closed)
- ✅ Trade history
- ✅ Equity curve
- ✅ Signal log

It **does not** touch:

- ❌ Strategy registry (which strategies exist, enabled/disabled)
- ❌ Lifetime stats per strategy
- ❌ Capital weights

So you can stress-test a new starting balance without throwing away
months of learning.

## Files this creates

```
P1/
├── paper_trader.db              ← resettable: balance, positions, trades
├── strategies_state.db          ← persistent: strategy stats
├── paper_trader.lock            ← lock file (deleted on clean exit)
└── logs/
    ├── agent.log                ← daily log, rotated, 90-day retention
    ├── agent.log.2026-05-13     ← yesterday's
    └── ...
```

## Troubleshooting

**Dashboard shows 🔴 Down:**
- Open Task Scheduler (`taskschd.msc`) → find PaperTraderAgent → check
  "Last Run Result" and "Last Run Time"
- Tail `logs/agent.log` — most errors are logged with full traceback
- The lock file in the project root holds the PID of the running
  instance; if it's stuck, delete `paper_trader.lock` and restart

**"Database is locked" errors:**
- WAL mode is enabled, so this should be rare. If it happens during
  active development (e.g. you have a debugger paused inside a
  transaction), restart both the agent and the Streamlit app.

**Strategy never fires:**
- Check the **Signal log** (added in a future version) — for now,
  inspect `signals` table in `paper_trader.db` with any SQLite browser.
- The thresholds in `paper_trader/strategies.py` are conservative; you
  can lower them but expect more false signals.

## Stopping the agent

- **Pause** (keep daemon running, just skip cycles): click ⏸️ Pause
  in the Paper Trader tab. Toggleable.
- **Hard stop**: `Stop-ScheduledTask -TaskName PaperTraderAgent` in
  PowerShell.
- **Uninstall**: `Unregister-ScheduledTask -TaskName PaperTraderAgent
  -Confirm:$false`.

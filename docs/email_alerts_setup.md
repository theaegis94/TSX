# Daily Email Alerts — Setup

The daily alerts run every weekday at **9:30 AM Eastern** via GitHub Actions
(free tier). They scan **TSX + TSX Venture**, rank tickers by upside
potential, and send you a digest. Plus: any saved rule set tagged 🔔 Alert
in the app fires its own match list.

## One-time setup (5 minutes)

### 1. Get a Gmail App Password

You'll send email through Gmail's SMTP. App Passwords let scripts log in
without using your real Google password.

1. Enable 2-Factor Authentication on your Google account if not already:
   https://myaccount.google.com/security
2. Go to https://myaccount.google.com/apppasswords
3. **App**: select "Mail"
4. **Device**: pick "Other (Custom name)" → name it `StockSignals`
5. Click **Generate**. Copy the 16-character password (no spaces).

### 2. Add GitHub Secrets

On the GitHub repo:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** for each of these:

| Secret name        | Value                                                         |
| ------------------ | ------------------------------------------------------------- |
| `SMTP_USER`        | Your Gmail address (e.g., `you@gmail.com`)                    |
| `SMTP_PASS`        | The 16-char App Password from step 1                          |
| `ALERT_TO`         | Where to send (defaults to `SMTP_USER` if omitted)            |
| `FINNHUB_API_KEY`  | Your Finnhub API key (for news sentiment in the email)        |

The workflow file `.github/workflows/daily_alerts.yml` is already configured
to use these secrets — no edits needed.

### 3. Pick which rule sets to alert on

In the app:

1. Open **Custom Patterns** tab
2. Build your rules (or load a preset, then tweak)
3. Click **💾 Save / load rule sets** → save with a name
4. Click the **🔔 Alert** button next to the saved set → toggles to **🔕 Mute**

The next morning's email will include matches from any rule set tagged
🔔 Alert.

### 4. Commit the config

The alert toggle writes to `saved_rules.json` in your local repo. Commit
and push that file:

```bash
git add saved_rules.json
git commit -m "configure email alerts"
git push origin main
```

GitHub Actions reads `saved_rules.json` from the repo, so without pushing
it, the cron job won't know which rules you marked.

## What the daily email looks like

- **Header**: today's date, scan stats
- **🎯 Top 15 Upside Candidates** — TSX/TSXV tickers ranked by composite
  score (CONVICTION + VOL_OUTLOOK + trend + news). Each row links to Yahoo
  Finance for that ticker.
- **🎯 Per-rule matches** — for each rule set you tagged 🔔 Alert, the
  list of tickers that match all conditions in the rule.

## Testing without waiting for cron

Two ways to test:

### A) Trigger the workflow manually
GitHub → **Actions** tab → **Daily Email Alerts** → **Run workflow**.

### B) Run locally
Set the env vars in your shell, then run:

```bash
export SMTP_USER="you@gmail.com"
export SMTP_PASS="xxxxxxxxxxxxxxxx"
export ALERT_TO="you@gmail.com"
export FINNHUB_API_KEY="your_finnhub_key"
export ALERTS_UNIVERSE="tsx_and_tsxv"
python scripts/email_alerts.py
```

Or quickly with a smaller universe to test:

```bash
ALERTS_UNIVERSE=tsx60 python scripts/email_alerts.py
```

## Schedule

Default: weekdays at **9:30 AM Eastern**. The workflow has two cron entries
to handle DST (one for EDT, one for EST) — GitHub Actions doesn't auto-
adjust. To change frequency, edit `.github/workflows/daily_alerts.yml`:

- Multiple times per day: add more `cron:` lines (each in UTC)
- Different time: convert your local time to UTC, change the cron
- Different days: cron field 5 is day-of-week (1=Mon ... 7=Sun)

Free GitHub Actions tier gives you 2000 minutes/month — daily runs use
~10-15 min/day, so ~300 min/month, well under the limit.

## Limitations

- Free tier scan of full TSX+TSXV takes 10-15 minutes. If you want faster,
  edit `ALERTS_UNIVERSE` in the workflow to `tsx_composite` (~250 tickers,
  2 min) or `tsx60` (~60 tickers, 30 sec).
- `ALERTS_MIN_VOL` (default 50000) filters out micro-cap noise. Lower it to
  see more TSXV penny stocks; raise it to focus on liquid names only.
- Finnhub TSX news coverage is sparse — the script falls back to yfinance
  via Yahoo Canada when available.
- Emails are HTML-formatted. Some old email clients may render plain text;
  Gmail/Outlook/Apple Mail handle it fine.

## Troubleshooting

- **No email received**: check Actions → recent workflow run logs. Common
  issues: wrong App Password (regenerate), Gmail blocked the login (check
  https://myaccount.google.com/notifications for "Critical security alert").
- **"No high-conviction setups"**: real result on flat market days. The
  scan ran fine; just no tickers scored positive.
- **Email goes to Spam**: gmail-to-gmail rarely does, but cross-provider
  might. Mark as Not Spam once and Gmail learns.

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from binance.client import Client
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from time import sleep

load_dotenv()

# ====================== CONFIG ======================
CHECK_INTERVAL_HOURS = 12
MODEL                = "grok-4.20-beta-0309-reasoning"
SYMBOL               = "SOLUSDT"

client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.x.ai/v1")
binance = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")


# ====================== RETRY HELPER ======================
def with_retry(func, max_attempts=3, backoff=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in ("connection", "timeout", "ssl", "rate limit"))
            if is_transient and attempt < max_attempts - 1:
                print(f"⚠️  Transient error (attempt {attempt+1}/{max_attempts}) — retrying in {backoff}s...")
                sleep(backoff)
                backoff *= 2
            else:
                print(f"❌ All retry attempts failed: {e}")
                raise


# ====================== EMAIL ======================
def send_trend_email(report):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASS]):
        return
    subject = (
        f"SOL Trend Agent — {report['regime']} | "
        f"Conf {report['confidence']}/10 | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
    )

    body = f"""SOL Macro Trend Agent — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

📈 Regime: {report['regime']} | Confidence: {report['confidence']}/10

📝 {report['summary']}

━━━━━━━━━━━━━━━━━━━━━━━━
OUTLOOK
━━━━━━━━━━━━━━━━━━━━━━━━
1-Month:   {report['outlook_1m']}
2-3 Month: {report['outlook_2_3m']}

━━━━━━━━━━━━━━━━━━━━━━━━
KEY LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━
Near-term Support:  {report['near_support']}
Near-term Resist:   {report['near_resistance']}
Major Support:      {report['major_support']}
Major Resistance:   {report['major_resistance']}
Key Fib Levels:     {report['key_fibs']}

━━━━━━━━━━━━━━━━━━━━━━━━
MARKET STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━
Weekly: {report['weekly_structure']} | RSI: {report['weekly_rsi']} | EMA50/200: {report['weekly_ema']}
Daily:  {report['daily_structure']}  | RSI: {report['daily_rsi']}  | EMA50/200: {report['daily_ema']}

━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIOS (Next 1-3 Months)
━━━━━━━━━━━━━━━━━━━━━━━━
{report['scenarios']}

━━━━━━━━━━━━━━━━━━━━━━━━
INVALIDATION LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━
Bull Invalidation: {report['bull_invalidation']}
Bear Invalidation: {report['bear_invalidation']}

━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━
{report['recommendation']}
"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("📧 Trend Agent email sent")
    except Exception as e:
        print(f"Email failed: {e}")


# ====================== DATA FETCH ======================
def get_klines(interval, limit=500):
    def _fetch():
        klines = binance.futures_klines(symbol=SYMBOL, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'n', 'tbav', 'tbqav', 'ignore'
        ])
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df.set_index('open_time')
    return with_retry(_fetch)


# ====================== TA CALCULATIONS ======================
def calculate_rsi(df, period=14):
    if len(df) < period + 10:
        return 50.0
    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)


def calculate_ema(df, span):
    return df['close'].ewm(span=span, adjust=False).mean().iloc[-1]


def ema_label(price, ema50, ema200):
    if price > ema50 > ema200:
        return "above both (bullish)"
    elif price < ema50 < ema200:
        return "below both (bearish)"
    elif price > ema50 and ema50 < ema200:
        return "above EMA50 / EMA50 below EMA200 (mixed)"
    else:
        return "mixed"


def get_fib_levels(high, low):
    diff = high - low
    return {
        "0.000": round(low, 2),
        "0.236": round(low + 0.236 * diff, 2),
        "0.382": round(low + 0.382 * diff, 2),
        "0.500": round(low + 0.500 * diff, 2),
        "0.618": round(low + 0.618 * diff, 2),
        "0.786": round(low + 0.786 * diff, 2),
        "1.000": round(high, 2),
    }


def check_structure(df, lookback_candles):
    """
    Compares current price to the high/low of the prior lookback window.
    Returns a plain-language structure label.
    """
    price      = df['close'].iloc[-1]
    prior_high = df['high'].iloc[-(lookback_candles + 1):-1].max()
    prior_low  = df['low'].iloc[-(lookback_candles + 1):-1].min()
    if price > prior_high:
        return "Breaking out / Higher highs"
    elif price < prior_low:
        return "Breaking down / Lower lows"
    else:
        return "Ranging within prior range"


def build_macro_dashboard(df_weekly, df_daily, df_4h):
    """
    Computes all higher-timeframe metrics and returns a formatted
    dashboard string (fed to AI) plus a structured metrics dict.
    """
    # ── Weekly ──
    w_price    = df_weekly['close'].iloc[-1]
    w_ema50    = calculate_ema(df_weekly, 50)
    w_ema200   = calculate_ema(df_weekly, 200)
    w_rsi      = calculate_rsi(df_weekly)
    w_vol_avg  = df_weekly['volume'].tail(20).mean()
    w_vol_now  = df_weekly['volume'].iloc[-1]
    w_structure = check_structure(df_weekly, 8)

    # ── Daily ──
    d_price    = df_daily['close'].iloc[-1]
    d_ema21    = calculate_ema(df_daily, 21)
    d_ema50    = calculate_ema(df_daily, 50)
    d_ema200   = calculate_ema(df_daily, 200)
    d_rsi      = calculate_rsi(df_daily)
    d_structure = check_structure(df_daily, 25)

    # ── 4H (near-term context only) ──
    h4_price = df_4h['close'].iloc[-1]
    h4_ema50 = calculate_ema(df_4h, 50)
    h4_rsi   = calculate_rsi(df_4h)

    # ── Fib levels ──
    # Long-term: last ~8 months of daily data
    long_high = df_daily['high'].tail(250).max()
    long_low  = df_daily['low'].tail(250).min()
    long_fibs = get_fib_levels(long_high, long_low)

    # Short-term: last ~2 months of daily data
    short_high = df_daily['high'].tail(60).max()
    short_low  = df_daily['low'].tail(60).min()
    short_fibs = get_fib_levels(short_high, short_low)

    # ── Weekly RSI zone label (Cowen-style) ──
    if w_rsi >= 70:
        w_rsi_zone = "Overheated (≥70) — late cycle caution"
    elif w_rsi >= 55:
        w_rsi_zone = "Bullish momentum (55–70)"
    elif w_rsi >= 45:
        w_rsi_zone = "Neutral / transitioning (45–55)"
    elif w_rsi >= 30:
        w_rsi_zone = "Bearish momentum (30–45)"
    else:
        w_rsi_zone = "Oversold (≤30) — potential bottom zone"

    fib_str_long  = " | ".join(f"{k}=${v}" for k, v in long_fibs.items()
                                if k in ("0.382", "0.500", "0.618", "0.786"))
    fib_str_short = " | ".join(f"{k}=${v}" for k, v in short_fibs.items()
                                if k in ("0.382", "0.500", "0.618"))

    dashboard = f"""
=== WEEKLY (Macro Regime) ===
Price:       ${w_price:.2f}
EMA50:       ${w_ema50:.2f}
EMA200:      ${w_ema200:.2f}
EMA Status:  {ema_label(w_price, w_ema50, w_ema200)}
RSI(14):     {w_rsi} → {w_rsi_zone}
Structure:   {w_structure}
Volume:      {'Above 20-week avg (momentum)' if w_vol_now > w_vol_avg else 'Below 20-week avg (weak/range)'}

=== DAILY (Intermediate Trend) ===
Price:       ${d_price:.2f}
EMA21:       ${d_ema21:.2f}
EMA50:       ${d_ema50:.2f}
EMA200:      ${d_ema200:.2f}
EMA Status:  {ema_label(d_price, d_ema50, d_ema200)}
RSI(14):     {d_rsi}
Structure:   {d_structure}

=== 4H (Near-term Context Only) ===
Price:       ${h4_price:.2f}
EMA50:       ${h4_ema50:.2f}
RSI(14):     {h4_rsi}

=== FIB LEVELS — Long-term Swing (${long_low:.2f} → ${long_high:.2f}, ~8mo) ===
{fib_str_long}

=== FIB LEVELS — Short-term Swing (${short_low:.2f} → ${short_high:.2f}, ~2mo) ===
{fib_str_short}
"""

    metrics = {
        "w_price":      round(w_price, 2),
        "w_ema50":      round(w_ema50, 2),
        "w_ema200":     round(w_ema200, 2),
        "w_rsi":        w_rsi,
        "w_rsi_zone":   w_rsi_zone,
        "w_structure":  w_structure,
        "w_ema_label":  ema_label(w_price, w_ema50, w_ema200),
        "d_price":      round(d_price, 2),
        "d_ema50":      round(d_ema50, 2),
        "d_ema200":     round(d_ema200, 2),
        "d_rsi":        d_rsi,
        "d_ema_label":  ema_label(d_price, d_ema50, d_ema200),
        "d_structure":  d_structure,
        "long_high":    round(long_high, 2),
        "long_low":     round(long_low, 2),
        "long_fibs":    long_fibs,
        "short_fibs":   short_fibs,
    }

    return dashboard, metrics


# ====================== AI REGIME ANALYSIS ======================
def get_ai_regime(dashboard, metrics):
    """
    Sends the full macro dashboard to Grok for a Cowen-style regime read.
    Returns a structured dict with all report fields.
    """
    def _call():
        prompt = f"""You are a macro crypto analyst in the style of Benjamin Cowen — data-driven, probabilistic, unemotional. Your job is to read the higher timeframe structure of SOL/USDT and deliver a clear regime classification with a concrete 1–3 month outlook. You are NOT a short-term trader. You use weekly and daily charts to understand where we are in the cycle.

You have been given pre-computed TA metrics. Reason through them carefully before forming your conclusion.

{dashboard}

=== ANALYTICAL FRAMEWORK ===

1. WEEKLY RSI is your primary cycle indicator:
   ≥70 → Late cycle / overheated — corrections are sharp
   55–70 → Healthy bull trend — dips are buyable
   45–55 → Transition zone — no conviction either way
   30–45 → Bear market / distribution — rallies get sold
   ≤30 → Capitulation / bottom territory — watch for reversal

2. EMA STACK is your trend confirmation:
   Price > EMA50 > EMA200 = aligned bull trend
   Price < EMA50 < EMA200 = aligned bear trend
   Mixed = choppy, wait for resolution

3. STRUCTURE (weekly more important than daily):
   Breaking out with volume = momentum continuation likely
   Breaking down = sellers in control
   Ranging = accumulation or distribution — watch for the break

4. FIB LEVELS are key macro support/resistance:
   Long-term fibs (8-month swing) define the macro zones
   Short-term fibs are the near-term reaction points

5. REGIME LABELS — choose exactly one:
   "Bull Market — Early Stage"
   "Bull Market — Mid Stage"
   "Bull Market — Late Stage / Overheated"
   "Correction Within Bull Market"
   "Bear Market — Early Stage"
   "Bear Market — Confirmed"
   "Accumulation / Bottom Formation"
   "Chop / Transitioning"

6. INVALIDATION LEVELS define when you are wrong — always include them.

Output ONLY valid JSON in exactly this format — no preamble, no markdown:
{{
  "regime": "one of the regime labels above",
  "confidence": 7.5,
  "summary": "3-4 sentences. What is the dominant story on the higher timeframes right now? Be direct and specific.",
  "outlook_1m": "Most probable path over the next month. Mention specific price levels.",
  "outlook_2_3m": "Longer view — where does this resolve in 2-3 months? What are the key conditions for each path?",
  "near_support": "$XX.XX — $XX.XX",
  "near_resistance": "$XX.XX — $XX.XX",
  "major_support": "$XX.XX — $XX.XX",
  "major_resistance": "$XX.XX — $XX.XX",
  "key_fibs": "0.618=$XX.XX | 0.786=$XX.XX (the most relevant fib levels right now)",
  "scenarios": "• Bull case (XX%): ...\\n• Base case (XX%): ...\\n• Bear case (XX%): ...",
  "bull_invalidation": "$XX.XX — specific level or condition that definitively breaks the bull thesis",
  "bear_invalidation": "$XX.XX — specific level or condition that flips the bear thesis",
  "recommendation": "Concrete guidance for a longer-term holder or swing trader. What should they be doing right now?"
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    return with_retry(_call)


# ====================== MAIN ANALYSIS ======================
def analyze_trend():
    print("📡 Fetching weekly, daily, and 4H data...")
    df_weekly = get_klines("1w", 300)
    df_daily  = get_klines("1d", 500)
    df_4h     = get_klines("4h", 400)

    if df_weekly.empty or df_daily.empty:
        print("❌ Data fetch failed — skipping this run")
        return

    dashboard, metrics = build_macro_dashboard(df_weekly, df_daily, df_4h)

    print(f"\n📊 Macro Dashboard:{dashboard}")
    print("🤖 Querying Grok for regime analysis...")

    report = get_ai_regime(dashboard, metrics)

    # Inject raw metric fields the email needs that aren't in the AI response
    report["weekly_rsi"]       = metrics["w_rsi"]
    report["weekly_ema"]       = metrics["w_ema_label"]
    report["weekly_structure"] = metrics["w_structure"]
    report["daily_rsi"]        = metrics["d_rsi"]
    report["daily_ema"]        = metrics["d_ema_label"]
    report["daily_structure"]  = metrics["d_structure"]

    send_trend_email(report)

    print(f"\n{'='*60}")
    print(f"📈 Regime:     {report['regime']} | Confidence: {report['confidence']}/10")
    print(f"📝 Summary:    {report['summary']}")
    print(f"🗓  1M Outlook: {report['outlook_1m']}")
    print(f"🗓  2-3M:       {report['outlook_2_3m']}")
    print(f"💡 Rec:        {report['recommendation']}")
    print(f"{'='*60}\n")

    # Log every report for historical record keeping
    with open("trend_reports.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price":     metrics["w_price"],
            **report
        }) + "\n")

    return report


# ====================== MAIN LOOP ======================
if __name__ == "__main__":
    print("🚀 SOL Macro Trend Agent started")
    print(f"   Runs every {CHECK_INTERVAL_HOURS} hours | Model: {MODEL}\n")

    while True:
        try:
            analyze_trend()
        except Exception as e:
            print(f"❌ Trend Agent error: {e}")

        print(f"⏳ Next run in {CHECK_INTERVAL_HOURS} hours...")
        time.sleep(CHECK_INTERVAL_HOURS * 3600)
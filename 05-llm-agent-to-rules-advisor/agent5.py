# Copyright (c) 2026 James Staples. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, or derivative works are prohibited.
# No license is granted for training machine learning models on this code.
import os
import re
import json
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import requests
import feedparser
import base64
import smtplib
from email.mime.text import MIMEText
from solders.transaction import VersionedTransaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client as SolanaClient
from binance.client import Client as BinanceClient
from spl.token.instructions import get_associated_token_address
import atexit
import numpy as np

load_dotenv()

# ====================== CONFIG ======================
LIVE_TRADING           = False
CHECK_INTERVAL         = 900          # seconds between loops (15 min)
MODEL                  = "grok-4.20-beta-0309-reasoning"
MEMORY_FILE            = "agent5_memory.json"
POSITION_STATE_FILE    = "live_position_agent5.json"
LOCK_FILE              = "agent5.lock"
UNCONFIRMED_LOG        = "unconfirmed_exits.log"
DECISIONS_FILE         = "agent5_decisions.jsonl"

JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"
JUPITER_HEADERS   = {"x-api-key": os.getenv("JUPITER_API_KEY", "")}
SOL_RESERVE       = 0.07
SLIPPAGE_BPS      = 100

# ── Self-backtest config ────────────────────────────────────────────────────
OUTCOME_LOOKFORWARD_BARS = 12   # 12 × 4H = 48 hours forward window
MIN_OUTCOMES_FOR_STATS   = 8

# ── Regime-conditional hold limits ──────────────────────────────────────────
MAX_HOLD_HOURS = {
    "bear":         36,
    "chop":         48,
    "bull_chop":    72,
    "bull":        120,
    "strong_bull": 168,
}
DEFAULT_MAX_HOLD = 72

# ── Regime-conditional trail multiplier ─────────────────────────────────────
TRAIL_MULTIPLIER = {
    "bear":         1.2,
    "chop":         1.4,
    "bull_chop":    1.6,
    "bull":         1.8,
    "strong_bull":  2.2,
}
DEFAULT_TRAIL = 1.4

# ── Regime-conditional BE trigger multiplier (× risk) ───────────────────────
# Bear/chop: promote to BE early to protect capital
# Bull: give it room — don't get stopped at BE on a normal pullback
BE_TRIGGER_MULT = {
    "bear":         0.5,
    "chop":         0.6,
    "bull_chop":    0.7,
    "bull":         1.0,
    "strong_bull":  1.2,
}
DEFAULT_BE_MULT = 0.7

CIRCUIT_BREAKER_LOSSES = 4

SOL_MINT  = Pubkey.from_string("So11111111111111111111111111111111111111112")
USDT_MINT = Pubkey.from_string("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")

FOMC_DATES = [
    "2026-03-17", "2026-03-18", "2026-04-28", "2026-04-29",
    "2026-06-16", "2026-06-17", "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16", "2026-10-27", "2026-10-28",
    "2026-12-08", "2026-12-09",
]

solana_client  = None
keypair        = None
wallet_pubkey  = None
binance_client = None

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.x.ai/v1")


# ====================== LOCK ======================
def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                ts = float(f.read().strip())
            if time.time() - ts > 600:
                print(f"⚠️  Removing stale lock ({time.time() - ts:.0f}s old)")
                os.remove(LOCK_FILE)
            else:
                print("❌ Another instance running. Exiting.")
                exit(1)
        except Exception:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
    with open(LOCK_FILE, "w") as f:
        f.write(str(time.time()))
    print("✅ Lock acquired — Agent5 running")


def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


atexit.register(release_lock)


# ====================== WALLET / SWAP ======================
def get_token_balance(mint):
    if mint == SOL_MINT:
        try:
            return solana_client.get_balance(wallet_pubkey).value / 1_000_000_000
        except Exception as e:
            print(f"Balance error (SOL): {e}")
            return 0.0
    else:
        ata = get_associated_token_address(wallet_pubkey, mint)
        try:
            resp = solana_client.get_token_account_balance(ata)
            return int(resp.value.amount) / (10 ** resp.value.decimals)
        except Exception as e:
            print(f"Balance error (USDT): {e}")
            return 0.0


def get_jupiter_quote(input_mint, output_mint, amount):
    params = {
        "inputMint":   str(input_mint),
        "outputMint":  str(output_mint),
        "amount":      str(int(amount)),
        "slippageBps": str(SLIPPAGE_BPS),
    }
    r = requests.get(JUPITER_QUOTE_URL, params=params, headers=JUPITER_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def get_swap_transaction(quote):
    payload = {
        "quoteResponse":    quote,
        "userPublicKey":    str(wallet_pubkey),
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "priorityLevel": "veryHigh",
                "maxLamports":   5000000,
            }
        },
    }
    r = requests.post(JUPITER_SWAP_URL, json=payload, headers=JUPITER_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()["swapTransaction"]


def execute_swap(swap_tx_b64, direction="buy"):
    usdt_before = get_token_balance(USDT_MINT)
    sol_before  = get_token_balance(SOL_MINT)
    try:
        swap_tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(swap_tx_bytes)
        tx = VersionedTransaction(tx.message, [keypair])
        from solana.rpc.types import TxOpts
        resp = solana_client.send_raw_transaction(
            bytes(tx),
            opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")
        )
        sig = resp.value
        print(f"✅ TX SENT → https://solscan.io/tx/{sig}")

        for i in range(10):
            time.sleep(2)
            try:
                status_resp = solana_client.get_signature_statuses([sig], search_transaction_history=True)
                status = status_resp.value[0] if status_resp.value else None
                if status and status.confirmation_status in ("confirmed", "finalized"):
                    print(f"✅ TX CONFIRMED ({status.confirmation_status}) after {(i+1)*2}s")
                    return sig, True
            except Exception:
                pass

        print("⏳ Polling timed out — verifying via balance...")
        time.sleep(5)
        usdt_after = get_token_balance(USDT_MINT)
        sol_after  = get_token_balance(SOL_MINT)
        print(f"   Before: USDT ${usdt_before:.2f} | SOL {sol_before:.6f}")
        print(f"   After:  USDT ${usdt_after:.2f} | SOL {sol_after:.6f}")
        if direction == "buy" and sol_after > sol_before + 0.1:
            print(f"✅ Balance confirms BUY: {sig}")
            return sig, True
        elif direction == "sell" and (usdt_after > usdt_before + 1.0 or sol_after <= SOL_RESERVE + 0.05):
            print(f"✅ Balance confirms SELL: {sig}")
            return sig, True

        print(f"⚠️  Could not confirm after all checks: {sig}")
        if direction == "sell":
            log_unconfirmed_sell(sig, "Polling + balance fallback failed")
        return sig, False

    except Exception as e:
        print(f"Swap execution failed: {e}")
        return None, False


def robust_buy(amount_usdt):
    spend = amount_usdt * 0.99
    for attempt in range(7):
        try:
            print(f"🔄 BUY attempt {attempt + 1}/7...")
            quote  = get_jupiter_quote(USDT_MINT, SOL_MINT, int(spend * 1_000_000))
            tx_b64 = get_swap_transaction(quote)
            return execute_swap(tx_b64, direction="buy")
        except Exception as e:
            print(f"⚠️  Buy attempt {attempt + 1} failed: {e}")
            if attempt < 6:
                time.sleep(3)
    print("❌ All buy attempts failed — skipping this setup")
    return None, False


def robust_sell():
    for attempt in range(7):
        try:
            print(f"🔄 SELL attempt {attempt + 1}/7...")
            fresh_sol = get_token_balance(SOL_MINT)
            if fresh_sol <= SOL_RESERVE:
                print("⚠️  Not enough SOL to sell after reserve")
                return None, False
            amount_lamports = int((fresh_sol - SOL_RESERVE) * 1_000_000_000)
            quote  = get_jupiter_quote(SOL_MINT, USDT_MINT, amount_lamports)
            tx_b64 = get_swap_transaction(quote)
            return execute_swap(tx_b64, direction="sell")
        except Exception as e:
            print(f"⚠️  Sell attempt {attempt + 1} failed: {e}")
            if attempt < 6:
                time.sleep(3)
    print("❌ All sell attempts failed")
    return None, False


def log_unconfirmed_sell(sig, context):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sig":       str(sig),
        "context":   context,
        "solscan":   f"https://solscan.io/tx/{sig}",
    }
    with open(UNCONFIRMED_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("⚠️  Unconfirmed sell logged — MANUAL CHECK REQUIRED")


def send_alert(subject, body):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASS]):
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("✅ Gmail alert sent")
    except Exception as e:
        print(f"Email failed: {e}")


# ====================== TA ======================
def get_klines_df(interval, limit):
    """Fetch klines, chunking if needed (Binance caps 4H at ~500 per call)."""
    try:
        max_per_call = 500
        if limit <= max_per_call:
            k = binance_client.get_klines(symbol='SOLUSDT', interval=interval, limit=limit)
        else:
            # Chunk fetch for large requests
            k = []
            remaining = limit
            end_time = None
            while remaining > 0:
                batch = min(remaining, max_per_call)
                kwargs = {"symbol": "SOLUSDT", "interval": interval, "limit": batch}
                if end_time:
                    kwargs["endTime"] = end_time
                chunk = binance_client.get_klines(**kwargs)
                if not chunk:
                    break
                k = chunk + k  # prepend — older data first
                end_time = chunk[0][0] - 1
                remaining -= len(chunk)
                if len(chunk) < batch:
                    break
            # Deduplicate by open time
            seen = set()
            deduped = []
            for candle in k:
                if candle[0] not in seen:
                    seen.add(candle[0])
                    deduped.append(candle)
            k = sorted(deduped, key=lambda x: x[0])

        df = pd.DataFrame(k, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Klines error ({interval}): {e}")
        return pd.DataFrame()


def get_klines_df_indexed(interval, limit):
    """Like get_klines_df but returns a tz-aware DatetimeIndex version for backtesting."""
    df = get_klines_df(interval, limit)
    if df.empty:
        return df
    df.index = pd.DatetimeIndex(df['timestamp']).tz_localize('UTC')
    return df


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
    return round(rsi.iloc[-1], 1) if not rsi.empty else 50.0


def detect_rsi_divergence(df, lookback=25):
    if len(df) < lookback + 14:
        return "none"

    close  = df['close'].iloc[-lookback:].reset_index(drop=True)
    volume = df['volume'].iloc[-lookback:].reset_index(drop=True)

    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    rsi_full = 100 - (100 / (1 + rs))
    rsi      = rsi_full.iloc[-lookback:].reset_index(drop=True)

    def find_swing_lows(series, window=4):
        lows = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i - window:i + window + 1].min():
                lows.append((i, float(series.iloc[i])))
        deduped = []
        for idx, val in lows:
            if deduped and idx - deduped[-1][0] <= window:
                if val < deduped[-1][1]:
                    deduped[-1] = (idx, val)
            else:
                deduped.append((idx, val))
        return deduped

    def find_swing_highs(series, window=4):
        highs = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i - window:i + window + 1].max():
                highs.append((i, float(series.iloc[i])))
        deduped = []
        for idx, val in highs:
            if deduped and idx - deduped[-1][0] <= window:
                if val > deduped[-1][1]:
                    deduped[-1] = (idx, val)
            else:
                deduped.append((idx, val))
        return deduped

    p_lows  = find_swing_lows(close)
    p_highs = find_swing_highs(close)
    r_lows  = find_swing_lows(rsi)
    r_highs = find_swing_highs(rsi)

    if len(p_lows) >= 2 and len(r_lows) >= 2:
        pl1, pl2 = p_lows[-2], p_lows[-1]
        rl1, rl2 = r_lows[-2], r_lows[-1]
        vol_ok   = volume.iloc[pl2[0]] > volume.iloc[pl1[0]]
        if pl2[1] < pl1[1] and rl2[1] > rl1[1] and vol_ok:
            return "regular_bullish"
        if pl2[1] > pl1[1] and rl2[1] < rl1[1] and vol_ok:
            return "hidden_bullish"

    if len(p_highs) >= 2 and len(r_highs) >= 2:
        ph1, ph2 = p_highs[-2], p_highs[-1]
        rh1, rh2 = r_highs[-2], r_highs[-1]
        vol_ok   = volume.iloc[ph2[0]] > volume.iloc[ph1[0]]
        if ph2[1] > ph1[1] and rh2[1] < rh1[1] and vol_ok:
            return "regular_bearish"
        if ph2[1] < ph1[1] and rh2[1] > rh1[1] and vol_ok:
            return "hidden_bearish"

    return "none"


def calculate_macd(df):
    """Returns (cross_str, hist_value, is_improving) for both display and scoring."""
    ema12     = df['close'].ewm(span=12, adjust=False).mean()
    ema26     = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - signal
    last, prev         = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_last, sig_prev = signal.iloc[-1], signal.iloc[-2]
    if last > sig_last and prev <= sig_prev:
        cross = "bullish_cross"
    elif last < sig_last and prev >= sig_prev:
        cross = "bearish_cross"
    else:
        cross = "no_cross"
    improving = hist.iloc[-1] > hist.iloc[-2]
    return cross, round(hist.iloc[-1], 4), improving


def calculate_macd_display(df):
    """Human-readable MACD string for display."""
    cross, hist_val, improving = calculate_macd(df)
    return f"{cross.replace('_', ' ')} | hist {hist_val:.4f} ({'improving' if improving else 'deteriorating'})"


def calculate_ema_status(df):
    ema50  = df['close'].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    price  = df['close'].iloc[-1]
    if price > ema50 > ema200:
        return "above_both_bullish"
    elif price < ema50 < ema200:
        return "below_both_bearish"
    elif price > ema50 and ema50 < ema200:
        return "above_ema50_mixed"
    return "mixed"


def calculate_atr(df, period=14):
    high_low   = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close  = np.abs(df['low']  - df['close'].shift())
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return round(tr.rolling(window=period).mean().iloc[-1], 2)


def get_fib_levels(df, tail=120):
    recent = df.tail(tail)
    high   = recent['high'].max()
    low    = recent['low'].min()
    diff   = high - low
    levels = {
        "0.236": round(low + 0.236 * diff, 2),
        "0.382": round(low + 0.382 * diff, 2),
        "0.500": round(low + 0.500 * diff, 2),
        "0.618": round(low + 0.618 * diff, 2),
        "0.786": round(low + 0.786 * diff, 2),
    }
    price   = df['close'].iloc[-1]
    closest = min(levels, key=lambda k: abs(levels[k] - price))
    fib_str = " | ".join(f"{k}=${v}" for k, v in levels.items())
    return f"nearest={closest} (${levels[closest]}) | {fib_str}", levels, closest


def get_macro_regime(df_daily, df_weekly):
    w_price  = df_weekly['close'].iloc[-1]
    w_ema50  = df_weekly['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    w_rsi    = calculate_rsi(df_weekly)

    d_price  = df_daily['close'].iloc[-1]
    d_ema50  = df_daily['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    d_ema200 = df_daily['close'].ewm(span=200, adjust=False).mean().iloc[-1]

    bull_weekly = w_price > w_ema50 and w_rsi > 50
    bear_weekly = w_price < w_ema50 and w_rsi < 50
    bull_daily  = d_price > d_ema50 > d_ema200
    bear_daily  = d_price < d_ema50 < d_ema200

    if bull_weekly and bull_daily and w_rsi > 60:
        return "strong_bull"
    elif bull_weekly and bull_daily:
        return "bull"
    elif bull_weekly and not bear_daily:
        return "bull_chop"
    elif bear_weekly and bear_daily:
        return "bear"
    else:
        return "chop"


# ====================== CODE-COMPUTED CONVICTION ======================
def compute_conviction(regime, div_4h, div_1h, macd_cross, macd_improving,
                       ema_status, fib_closest, price, fib_levels, is_fomc):
    """
    Deterministic conviction score computed in Python — the AI cannot override this.
    Returns (score, breakdown_str) so we can show exactly why.
    """
    score = 5.0
    reasons = []

    # ── Positive signals ────────────────────────────────────────────────
    # Multi-TF Fib confluence: price within 1% of a key Fib on 4H
    if fib_levels:
        nearest_val = fib_levels.get(fib_closest, 0)
        if nearest_val and abs(price - nearest_val) / price < 0.01:
            score += 1.5
            reasons.append(f"+1.5 near Fib {fib_closest} (${nearest_val})")

    # 4H RSI divergence
    if div_4h in ("regular_bullish", "hidden_bullish"):
        score += 1.5
        reasons.append(f"+1.5 4H {div_4h}")
    elif div_4h in ("regular_bearish", "hidden_bearish"):
        score -= 1.0
        reasons.append(f"-1.0 4H {div_4h}")

    # MACD cross
    if macd_cross == "bullish_cross":
        score += 1.0
        reasons.append("+1.0 4H MACD bullish cross")
    elif macd_cross == "bearish_cross":
        score -= 1.0
        reasons.append("-1.0 4H MACD bearish cross")
    elif macd_improving:
        score += 0.5
        reasons.append("+0.5 MACD histogram improving")

    # EMA position
    if ema_status == "above_both_bullish":
        score += 1.0
        reasons.append("+1.0 above both daily EMAs")
    elif ema_status == "below_both_bearish":
        score -= 1.5
        reasons.append("-1.5 below both daily EMAs")

    # 1H divergence confirmation
    if div_1h in ("regular_bullish", "hidden_bullish"):
        score += 0.5
        reasons.append(f"+0.5 1H confirms {div_1h}")
    elif div_1h in ("regular_bearish", "hidden_bearish"):
        score -= 0.5
        reasons.append(f"-0.5 1H {div_1h}")

    # ── Regime penalties ────────────────────────────────────────────────
    if regime == "bear":
        score -= 3.0
        reasons.append("-3.0 BEAR regime")
    elif regime == "chop":
        score -= 1.0
        reasons.append("-1.0 CHOP regime")

    # ── FOMC ────────────────────────────────────────────────────────────
    if is_fomc:
        score -= 1.0
        reasons.append("-1.0 FOMC window")

    score = max(0, min(10, round(score, 1)))
    breakdown = " | ".join(reasons)
    return score, breakdown


# ====================== SELF-BACKTEST ======================
def get_extended_4h_history():
    """Fetch extended 4H history for backtesting (2000 candles ≈ 333 days)."""
    print("   📈 Fetching extended 4H history for backtest...")
    return get_klines_df_indexed('4h', 2000)


def compute_signal_fingerprint(regime, div_4h, macd_cross, ema_status):
    """Create a hashable fingerprint of the current signal combination."""
    return f"{regime}+div={div_4h}+macd={macd_cross}+ema={ema_status}"

def classify_divergence_series(df_4h, rsi, lookback=25, swing_window=4):
    """
    Pre-compute RSI divergence classification for every bar in df_4h.
    Returns a list of strings same length as df_4h.
    """
    close  = df_4h['close'].values
    volume = df_4h['volume'].values
    rsi_v  = rsi.values
    n      = len(df_4h)
    result = ["none"] * n

    for i in range(lookback + 14, n):
        seg_c = close[i - lookback:i + 1]
        seg_r = rsi_v[i - lookback:i + 1]
        seg_v = volume[i - lookback:i + 1]
        slen  = len(seg_c)

        # Find swing lows
        p_lows, r_lows = [], []
        for j in range(swing_window, slen - swing_window):
            if seg_c[j] == min(seg_c[j - swing_window:j + swing_window + 1]):
                if p_lows and j - p_lows[-1][0] <= swing_window:
                    if seg_c[j] < p_lows[-1][1]:
                        p_lows[-1] = (j, seg_c[j])
                else:
                    p_lows.append((j, seg_c[j]))
            if seg_r[j] == min(seg_r[j - swing_window:j + swing_window + 1]):
                if r_lows and j - r_lows[-1][0] <= swing_window:
                    if seg_r[j] < r_lows[-1][1]:
                        r_lows[-1] = (j, seg_r[j])
                else:
                    r_lows.append((j, seg_r[j]))

        # Find swing highs
        p_highs, r_highs = [], []
        for j in range(swing_window, slen - swing_window):
            if seg_c[j] == max(seg_c[j - swing_window:j + swing_window + 1]):
                if p_highs and j - p_highs[-1][0] <= swing_window:
                    if seg_c[j] > p_highs[-1][1]:
                        p_highs[-1] = (j, seg_c[j])
                else:
                    p_highs.append((j, seg_c[j]))
            if seg_r[j] == max(seg_r[j - swing_window:j + swing_window + 1]):
                if r_highs and j - r_highs[-1][0] <= swing_window:
                    if seg_r[j] > r_highs[-1][1]:
                        r_highs[-1] = (j, seg_r[j])
                else:
                    r_highs.append((j, seg_r[j]))

        # Check bullish divergences
        if len(p_lows) >= 2 and len(r_lows) >= 2:
            pl1, pl2 = p_lows[-2], p_lows[-1]
            rl1, rl2 = r_lows[-2], r_lows[-1]
            vol_ok = seg_v[pl2[0]] > seg_v[pl1[0]]
            if pl2[1] < pl1[1] and rl2[1] > rl1[1] and vol_ok:
                result[i] = "regular_bullish"
                continue
            if pl2[1] > pl1[1] and rl2[1] < rl1[1] and vol_ok:
                result[i] = "hidden_bullish"
                continue

        # Check bearish divergences
        if len(p_highs) >= 2 and len(r_highs) >= 2:
            ph1, ph2 = p_highs[-2], p_highs[-1]
            rh1, rh2 = r_highs[-2], r_highs[-1]
            vol_ok = seg_v[ph2[0]] > seg_v[ph1[0]]
            if ph2[1] > ph1[1] and rh2[1] < rh1[1] and vol_ok:
                result[i] = "regular_bearish"
                continue
            if ph2[1] < ph1[1] and rh2[1] > rh1[1] and vol_ok:
                result[i] = "hidden_bearish"
                continue

    return result


def compute_historical_edge(df_4h, regime, div_4h, macd_cross, ema_status):
    """
    Walk through historical 4H candles, find bars where the same signal
    combination occurred, and compute forward returns.
    Returns a summary string for the AI prompt.
    """
    if df_4h.empty or len(df_4h) < 300:
        return "Insufficient history for edge computation"

    try:
        fwd_bars = OUTCOME_LOOKFORWARD_BARS
        fingerprint = compute_signal_fingerprint(regime, div_4h, macd_cross, ema_status)

        # Pre-compute indicators for the full history
        close = df_4h['close']
        delta = close.diff()
        gain  = delta.where(delta > 0, 0)
        loss  = -delta.where(delta < 0, 0)
        avg_g = gain.ewm(com=13, min_periods=14).mean()
        avg_l = loss.ewm(com=13, min_periods=14).mean()
        rs    = avg_g / avg_l
        rsi   = 100 - (100 / (1 + rs))

        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal    = macd_line.ewm(span=9, adjust=False).mean()

        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        # Pre-compute divergence for all bars
        print("   🔍 Computing historical edge...")
        div_series = classify_divergence_series(df_4h, rsi)

        matches       = []
        long_wins     = 0
        long_total    = 0

        # Walk from bar 200 to (end - fwd_bars) to have enough lookback + lookforward
        for i in range(200, len(df_4h) - fwd_bars):
            # Classify regime at bar i
            p = close.iloc[i]
            e50  = ema50.iloc[i]
            e200 = ema200.iloc[i]
            r    = rsi.iloc[i]

            if p > e50 > e200 and r > 60:
                bar_regime = "strong_bull"
            elif p > e50 > e200:
                bar_regime = "bull"
            elif p > e50:
                bar_regime = "bull_chop"
            elif p < e50 < e200:
                bar_regime = "bear"
            else:
                bar_regime = "chop"

            # Classify MACD cross at bar i
            ml, ml_prev = macd_line.iloc[i], macd_line.iloc[i-1]
            sl, sl_prev = signal.iloc[i], signal.iloc[i-1]
            if ml > sl and ml_prev <= sl_prev:
                bar_macd = "bullish_cross"
            elif ml < sl and ml_prev >= sl_prev:
                bar_macd = "bearish_cross"
            else:
                bar_macd = "no_cross"

            # Classify EMA status
            if p > e50 > e200:
                bar_ema = "above_both_bullish"
            elif p < e50 < e200:
                bar_ema = "below_both_bearish"
            else:
                bar_ema = "mixed"

            # Divergence from pre-computed series
            bar_div = div_series[i]

            bar_fp = compute_signal_fingerprint(bar_regime, bar_div, bar_macd, bar_ema)

            if bar_fp == fingerprint:
                # Compute forward returns
                entry = close.iloc[i]
                fwd   = df_4h.iloc[i+1:i+1+fwd_bars]
                high_pct  = ((fwd['high'].max() - entry) / entry) * 100
                low_pct   = ((fwd['low'].min() - entry) / entry) * 100
                close_pct = ((fwd['close'].iloc[-1] - entry) / entry) * 100

                long_win = high_pct >= 1.5
                long_total += 1
                long_wins  += int(long_win)

                matches.append({
                    "high_pct":  round(high_pct, 2),
                    "low_pct":   round(low_pct, 2),
                    "close_pct": round(close_pct, 2),
                })

        if not matches:
            return f"No historical matches for {fingerprint}"

        avg_high  = round(np.mean([m['high_pct'] for m in matches]), 2)
        avg_low   = round(np.mean([m['low_pct'] for m in matches]), 2)
        avg_close = round(np.mean([m['close_pct'] for m in matches]), 2)
        long_wr   = round(long_wins / long_total * 100) if long_total else 0

        edge_str = (
            f"Historical edge ({long_total} matches | {fingerprint} | fwd={fwd_bars*4}h):\n"
            f"  LONG WR: {long_wr}% | avg high: +{avg_high}% | avg low: {avg_low}% | avg close: {avg_close:+.2f}%"
        )
        print(f"   🔍 {edge_str}")
        return edge_str

    except Exception as e:
        print(f"   ⚠️  Edge computation error: {e}")
        return f"Edge computation error: {e}"


# ====================== OUTCOME TAGGING ======================
def tag_outcomes_in_decisions(df_4h):
    """
    Tag forward-looking outcomes on past decisions for self-evaluation.
    Computes four cumulative forward windows (12h, 24h, 36h, 48h) on each decision.
    Existing field names (outcome_high_pct etc.) represent the 48h window for back-compat.
    Suffixed fields added: _12h, _24h, _36h.
    
    Also captures the per-4H-bar forward price path (12 bars of high/low/close),
    enabling time-to-high analysis, drawdown shape, dynamic TP, and trailing stop
    research without affecting any decision logic.
    
    Path fields: outcome_path_h_4h..h_48h, outcome_path_l_4h..l_48h, outcome_path_c_4h..c_48h
    """
    if not os.path.exists(DECISIONS_FILE):
        return

    # Forward windows in 4H bars: 3 bars = 12h, 6 = 24h, 9 = 36h, 12 = 48h
    WINDOWS = {"12h": 3, "24h": 6, "36h": 9, "48h": 12}

    try:
        with open(DECISIONS_FILE, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return

        records  = [json.loads(l) for l in lines]
        modified = False
        newly_tagged = 0
        path_added_count = 0

        for rec in records:
            # If already fully tagged with all 4 windows AND the path data, skip
            already_full = (
                rec.get("outcome_tagged") and
                "outcome_high_pct_12h" in rec and
                "outcome_high_pct_24h" in rec and
                "outcome_high_pct_36h" in rec and
                "outcome_path_h_48h" in rec  # NEW: also requires path data present
            )
            if already_full:
                continue

            sig_time_str = rec.get("timestamp")
            if not sig_time_str:
                continue

            sig_time = pd.Timestamp(sig_time_str)
            if sig_time.tzinfo is None:
                sig_time = sig_time.tz_localize("UTC")
            else:
                sig_time = sig_time.tz_convert("UTC")

            # Match resolution to index to avoid "Cannot losslessly convert units"
            sig_time = sig_time.as_unit(df_4h.index.dtype.str.split('[')[-1].rstrip(']'))

            idx_arr = df_4h.index.searchsorted(sig_time)
            if idx_arr < 1:
                continue

            fwd_start = idx_arr
            fwd_end_full = fwd_start + OUTCOME_LOOKFORWARD_BARS
            if fwd_end_full > len(df_4h):
                pass  # might still tag shorter windows below

            entry_price = rec.get("price")
            if not entry_price or entry_price <= 0:
                continue

            rec_modified = False

            # ── Existing cumulative-window logic (unchanged) ──
            for label, bars in WINDOWS.items():
                end = fwd_start + bars
                if end > len(df_4h):
                    continue
                field_suffix = "" if label == "48h" else f"_{label}"
                if f"outcome_high_pct{field_suffix}" in rec:
                    continue
                fwd_slice = df_4h.iloc[fwd_start:end]
                high_pct  = round((fwd_slice['high'].max()  - entry_price) / entry_price * 100, 3)
                low_pct   = round((fwd_slice['low'].min()   - entry_price) / entry_price * 100, 3)
                close_pct = round((fwd_slice['close'].iloc[-1] - entry_price) / entry_price * 100, 3)
                rec[f"outcome_high_pct{field_suffix}"]  = high_pct
                rec[f"outcome_low_pct{field_suffix}"]   = low_pct
                rec[f"outcome_close_pct{field_suffix}"] = close_pct
                rec_modified = True

            # ── NEW: per-bar forward path capture ──
            # Only run if we have the full 48h window AND haven't already captured.
            # Captures bar-level OHLC%-from-entry at +4h, +8h, ..., +48h.
            if fwd_end_full <= len(df_4h) and "outcome_path_h_48h" not in rec:
                fwd_full = df_4h.iloc[fwd_start:fwd_end_full]
                # 12 bars expected. Each bar i represents the +(i+1)*4h timestamp.
                for i in range(OUTCOME_LOOKFORWARD_BARS):
                    bar = fwd_full.iloc[i]
                    hours = (i + 1) * 4
                    rec[f"outcome_path_h_{hours}h"] = round(
                        (bar['high'] - entry_price) / entry_price * 100, 3
                    )
                    rec[f"outcome_path_l_{hours}h"] = round(
                        (bar['low'] - entry_price) / entry_price * 100, 3
                    )
                    rec[f"outcome_path_c_{hours}h"] = round(
                        (bar['close'] - entry_price) / entry_price * 100, 3
                    )
                path_added_count += 1
                rec_modified = True

            # Mark outcome_tagged=True only when the 48h window is filled
            if "outcome_close_pct" in rec and not rec.get("outcome_tagged"):
                rec["outcome_tagged"] = True
                newly_tagged += 1
                rec_modified = True

            if rec_modified:
                modified = True

        if modified:
            with open(DECISIONS_FILE, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            tagged_count = sum(1 for r in records if r.get("outcome_tagged"))
            with_path = sum(1 for r in records if "outcome_path_h_48h" in r)
            print(f"   📊 Outcome tagging: {tagged_count} total tagged (48h), "
                  f"{newly_tagged} newly tagged, {path_added_count} forward paths added "
                  f"(total with path: {with_path})")

    except Exception as e:
        print(f"   ⚠️  Outcome tagging error: {e}")
        

def get_signal_performance_summary():
    """Reads tagged outcomes and computes win-rate stats for the AI prompt."""
    if not os.path.exists(DECISIONS_FILE):
        return "No decision history yet — building track record."

    try:
        with open(DECISIONS_FILE, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]

        tagged = [r for r in records if r.get("outcome_tagged")]
        if len(tagged) < MIN_OUTCOMES_FOR_STATS:
            return f"Track record building ({len(tagged)} outcomes so far — need {MIN_OUTCOMES_FOR_STATS}+ for stats)."

        stats = {}
        for r in tagged:
            regime = r.get("regime", "unknown")
            action = r.get("action", "stay").lower()
            if action == "stay":
                continue

            high_pct  = r.get("outcome_high_pct", 0)
            close_pct = r.get("outcome_close_pct", 0)

            win = high_pct >= 1.5 if action == "buy" else False

            key = (regime, action)
            if key not in stats:
                stats[key] = {"wins": 0, "total": 0, "avg_close": []}
            stats[key]["wins"]  += int(win)
            stats[key]["total"] += 1
            stats[key]["avg_close"].append(close_pct)

        if not stats:
            return "No actionable signals with outcomes yet."

        lines = []
        for (regime, action), s in sorted(stats.items()):
            wr      = round(s["wins"] / s["total"] * 100)
            avg_cls = round(sum(s["avg_close"]) / len(s["avg_close"]), 2)
            n       = s["total"]
            quality = "✅" if wr >= 55 else "⚠️" if wr >= 45 else "❌"
            lines.append(
                f"  {quality} {regime.upper()} {action.upper()}: "
                f"{n} signals | WR {wr}% | avg net {avg_cls:+.2f}% over {OUTCOME_LOOKFORWARD_BARS*4}h"
            )

        total_dir = sum(s["total"] for s in stats.values())
        total_win = sum(s["wins"]  for s in stats.values())
        overall   = round(total_win / total_dir * 100) if total_dir else 0

        return (
            f"Signal track record ({total_dir} directional, {OUTCOME_LOOKFORWARD_BARS*4}h window):\n"
            + "\n".join(lines)
            + f"\n  Overall directional WR: {overall}%"
        )

    except Exception as e:
        return f"Performance summary error: {e}"


def log_decision(action, conviction, regime, price, ta_signals, edge_str,
                 signals=None, funding_oi="", fear_greed="", atr=0, btc_change=0,
                 ai_thought="", ai_reason="", ai_action_raw="", ai_called=False, ai_error=""):
    """Log every decision (BUY/SELL/STAY) for outcome tagging later."""
    rec = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "action":     action,
        "conviction": conviction,
        "regime":     regime,
        "price":      price,
        "signals":    ta_signals,
        "edge":       edge_str,
        # ── AI reasoning capture (May 2026 addition) ──
        "ai_called":     ai_called,        # was AI invoked this cycle (conviction >= 6.5)
        "ai_action_raw": ai_action_raw,    # what AI returned BEFORE code overrides
        "ai_thought":    ai_thought,       # AI's reasoning narrative
        "ai_reason":     ai_reason,        # AI's stated reason for action
        "ai_error":      ai_error,         # error string if AI call failed, empty otherwise
        # ── Extended data for future analysis ──
        "rsi_4h":     round(signals.get("rsi_4h", 0), 1) if signals else 0,
        # ... (rest unchanged)
        "rsi_1h":     round(signals.get("rsi_1h", 0), 1) if signals else 0,
        "div_4h":     signals.get("div_4h", "none") if signals else "none",
        "div_1h":     signals.get("div_1h", "none") if signals else "none",
        "macd_cross": signals.get("macd_cross", "no_cross") if signals else "no_cross",
        "macd_hist":  round(signals.get("macd_hist", 0), 4) if signals else 0,
        "ema_status": signals.get("ema_status", "mixed") if signals else "mixed",
        "fib_closest": signals.get("fib_closest", "") if signals else "",
        "fib_distance_pct": round(signals.get("fib_distance_pct", 0), 3) if signals else 0,
        "distance_from_daily_ema50_pct": round(signals.get("distance_from_daily_ema50_pct", 0), 3) if signals else 0,
        "vol_ratio_4h": round(signals.get("vol_ratio_4h", 1.0), 3) if signals else 1.0,
        "vol_ratio_1h": round(signals.get("vol_ratio_1h", 1.0), 3) if signals else 1.0,
        "vol_4h_raw":   round(signals.get("vol_4h_raw", 0), 2) if signals else 0,
        "atr":        round(atr, 4),
        "funding_oi": funding_oi,
        "fear_greed": fear_greed,
        "btc_change": round(btc_change, 2),
    }
    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ====================== DATA CONTEXT ======================
def get_ta_summary():
    """Returns (summary_str, regime, atr, structured_signals_dict)."""
    try:
        df_1h     = get_klines_df('1h', 500)
        df_4h     = get_klines_df('4h', 400)
        df_daily  = get_klines_df('1d', 200)
        df_weekly = get_klines_df('1w', 100)

        if df_1h.empty or df_4h.empty:
            return "TA unavailable", "chop", 2.0, {}

        regime = get_macro_regime(df_daily, df_weekly) if not df_daily.empty and not df_weekly.empty else "chop"
        atr    = calculate_atr(df_4h)

        # Structured signals for conviction scoring
        div_4h = detect_rsi_divergence(df_4h)
        div_1h = detect_rsi_divergence(df_1h, lookback=18)
        macd_cross, macd_hist, macd_improving = calculate_macd(df_4h)
        ema_status = calculate_ema_status(df_4h)
        fib_str, fib_levels, fib_closest = get_fib_levels(df_4h)
        rsi_4h = calculate_rsi(df_4h)
        rsi_1h = calculate_rsi(df_1h)

        # Volume — use LAST CLOSED bar (iloc[-2]) for consistent readings
        # regardless of when in the bar cycle the decision fires.
        # Current in-progress bar would give inconsistent partial-bar values.
        vol_4h       = float(df_4h['volume'].iloc[-2]) if len(df_4h) >= 2 else 0.0
        vol_4h_ma20  = float(df_4h['volume'].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1]) if len(df_4h) >= 2 else 0.0
        vol_ratio_4h = round(vol_4h / vol_4h_ma20, 3) if vol_4h_ma20 > 0 else 1.0
        vol_1h       = float(df_1h['volume'].iloc[-2]) if len(df_1h) >= 2 else 0.0
        vol_1h_ma20  = float(df_1h['volume'].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1]) if len(df_1h) >= 2 else 0.0
        vol_ratio_1h = round(vol_1h / vol_1h_ma20, 3) if vol_1h_ma20 > 0 else 1.0

        w_rsi   = calculate_rsi(df_weekly) if not df_weekly.empty else "N/A"
        d_rsi   = calculate_rsi(df_daily)  if not df_daily.empty  else "N/A"
        d_ema50 = round(df_daily['close'].ewm(span=50, adjust=False).mean().iloc[-1], 2) if not df_daily.empty else "N/A"

        macd_display = f"{macd_cross.replace('_', ' ')} | hist {macd_hist:.4f} ({'improving' if macd_improving else 'deteriorating'})"

        summary = (
            f"Macro Regime: {regime.upper()}\n"
            f"Weekly RSI={w_rsi} | Daily RSI={d_rsi} | Daily EMA50=${d_ema50}\n"
            f"4H: Fibs={fib_str} | RSI={rsi_4h} "
            f"| Div={div_4h} | MACD={macd_display} "
            f"| EMA={ema_status}\n"
            f"1H: Fibs={get_fib_levels(df_1h)[0]} | RSI={rsi_1h} "
            f"| Div={div_1h}\n"
            f"ATR(14)=${atr}"
        )

        signals = {
            "div_4h":         div_4h,
            "div_1h":         div_1h,
            "macd_cross":     macd_cross,
            "macd_hist":      macd_hist,
            "macd_improving": macd_improving,
            "ema_status":     ema_status,
            "fib_closest":    fib_closest,
            "fib_levels":     fib_levels,
            "fib_distance_pct": round((df_4h['close'].iloc[-1] - fib_levels.get(fib_closest, df_4h['close'].iloc[-1])) / df_4h['close'].iloc[-1] * 100, 3) if fib_levels and fib_closest in fib_levels else 0.0,
            "distance_from_daily_ema50_pct": round((df_4h['close'].iloc[-1] - df_daily['close'].ewm(span=50, adjust=False).mean().iloc[-1]) / df_4h['close'].iloc[-1] * 100, 3) if not df_daily.empty else 0.0,
            "rsi_4h":         rsi_4h,
            "rsi_1h":         rsi_1h,
            "price":          df_4h['close'].iloc[-1],
            "vol_ratio_4h":   vol_ratio_4h,
            "vol_ratio_1h":   vol_ratio_1h,
            "vol_4h_raw":     round(vol_4h, 2),
        }
        return summary, regime, atr, signals

    except Exception as e:
        print(f"TA error: {e}")
        return "TA dashboard unavailable", "chop", 2.0, {}


def is_fomc_window():
    today = pd.Timestamp.now('UTC').strftime('%Y-%m-%d')
    for d in FOMC_DATES:
        if abs((pd.to_datetime(today) - pd.to_datetime(d)).days) <= 2:
            return True, f"⚠️ FOMC WINDOW ACTIVE (near {d})"
    return False, "No FOMC window"


def get_funding_oi(current_price):
    try:
        fr      = binance_client.futures_funding_rate(symbol='SOLUSDT', limit=1)[0]
        funding = float(fr['fundingRate']) * 100
        oi      = binance_client.futures_open_interest(symbol='SOLUSDT')
        oi_val  = float(oi['openInterest']) * current_price / 1_000_000
        return f"Funding: {funding:+.4f}%/8h | OI: ${oi_val:.1f}M"
    except Exception:
        return "Funding/OI unavailable"


def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        d = r.json()['data'][0]
        return f"F&G: {d['value']} ({d['value_classification']})"
    except Exception:
        return "F&G N/A"


def get_multi_news():
    sources = [
        ("CoinDesk",     "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph","https://cointelegraph.com/rss"),
        ("The Block",    "https://www.theblock.co/rss"),
    ]
    headlines = []
    for name, url in sources:
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                headlines.append(f"{name}: {feed.entries[0].title}")
        except Exception:
            pass
    return " | ".join(headlines) if headlines else "No major news"


# ====================== STATE & MEMORY ======================
def load_position_state():
    if os.path.exists(POSITION_STATE_FILE):
        try:
            with open(POSITION_STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Corrupt position state: {e}")
            os.remove(POSITION_STATE_FILE)
    return {
        "in_position": False, "entry_price": 0.0, "stop_price": 0.0,
        "target_price": 0.0, "be_trigger": 0.0, "at_be": False,
        "trail_stop": 0.0, "risk": 0.0, "entry_time": "",
        "reason_entry": "", "news_at_entry": "", "conviction": 7,
        "regime_at_entry": "chop",
    }


def save_position_state(state):
    with open(POSITION_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_trade_history():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_trade_history(history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def save_trade(trade_data):
    history = load_trade_history()
    history.append(trade_data)
    save_trade_history(history)


def get_history_summary(history):
    if not history:
        return "No past trades yet."
    total    = len(history)
    wins     = sum(1 for t in history if t.get("pnl_pct", 0) > 0)
    losses   = total - wins
    win_rate = round(wins / total * 100, 1) if total else 0
    avg_win  = round(sum(t["pnl_pct"] for t in history if t.get("pnl_pct", 0) > 0) / wins,  2) if wins   else 0
    avg_loss = round(sum(t["pnl_pct"] for t in history if t.get("pnl_pct", 0) <= 0) / losses, 2) if losses else 0
    total_pnl = round(sum(t.get("pnl_pct", 0) for t in history), 2)

    regime_pnl = {}
    for t in history:
        r = t.get("regime_at_entry", "unknown")
        regime_pnl.setdefault(r, []).append(t.get("pnl_pct", 0))
    regime_lines = []
    for r, pnls in regime_pnl.items():
        avg = round(sum(pnls) / len(pnls), 2)
        regime_lines.append(f"  {r}: {len(pnls)} trades | avg {avg:+.2f}%")

    recent = "\n".join(
        f"  [{t.get('regime_at_entry','?')}] {t['entry_time'][:10]}: "
        f"{t.get('pnl_pct', '?'):+.2f}% | Entry: {t.get('reason_entry','')[:80]} | "
        f"Lesson: {t.get('lesson','')[:120]}"
        for t in history[-15:]
    )

    return (
        f"Lifetime: {total} trades | Win rate: {win_rate}% | "
        f"Avg win: +{avg_win}% | Avg loss: {avg_loss}% | Total PnL: {total_pnl:+.2f}%\n"
        f"By regime:\n" + "\n".join(regime_lines) + f"\nRecent 15:\n{recent}"
    )


def build_position_info(state, current_price):
    if not state["in_position"]:
        return "CASH"
    effective_stop = max(state.get("stop_price", 0), state.get("trail_stop", 0))
    pnl = (current_price - state["entry_price"]) / state["entry_price"] * 100
    hours_held = 0
    try:
        entry_dt = datetime.fromisoformat(state.get("entry_time", ""))
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
    except Exception:
        pass
    return (
        f"In position | Entry ${state['entry_price']:.4f} | PnL {pnl:+.1f}% | "
        f"Stop ${effective_stop:.4f} | Held {hours_held:.1f}h | "
        f"Regime at entry: {state.get('regime_at_entry', '?')}"
    )


# ====================== CIRCUIT BREAKER ======================
def is_circuit_breaker_active(history):
    if len(history) < CIRCUIT_BREAKER_LOSSES:
        return False
    recent = history[-CIRCUIT_BREAKER_LOSSES:]
    return all(t.get("pnl_pct", 0) <= -1 for t in recent)


# ====================== REFLECTION ======================
def generate_lesson(trade_data):
    pnl  = trade_data['pnl_pct']
    exit = trade_data.get('reason_exit', '')

    prompt = f"""You are reviewing a completed SOL spot trade to extract one strategic lesson.

Trade data:
  Regime at entry: {trade_data.get('regime_at_entry', 'unknown')}
  Entry price:     ${trade_data['entry_price']:.4f}
  Exit price:      ${trade_data['exit_price']:.4f}
  PnL:             {pnl:+.2f}%
  Held:            {trade_data.get('hours_held', '?')}h
  Entry reason:    {trade_data.get('reason_entry', '')}
  Exit reason:     {exit}
  Conviction:      {trade_data.get('conviction_at_entry', '?')}/10
  Conviction breakdown: {trade_data.get('conviction_breakdown', 'N/A')}

RULES:
1. Mechanical exits (ATR trail, stop, target) are always correct — never question them.
2. Focus on ONE of: (a) Was the regime right for a long? (b) Was there real confluence or was it forced? (c) Was hold time appropriate? (d) Did it meet minimum conviction threshold?
3. If regime was BEAR/CHOP and it lost money — address whether we should have entered at all.
4. If profitable — name the specific signals that worked.
5. One sentence. Direct. No hedging.

Write only the lesson sentence."""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Exited on {exit} with {pnl:+.2f}%"


def close_position(state, exit_price, exit_reason, current_regime="unknown"):
    pnl_pct = ((exit_price - state["entry_price"]) / state["entry_price"]) * 100
    send_alert("Agent5 EXIT", f"{exit_reason} @ {exit_price:.4f} | PnL {pnl_pct:+.1f}%")

    hours_held = 0
    try:
        entry_dt = datetime.fromisoformat(state.get("entry_time", ""))
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        hours_held = round((datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600, 1)
    except Exception:
        pass

    trade_data = {
        "entry_time":           state.get("entry_time", "unknown"),
        "entry_price":          state["entry_price"],
        "exit_time":            datetime.now(timezone.utc).isoformat(),
        "exit_price":           exit_price,
        "pnl_pct":              round(pnl_pct, 4),
        "hours_held":           hours_held,
        "reason_entry":         state.get("reason_entry", "unknown"),
        "reason_exit":          exit_reason,
        "news_at_entry":        state.get("news_at_entry", "unknown"),
        "regime_at_entry":      state.get("regime_at_entry", "unknown"),
        "regime_at_exit":       current_regime,
        "conviction_at_entry":  state.get("conviction", "?"),
        "conviction_breakdown": state.get("conviction_breakdown", "N/A"),
        "lesson":               "Pending...",
    }

    trade_data["lesson"] = generate_lesson(trade_data)
    print(f"📝 Lesson: {trade_data['lesson']}")

    save_trade(trade_data)
    if os.path.exists(POSITION_STATE_FILE):
        os.remove(POSITION_STATE_FILE)
    return load_position_state()


# ====================== CORE AGENT LOOP ======================
def run_agent():
    global solana_client, keypair, wallet_pubkey, binance_client
    acquire_lock()
    solana_client  = SolanaClient(os.getenv("RPC_URL"))
    keypair        = Keypair.from_base58_string(os.getenv("PRIVATE_KEY"))
    wallet_pubkey  = keypair.pubkey()
    binance_client = BinanceClient()

    state = load_position_state()
    print("🚀 Agent5 started (v2 — self-backtesting + code-computed conviction)")

    while True:
        try:
            usdt          = get_token_balance(USDT_MINT)
            sol           = get_token_balance(SOL_MINT)
            ticker        = binance_client.get_symbol_ticker(symbol='SOLUSDT')
            current_price = float(ticker['price'])
            now_str       = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M')

            print(f"\n{'='*70}")
            print(f"[{now_str}] WALLET — USDT: ${usdt:,.2f} | SOL: {sol:.6f} | Price: ${current_price:.2f}")

            # ── Gather context ──────────────────────────────────────────────
            news        = get_multi_news()
            funding_oi  = get_funding_oi(current_price)
            fear_greed  = get_fear_greed()
            is_fomc, fomc_str = is_fomc_window()
            btc_ticker  = binance_client.get_ticker(symbol='BTCUSDT')
            btc_change  = float(btc_ticker.get('priceChangePercent', 0))

            ta_summary, current_regime, atr, signals = get_ta_summary()

            print(f"   📊 Regime: {current_regime.upper()}")
            print(f"   📈 TA: {ta_summary[:300]}...")
            print(f"   💰 {funding_oi} | {fear_greed}")

            history          = load_trade_history()
            history_summary  = get_history_summary(history)
            position_info    = build_position_info(state, current_price)

            # ── Self-backtest: tag outcomes + compute edge ──────────────────
            df_4h_extended = get_extended_4h_history()
            if not df_4h_extended.empty:
                print("   🏷️  Tagging signal outcomes...")
                tag_outcomes_in_decisions(df_4h_extended)

            track_record = get_signal_performance_summary()
            print(f"   📊 Track record: {track_record[:120]}...")

            # Historical edge for current signal combo
            edge_str = "No edge data"
            if signals and not df_4h_extended.empty:
                edge_str = compute_historical_edge(
                    df_4h_extended,
                    current_regime,
                    signals.get("div_4h", "none"),
                    signals.get("macd_cross", "no_cross"),
                    signals.get("ema_status", "mixed"),
                )

            # ── Code-computed conviction ────────────────────────────────────
            conviction_score, conviction_breakdown = 0, "No signals"
            if signals:
                conviction_score, conviction_breakdown = compute_conviction(
                    current_regime,
                    signals.get("div_4h", "none"),
                    signals.get("div_1h", "none"),
                    signals.get("macd_cross", "no_cross"),
                    signals.get("macd_improving", False),
                    signals.get("ema_status", "mixed"),
                    signals.get("fib_closest", "0.500"),
                    current_price,
                    signals.get("fib_levels", {}),
                    is_fomc,
                )

            print(f"   🧮 Conviction: {conviction_score}/10 — {conviction_breakdown}")

            if state.get("in_position"):
                effective_stop = max(state.get("stop_price", 0), state.get("trail_stop", 0))
                unrealized     = ((current_price - state["entry_price"]) / state["entry_price"]) * 100
                print(f"   🛑 Stop: ${effective_stop:.4f} | Trail: ${state.get('trail_stop',0):.4f} | Unrealized: {unrealized:+.1f}%")

            # ── Circuit breaker ─────────────────────────────────────────────
            if is_circuit_breaker_active(history):
                print("🔴 CIRCUIT BREAKER — 4 consecutive losses — pausing 6h")
                time.sleep(21600)
                continue

            # ── Hard gate: conviction too low → skip AI call entirely ───────
            # If not in position and conviction < 6.5, we know it's a STAY.
            # Save the API call and the risk of the AI talking itself into a trade.
            if not state["in_position"] and conviction_score < 6.5:
                print(f"   ⏸  STAY — conviction {conviction_score}/10 below 6.5 threshold (skipping AI call)")
                log_decision("STAY", conviction_score, current_regime, current_price,
                             conviction_breakdown, edge_str,
                             signals=signals, funding_oi=funding_oi,
                             fear_greed=fear_greed, atr=atr, btc_change=btc_change)
                time.sleep(CHECK_INTERVAL)
                continue

            # ── Build prompt (much shorter — conviction is pre-computed) ─────
            prompt = f"""You are a disciplined autonomous SOL spot trader. Conviction has been pre-computed by the system — you cannot change it.

SYSTEM-COMPUTED CONVICTION: {conviction_score}/10
BREAKDOWN: {conviction_breakdown}
THRESHOLD: ≥6.5 = BUY eligible | <6.5 = STAY

Price: ${current_price:.2f} | Regime: {current_regime.upper()}

=== CONTEXT (for reasoning, not entry/exit triggers) ===
{funding_oi} | {fear_greed} | BTC 24h: {btc_change:+.1f}%
{fomc_str}

=== POSITION ===
{position_info}

=== TA ===
{ta_summary}

=== HISTORICAL EDGE ===
{edge_str}

=== SIGNAL TRACK RECORD (how past signals with similar conviction performed) ===
{track_record}

=== TRADE HISTORY ===
{history_summary}

RULES:
1. Conviction is computed by code — you CANNOT override or recompute it.
2. If conviction < 6.5 and not in position → action MUST be STAY.
3. If regime is bear → action MUST be STAY unless conviction ≥ 8.0 AND 4H bullish divergence AND price at 0.236 or 0.382 Fib
4. When in position: evaluate SELL only on mechanical triggers (stop/trail/target hit) or confirmed TA breakdown (2 of: loses key Fib, 4H MACD bearish cross, below both EMAs 6h+ with volume).
5. Sentiment/news/F&G/funding are NEVER entry or exit reasons.
6. A 1.5-2% locked win via trail in chop/bear is a perfect outcome.

Output ONLY this JSON:
{{
  "thought": "brief reasoning — what you see in the TA that the code-computed score might miss",
  "action": "BUY | SELL | STAY",
  "reason": "one sentence — TA signals only",
  "rr_ratio": 2.5,
  "key_risks": ["risk1", "risk2"]
}}"""

            # ── AI call ─────────────────────────────────────────────────────
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw, flags=re.MULTILINE).strip()
                decision = json.loads(raw)
            except Exception as e:
                print(f"⚠️  AI call failed: {e} — defaulting to STAY")
                decision = {"action": "STAY", "thought": f"AI error: {e}", "rr_ratio": 0}

            print(f"\n🤖 THOUGHT: {decision.get('thought', '')[:300]}")
            print(f"   ACTION: {decision.get('action')} | System conviction: {conviction_score}/10")
            print(f"   Reason: {decision.get('reason','')}")

            # ── Hard overrides (code always wins) ───────────────────────────
            ai_action_raw = decision.get("action", "STAY").upper()  # capture before override
            ai_thought_raw = decision.get("thought", "")
            ai_reason_raw  = decision.get("reason", "")
            ai_error_raw   = ""
            # If the decision dict came from an exception path, "thought" starts with "AI error:"
            if ai_thought_raw.startswith("AI error:"):
                ai_error_raw = ai_thought_raw.replace("AI error:", "").strip()
            
            action = ai_action_raw

            if action == "BUY" and conviction_score < 6.5:
                action = "STAY"
                print(f"   ⛔ Code override: conviction {conviction_score} < 7.5 — STAY")

            if action == "BUY" and current_regime == "bear" and not (
                conviction_score >= 8.0 and
                signals.get("div_4h") in ("regular_bullish", "hidden_bullish") and
                signals.get("fib_closest") in ("0.236", "0.382")
            ):
                action = "STAY"
                print("   ⛔ Code override: bear regime gate — STAY")

            # Log every decision for self-backtesting (now includes AI reasoning)
            log_decision(action, conviction_score, current_regime, current_price,
                         conviction_breakdown, edge_str,
                         signals=signals, funding_oi=funding_oi,
                         fear_greed=fear_greed, atr=atr, btc_change=btc_change,
                         ai_called=True,
                         ai_action_raw=ai_action_raw,
                         ai_thought=ai_thought_raw,
                         ai_reason=ai_reason_raw,
                         ai_error=ai_error_raw)

            if LIVE_TRADING:
                force_exit   = False
                exit_reason  = None

                # ── Position management ─────────────────────────────────────
                if state["in_position"]:
                    df_4h_atr = get_klines_df('4h', 50)
                    atr       = calculate_atr(df_4h_atr) if not df_4h_atr.empty else atr

                    entry_regime   = state.get("regime_at_entry", "chop")
                    trail_mult     = TRAIL_MULTIPLIER.get(entry_regime, DEFAULT_TRAIL)
                    max_hold       = MAX_HOLD_HOURS.get(entry_regime, DEFAULT_MAX_HOLD)

                    # ── Breakeven promotion (regime-conditional) ────────────
                    if not state.get("at_be", False) and current_price >= state.get("be_trigger", 0):
                        state["stop_price"] = state["entry_price"]
                        state["at_be"]      = True
                        print(f"   🔒 BE stop promoted to ${state['stop_price']:.4f}")

                    # ── Trailing stop — starts at 0.5× risk profit ──────────
                    # OLD: required entry_price + risk (1.8× ATR) before trailing
                    # NEW: starts trailing once 0.5× risk in profit (catches gains in chop)
                    trail_activation = state["entry_price"] + (state.get("risk", 0) * 0.5)
                    if current_price > trail_activation:
                        new_trail = current_price - (trail_mult * atr)
                        state["trail_stop"] = max(state.get("trail_stop", 0), new_trail)
                        print(f"   📈 Trail updated → ${state['trail_stop']:.4f} ({trail_mult}x ATR)")

                    effective_stop = max(state.get("stop_price", 0), state.get("trail_stop", 0))

                    # ── Max hold check ──────────────────────────────────────
                    try:
                        entry_dt = datetime.fromisoformat(state.get("entry_time", ""))
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                        hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                        if hours_held > max_hold:
                            if current_regime in ("bull", "strong_bull") and conviction_score >= 8:
                                print(f"   ⏳ Max hold {max_hold}h reached but bull + high conviction — extending")
                            else:
                                force_exit  = True
                                exit_reason = f"MAX HOLD ({max_hold}h for {entry_regime} regime)"
                    except Exception as e:
                        print(f"   ⚠️  Hold-time check skipped: {e}")

                    # ── Stop / Target ───────────────────────────────────────
                    if current_price <= effective_stop:
                        force_exit  = True
                        exit_reason = "TRAIL STOP (ATR)" if state.get("trail_stop", 0) > state.get("stop_price", 0) else "STOP LOSS"
                    elif current_price >= state.get("target_price", 0):
                        force_exit  = True
                        exit_reason = "TARGET HIT"

                    save_position_state(state)

                # ── Forced exits take priority ──────────────────────────────
                if force_exit and state["in_position"]:
                    print(f"🚨 FORCE EXIT: {exit_reason}")
                    sig, confirmed = robust_sell()
                    if sig:
                        if not confirmed:
                            log_unconfirmed_sell(sig, f"force_exit={exit_reason}")
                            send_alert("URGENT: Unconfirmed Force Exit",
                                       f"TX sent but NOT confirmed!\nSig: {sig}\nMANUAL intervention required!")
                        else:
                            try:
                                exit_ticker   = binance_client.get_symbol_ticker(symbol='SOLUSDT')
                                record_price  = float(exit_ticker['price'])
                            except Exception:
                                record_price  = current_price
                            state = close_position(state, record_price, exit_reason, current_regime)

                # ── BUY ────────────────────────────────────────────────────
                elif not state["in_position"] and action == "BUY" and usdt > 20:
                    print("🚀 BUY SIGNAL — executing...")
                    sig, confirmed = robust_buy(usdt)
                    if sig and confirmed:
                        time.sleep(4)
                        try:
                            entry_ticker  = binance_client.get_symbol_ticker(symbol='SOLUSDT')
                            entry_price   = float(entry_ticker['price'])
                        except Exception:
                            entry_price   = current_price

                        df_4h_entry   = get_klines_df('4h', 50)
                        atr           = calculate_atr(df_4h_entry) if not df_4h_entry.empty else atr

                        risk_distance = max(atr * 1.8, entry_price * 0.035)

                        if current_regime in ("bear", "chop"):
                            rr = min(decision.get("rr_ratio", 2.0), 2.0)
                        elif current_regime == "bull_chop":
                            rr = min(decision.get("rr_ratio", 2.5), 2.5)
                        else:
                            rr = decision.get("rr_ratio", 2.8)

                        target_price = entry_price + (rr * risk_distance)
                        be_mult      = BE_TRIGGER_MULT.get(current_regime, DEFAULT_BE_MULT)
                        be_trigger   = entry_price + (risk_distance * be_mult)

                        state = {
                            "in_position":          True,
                            "entry_price":          entry_price,
                            "stop_price":           entry_price - risk_distance,
                            "target_price":         target_price,
                            "be_trigger":           be_trigger,
                            "at_be":                False,
                            "trail_stop":           entry_price - risk_distance,
                            "risk":                 risk_distance,
                            "entry_time":           datetime.now(timezone.utc).isoformat(),
                            "reason_entry":         decision.get("reason"),
                            "news_at_entry":        news,
                            "conviction":           conviction_score,
                            "conviction_breakdown": conviction_breakdown,
                            "regime_at_entry":      current_regime,
                        }
                        save_position_state(state)
                        send_alert(
                            "Agent5 BUY",
                            f"BUY @ ${entry_price:.4f} | Regime: {current_regime}\n"
                            f"Conviction: {conviction_score}/10 — {conviction_breakdown}\n"
                            f"{decision.get('reason')}\n"
                            f"Stop ${entry_price - risk_distance:.4f} | Target ${target_price:.4f} | RR {rr:.1f}:1"
                        )
                        print(f"✅ Position opened | Entry ${entry_price:.4f} | Stop ${entry_price - risk_distance:.4f} | Target ${target_price:.4f}")

                    elif sig and not confirmed:
                        print(f"⚠️  Buy TX unconfirmed — NOT entering position: {sig}")

                # ── AI SELL ────────────────────────────────────────────────
                elif state["in_position"] and action == "SELL":
                    print("🔻 AI SELL SIGNAL — executing...")
                    sig, confirmed = robust_sell()
                    if sig:
                        if not confirmed:
                            log_unconfirmed_sell(sig, f"ai_sell reason={decision.get('reason')}")
                            send_alert("URGENT: Unconfirmed AI Sell",
                                       f"TX sent but NOT confirmed!\nSig: {sig}\nMANUAL intervention required!")
                        else:
                            try:
                                exit_ticker  = binance_client.get_symbol_ticker(symbol='SOLUSDT')
                                record_price = float(exit_ticker['price'])
                            except Exception:
                                record_price = current_price
                            state = close_position(state, record_price, decision.get("reason", "AI SELL"), current_regime)

                else:
                    print(f"   ⏸  STAY — {decision.get('reason','No action')}")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"❌ Loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    run_agent()
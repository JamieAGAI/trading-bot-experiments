# Copyright (c) 2026 Laura Staples. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, or derivative works are prohibited.
# No license is granted for training machine learning models on this code.
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
import argparse
import os
import csv
import requests
from dotenv import load_dotenv
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.pubkey import Pubkey
import base64
from spl.token.instructions import get_associated_token_address
import datetime as dt
import smtplib
from email.mime.text import MIMEText
import atexit
from binance import ThreadedWebsocketManager
import threading
import json
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ====================== JUPITER v1 (Feb 2026) ======================
JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"
JUPITER_API_KEY   = os.getenv("JUPITER_API_KEY")

SOL_MINT  = Pubkey.from_string("So11111111111111111111111111111111111111112")
USDT_MINT = Pubkey.from_string("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")
RPC_URL     = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
SLIPPAGE_BPS = 100
SOL_RESERVE  = 0.07

# Email config
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

if not JUPITER_API_KEY:
    raise ValueError("Add JUPITER_API_KEY=your_key_here to .env")
if not RPC_URL or not PRIVATE_KEY:
    raise ValueError("RPC_URL and PRIVATE_KEY required in .env")

solana_client = SolanaClient(RPC_URL)
keypair       = Keypair.from_base58_string(PRIVATE_KEY)
wallet_pubkey = keypair.pubkey()

HEADERS = {"x-api-key": JUPITER_API_KEY}

# =============================================================================
#  STRATEGY PARAMS
# =============================================================================

SYMBOL             = 'SOLUSDT'
INTERVAL_4H        = Client.KLINE_INTERVAL_4HOUR
INTERVAL_1D        = Client.KLINE_INTERVAL_1DAY
EMA_TREND          = 35
EMA_VALUE          = 15
EMA_DAILY          = 20
RSI_LEN            = 14
ATR_LEN            = 14
REWARD_RATIO       = 2.5
ATR_BUFFER         = 0.45
MAX_HOLD_BARS      = 50
RSI_MIN            = 53
RSI_MAX            = 65
FEE_PCT            = 0.003
VOLUME_MULTIPLIER  = 1.1

POSITION_SIZE      = 1.0
MAX_RISK_PCT       = 0.10  

CACHE_4H   = 'sol_4h_cache.csv'
CACHE_1D   = 'sol_1d_cache.csv'
TRADES_CSV = 'trades_v10.csv'
POSITION_STATE_FILE = 'live_position.json'

WS_TIMEOUT_SECONDS = 600

# ── Globals & Lock ───────────────────────────────────────────────────────────
df4_lock = threading.Lock()

current_4h_bar    = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': 0, 'start_time': None}
twm               = None
last_ws_msg_time  = time.time()
df4               = None
df1d              = None

# ── Fetch helpers ────────────────────────────────────────────────────────────
def _time_range(days):
    end   = int(time.time() * 1000)
    start = end - (days * 86400 * 1000)
    return end, start

def _klines_to_df(klines):
    df = pd.DataFrame(
        klines,
        columns=['timestamp','open','high','low','close','volume',
                 'close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore']
    )
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.set_index(pd.to_datetime(df['timestamp'], unit='ms', utc=True))
    df.index.name = 'timestamp'
    df = df[['open','high','low','close','volume']]

    # Drop the last bar if it's still in-progress.
    # Infer interval from spacing of last two bars; default to 4h
    # if only one row exists.
    if len(df) >= 2:
        interval = df.index[-1] - df.index[-2]
    else:
        interval = pd.Timedelta(hours=4)
    now_utc = pd.Timestamp.now('UTC')
    if df.index[-1] + interval > now_utc:
        df = df.iloc[:-1]

    return df

def _cache_is_stale(cache_path, max_age_hours):
    if not os.path.exists(cache_path):
        return True
    try:
        tmp = pd.read_csv(cache_path, index_col='timestamp', parse_dates=True)
        if tmp.empty:
            return True
        last_ts = tmp.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        age = (pd.Timestamp.now('UTC') - last_ts).total_seconds() / 3600
        return age > max_age_hours
    except Exception:
        return True

def fetch_data(days=730):
    for attempt in range(3):
        try:
            binance_client = Client()
            end, start = _time_range(days)

            if not _cache_is_stale(CACHE_4H, max_age_hours=4):
                logger.info("Loading 4h from cache (fresh)...")
                df4_local = pd.read_csv(CACHE_4H, index_col='timestamp', parse_dates=True)
                df4_local = df4_local[['open','high','low','close','volume']]
                if df4_local.index.tz is None:
                    df4_local.index = df4_local.index.tz_localize('UTC')
            else:
                logger.info("Fetching fresh 4h data from Binance...")
                df4_local = _klines_to_df(
                    binance_client.get_historical_klines(SYMBOL, INTERVAL_4H, start_str=start, end_str=end)
                )
                df4_local.to_csv(CACHE_4H)

            logger.info("Fetching fresh daily data from Binance...")
            df1d_local = _klines_to_df(
                binance_client.get_historical_klines(SYMBOL, INTERVAL_1D, start_str=start, end_str=end)
            )
            df1d_local.to_csv(CACHE_1D)

            return df4_local, df1d_local

        except Exception as e:
            logger.warning(f"Fetch attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(10)
    raise ValueError("Data fetch failed after 3 retries")

# ── Indicators ───────────────────────────────────────────────────────────────
def compute_4h(df, dropna=True):
    df = df.copy()
    df['ema_trend']  = ta.ema(df['close'], length=EMA_TREND)
    df['ema_value']  = ta.ema(df['close'], length=EMA_VALUE)
    df['rsi']        = ta.rsi(df['close'], length=RSI_LEN)
    df['atr']        = ta.atr(df['high'], df['low'], df['close'], length=ATR_LEN)
    df['vol_ma']     = df['volume'].rolling(20, min_periods=1).mean()
    df['bullish']    = (df['close'] > df['open']).astype(int)
    return df.dropna() if dropna else df

def compute_daily(df):
    df = df.copy()
    df['ema_daily'] = ta.ema(df['close'], length=EMA_DAILY)
    return df.dropna()

def attach_daily(df4_in, df1d_in):
    df1d_in = df1d_in.copy()
    df4_in  = df4_in.copy()
    if df1d_in.index.tz is None:
        df1d_in.index = df1d_in.index.tz_localize('UTC')
    if df4_in.index.tz is None:
        df4_in.index = df4_in.index.tz_localize('UTC')
    df1d_in['daily_bull'] = (df1d_in['close'] > df1d_in['ema_daily']).astype(int)
    daily_bull = df1d_in['daily_bull'].reindex(df4_in.index, method='ffill')
    df4_out = df4_in.copy()
    df4_out['daily_bull'] = daily_bull
    return df4_out.dropna()

# ── Signal ───────────────────────────────────────────────────────────────────
def buy_signal(df, i, debug=False):
    if i < 1:
        return False
    c  = df.iloc[i]
    p1 = df.iloc[i-1]
    recent_low    = df.iloc[max(0, i-8):i+1]['low'].min()
    in_value_zone = recent_low <= c['ema_value'] * 1.02

    conditions = [
        (c['daily_bull'] == 1,             "daily_bull"),
        (c['close'] > c['ema_trend'],      "close > ema_trend"),
        (in_value_zone,                     "in_value_zone"),
        (c['low'] > p1['low'],             "higher low"),
        (c['close'] > p1['high'],          "close > prev high"),
        (c['bullish'] == 1,                 "bullish candle"),
        (RSI_MIN < c['rsi'] < RSI_MAX,     "RSI range"),
        (c['volume'] >= c['vol_ma'] * VOLUME_MULTIPLIER, "volume threshold"),
        (c['close'] > c['ema_value'],      "close > ema_value"),
    ]

    for passed, name in conditions:
        if not passed:
            if debug:
                logger.info(f"  Signal FAILED: {name} | "
                            f"daily_bull={c['daily_bull']}, close={c['close']:.2f}, "
                            f"ema_trend={c['ema_trend']:.2f}, in_value_zone={in_value_zone}, "
                            f"low={c['low']:.2f}, p1_low={p1['low']:.2f}, "
                            f"close_vs_p1_high={c['close'] > p1['high']}, "
                            f"bullish={c['bullish']}, rsi={c['rsi']:.1f}, "
                            f"vol={c['volume']:.0f}, vol_ma*{VOLUME_MULTIPLIER}={c['vol_ma']*VOLUME_MULTIPLIER:.0f}, "
                            f"close_vs_ema_value={c['close'] > c['ema_value']}")
            return False

    if debug:
        logger.info("  Signal: ALL conditions passed!")
    return True

# ── Backtest ─────────────────────────────────────────────────────────────────
def backtest(df, initial_capital=10000):
    capital      = initial_capital
    position     = 0
    entry_price  = 0
    stop_price   = 0
    target_price = 0
    be_trigger   = 0
    at_be        = False
    trail_stop   = 0
    entry_bar    = 0
    trades       = []
    wins = losses = 0

    for i in range(8, len(df) - 1):
        if position == 0 and buy_signal(df, i):
            next_open   = df.iloc[i+1]['open']
            entry_price = next_open
            atr         = df.iloc[i]['atr']
            swing_low   = df.iloc[i-2:i+1]['low'].min()
            stop_price  = swing_low - (ATR_BUFFER * atr)
            risk        = entry_price - stop_price
            if risk <= 0 or (risk / entry_price) > MAX_RISK_PCT:
                continue

            target_price = entry_price + (REWARD_RATIO * risk)
            be_trigger   = entry_price + risk
            deploy       = capital * POSITION_SIZE
            shares       = deploy / entry_price
            position     = shares
            entry_bar    = i + 1
            trail_stop   = stop_price

            trades.append({
                'entry_date' : df.iloc[i+1].name,
                'price'      : round(entry_price, 4),
                'stop'       : round(stop_price, 4),
                'target'     : round(target_price, 4),
                'risk'       : round(risk, 4),
                'rr'         : round(REWARD_RATIO, 2),
                'shares'     : round(shares, 6),
                'profit'     : None,
                'capital'    : round(capital, 2),
                'exit_reason': None,
            })

        elif position > 0:
            cur       = df.iloc[i]
            bars_held = i - entry_bar

            if not at_be and cur['high'] >= be_trigger:
                stop_price = entry_price
                at_be      = True

            exit_price  = None
            exit_reason = None

            if cur['close'] > entry_price + (2 * risk):
                recent_high = df.iloc[max(0, i-2):i+1]['high'].max()
                new_trail   = recent_high - (2 * cur['atr'])
                trail_stop  = max(trail_stop, new_trail)

            effective_stop = max(stop_price, trail_stop)
            if cur['low'] <= effective_stop:
                exit_price  = effective_stop
                exit_reason = 'Trailing Stop' if trail_stop > stop_price and cur['low'] <= trail_stop else ('Breakeven Stop' if at_be else 'Stop Loss')
            elif cur['high'] >= target_price:
                exit_price  = target_price
                exit_reason = 'Take Profit'
            elif (cur['close'] < cur['ema_trend'] and df.iloc[i-1]['close'] < df.iloc[i-1]['ema_trend']):
                exit_price  = cur['close']
                exit_reason = 'Trend Exit'
            elif cur['daily_bull'] == 0:
                exit_price  = cur['close']
                exit_reason = 'Daily Trend Exit'
            elif bars_held >= MAX_HOLD_BARS:
                exit_price  = cur['close']
                exit_reason = 'Timeout'

            if exit_price is not None:
                gross_profit = (exit_price - entry_price) * position
                entry_fee    = (entry_price * position) * FEE_PCT
                exit_fee     = (exit_price * position) * FEE_PCT
                net_profit   = gross_profit - entry_fee - exit_fee
                capital     += net_profit
                position     = 0
                if net_profit > 0:
                    wins += 1
                else:
                    losses += 1
                trades[-1].update({
                    'exit_date'  : cur.name,
                    'exit_price' : round(exit_price, 4),
                    'profit'     : round(net_profit, 2),
                    'exit_reason': exit_reason,
                    'bars_held'  : bars_held,
                    'at_be'      : at_be,
                })

    if position > 0:
        exit_price = df.iloc[-1]['close']
        gross_profit = (exit_price - entry_price) * position
        entry_fee    = (entry_price * position) * FEE_PCT
        exit_fee     = (exit_price * position) * FEE_PCT
        net_profit   = gross_profit - entry_fee - exit_fee
        capital     += net_profit
        if net_profit > 0:
            wins += 1
        else:
            losses += 1
        trades[-1].update({
            'exit_date'  : df.iloc[-1].name,
            'exit_price' : round(exit_price, 4),
            'profit'     : round(net_profit, 2),
            'exit_reason': 'End of Data',
            'bars_held'  : len(df) - 1 - entry_bar,
            'at_be'      : at_be,
        })

    closed     = [t for t in trades if t.get('exit_reason')]
    profits    = [t['profit'] for t in closed if t['profit'] is not None]
    total      = len(closed)
    pct        = ((capital - initial_capital) / initial_capital) * 100
    wr         = (wins / total * 100) if total > 0 else 0
    avg_w      = sum(p for p in profits if p > 0) / wins if wins else 0
    avg_l      = sum(p for p in profits if p <= 0) / losses if losses else 0
    exp        = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)

    logger.info("\n" + "═"*47)
    logger.info(f"BACKTEST — {SYMBOL} 4h + Daily (v10)  VOLUME x{VOLUME_MULTIPLIER}")
    logger.info(f"{df.index[0].date()} → {df.index[-1].date()}")
    logger.info("═"*47)
    logger.info(f"Starting Capital : ${initial_capital:,.2f}")
    logger.info(f"Final Capital    : ${capital:,.2f}")
    logger.info(f"Total Return     : {pct:+.2f}%")
    logger.info(f"Total Trades     : {total}")
    logger.info(f"Wins / Losses    : {wins} / {losses}")
    logger.info(f"Win Rate         : {wr:.1f}%")
    logger.info(f"Expectancy/trade : ${exp:,.2f}")
    logger.info("═"*47)

    fieldnames = ['entry_date','price','stop','target','risk','rr','shares','exit_date','exit_price','profit','capital','exit_reason','bars_held','at_be']
    with open(TRADES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(closed)
    logger.info(f"Trade log → {TRADES_CSV}")
    return capital, trades

# ── Jupiter Swaps ────────────────────────────────────────────────────────────
def get_token_balance(mint):
    if mint == SOL_MINT:
        try:
            balance_resp = solana_client.get_balance(wallet_pubkey)
            return balance_resp.value / 1_000_000_000
        except Exception as e:
            logger.warning(f"Native SOL balance fetch failed: {e}")
            return 0.0
    else:
        ata = get_associated_token_address(wallet_pubkey, mint)
        try:
            balance_resp = solana_client.get_token_account_balance(ata)
            if hasattr(balance_resp, 'value'):
                return int(balance_resp.value.amount) / (10 ** balance_resp.value.decimals)
        except Exception as e:
            logger.warning(f"Token balance fetch failed for {mint}: {e}")
            return 0.0
    return 0.0

def get_jupiter_quote(input_mint, output_mint, amount):
    params = {
        "inputMint": str(input_mint),
        "outputMint": str(output_mint),
        "amount": str(int(amount)),
        "slippageBps": str(SLIPPAGE_BPS)
    }
    response = requests.get(JUPITER_QUOTE_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()

def get_swap_transaction(quote_response):
    payload = {
        "quoteResponse": quote_response,
        "userPublicKey": str(wallet_pubkey),
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "priorityLevel": "veryHigh",
                "maxLamports": 5000000,
            }
        },
    }
    response = requests.post(JUPITER_SWAP_URL, json=payload, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()["swapTransaction"]

# ─── Transaction validation (Apr 27, 2026) ───────────────────────────────────
# Allowlist of program IDs that may legitimately appear in a Jupiter swap.
# Verified against real $2 buy + 0.02 sell + 0.01 sell on Apr 27, 2026
# via swap_test.py.
JUPITER_ALLOWED_PROGRAMS = {
    "11111111111111111111111111111111",                  # System Program
    "ComputeBudget111111111111111111111111111111",       # Compute Budget
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",       # Jupiter Aggregator
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",       # SPL Token
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",      # Associated Token Program
}


def validate_swap_transaction(tx, quote, direction):
    """
    Inspect a decoded VersionedTransaction before signing.
    Returns (ok: bool, reason: str). ok=False means DO NOT SIGN.
    """
    try:
        msg = tx.message
        account_keys = list(msg.account_keys)
        instructions = list(msg.instructions)

        # Check 1: every instruction's program must be allowlisted
        for i, ix in enumerate(instructions):
            program_idx = ix.program_id_index
            if program_idx >= len(account_keys):
                return False, f"instr {i}: program_id_index {program_idx} out of range"
            program = str(account_keys[program_idx])
            if program not in JUPITER_ALLOWED_PROGRAMS:
                return False, f"instr {i}: non-allowlisted program {program}"

        # Check 2: our wallet must appear in account_keys
        wallet_str = str(wallet_pubkey)
        if not any(str(k) == wallet_str for k in account_keys):
            return False, f"our wallet {wallet_str} not in account_keys"

        # Check 3: quote sanity
        try:
            in_amount  = int(quote.get("inAmount", 0))
            out_amount = int(quote.get("outAmount", 0))
            min_out    = int(quote.get("otherAmountThreshold", 0))
        except (TypeError, ValueError):
            return False, "quote amounts not parseable as integers"

        if in_amount <= 0 or out_amount <= 0 or min_out <= 0:
            return False, (f"quote has non-positive amounts "
                           f"(in={in_amount} out={out_amount} min={min_out})")

        if min_out > out_amount:
            return False, f"otherAmountThreshold {min_out} > outAmount {out_amount} (impossible)"

        max_slip_bps = SLIPPAGE_BPS * 2
        expected_min = out_amount * (10000 - max_slip_bps) // 10000
        if min_out < expected_min:
            return False, (f"min_out {min_out} below expected floor {expected_min} "
                           f"(slippage exceeds {max_slip_bps} bps)")

        # Check 4: direction matches quote mints
        in_mint  = quote.get("inputMint", "")
        out_mint = quote.get("outputMint", "")
        if direction == "buy":
            if in_mint != str(USDT_MINT) or out_mint != str(SOL_MINT):
                return False, f"buy direction but quote mints are {in_mint} -> {out_mint}"
        elif direction == "sell":
            if in_mint != str(SOL_MINT) or out_mint != str(USDT_MINT):
                return False, f"sell direction but quote mints are {in_mint} -> {out_mint}"
        else:
            return False, f"unknown direction: {direction}"

        return True, "ok"

    except Exception as e:
        return False, f"validation exception: {type(e).__name__}: {e}"


def execute_swap(swap_tx_b64, quote, direction="buy", pre_usdt=0, pre_sol=0):
    """direction: 'buy' (USDT→SOL) or 'sell' (SOL→USDT). Returns (sig, confirmed).

    Apr 27 2026: validates the Jupiter transaction before signing.
    On validation failure: log, return (None, False) so the calling
    perform_buy/perform_sell treats it as a failed attempt and retries.
    """
    try:
        swap_tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(swap_tx_bytes)

        # ─── VALIDATE BEFORE SIGNING ────────────────────────────────────
        ok, reason = validate_swap_transaction(tx, quote, direction)
        if not ok:
            logger.warning(f"🛑 Transaction validation REJECTED: {reason}")
            logger.warning(f"   Direction: {direction} | Quote in/out: "
                           f"{quote.get('inAmount')}/{quote.get('outAmount')}")
            logger.warning("   Refusing to sign. Treating as failed attempt.")
            return None, False
        logger.info("✓ Transaction validated — signing")
        # ────────────────────────────────────────────────────────────────

        tx = VersionedTransaction(tx.message, [keypair])
        from solana.rpc.types import TxOpts
        resp = solana_client.send_raw_transaction(
            bytes(tx),
            opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")
        )
        sig = resp.value
        logger.info(f"✅ TX SENT → https://solscan.io/tx/{sig}")

        # Extended poll — 60s
        for i in range(30):
            time.sleep(2)
            try:
                status_resp = solana_client.get_signature_statuses([sig], search_transaction_history=True)
                status = status_resp.value[0] if status_resp.value else None
                if status and status.confirmation_status in ("confirmed", "finalized"):
                    logger.info(f"✅ TX CONFIRMED ({status.confirmation_status}) after {(i+1)*2}s")
                    return sig, True
            except:
                pass

        # Balance fallback — retry 3 times over 90s
        logger.warning("⏳ Polling timed out after 60s — verifying via balance (will retry 3x)...")
        for check_attempt in range(3):
            time.sleep(30)
            post_usdt = get_token_balance(USDT_MINT)
            post_sol  = get_token_balance(SOL_MINT)
            elapsed = 60 + (check_attempt + 1) * 30
            logger.info(f"   Check {check_attempt+1}/3 (t+{elapsed}s) — USDT ${post_usdt:.2f} | SOL {post_sol:.6f}")
            if direction == "buy" and post_sol > pre_sol + 0.1:
                logger.info(f"✅ Balance confirms BUY executed after {elapsed}s")
                return sig, True
            if direction == "sell" and (post_usdt > pre_usdt + 1.0 or post_sol <= SOL_RESERVE + 0.05):
                logger.info(f"✅ Balance confirms SELL executed after {elapsed}s")
                return sig, True

        logger.warning("⚠️ TX unconfirmed after 150s total — flagging for manual review")
        return sig, False

    except Exception as e:
        logger.warning(f"Swap execution failed: {e}")
        return None, False

def perform_buy(amount_usdt_lamports):
    if amount_usdt_lamports <= 0:
        logger.warning("No USDT available for swap.")
        return None, False
    adjusted = int(amount_usdt_lamports * 0.99)
    pre_usdt = get_token_balance(USDT_MINT)
    pre_sol  = get_token_balance(SOL_MINT)
    for attempt in range(7):
        try:
            logger.info(f"🔄 BUY attempt {attempt+1}/7...")
            quote = get_jupiter_quote(USDT_MINT, SOL_MINT, adjusted)
            swap_tx_b64 = get_swap_transaction(quote)
            sig, confirmed = execute_swap(swap_tx_b64, quote, direction="buy", pre_usdt=pre_usdt, pre_sol=pre_sol)
            if sig and confirmed:
                return sig, True
            elif sig and not confirmed:
                logger.warning(f"⚠️ Buy TX unconfirmed: {sig}")
                return sig, False
        except Exception as e:
            logger.warning(f"⚠️ Buy attempt {attempt+1} failed: {e}")
            if attempt < 6:
                time.sleep(3)
    logger.warning("❌ All buy attempts failed")
    return None, False

def perform_sell(amount_sol_lamports):
    if amount_sol_lamports <= 0:
        logger.warning("No SOL available for swap after reserve.")
        return None, False
    pre_usdt = get_token_balance(USDT_MINT)
    pre_sol  = get_token_balance(SOL_MINT)
    for attempt in range(7):
        try:
            logger.info(f"🔄 SELL attempt {attempt+1}/7...")
            # Re-read balance each attempt in case partial went through
            fresh_sol = get_token_balance(SOL_MINT)
            if fresh_sol <= SOL_RESERVE:
                logger.warning("⚠️ Not enough SOL after reserve")
                return None, False
            fresh_lamports = int((fresh_sol - SOL_RESERVE) * 1_000_000_000)
            quote = get_jupiter_quote(SOL_MINT, USDT_MINT, fresh_lamports)
            swap_tx_b64 = get_swap_transaction(quote)
            sig, confirmed = execute_swap(swap_tx_b64, quote, direction="sell", pre_usdt=pre_usdt, pre_sol=pre_sol)
            if sig and confirmed:
                return sig, True
            elif sig and not confirmed:
                logger.warning(f"⚠️ Sell TX unconfirmed: {sig}")
                return sig, False
        except Exception as e:
            logger.warning(f"⚠️ Sell attempt {attempt+1} failed: {e}")
            if attempt < 6:
                time.sleep(3)
    logger.warning("❌ All sell attempts failed")
    return None, False

# ── Email Alert ──────────────────────────────────────────────────────────────
def send_alert(subject, body):
    if not EMAIL_FROM or not EMAIL_TO or not EMAIL_PASS:
        logger.warning("Email config missing -- skipping alert")
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info("Alert sent!")
    except Exception as e:
        logger.warning(f"Failed to send alert: {e}")

# ── ROBUST WS (STARLINK-PROOF) ───────────────────────────────────────────────
def safe_handle_kline(msg):
    global df4, last_ws_msg_time
    try:
        if 'e' in msg and msg['e'] == 'error':
            logger.warning(f"[WS] Binance error: {msg}")
            return

        kline = msg['k']
        is_closed = kline['x']
        open_price = float(kline['o'])
        high = float(kline['h'])
        low = float(kline['l'])
        close = float(kline['c'])
        volume = float(kline['v'])
        start_time = pd.to_datetime(kline['t'], unit='ms', utc=True)

        if current_4h_bar['start_time'] is None or start_time != current_4h_bar['start_time']:
            current_4h_bar.update({
                'open': open_price, 'high': high, 'low': low,
                'close': close, 'volume': volume, 'start_time': start_time
            })
        else:
            current_4h_bar.update({
                'high': max(current_4h_bar['high'], high),
                'low': min(current_4h_bar['low'], low),
                'close': close,
                'volume': volume
            })

        last_ws_msg_time = time.time()

        if is_closed:
            closed_ts = current_4h_bar['start_time']
            logger.info(f"[WS] 4h bar closed: {closed_ts} | C={current_4h_bar['close']:.2f}")

            new_row = pd.DataFrame([{
                'open': current_4h_bar['open'],
                'high': current_4h_bar['high'],
                'low': current_4h_bar['low'],
                'close': current_4h_bar['close'],
                'volume': current_4h_bar['volume']
            }], index=[closed_ts])

            with df4_lock:
                if closed_ts in df4.index:
                    logger.debug(f"[WS] Skipped duplicate bar at {closed_ts}")
                else:
                    df4 = pd.concat([df4, new_row])
                    df4 = df4.sort_index()
                    df4 = compute_4h(df4, dropna=False)
                    logger.info(f"[WS] Appended new bar → len={len(df4)} | last close={df4['close'].iloc[-1]:.2f}")

            current_4h_bar['start_time'] = None

    except Exception as e:
        logger.warning(f"[WS] Non-fatal error (Starlink OK): {e}")

def run_websocket_forever():
    global twm
    backoff = 5
    while True:
        try:
            logger.info("Starting Binance SOLUSDT 4h WebSocket...")
            twm = ThreadedWebsocketManager()
            twm.daemon = True
            twm.start()

            twm.start_kline_socket(
                callback=safe_handle_kline,
                symbol=SYMBOL.lower(),
                interval=INTERVAL_4H
            )

            logger.info("✓ WebSocket connected successfully")

            while True:
                time.sleep(45)
                if time.time() - last_ws_msg_time > WS_TIMEOUT_SECONDS:
                    logger.warning("WS watchdog timeout → forcing reconnect")
                    break

        except Exception as e:
            logger.warning(f"WebSocket dropped (Starlink normal): {e}")
        finally:
            if twm:
                try:
                    twm.stop()
                except:
                    pass
                twm = None

        logger.info(f"Reconnecting in {backoff}s...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 300)

def stop_ws():
    global twm
    if twm:
        try:
            twm.stop()
        except:
            pass
        twm = None
    logger.info("WebSocket stopped.")

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--days', type=int, default=1095)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    if args.fresh:
        for f in [CACHE_4H, CACHE_1D, POSITION_STATE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Cache and position state cleared.")

    df4, df1d = fetch_data(days=args.days)
    df1d = compute_daily(df1d)
    df4  = compute_4h(df4)
    df4  = attach_daily(df4, df1d)

    if args.backtest:
        backtest(df4)
    else:
        logger.info(f"Live — {SYMBOL} 4h+Daily — checking every 4 hours... (VOLUME x{VOLUME_MULTIPLIER})")

        ws_thread = threading.Thread(target=run_websocket_forever, daemon=True)
        ws_thread.start()

        # === POSITION STATE RECOVERY ON STARTUP (restart-proof) ===
        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        be_trigger = 0.0
        at_be = False
        trail_stop = 0.0
        entry_bar = 0
        risk = 0.0
        binance_client = Client()
        last_daily_refresh = time.time()
        DAILY_REFRESH_INTERVAL = 43200  # 12 hours

        sol_bal = get_token_balance(SOL_MINT)
        if os.path.exists(POSITION_STATE_FILE) and sol_bal > SOL_RESERVE + 0.005:
            try:
                with open(POSITION_STATE_FILE, 'r') as f:
                    state = json.load(f)
                in_position = state['in_position']
                entry_price = state['entry_price']
                stop_price = state['stop_price']
                target_price = state['target_price']
                be_trigger = state['be_trigger']
                at_be = state['at_be']
                trail_stop = state['trail_stop']
                entry_bar = state.get('entry_bar', len(df4) - 1)
                entry_ts_str = state.get('entry_ts')
                if entry_ts_str:
                    try:
                        entry_ts = pd.Timestamp(entry_ts_str)
                        if entry_ts.tzinfo is None:
                            entry_ts = entry_ts.tz_localize('UTC')
                        if entry_ts in df4.index:
                            entry_bar = df4.index.get_loc(entry_ts)
                            logger.info(f"Recovered entry_bar={entry_bar} from timestamp {entry_ts}")
                        else:
                            logger.warning(f"Entry timestamp {entry_ts} not in df4 — using saved index {entry_bar}")
                    except Exception as e:
                        logger.warning(f"Entry timestamp parse failed: {e} — using saved index {entry_bar}")
                risk = state.get('risk', 0.0)
                # Enhanced recovery log — show entry timestamp and hours held
                hours_held_str = ""
                if entry_ts_str:
                    try:
                        et = pd.Timestamp(entry_ts_str)
                        if et.tzinfo is None:
                            et = et.tz_localize('UTC')
                        hours_held = (pd.Timestamp.now('UTC') - et).total_seconds() / 3600
                        hours_held_str = f" | Entered {entry_ts_str[:16]} UTC ({hours_held:.1f}h ago)"
                    except Exception:
                        pass
                logger.info(f"✅ RECOVERED live position from disk! Entry ~{entry_price:.4f} | SOL: {sol_bal:.6f}{hours_held_str}")
            except Exception as e:
                logger.warning(f"State file corrupted: {e}")
                if os.path.exists(POSITION_STATE_FILE):
                    os.remove(POSITION_STATE_FILE)
        elif os.path.exists(POSITION_STATE_FILE) and sol_bal <= SOL_RESERVE + 0.005:
            # SELF-HEAL: state file says in-position but wallet shows no SOL.
            # Happens when a sell confirmed on-chain but execute_swap's balance check
            # returned False (RPC latency etc). State is stale — clear it.
            logger.warning(f"⚠️  State file says in-position but SOL balance ({sol_bal:.6f}) is at or below reserve.")
            logger.warning("⚠️  Position appears to have been closed (sell confirmed on-chain after bot gave up). Clearing stale state.")
            try:
                with open(POSITION_STATE_FILE, 'r') as f:
                    stale = json.load(f)
                logger.info(f"   Stale state: entry={stale.get('entry_price')}, stop={stale.get('stop_price')}, target={stale.get('target_price')}")
            except Exception:
                pass
            os.remove(POSITION_STATE_FILE)
            send_alert("Bot Self-Healed: Stale Position Cleared",
                       f"Found state file claiming in-position but wallet has only {sol_bal:.6f} SOL (<= reserve). "
                       f"The earlier sell must have confirmed after the bot gave up. State file cleared automatically. "
                       f"Bot continuing normally, flat, waiting for next signal.")
            in_position = False
        elif sol_bal > SOL_RESERVE + 0.25:
            logger.warning(f"⚠️  SOL detected in wallet ({sol_bal:.6f}) but no state file found — MANUAL INTERVENTION REQUIRED. Bot paused.")
            send_alert("URGENT: Manual Intervention Required", f"SOL balance {sol_bal:.6f} found but no position state file.\nBot will not trade until state is restored manually.")
            raise SystemExit(1)

        while True:
            with df4_lock:
                df4_snap = df4.copy()
            logger.info(f"df4 len={len(df4_snap)}, last close={df4_snap['close'].iloc[-1]:.2f}")

            # ── Refresh daily data periodically ──────────────────────────
            if time.time() - last_daily_refresh > DAILY_REFRESH_INTERVAL:
                try:
                    logger.info("Refreshing daily data from Binance...")
                    _, df1d_fresh = fetch_data(days=args.days)
                    df1d = compute_daily(df1d_fresh)
                    last_daily_refresh = time.time()
                    logger.info(f"Daily data refreshed — last bar: {df1d.index[-1]}")
                except Exception as e:
                    logger.warning(f"Daily refresh failed (will retry next cycle): {e}")

            df4_snap = attach_daily(df4_snap, df1d)
            i = len(df4_snap) - 1
            c = df4_snap.iloc[i]
            p1 = df4_snap.iloc[i-1] if i > 0 else None
            now_str = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M')

            usdt_bal = get_token_balance(USDT_MINT)
            sol_bal  = get_token_balance(SOL_MINT)

            # RUNTIME SELF-HEAL: catch stale in_position during live run
            # (not just on startup). If we think we're in position but wallet
            # shows no SOL, a sell confirmed late on-chain — clear state.
            if in_position and sol_bal <= SOL_RESERVE + 0.005:
                logger.warning(f"⚠️  RUNTIME SELF-HEAL: in_position=True but SOL balance is only {sol_bal:.6f} (<= reserve)")
                if os.path.exists(POSITION_STATE_FILE):
                    os.remove(POSITION_STATE_FILE)
                in_position = False
                logger.info("✅ Stale position state cleared during runtime. Bot is now flat.")
                send_alert("SOL Bot Self-Healed (Runtime)",
                           f"Detected stale in-position flag during live trading.\n"
                           f"Wallet SOL balance: {sol_bal:.6f} (<= reserve). Previous sell likely confirmed late.\n"
                           f"State file cleared. Bot continuing flat.")

            logger.info(f"[{now_str}] WALLET — USDT: ${usdt_bal:,.2f} | SOL: {sol_bal:.6f} | In Position: {in_position}")

            macro = "DAILY BULL" if c['daily_bull'] == 1 else "DAILY BEAR"
            trend = "UP" if c['close'] > c['ema_trend'] else "DOWN"

            # ── Not in position: check for entry ─────────────────────────────
            if not in_position:
                signal = buy_signal(df4_snap, i, debug=True)
                if signal:
                    logger.info(f"\n{'='*44}\n  *** BUY SIGNAL — {now_str} ***\n  {macro} | 4h {trend}")
                    try:
                        entry_price = float(binance_client.get_symbol_ticker(symbol=SYMBOL)['price'])
                    except Exception as e:
                        logger.warning(f"Failed to fetch entry price: {e}")
                        time.sleep(60)
                        continue

                    atr        = c['atr']
                    swing_low  = df4_snap.iloc[max(0, i-2):i+1]['low'].min()
                    stop_price = swing_low - (ATR_BUFFER * atr)
                    risk       = entry_price - stop_price

                    if risk <= 0 or (risk / entry_price) > MAX_RISK_PCT:
                        logger.info(f"Skipping — risk {risk/entry_price*100:.1f}% invalid")
                        time.sleep(60)
                        continue

                    target_price = entry_price + (REWARD_RATIO * risk)
                    be_trigger   = entry_price + risk
                    trail_stop   = stop_price

                    deploy_usdt          = usdt_bal * POSITION_SIZE
                    amount_usdt_lamports = int(deploy_usdt * 10**6)

                    logger.info(f"Deploying: {deploy_usdt:.2f} USDT | Entry: ~{entry_price:.4f} | Stop: {stop_price:.4f} | Target: {target_price:.4f}")

                    tx_sig, confirmed = perform_buy(amount_usdt_lamports)
                    if tx_sig and confirmed:
                        in_position = True
                        entry_bar   = i
                        at_be       = False
                        effective_stop = max(stop_price, trail_stop)

                        state = {
                            'in_position': True,
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'target_price': target_price,
                            'be_trigger': be_trigger,
                            'at_be': at_be,
                            'trail_stop': trail_stop,
                            'entry_bar': entry_bar,
                            'entry_ts': df4_snap.index[i].isoformat(),
                            'risk': risk
                        }
                        with open(POSITION_STATE_FILE, 'w') as f:
                            json.dump(state, f)
                        logger.info("💾 Position state saved to disk")
                        logger.info(f"BUY EXECUTED! Tx: {str(tx_sig)[:20]}...")
                        send_alert(
                            "SOLUSDT Bot: BUY Executed",
                            f"BUY at {now_str}\nEntry: {entry_price:.4f}\nStop: {stop_price:.4f}\nTarget: {target_price:.4f}"
                        )
                    elif tx_sig and not confirmed:
                        logger.warning(f"⚠️ Buy TX unconfirmed — NOT entering position: {tx_sig}")
                    else:
                        logger.warning("Buy transaction failed")
                    logger.info(f"{'='*44}\n")
                else:
                    logger.info(f"[{now_str}] {macro} | 4h {trend} | Close={c['close']:.2f} RSI={c['rsi']:.1f} "
                                f"Vol={c['volume']:.0f} VolMA*{VOLUME_MULTIPLIER}={c['vol_ma']*VOLUME_MULTIPLIER:.0f}")

            # ── In position: check exits ──────────────────────────────────────
            else:
                bars_held = i - entry_bar

                if not at_be and c['high'] >= be_trigger:
                    stop_price = entry_price
                    at_be      = True
                    logger.info(f"Breakeven stop activated at {entry_price:.4f}")

                if c['close'] > entry_price + (2 * risk):
                    recent_high = df4_snap.iloc[max(0, i-2):i+1]['high'].max()
                    new_trail   = recent_high - (2 * c['atr'])
                    trail_stop  = max(trail_stop, new_trail)

                effective_stop = max(stop_price, trail_stop)

                # Save updated stop levels to disk every cycle
                try:
                    with open(POSITION_STATE_FILE, 'r') as f:
                        saved_state = json.load(f)
                    saved_state['trail_stop'] = trail_stop
                    saved_state['stop_price'] = stop_price
                    saved_state['at_be'] = at_be
                    with open(POSITION_STATE_FILE, 'w') as f:
                        json.dump(saved_state, f)
                except Exception as e:
                    logger.warning(f"Failed to save trail stop: {e}")

                exit_reason = None
                exit_price  = None

                if c['low'] <= effective_stop:
                    exit_price  = effective_stop
                    exit_reason = 'Trailing Stop' if trail_stop > stop_price and c['low'] <= trail_stop else ('Breakeven Stop' if at_be else 'Stop Loss')
                elif c['high'] >= target_price:
                    exit_price  = target_price
                    exit_reason = 'Take Profit'
                elif p1 is not None and c['close'] < c['ema_trend'] and p1['close'] < p1['ema_trend']:
                    exit_price  = c['close']
                    exit_reason = 'Trend Exit'
                elif c['daily_bull'] == 0:
                    exit_price  = c['close']
                    exit_reason = 'Daily Trend Exit'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price  = c['close']
                    exit_reason = 'Timeout'

                if exit_reason:
                    logger.info(f"\n{'='*44}\n  *** EXIT — {exit_reason} — {now_str} ***")
                    sol_balance         = get_token_balance(SOL_MINT)
                    amount_sol          = max(sol_balance - SOL_RESERVE, 0)
                    amount_sol_lamports = int(amount_sol * 10**9)

                    tx_sig, confirmed = perform_sell(amount_sol_lamports)
                    if tx_sig and confirmed:
                        logger.info(f"SELL EXECUTED! Tx: {str(tx_sig)[:20]}...")
                        if os.path.exists(POSITION_STATE_FILE):
                            os.remove(POSITION_STATE_FILE)
                            logger.info("💾 Position state cleared")
                        send_alert("SOLUSDT Bot: EXIT Executed", f"EXIT: {exit_reason} at {now_str}\nExit Price: {exit_price:.4f}")
                        in_position = False  # ← only on confirmed
                    elif tx_sig and not confirmed:
                        logger.warning(f"⚠️ Sell TX unconfirmed — MANUAL CHECK REQUIRED: {tx_sig}")
                        send_alert("URGENT: Unconfirmed Sell", f"Tx sent but NOT confirmed!\nSig: {tx_sig}\nMANUAL check required!")
                        # leave in_position = True — retry next bar
                    else:
                        logger.warning("Sell failed — will retry next bar")
                        # leave in_position = True — retry next bar
                    logger.info(f"{'='*44}\n")
                else:
                    pnl_pct = ((c['close'] - entry_price) / entry_price) * 100
                    logger.info(f"[{now_str}] In Position | Entry={entry_price:.4f} | Current={c['close']:.4f} | PnL={pnl_pct:+.2f}% | Stop={effective_stop:.4f}")

            # ── Intra-bar monitoring (runs for any open position) ─────────
            if in_position:
                # Belt-and-suspenders: ensure effective_stop is defined
                # regardless of which code path got us here.
                effective_stop = max(stop_price, trail_stop)

                last_bar_time = df4_snap.index[-1]
                if last_bar_time.tzinfo is None:
                    last_bar_time = last_bar_time.tz_localize('UTC')
                next_bar_time = last_bar_time + dt.timedelta(hours=8)

                logger.info(f"Monitoring intra-bar until {next_bar_time.strftime('%Y-%m-%d %H:%M')} UTC")

                while pd.Timestamp.now('UTC') < next_bar_time:
                    current_price = None
                    for attempt in range(3):
                        try:
                            current_price = float(binance_client.get_symbol_ticker(symbol=SYMBOL)['price'])
                            break
                        except:
                            time.sleep(10)
                    if current_price is None:
                        time.sleep(60)
                        continue

                    now_intra = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S')

                    if current_price <= effective_stop:
                        logger.info(f"\n{'='*44}\n  *** INTRA-BAR STOP HIT — {now_intra} ***")
                        sol_balance         = get_token_balance(SOL_MINT)
                        amount_sol          = max(sol_balance - SOL_RESERVE, 0)
                        amount_sol_lamports = int(amount_sol * 10**9)
                        sig, confirmed = perform_sell(amount_sol_lamports)
                        if confirmed:
                            if os.path.exists(POSITION_STATE_FILE):
                                os.remove(POSITION_STATE_FILE)
                            send_alert("SOLUSDT Bot: EXIT Executed", f"EXIT: Intra-bar Stop at {now_intra}\nExit Price: {current_price:.4f}")
                            in_position = False
                            break
                        elif sig and not confirmed:
                            logger.warning(f"⚠️ Intra-bar stop sell unconfirmed — MANUAL CHECK: {sig}")
                            send_alert("URGENT: Unconfirmed Intra-bar Stop Sell", f"Tx sent but NOT confirmed!\nSig: {sig}\nMANUAL check required!")
                            break
                        else:
                            logger.warning("Intra-bar stop sell failed — retrying next loop")

                    elif current_price >= target_price:
                        logger.info(f"\n{'='*44}\n  *** INTRA-BAR TARGET HIT — {now_intra} ***")
                        sol_balance         = get_token_balance(SOL_MINT)
                        amount_sol          = max(sol_balance - SOL_RESERVE, 0)
                        amount_sol_lamports = int(amount_sol * 10**9)
                        sig, confirmed = perform_sell(amount_sol_lamports)
                        if confirmed:
                            if os.path.exists(POSITION_STATE_FILE):
                                os.remove(POSITION_STATE_FILE)
                            send_alert("SOLUSDT Bot: EXIT Executed", f"EXIT: Intra-bar Target at {now_intra}\nExit Price: {current_price:.4f}")
                            in_position = False
                            break
                        
                        elif sig and not confirmed:
                            logger.warning(f"⚠️ Intra-bar target sell unconfirmed — MANUAL CHECK: {sig}")
                            send_alert("URGENT: Unconfirmed Intra-bar Target Sell", f"Tx sent but NOT confirmed!\nSig: {sig}\nMANUAL check required!")
                            break
                        else:
                            logger.warning("Intra-bar target sell failed — retrying next loop")

                    else:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"  [{now_intra}] Price={current_price:.4f} | PnL={pnl_pct:+.2f}%")

                    time.sleep(60)

            # ── Sleep until next bar close (only when flat) ────────────────
            if not in_position:
                with df4_lock:
                    last_bar_time = df4.index[-1]
                if last_bar_time.tzinfo is None:
                    last_bar_time = last_bar_time.tz_localize('UTC')

                next_check = last_bar_time + pd.Timedelta(hours=4)
                while next_check <= pd.Timestamp.now('UTC'):
                    next_check += pd.Timedelta(hours=4)

                sleep_secs = (next_check - pd.Timestamp.now('UTC')).total_seconds() + 10
                if sleep_secs > 0:
                    logger.info(f"Sleeping {sleep_secs/3600:.2f}h until {next_check.strftime('%Y-%m-%d %H:%M')} UTC...")
                    time.sleep(sleep_secs)
                else:
                    time.sleep(60)
            else:
                # In position — skip the long sleep, go straight back to monitoring.
                # Short sleep to let the WS catch up with any just-closed bar.
                time.sleep(5)

atexit.register(stop_ws)
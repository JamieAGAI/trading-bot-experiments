import os
import time
import argparse
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from solders.keypair import Keypair
import base58
import tensorflow as tf
from collections import deque
import random
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.message import to_bytes_versioned
import base64
import pickle
from datetime import datetime, timedelta

# Load environment variables
load_dotenv(override=True)

# Constants
SOL_MINT = "So11111111111111111111111111111111111111112"  # WSOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
EMA_SHORT, EMA_LONG, RSI_PERIOD, ADX_PERIOD, ATR_PERIOD = 9, 21, 14, 14, 14
TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX = 700, 1500  # Start small, scale with profits
MIN_PROFIT_THRESHOLD = 0.002  # 0.2%
POST_SELL_BUY_DELAY = 300  # 5 min
MAX_SOL_POSITION, INITIAL_USDC = 6, 1500  # Small start
LOG_FILE = "trades.csv"
MODEL_DIR = "dql_model"
BULL_MODEL_FILE = f"{MODEL_DIR}/bull_model.keras"
BEAR_MODEL_FILE = f"{MODEL_DIR}/bear_model.keras"
SIDEWAYS_MODEL_FILE = f"{MODEL_DIR}/sideways_model.keras"
BULL_EPS_FILE = f"{MODEL_DIR}/bull_epsilon.pkl"
BEAR_EPS_FILE = f"{MODEL_DIR}/bear_epsilon.pkl"
SIDEWAYS_EPS_FILE = f"{MODEL_DIR}/sideways_epsilon.pkl"
SLIPPAGE = 0.01  # 1%
STOP_LOSS, TAKE_PROFIT = -0.03, 0.06  # Tighter: 3% SL, 6% TP
REFRESH_INTERVAL = 300  # 5 min
MAX_LOSS = -100  # Stop if total loss >$200
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
DATA_FILE = "sol_15min_data.csv"
BITGET_BASE = "https://api.bitget.com"

# DQL Params (tuned for stability)
STATE_SIZE, ACTION_SIZE = 17, 3  # Added MACD hist and regime code
GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY = 0.95, 0.3, 0.05, 0.99998  # Slower decay
LEARNING_RATE, MEMORY_SIZE, BATCH_SIZE = 0.0005, 20000, 32  # Larger memory
TARGET_UPDATE_FREQ = 500

# Solana setup
alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
if not alchemy_api_key:
    print("ALCHEMY_API_KEY not found. Exiting.")
    exit(1)
solana_client = Client(f"https://solana-mainnet.g.alchemy.com/v2/{alchemy_api_key}")

# Wallet (only print if not backtest)
try:
    private_key_base58 = os.getenv("PHANTOM_PRIVATE_KEY")
    private_key_bytes = base58.b58decode(private_key_base58)
    wallet = Keypair.from_bytes(private_key_bytes)
except Exception as e:
    print(f"Wallet error: {e}")
    exit(1)

# Jupiter
JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Rate limiters
solana_requests = deque(maxlen=100)
BITGET_RATE_LIMIT = 20
BITGET_WINDOW = 1
bitget_requests = deque(maxlen=100)  # Separate for Bitget

def enforce_rate_limit(requests_deque, limit, window):
    current = time.time()
    while requests_deque and current - requests_deque[0] > window:
        requests_deque.popleft()
    if len(requests_deque) >= limit:
        sleep = window - (current - requests_deque[0])
        if sleep > 0:
            time.sleep(sleep)
    requests_deque.append(current)

# Bitget functions (fixed with v2 endpoints and 8 columns)
def fetch_bitget_price():
    url = f"{BITGET_BASE}/api/v2/spot/market/tickers?symbol=SOLUSDT"
    try:
        enforce_rate_limit(bitget_requests, BITGET_RATE_LIMIT, BITGET_WINDOW)
        resp = requests.get(url).json()
        if resp.get('code') == '00000':
            return float(resp['data'][0]['lastPr'])
        else:
            print(f"Price API error: {resp}")
            return None
    except Exception as e:
        print(f"Price fetch error: {e}")
        return None

def fetch_bitget_historical(interval='15min', days=7, symbol='SOLUSDT'):
    if days <= 0:
        days = 1  # Minimum
    all_data = []
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    interval_ms = int(pd.Timedelta(interval).total_seconds() * 1000)
    chunk_size = 500 * interval_ms  # 1000 candles worth of ms
    current_end = end_time
    while current_end > start_time:
        current_start = max(start_time, current_end - chunk_size)
        url = f"{BITGET_BASE}/api/v2/spot/market/candles?symbol={symbol}&granularity={interval}&startTime={current_start}&endTime={current_end}&limit=1000"
        try:
            enforce_rate_limit(bitget_requests, BITGET_RATE_LIMIT, BITGET_WINDOW)
            resp = requests.get(url).json()
            if resp.get('code') != '00000':
                print(f"API error: {resp}")
                break
            candles = resp.get('data', [])
            if candles:
                df_chunk = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'baseVol', 'quoteVol', 'usdtVol'])
                df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'].astype(int), unit='ms')
                df_chunk = df_chunk.sort_values('timestamp').drop_duplicates('timestamp')  # Sort oldest to newest
                all_data.append(df_chunk)
                current_end = int(df_chunk['timestamp'].iloc[0].timestamp() * 1000) - 1  # Move to before this chunk's start
            else:
                print(f"No candles in chunk {current_start} to {current_end}. Skipping.")
                current_end = current_start - 1  # Avoid loop
        except Exception as e:
            print(f"Fetch error: {e}")
            break
        time.sleep(0.1)  # Rate limit buffer
    if all_data:
        df = pd.concat(all_data).sort_values('timestamp').reset_index(drop=True)
        df[['open', 'high', 'low', 'close', 'baseVol', 'quoteVol', 'usdtVol']] = df[['open', 'high', 'low', 'close', 'baseVol', 'quoteVol', 'usdtVol']].astype(float)
        df.to_csv(DATA_FILE, index=False)
        print(f"Fetched {len(df)} rows of data.")
        return df
    else:
        print("No data fetched. Trying cached...")
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Used cached {len(df)} rows.")
            return df
        return pd.DataFrame()

# Indicators
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral fill for NaNs

def calculate_macd(prices, short=12, long=26, signal=9):
    ema_short = calculate_ema(prices, short)
    ema_long = calculate_ema(prices, long)
    macd = ema_short - ema_long
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_hist

def calculate_bollinger_width(prices, period=20, std=2):
    sma = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    width = (upper - lower) / sma  # Normalized width
    return width.fillna(0)

def calculate_tr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean()
    return tr.fillna(0)

def calculate_di(high, low, close, period=14):
    dm_plus = ((high - high.shift()) > (low.shift() - low)) * (high - high.shift()).clip(lower=0)
    dm_minus = ((low.shift() - low) > (high - high.shift())) * (low.shift() - low).clip(lower=0)
    tr = calculate_tr(high, low, close, period)
    di_plus = 100 * (dm_plus.rolling(period).mean() / tr)
    di_minus = 100 * (dm_minus.rolling(period).mean() / tr)
    return di_plus.fillna(0), di_minus.fillna(0)

def calculate_adx(high, low, close, period=14):
    di_plus, di_minus = calculate_di(high, low, close, period)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)  # Avoid div0
    adx = dx.rolling(period).mean()
    return adx.fillna(0), di_plus, di_minus

def calculate_atr(high, low, close, period=14):
    tr = calculate_tr(high, low, close, period)
    return tr.rolling(period).mean().fillna(0)

def normalize(series):
    if len(series) == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val < 1e-8:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

def calculate_success_score(df, i, norm_ema_s, norm_ema_l, norm_rsi, norm_macd, norm_bb, norm_adx, norm_di_p, norm_di_m, norm_atr, recent_returns, norm_volume):
    if i < 0 or len(df) <= i:
        return 0.5
    ema_diff = norm_ema_s.iloc[i] - norm_ema_l.iloc[i]
    rsi_val = norm_rsi.iloc[i]
    macd_val = norm_macd.iloc[i]
    adx_val = norm_adx.iloc[i]
    di_diff = norm_di_p.iloc[i] - norm_di_m.iloc[i]
    atr_val = norm_atr.iloc[i]
    recent_ret = recent_returns[-1] if len(recent_returns) > 0 else 0
    bb_val = norm_bb.iloc[i]
    norm_volume_val = norm_volume.iloc[i] if isinstance(norm_volume, pd.Series) else norm_volume
    
    trend_score = 0.20 * ema_diff + 0.10 * macd_val + 0.10 * di_diff + 0.10 * recent_ret
    rev_score = 0.30 * (0.5 - abs(rsi_val - 0.5))
    vol_score = 0.20 * (1 - atr_val) - 0.10 * bb_val + 0.10 * (norm_volume_val - 0.5)
    bear_penalty = -0.15 * di_diff if di_diff < 0 else 0
    score = trend_score + rev_score + vol_score + adx_val * 0.1 + bear_penalty
    return max(0, min(1, (score + 1) / 2))

# 1h trend filter
def get_1h_trend(hist_df=None):
    try:
        if hist_df is None:
            df_1h = fetch_bitget_historical('1h', days=2)
        else:
            df_1h = hist_df.set_index('timestamp').resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'usdtVol': 'sum'
            }).dropna().reset_index()
        if len(df_1h) < EMA_LONG:
            return 0
        ema_s_1h = calculate_ema(df_1h['close'], EMA_SHORT)
        ema_l_1h = calculate_ema(df_1h['close'], EMA_LONG)
        if pd.isna(ema_s_1h.iloc[-1]) or pd.isna(ema_l_1h.iloc[-1]):
            return 0
        return 1 if ema_s_1h.iloc[-1] > ema_l_1h.iloc[-1] else -1
    except Exception as e:
        print(f"1h trend error: {e}")
        return 0

# Market regime
def get_market_regime(norm_adx, trend_1h, norm_bb):
    if norm_adx > 0.3 and trend_1h == 1:
        return "bull"
    elif norm_adx > 0.3 and trend_1h == -1:
        return "bear"
    else:
        return "sideways"

# Wallet balances (cached)
last_balance = 0
cached_sol, cached_usdc = 0, 0
CACHE_DUR = 300

def get_balances(force=False):
    global last_balance, cached_sol, cached_usdc
    if not force and time.time() - last_balance < CACHE_DUR:
        return cached_sol, cached_usdc
    enforce_rate_limit(solana_requests, 100, 1)
    try:
        sol = solana_client.get_balance(wallet.pubkey()).value / 1e9
        usdc_ata = Pubkey.find_program_address([bytes(wallet.pubkey()), bytes(SPL_TOKEN_PROGRAM_ID), bytes(Pubkey.from_string(USDC_MINT))], ASSOCIATED_TOKEN_PROGRAM_ID)[0]
        usdc_resp = solana_client.get_token_account_balance(usdc_ata)
        usdc = usdc_resp.value.ui_amount if usdc_resp.value else 0
        cached_sol, cached_usdc = sol, usdc
        last_balance = time.time()
        return sol, usdc
    except Exception as e:
        print(f"Balance fetch error: {e}")
        return cached_sol, cached_usdc

# Log
def log_trade(ts, typ, price, sol_amt, usdc_amt, profit, fees, curr_price, pnl, drawdown):
    net = profit - fees if typ == 'SELL' else 0
    entry = pd.DataFrame({
        'timestamp': pd.to_datetime(ts, unit='s'),
        'unix': ts,
        'type': typ,
        'price_usd_sol': price,
        'sol': sol_amt,
        'usdc': usdc_amt,
        'profit': profit,
        'fees': fees,
        'net': net,
        'curr_price': curr_price,
        'pnl': pnl,
        'drawdown': drawdown
    }, index=[0])
    if not os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    print(f"Trade: {typ} {price:.2f} USD/SOL, SOL:{sol_amt:.4f}, USDC:{usdc_amt:.2f}, Profit:{profit:.2f}, Fees:{fees:.2f}")

# ATA init
def initialize_ata(mint):
    mint_pk = Pubkey.from_string(mint)
    ata, _ = Pubkey.find_program_address([bytes(wallet.pubkey()), bytes(SPL_TOKEN_PROGRAM_ID), bytes(mint_pk)], ASSOCIATED_TOKEN_PROGRAM_ID)
    try:
        solana_client.get_account_info(ata)
        return ata
    except:
        pass
    instr = Instruction(
        program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
        accounts=[
            AccountMeta(wallet.pubkey(), True, True),
            AccountMeta(ata, False, True),
            AccountMeta(wallet.pubkey(), False, False),
            AccountMeta(mint_pk, False, False),
            AccountMeta(Pubkey.from_string('11111111111111111111111111111111'), False, False),  # System
            AccountMeta(SPL_TOKEN_PROGRAM_ID, False, False)
        ],
        data=bytes([])
    )
    for _ in range(3):
        try:
            bh = solana_client.get_latest_blockhash().value.blockhash
            msg = MessageV0.try_compile(
                payer=wallet.pubkey(),
                instructions=[instr],
                address_lookup_table_accounts=[],
                recent_blockhash=bh
            )
            tx = VersionedTransaction(msg, [wallet])
            sig = solana_client.send_transaction(tx).value
            solana_client.confirm_transaction(sig, commitment=Confirmed)
            return ata
        except Exception as e:
            print(f"ATA init retry error: {e}")
            time.sleep(5)
    print("ATA init failed after retries.")
    return None

# DQL Agent (fixed model with Input layer to avoid warning)
class DQNAgent:
    def __init__(self, state_size, action_size, model_file, eps_file):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model_file = model_file
        self.eps_file = eps_file
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.load_model()
        self.updates = 0

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = tf.keras.models.load_model(self.model_file)
            self.target_model = tf.keras.models.load_model(self.model_file)
            self.target_model.set_weights(self.model.get_weights())
        if os.path.exists(self.eps_file):
            with open(self.eps_file, 'rb') as f:
                self.epsilon = pickle.load(f)

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(self.model_file)
        with open(self.eps_file, 'wb') as f:
            pickle.dump(self.epsilon, f)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        targets = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        self.updates += 1
        if self.updates % TARGET_UPDATE_FREQ == 0:
            self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Ensemble DQN Agent
class EnsembleDQNAgent:
    def __init__(self):
        self.bull_agent = DQNAgent(STATE_SIZE, ACTION_SIZE, BULL_MODEL_FILE, BULL_EPS_FILE)
        self.bear_agent = DQNAgent(STATE_SIZE, ACTION_SIZE, BEAR_MODEL_FILE, BEAR_EPS_FILE)
        self.sideways_agent = DQNAgent(STATE_SIZE, ACTION_SIZE, SIDEWAYS_MODEL_FILE, SIDEWAYS_EPS_FILE)

    def act(self, state, regime):
        if regime == "bull":
            return self.bull_agent.act(state)
        elif regime == "bear":
            return self.bear_agent.act(state)
        else:
            return self.sideways_agent.act(state)

    def remember(self, state, action, reward, next_state, done, regime):
        if regime == "bull":
            self.bull_agent.remember(state, action, reward, next_state, done)
        elif regime == "bear":
            self.bear_agent.remember(state, action, reward, next_state, done)
        else:
            self.sideways_agent.remember(state, action, reward, next_state, done)

    def replay(self, regime):
        if regime == "bull":
            self.bull_agent.replay()
        elif regime == "bear":
            self.bear_agent.replay()
        else:
            self.sideways_agent.replay()

    def save_model(self):
        self.bull_agent.save_model()
        self.bear_agent.save_model()
        self.sideways_agent.save_model()

# Jupiter swap
def execute_swap(input_mint, output_mint, amount, retries=3):
    decimals_in = 6 if input_mint == USDC_MINT else 9
    amount_lamports = int(amount * 10**decimals_in)
    if amount_lamports < 10**(decimals_in - 2):  # Min 0.01
        return False, None, 0, 0

    sol, usdc = get_balances()
    if input_mint == USDC_MINT and usdc < amount or input_mint == SOL_MINT and sol < amount:
        return False, None, 0, 0

    input_ata = initialize_ata(input_mint)
    output_ata = initialize_ata(output_mint)
    if not input_ata or not output_ata:
        return False, None, 0, 0

    quote_url = f"{JUPITER_API_BASE}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}&slippageBps={int(SLIPPAGE*10000)}"
    for attempt in range(retries):
        try:
            quote = requests.get(quote_url).json()
            if 'error' in quote:
                print(f"Quote error: {quote['error']}")
                time.sleep(5)
                continue
            in_amt = int(quote['inAmount'])
            out_amt = int(quote['outAmount'])
            decimals_out = 9 if output_mint == SOL_MINT else 6
            out_decimal = out_amt / 10**decimals_out
            price = amount / out_decimal if input_mint == USDC_MINT else out_decimal / amount  # USD/SOL approx

            jup_fee = out_decimal * 0.001  # 0.1%
            net_fee_usd = jup_fee * price if output_mint == SOL_MINT else jup_fee
            net_fee_usd += 0.000005 * price  # Solana fee

            payload = {
                "quoteResponse": quote,
                "userPublicKey": str(wallet.pubkey()),
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": 100000
            }
            swap_resp = requests.post(f"{JUPITER_API_BASE}/swap", json=payload).json()
            print(f"[Debug] Swap response: {swap_resp}")
            if 'error' in swap_resp:
                print(f"Swap error: {swap_resp['error']}")
                time.sleep(5)
                continue
            tx_bytes = base64.b64decode(swap_resp['swapTransaction'])
            tx = VersionedTransaction.from_bytes(tx_bytes)
            msg_bytes = to_bytes_versioned(tx.message)
            sig = wallet.sign_message(msg_bytes)
            tx.signatures = [sig]  # Replace the null signature with your actual one

            bh = solana_client.get_latest_blockhash().value.blockhash
            sig = solana_client.send_raw_transaction(bytes(tx)).value
            solana_client.confirm_transaction(sig, commitment=Confirmed, sleep_seconds=5)
            return True, price, out_decimal, net_fee_usd
        except Exception as e:
            print(f"Swap attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False, None, 0, 0

# Backtest function (fixed normalization and iloc errors)
def backtest(days=7):
    print("Starting backtest...")
    df = fetch_bitget_historical(days=days)
    prices = df['close']
    high = df['high']
    low = df['low']
    close = df['close']
    usdtVol = df['usdtVol']  # Add this for volume
    if len(df) < EMA_LONG + ADX_PERIOD + 10:  # Need more data for indicators
        print("Insufficient data for backtest.")
        return

    agent = EnsembleDQNAgent()
    position = None
    total_pnl = 0
    trades = 0
    wins = 0
    recent_returns = deque(maxlen=5)
    usdc = INITIAL_USDC
    sol = 0
    max_usdc = INITIAL_USDC

    for i in range(EMA_LONG + ADX_PERIOD, len(df)):  # Start after enough data for all indicators
        row = df.iloc[i]
        price = row['close']

        # Slice historical data up to i
        hist_prices = prices.iloc[:i+1]
        hist_high = high.iloc[:i+1]
        hist_low = low.iloc[:i+1]
        hist_close = close.iloc[:i+1]  # Alias for clarity
        hist_usdtVol = usdtVol.iloc[:i+1]  # Add historical volume slice

        # Compute indicators and normalizations on historical data only
        norm_prices = normalize(hist_prices)
        ema_s_hist = calculate_ema(hist_prices, EMA_SHORT)
        norm_ema_s = normalize(ema_s_hist)
        ema_l_hist = calculate_ema(hist_prices, EMA_LONG)
        norm_ema_l = normalize(ema_l_hist)
        rsi_hist = calculate_rsi(hist_prices, RSI_PERIOD)
        norm_rsi = normalize(rsi_hist)
        macd_hist, macd_hist_hist = calculate_macd(hist_prices)  # Updated
        norm_macd = normalize(macd_hist)
        norm_macd_hist = normalize(macd_hist_hist)  # New
        bb_hist = calculate_bollinger_width(hist_prices)
        norm_bb = normalize(bb_hist)
        adx_hist, di_p_hist, di_m_hist = calculate_adx(hist_high, hist_low, hist_close, ADX_PERIOD)
        norm_adx = normalize(adx_hist)
        norm_di_p = normalize(di_p_hist)
        norm_di_m = normalize(di_m_hist)
        atr_hist = calculate_atr(hist_high, hist_low, hist_close, ATR_PERIOD)
        norm_atr = normalize(atr_hist)
        norm_usdtVol = normalize(hist_usdtVol)  # Add normalized volume

        # Get current (last) values
        norm_price_val = norm_prices.iloc[-1]
        norm_ema_s_val = norm_ema_s.iloc[-1]
        norm_ema_l_val = norm_ema_l.iloc[-1]
        norm_rsi_val = norm_rsi.iloc[-1]
        norm_macd_val = norm_macd.iloc[-1]
        norm_bb_val = norm_bb.iloc[-1]
        norm_adx_val = norm_adx.iloc[-1]
        norm_di_p_val = norm_di_p.iloc[-1]
        norm_di_m_val = norm_di_m.iloc[-1]
        norm_atr_val = norm_atr.iloc[-1]
        norm_volume_val = norm_usdtVol.iloc[-1]  # Add current normalized volume
        norm_macd_hist_val = norm_macd_hist.iloc[-1]  # New

        # Recent return
        recent_ret = (hist_prices.iloc[-1] - hist_prices.iloc[-2]) / hist_prices.iloc[-2] if len(hist_prices) > 1 else 0
        recent_returns.append(recent_ret)

        trend_1h = get_1h_trend(df.iloc[:i+1])

        regime = get_market_regime(norm_adx_val, trend_1h, norm_bb_val)
        regime_code = 0 if regime == "bull" else 1 if regime == "bear" else 2  # Add to state

        state = [
            norm_price_val, norm_ema_s_val, norm_ema_l_val,
            norm_rsi_val, norm_macd_val, norm_bb_val,
            norm_adx_val, norm_di_p_val, norm_di_m_val, norm_atr_val,
            1 if position else 0, position['entry_price'] / price if position else 0.5,
            trend_1h / 2 + 0.5, recent_ret, norm_volume_val,
            norm_macd_hist_val, regime_code / 2  # Added hist and regime
        ]

        action = agent.act(state, regime)
        reward = 0
        done = False
        unrealized = (price - position['entry_price']) * position['sol_amt'] if position else 0
        drawdown = max_usdc - (usdc + sol * price)
        reward -= 0.1 * drawdown if drawdown < 0 else 0  # Penalize drawdown

        cooldown = max(3600, 7200 * norm_atr_val)  # Higher ATR = longer wait (in seconds, approx 15min intervals)

        if position:
            ret = (price - position['entry_price']) / position['entry_price']
            if ret <= STOP_LOSS or ret >= TAKE_PROFIT or action == 2:
                action = 2  # Force sell on SL/TP
            if action == 2 and (i - position['idx']) * 15 >= cooldown / 60:  # Simulate time (15min per candle)
                profit = (price - position['entry_price']) * position['sol_amt']
                fees = position['fees'] * 2  # Round trip est
                total_pnl += profit - fees
                usdc += position['sol_amt'] * price
                sol -= position['sol_amt']
                log_trade(int(row['timestamp'].timestamp()), 'SELL', price, position['sol_amt'], position['sol_amt'] * price, profit, fees, price, 0, drawdown)
                trades += 1
                if profit > fees:
                    wins += 1
                reward = profit - fees
                position = None
                done = True
            else:
                reward = 0.05 * unrealized  # Small unrealized reward

        else:
            if action == 1 and usdc >= TRADE_AMOUNT_MIN and sol <= MAX_SOL_POSITION and norm_rsi_val < (RSI_OVERBOUGHT / 100) and norm_adx_val > 0.3 and trend_1h == 1 and regime == "bull":
                score = calculate_success_score(df.iloc[:i+1], i, norm_ema_s, norm_ema_l, norm_rsi, norm_macd, norm_bb, norm_adx, norm_di_p, norm_di_m, norm_atr, recent_returns, norm_usdtVol)
                if score > 0.6:
                    risk_factor = 1 - norm_atr_val  # Smaller size in high ATR
                    buy_amt = TRADE_AMOUNT_MIN + (TRADE_AMOUNT_MAX - TRADE_AMOUNT_MIN) * score * risk_factor
                    buy_amt = min(buy_amt, usdc)
                    sol_bought = buy_amt / price
                    if sol + sol_bought > MAX_SOL_POSITION:
                        sol_bought = MAX_SOL_POSITION - sol
                        buy_amt = sol_bought * price
                    fees = (buy_amt / price) * 0.001 * price  # 0.1%
                    position = {'entry_price': price, 'sol_amt': sol_bought, 'usdc_in': buy_amt, 'fees': fees, 'idx': i}
                    usdc -= buy_amt
                    sol += sol_bought
                    log_trade(int(row['timestamp'].timestamp()), 'BUY', price, sol_bought, buy_amt, 0, fees, price, 0, drawdown)
                    reward = 0

        next_state = state
        agent.remember(state, action, reward, next_state, done, regime)
        agent.replay(regime)

        max_usdc = max(max_usdc, usdc + sol * price)

    if position:
        final_price = df.iloc[-1]['close']
        profit = (final_price - position['entry_price']) * position['sol_amt']
        fees = position['fees'] * 2  # Round trip est
        total_pnl += profit - fees
        usdc += position['sol_amt'] * final_price

    win_rate = (wins / trades * 100) if trades else 0
    print(f"Backtest complete: Total PnL: ${total_pnl:.2f}, Win Rate: {win_rate:.1f}%, Trades: {trades}, Final USDC: {usdc:.2f}")
    agent.save_model()

# Live run
def run_live():
    print(f"Wallet: {wallet.pubkey()}")
    print("Starting live bot...")
    agent = EnsembleDQNAgent()
    position = None
    last_buy = last_sell = 0
    total_pnl = total_fees = 0
    recent_returns = deque(maxlen=5)
    last_save = 0
    sol_ata = initialize_ata(SOL_MINT)
    usdc_ata = initialize_ata(USDC_MINT)

    while True:
        try:
            price = fetch_bitget_price()
            if not price:
                time.sleep(REFRESH_INTERVAL)
                continue

            df = fetch_bitget_historical(days=1)
            if len(df) < EMA_LONG + ADX_PERIOD:
                time.sleep(REFRESH_INTERVAL)
                continue

            # Pre-compute for live (similar to backtest)
            prices = df['close']
            high = df['high']
            low = df['low']
            close = df['close']
            usdtVol = df['usdtVol']

            ema_s = calculate_ema(prices, EMA_SHORT).iloc[-1]
            ema_l = calculate_ema(prices, EMA_LONG).iloc[-1]
            rsi = calculate_rsi(prices, RSI_PERIOD).iloc[-1]
            macd, macd_hist = calculate_macd(prices)
            macd_val = macd.iloc[-1]
            macd_hist_val = macd_hist.iloc[-1]  # New
            bb_w = calculate_bollinger_width(prices).iloc[-1]
            adx, di_p, di_m = calculate_adx(high, low, close, ADX_PERIOD)
            adx_val = adx.iloc[-1]
            atr_series = calculate_atr(high, low, close, ATR_PERIOD)
            atr = atr_series.iloc[-1]
            recent_ret = (price - prices.iloc[-2]) / prices.iloc[-2] if len(prices) > 1 else 0
            recent_returns.append(recent_ret)

            norm_prices = normalize(prices)
            norm_ema_s = normalize(calculate_ema(prices, EMA_SHORT))
            norm_ema_l = normalize(calculate_ema(prices, EMA_LONG))
            norm_rsi = normalize(calculate_rsi(prices, RSI_PERIOD))
            norm_macd = normalize(macd)
            norm_macd_hist = normalize(macd_hist)  # New
            norm_bb = normalize(calculate_bollinger_width(prices))
            norm_adx = normalize(adx)
            norm_di_p = normalize(di_p)
            norm_di_m = normalize(di_m)
            norm_atr = normalize(atr_series)
            norm_usdtVol = normalize(usdtVol)

            i = len(prices) - 1
            score = calculate_success_score(df, i, norm_ema_s, norm_ema_l, norm_rsi, norm_macd, norm_bb, norm_adx, norm_di_p, norm_di_m, norm_atr, recent_returns, norm_usdtVol.iloc[-1])
            trend_1h = get_1h_trend(None)

            regime = get_market_regime(norm_adx.iloc[-1], trend_1h, norm_bb.iloc[-1])
            regime_code = 0 if regime == "bull" else 1 if regime == "bear" else 2

            state = [
                norm_prices.iloc[-1], norm_ema_s.iloc[-1], norm_ema_l.iloc[-1],
                norm_rsi.iloc[-1], norm_macd.iloc[-1], norm_bb.iloc[-1],
                norm_adx.iloc[-1], norm_di_p.iloc[-1], norm_di_m.iloc[-1], norm_atr.iloc[-1],
                1 if position else 0, position['entry_price'] / price if position else 0.5,
                trend_1h / 2 + 0.5, recent_ret, norm_usdtVol.iloc[-1],
                norm_macd_hist.iloc[-1], regime_code / 2
            ]

            sol, usdc = get_balances()
            action = agent.act(state, regime)
            ts = int(time.time())
            unrealized = (price - position['entry_price']) * position['sol_amt'] if position else 0
            drawdown = INITIAL_USDC - (usdc + sol * price)
            reward = 0.1 * unrealized if position else 0
            reward -= 0.1 * drawdown if drawdown < 0 else 0

            cooldown = max(3600, 7200 * (atr / price))

            if position:
                ret = (price - position['entry_price']) / position['entry_price']
                if ret <= STOP_LOSS or ret >= TAKE_PROFIT or action == 2:
                    action = 2
                if action == 2 and (ts - last_sell) >= cooldown and (ts - position['ts']) >= cooldown:
                    success, sell_price, usdc_out, fee = execute_swap(SOL_MINT, USDC_MINT, position['sol_amt'])
                    if success:
                        profit = usdc_out - position['usdc_in']
                        total_fees += fee + position['fee']
                        total_pnl += profit - total_fees
                        log_trade(ts, 'SELL', sell_price, position['sol_amt'], usdc_out, profit, total_fees, price, 0, drawdown)
                        last_sell = ts
                        position = None
                        reward = profit - total_fees

            else:
                if action == 1 and usdc >= TRADE_AMOUNT_MIN and sol <= MAX_SOL_POSITION and (ts - last_buy) >= cooldown and (ts - last_sell) >= POST_SELL_BUY_DELAY and rsi < RSI_OVERBOUGHT and adx_val > 0.3 and trend_1h == 1 and regime == "bull":
                    risk_factor = 1 - (atr / price)
                    buy_amt = TRADE_AMOUNT_MIN + (TRADE_AMOUNT_MAX - TRADE_AMOUNT_MIN) * score * risk_factor
                    buy_amt = min(buy_amt, usdc * 0.5)
                    success, buy_price, sol_bought, fee = execute_swap(USDC_MINT, SOL_MINT, buy_amt)
                    if success:
                        if sol + sol_bought > MAX_SOL_POSITION:
                            sol_bought = MAX_SOL_POSITION - sol
                        position = {'entry_price': buy_price, 'sol_amt': sol_bought, 'usdc_in': buy_amt, 'fee': fee, 'ts': ts}
                        last_buy = ts
                        log_trade(ts, 'BUY', buy_price, sol_bought, buy_amt, 0, fee, price, 0, drawdown)

            if total_pnl < MAX_LOSS:
                print("Max loss hit. Stopping.")
                break

            print(f"SOL:{sol:.4f}, USDC:{usdc:.2f}, Price:{price:.2f}, PnL:{total_pnl:.2f}, Action:{action}, Epsilon:{agent.bull_agent.epsilon:.4f}")

            next_state = state
            agent.remember(state, action, reward, next_state, False, regime)
            agent.replay(regime)

            if time.time() - last_save > 300:
                agent.save_model()
                last_save = time.time()

            time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            agent.save_model()
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.backtest:
        backtest(days=90)
    elif args.live:
        run_live()
    else:
        print("Use --backtest or --live")
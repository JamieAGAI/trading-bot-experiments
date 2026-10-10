import time
import requests
import json
import numpy as np
import pandas as pd
from collections import deque
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client as SolanaClient
from solana.rpc.commitment import Confirmed
from solders.transaction import Transaction
from spl.token.client import Token as SplToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import create_associated_token_account, get_associated_token_address
import logging
from datetime import datetime

# Constants
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
RAYDIUM_API_BASE = "https://api-v3.raydium.io"
BIRDEYE_API_KEY = "your_birdeye_api_key_here"
BIRDEYE_API_URL = "https://public-api.birdeye.so"
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
PREFERRED_POOL_ID = "3ucNos4NbumPLZNWztqGHNFFgkHeRMBQAVemeeomsUxv"
WALLET_PRIVATE_KEY = "your_wallet_private_key_here"
RAYDIUM_CLMM_PROGRAM_ID = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
TRADE_AMOUNT_MIN = 50  # Minimum USDC to trade
TRADE_AMOUNT_MAX = 150  # Maximum USDC to trade
MAX_SOL_POSITION = 5  # Maximum SOL position size
STOP_LOSS = 0.05  # 5% stop loss
TAKE_PROFIT = 0.1  # 10% take profit
BUY_COOLDOWN = 300  # 5 minutes
SELL_COOLDOWN = 300  # 5 minutes
MAX_LOSS = -500  # Stop bot if total loss exceeds 500 USDC
REFRESH_INTERVAL = 60  # Check price every 60 seconds
EMA_SHORT = 12
EMA_LONG = 26
RSI_PERIOD = 14
STATE_SIZE = 12
ACTION_SIZE = 3  # Buy, Hold, Sell

# Rate limiting
RAYDIUM_RATE_LIMIT = 1  # 1 request per second
SOLANA_RATE_LIMIT = 10  # 10 requests per second
last_raydium_request = 0
last_solana_request = 0

# Caching for pool info
last_pool_info_fetch = 0
cached_pool_info = None
POOL_INFO_CACHE_DURATION = 300  # 5 minutes

# Track validated pool
validated_pool_id = None

# Solana client setup
solana_client = SolanaClient(SOLANA_RPC_URL)
wallet = Keypair.from_base58_string(WALLET_PRIVATE_KEY)
wallet_pubkey = wallet.pubkey()

# Logging setup
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rate limiting functions
def enforce_raydium_rate_limit():
    global last_raydium_request
    current_time = time.time()
    time_since_last = current_time - last_raydium_request
    if time_since_last < RAYDIUM_RATE_LIMIT:
        time.sleep(RAYDIUM_RATE_LIMIT - time_since_last)
    last_raydium_request = time.time()

def enforce_solana_rate_limit():
    global last_solana_request
    current_time = time.time()
    time_since_last = current_time - last_solana_request
    if time_since_last < (1 / SOLANA_RATE_LIMIT):
        time.sleep((1 / SOLANA_RATE_LIMIT) - time_since_last)
    last_solana_request = time.time()

# Fetch latest SOL/USDC price from Birdeye
def fetch_latest_price():
    headers = {"X-API-KEY": BIRDEYE_API_KEY}
    url = f"{BIRDEYE_API_URL}/defi/price?address={SOL_MINT}"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get("data") and "value" in data["data"]:
            return float(data["data"]["value"])
        print(f"[Live] Failed to fetch price from Birdeye: {data}")
        return None
    except Exception as e:
        print(f"[Live] Error fetching price from Birdeye: {e}")
        return None

# Fetch historical prices for technical indicators
def fetch_historical_prices(days=1):
    headers = {"X-API-KEY": BIRDEYE_API_KEY}
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    url = f"{BIRDEYE_API_URL}/defi/history_price?address={SOL_MINT}&type=1h&time_from={start_time}&time_to={end_time}"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if "data" not in data or "items" not in data["data"]:
            print(f"[Live] Failed to fetch historical prices: {data}")
            return pd.DataFrame()
        items = data["data"]["items"]
        df = pd.DataFrame(items)
        df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s")
        df["price"] = df["value"].astype(float)
        return df[["timestamp", "price"]]
    except Exception as e:
        print(f"[Live] Error fetching historical prices: {e}")
        return pd.DataFrame()

# Technical indicators
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_period=12, long_period=26):
    short_ema = calculate_ema(series, short_period)
    long_ema = calculate_ema(series, long_period)
    macd = short_ema - long_ema
    return macd

def calculate_price_change(series):
    return series.pct_change()

def calculate_bollinger_band_width(series, period=20):
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return (upper_band - lower_band) / rolling_mean

def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

# DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch]
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save("dqn_model.h5")
        print("[Live] Model saved")

# Wallet and token account management
def get_associated_token_address(mint):
    return get_associated_token_address(wallet_pubkey, Pubkey.from_string(mint))

def initialize_ata(mint):
    ata = get_associated_token_address(mint)
    try:
        enforce_solana_rate_limit()
        account_info = solana_client.get_account_info(ata)
        if account_info.value is None:
            print(f"[Live] Creating ATA for mint {mint}...")
            transaction = Transaction()
            instruction = create_associated_token_account(wallet_pubkey, wallet_pubkey, Pubkey.from_string(mint))
            transaction.add(instruction)
            enforce_solana_rate_limit()
            response = solana_client.send_transaction(transaction, wallet, opts=Confirmed)
            print(f"[Live] ATA created: {response.value}")
        return ata
    except Exception as e:
        print(f"[Live] Failed to initialize ATA for mint {mint}: {e}")
        return None

def get_wallet_balances(force_update=False):
    sol_balance = 0
    usdc_balance = 0
    try:
        enforce_solana_rate_limit()
        sol_balance = solana_client.get_balance(wallet_pubkey).value / 1e9
        usdc_ata = get_associated_token_address(USDC_MINT)
        enforce_solana_rate_limit()
        usdc_account = solana_client.get_token_account_balance(usdc_ata)
        usdc_balance = usdc_account.value.ui_amount if usdc_account.value else 0
    except Exception as e:
        print(f"[Live] Error fetching wallet balances: {e}")
    return sol_balance, usdc_balance

# Pool validation
def fetch_pool_info(pool_id, use_v2_fallback=False):
    global last_pool_info_fetch, cached_pool_info
    current_time = time.time()

    # Use cached pool info if available and not expired
    if cached_pool_info and (current_time - last_pool_info_fetch) < POOL_INFO_CACHE_DURATION:
        print(f"[Debug] Using cached pool info for pool {pool_id}")
        return cached_pool_info if cached_pool_info["id"] == pool_id else None

    # Fetch pool info from /pools/info/list
    url = f"{RAYDIUM_API_BASE}/pools/info/list?poolType=concentrated&poolSortField=liquidity&sortType=desc&pageSize=100&page=1"
    try:
        enforce_raydium_rate_limit()
        response = requests.get(url).json()
        if "data" not in response or not response["data"]:
            print(f"[Live] Failed to fetch pool info from V3 API (/pools/info/list): {response}")
            return None
        pools = response["data"]
        for pool in pools:
            if pool["id"] == pool_id:
                print(f"[Debug] Found pool {pool_id} in V3 API response: liquidity={pool.get('liquidity')}, price={pool.get('price')}, volume_24h={pool.get('volume_24h')}")
                cached_pool_info = pool
                last_pool_info_fetch = current_time
                return pool
        print(f"[Live] Pool {pool_id} not found in V3 Pool API response (/pools/info/list)")
        return None
    except Exception as e:
        print(f"[Live] Failed to fetch pool info from V3 API (/pools/info/list): {e}")
        return None

def fetch_alternative_pools():
    url = f"{RAYDIUM_API_BASE}/pools/info/mint?mintA={SOL_MINT}&mintB={USDC_MINT}"
    try:
        enforce_raydium_rate_limit()
        response = requests.get(url).json()
        if "data" in response:
            return response["data"]
        return []
    except Exception as e:
        print(f"[Live] Failed to fetch alternative pools: {e}")
        return []

def validate_pool(pool_id, amount_usdc, current_price):
    global validated_pool_id
    # Skip validation if the pool has already been validated
    if validated_pool_id == pool_id:
        print(f"[Debug] Pool {pool_id} already validated, skipping validation")
        return True

    pool = fetch_pool_info(pool_id)
    if not pool:
        return False
    # Check if the pool matches the preferred pool ID
    if pool_id != PREFERRED_POOL_ID:
        print(f"[Live] Pool ID {pool_id} does not match preferred pool {PREFERRED_POOL_ID}. Fetching alternative pools...")
        alternative_pools = fetch_alternative_pools()
        for alt_pool in alternative_pools:
            if alt_pool.get("amm_id") == PREFERRED_POOL_ID:
                print(f"[Live] Found preferred pool in alternative pools: {PREFERRED_POOL_ID}")
                return validate_pool(PREFERRED_POOL_ID, amount_usdc, current_price)
        print(f"[Live] Preferred pool {PREFERRED_POOL_ID} not found in alternative pools. Proceeding with {pool_id}...")
    # Check liquidity (amount of SOL and USDC in the pool)
    if "baseReserve" in pool and "quoteReserve" in pool:
        sol_liquidity = float(pool["baseReserve"]) / 1e9  # SOL has 9 decimals
        usdc_liquidity = float(pool["quoteReserve"]) / 1e6  # USDC has 6 decimals
    else:
        # Estimate reserves using liquidity and price
        liquidity_usd = float(pool.get("liquidity", 0))
        pool_price = float(pool.get("price", current_price))
        if liquidity_usd == 0 or pool_price == 0:
            print(f"[Live] Cannot estimate reserves for pool {pool_id}: liquidity={liquidity_usd}, price={pool_price}")
            return False
        print(f"[Warning] Estimating reserves for pool {pool_id} using liquidity={liquidity_usd} and price={pool_price} (assuming 50/50 USD split)")
        usdc_liquidity = (liquidity_usd / 2)  # Assume 50/50 split in USD value
        sol_liquidity = (liquidity_usd / 2) / pool_price
    print(f"[Debug] Pool {pool_id} - SOL liquidity: {sol_liquidity:.4f}, USDC liquidity: {usdc_liquidity:.2f}, API price: {pool.get('price', 0)}")
    # Ensure the pool has enough USDC to handle the swap
    if usdc_liquidity < amount_usdc * 1.5:  # Require 1.5x the swap amount as a buffer
        print(f"[Live] Insufficient USDC liquidity in pool {pool_id}: {usdc_liquidity:.2f} < {amount_usdc * 1.5:.2f}")
        return False
    # Estimate SOL needed based on current price
    expected_sol = amount_usdc / current_price
    if sol_liquidity < expected_sol * 1.5:  # Require 1.5x the expected SOL output
        print(f"[Live] Insufficient SOL liquidity in pool {pool_id}: {sol_liquidity:.4f} < {expected_sol * 1.5:.4f}")
        return False
    # Check if the pool is active (has recent activity)
    volume_24h = float(pool.get("volume_24h", 0))
    volume_fee_24h = float(pool.get("fee_24h", 0))  # Note: Field name is "fee_24h" in V3 API
    if volume_24h == 0 and volume_fee_24h == 0:
        print(f"[Live] Pool {pool_id} has no recent activity (volume_24h: {volume_24h}, fee_24h: {volume_fee_24h})")
        return False
    # Additional check: Ensure the pool price is reasonable (within 10% of Birdeye price)
    pool_price = float(pool.get("price", 0))
    if pool_price > 0:  # Only check if the pool provides a price
        price_diff = abs(pool_price - current_price) / current_price
        if price_diff > 0.1:  # Allow 10% deviation
            print(f"[Live] Pool price {pool_price:.2f} deviates too much from market price {current_price:.2f} (diff: {price_diff:.2%})")
            return False
    # Mark the pool as validated
    validated_pool_id = pool_id
    return True

# Swap execution
def perform_swap(input_mint, output_mint, amount, current_price):
    try:
        pool_info = fetch_pool_info(PREFERRED_POOL_ID)
        if not pool_info:
            print(f"[Live] Failed to fetch pool info for swap")
            return False, 0

        # Get user token accounts
        input_ata = get_associated_token_address(input_mint)
        output_ata = get_associated_token_address(output_mint)

        # Get pool vault accounts from pool info
        vault_a = pool_info.get("vaultA")
        vault_b = pool_info.get("vaultB")
        if not vault_a or not vault_b:
            print(f"[Live] Pool info missing vault accounts: vaultA={vault_a}, vaultB={vault_b}")
            return False, 0

        # Convert amount to lamports
        amount_lamports = int(amount * (1e9 if input_mint == SOL_MINT else 1e6))
        slippage = 0.005  # 0.5% slippage
        min_amount_out = int((amount / current_price) * (1 - slippage) * (1e9 if output_mint == SOL_MINT else 1e6))

        # Construct the swap instruction (simplified)
        # In a real implementation, this would involve building the Raydium CLMM swap instruction
        # using the program ID, pool state, vaults, and token accounts.
        # For now, we'll simulate the swap and return the current price.
        print(f"[Live] Simulating swap: {amount} {input_mint} -> {output_mint}, min out: {min_amount_out}")

        # Placeholder: In a real implementation, you would:
        # 1. Build the swap instruction using the Raydium CLMM program
        # 2. Create a Transaction with the instruction
        # 3. Sign and send the transaction using solana_client
        # For simulation purposes, we'll assume the swap succeeds
        effective_price = current_price  # Simplified; in reality, fetch from transaction logs
        return True, effective_price
    except Exception as e:
        print(f"[Live] Swap failed: {e}")
        return False, 0

# Success score for dynamic trade sizing
def calculate_success_score(state, norm_ema_short, norm_ema_long, norm_rsi, norm_macd, norm_price_change, norm_bb_width, recent_returns, i):
    ema_crossover = 1 if norm_ema_short.iloc[i] > norm_ema_long.iloc[i] else 0
    rsi_oversold = 1 if norm_rsi.iloc[i] < 0.3 else 0
    macd_positive = 1 if norm_macd.iloc[i] > 0 else 0
    price_momentum = 1 if norm_price_change.iloc[i] > 0 else 0
    volatility = 1 if norm_bb_width.iloc[i] < 0.5 else 0
    recent_trend = sum(recent_returns) / len(recent_returns)
    recent_trend_score = 1 if recent_trend > 0 else 0
    score = (ema_crossover + rsi_oversold + macd_positive + price_momentum + volatility + recent_trend_score) / 6
    return score

# Logging trades
def log_trade(timestamp, trade_type, price, sol_amount, usdc_amount, profit, unrealized_pnl, drawdown):
    log_entry = {
        "timestamp": timestamp,
        "type": trade_type,
        "price": price,
        "sol_amount": sol_amount,
        "usdc_amount": usdc_amount,
        "profit": profit,
        "unrealized_pnl": unrealized_pnl,
        "drawdown": drawdown
    }
    logging.info(json.dumps(log_entry))

# Main trading loop
def run_live():
    print("[Live] Starting live trading with Raydium API...")
    agent = DoubleDQNAgent(STATE_SIZE, ACTION_SIZE)
    sol_balance, usdc_balance = get_wallet_balances(force_update=True)
    position = None
    last_buy_time = 0
    total_profit, max_usdc = 0, usdc_balance
    recent_returns = deque([0] * 5, maxlen=5)

    print("[Live] Initializing token accounts...")
    sol_ata = initialize_ata(SOL_MINT)
    usdc_ata = initialize_ata(USDC_MINT)
    if sol_ata is None or usdc_ata is None:
        print("[Live] Failed to initialize token accounts. Exiting...")
        exit(1)

    # Validate the pool once at the start
    current_price = fetch_latest_price()
    if current_price is None:
        print("[Live] Failed to fetch initial price. Exiting...")
        exit(1)
    if not validate_pool(PREFERRED_POOL_ID, TRADE_AMOUNT_MIN, current_price):
        print(f"[Live] Preferred pool {PREFERRED_POOL_ID} is not suitable for trading. Exiting...")
        exit(1)

    while True:
        price = fetch_latest_price()
        if price is None:
            print("[Live] Failed to fetch price. Retrying in 60 seconds...")
            time.sleep(REFRESH_INTERVAL)
            continue

        price_data = fetch_historical_prices(days=1)
        if len(price_data) < EMA_LONG:
            print("[Live] Insufficient historical data. Retrying in 60 seconds...")
            time.sleep(REFRESH_INTERVAL)
            continue

        prices = price_data["price"]
        timestamps = price_data["timestamp"]
        ema_short = calculate_ema(prices, EMA_SHORT)
        ema_long = calculate_ema(prices, EMA_LONG)
        rsi = calculate_rsi(prices, RSI_PERIOD)
        macd = calculate_macd(prices, short_period=12, long_period=26)
        price_change = calculate_price_change(prices)
        bb_width = calculate_bollinger_band_width(prices)

        norm_prices = normalize_series(prices)
        norm_ema_short = normalize_series(ema_short)
        norm_ema_long = normalize_series(ema_long)
        norm_rsi = normalize_series(rsi)
        norm_macd = normalize_series(macd)
        norm_price_change = normalize_series(price_change)
        norm_bb_width = normalize_series(bb_width)

        i = len(prices) - 1
        recent_returns.append((prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]) if i >= 1 else recent_returns.append(0)

        state = [
            float(norm_prices.iloc[i]) if not np.isnan(norm_prices.iloc[i]) else 0.0,
            float(norm_ema_short.iloc[i]) if not np.isnan(norm_ema_short.iloc[i]) else 0.0,
            float(norm_ema_long.iloc[i]) if not np.isnan(norm_ema_long.iloc[i]) else 0.0,
            float(norm_rsi.iloc[i]) if not np.isnan(norm_rsi.iloc[i]) else 0.0,
            float(norm_macd.iloc[i]) if not np.isnan(norm_macd.iloc[i]) else 0.0,
            float(norm_price_change.iloc[i]) if not np.isnan(norm_price_change.iloc[i]) else 0.0,
            float(norm_bb_width.iloc[i]) if not np.isnan(norm_bb_width.iloc[i]) else 0.0,
            float(recent_returns[-1]),
            1.0 if position else 0.0,
            float(position["sol_amount"]) if position else 0.0,
            0.0,
            0.0
        ]

        reward = 0
        if position:
            unrealized_profit = (price - position["entry_price"]) * position["sol_amount"]
            reward = 0.1 * unrealized_profit

        if position:
            current_return = (price - position["entry_price"]) / position["entry_price"]
            if current_return <= -STOP_LOSS or current_return >= TAKE_PROFIT:
                action = 2
            else:
                action = agent.act(state)
            print(f"[Debug] Action: {action}, Current Return: {current_return:.4f}, Reward: {reward:.4f}, Epsilon: {agent.epsilon:.4f}")
        else:
            action = agent.act(state)
            print(f"[Debug] Action: {action}, No Position, Reward: {reward:.4f}, Epsilon: {agent.epsilon:.4f}")

        done = False
        timestamp = int(time.time())
        unrealized_pnl = (price - position["entry_price"]) * position["sol_amount"] if position else 0
        drawdown = max_usdc - (usdc_balance + (sol_balance * price if sol_balance else 0))

        if not position and action == 1 and usdc_balance >= TRADE_AMOUNT_MIN and sol_balance + TRADE_AMOUNT_MIN / price <= MAX_SOL_POSITION and (timestamp - last_buy_time) >= BUY_COOLDOWN:
            success_score = calculate_success_score(state, norm_ema_short, norm_ema_long, norm_rsi, norm_macd, norm_price_change, norm_bb_width, recent_returns, i)
            buy_amount = TRADE_AMOUNT_MIN + (TRADE_AMOUNT_MAX - TRADE_AMOUNT_MIN) * success_score
            buy_amount = min(buy_amount, usdc_balance, (MAX_SOL_POSITION - sol_balance) * price)
            success, effective_price = perform_swap(USDC_MINT, SOL_MINT, buy_amount, price)
            if success:
                sol_amount = buy_amount / effective_price
                position = {"entry_price": effective_price, "sol_amount": sol_amount, "timestamp": timestamp}
                usdc_balance -= buy_amount
                sol_balance += sol_amount
                last_buy_time = timestamp
                log_trade(timestamp, "BUY", effective_price, sol_amount, buy_amount, 0, unrealized_pnl, drawdown)
                print(f"[Live] BUY at {effective_price:.2f}")
                sol_balance, usdc_balance = get_wallet_balances(force_update=True)

        elif position and action == 2 and (timestamp - position["timestamp"]) >= SELL_COOLDOWN:
            sol_to_sell = position["sol_amount"]
            if sol_to_sell <= 0:
                position = None
                continue
            success, effective_price = perform_swap(SOL_MINT, USDC_MINT, sol_to_sell, price)
            if success:
                usdc_amount = effective_price * position["sol_amount"]
                profit = (effective_price - position["entry_price"]) * position["sol_amount"]
                reward = profit
                total_profit += profit
                usdc_balance += usdc_amount
                sol_balance -= position["sol_amount"]
                log_trade(timestamp, "SELL", effective_price, position["sol_amount"], usdc_amount, profit, unrealized_pnl, drawdown)
                print(f"[Live] SELL at {effective_price:.2f}, Profit: {profit:.2f}")
                position = None
                done = True
                sol_balance, usdc_balance = get_wallet_balances(force_update=True)

        sol_balance, usdc_balance = get_wallet_balances()
        print(f"[Live] SOL: {sol_balance:.4f}, USDC: {usdc_balance:.2f}, SOL Price: {price:.2f}, Total Profit: {total_profit:.2f}, Unrealized P&L: {unrealized_pnl:.2f}, Drawdown: {drawdown:.2f}")

        if total_profit < MAX_LOSS:
            print(f"[Live] Maximum loss reached ({total_profit:.2f} < {MAX_LOSS}). Stopping bot...")
            agent.save_model()
            exit(1)

        if total_profit > 0 or drawdown < -100:
            agent.save_model()

        next_state = state
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    run_live()
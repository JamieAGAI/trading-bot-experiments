import os
import time
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
import random
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction as SoldersTransaction
from solders.instruction import Instruction, AccountMeta
from solders import message
from spl.token import instructions as spl_instructions
from spl.token.constants import TOKEN_PROGRAM_ID
from dotenv import load_dotenv
import base58
import struct

# Load environment variables
load_dotenv()

# Constants (using base58.b58decode to avoid Pubkey.from_string issues)
try:
    print("Decoding SOL_MINT...")
    SOL_MINT = Pubkey(base58.b58decode("So11111111111111111111111111111111111111112"))
    print("Decoding USDC_MINT...")
    USDC_MINT = Pubkey(base58.b58decode("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"))
    print("Decoding ORCA_WHIRLPOOL_PROGRAM_ID...")
    ORCA_WHIRLPOOL_PROGRAM_ID = Pubkey(base58.b58decode("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"))
    print("Decoding ORCA_WHIRLPOOL_CONFIG...")
    ORCA_WHIRLPOOL_CONFIG = Pubkey(base58.b58decode("2LecshUwdy9xi7meFgHtFJQNSKk4KdTrcpvaB56dP2NQ"))
    print("Decoding ORCA_POOL_ID...")
    ORCA_POOL_ID = Pubkey(base58.b58decode("Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crryfu44zE"))
    print("Decoding TOKEN_VAULT_A...")
    TOKEN_VAULT_A = Pubkey(base58.b58decode("EUuUbDcafPrmVTD5M6qoJAoyyNbihBhugADAxRMn5he9"))
    print("Decoding TOKEN_VAULT_B...")
    TOKEN_VAULT_B = Pubkey(base58.b58decode("2WLWEuKDgkDUccTpbwYp1GToYktiSB1cXvreHUwiSUVP"))
except Exception as e:
    print(f"[Error] Failed to construct public keys: {e}")
    exit(1)

EMA_SHORT, EMA_LONG, RSI_PERIOD = 9, 21, 7
TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX = 50, 150
MAX_SOL_POSITION, INITIAL_USDC = 5, 3000
LOG_FILE = "live_trades.csv"
MODEL_DIR = "C:/Users/jamie/SOLPY"
MODEL_FILE = f"{MODEL_DIR}/model.keras"
SLIPPAGE = 0.01
STOP_LOSS, TAKE_PROFIT = 0.05, 0.10
BUY_COOLDOWN, SELL_COOLDOWN = 600, 600
REFRESH_INTERVAL = 20
MAX_LOSS = -100

# DQL Parameters
STATE_SIZE, ACTION_SIZE = 12, 3
GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY = 0.95, 0.5, 0.01, 0.98
LEARNING_RATE, MEMORY_SIZE, BATCH_SIZE = 0.0001, 5000, 64
TARGET_UPDATE_FREQ = 1000

# Solana client
solana_client = Client("https://api.mainnet-beta.solana.com")

# Wallet setup
try:
    private_key_base58 = os.getenv("PHANTOM_PRIVATE_KEY")
    if not private_key_base58:
        raise ValueError("PHANTOM_PRIVATE_KEY not found in .env file")
    wallet = Keypair.from_base58_string(private_key_base58)
    print(f"[Live] Wallet Public Key: {wallet.pubkey()}")
except Exception as e:
    print(f"[Live] Failed to decode PHANTOM_PRIVATE_KEY: {e}")
    exit(1)

# Rate limiter for Solana API calls
solana_request_timestamps = []
SOLANA_RATE_LIMIT = 10
SOLANA_RATE_LIMIT_WINDOW = 1

def enforce_solana_rate_limit():
    current_time = time.time()
    while solana_request_timestamps and current_time - solana_request_timestamps[0] > SOLANA_RATE_LIMIT_WINDOW:
        solana_request_timestamps.pop(0)
    if len(solana_request_timestamps) >= SOLANA_RATE_LIMIT:
        sleep_time = SOLANA_RATE_LIMIT_WINDOW - (current_time - solana_request_timestamps[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
    solana_request_timestamps.append(time.time())

# Balance cache
last_balance_check = 0
cached_sol_balance = 0
cached_usdc_balance = 0
BALANCE_CACHE_DURATION = 300

def get_wallet_balances(force_update=False):
    global last_balance_check, cached_sol_balance, cached_usdc_balance
    current_time = time.time()
    if not force_update and (current_time - last_balance_check) < BALANCE_CACHE_DURATION:
        return cached_sol_balance, cached_usdc_balance
    try:
        enforce_solana_rate_limit()
        sol_balance = solana_client.get_balance(wallet.pubkey()).value / 1_000_000_000

        usdc_ata, _ = Pubkey.find_program_address(
            [bytes(wallet.pubkey()), bytes(TOKEN_PROGRAM_ID), bytes(USDC_MINT)],
            Pubkey(base58.b58decode("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"))
        )
        enforce_solana_rate_limit()
        usdc_account = solana_client.get_token_account_balance(usdc_ata)
        usdc_balance = usdc_account.value.ui_amount if usdc_account.value.ui_amount is not None else 0

        cached_sol_balance = sol_balance
        cached_usdc_balance = usdc_balance
        last_balance_check = current_time
        return sol_balance, usdc_balance
    except Exception as e:
        print(f"[Live] Error fetching wallet balances: {e}")
        return 0, 0

# Fetch latest price
def fetch_latest_price():
    url = f"https://public-api.birdeye.so/defi/price?address={str(SOL_MINT)}"
    headers = {"X-API-KEY": os.getenv("BIRDEYE_API_KEY")}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return data["data"]["value"] if data["success"] else None
    except Exception as e:
        print(f"[Live] Error fetching price: {e}")
        return None

# Fetch historical prices
def fetch_historical_prices(days=1):
    time_to = int(time.time())
    time_from = time_to - days * 24 * 3600
    url = f"https://public-api.birdeye.so/defi/history_price?address={str(SOL_MINT)}&address_type=token&type=5m&time_from={time_from}&time_to={time_to}"
    headers = {"X-API-KEY": os.getenv("BIRDEYE_API_KEY")}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data["success"]:
            return [{"timestamp": item["unixTime"], "price": item["value"]} for item in data["data"]["items"]]
        return []
    except Exception as e:
        print(f"[Live] Error fetching historical prices: {e}")
        return []

# Technical indicators
def calculate_ema(prices, period):
    k = 2 / (period + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append(prices[i] * k + ema[-1] * (1 - k))
    return ema

def calculate_rsi(prices, period):
    deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    gain, loss = 0, 0
    for i in range(period):
        delta = deltas[i]
        if delta > 0:
            gain += delta
        else:
            loss -= delta
    avg_gain = gain / period
    avg_loss = loss / period
    rsi = []
    for i in range(period, len(deltas)):
        delta = deltas[i]
        avg_gain = (avg_gain * (period - 1) + (delta if delta > 0 else 0)) / period
        avg_loss = (avg_loss * (period - 1) + (-delta if delta < 0 else 0)) / period
        rsi.append(100 - (100 / (1 + avg_gain / (avg_loss or 1))))
    return rsi

# Fetch Whirlpool tickCurrentIndex
def fetch_whirlpool_tick_index(pool_id):
    try:
        enforce_solana_rate_limit()
        account_info = solana_client.get_account_info(pool_id).value
        if not account_info:
            print(f"[Live] Failed to fetch Whirlpool account for {pool_id}")
            return -21418  # Fallback to the last known value

        # Parse the tickCurrentIndex (i32 at offset 132)
        data = account_info.data
        tick_current_index = struct.unpack("<i", data[132:136])[0]
        return tick_current_index
    except Exception as e:
        print(f"[Live] Failed to fetch tickCurrentIndex: {e}")
        return -21418  # Fallback to the last known value

# Orca Whirlpool swap function
import hashlib

import hashlib

import hashlib
import struct
from solders.transaction import Transaction as SoldersTransaction
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta

def execute_orca_swap(input_mint, output_mint, amount):
    print("[Debug] Entering execute_orca_swap with updated transaction construction")
    try:
        # Determine token order (SOL -> USDC or USDC -> SOL)
        is_sol_to_usdc = input_mint == SOL_MINT
        token_mint_a = SOL_MINT if is_sol_to_usdc else USDC_MINT
        token_mint_b = USDC_MINT if is_sol_to_usdc else SOL_MINT
        token_vault_a = TOKEN_VAULT_A if is_sol_to_usdc else TOKEN_VAULT_B
        token_vault_b = TOKEN_VAULT_B if is_sol_to_usdc else TOKEN_VAULT_A

        # Initialize token accounts
        input_ata, _ = Pubkey.find_program_address(
            [bytes(wallet.pubkey()), bytes(TOKEN_PROGRAM_ID), bytes(token_mint_a)],
            Pubkey(base58.b58decode("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"))
        )
        output_ata, _ = Pubkey.find_program_address(
            [bytes(wallet.pubkey()), bytes(TOKEN_PROGRAM_ID), bytes(token_mint_b)],
            Pubkey(base58.b58decode("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"))
        )

        # Create token accounts if they don't exist
        for ata, mint in [(input_ata, token_mint_a), (output_ata, token_mint_b)]:
            enforce_solana_rate_limit()
            account_info = solana_client.get_account_info(ata).value
            if not account_info:
                instruction = spl_instructions.create_associated_token_account(
                    payer=wallet.pubkey(),
                    owner=wallet.pubkey(),
                    mint=mint
                )
                blockhash_resp = solana_client.get_latest_blockhash()
                if not blockhash_resp or not hasattr(blockhash_resp, "value") or not blockhash_resp.value.blockhash:
                    print("[Live] Failed to fetch recent blockhash for token account creation")
                    return False, None
                recent_blockhash = blockhash_resp.value.blockhash
                tx = SoldersTransaction.new_signed_with_payer(
                    instructions=[instruction],
                    payer=wallet.pubkey(),
                    signing_keypairs=[wallet],
                    recent_blockhash=recent_blockhash
                )
                solana_client.send_raw_transaction(bytes(tx))

        # Check balances
        sol_balance, usdc_balance = get_wallet_balances()
        if sol_balance < 0.01:
            print(f"[Live] Insufficient SOL for fees: {sol_balance:.4f}")
            return False, None
        if input_mint == USDC_MINT and usdc_balance < amount:
            print(f"[Live] Insufficient USDC: {usdc_balance:.2f} < {amount:.2f}")
            return False, None

        # Calculate swap amounts
        amount_in = int(amount * (10 ** (9 if input_mint == SOL_MINT else 6)))
        current_price = fetch_latest_price()
        if not current_price:
            return False, None
        expected_out = (amount / current_price) if is_sol_to_usdc else (amount * current_price)
        min_amount_out = int(expected_out * (1 - SLIPPAGE) * (10 ** (6 if output_mint == USDC_MINT else 9)))

        # Fetch the current tick index
        tick_spacing = 4
        current_tick_index = fetch_whirlpool_tick_index(ORCA_POOL_ID)
        ticks_per_array = 88 * tick_spacing  # 352 ticks per array
        start_tick_index_0 = (current_tick_index // ticks_per_array) * ticks_per_array
        start_tick_index_1 = start_tick_index_0 + ticks_per_array
        start_tick_index_2 = start_tick_index_1 + ticks_per_array

        tick_array_0, _ = Pubkey.find_program_address(
            [b"tick_array", bytes(ORCA_POOL_ID), struct.pack("<i", start_tick_index_0)],
            ORCA_WHIRLPOOL_PROGRAM_ID
        )
        tick_array_1, _ = Pubkey.find_program_address(
            [b"tick_array", bytes(ORCA_POOL_ID), struct.pack("<i", start_tick_index_1)],
            ORCA_WHIRLPOOL_PROGRAM_ID
        )
        tick_array_2, _ = Pubkey.find_program_address(
            [b"tick_array", bytes(ORCA_POOL_ID), struct.pack("<i", start_tick_index_2)],
            ORCA_WHIRLPOOL_PROGRAM_ID
        )

        # Compute oracle address
        oracle, _ = Pubkey.find_program_address(
            [b"oracle", bytes(ORCA_POOL_ID)],
            ORCA_WHIRLPOOL_PROGRAM_ID
        )

        # Build swap instruction with correct discriminator and padded booleans
        swap_discriminator = hashlib.sha256("global:swap".encode("utf-8")).digest()[:8]
        swap_data = (
            swap_discriminator +  # 8-byte discriminator for swap
            struct.pack("<Q", amount_in) +
            struct.pack("<Q", min_amount_out) +
            struct.pack("<Q", 0) +  # sqrt_price_limit (0 to disable)
            bytes([1]) + bytes([0] * 7) +  # amount_specified_is_input: true, padded to 8 bytes
            bytes([1 if is_sol_to_usdc else 0]) + bytes([0] * 7)  # a_to_b, padded to 8 bytes
        )

        swap_instruction = Instruction(
            program_id=ORCA_WHIRLPOOL_PROGRAM_ID,
            accounts=[
                AccountMeta(pubkey=ORCA_WHIRLPOOL_CONFIG, is_signer=False, is_writable=False),
                AccountMeta(pubkey=ORCA_POOL_ID, is_signer=False, is_writable=True),
                AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
                AccountMeta(pubkey=input_ata, is_signer=False, is_writable=True),
                AccountMeta(pubkey=token_vault_a, is_signer=False, is_writable=True),
                AccountMeta(pubkey=output_ata, is_signer=False, is_writable=True),
                AccountMeta(pubkey=token_vault_b, is_signer=False, is_writable=True),
                AccountMeta(pubkey=tick_array_0, is_signer=False, is_writable=True),
                AccountMeta(pubkey=tick_array_1, is_signer=False, is_writable=True),
                AccountMeta(pubkey=tick_array_2, is_signer=False, is_writable=True),
                AccountMeta(pubkey=oracle, is_signer=False, is_writable=True),
                AccountMeta(pubkey=wallet.pubkey(), is_signer=True, is_writable=False),
            ],
            data=swap_data
        )

        # Send transaction
        blockhash_resp = solana_client.get_latest_blockhash()
        if not blockhash_resp or not hasattr(blockhash_resp, "value") or not blockhash_resp.value.blockhash:
            print("[Live] Failed to fetch recent blockhash for swap")
            return False, None
        recent_blockhash = blockhash_resp.value.blockhash
        print(f"[Debug] Fetched recent_blockhash: {recent_blockhash}")

        # Create and sign the transaction in one step
        tx = SoldersTransaction.new_signed_with_payer(
            instructions=[swap_instruction],
            payer=wallet.pubkey(),
            signing_keypairs=[wallet],
            recent_blockhash=recent_blockhash
        )
        print(f"[Debug] Constructed and signed transaction: {tx}")
        solana_client.send_raw_transaction(bytes(tx))
        print(f"[Debug] Sent transaction")
        effective_price = current_price
        return True, effective_price
    except Exception as e:
        print(f"[Live] Orca Swap failed: {e}")
        return False, None

# Log trades
def log_trade(timestamp, trade_type, price, sol_amount, usdc_amount, profit, unrealized_pnl, drawdown):
    log_entry = f"{timestamp},{trade_type},{price},{sol_amount},{usdc_amount},{profit},{unrealized_pnl},{drawdown}\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC),unrealizedPnL(USDC),drawdown(USDC)\n")
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)
    print(f"[Live] Trade: {trade_type} at {price:.2f}, SOL: {sol_amount:.4f}, USDC: {usdc_amount:.2f}, Profit: {profit:.2f}")

# DQL Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_counter = 0
        self.load_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
        return model

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = tf.keras.models.load_model(MODEL_FILE)
                self.target_model = tf.keras.models.load_model(MODEL_FILE)
                print(f"[DQL] Loaded model from {MODEL_FILE}")
                self.target_model.set_weights(self.model.get_weights())
            except Exception as e:
                print(f"[DQL] Failed to load model: {e}")

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(MODEL_FILE)
        print(f"[DQL] Saved model to {MODEL_FILE}")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.array([state])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                next_action = np.argmax(self.model.predict(np.array([next_states[i]]), verbose=0)[0])
                target += self.gamma * next_q_values[i][next_action]
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("[DQL] Updated target model")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main trading loop
def run_live():
    print("[Live] Starting live trading with Orca...")
    agent = DoubleDQNAgent(STATE_SIZE, ACTION_SIZE)
    sol_balance, usdc_balance = get_wallet_balances(True)
    position = None
    last_buy_time = 0
    total_profit = 0
    max_usdc = usdc_balance

    while True:
        price = fetch_latest_price()
        if price is None:
            print("[Live] Failed to fetch price, retrying...")
            time.sleep(REFRESH_INTERVAL)
            continue

        price_data = fetch_historical_prices(1)
        if len(price_data) < EMA_LONG:
            print("[Live] Not enough historical data, retrying...")
            time.sleep(REFRESH_INTERVAL)
            continue

        prices = [d["price"] for d in price_data]
        ema_short = calculate_ema(prices, EMA_SHORT)
        ema_long = calculate_ema(prices, EMA_LONG)
        rsi = calculate_rsi(prices, RSI_PERIOD)

        state = [
            prices[-1] / max(prices),
            ema_short[-1] / max(ema_short),
            ema_long[-1] / max(ema_long),
            rsi[-1] / 100,
            0, 0, 0, 0,
            1 if position else 0,
            position["sol_amount"] if position else 0,
            0, 0
        ]

        reward = 0
        if position:
            unrealized_profit = (price - position["entry_price"]) * position["sol_amount"]
            reward = 0.1 * unrealized_profit

        action = agent.act(state)
        timestamp = int(time.time())
        unrealized_pnl = (price - position["entry_price"]) * position["sol_amount"] if position else 0
        drawdown = max_usdc - (usdc_balance + (sol_balance * price or 0))

        if not position and action == 1 and usdc_balance >= TRADE_AMOUNT_MIN and sol_balance + TRADE_AMOUNT_MIN / price <= MAX_SOL_POSITION and (timestamp - last_buy_time) >= BUY_COOLDOWN:
            buy_amount = min(TRADE_AMOUNT_MAX, max(TRADE_AMOUNT_MIN, usdc_balance))
            success, effective_price = execute_orca_swap(USDC_MINT, SOL_MINT, buy_amount)
            if success:
                sol_amount = buy_amount / effective_price
                position = {"entry_price": effective_price, "sol_amount": sol_amount, "timestamp": timestamp}
                usdc_balance -= buy_amount
                sol_balance += sol_amount
                last_buy_time = timestamp
                log_trade(timestamp, "BUY", effective_price, sol_amount, buy_amount, 0, unrealized_pnl, drawdown)
                print(f"[Live] BUY at {effective_price:.2f}")
                sol_balance, usdc_balance = get_wallet_balances(True)

        elif position and action == 2 and (timestamp - position["timestamp"]) >= SELL_COOLDOWN:
            sol_to_sell = position["sol_amount"]
            success, effective_price = execute_orca_swap(SOL_MINT, USDC_MINT, sol_to_sell)
            if success:
                usdc_amount = effective_price * sol_to_sell
                profit = (effective_price - position["entry_price"]) * sol_to_sell
                total_profit += profit
                usdc_balance += usdc_amount
                sol_balance -= sol_to_sell
                log_trade(timestamp, "SELL", effective_price, sol_to_sell, usdc_amount, profit, unrealized_pnl, drawdown)
                print(f"[Live] SELL at {effective_price:.2f}, Profit: {profit:.2f}")
                position = None
                sol_balance, usdc_balance = get_wallet_balances(True)

        if position:
            current_pnl = (price - position["entry_price"]) / position["entry_price"]
            if current_pnl <= -STOP_LOSS or current_pnl >= TAKE_PROFIT:
                sol_to_sell = position["sol_amount"]
                success, effective_price = execute_orca_swap(SOL_MINT, USDC_MINT, sol_to_sell)
                if success:
                    usdc_amount = effective_price * sol_to_sell
                    profit = (effective_price - position["entry_price"]) * sol_to_sell
                    total_profit += profit
                    usdc_balance += usdc_amount
                    sol_balance -= sol_to_sell
                    log_trade(timestamp, "SELL (SL/TP)", effective_price, sol_to_sell, usdc_amount, profit, unrealized_pnl, drawdown)
                    print(f"[Live] SELL (SL/TP) at {effective_price:.2f}, Profit: {profit:.2f}")
                    position = None
                    sol_balance, usdc_balance = get_wallet_balances(True)

        print(f"[Live] SOL: {sol_balance:.4f}, USDC: {usdc_balance:.2f}, SOL Price: {price:.2f}, Total Profit: {total_profit:.2f}")

        if total_profit < MAX_LOSS:
            print(f"[Live] Maximum loss reached ({total_profit:.2f} < {MAX_LOSS}). Stopping bot...")
            agent.save_model()
            exit(1)

        if total_profit > 0 or drawdown < -50:
            agent.save_model()

        next_state = state
        agent.remember(state, action, reward, next_state, bool(position))
        agent.replay()

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    run_live()
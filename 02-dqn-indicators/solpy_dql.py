import os
import time
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from solders.keypair import Keypair
import base58
import tensorflow as tf
from collections import deque
import random
import pickle
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction, AccountMeta
from solders import message
from solders import system_program
import base64
from http.client import HTTPException

# Load environment variables
load_dotenv()

# Constants
SOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL (WSOL)
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
EMA_SHORT, EMA_LONG, RSI_PERIOD = 9, 21, 7
TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX = 150, 250
MAX_SOL_POSITION, INITIAL_USDC = 5, 800
LOG_FILE = "live_trades.csv"
MODEL_DIR = "dql_model"
MODEL_FILE = f"{MODEL_DIR}/model.keras"
POSITION_FILE = "position.pkl"
SLIPPAGE = 0.01  # 1% slippage
STOP_LOSS, TAKE_PROFIT = 0.05, 0.10  # 5% stop loss, 10% take profit
BUY_COOLDOWN, SELL_COOLDOWN = 7200, 7200  # 2 hours
REFRESH_INTERVAL = 10  # 10 seconds
MAX_LOSS = -100  # Stop if total profit drops below -100 USDC
NETWORK_FEE_LAMPORTS = 5000  # Approx 0.00005 SOL network fee
RSI_OVERBOUGHT = 70  # RSI threshold for overbought condition

# DQL Parameters
STATE_SIZE, ACTION_SIZE = 12, 3
GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY = 0.95, 0.5, 0.1, 0.999985
LEARNING_RATE, MEMORY_SIZE, BATCH_SIZE = 0.0001, 5000, 64
TARGET_UPDATE_FREQ = 1000

# Solana client
alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
if not alchemy_api_key:
    print("[Live] ALCHEMY_API_KEY not found in .env file. Exiting...")
    exit(1)
solana_client = Client(f"https://solana-mainnet.g.alchemy.com/v2/{alchemy_api_key}")

# Wallet setup
try:
    private_key_base58 = os.getenv("PHANTOM_PRIVATE_KEY")
    private_key_bytes = base58.b58decode(private_key_base58)
    wallet = Keypair.from_bytes(private_key_bytes)
    print(f"[Live] Wallet Public Key: {wallet.pubkey()}")
except Exception as e:
    print(f"[Live] Failed to decode PHANTOM_PRIVATE_KEY: {e}")
    exit(1)

# Jupiter API setup
JUPITER_API_BASE = "https://api.jup.ag/swap/v1"
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Rate limiter for Solana API calls
solana_request_timestamps = deque(maxlen=100)
SOLANA_RATE_LIMIT = 40  # Matches ~1000 CUPS at ~25 CUs/request
SOLANA_RATE_LIMIT_WINDOW = 1  # 1 second window

def enforce_solana_rate_limit():
    current_time = time.time()
    while solana_request_timestamps and current_time - solana_request_timestamps[0] > SOLANA_RATE_LIMIT_WINDOW:
        solana_request_timestamps.popleft()
    if len(solana_request_timestamps) >= SOLANA_RATE_LIMIT:
        sleep_time = SOLANA_RATE_LIMIT_WINDOW - (current_time - solana_request_timestamps[0])
        if sleep_time > 0:
            print(f"[Live] Rate limit reached ({len(solana_request_timestamps)} requests). Waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    solana_request_timestamps.append(time.time())

# Cache for wallet balances
last_balance_check = 0
cached_sol_balance = 0
cached_usdc_balance = 0
BALANCE_CACHE_DURATION = 600  # 10 minutes

def fetch_latest_price():
    url = f"https://public-api.birdeye.so/defi/price?address={SOL_MINT}"
    headers = {"X-API-KEY": os.getenv("BIRDEYE_API_KEY")}
    try:
        response = requests.get(url, headers=headers).json()
        if response.get("success"):
            return response["data"]["value"]
        return None
    except Exception:
        return None

def fetch_historical_prices(days=1):
    time_to = int(pd.Timestamp.now().timestamp())
    time_from = time_to - days * 24 * 3600
    url = f"https://public-api.birdeye.so/defi/history_price?address={SOL_MINT}&address_type=token&type=15m&time_from={time_from}&time_to={time_to}"
    headers = {"X-API-KEY": os.getenv("BIRDEYE_API_KEY")}
    try:
        response = requests.get(url, headers=headers).json()
        if response.get("success"):
            data = pd.DataFrame(response["data"]["items"]).rename(columns={"unixTime": "timestamp", "value": "price"})
            return data
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26):
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    return ema_short - ema_long

def calculate_price_change(prices, window=5):
    return prices.pct_change(window).fillna(0)

def calculate_bollinger_band_width(prices, period=20, num_std=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return (upper_band - lower_band).fillna(0)

def normalize_series(series):
    min_val, max_val = series.min(), series.max()
    return (series - min_val) / (max_val - min_val + 1e-6)

def calculate_success_score(state, norm_ema_short, norm_ema_long, norm_rsi, norm_macd, norm_price_change, norm_bb_width, recent_returns, i):
    ema_diff = norm_ema_short.iloc[i] - norm_ema_long.iloc[i]
    macd = norm_macd.iloc[i]
    price_change = norm_price_change.iloc[i]
    recent_return = recent_returns[-1] if i >= 5 else 0
    rsi = norm_rsi.iloc[i]
    bb_width = norm_bb_width.iloc[i]
    trend_score = 0.3 * ema_diff + 0.2 * macd + 0.2 * price_change + 0.1 * recent_return
    mean_reversion_score = 0.1 * (0.5 - rsi)
    success_score = trend_score + mean_reversion_score
    success_score = max(0, min(1, (success_score + 1) / 2)) ** 2
    return success_score

def get_wallet_balances(force_update=False, max_retries=5, retry_delay=10):
    global last_balance_check, cached_sol_balance, cached_usdc_balance
    current_time = time.time()
    if not force_update and (current_time - last_balance_check) < BALANCE_CACHE_DURATION:
        return cached_sol_balance, cached_usdc_balance

    for attempt in range(max_retries):
        try:
            enforce_solana_rate_limit()
            sol_balance = solana_client.get_balance(wallet.pubkey()).value / 1e9
            enforce_solana_rate_limit()
            usdc_mint = Pubkey.from_string(USDC_MINT)
            ata, _ = Pubkey.find_program_address(
                seeds=[bytes(wallet.pubkey()), bytes(SPL_TOKEN_PROGRAM_ID), bytes(usdc_mint)],
                program_id=ASSOCIATED_TOKEN_PROGRAM_ID
            )
            try:
                enforce_solana_rate_limit()
                account_info = solana_client.get_token_account_balance(ata)
                usdc_balance = account_info.value.ui_amount if account_info.value.ui_amount is not None else 0
            except Exception:
                usdc_balance = 0
            cached_sol_balance, cached_usdc_balance = sol_balance, usdc_balance
            last_balance_check = current_time
            print(f"[Live] Fetched balances: SOL={sol_balance:.6f}, USDC={usdc_balance:.2f}")
            return sol_balance, usdc_balance
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"[Live] Solana API rate limit hit. Waiting 60 seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(60)
                continue
            print(f"[Live] HTTP error fetching wallet balances (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            print(f"[Live] Error fetching wallet balances (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("[Live] Failed to fetch wallet balances after max retries. Using cached values.")
                return cached_sol_balance, cached_usdc_balance
            time.sleep(retry_delay)
    return cached_sol_balance, cached_usdc_balance

def log_trade(timestamp, trade_type, price, sol_amount, usdc_amount, profit, unrealized_pnl, drawdown, fees):
    log_entry = pd.DataFrame([{
        "timestamp": pd.Timestamp(timestamp, unit="s"),
        "unixTimestamp": timestamp,
        "tradeType": trade_type,
        "price(SOL/USDC)": price,
        "solAmount": sol_amount,
        "usdcAmount": usdc_amount,
        "profit(USDC)": profit,
        "unrealizedPnL(USDC)": unrealized_pnl,
        "drawdown(USDC)": drawdown,
        "fees(USDC)": fees
    }])
    if not os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    print(f"[Live] Trade: {trade_type} at {price:.2f}, SOL: {sol_amount:.4f}, USDC: {usdc_amount:.2f}, Profit: {profit:.2f}, Fees: {fees:.4f}")

def invalidate_balance_cache():
    global last_balance_check
    last_balance_check = 0  # Force a fresh balance fetch on the next call

def initialize_ata(mint: str) -> Pubkey:
    mint_pubkey = Pubkey.from_string(mint)
    ata, _ = Pubkey.find_program_address(
        seeds=[bytes(wallet.pubkey()), bytes(SPL_TOKEN_PROGRAM_ID), bytes(mint_pubkey)],
        program_id=ASSOCIATED_TOKEN_PROGRAM_ID
    )
    try:
        enforce_solana_rate_limit()
        account_info = solana_client.get_account_info(ata)
        if account_info.value is not None:
            return ata
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("[Live] Solana API rate limit hit during ATA check. Waiting 60 seconds...")
            time.sleep(60)
            return initialize_ata(mint)
        raise
    except Exception:
        pass

    instruction = Instruction(
        program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
        accounts=[
            AccountMeta(pubkey=wallet.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
            AccountMeta(pubkey=wallet.pubkey(), is_signer=False, is_writable=False),
            AccountMeta(pubkey=mint_pubkey, is_signer=False, is_writable=False),
            AccountMeta(pubkey=system_program.ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SPL_TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
        ],
        data=bytes([])
    )
    for attempt in range(3):
        try:
            enforce_solana_rate_limit()
            recent_blockhash_response = solana_client.get_latest_blockhash()
            recent_blockhash = recent_blockhash_response.value.blockhash
            msg = message.Message.new_with_blockhash(
                instructions=[instruction],
                payer=wallet.pubkey(),
                blockhash=recent_blockhash
            )
            tx = VersionedTransaction(msg, [wallet])
            enforce_solana_rate_limit()
            send_response = solana_client.send_transaction(tx)
            tx_signature = send_response.value
            enforce_solana_rate_limit()
            confirmation = solana_client.confirm_transaction(tx_signature, commitment=Confirmed, sleep_seconds=2)
            if confirmation.value is not None:
                return ata
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"[Live] Solana API rate limit hit during ATA initialization. Waiting 60 seconds... (Attempt {attempt + 1}/3)")
                time.sleep(60)
                continue
            raise
        except Exception as e:
            print(f"[Live] Failed to initialize ATA (Attempt {attempt + 1}/3): {e}")
            if attempt == 2:
                return None
    print("[Live] Failed to initialize ATA after 3 attempts.")
    return None

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
        self.load_model()
        self.update_counter = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
        return model

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.model = tf.keras.models.load_model(MODEL_FILE)
            self.target_model = tf.keras.models.load_model(MODEL_FILE)
            print(f"[DQL] Loaded model from {MODEL_FILE}")
            self.target_model.set_weights(self.model.get_weights())
        epsilon_file = os.path.join(MODEL_DIR, "epsilon.pkl")
        if os.path.exists(epsilon_file):
            with open(epsilon_file, "rb") as f:
                self.epsilon = pickle.load(f)
            print(f"[DQL] Loaded epsilon: {self.epsilon}")

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(MODEL_FILE)
        epsilon_file = os.path.join(MODEL_DIR, "epsilon.pkl")
        with open(epsilon_file, "wb") as f:
            pickle.dump(self.epsilon, f)
        print(f"[DQL] Saved model to {MODEL_FILE}, epsilon: {self.epsilon}")

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state)
        state = np.reshape(state, (1, self.state_size))
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                action_next = np.argmax(self.model.predict(np.reshape(next_states[i], (1, self.state_size)), verbose=0)[0])
                target += self.gamma * next_q_values[i][action_next]
            targets[i][actions[i]] = target
        self.model.fit(states, targets, epochs=1, verbose=0)
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("[DQL] Updated target model")
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

last_request_time = 0
request_interval = 2  # Minimum 2 seconds between Jupiter requests

def execute_jupiter_swap(input_mint, output_mint, amount, max_retries=3, retry_delay=5):
    global last_request_time
    decimals = 9 if input_mint == SOL_MINT else 6
    amount_in_decimals = int(amount * (10 ** decimals))
    min_amount = 0.01 * (10 ** decimals)
    if amount_in_decimals < min_amount or amount_in_decimals <= 0:
        print(f"[Live] Swap amount {amount_in_decimals} below minimum {min_amount} or invalid.")
        return False, None, 0

    sol_balance, usdc_balance = get_wallet_balances(force_update=True)
    pre_sol_balance, pre_usdc_balance = sol_balance, usdc_balance
    print(f"[Live] Pre-swap balances: SOL={pre_sol_balance:.6f}, USDC={pre_usdc_balance:.2f}")

    if sol_balance < 0.001:
        print(f"[Live] Insufficient SOL balance for transaction fees: {sol_balance:.6f} SOL")
        return False, None, 0
    if input_mint == USDC_MINT and usdc_balance < amount:
        print(f"[Live] Insufficient USDC balance: {usdc_balance:.2f} < {amount:.2f}")
        return False, None, 0
    if input_mint == SOL_MINT and sol_balance < amount:
        print(f"[Live] Insufficient SOL balance: {sol_balance:.6f} < {amount:.6f}")
        return False, None, 0

    input_ata = initialize_ata(input_mint)
    output_ata = initialize_ata(output_mint)
    if not input_ata or not output_ata:
        print(f"[Live] Failed to initialize ATAs: input_ata={input_ata}, output_ata={output_ata}")
        return False, None, 0

    quote_url = f"{JUPITER_API_BASE}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_in_decimals}&slippageBps={int(SLIPPAGE * 10000)}"
    swap_url = f"{JUPITER_API_BASE}/swap"

    for attempt in range(max_retries):
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < request_interval:
            time.sleep(request_interval - time_since_last_request)
        last_request_time = time.time()

        try:
            response = requests.get(quote_url)
            response.raise_for_status()
            quote_response = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Live] Failed to fetch quote (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        required_fields = ["inputMint", "outputMint", "inAmount", "outAmount"]
        if not all(field in quote_response for field in required_fields):
            print(f"[Live] Invalid quote response: {quote_response}")
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        in_amount = int(quote_response["inAmount"]) / (10 ** (6 if input_mint == USDC_MINT else 9))
        out_amount = int(quote_response["outAmount"]) / (10 ** (9 if output_mint == SOL_MINT else 6))
        if in_amount <= 0 or out_amount <= 0:
            print(f"[Live] Invalid swap amounts: in_amount={in_amount:.2f}, out_amount={out_amount:.6f}")
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        effective_price = out_amount / in_amount if input_mint == SOL_MINT else in_amount / out_amount
        print(f"[Live] Quote: inAmount={in_amount:.2f}, outAmount={out_amount:.6f}, effective_price={effective_price:.2f}")

        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": str(wallet.pubkey()),
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": 200000
        }

        try:
            swap_response = requests.post(swap_url, json=payload)
            swap_response.raise_for_status()
            swap_response = swap_response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Live] Failed to fetch swap transaction (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        if "swapTransaction" not in swap_response:
            print(f"[Live] Swap response missing swapTransaction: {swap_response}")
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        try:
            tx_bytes = base64.b64decode(swap_response["swapTransaction"])
            tx = VersionedTransaction.from_bytes(tx_bytes)
        except Exception as e:
            print(f"[Live] Failed to decode swap transaction (Attempt {attempt + 1}/{max_retries}): {e}")
            if "UnsupportedTransactionVersionMessage" in str(e):
                print("[Live] Retrying swap due to unsupported transaction version...")
                if attempt == max_retries - 1:
                    return False, None, 0
                time.sleep(retry_delay)
                continue
            if attempt == max_retries - 1:
                return False, None, 0
            time.sleep(retry_delay)
            continue

        tx_signature = None
        for attempt_solana in range(3):
            try:
                enforce_solana_rate_limit()
                blockhash_response = solana_client.get_latest_blockhash()
                recent_blockhash = blockhash_response.value.blockhash
                msg = tx.message
                new_msg = message.MessageV0(
                    header=msg.header,
                    account_keys=msg.account_keys,
                    recent_blockhash=recent_blockhash,
                    instructions=msg.instructions,
                    address_table_lookups=msg.address_table_lookups
                )
                tx = VersionedTransaction(new_msg, [wallet])
                enforce_solana_rate_limit()
                send_response = solana_client.send_raw_transaction(bytes(tx))
                tx_signature = send_response.value
                enforce_solana_rate_limit()
                confirmation = solana_client.confirm_transaction(tx_signature, commitment=Confirmed, sleep_seconds=2)
                if confirmation.value is not None:
                    enforce_solana_rate_limit()
                    tx_status = solana_client.get_transaction(tx_signature, commitment=Confirmed)
                    if not tx_status.value or tx_status.value.transaction.meta.err:
                        print(f"[Live] Transaction {tx_signature} failed or has errors: {tx_status.value.transaction.meta.err}")
                        if attempt_solana == 2:
                            return False, None, 0
                        time.sleep(retry_delay)
                        continue

                    max_balance_retries = 5
                    for balance_attempt in range(max_balance_retries):
                        post_sol_balance, post_usdc_balance = get_wallet_balances(force_update=True, max_retries=5, retry_delay=10)
                        usdc_diff = abs(pre_usdc_balance - post_usdc_balance)
                        sol_diff = abs(post_sol_balance - pre_sol_balance)
                        if (input_mint == USDC_MINT and usdc_diff > in_amount * 0.5) or \
                           (input_mint == SOL_MINT and sol_diff > out_amount * 0.5):
                            break
                        print(f"[Live] Balances unchanged after swap (Attempt {balance_attempt + 1}/{max_balance_retries}). Retrying...")
                        if balance_attempt == max_balance_retries - 1:
                            print("[Live] Balances did not update after swap. Using expected amounts for fee calculation.")
                            post_usdc_balance = pre_usdc_balance - in_amount if input_mint == USDC_MINT else pre_usdc_balance + out_amount
                            post_sol_balance = pre_sol_balance + out_amount - (NETWORK_FEE_LAMPORTS / 1e9) if input_mint == USDC_MINT else pre_sol_balance - in_amount - (NETWORK_FEE_LAMPORTS / 1e9)
                        time.sleep(15)
                    print(f"[Live] Post-swap balances: SOL={post_sol_balance:.6f}, USDC={post_usdc_balance:.2f}")

                    network_fee_sol = NETWORK_FEE_LAMPORTS / 1e9
                    network_fee_usdc = network_fee_sol * effective_price

                    if input_mint == USDC_MINT:
                        actual_sol_received = post_sol_balance - pre_sol_balance + network_fee_sol
                        actual_usdc_spent = pre_usdc_balance - post_usdc_balance
                        expected_usdc = actual_sol_received * effective_price
                        jupiter_fee_usdc = actual_usdc_spent - expected_usdc
                    else:
                        actual_sol_spent = pre_sol_balance - post_sol_balance - network_fee_sol
                        actual_usdc_received = post_usdc_balance - pre_usdc_balance
                        expected_usdc = actual_sol_spent * effective_price
                        jupiter_fee_usdc = expected_usdc - actual_usdc_received

                    total_fees = jupiter_fee_usdc + network_fee_usdc
                    total_fees = max(total_fees, network_fee_usdc)
                    total_fees = min(total_fees, (in_amount if input_mint == USDC_MINT else out_amount) * 0.01)
                    if total_fees < 0:
                        print(f"[Live] Warning: Calculated negative fees ({total_fees:.6f}). Falling back to estimate.")
                        jupiter_fee_usdc = (in_amount if input_mint == USDC_MINT else out_amount) * 0.003
                        total_fees = jupiter_fee_usdc + network_fee_usdc

                    print(f"[Live] Fee Breakdown: Jupiter Fee={jupiter_fee_usdc:.6f} USDC, Network Fee={network_fee_usdc:.6f} USDC, Total Fees={total_fees:.6f} USDC")
                    invalidate_balance_cache()
                    return True, effective_price, total_fees
                print(f"[Live] Swap confirmation failed for tx {tx_signature}")
                if attempt_solana == 2:
                    return False, None, 0
                time.sleep(retry_delay)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"[Live] Solana API rate limit hit during swap. Waiting 60 seconds... (Attempt {attempt_solana + 1}/3)")
                    time.sleep(60)
                    continue
                print(f"[Live] HTTP error during swap (Attempt {attempt_solana + 1}/3): {e}")
                if attempt_solana == 2:
                    return False, None, 0
                time.sleep(retry_delay)
            except Exception as e:
                print(f"[Live] Error during swap transaction (Attempt {attempt_solana + 1}/3): {e}")
                if attempt_solana == 2:
                    return False, None, 0
                time.sleep(retry_delay)
    print("[Live] Swap failed after max retries.")
    return False, None, 0

def save_position(position, last_buy_time):
    try:
        state = {"position": position, "last_buy_time": last_buy_time}
        with open(POSITION_FILE, "wb") as f:
            pickle.dump(state, f)
        print(f"[Live] Saved position state to {POSITION_FILE}: {state}")
    except Exception as e:
        print(f"[Live] Failed to save position state: {e}")

def load_position():
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, "rb") as f:
                state = pickle.load(f)
            print(f"[Live] Loaded position state: {state}")
            return state["position"], state["last_buy_time"]
        except Exception as e:
            print(f"[Live] Failed to load position state: {e}")
    print(f"[Live] No position file found at {POSITION_FILE}")
    return None, 0

def infer_position_from_balances(sol_balance, usdc_balance, last_price):
    MIN_SOL_FOR_POSITION = TRADE_AMOUNT_MIN / last_price  # E.g., 150 USDC / price
    if sol_balance > MIN_SOL_FOR_POSITION and usdc_balance < INITIAL_USDC * 0.9:  # USDC should be lower if we bought
        if os.path.exists(LOG_FILE):
            try:
                trades = pd.read_csv(LOG_FILE)
                if "tradeType" not in trades.columns or "price(SOL/USDC)" not in trades.columns:
                    print("[Live] Trade log missing required columns.")
                    return None, 0
                buy_trades = trades[trades["tradeType"] == "BUY"].sort_values(by="unixTimestamp", ascending=False)
                sell_trades = trades[trades["tradeType"] == "SELL"].sort_values(by="unixTimestamp", ascending=False)
                if not buy_trades.empty:
                    last_buy = buy_trades.iloc[0]
                    last_buy_time = int(last_buy["unixTimestamp"])
                    # Check if there's a sell after the last buy
                    if not sell_trades.empty and sell_trades.iloc[0]["unixTimestamp"] > last_buy_time:
                        print("[Live] Last action was a sell. No position exists.")
                        return None, last_buy_time
                    entry_price = last_buy["price(SOL/USDC)"]
                    sol_amount = last_buy["solAmount"]
                    timestamp = last_buy_time
                    print(f"[Live] Inferred position from trade log: entry_price={entry_price}, sol_amount={sol_amount}, timestamp={timestamp}")
                    return {"entry_price": entry_price, "sol_amount": sol_amount, "timestamp": timestamp}, timestamp
            except Exception as e:
                print(f"[Live] Error reading trade log: {e}")
    print("[Live] Could not infer position from balances and trade log.")
    return None, 0

def run_live():
    print("[Live] Starting live trading with Jupiter Aggregator...")
    agent = DoubleDQNAgent(STATE_SIZE, ACTION_SIZE)
    sol_balance, usdc_balance = get_wallet_balances(force_update=True)

    # Load position from file
    position, last_buy_time = load_position()

    # If no position loaded, try to infer from balances and trade log
    if position is None:
        price = fetch_latest_price()
        if price is None:
            print("[Live] Failed to fetch price for position inference. Assuming no position.")
        else:
            position, last_buy_time = infer_position_from_balances(sol_balance, usdc_balance, price)
            if position:
                save_position(position, last_buy_time)

    # Log initial position state
    if position:
        print(f"[Live] Starting with position: entry_price={position['entry_price']}, sol_amount={position['sol_amount']}, timestamp={position['timestamp']}")
    else:
        print("[Live] Starting with no position.")

    total_profit, max_usdc = 0, usdc_balance
    recent_returns = deque([0] * 5, maxlen=5)

    print("[Live] Initializing token accounts...")
    sol_ata = initialize_ata(SOL_MINT)
    usdc_ata = initialize_ata(USDC_MINT)
    if sol_ata is None or usdc_ata is None:
        print("[Live] Failed to initialize token accounts. Exiting...")
        exit(1)

    while True:
        price = fetch_latest_price()
        if price is None:
            print("[Live] Failed to fetch latest price. Retrying in 10 seconds...")
            time.sleep(REFRESH_INTERVAL)
            continue

        price_data = fetch_historical_prices(days=1)
        if len(price_data) < EMA_LONG:
            print(f"[Live] Insufficient historical data: {len(price_data)} < {EMA_LONG}. Retrying in 10 seconds...")
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

        # Check RSI for overbought condition
        current_rsi = rsi.iloc[i]
        print(f"[Live] 15m RSI: {current_rsi:.2f}")

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
            0.0,  # Padding 1
            0.0   # Padding 2
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
        if position:
            print(f"[Debug] Unrealized P&L calculation: price={price}, entry_price={position['entry_price']}, sol_amount={position['sol_amount']}, unrealized_pnl={unrealized_pnl}")
        else:
            print("[Debug] No position for unrealized P&L calculation.")
        drawdown = max_usdc - (usdc_balance + (sol_balance * price if sol_balance else 0))

        # Buy logic with RSI check
        if not position and action == 1 and usdc_balance >= TRADE_AMOUNT_MIN and sol_balance + TRADE_AMOUNT_MIN / price <= MAX_SOL_POSITION and (timestamp - last_buy_time) >= BUY_COOLDOWN:
            # Additional check: if SOL balance is high, we might have a position that wasn't loaded correctly
            if sol_balance > TRADE_AMOUNT_MIN / price:
                print(f"[Live] SOL balance ({sol_balance:.4f}) suggests a position exists, but position state is None. Skipping buy to avoid overbuying.")
                # Try to reinfer the position
                position, last_buy_time = infer_position_from_balances(sol_balance, usdc_balance, price)
                if position:
                    save_position(position, last_buy_time)
                continue
            if current_rsi > RSI_OVERBOUGHT:
                print(f"[Live] Skipping buy: SOL is overbought (RSI = {current_rsi:.2f} > {RSI_OVERBOUGHT})")
            else:
                success_score = calculate_success_score(state, norm_ema_short, norm_ema_long, norm_rsi, norm_macd, norm_price_change, norm_bb_width, recent_returns, i)
                buy_amount = TRADE_AMOUNT_MIN + (TRADE_AMOUNT_MAX - TRADE_AMOUNT_MIN) * success_score
                buy_amount = min(buy_amount, usdc_balance, (MAX_SOL_POSITION - sol_balance) * price)
                success, effective_price, fees = execute_jupiter_swap(USDC_MINT, SOL_MINT, buy_amount)
                if success:
                    sol_amount = buy_amount / effective_price
                    position = {"entry_price": effective_price, "sol_amount": sol_amount, "timestamp": timestamp}
                    usdc_balance -= buy_amount
                    sol_balance += sol_amount
                    last_buy_time = timestamp
                    log_trade(timestamp, "BUY", effective_price, sol_amount, buy_amount, 0, unrealized_pnl, drawdown, fees)
                    print(f"[Live] BUY at {effective_price:.2f}, Fees: {fees:.4f}")
                    sol_balance, usdc_balance = get_wallet_balances(force_update=True)
                    save_position(position, last_buy_time)

        elif position and action == 2 and (timestamp - position["timestamp"]) >= SELL_COOLDOWN:
            sol_to_sell = position["sol_amount"]
            if sol_to_sell <= 0:
                position = None
                save_position(position, last_buy_time)
                continue
            success, effective_price, fees = execute_jupiter_swap(SOL_MINT, USDC_MINT, sol_to_sell)
            if success:
                usdc_amount = effective_price * position["sol_amount"]
                profit = (effective_price - position["entry_price"]) * position["sol_amount"]
                reward = profit
                total_profit += profit
                usdc_balance += usdc_amount
                sol_balance -= position["sol_amount"]
                log_trade(timestamp, "SELL", effective_price, position["sol_amount"], usdc_amount, profit, unrealized_pnl, drawdown, fees)
                print(f"[Live] SELL at {effective_price:.2f}, Profit: {profit:.2f}, Fees: {fees:.4f}")
                position = None
                done = True
                sol_balance, usdc_balance = get_wallet_balances(force_update=True)
                save_position(position, last_buy_time)

        sol_balance, usdc_balance = get_wallet_balances()
        print(f"[Live] SOL: {sol_balance:.4f}, USDC: {usdc_balance:.2f}, SOL Price: {price:.2f}, Total Profit: {total_profit:.2f}, Unrealized P&L: {unrealized_pnl:.2f}, Drawdown: {drawdown:.2f}")

        if total_profit < MAX_LOSS:
            print(f"[Live] Maximum loss reached ({total_profit:.2f} < {MAX_LOSS}). Stopping bot...")
            agent.save_model()
            save_position(position, last_buy_time)
            exit(1)

        if total_profit > 0 or drawdown < -100:
            agent.save_model()
        save_position(position, last_buy_time)  # Save position state periodically

        next_state = state
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    run_live()
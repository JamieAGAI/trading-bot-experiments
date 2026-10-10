import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch  # Keep for device, but PPO uses it internally
from torch.utils.data import Dataset, DataLoader  # Optional, if needed elsewhere
from collections import deque
import requests
from binance.client import Client as BinanceClient
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.transaction import Transaction, VersionedTransaction
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import get_associated_token_address, create_associated_token_account
from solders.system_program import create_account, CreateAccountParams
import base58
import base64
from solana.rpc.commitment import Confirmed
from solders.message import to_bytes_versioned
from tqdm import tqdm
import random
import zipfile
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob  # For sentiment analysis

# New imports for PPO and Gym
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # For single env
from stable_baselines3.common.callbacks import BaseCallback

# Load .env
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()
ALCHEMY_RPC = os.getenv('ALCHEMY_RPC', 'https://solana-mainnet.g.alchemy.com/v2/your_key')
WALLET_PRIVATE_KEY = os.getenv('WALLET_PRIVATE_KEY', 'your_base58_key')
TWITTER_BEARER = os.getenv('TWITTER_BEARER', 'your_token')

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Constants (unchanged, but adjust if needed)
USDC_MINT = Pubkey.from_string('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
SOL_MINT = Pubkey.from_string('So11111111111111111111111111111111111111112')
JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
lookback_left = 6
lookback_right = 6
atr_len = 14
atr_mult = 2
rsi_buy_thresh = 50
rsi_sell_thresh = 60
rsi_len = 6
SLIPPAGE = 0.005
BUY_PERCENTAGE = 0.2
SELL_PERCENTAGE = 0.2
DATA_CACHE = 'sol_usdt_1700_days_15min.csv'
MIN_TRADE_AMOUNT = 10.0
TRADE_FEE = 0.001
IMITATION_EPOCHS = 300  # Increased (but skipping imitation for PPO simplicity)
IMITATION_BIAS = 0.6  # Not used in PPO
SIGNALS_CSV = 'solana_strategy_signals.csv'  # Skipping pre-train for now
SEQ_LEN = 10  # Increased for trends
MAX_DD_THRESHOLD = 0.05  # Temp higher for training
EPISODES = 20  # As discussed
DAYS_PER_EPISODE = 500  # As discussed

# News sites for sentiment (unchanged)
NEWS_SITES = [
    'https://coindesk.com/search?s=solana',
    'https://cointelegraph.com/search?query=solana',
    'https://cryptonews.com/search/?q=solana',
    'https://solana.com/news',
    'https://www.reddit.com/r/solana/new.json'
]

# Helpers: Get balances (unchanged)
def get_balances(wallet_pubkey, solana_client):
    sol = solana_client.get_balance(wallet_pubkey).value / 1e9
    usdc_ata = get_associated_token_address(wallet_pubkey, USDC_MINT)
    usdc = solana_client.get_token_account_balance(usdc_ata).value.ui_amount or 0.0
    return sol, usdc

# Helpers: Initialize ATA (unchanged)
def initialize_ata(mint, wallet_keypair, solana_client):
    ata = get_associated_token_address(wallet_keypair.pubkey(), mint)
    try:
        solana_client.get_token_account_balance(ata)
    except:
        tx = Transaction.new_signed(
            wallet_keypair,
            [create_associated_token_account(wallet_keypair.pubkey(), wallet_keypair.pubkey(), mint)],
            solana_client.get_latest_blockhash().value.blockhash
        )
        solana_client.send_transaction(tx)
    return ata

# Jupiter swap function (unchanged)
def jupiter_swap(wallet_keypair, input_mint, output_mint, amount, solana_client, retries=3):
    if amount < MIN_TRADE_AMOUNT:
        print(f"Skipping small swap: {amount} below min {MIN_TRADE_AMOUNT}")
        return False
    decimals_in = 6 if input_mint == USDC_MINT else 9
    amount_lamports = int(amount * 10**decimals_in)
    if amount_lamports < 10**(decimals_in - 2):
        return False

    wallet_pubkey = wallet_keypair.pubkey()
    sol_bal, usdc_bal = get_balances(wallet_pubkey, solana_client)
    if (input_mint == USDC_MINT and usdc_bal < amount) or (input_mint == SOL_MINT and sol_bal < amount):
        return False

    input_ata = initialize_ata(input_mint, wallet_keypair, solana_client)
    output_ata = initialize_ata(output_mint, wallet_keypair, solana_client)
    if not input_ata or not output_ata:
        return False

    quote_url = f"{JUPITER_API_BASE}/quote?inputMint={str(input_mint)}&outputMint={str(output_mint)}&amount={amount_lamports}&slippageBps={int(SLIPPAGE*10000)}"
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
            price = amount / out_decimal if input_mint == USDC_MINT else out_decimal / amount

            jup_fee = out_decimal * 0.002
            net_fee_usd = jup_fee * price if output_mint == SOL_MINT else jup_fee
            net_fee_usd += 0.000005 * price

            payload = {
                "quoteResponse": quote,
                "userPublicKey": str(wallet_pubkey),
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": 100000
            }
            swap_resp = requests.post(f"{JUPITER_API_BASE}/swap", json=payload).json()
            if 'error' in swap_resp:
                print(f"Swap error: {swap_resp['error']}")
                time.sleep(5)
                continue
            tx_bytes = base64.b64decode(swap_resp['swapTransaction'])
            tx = VersionedTransaction.from_bytes(tx_bytes)
            msg_bytes = to_bytes_versioned(tx.message)
            sig = wallet_keypair.sign_message(msg_bytes)
            tx.signatures = [sig]

            bh = solana_client.get_latest_blockhash(Confirmed).value.blockhash
            send_sig = solana_client.send_raw_transaction(bytes(tx), Confirmed).value
            solana_client.confirm_transaction(send_sig, commitment=Confirmed, sleep_seconds=5)
            print(f"Swap executed: {amount} {input_mint} to {out_decimal} {output_mint}, Fees ~{net_fee_usd:.4f} USD")
            return True
        except Exception as e:
            print(f"Swap attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False

# Fetch historical data (with error handling)
def fetch_historical_data(days=1700):
    if os.path.exists(DATA_CACHE):
        print("Loading cached data...")
        try:
            return pd.read_csv(DATA_CACHE, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print("Fetching new data...")
    client = BinanceClient('', '')
    interval = BinanceClient.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(DATA_CACHE)
    return df

# Compute indicators (complete version with all calculations, plus NaN filling)
def compute_indicators(df):
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma100'] = df['close'].rolling(100).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_len).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['vol_ma'] = df['volume'].rolling(5).mean()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(atr_len).mean()
    
    pivot_window = lookback_left + lookback_right + 1
    df['pivot_high'] = df['high'].rolling(window=pivot_window, center=True).max()
    df['pivot_low'] = df['low'].rolling(window=pivot_window, center=True).min()
    
    lookback_fib = lookback_left * 2
    df['swing_high'] = df['high'].rolling(lookback_fib).max()
    df['swing_low'] = df['low'].rolling(lookback_fib).min()
    fib_range = df['swing_high'] - df['swing_low']
    df['fib382'] = df['swing_high'] - fib_range * 0.382
    df['fib50'] = df['swing_high'] - fib_range * 0.5
    df['fib618'] = df['swing_high'] - fib_range * 0.618
    
    # ADX, DI+, DI-
    dm_plus = ((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low'])) * (df['high'] - df['high'].shift())
    dm_minus = ((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift())) * (df['low'].shift() - df['low'])
    dm_plus = dm_plus.rolling(atr_len).mean()
    dm_minus = dm_minus.rolling(atr_len).mean()
    di_plus = 100 * dm_plus / df['atr']
    di_minus = 100 * dm_minus / df['atr']
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    df['adx'] = dx.rolling(atr_len).mean()
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    # Add price_change as a column (to fix KeyError)
    df['price_change'] = df['close'].pct_change().fillna(0)
    
    # Fill NaNs with 0 to prevent propagation
    df = df.fillna(0)
    
    return df

# Sentiment functions
def fetch_news_sentiment():
    try:
        url = NEWS_SITES[0]  # e.g., Coindesk
        response = requests.get(url)
        text = response.text[:1000]  # Limit to avoid overload
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Between -1 and 1
    except:
        return 0.0  # Fallback

def fetch_twitter_sentiment():
    try:
        url = "https://api.twitter.com/2/tweets/search/recent?query=solana&max_results=10"
        headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
        response = requests.get(url, headers=headers).json()
        tweets = [tweet['text'] for tweet in response.get('data', [])]
        sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
        return np.mean(sentiments) if sentiments else 0.0
    except:
        return 0.0  # Fallback

# Trading Callback for logging (fixed for PPO locals)
class TradingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward per step
        self.current_episode_reward += self.locals['rewards'][0]  # For single env
        
        if self.locals['dones'][0]:  # Check for episode end using 'dones' (vec env)
            self.episode_rewards.append(self.current_episode_reward)
            print(f"Episode {len(self.episode_rewards)}: Reward {self.current_episode_reward}")
            self.current_episode_reward = 0.0  # Reset for next episode
        return True

# Trading Env class (updated for Gym and PPO with flattening)
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=60000, seq_len=SEQ_LEN):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.current_step = seq_len - 1  # Start after enough for sequence
        self.balance = initial_balance
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [initial_balance]
        self.max_net_worth = initial_balance
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.seq_len = seq_len
        self.state_history = deque(maxlen=seq_len)  # For sequences
        self.scaler = StandardScaler()  # For state normalization
        self._fit_scaler()
        
        # Define spaces for Gym (flattened for MlpPolicy)
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_len * 17,), dtype=np.float32)  # Flattened

    def _fit_scaler(self):
        # Fit scaler on existing columns (removed 'price_change' since it's added later; removed duplicate 'close')
        scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
        features = self.df[scaler_columns].dropna().values  # Drop NaNs before fitting to avoid errors
        if len(features) > 0:
            self.scaler.fit(features)
        else:
            print("Warning: No data for scaler fit - using identity")
            # Fallback if all NaN

    def reset(self, *, seed=None, options=None):
        self.current_step = self.seq_len - 1
        self.balance = 60000
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [60000]
        self.max_net_worth = 60000
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.state_history = deque(maxlen=self.seq_len)
        for _ in range(self.seq_len):
            self.state_history.append(self._get_single_state(0))  # Initialize with first state
        return np.array(self.state_history).flatten(), {}  # Flatten to 1D

    def _get_single_state(self, step):
        row = self.df.iloc[step]
        prev_close = self.df.iloc[step-1]['close'] if step > 0 else row['close']
        price_change = (row['close'] - prev_close) / prev_close if step > 0 else 0
        
        dyn_tolerance = row['atr'] * atr_mult if 'atr' in row else row.get('atr_val', 1) * atr_mult
        near_support = abs(row['close'] - row.get('pivot_low', row['close'])) < dyn_tolerance
        buy_signal = near_support and row['close'] > row.get('ma50', row['close']) and row['rsi'] < rsi_buy_thresh and row['close'] > row['ma200'] and row['volume'] > row.get('vol_ma', row['volume'])
        near_resistance = abs(row['close'] - row.get('pivot_high', row['close'])) < dyn_tolerance
        sell_signal = near_resistance and row['rsi'] > rsi_sell_thresh and row['close'] > row['ma200']
        indicator_action = 1 if buy_signal else (2 if sell_signal else 0)
        
        # Simulate sentiment for training/backtest (correlated with price trend)
        news_sent = np.clip(np.random.normal(price_change * 5, 0.2), -1, 1)  # Biased by trend
        twitter_sent = np.clip(np.random.normal(price_change * 3, 0.3), -1, 1)
        
        raw_state = np.array([
            row['close'], row['rsi'], row.get('atr', row.get('atr_val', 1)), row['ma200'],
            row.get('pivot_high', row['close']) - row['close'],
            row['close'] - row.get('fib618', row['close']),
            row.get('adx', 25), row.get('di_plus', 0), row.get('di_minus', 0)
        ])
        normalized_features = self.scaler.transform(raw_state.reshape(1, -1)).flatten()
        
        state = np.concatenate([
            normalized_features,
            np.array([
                1 if row['close'] > row['ma200'] else 0,
                1 if row['volume'] > row['vol_ma'] else 0,
                1 if row['rsi'] < rsi_buy_thresh else 0,
                row['close'] - row.get('pivot_low', row['close']),
                price_change,  # Moved here, not normalized (dynamic)
                indicator_action,
                news_sent, twitter_sent
            ])
        ]).astype(np.float32)
        state = np.nan_to_num(state, nan=0.0)  # Safeguard against NaNs
        return state

    def get_state(self):
        return np.array(self.state_history)  # 2D internally

    def step(self, action):
        row = self.df.iloc[self.current_step]
        reward = 0
        unrealized = (row['close'] - self.entry_price) * self.position if self.position > 0 and self.entry_price is not None else 0
        
        if action == 1 and self.balance > 100:  # Skip if low balance
            buy_value = self.balance * BUY_PERCENTAGE
            buy_amount = buy_value / row['close']
            self.position += buy_amount
            self.entry_price = row['close']  # Set entry price
            self.balance -= buy_value
            self.balance *= (1 - TRADE_FEE)
            reward += 0.2 if row['rsi'] < rsi_buy_thresh and row.get('adx', 0) > 25 else 0.1  # Boost for strong signals
            self.trade_count += 1
        elif action == 2 and self.position > 0:
            sell_value = self.position * row['close']
            profit = sell_value - (self.position * self.entry_price) if self.entry_price is not None else 0
            self.balance += sell_value * (1 - TRADE_FEE)
            self.position = 0
            self.entry_price = None
            reward += (profit / 60000 * 5) if profit > 0 else -0.1  # Penalize negative sells
            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1
            self.trade_count += 1
        
        # Advance step and check done components
        self.current_step += 1
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        current_dd = max(0, 1 - net_worth / self.max_net_worth) if self.max_net_worth > 0 else 0
        terminated = current_dd > MAX_DD_THRESHOLD  # Failure condition (DD threshold)
        truncated = self.current_step >= len(self.df) - 1  # Step limit (end of data)
        done = terminated or truncated
        self.net_worth_history.append(net_worth)
        self.max_net_worth = max(self.max_net_worth, net_worth)
        
        # Update state history with new state
        new_state = self._get_single_state(self.current_step)
        self.state_history.append(new_state)
        
        # Reward shaping: Include Sharpe-like (mean return / std) over recent history
        if len(self.net_worth_history) > 20:
            recent_history = self.net_worth_history[-21:]
            returns = np.diff(recent_history) / recent_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 96)  # Annualized for 15min
            reward += sharpe * 0.1  # Small boost for consistent performance
        
        # Penalize large unrealized losses
        if unrealized < -0.05 * self.balance:  # e.g., >5% unrealized loss
            reward -= 0.2
        
        return self.get_state().flatten(), reward, terminated, truncated, {}  # Now 5 values

def main(backtest=False):
    df_full = fetch_historical_data(days=1700)  # Updated to 1700 as suggested
    df_full = compute_indicators(df_full)
    print(f"Full data length: {len(df_full)} rows")
    if len(df_full) < 1000:
        print("Error: Data too short! Delete cache and retry.")
        return

    if backtest:
        # True holdout split
        holdout_rows = 9600 + SEQ_LEN  # ~100 days holdout
        train_rows = int(DAYS_PER_EPISODE * 96) + SEQ_LEN  # ~500 days for train

        # Split: Train on older data, holdout on recent unseen
        train_start_idx = len(df_full) - holdout_rows - train_rows
        train_df = df_full.iloc[train_start_idx : train_start_idx + train_rows]
        holdout_df = df_full.tail(holdout_rows)

        # Debug prints to verify no overlap
        print(f"Train df length: {len(train_df)}")
        print(f"Train time range: {train_df.index[0]} to {train_df.index[-1]}")
        print(f"Holdout df length: {len(holdout_df)}")
        print(f"Holdout time range: {holdout_df.index[0]} to {holdout_df.index[-1]}")

        # Use train_df for training
        df = train_df

        days_per_episode = DAYS_PER_EPISODE
        expected_rows = int(days_per_episode * 96)  # 96 15-min intervals per day
        print(f"Subsampled length: {len(df)} rows for {days_per_episode} days")
        
        # Wrap env in VecEnv for SB3
        env = DummyVecEnv([lambda: TradingEnv(df)])
        
        # Calculate total timesteps equivalent to original episodes
        steps_per_episode = len(df) - SEQ_LEN  # Approx steps per episode
        total_timesteps = EPISODES * steps_per_episode
        
        # Load or create model
        if os.path.exists('ppo_model.zip'):
            model = PPO.load('ppo_model', env=env)
            print("Loaded existing model for continued training.")
            # Continue learning with half original timesteps (adjust as needed)
            additional_timesteps = total_timesteps // 2
            callback = TradingCallback()
            model.learn(total_timesteps=additional_timesteps, progress_bar=True, callback=callback)
        else:
            model = PPO("MlpPolicy", env, verbose=1, device=device)
            print("Created new model.")
            callback = TradingCallback()
            model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
        
        # Save model (SB3 format) - already zipped, no extra zip needed
        model.save('ppo_model')
        
        # Final eval (adapted)
        eval_env = TradingEnv(holdout_df)
        obs, _ = eval_env.reset()  # Get obs
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = eval_env.step(action)
        eval_max_dd = max(0, 1 - min(eval_env.net_worth_history) / eval_env.max_net_worth) if eval_env.max_net_worth > 0 else 0
        profit_pct = ((eval_env.balance - 60000) / 60000) * 100
        eval_total_closed = eval_env.wins + eval_env.losses
        eval_win_rate = eval_env.wins / eval_total_closed if eval_total_closed > 0 else 0
        print(f"Final Eval: Balance {eval_env.balance:.2f} ({profit_pct:.2f}% profit), DD {eval_max_dd*100:.2f}%, Trades {eval_env.trade_count}, Win Rate {eval_win_rate*100:.2f}%")
        
    else:
        solana_client = SolanaClient(ALCHEMY_RPC)
        wallet_keypair = Keypair.from_base58(WALLET_PRIVATE_KEY)
        wallet_pubkey = wallet_keypair.pubkey()
        
        # Load PPO model
        dummy_env = DummyVecEnv([lambda: TradingEnv(df_full.tail(1000 + SEQ_LEN))])  # Dummy env for loading
        model = PPO.load('ppo_model', env=dummy_env)
        
        state_history = deque(maxlen=SEQ_LEN)
        for _ in range(SEQ_LEN):
            state_history.append(np.zeros(17, dtype=np.float32))  # Updated dim
        while True:
            latest_df = fetch_historical_data(days=1)
            latest_df = compute_indicators(latest_df)
            if len(latest_df) < SEQ_LEN:
                time.sleep(300)
                continue
            # Build real sequence for live
            for i in range(-SEQ_LEN, 0):
                row = latest_df.iloc[i]
                prev_close = latest_df.iloc[i-1]['close'] if i > -SEQ_LEN else row['close']
                price_change = (row['close'] - prev_close) / prev_close if i > -SEQ_LEN else 0
                dyn_tolerance = row['atr'] * atr_mult
                near_support = abs(row['close'] - row.get('pivot_low', row['close'])) < dyn_tolerance
                buy_signal = near_support and row['close'] > row.get('ma50', row['close']) and row.get('rsi', 50) < rsi_buy_thresh and row['close'] > row['ma200'] and row['volume'] > row.get('vol_ma', row['volume'])
                near_resistance = abs(row['close'] - row.get('pivot_high', row['close'])) < dyn_tolerance
                sell_signal = near_resistance and row.get('rsi', 50) > rsi_sell_thresh and row['close'] > row['ma200']
                indicator_action = 1 if buy_signal else (2 if sell_signal else 0)
                state = np.array([
                    row['close'], row.get('rsi', 50), row.get('atr', 1), row.get('ma200', row['close']),
                    1 if row['close'] > row.get('ma200', row['close']) else 0,
                    row['volume'] > row.get('vol_ma', row['volume']),
                    row['close'] - row.get('pivot_low', row['close']),
                    row.get('pivot_high', row['close']) - row['close'],
                    1 if row.get('rsi', 50) < rsi_buy_thresh else 0,
                    row['close'] - row.get('fib618', row['close']),
                    row.get('adx', 25), row.get('di_plus', 0), row.get('di_minus', 0),
                    price_change,  # Added
                    indicator_action,
                    fetch_news_sentiment(), fetch_twitter_sentiment()
                ], dtype=np.float32)
                state_history.append(state)
            state = np.array(state_history).flatten()  # Flatten for PPO
            
            # Use PPO to predict action
            action, _ = model.predict(state)
            
            latest = latest_df.iloc[-1]
            dyn_tolerance = latest['atr'] * atr_mult
            near_support = abs(latest['close'] - latest.get('pivot_low', latest['close'])) < dyn_tolerance
            buy_signal = near_support and latest['close'] > latest.get('ma50', latest['close']) and latest.get('rsi', 50) < rsi_buy_thresh and latest['close'] > latest['ma200'] and latest['volume'] > latest.get('vol_ma', latest['volume'])
            near_resistance = abs(latest['close'] - latest.get('pivot_high', latest['close'])) < dyn_tolerance
            sell_signal = near_resistance and latest.get('rsi', 50) > rsi_sell_thresh and latest['close'] > latest['ma200']
            
            usdc_bal, sol_bal = get_balances(wallet_pubkey, solana_client)
            if (action == 1 or (buy_signal and state[-2] > 0 and state[-1] > 0)) and usdc_bal > MIN_TRADE_AMOUNT:
                amount_in = usdc_bal * BUY_PERCENTAGE
                jupiter_swap(wallet_keypair, USDC_MINT, SOL_MINT, amount_in, solana_client)
            elif (action == 2 or sell_signal) and sol_bal * latest['close'] > MIN_TRADE_AMOUNT:
                amount_in = sol_bal * SELL_PERCENTAGE
                jupiter_swap(wallet_keypair, SOL_MINT, USDC_MINT, amount_in, solana_client)
            
            time.sleep(300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    args = parser.parse_args()
    main(args.backtest)
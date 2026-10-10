import argparse
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque
from sklearn.preprocessing import RobustScaler
from binance.client import Client as BinanceClient
import time
import csv

np.random.seed(42)  # For determinism

# Constants from v3.py (trial 32)
SLIPPAGE = 0.005  # Unused in eval, included for completeness
MIN_TRADE_AMOUNT = 10.0
TRADE_FEE = 0.003
SEQ_LEN = 40
MAX_DD_THRESHOLD = 0.15
rsi_len = 6
rsi_sell_thresh = 60
atr_len = 14
STOP_LOSS_THRESHOLD = -0.03
BUY_PERCENTAGE = 1  # Conservative sizing
lookback_left = 6
lookback_right = 4
atr_mult = 2.9290654217059315
rsi_buy_thresh = 53
action_threshold = 0.309719234200141

# Reward params from v3.py
sharpe_weight = 0.17182410053386737
sortino_weight = 0.1918212095779503
unrealized_penalty = 0.18071713763630204
dd_penalty_weight = 0.09242998397703278
activity_incentive = 0.030979983572360145
hold_penalty = -0.023654420232985213
unrealized_threshold = -0.05
dd_threshold = 0.1

# Fetch historical data
def fetch_historical_data(days=730, force_fetch=False):
    data_cache = f'sol_usdt_{days}_days_15min.csv'
    if not force_fetch and os.path.exists(data_cache):
        print(f"Loading cached data for {days} days...")
        try:
            return pd.read_csv(data_cache, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print(f"Fetching new data for {days} days...")
    client = BinanceClient('', '')
    interval = BinanceClient.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = []
    while True:
        try:
            new_klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time, limit=1000)
            if not new_klines:
                break
            klines += new_klines
            start_time = new_klines[-1][0] + 1
        except Exception as e:
            print(f"API fetch error: {e}. Retrying in 5s...")
            time.sleep(5)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(data_cache)
    return df

# Compute indicators
def compute_indicators(df, lookback_left, lookback_right):
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
    df['pivot_high'] = df['high'].rolling(window=pivot_window, center=False).max()
    df['pivot_low'] = df['low'].rolling(window=pivot_window, center=False).min()
    
    lookback_fib = lookback_left * 2
    df['swing_high'] = df['high'].rolling(lookback_fib).max()
    df['swing_low'] = df['low'].rolling(lookback_fib).min()
    fib_range = df['swing_high'] - df['swing_low']
    df['fib382'] = df['swing_high'] - fib_range * 0.382
    df['fib50'] = df['swing_high'] - fib_range * 0.5
    df['fib618'] = df['swing_high'] - fib_range * 0.618
    
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

    df['price_change'] = df['close'].pct_change().fillna(0)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

# MDD calculation
def calculate_mdd(net_worth_history):
    if not net_worth_history:
        return 0.0, 0, 0
    mdd = 0.0
    peak = net_worth_history[0]
    peak_index = 0
    trough_index = 0
    for i, value in enumerate(net_worth_history):
        if value > peak:
            peak = value
            peak_index = i
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
            trough_index = i
    return mdd, peak_index, trough_index

# TradingEnv from v3.py
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=60000, seq_len=SEQ_LEN, scaler=None, fixed_buy_pct=None, is_eval=True,
                 sharpe_weight=sharpe_weight, sortino_weight=sortino_weight, unrealized_penalty=unrealized_penalty,
                 dd_penalty_weight=dd_penalty_weight, activity_incentive=activity_incentive, hold_penalty=hold_penalty,
                 unrealized_threshold=unrealized_threshold, dd_threshold=dd_threshold):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.current_step = seq_len - 1
        self.balance = initial_balance
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [initial_balance]
        self.max_net_worth = initial_balance
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.seq_len = seq_len
        self.state_history = deque(maxlen=seq_len)
        self.scaler = scaler if scaler is not None else RobustScaler()
        self.lookback_left = lookback_left
        self.lookback_right = lookback_right
        self.atr_mult = atr_mult
        self.rsi_buy_thresh = rsi_buy_thresh
        self.action_threshold = action_threshold
        self.trade_drawdowns = []
        self.current_trade_net_worths = []
        self.current_trade_max = initial_balance
        self.trade_log = []
        self.fixed_buy_pct = fixed_buy_pct
        self.is_eval = is_eval
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.unrealized_penalty = unrealized_penalty
        self.dd_penalty_weight = dd_penalty_weight
        self.activity_incentive = activity_incentive
        self.hold_penalty = hold_penalty
        self.unrealized_threshold = unrealized_threshold
        self.dd_threshold = dd_threshold

        if scaler is None:
            self._fit_scaler()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_len * 18,), dtype=np.float32)

    def _fit_scaler(self):
        scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
        features_df = self.df[scaler_columns].replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        features = features_df.values
        if len(features) > 1:
            self.scaler.fit(features)
        else:
            print("Warning: Insufficient data for scaler fit - using identity")

    def _get_single_state(self, step):
        row = self.df.iloc[step]
        prev_close = self.df.iloc[step-1]['close'] if step > 0 else row['close']
        price_change = (row['close'] - prev_close) / prev_close if step > 0 else 0
        
        dyn_tolerance = row['atr'] * self.atr_mult
        near_support = abs(row['close'] - row.get('pivot_low', row['close'])) < dyn_tolerance
        buy_signal = near_support and row['close'] > row.get('ma50', row['close']) and row['rsi'] < self.rsi_buy_thresh and row['close'] > row['ma200'] and row['volume'] > row.get('vol_ma', row['volume'])
        near_resistance = abs(row['close'] - row.get('pivot_high', row['close'])) < dyn_tolerance
        sell_signal = near_resistance and row['rsi'] > rsi_sell_thresh and row['close'] > row['ma200']
        indicator_action = 1 if buy_signal else (2 if sell_signal else 0)
        
        raw_state = np.array([
            row['close'], row['rsi'], row.get('atr', 1), row['ma200'],
            row.get('pivot_high', row['close']) - row['close'],
            row['close'] - row.get('fib618', row['close']),
            row.get('adx', 25), row.get('di_plus', 0), row.get('di_minus', 0)
        ])
        normalized_features = self.scaler.transform(raw_state.reshape(1, -1)).flatten()
        
        # Portfolio state features
        unrealized = (row['close'] - self.entry_price) * self.position if self.position > 0 and self.entry_price is not None else 0
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        position_norm = self.position / (self.balance / row['close']) if self.balance > 0 and row['close'] > 0 else 0
        unrealized_pct = unrealized / self.balance if self.balance > 0 else 0
        net_worth_ratio = net_worth / self.initial_balance
        
        state = np.concatenate([
            normalized_features,
            np.array([
                1 if row['close'] > row['ma200'] else 0,
                1 if row['volume'] > row['vol_ma'] else 0,
                1 if row['rsi'] < self.rsi_buy_thresh else 0,
                row['close'] - row.get('pivot_low', row['close']),
                price_change,
                indicator_action,
                position_norm,
                unrealized_pct,
                net_worth_ratio
            ])
        ]).astype(np.float32)
        state = np.nan_to_num(state, nan=0.0)
        return state

    def get_state(self):
        return np.array(self.state_history)

    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [self.initial_balance]
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.trade_drawdowns = []
        self.current_trade_net_worths = []
        self.current_trade_max = self.initial_balance
        self.trade_log = []
        self.state_history = deque(maxlen=self.seq_len)
        for i in range(self.seq_len):
            self.state_history.append(self._get_single_state(i))
        self.current_step = self.seq_len - 1
        return self.get_state().flatten(), {}

    def step(self, action):
        action = action[0]  # Unpack
        row = self.df.iloc[self.current_step]
        reward = 0
        unrealized = (row['close'] - self.entry_price) * self.position if self.position > 0 and self.entry_price is not None else 0
        
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        buy_percentage = self.fixed_buy_pct if self.fixed_buy_pct is not None else BUY_PERCENTAGE
        
        if self.position > 0 and unrealized < STOP_LOSS_THRESHOLD * (self.position * row['close']):
            action = -1
            reward -= 0.5
        
        if action > self.action_threshold:
            buy_value = self.balance * buy_percentage
            if buy_value >= MIN_TRADE_AMOUNT and buy_value <= self.balance:
                buy_amount = buy_value / row['close']
                self.position += buy_amount
                self.entry_price = (self.entry_price * (self.position - buy_amount) + row['close'] * buy_amount) / self.position if self.position > buy_amount else row['close']
                self.balance -= buy_value
                self.balance *= (1 - TRADE_FEE)
                reward += 0.2 if row['rsi'] < self.rsi_buy_thresh and row.get('adx', 0) > 25 else 0.1
                reward += self.activity_incentive
                self.trade_count += 1
                self.trade_log.append([self.current_step, 'Buy', row['close'], buy_amount, 0, net_worth, 0])
                self.current_trade_net_worths = [net_worth]  # Reset trade tracking on buy
                self.current_trade_max = net_worth
        elif action < -self.action_threshold:
            sell_amount = self.position * 1.0
            if sell_amount * row['close'] >= MIN_TRADE_AMOUNT and sell_amount > 0:
                sell_value = sell_amount * row['close']
                profit = sell_value - (sell_amount * self.entry_price) if self.entry_price is not None else 0
                self.balance += sell_value * (1 - TRADE_FEE)
                self.position -= sell_amount
                if self.position <= 0:
                    self.entry_price = None
                reward += (profit / 60000 * 5) if profit > 0 else -0.1
                reward += self.activity_incentive
                if profit > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.trade_count += 1
                trade_dd = max(0, 1 - min(self.current_trade_net_worths) / self.current_trade_max) if self.current_trade_net_worths else 0
                self.trade_drawdowns.append(trade_dd)
                self.trade_log.append([self.current_step, 'Sell', row['close'], sell_amount, profit, net_worth, trade_dd * 100])
                self.current_trade_net_worths = []  # Reset after sell
                self.current_trade_max = 0
        else:
            if self.position > 0:
                reward += self.hold_penalty
        
        self.current_step += 1
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        if self.position > 0:
            self.current_trade_net_worths.append(net_worth)
            self.current_trade_max = max(self.current_trade_max, net_worth)
        current_dd = max(0, 1 - net_worth / self.max_net_worth) if self.max_net_worth > 0 else 0
        terminated = current_dd > MAX_DD_THRESHOLD
        truncated = self.current_step >= len(self.df) - 1
        self.net_worth_history.append(net_worth)
        self.max_net_worth = max(self.max_net_worth, net_worth)
        
        if not self.is_eval or len(self.net_worth_history) > 20:
            recent_history = self.net_worth_history[-21:]
            returns = np.diff(recent_history) / recent_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 96)
            reward += sharpe * self.sharpe_weight
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                sortino_std = np.std(downside_returns)
                sortino = np.mean(returns) / (sortino_std + 1e-6) * np.sqrt(252 * 96)
            else:
                sortino = 0
            reward += sortino * self.sortino_weight
        
        if unrealized < self.unrealized_threshold * self.balance:
            reward -= self.unrealized_penalty
        
        reward -= current_dd * self.dd_penalty_weight if current_dd > self.dd_threshold else 0
        
        if self.current_step < len(self.df):
            self.state_history.append(self._get_single_state(self.current_step))
        
        self.balance = max(self.balance, 0)  # Prevent negative balance
        
        return self.get_state().flatten(), reward, terminated, truncated, {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ppo_trial32.zip', help='Path to model zip')
    parser.add_argument('--days', type=int, default=730, help='Number of days to fetch data for')
    parser.add_argument('--holdout_days', type=int, default=365, help='Number of holdout days for evaluation')
    parser.add_argument('--balance', type=float, default=60000, help='Initial balance for eval')
    parser.add_argument('--buy_pct', type=float, default=None, help='Fixed buy percentage (override dynamic)')
    args = parser.parse_args()

    # Init log CSV
    with open('trade_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Action', 'Price', 'Amount', 'Profit', 'Net_Worth', 'Trade_DD (%)'])

    df_full = fetch_historical_data(days=args.days)
    df_full = compute_indicators(df_full, lookback_left=lookback_left, lookback_right=lookback_right)
    print(f"Full data length: {len(df_full)} rows")

    holdout_rows = args.holdout_days * 96 + SEQ_LEN
    holdout_start = len(df_full) - holdout_rows
    holdout_df = df_full.iloc[holdout_start:].reset_index()
    print(f"Holdout data days approx: {len(holdout_df) / 96:.0f}")

    # Fit scaler on pre-holdout data (for fairness, but since eval, could use full if desired)
    scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
    pre_holdout_df = df_full.iloc[:holdout_start]
    features_df = pre_holdout_df[scaler_columns].replace([np.inf, -np.inf], np.nan).dropna()
    features = features_df.values
    scaler = RobustScaler()
    if len(features) > 1:
        scaler.fit(features)
    else:
        print("Warning: Insufficient data for scaler fit - using identity")

    eval_env = TradingEnv(holdout_df, initial_balance=args.balance, scaler=scaler, fixed_buy_pct=args.buy_pct, is_eval=True,
                          sharpe_weight=sharpe_weight, sortino_weight=sortino_weight,
                          unrealized_penalty=unrealized_penalty, dd_penalty_weight=dd_penalty_weight,
                          activity_incentive=activity_incentive, hold_penalty=hold_penalty,
                          unrealized_threshold=unrealized_threshold, dd_threshold=dd_threshold)
    print(f"Observation space shape: {eval_env.observation_space.shape}")

    dummy_env = DummyVecEnv([lambda: TradingEnv(holdout_df, initial_balance=args.balance, scaler=scaler, fixed_buy_pct=args.buy_pct, is_eval=True,
                                                sharpe_weight=sharpe_weight, sortino_weight=sortino_weight,
                                                unrealized_penalty=unrealized_penalty, dd_penalty_weight=dd_penalty_weight,
                                                activity_incentive=activity_incentive, hold_penalty=hold_penalty,
                                                unrealized_threshold=unrealized_threshold, dd_threshold=dd_threshold)])

    if os.path.exists(args.model):
        try:
            model = PPO.load(args.model, env=dummy_env)
            print(f"Loaded {args.model} for evaluation.")
        except Exception as e:
            print(f"Load error: {e}")
            exit()
    else:
        print(f"No model found at {args.model}.")
        exit()

    obs, _ = eval_env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = eval_env.step(action)
    
    if terminated:
        print("Eval terminated early due to max DD threshold.")

    # Force sell if positioned
    if eval_env.position > 0:
        row = holdout_df.iloc[-1]
        sell_value = eval_env.position * row['close']
        profit = sell_value - (eval_env.position * eval_env.entry_price) if eval_env.entry_price else 0
        eval_env.balance += sell_value * (1 - TRADE_FEE)
        net_worth = eval_env.balance
        trade_dd = max(0, 1 - min(eval_env.current_trade_net_worths) / eval_env.current_trade_max) if eval_env.current_trade_net_worths else 0
        eval_env.trade_drawdowns.append(trade_dd)
        eval_env.trade_log.append([eval_env.current_step, 'Sell (force)', row['close'], eval_env.position, profit, net_worth, trade_dd * 100])
        eval_env.position = 0
        eval_env.trade_count += 1
        if profit > 0:
            eval_env.wins += 1
        else:
            eval_env.losses += 1
        eval_env.net_worth_history.append(net_worth)
        eval_env.max_net_worth = max(eval_env.max_net_worth, net_worth)

    # Calculate MDD
    eval_max_dd, peak_index, trough_index = calculate_mdd(eval_env.net_worth_history)
    profit_pct = ((eval_env.balance - args.balance) / args.balance) * 100
    total_closed = eval_env.wins + eval_env.losses
    win_rate = eval_env.wins / total_closed * 100 if total_closed > 0 else 0
    max_trade_dd = max(eval_env.trade_drawdowns) * 100 if eval_env.trade_drawdowns else 0

    # Write trade log to CSV
    with open('trade_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for log in eval_env.trade_log:
            writer.writerow(log)

    # Print net worth range and MDD details
    min_net_worth = min(eval_env.net_worth_history)
    max_net_worth = max(eval_env.net_worth_history)
    print(f"Net worth range: Min {min_net_worth:.2f}, Max {max_net_worth:.2f}")
    print(f"Largest drawdown from step {peak_index} (peak {eval_env.net_worth_history[peak_index]:.2f}) to step {trough_index} (trough {eval_env.net_worth_history[trough_index]:.2f})")

    print(f"Eval: Balance {eval_env.balance:.2f} ({profit_pct:.2f}% profit), Portfolio DD {eval_max_dd*100:.2f}%, Max Trade DD {max_trade_dd:.2f}%, Trades {eval_env.trade_count} (closed: {total_closed}), Win Rate {win_rate:.2f}%")
    print(f"Per-trade drawdowns (%): {[round(dd*100, 2) for dd in eval_env.trade_drawdowns if dd > 0]}")

    # Buy-and-hold benchmark
    initial_price = holdout_df.iloc[0]['close']
    final_price = holdout_df.iloc[-1]['close']
    bh_profit_pct = ((final_price - initial_price) / initial_price) * 100
    print(f"Buy-and-hold benchmark profit: {bh_profit_pct:.2f}%")
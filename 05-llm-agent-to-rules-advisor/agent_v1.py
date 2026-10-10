import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import pandas_ta as ta
import requests
import feedparser
import base64
import smtplib
from email.mime.text import MIMEText
from solders.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client as SolanaClient
from binance.client import Client as BinanceClient
from spl.token.instructions import get_associated_token_address
import atexit

load_dotenv()

# ====================== CONFIG ======================
LIVE_TRADING = True
CHECK_INTERVAL = 900
MODEL = "grok-4-1-fast-reasoning"
MEMORY_FILE = "agent_memory.json"
POSITION_STATE_FILE = "live_position.json"
LOCK_FILE = "agent.lock"

JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"
HEADERS = {"x-api-key": os.getenv("JUPITER_API_KEY")}
SOL_RESERVE = 0.05
SLIPPAGE_BPS = 50

SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
USDT_MINT = Pubkey.from_string("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")

solana_client = None
keypair = None
wallet_pubkey = None
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
                content = f.read().strip()
                if content:
                    old_pid, old_ts = content.split("|")
                    if time.time() - float(old_ts) > 600:
                        os.remove(LOCK_FILE)
                    else:
                        print(f"❌ Another instance running. Exiting.")
                        exit(1)
        except:
            os.remove(LOCK_FILE)
    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()}|{time.time()}")
    print("✅ Lock acquired — single instance running")

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

atexit.register(release_lock)

# ====================== TESTED SWAPS (your swap.py) ======================
def get_token_balance(mint):
    if mint == SOL_MINT:
        try:
            return solana_client.get_balance(wallet_pubkey).value / 1_000_000_000
        except:
            return 0.0
    else:
        ata = get_associated_token_address(wallet_pubkey, mint)
        try:
            resp = solana_client.get_token_account_balance(ata)
            return int(resp.value.amount) / (10 ** resp.value.decimals)
        except:
            return 0.0

def get_jupiter_quote(input_mint, output_mint, amount):
    params = {"inputMint": str(input_mint), "outputMint": str(output_mint),
              "amount": str(int(amount)), "slippageBps": str(SLIPPAGE_BPS)}
    r = requests.get(JUPITER_QUOTE_URL, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()

def get_swap_transaction(quote):
    payload = {
        "quoteResponse": quote,
        "userPublicKey": str(wallet_pubkey),
        "wrapAndUnwrapSol": True,
        "asLegacyTransaction": True,
        "dynamicComputeUnitLimit": False,
        "computeUnitLimit": 60000,
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "priorityLevel": "veryHigh",
                "maxLamports": 5000000
            }
        }
    }
    r = requests.post(JUPITER_SWAP_URL, json=payload, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()["swapTransaction"]

def execute_swap(swap_tx_b64):
    for attempt in range(3):
        try:
            swap_tx_bytes = base64.b64decode(swap_tx_b64)
            tx = Transaction.from_bytes(swap_tx_bytes)
            blockhash_resp = solana_client.get_latest_blockhash()
            tx.sign([keypair], blockhash_resp.value.blockhash)
            sig = solana_client.send_raw_transaction(bytes(tx))
            print(f"✅ TX SENT (attempt {attempt+1}) → https://solscan.io/tx/{sig}")
            return sig
        except Exception as e:
            print(f"Swap attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return None

def robust_buy(amount_usdt):
    if amount_usdt <= 10: return None
    for _ in range(3):
        try:
            quote = get_jupiter_quote(USDT_MINT, SOL_MINT, int(amount_usdt * 1_000_000))
            tx_b64 = get_swap_transaction(quote)
            return execute_swap(tx_b64)
        except Exception as e:
            print(f"Buy attempt failed: {e}")
            time.sleep(5)
    return None

def robust_sell(amount_sol):
    if amount_sol <= SOL_RESERVE: return None
    for _ in range(3):
        try:
            quote = get_jupiter_quote(SOL_MINT, USDT_MINT, int((amount_sol - SOL_RESERVE) * 1_000_000_000))
            tx_b64 = get_swap_transaction(quote)
            return execute_swap(tx_b64)
        except Exception as e:
            print(f"Sell attempt failed: {e}")
            time.sleep(5)
    return None

def send_alert(subject, body):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASS]): return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_FROM, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print("✅ Gmail alert sent")
    except: pass

# ====================== STRATEGY ======================
def fetch_data(days=730):
    end = int(time.time() * 1000)
    start = end - (days * 86400 * 1000)
    klines4 = binance_client.get_historical_klines('SOLUSDT', '4h', start_str=start, end_str=end)
    df4 = pd.DataFrame(klines4)[[0,1,2,3,4,5]].astype(float)
    df4.columns = ['timestamp','open','high','low','close','volume']
    df4['timestamp'] = pd.to_datetime(df4['timestamp'], unit='ms', utc=True)
    df4.set_index('timestamp', inplace=True)
    
    klines1d = binance_client.get_historical_klines('SOLUSDT', '1d', start_str=start, end_str=end)
    df1d = pd.DataFrame(klines1d)[[0,1,2,3,4,5]].astype(float)
    df1d.columns = ['timestamp','open','high','low','close','volume']
    df1d['timestamp'] = pd.to_datetime(df1d['timestamp'], unit='ms', utc=True)
    df1d.set_index('timestamp', inplace=True)
    return df4, df1d

def compute_4h(df):
    df = df.copy()
    df['ema_trend'] = ta.ema(df['close'], length=35)
    df['ema_value'] = ta.ema(df['close'], length=15)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    return df

def compute_daily(df):
    df = df.copy()
    df['ema_50'] = ta.ema(df['close'], length=21)
    return df

def attach_daily(df4, df1d):
    df1d = df1d.copy()
    df4 = df4.copy()
    df1d['daily_bull'] = (df1d['close'] > df1d['ema_50']).astype(int)
    daily_bull = df1d['daily_bull'].reindex(df4.index, method='ffill')
    df4['daily_bull'] = daily_bull
    return df4

def buy_signal(df, i):
    if i < 1: return False
    c = df.iloc[i]
    p1 = df.iloc[i-1]
    recent_low = df.iloc[max(0, i-8):i+1]['low'].min()
    in_value_zone = recent_low <= c['ema_value'] * 1.02
    return all([
        c['daily_bull'] == 1,
        c['close'] > c['ema_trend'],
        in_value_zone,
        c['low'] > p1['low'],
        c['close'] > p1['high'],
        c['close'] > c['open'],
        53 < c['rsi'] < 65,
        c['volume'] >= c['vol_ma'] * 1.1,
        c['close'] > c['ema_value']
    ])

def is_choppy(df):
    latest = df.iloc[-1]
    return latest['adx'] < 25 or (latest['ema_value']*0.98 < latest['close'] < latest['ema_trend']*1.02)

# ====================== INTELLIGENCE ======================
def get_latest_news():
    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        return feed.entries[0].title if feed.entries else "No major news"
    except:
        return "No major news"

def get_onchain_metrics():
    try:
        t = binance_client.get_ticker(symbol='SOLUSDT')
        return f"SOL 24h Vol: ${float(t['quoteVolume']):,.0f} | Price ${float(t['lastPrice']):.2f}"
    except:
        return "On-chain neutral"

def get_multi_tf_signals(df4):
    try:
        df1h = pd.DataFrame(binance_client.get_historical_klines('SOLUSDT', '1h', limit=200))
        df1h = df1h[[0,1,2,3,4,5]].astype(float)
        df1h.columns = ['ts','o','h','l','c','v']
        rsi_1h = ta.rsi(df1h['c'],14).iloc[-1]
        dfw = pd.DataFrame(binance_client.get_historical_klines('SOLUSDT', '1w', limit=104))
        dfw = dfw[[0,1,2,3,4,5]].astype(float)
        dfw.columns = ['ts','o','h','l','c','v']
        ema_w = ta.ema(dfw['c'],21)
        weekly_up = dfw['c'].iloc[-1] > ema_w.iloc[-1]
        return f"1H RSI:{rsi_1h:.1f} | 4H signal:{'PASS' if buy_signal(df4, len(df4)-1) else 'FAIL'} | Daily bull:{'ON' if df4.iloc[-1].get('daily_bull') else 'OFF'} | Weekly:{'UP' if weekly_up else 'DOWN'}"
    except:
        return "Multi-TF neutral"

def critic_review(decision, extra):
    prompt = f"Critic — capital safety first. Decision: {decision} Extra: {extra}. Reply ONLY SAFE or RISKY + 1 short sentence."
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.0)
        return resp.choices[0].message.content.strip()
    except:
        return "SAFE - critic unavailable"

# ====================== STATE ======================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f: return json.load(f)
    return {"trades": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f: json.dump(mem, f, indent=2)

def load_position_state():
    if os.path.exists(POSITION_STATE_FILE):
        try:
            with open(POSITION_STATE_FILE) as f:
                return json.load(f)
        except:
            os.remove(POSITION_STATE_FILE)
    return {"in_position": False, "entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0,
            "be_trigger": 0.0, "at_be": False, "trail_stop": 0.0, "risk": 0.0, "entry_bar": 0}

def save_position_state(state):
    with open(POSITION_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ====================== TRUE AGENT v2.6 ======================
def run_agent():
    global solana_client, keypair, wallet_pubkey, binance_client
    acquire_lock()
    solana_client = SolanaClient(os.getenv("RPC_URL"))
    keypair = Keypair.from_base58_string(os.getenv("PRIVATE_KEY"))
    wallet_pubkey = keypair.pubkey()
    binance_client = BinanceClient()

    mem = load_memory()
    state = load_position_state()
    print("🚀 SOLPY_Agent v2.6 — TRUE AI AGENT MODE (Grok has real trading knowledge + override power)")

    while True:
        try:
            df4, df1d = fetch_data()
            df1d = compute_daily(df1d)
            df4 = compute_4h(df4)
            df4 = attach_daily(df4, df1d)
            c = df4.iloc[-1]
            i = len(df4) - 1

            usdt = get_token_balance(USDT_MINT)
            sol = get_token_balance(SOL_MINT)
            ticker = binance_client.get_symbol_ticker(symbol='SOLUSDT')
            current_price = float(ticker['price'])

            now_str = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M')
            position_status = "IN POSITION" if state["in_position"] else "CASH"
            print(f"[{now_str}] WALLET — USDT: ${usdt:,.2f} | SOL: {sol:.6f} | {position_status}")

            mechanical_pass = buy_signal(df4, i)
            chop = is_choppy(df4)
            news = get_latest_news()
            print(f"📢 Latest CoinDesk headline: {news}")
            onchain = get_onchain_metrics()
            multi_tf = get_multi_tf_signals(df4)

            force_exit = False
            exit_reason = None
            if state["in_position"]:
                if not state["at_be"] and current_price >= state["be_trigger"]:
                    state["stop_price"] = state["entry_price"]
                    state["at_be"] = True
                if current_price > state["entry_price"] + (2 * state["risk"]):
                    recent_high = df4.iloc[max(0, i-2):i+1]['high'].max()
                    new_trail = recent_high - (2 * c['atr'])
                    state["trail_stop"] = max(state["trail_stop"], new_trail)
                effective_stop = max(state["stop_price"], state["trail_stop"])
                if current_price <= effective_stop:
                    force_exit = True
                    exit_reason = "MECHANICAL STOP"
                elif current_price >= state["target_price"]:
                    force_exit = True
                    exit_reason = "MECHANICAL TARGET"

            position_info = (f"In position | Entry ${state['entry_price']:.4f} | PnL {((current_price-state['entry_price'])/state['entry_price']*100):+.1f}% | Stop ${max(state['stop_price'],state['trail_stop']):.4f}"
                            if state["in_position"] else "CASH")

            prompt = f"""You are Jamie's autonomous SOL trading agent — an experienced trader with deep knowledge of the crypto market, SOL ecosystem, recent events, catalysts, and market sentiment.

Current price: ${current_price:.2f}
Mechanical signal: {'PASS' if mechanical_pass else 'FAIL'}
Chop: {'YES' if chop else 'NO'}
Multi-TF: {multi_tf}
News: {news}
On-chain: {onchain}
Wallet: USDT ${usdt:.2f} | SOL {sol:.6f}
Position: {position_info}

You have full decision power. Use your built-in knowledge of the crypto market + the live data above to act with real conviction. You may override the mechanical rules and BUY even if the signal is FAIL or chop is YES when the overall picture feels right. You may also SELL early if you see danger. Protect the bag first, but trade like a real experienced trader.

Output ONLY JSON: {{"thought":"...", "action":"BUY|SELL|STAY", "reason":"..."}}"""

            response = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.2)
            decision = json.loads(response.choices[0].message.content)

            critic = critic_review(decision, f"News:{news} | On-chain:{onchain} | Multi-TF:{multi_tf}")

            print(f"\n🤖 THOUGHT: {decision['thought']}")
            print(f"   CRITIC: {critic}")
            print(f"   ACTION: {decision['action']} | Reason: {decision['reason']}")

            if LIVE_TRADING:
                if force_exit and state["in_position"]:
                    sig = robust_sell(sol)
                    if sig:
                        exit_price = current_price
                        pnl_pct = ((exit_price - state["entry_price"]) / state["entry_price"]) * 100
                        send_alert("SOLPY_Agent FORCED EXIT", f"{exit_reason} @ {exit_price:.4f} | PnL {pnl_pct:+.1f}%")
                        mem["trades"].append({"exit_price": exit_price, "pnl_pct": pnl_pct})
                        save_memory(mem)
                        state = load_position_state()
                        if os.path.exists(POSITION_STATE_FILE): os.remove(POSITION_STATE_FILE)

                elif decision["action"] == "BUY" and not state["in_position"] and usdt > 10:
                    entry_price = current_price
                    atr = c['atr']
                    swing_low = df4.iloc[max(0, i-2):i+1]['low'].min()
                    stop_price = swing_low - (0.5 * atr)
                    risk = entry_price - stop_price
                    if risk > 0 and (risk / entry_price) <= 0.1:
                        target_price = entry_price + (2.5 * risk)
                        be_trigger = entry_price + risk
                        trail_stop = stop_price

                        sig = robust_buy(usdt)
                        if sig:
                            state = {"in_position": True, "entry_price": entry_price, "stop_price": stop_price,
                                     "target_price": target_price, "be_trigger": be_trigger, "at_be": False,
                                     "trail_stop": trail_stop, "risk": risk, "entry_bar": i}
                            save_position_state(state)
                            send_alert("SOLPY_Agent BUY", f"BUY @ {entry_price:.4f}\n{decision['reason']}")
                            mem["trades"].append({"entry_price": entry_price})
                            save_memory(mem)

                elif decision["action"] == "SELL" and state["in_position"]:
                    sig = robust_sell(sol)
                    if sig:
                        exit_price = current_price
                        pnl_pct = ((exit_price - state["entry_price"]) / state["entry_price"]) * 100
                        send_alert("SOLPY_Agent EXIT", f"EXIT @ {exit_price:.4f} | PnL {pnl_pct:+.1f}%")
                        mem["trades"].append({"exit_price": exit_price, "pnl_pct": pnl_pct})
                        save_memory(mem)
                        if os.path.exists(POSITION_STATE_FILE): os.remove(POSITION_STATE_FILE)
                        state = load_position_state()

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"Loop error (Starlink-safe): {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_agent()
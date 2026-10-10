"""
trade_recorder.py — Solana Wallet Trade Recorder
--------------------------------------------------
Reads transaction history from a Solana wallet and records
all SOL/USDT swaps to a clean CSV file for tax and record keeping.

Includes USD/CAD exchange rate from Bank of Canada API for CRA reporting.
Run it anytime to update your records — it only adds new trades,
never duplicates old ones.

Setup:
  1. Fill in the CONFIGURATION block below
  2. Run: python trade_recorder_agent5.py
  3. Open the OUTPUT_CSV file to see your trade history

Requirements:
  pip install requests python-dotenv
"""

import os
import csv
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ====================== CONFIGURATION ======================
# Bot identity — shows in console output
BOT_NAME       = "Agent5"

# Wallet to scan — paste the public address directly here
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")  # redacted

# RPC endpoint — falls back to .env if left blank
RPC_URL        = os.getenv("RPC_URL", "")

# Output files — keep these unique per bot so records don't mix
OUTPUT_CSV     = "trades_agent5.csv"
LAST_SEEN_FILE = "last_seen_agent5.txt"
# ===========================================================

# Token mints
SOL_MINT      = "So11111111111111111111111111111111111111112"
USDT_MINT     = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
USDT_DECIMALS = 6
SOL_DECIMALS  = 9

# CSV column headers
CSV_HEADERS = [
    "Date",
    "Time (UTC)",
    "Type",
    "SOL Amount",
    "USDT Amount",
    "Price (USDT/SOL)",
    "USDT Balance After",
    "SOL Balance After",
    "PnL (USDT)",
    "USD/CAD Rate",
    "PnL (CAD)",
    "Total PnL (CAD)", 
    "Trade Result",
    "Transaction ID",
]


# ====================== HELPERS ======================
def load_last_seen():
    """Load the last processed transaction signature."""
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r") as f:
            sig = f.read().strip()
            return sig if sig else None
    return None


def save_last_seen(sig):
    """Save the most recently processed transaction signature."""
    with open(LAST_SEEN_FILE, "w") as f:
        f.write(sig)


def load_existing_trades():
    """Load existing trades from CSV to avoid duplicates."""
    existing_sigs = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_sigs.add(row.get("Transaction ID", ""))
    return existing_sigs


def get_usdcad_rate(date_str):
    """
    Fetch USD/CAD rate from Bank of Canada for a given date (YYYY-MM-DD).
    Falls back up to 7 days prior to handle weekends and holidays.
    Returns (rate, actual_date) or (None, None) if unavailable.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    for days_back in range(7):
        check_date = (date - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = (
            f"https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json"
            f"?start_date={check_date}&end_date={check_date}"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            observations = r.json().get("observations", [])
            if observations:
                rate = float(observations[0]["FXUSDCAD"]["v"])
                if days_back > 0:
                    print(f"  ℹ️  No BoC rate for {date_str} — using {check_date} ({days_back}d back)")
                return rate, check_date
        except Exception as e:
            print(f"  ⚠️  BoC rate fetch failed for {check_date}: {e}")
    return None, None


def get_transactions(wallet, limit=50, before=None):
    """Fetch transaction signatures for a wallet from Helius RPC."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [
            wallet,
            {
                "limit": limit,
                **({"before": before} if before else {})
            }
        ]
    }
    try:
        r = requests.post(RPC_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        print(f"❌ Failed to fetch transactions: {e}")
        return []


def get_transaction_detail(sig):
    """Fetch full transaction details for a signature."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            sig,
            {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0,
                "commitment": "finalized"
            }
        ]
    }
    try:
        r = requests.post(RPC_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("result")
    except Exception as e:
        print(f"  ⚠️ Failed to fetch tx detail for {sig[:20]}...: {e}")
        return None


def parse_token_balances(tx, wallet):
    """
    Extract pre/post SOL and USDT balances from a transaction.
    Returns (pre_sol, post_sol, pre_usdt, post_usdt, sol_change, usdt_change)
    or None if not a swap.
    """
    try:
        meta = tx.get("meta", {})
        if meta.get("err"):
            return None  # Skip failed transactions

        # --- SOL balance changes ---
        account_keys = []
        msg = tx.get("transaction", {}).get("message", {})
        for acc in msg.get("accountKeys", []):
            if isinstance(acc, dict):
                account_keys.append(acc.get("pubkey", ""))
            else:
                account_keys.append(str(acc))

        wallet_idx = None
        for idx, key in enumerate(account_keys):
            if key == wallet:
                wallet_idx = idx
                break

        if wallet_idx is None:
            return None

        pre_balances  = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        pre_sol  = pre_balances[wallet_idx]  / 10**SOL_DECIMALS if wallet_idx < len(pre_balances) else 0
        post_sol = post_balances[wallet_idx] / 10**SOL_DECIMALS if wallet_idx < len(post_balances) else 0

        # --- USDT token balance changes ---
        pre_usdt  = 0.0
        post_usdt = 0.0
        pre_token_balances  = meta.get("preTokenBalances", [])
        post_token_balances = meta.get("postTokenBalances", [])

        for tb in pre_token_balances:
            if tb.get("mint") == USDT_MINT and tb.get("owner") == wallet:
                pre_usdt = float(tb.get("uiTokenAmount", {}).get("uiAmount") or 0)

        for tb in post_token_balances:
            if tb.get("mint") == USDT_MINT and tb.get("owner") == wallet:
                post_usdt = float(tb.get("uiTokenAmount", {}).get("uiAmount") or 0)

        sol_change  = post_sol  - pre_sol
        usdt_change = post_usdt - pre_usdt

        # Only care about actual swaps (both SOL and USDT changed meaningfully)
        if abs(sol_change) < 0.05 or abs(usdt_change) < 0.5:
            return None

        return pre_sol, post_sol, pre_usdt, post_usdt, sol_change, usdt_change

    except Exception as e:
        print(f"  ⚠️ Balance parse error: {e}")
        return None


def format_timestamp(unix_ts):
    """Convert unix timestamp to readable date and time strings."""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


# ====================== MAIN ======================
def run_recorder():
    print("\n" + "="*60)
    print(f"  SOLANA TRADE RECORDER — {BOT_NAME}")
    print("="*60)
    print(f"  Wallet : {WALLET_ADDRESS[:20]}...{WALLET_ADDRESS[-6:]}")
    print(f"  Output : {OUTPUT_CSV}")
    print("="*60 + "\n")

    if "YOUR_" in WALLET_ADDRESS:
        print("❌ Please set WALLET_ADDRESS in the CONFIGURATION block at the top of this script.")
        return

    if not RPC_URL:
        print("❌ Please set RPC_URL in your .env file.")
        return

    # Load existing data
    last_seen     = load_last_seen()
    existing_sigs = load_existing_trades()

    print(f"📂 Existing trades in CSV : {len(existing_sigs)}")
    print(f"🔖 Last seen signature    : {last_seen[:20] + '...' if last_seen else 'None (first run)'}\n")

        # === NEW: Load running total PnL (CAD) from existing trades ===
    running_pnl_cad = 0.0
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pnl_str = row.get("PnL (CAD)", "").strip()
                if pnl_str and pnl_str not in ("", "N/A", "—"):
                    try:
                        # remove the + sign and convert to float
                        running_pnl_cad += float(pnl_str.replace('+', ''))
                    except ValueError:
                        pass
    print(f"📊 Current running PnL (CAD) from CSV: ${running_pnl_cad:+.2f}\n")

    # Fetch recent transaction signatures
    print("🔍 Fetching transaction history...")
    sigs = get_transactions(WALLET_ADDRESS, limit=100)

    if not sigs:
        print("No transactions found or RPC error.")
        return

    print(f"   Found {len(sigs)} recent transactions\n")

    # Filter to only new transactions
    new_sigs = []
    for s in sigs:
        sig = s.get("signature", "")
        if sig == last_seen:
            break  # Stop when we hit the last processed one
        if sig not in existing_sigs:
            new_sigs.append(s)

    if not new_sigs:
        print("✅ No new transactions found — records are up to date!")
        return

    print(f"📋 Processing {len(new_sigs)} new transactions...\n")

    # Cache BoC rates by date to avoid duplicate API calls
    boc_rate_cache = {}

    # Process each new transaction
    new_trades     = []
    last_buy_price = None
    last_buy_usdt  = None

    for s in reversed(new_sigs):  # Process oldest first
        sig       = s.get("signature", "")
        unix_time = s.get("blockTime", 0)

        print(f"  Checking {sig[:20]}...")
        time.sleep(0.3)  # Be gentle with RPC rate limits

        tx = get_transaction_detail(sig)
        if not tx:
            continue

        result = parse_token_balances(tx, WALLET_ADDRESS)
        if not result:
            print(f"  → Not a SOL/USDT swap, skipping")
            continue

        pre_sol, post_sol, pre_usdt, post_usdt, sol_change, usdt_change = result
        date_str, time_str = format_timestamp(unix_time)

        # Fetch USD/CAD rate (cached per date)
        if date_str not in boc_rate_cache:
            rate, _ = get_usdcad_rate(date_str)
            boc_rate_cache[date_str] = rate
        usdcad = boc_rate_cache[date_str]

        # Determine trade type
        if sol_change > 0:
            trade_type = "BUY"   # SOL increased = bought SOL with USDT
            sol_amount  = round(abs(sol_change), 6)
            usdt_amount = round(abs(usdt_change), 2)
            price       = round(usdt_amount / sol_amount, 4) if sol_amount > 0 else 0
            pnl_usdt    = ""
            pnl_cad     = ""
            result_str  = "—"
            last_buy_price = price
            last_buy_usdt  = usdt_amount

        else:
            trade_type  = "SELL"  # SOL decreased = sold SOL for USDT
            sol_amount  = round(abs(sol_change), 6)
            usdt_amount = round(abs(usdt_change), 2)
            price       = round(usdt_amount / sol_amount, 4) if sol_amount > 0 else 0

            # Calculate PnL if we have a matching buy
            if last_buy_price and last_buy_usdt:
                pnl_val    = round(usdt_amount - last_buy_usdt, 2)
                pnl_usdt   = f"{pnl_val:+.2f}"

                if usdcad:
                    pnl_cad_val = round(pnl_val * usdcad, 2)
                    pnl_cad     = f"{pnl_cad_val:+.2f}"
                    running_pnl_cad += pnl_cad_val          # ← UPDATES RUNNING TOTAL
                else:
                    pnl_cad = ""

                result_str = "WIN ✅" if pnl_val > 0 else "LOSS ❌" if pnl_val < 0 else "SCRATCH"
                last_buy_price = None
                last_buy_usdt  = None
            else:
                pnl_usdt   = ""
                pnl_cad    = ""
                result_str = "—"

        trade = {
            "Date"              : date_str,
            "Time (UTC)"        : time_str,
            "Type"              : trade_type,
            "SOL Amount"        : sol_amount,
            "USDT Amount"       : usdt_amount,
            "Price (USDT/SOL)"  : price,
            "USDT Balance After": round(post_usdt, 2),
            "SOL Balance After" : round(post_sol, 6),
            "PnL (USDT)"        : pnl_usdt,
            "USD/CAD Rate"      : round(usdcad, 4) if usdcad else "N/A",
            "PnL (CAD)"         : pnl_cad,
            "Total PnL (CAD)"   : f"{running_pnl_cad:+.2f}",
            "Trade Result"      : result_str,
            "Transaction ID"    : sig,
        }

        new_trades.append(trade)
        cad_str = f" | CAD PnL: ${pnl_cad}" if pnl_cad else ""
        print(f"  → {trade_type} | {sol_amount} SOL @ ${price} | USDT: ${round(post_usdt, 2)} | {result_str}{cad_str}")

    if not new_trades:
        print("\n✅ No SOL/USDT swaps found in new transactions.")
        return

    # Write to CSV
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_trades)

    # Save last seen signature (most recent = first in list)
    save_last_seen(new_sigs[0].get("signature", ""))

    # Summary
    print("\n" + "="*60)
    print(f"  ✅ {len(new_trades)} new trades recorded to {OUTPUT_CSV}")
    wins   = sum(1 for t in new_trades if "WIN" in t["Trade Result"])
    losses = sum(1 for t in new_trades if "LOSS" in t["Trade Result"])
    pnls_usdt = [float(t["PnL (USDT)"]) for t in new_trades if t["PnL (USDT)"] not in ("", "—")]
    pnls_cad  = [float(t["PnL (CAD)"])  for t in new_trades if t["PnL (CAD)"]  not in ("", "—")]
    if pnls_usdt:
        print(f"  📊 This run  : {wins} wins | {losses} losses | PnL (USDT): ${sum(pnls_usdt):+.2f}", end="")
        if pnls_cad:
            print(f" | PnL (CAD): ${sum(pnls_cad):+.2f}", end="")
        print()
    print("="*60 + "\n")


if __name__ == "__main__":
    run_recorder()

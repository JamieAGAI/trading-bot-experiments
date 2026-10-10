"""
jup_perp_trades.py — Jupiter Perps Trade Recorder (FIXED v2)
---------------------------------------------------
Official CSV export + **automatic column cleaning** (leading spaces fixed)
Full debug output so we can see exactly what's happening.
"""

import os
import csv
import time
import requests
import io
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ====================== CONFIGURATION ======================
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")  # redacted
OUTPUT_CSV     = "perptrades.csv"
LAST_SEEN_FILE = "last_seen_jup_perps.txt"
# ===========================================================

CSV_HEADERS = [
    "Date", "Time (UTC)", "Market", "Side", "Action", "Size (USD)",
    "Collateral (USD)", "Entry Price", "Exit Price", "PnL (USD)",
    "Fees (USD)", "USD/CAD Rate", "PnL (CAD)", "Total PnL (CAD)",
    "Trade Result", "Transaction ID",
]

# ====================== HELPERS ======================
def load_last_seen():
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE) as f:
            sig = f.read().strip()
            return sig if sig else None
    return None

def save_last_seen(sig):
    with open(LAST_SEEN_FILE, "w") as f:
        f.write(sig)

def load_existing_sigs():
    existing = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row.get("Transaction ID", ""))
    return existing

def load_running_pnl_cad():
    running = 0.0
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("PnL (CAD)", "").strip()
                if val and val not in ("", "N/A", "—"):
                    try:
                        running += float(val.replace("+", ""))
                    except ValueError:
                        pass
    return running

def get_usdcad_rate(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    for days_back in range(7):
        check_date = (date - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date={check_date}&end_date={check_date}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                rate = float(obs[0]["FXUSDCAD"]["v"])
                if days_back > 0:
                    print(f"  ℹ️  No BoC rate for {date_str} — using {check_date} ({days_back}d back)")
                return rate, check_date
        except Exception as e:
            print(f"  ⚠️  BoC rate fetch failed for {check_date}: {e}")
    return None, None

def parse_created_at(dt_str):
    """Robust parser for Jupiter's timestamp"""
    if not dt_str:
        return "unknown", "unknown"
    try:
        clean = dt_str.split(" (")[0].strip()          # remove (Coordinated Universal Time)
        clean = clean.split(" GMT")[0].strip() + " GMT+0000"  # ensure timezone format
        dt = datetime.strptime(clean, "%a %b %d %Y %H:%M:%S %Z%z")
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except Exception as e:
        print(f"  ⚠️  Date parse failed: {dt_str[:80]}... | Error: {e}")
        return "parse_error", "parse_error"

# ====================== FETCH + CLEAN ======================
def fetch_perps_csv(wallet):
    url = f"https://perps-data.jup.ag/trades/export/{wallet}"
    print(f"🔍 Fetching official Perps CSV from {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    
    csv_reader = csv.DictReader(io.StringIO(r.text))
    
    # === CRITICAL FIX: Remove leading spaces from ALL column names ===
    raw_rows = []
    for orig_row in csv_reader:
        clean_row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in orig_row.items()}
        raw_rows.append(clean_row)
    
    print(f"  ✅ Fetched {len(raw_rows)} trades")
    if raw_rows:
        print("  CSV Columns (CLEANED):", list(raw_rows[0].keys()))
        print("  Sample row (first trade - CLEANED):")
        for k, v in list(raw_rows[0].items())[:12]:   # first 12 fields
            print(f"    {k}: {v}")
    return raw_rows

# ====================== MAIN ======================
def run_recorder():
    print("\n" + "=" * 80)
    print("  JUPITER PERPS TRADE RECORDER (FIXED v2 - Column spaces resolved)")
    print("=" * 80)

    wallet = WALLET_ADDRESS.strip()
    if not wallet:
        print("❌ WALLET_ADDRESS not set.")
        return

    print(f"  Wallet : {wallet[:20]}...{wallet[-6:]}")
    print(f"  Output : {OUTPUT_CSV}")
    print("=" * 80 + "\n")

    existing_sigs = load_existing_sigs()
    last_seen = load_last_seen()
    running_pnl_cad = load_running_pnl_cad()

    print(f"📂 Existing trades in CSV : {len(existing_sigs)}")
    print(f"🔖 Last seen tx           : {last_seen[:20] + '...' if last_seen else 'None (first run)'}")
    print(f"📊 Running PnL (CAD)      : ${running_pnl_cad:+.2f}\n")

    raw_rows = fetch_perps_csv(wallet)
    if not raw_rows:
        print("  No trades returned.")
        return

    # Filter new trades using CLEANED column names
    new_raw = []
    for row in raw_rows:
        tx_id = str(row.get("Transaction ID", "")).strip()
        if tx_id and tx_id not in existing_sigs:
            new_raw.append(row)

    print(f"\n📋 Found {len(new_raw)} NEW trades to process\n")

    if not new_raw:
        print("✅ No new trades — records are up to date!")
        return

    boc_cache = {}
    new_rows = []

    for raw in reversed(new_raw):   # oldest first so running total is correct
        try:
            tx_id = str(raw.get("Transaction ID", "")).strip()
            if not tx_id:
                print("  ⚠️  Skipping row with no Transaction ID")
                continue

            date_str, time_str = parse_created_at(raw.get("Created at", ""))
            market = f"{raw.get('Asset', 'UNKNOWN')}-PERP"
            side = raw.get("Position", "").upper()
            action = raw.get("Position change", "").upper()
            size_usd = float(raw.get("Trade size ($)", 0) or 0)
            pnl_usd = float(raw.get("Profit / Loss ($)", 0) or 0)
            fees_usd = float(raw.get("Trade fee ($)", 0) or 0) + float(raw.get("Liquidation fee ($)", 0) or 0)
            exec_price = float(raw.get("Execution price ($)", 0) or 0)

            print(f"  → Processing: {action} | {side} | Size ${size_usd:.0f} | PnL ${pnl_usd:+.2f} | Tx: {tx_id[:16]}...")

            # USD/CAD
            if date_str not in boc_cache and date_str != "unknown":
                rate, _ = get_usdcad_rate(date_str)
                boc_cache[date_str] = rate
            usdcad = boc_cache.get(date_str)

            is_closing = action in ("DECREASE", "CLOSE", "LIQUIDATION")
            pnl_usd_str = f"{pnl_usd:+.2f}" if is_closing and pnl_usd != 0 else ""
            pnl_cad_str = ""
            result_str = "—"

            if is_closing and pnl_usd != 0:
                if usdcad:
                    pnl_cad_val = round(pnl_usd * usdcad, 2)
                    pnl_cad_str = f"{pnl_cad_val:+.2f}"
                    running_pnl_cad += pnl_cad_val
                result_str = "WIN ✅" if pnl_usd > 0 else "LOSS ❌" if pnl_usd < 0 else "SCRATCH"

            row_dict = {
                "Date": date_str,
                "Time (UTC)": time_str,
                "Market": market,
                "Side": side,
                "Action": action,
                "Size (USD)": f"{size_usd:.2f}",
                "Collateral (USD)": raw.get("Deposit / Withdraw ($)", ""),
                "Entry Price": f"{exec_price:.4f}",
                "Exit Price": f"{exec_price:.4f}" if is_closing else "",
                "PnL (USD)": pnl_usd_str,
                "Fees (USD)": f"{fees_usd:.4f}" if fees_usd else "",
                "USD/CAD Rate": round(usdcad, 4) if usdcad else "N/A",
                "PnL (CAD)": pnl_cad_str,
                "Total PnL (CAD)": f"{running_pnl_cad:+.2f}",
                "Trade Result": result_str,
                "Transaction ID": tx_id,
            }

            new_rows.append(row_dict)
            cad_note = f" | CAD: ${pnl_cad_str}" if pnl_cad_str else ""
            print(f"    ✅ ADDED → {action:10s} | {side:5s} | {market} | Size ${size_usd:.0f} | {result_str}{cad_note}")

        except Exception as e:
            print(f"  ⚠️  Error processing row: {e}")
            continue

    # Write to CSV
    if new_rows:
        file_exists = os.path.exists(OUTPUT_CSV)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)

        save_last_seen(new_rows[-1]["Transaction ID"])

        print("\n" + "=" * 80)
        print(f"  ✅ {len(new_rows)} trades successfully recorded to {OUTPUT_CSV} !")
        wins = sum(1 for r in new_rows if "WIN" in r.get("Trade Result", ""))
        losses = sum(1 for r in new_rows if "LOSS" in r.get("Trade Result", ""))
        print(f"  📊 This run: {wins}W / {losses}L | Running PnL (CAD): ${running_pnl_cad:+.2f}")
        print("=" * 80 + "\n")
        print("📁 Check your new file: perptrades.csv")
    else:
        print("\n❌ Still 0 trades added. (This should not happen now)")

if __name__ == "__main__":
    run_recorder()
"""
swap_test.py — Manual swap script for testing tiny real SOL/USDT swaps.

Mirrors main.py's Jupiter v1 swap pipeline 1:1. Used as a safe sandbox
to verify swap behaviour and to test transaction validation logic before
porting it to the live bot.

PHASE 2 (Apr 27, 2026): Transaction validation is now active. Before
signing, the script decodes the Jupiter-returned VersionedTransaction
and verifies it only touches allowlisted Solana programs and matches
the quote we requested. On validation failure, the script aborts before
signing — no key signs an unverified transaction.

Save in the SOLPY folder so it reads the same .env as main.py.

Usage:
    python swap_test.py buy 2.0           # dry-run: 2 USDT -> SOL
    python swap_test.py sell 0.05         # dry-run: 0.05 SOL -> USDT
    python swap_test.py buy 2.0 --confirm # actually execute (validates first)
    python swap_test.py --help

Hard safety caps:
    buy  > 5.0 USDT  -> abort
    sell > 0.1 SOL   -> abort
    Without --confirm, NO signing or sending happens.

Exit codes:
    0  success (or successful dry-run)
    1  invalid args / safety cap exceeded
    2  Jupiter API failure (quote or swap-tx fetch)
    3  send / signing failure
    4  unconfirmed after 150s (check solscan manually)
    5  transaction validation rejected the swap-tx (potential security event)
"""

import argparse
import base64
import json
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv

from solana.rpc.api import Client as SolanaClient
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from spl.token.instructions import get_associated_token_address


# ====================== CONFIG (mirrors main.py exactly) ======================
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"
JUPITER_API_KEY   = os.getenv("JUPITER_API_KEY")

SOL_MINT  = Pubkey.from_string("So11111111111111111111111111111111111111112")
USDT_MINT = Pubkey.from_string("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")

RPC_URL      = os.getenv("RPC_URL")
PRIVATE_KEY  = os.getenv("PRIVATE_KEY")
SLIPPAGE_BPS = 100
SOL_RESERVE  = 0.07

# Hard safety caps — refused regardless of --confirm flag
MAX_BUY_USDT = 5.0
MAX_SELL_SOL = 0.1

if not JUPITER_API_KEY:
    raise ValueError("Add JUPITER_API_KEY=your_key_here to .env")
if not RPC_URL or not PRIVATE_KEY:
    raise ValueError("RPC_URL and PRIVATE_KEY required in .env")

solana_client = SolanaClient(RPC_URL)
keypair       = Keypair.from_base58_string(PRIVATE_KEY)
wallet_pubkey = keypair.pubkey()

HEADERS = {"x-api-key": JUPITER_API_KEY}


# ====================== HELPERS (mirror main.py exactly) ======================
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
        "inputMint":   str(input_mint),
        "outputMint":  str(output_mint),
        "amount":      str(int(amount)),
        "slippageBps": str(SLIPPAGE_BPS),
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


# ====================== TRANSACTION VALIDATION (Phase 2, Apr 27 2026) ======================
# Allowlist of program IDs that may legitimately appear in a Jupiter swap.
# Verified against real $2 buy and 0.02 sell on Apr 27, 2026.
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

    Checks:
      1. Every instruction's program_id is on the allowlist
      2. Our wallet pubkey appears in account_keys
      3. Quote amounts parse as positive integers and slippage is sane
         (otherAmountThreshold <= outAmount and within 2x SLIPPAGE_BPS)
      4. Quote inputMint/outputMint match the requested direction
    """
    try:
        msg = tx.message
        account_keys = list(msg.account_keys)
        instructions = list(msg.instructions)

        # Check 1: every instruction's program must be on the allowlist
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

        # Slippage tolerance reasonable: min_out >= out_amount * (1 - 2*SLIPPAGE_BPS/10000)
        # 2x slack catches "way too loose" but allows normal 1% slippage.
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


# ====================== DECODING & DISPLAY ======================
def display_decoded_tx(tx, swap_tx_b64):
    """Print every detail we can pull out of the VersionedTransaction.
    No validation here yet — pure inspection."""
    msg = tx.message

    logger.info("─" * 70)
    logger.info("DECODED TRANSACTION")
    logger.info("─" * 70)
    logger.info(f"  Raw tx base64 length:  {len(swap_tx_b64)} chars")
    logger.info(f"  Raw tx bytes length:   {len(base64.b64decode(swap_tx_b64))} bytes")

    sigs = list(tx.signatures)
    logger.info(f"  Signatures (slots):    {len(sigs)}")

    account_keys = list(msg.account_keys)
    logger.info(f"  Account keys count:    {len(account_keys)}")
    logger.info("  Account keys:")
    for i, key in enumerate(account_keys):
        marker = ""
        if str(key) == str(wallet_pubkey):
            marker = "  <-- OUR WALLET"
        logger.info(f"    [{i:>2}]  {str(key)}{marker}")

    instructions = list(msg.instructions)
    logger.info(f"  Instructions count:    {len(instructions)}")
    logger.info("  Instructions:")
    for i, ix in enumerate(instructions):
        program_idx = ix.program_id_index
        program_key = account_keys[program_idx] if program_idx < len(account_keys) else "?"
        account_indices = list(ix.accounts)
        data_len = len(bytes(ix.data))
        logger.info(f"    [{i:>2}] program_idx={program_idx}  program={str(program_key)}")
        logger.info(f"         accounts (idx)={account_indices}  data_len={data_len} bytes")

    logger.info("─" * 70)


def display_quote(quote, direction):
    logger.info("─" * 70)
    logger.info("JUPITER QUOTE")
    logger.info("─" * 70)
    in_amount  = int(quote.get("inAmount", 0))
    out_amount = int(quote.get("outAmount", 0))
    other_min  = int(quote.get("otherAmountThreshold", 0))

    if direction == "buy":
        in_human  = in_amount  / 1_000_000          # USDT, 6 decimals
        out_human = out_amount / 1_000_000_000      # SOL, 9 decimals
        min_human = other_min  / 1_000_000_000
        logger.info(f"  Direction:           buy (USDT -> SOL)")
        logger.info(f"  inAmount (USDT):     {in_human:.6f} USDT  ({in_amount} lamports)")
        logger.info(f"  outAmount (SOL):     {out_human:.9f} SOL   ({out_amount} lamports)")
        logger.info(f"  Min out (slippage):  {min_human:.9f} SOL   ({other_min} lamports)")
        if out_human > 0:
            logger.info(f"  Implied price:       ${in_human/out_human:.4f}/SOL")
    else:
        in_human  = in_amount  / 1_000_000_000      # SOL, 9 decimals
        out_human = out_amount / 1_000_000          # USDT, 6 decimals
        min_human = other_min  / 1_000_000
        logger.info(f"  Direction:           sell (SOL -> USDT)")
        logger.info(f"  inAmount (SOL):      {in_human:.9f} SOL   ({in_amount} lamports)")
        logger.info(f"  outAmount (USDT):    {out_human:.6f} USDT  ({out_amount} lamports)")
        logger.info(f"  Min out (slippage):  {min_human:.6f} USDT  ({other_min} lamports)")
        if in_human > 0:
            logger.info(f"  Implied price:       ${out_human/in_human:.4f}/SOL")

    logger.info(f"  Slippage BPS:        {quote.get('slippageBps')}")
    logger.info(f"  Price impact:        {quote.get('priceImpactPct')}")
    logger.info(f"  Route plan steps:    {len(quote.get('routePlan', []))}")
    logger.info("  Full quote JSON (pretty):")
    pretty = json.dumps(quote, indent=2)
    for line in pretty.splitlines():
        logger.info(f"    {line}")
    logger.info("─" * 70)


# ====================== SEND & CONFIRM (mirrors main.py) ======================
def send_and_confirm(tx, direction, pre_usdt, pre_sol):
    """Send signed tx, poll 60s, then balance fallback 3x at 30s.
    Mirrors main.py execute_swap() lines 437-472."""
    resp = solana_client.send_raw_transaction(
        bytes(tx),
        opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"),
    )
    sig = resp.value
    logger.info(f"✅ TX SENT -> https://solscan.io/tx/{sig}")

    # Poll signature status — 60s window
    for i in range(30):
        time.sleep(2)
        try:
            status_resp = solana_client.get_signature_statuses([sig], search_transaction_history=True)
            status = status_resp.value[0] if status_resp.value else None
            if status and status.confirmation_status in ("confirmed", "finalized"):
                logger.info(f"✅ TX CONFIRMED ({status.confirmation_status}) after {(i+1)*2}s")
                return sig, True
        except Exception:
            pass

    # Balance fallback — 3 retries at 30s each
    logger.warning("⏳ Polling timed out after 60s — verifying via balance (will retry 3x)...")
    for check in range(3):
        time.sleep(30)
        post_usdt = get_token_balance(USDT_MINT)
        post_sol  = get_token_balance(SOL_MINT)
        elapsed = 60 + (check + 1) * 30
        logger.info(f"   Check {check+1}/3 (t+{elapsed}s) — USDT ${post_usdt:.4f} | SOL {post_sol:.6f}")
        if direction == "buy" and post_sol > pre_sol + 0.001:
            logger.info(f"✅ Balance confirms BUY executed after {elapsed}s")
            return sig, True
        if direction == "sell" and (post_usdt > pre_usdt + 0.1 or post_sol <= SOL_RESERVE + 0.001):
            logger.info(f"✅ Balance confirms SELL executed after {elapsed}s")
            return sig, True

    logger.warning("⚠️  TX unconfirmed after 150s total — check solscan link manually")
    return sig, False


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(
        description="Manual swap script for testing tiny real SOL/USDT swaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("direction", choices=["buy", "sell"],
                        help="buy = USDT->SOL, sell = SOL->USDT")
    parser.add_argument("amount", type=float,
                        help="Amount to swap (USDT for buy, SOL for sell)")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually sign and send. Without this, dry-run only.")
    args = parser.parse_args()

    # ── Safety caps ──────────────────────────────────────────────────────
    if args.amount <= 0:
        logger.error("ABORT: amount must be positive")
        sys.exit(1)
    if args.direction == "buy" and args.amount > MAX_BUY_USDT:
        logger.error(f"ABORT: buy capped at {MAX_BUY_USDT} USDT (got {args.amount})")
        sys.exit(1)
    if args.direction == "sell" and args.amount > MAX_SELL_SOL:
        logger.error(f"ABORT: sell capped at {MAX_SELL_SOL} SOL (got {args.amount})")
        sys.exit(1)

    # ── Wallet / config print ────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SWAP TEST")
    logger.info("=" * 70)
    logger.info(f"  Mode:          {'EXECUTE (--confirm)' if args.confirm else 'DRY RUN'}")
    logger.info(f"  Direction:     {args.direction}")
    logger.info(f"  Amount:        {args.amount} {'USDT' if args.direction == 'buy' else 'SOL'}")
    logger.info(f"  Wallet:        {wallet_pubkey}")
    rpc_host = RPC_URL.split("//")[-1].split("/")[0] if "//" in RPC_URL else "?"
    logger.info(f"  RPC host:      {rpc_host}")

    pre_usdt = get_token_balance(USDT_MINT)
    pre_sol  = get_token_balance(SOL_MINT)
    logger.info(f"  Pre USDT:      ${pre_usdt:.4f}")
    logger.info(f"  Pre SOL:       {pre_sol:.6f}")

    # ── Pre-flight balance checks ────────────────────────────────────────
    if args.direction == "buy":
        if pre_usdt < args.amount:
            logger.error(f"ABORT: insufficient USDT (have {pre_usdt:.4f}, need {args.amount})")
            sys.exit(1)
        input_mint  = USDT_MINT
        output_mint = SOL_MINT
        amount_lamports = int(args.amount * 1_000_000)  # USDT 6 decimals
    else:
        # Don't allow drop below SOL_RESERVE
        if pre_sol - args.amount < SOL_RESERVE:
            logger.error(f"ABORT: sell would drop SOL below reserve "
                         f"(have {pre_sol:.6f}, after sell {pre_sol - args.amount:.6f}, "
                         f"reserve {SOL_RESERVE})")
            sys.exit(1)
        input_mint  = SOL_MINT
        output_mint = USDT_MINT
        amount_lamports = int(args.amount * 1_000_000_000)  # SOL 9 decimals

    # ── Get quote ────────────────────────────────────────────────────────
    try:
        quote = get_jupiter_quote(input_mint, output_mint, amount_lamports)
    except Exception as e:
        logger.error(f"Jupiter quote failed: {e}")
        sys.exit(2)

    display_quote(quote, args.direction)

    # ── Get swap transaction ─────────────────────────────────────────────
    try:
        swap_tx_b64 = get_swap_transaction(quote)
    except Exception as e:
        logger.error(f"Jupiter swap-tx fetch failed: {e}")
        sys.exit(2)

    # ── Decode and display ───────────────────────────────────────────────
    try:
        tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
    except Exception as e:
        logger.error(f"Transaction decode failed: {e}")
        sys.exit(2)

    display_decoded_tx(tx, swap_tx_b64)

    # ── VALIDATE TRANSACTION (Phase 2) ───────────────────────────────────
    # Runs whether dry-run or --confirm. Lets us see validation status
    # in dry-run mode and blocks signing if validation fails.
    logger.info("=" * 70)
    logger.info("TRANSACTION VALIDATION")
    logger.info("=" * 70)
    ok, reason = validate_swap_transaction(tx, quote, args.direction)
    if ok:
        logger.info(f"  ✓ VALIDATION PASSED")
        logger.info(f"  ✓ All instructions touch only allowlisted programs")
        logger.info(f"  ✓ Wallet pubkey present in account_keys")
        logger.info(f"  ✓ Quote mints match direction ({args.direction})")
        logger.info(f"  ✓ Slippage within bounds (max {SLIPPAGE_BPS * 2} bps)")
    else:
        logger.error(f"  🛑 VALIDATION REJECTED: {reason}")
        logger.error(f"  🛑 Refusing to sign this transaction.")
        logger.error(f"  🛑 If --confirm was set, no signing took place.")
        logger.error("=" * 70)
        sys.exit(5)
    logger.info("=" * 70)

    # ── Stop here if dry run ─────────────────────────────────────────────
    if not args.confirm:
        logger.info("=" * 70)
        logger.info("DRY RUN COMPLETE — validation passed, no transaction signed or sent")
        logger.info("Re-run with --confirm to actually execute.")
        logger.info("=" * 70)
        sys.exit(0)

    # ── Sign and send ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SIGNING AND SENDING")
    logger.info("=" * 70)
    try:
        signed_tx = VersionedTransaction(tx.message, [keypair])
    except Exception as e:
        logger.error(f"Signing failed: {e}")
        sys.exit(3)

    try:
        sig, confirmed = send_and_confirm(signed_tx, args.direction, pre_usdt, pre_sol)
    except Exception as e:
        logger.error(f"Send failed: {e}")
        sys.exit(3)

    # ── Final state ──────────────────────────────────────────────────────
    post_usdt = get_token_balance(USDT_MINT)
    post_sol  = get_token_balance(SOL_MINT)

    logger.info("=" * 70)
    logger.info("RESULT")
    logger.info("=" * 70)
    logger.info(f"  Confirmed:     {confirmed}")
    logger.info(f"  Signature:     {sig}")
    logger.info(f"  Solscan:       https://solscan.io/tx/{sig}")
    logger.info(f"  Pre  USDT:     ${pre_usdt:.4f}    Post USDT:  ${post_usdt:.4f}    "
                f"Δ {post_usdt - pre_usdt:+.4f}")
    logger.info(f"  Pre  SOL:      {pre_sol:.6f}    Post SOL:   {post_sol:.6f}    "
                f"Δ {post_sol - pre_sol:+.6f}")
    logger.info("=" * 70)

    if not confirmed:
        sys.exit(4)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
analyze_shadow_perps.py
=======================
Read-only stratification analysis of advisor shadow perp trades.

Answers ONE question: when do shadow short trades win vs lose, and which
features distinguish winners from losers?

Reports both RAW win rates (every logged hourly entry counted separately)
and DE-STACKED win rates (entries grouped into logical trades to avoid
inflating wins/losses on stacked dump or ramp days). The de-stacked column
is the primary metric for decision-making.

Win Rate (WR) = % of trades that exited at TP (tp_primary or tp_stretch)
                rather than stop or timeout.

Usage:
    python analyze_shadow_perps.py
    python analyze_shadow_perps.py --input path/to/file.jsonl
    python analyze_shadow_perps.py --group-by exit
    python analyze_shadow_perps.py --export csv

This script is strictly read-only. It reads one JSONL file and writes
nothing back. No API calls. No modifications to any other code.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ============================================================
# CONSTANTS (locked for reproducibility — DO NOT change without
# updating analyses that compare runs over time)
# ============================================================

VOL_LOW_CUTOFF  = 0.7    # vol_ratio_4h < this  -> 'low'
VOL_HIGH_CUTOFF = 1.3    # vol_ratio_4h > this  -> 'high'

WIN_OUTCOMES  = {'tp_primary', 'tp_stretch'}
LOSS_OUTCOMES = {'stop', 'timeout'}

NOISE_N    = 5    # n < this triggers [NOISE] tag
LOW_CONF_N = 10   # n < this triggers [LOW]   tag

DEFAULT_INPUT     = 'advisor_shadow_perps.jsonl'
DEFAULT_GROUP_BY  = 'time'
TIME_WINDOW_HOURS = 4   # de-stacking window for --group-by time


# ============================================================
# I/O
# ============================================================

def load_records(path):
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: skipping line {line_no}: {e}", file=sys.stderr)
    return records


# ============================================================
# PARSING
# ============================================================

def parse_macd(macd_str):
    """Parse 'bearish_cross | hist -0.0459 (deteriorating)' into parts.
    Returns (cross, hist, improving). Any field is None if unparseable."""
    if not isinstance(macd_str, str) or '|' not in macd_str:
        return None, None, None
    parts = [p.strip() for p in macd_str.split('|')]
    cross = parts[0] if parts and parts[0] else None
    hist = None
    improving = None
    for p in parts[1:]:
        low = p.lower()
        if 'hist' in low:
            m = re.search(r'-?\d+\.?\d*', p)
            if m:
                try:
                    hist = float(m.group())
                except ValueError:
                    pass
        if 'improving' in low:
            improving = True
        elif 'deteriorating' in low:
            improving = False
    return cross, hist, improving


def vol_bucket(vol_ratio):
    if vol_ratio is None:
        return None
    if vol_ratio < VOL_LOW_CUTOFF:
        return 'low'
    if vol_ratio > VOL_HIGH_CUTOFF:
        return 'high'
    return 'medium'


def confidence_band(c):
    if c is None:
        return None
    if c < 7.0:
        return '6.5-6.9'
    if c < 7.5:
        return '7.0-7.4'
    if c < 8.0:
        return '7.5-7.9'
    return '8.0+'


def enrich(record):
    """Add parsed convenience fields to record. Mutates in place."""
    cross, hist, improving = parse_macd(record.get('macd_4h'))
    record['_macd_cross']     = cross
    record['_macd_hist']      = hist
    record['_macd_improving'] = improving
    record['_vol_bucket']     = vol_bucket(record.get('vol_ratio_4h'))
    record['_conf_band']      = confidence_band(record.get('confidence'))
    return record


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


# ============================================================
# OUTCOME CLASSIFICATION
# ============================================================

def is_win(outcome_result):
    return outcome_result in WIN_OUTCOMES


def confidence_tag(n):
    if n == 0:
        return ''
    if n < NOISE_N:
        return '[NOISE]'
    if n < LOW_CONF_N:
        return '[LOW]'
    return ''


# ============================================================
# DE-STACKING
# ============================================================

def grouping_key(record, group_by):
    """Return key shared by records belonging to the same logical trade.
    Bias is always part of the key (longs and shorts never group together)."""
    bias = record.get('bias', '?')
    if group_by == 'time':
        ts = parse_iso(record.get('timestamp'))
        if ts is None:
            return None
        # Floor to a 4h window aligned with UTC bar boundaries (0,4,8,12,16,20)
        floor_hour = (ts.hour // TIME_WINDOW_HOURS) * TIME_WINDOW_HOURS
        window = ts.replace(hour=floor_hour, minute=0, second=0, microsecond=0)
        return (bias, 'time', window.isoformat())
    if group_by == 'exit':
        exit_price = record.get('outcome_exit_price')
        result     = record.get('outcome_result')
        if exit_price is None or result is None:
            return None
        return (bias, 'exit', exit_price, result)
    raise ValueError(f"unknown group_by: {group_by}")


def make_logical_trades(records, group_by):
    """Group records into logical trades. Each logical trade is a dict:
       - records: list of raw records in this group
       - representative: earliest record (used for feature stratification)
       - outcome_result: modal outcome across the group
       - avg_pnl: average outcome_pnl_leveraged_pct across the group
       - size: how many raw entries collapsed into this trade
    Records that can't be grouped become singleton logical trades."""
    grouped = defaultdict(list)
    singletons = []
    for r in records:
        key = grouping_key(r, group_by)
        if key is None:
            singletons.append([r])
        else:
            grouped[key].append(r)

    logical_trades = []
    for group in list(grouped.values()) + singletons:
        # Representative = earliest record by timestamp, fallback to first.
        # Semantic: "the trade you'd have taken if you only acted on the first
        # signal of each window." Outcome and PnL come from the representative,
        # not aggregated across the group, because the other entries in the
        # group are *additional* paper trades, not the same trade.
        with_ts = [(parse_iso(r.get('timestamp')), r) for r in group]
        with_ts = [(ts, r) for ts, r in with_ts if ts is not None]
        rep = min(with_ts, key=lambda x: x[0])[1] if with_ts else group[0]
        logical_trades.append({
            'records': group,
            'representative': rep,
            'outcome_result': rep.get('outcome_result'),
            'avg_pnl': rep.get('outcome_pnl_leveraged_pct'),
            'size': len(group),
        })
    return logical_trades


# ============================================================
# STRATIFICATION
# ============================================================

def stratum_for(record, dimension):
    """Get the stratum value for a record on a given dimension.
    Returns None if the field is missing -- caller must skip None."""
    if dimension == 'div_4h':
        v = record.get('div_4h')
        return v if v else None
    if dimension == 'ema':
        v = record.get('ema_4h') or record.get('ema_status')
        return v if v else None
    if dimension == 'macd_cross':
        return record.get('_macd_cross')
    if dimension == 'macd_improving':
        v = record.get('_macd_improving')
        if v is None:
            return None
        return 'improving' if v else 'deteriorating'
    if dimension == 'vol_bucket':
        return record.get('_vol_bucket')
    if dimension == 'regime':
        v = record.get('local_regime')
        return v if v else None
    if dimension == 'conf_band':
        return record.get('_conf_band')
    raise ValueError(f"unknown dimension: {dimension}")


def stratify_1d(items, dimension, get_record):
    by = defaultdict(list)
    skipped = 0
    for item in items:
        s = stratum_for(get_record(item), dimension)
        if s is None:
            skipped += 1
            continue
        by[s].append(item)
    return by, skipped


def stratify_2d(items, dim_a, dim_b, get_record):
    by = defaultdict(list)
    skipped = 0
    for item in items:
        rec = get_record(item)
        a = stratum_for(rec, dim_a)
        b = stratum_for(rec, dim_b)
        if a is None or b is None:
            skipped += 1
            continue
        by[(a, b)].append(item)
    return by, skipped


def stats_raw(records):
    n = len(records)
    if n == 0:
        return {'n': 0, 'wins': 0, 'wr': None, 'avg_pnl': None}
    wins = sum(1 for r in records if is_win(r.get('outcome_result')))
    pnls = [r.get('outcome_pnl_leveraged_pct') for r in records
            if r.get('outcome_pnl_leveraged_pct') is not None]
    avg = sum(pnls) / len(pnls) if pnls else None
    return {'n': n, 'wins': wins, 'wr': 100.0 * wins / n, 'avg_pnl': avg}


def stats_destacked(logical_trades):
    n = len(logical_trades)
    if n == 0:
        return {'n': 0, 'wins': 0, 'wr': None, 'avg_pnl': None}
    wins = sum(1 for lt in logical_trades if is_win(lt['outcome_result']))
    pnls = [lt['avg_pnl'] for lt in logical_trades if lt['avg_pnl'] is not None]
    avg = sum(pnls) / len(pnls) if pnls else None
    return {'n': n, 'wins': wins, 'wr': 100.0 * wins / n, 'avg_pnl': avg}


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def fmt_n(s):     return f"{s['n']:3d}"
def fmt_wr(s):    return f"{s['wr']:5.1f}%" if s['wr'] is not None else "  -  "
def fmt_pnl(s):   return f"{s['avg_pnl']:+6.2f}%" if s['avg_pnl'] is not None else "   -  "


def fmt_row(label, raw, ds, indent=0):
    pad = '  ' * indent
    return (f"{pad}{label:<28s}  "
            f"{fmt_n(raw)}  {fmt_wr(raw)}  {fmt_pnl(raw)}  {confidence_tag(raw['n']):7s}    "
            f"{fmt_n(ds)}  {fmt_wr(ds)}  {fmt_pnl(ds)}  {confidence_tag(ds['n'])}")


def print_section_header(title):
    print()
    print('=' * 92)
    print(title)
    print('=' * 92)
    print(f"  {'':<28s}  {'Raw view':^32s}    {'De-stacked view':^28s}")
    print(f"  {'':<28s}    n     WR    avgPnL              n     WR    avgPnL")
    print('-' * 92)


def print_1d(title, items_raw, logical_trades, dimension, get_record_raw, get_record_lt):
    raw_by,  raw_skip  = stratify_1d(items_raw, dimension, get_record_raw)
    ds_by,   ds_skip   = stratify_1d(logical_trades, dimension, get_record_lt)
    print_section_header(title)
    keys = sorted(set(raw_by) | set(ds_by))
    for k in keys:
        print(fmt_row(str(k), stats_raw(raw_by.get(k, [])), stats_destacked(ds_by.get(k, []))))
    if raw_skip or ds_skip:
        print(f"  (records skipped — missing field: raw={raw_skip}, destacked={ds_skip})")


def print_2d(title, items_raw, logical_trades, dim_a, dim_b, get_record_raw, get_record_lt):
    raw_by,  raw_skip  = stratify_2d(items_raw, dim_a, dim_b, get_record_raw)
    ds_by,   ds_skip   = stratify_2d(logical_trades, dim_a, dim_b, get_record_lt)
    print_section_header(title)
    keys = sorted(set(raw_by) | set(ds_by))
    last_a = None
    for (a, b) in keys:
        if a != last_a:
            print(f"  [{a}]")
            last_a = a
        print(fmt_row(str(b), stats_raw(raw_by.get((a, b), [])),
                              stats_destacked(ds_by.get((a, b), [])), indent=1))
    if raw_skip or ds_skip:
        print(f"  (records skipped — missing field: raw={raw_skip}, destacked={ds_skip})")


def print_summary(items_raw, logical_trades):
    """Highlight strongest and weakest divergence × EMA combinations."""
    raw_by,  _ = stratify_2d(items_raw,      'div_4h', 'ema', lambda r: r)
    ds_by,   _ = stratify_2d(logical_trades, 'div_4h', 'ema', lambda lt: lt['representative'])

    rows = []
    for (a, b), trades in ds_by.items():
        s_ds  = stats_destacked(trades)
        s_raw = stats_raw(raw_by.get((a, b), []))
        rows.append((f"{a} + {b}", s_raw, s_ds))

    qualifying = [r for r in rows if r[2]['n'] >= NOISE_N and r[2]['wr'] is not None]
    qualifying.sort(key=lambda r: -r[2]['wr'])

    print()
    print('=' * 92)
    print('SUMMARY — Divergence × EMA combinations ranked by de-stacked WR (n >= 5)')
    print('=' * 92)
    if not qualifying:
        print('  No combinations meet the n >= 5 threshold for de-stacked view.')
        print('  Wait for more tagged shorts to accumulate before drawing conclusions.')
        return

    # If we have 6+ qualifying, show top 3 and bottom 3.
    # Otherwise show all of them ranked, no split.
    if len(qualifying) >= 6:
        print('  Strongest (top 3):')
        for label, s_raw, s_ds in qualifying[:3]:
            print(f"    {label:<48s}  raw n={s_raw['n']:3d} WR {fmt_wr(s_raw):>6s}  |  "
                  f"destack n={s_ds['n']:3d} WR {fmt_wr(s_ds):>6s}  {bar_chart(s_ds['wr'])}")
        print('  Weakest (bottom 3):')
        for label, s_raw, s_ds in qualifying[-3:][::-1]:
            print(f"    {label:<48s}  raw n={s_raw['n']:3d} WR {fmt_wr(s_raw):>6s}  |  "
                  f"destack n={s_ds['n']:3d} WR {fmt_wr(s_ds):>6s}  {bar_chart(s_ds['wr'])}")
    else:
        print(f'  All {len(qualifying)} qualifying combinations (ranked, best first):')
        for label, s_raw, s_ds in qualifying:
            print(f"    {label:<48s}  raw n={s_raw['n']:3d} WR {fmt_wr(s_raw):>6s}  |  "
                  f"destack n={s_ds['n']:3d} WR {fmt_wr(s_ds):>6s}  {bar_chart(s_ds['wr'])}")
        print(f"  (Only {len(qualifying)} combination(s) clear the n >= 5 bar.")
        print(f"   Most strata are still too small for confident comparison.)")


def bar_chart(wr_pct, width=10):
    if wr_pct is None:
        return ''
    filled = int(round(wr_pct / 100.0 * width))
    filled = max(0, min(width, filled))
    return '[' + '#' * filled + '.' * (width - filled) + ']'


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(path, items_raw, logical_trades):
    """Flat row-per-stratum dump across all 1D and 2D analyses."""
    dimensions_1d = ['div_4h', 'ema', 'macd_cross', 'macd_improving',
                     'vol_bucket', 'regime', 'conf_band']
    crosstabs = [('div_4h', 'ema'), ('macd_cross', 'vol_bucket'),
                 ('div_4h', 'macd_cross'), ('ema', 'vol_bucket')]

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['analysis', 'dim_a', 'value_a', 'dim_b', 'value_b',
                    'view', 'n', 'wins', 'wr_pct', 'avg_pnl_leveraged_pct',
                    'confidence_tag'])

        def write(analysis, dim_a, val_a, dim_b, val_b, view, s):
            w.writerow([analysis, dim_a, val_a, dim_b or '', val_b or '',
                        view, s['n'], s['wins'],
                        f"{s['wr']:.2f}" if s['wr'] is not None else '',
                        f"{s['avg_pnl']:.4f}" if s['avg_pnl'] is not None else '',
                        confidence_tag(s['n']).strip('[]')])

        for d in dimensions_1d:
            raw_by, _ = stratify_1d(items_raw, d, lambda r: r)
            ds_by,  _ = stratify_1d(logical_trades, d, lambda lt: lt['representative'])
            keys = sorted(set(raw_by) | set(ds_by))
            for k in keys:
                write(f'1d:{d}', d, k, None, None, 'raw',     stats_raw(raw_by.get(k, [])))
                write(f'1d:{d}', d, k, None, None, 'destack', stats_destacked(ds_by.get(k, [])))

        for (a, b) in crosstabs:
            raw_by, _ = stratify_2d(items_raw,      a, b, lambda r: r)
            ds_by,  _ = stratify_2d(logical_trades, a, b, lambda lt: lt['representative'])
            keys = sorted(set(raw_by) | set(ds_by))
            for (va, vb) in keys:
                write(f'2d:{a}x{b}', a, va, b, vb, 'raw',
                      stats_raw(raw_by.get((va, vb), [])))
                write(f'2d:{a}x{b}', a, va, b, vb, 'destack',
                      stats_destacked(ds_by.get((va, vb), [])))

    print(f"\nCSV exported -> {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stratified analysis of advisor shadow perp shorts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'Path to JSONL input (default: {DEFAULT_INPUT})')
    parser.add_argument('--group-by', choices=['time', 'exit'],
                        default=DEFAULT_GROUP_BY,
                        help=f'De-stacking method (default: {DEFAULT_GROUP_BY})')
    parser.add_argument('--export', choices=['csv', 'none'], default='none',
                        help='Optional flat-table export')
    parser.add_argument('--export-path', default='shadow_perps_analysis.csv',
                        help='Path for CSV export (default: shadow_perps_analysis.csv)')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(in_path)
    for r in records:
        enrich(r)

    shorts  = [r for r in records if r.get('bias') == 'short']
    tagged  = [r for r in shorts  if r.get('outcome_tagged')]
    pending = [r for r in shorts  if not r.get('outcome_tagged')]

    print()
    print('=' * 92)
    print('ADVISOR SHADOW PERPS — STRATIFIED ANALYSIS')
    print('=' * 92)
    print(f"  Input:          {in_path}")
    print(f"  Total records:  {len(records)}")
    print(f"  Shorts:         {len(shorts)} ({len(tagged)} tagged, {len(pending)} pending)")
    print(f"  De-stacking:    --group-by {args.group_by}", end='')
    if args.group_by == 'time':
        print(f"  ({TIME_WINDOW_HOURS}h windows aligned to UTC bar boundaries)")
    else:
        print(f"  (grouped by shared exit_price + outcome_result)")
    print()
    print(f"  Win Rate (WR) = % of trades exiting at TP (tp_primary or tp_stretch)")
    print(f"                  rather than stop or timeout.")
    print(f"  Sample size tags:  [NOISE] = n<{NOISE_N},  [LOW] = n<{LOW_CONF_N}")

    if not tagged:
        print("\nNo tagged shorts to analyze. Exiting.")
        return

    logical_trades = make_logical_trades(tagged, args.group_by)

    overall_raw = stats_raw(tagged)
    overall_ds  = stats_destacked(logical_trades)
    print()
    print('-' * 92)
    print('OVERALL TAGGED SHORTS')
    print('-' * 92)
    print(f"  Raw view:        {overall_raw['n']:3d} entries  "
          f"{overall_raw['wins']:3d} wins  WR {overall_raw['wr']:5.1f}%  "
          f"avg PnL {overall_raw['avg_pnl']:+6.2f}% leveraged")
    print(f"  De-stacked view: {overall_ds['n']:3d} trades   "
          f"{overall_ds['wins']:3d} wins  WR {overall_ds['wr']:5.1f}%  "
          f"avg PnL {overall_ds['avg_pnl']:+6.2f}% leveraged")
    if pending:
        print(f"\n  ({len(pending)} pending shorts excluded — outcomes not yet tagged)")

    get_raw = lambda r: r
    get_lt  = lambda lt: lt['representative']

    print_1d('1D: Divergence (4H)',     tagged, logical_trades, 'div_4h',         get_raw, get_lt)
    print_1d('1D: EMA position (4H)',   tagged, logical_trades, 'ema',            get_raw, get_lt)
    print_1d('1D: MACD cross state',    tagged, logical_trades, 'macd_cross',     get_raw, get_lt)
    print_1d('1D: MACD improving flag', tagged, logical_trades, 'macd_improving', get_raw, get_lt)
    print_1d('1D: Volume bucket',       tagged, logical_trades, 'vol_bucket',     get_raw, get_lt)
    print_1d('1D: Local regime',        tagged, logical_trades, 'regime',         get_raw, get_lt)
    print_1d('1D: Confidence band',     tagged, logical_trades, 'conf_band',      get_raw, get_lt)

    print_2d('2D: Divergence × EMA position (PRIMARY CROSS-TAB)',
             tagged, logical_trades, 'div_4h', 'ema', get_raw, get_lt)
    print_2d('2D: MACD cross × Volume bucket',
             tagged, logical_trades, 'macd_cross', 'vol_bucket', get_raw, get_lt)
    print_2d('2D: Divergence × MACD cross',
             tagged, logical_trades, 'div_4h', 'macd_cross', get_raw, get_lt)
    print_2d('2D: EMA × Volume bucket',
             tagged, logical_trades, 'ema', 'vol_bucket', get_raw, get_lt)

    print_summary(tagged, logical_trades)

    if args.export == 'csv':
        export_csv(args.export_path, tagged, logical_trades)

    print()


if __name__ == '__main__':
    main()
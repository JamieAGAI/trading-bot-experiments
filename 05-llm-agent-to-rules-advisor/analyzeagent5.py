"""
Agent5 Decision Analyzer
Run from your SOLPY folder:  python analyze_agent5.py
Reads agent5_decisions.jsonl → prints fingerprint report + saves agent5_report.xlsx
"""
import json, re, os, sys
from collections import defaultdict
from datetime import datetime

DECISIONS_FILE = "agent5_decisions.jsonl"
OUTPUT_XLSX = "agent5_report.xlsx"

def get_fingerprint(e):
    edge = e.get('edge', '')
    m = re.search(r'bear\+div=(\S+)\+macd=(\S+)\+ema=(\S+)', edge)
    if m:
        return f"div={m.group(1)}|macd={m.group(2)}|ema={m.group(3)}"
    return "no_match"

def parse_funding_oi(s):
    funding, oi = None, None
    if not s: return funding, oi
    m = re.search(r'Funding:\s*([+-]?[\d.]+)%', str(s))
    if m: funding = float(m.group(1))
    m = re.search(r'OI:\s*\$?([\d.]+)M', str(s))
    if m: oi = float(m.group(1))
    return funding, oi

def parse_fg(s):
    if not s: return None
    m = re.search(r'F&G:\s*(\d+)', str(s))
    return int(m.group(1)) if m else None

def print_report(entries):
    tagged = [e for e in entries if e.get('outcome_tagged')]
    print(f"\n{'='*60}")
    print(f"  AGENT5 DECISION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  Total entries:  {len(entries)}")
    print(f"  Tagged:         {len(tagged)}")
    print(f"  Untagged:       {len(entries) - len(tagged)}")
    print(f"  Date range:     {entries[0]['timestamp'][:16]} → {entries[-1]['timestamp'][:16]}")

    fp_data = defaultdict(list)
    for e in entries:
        fp_data[get_fingerprint(e)].append(e)

    print(f"  Fingerprints:   {len(fp_data)}")
    print(f"\n{'─'*60}")
    print(f"  FINGERPRINT BREAKDOWN")
    print(f"{'─'*60}\n")

    ranked = []
    for fp, group in fp_data.items():
        t = [e for e in group if e.get('outcome_tagged')]
        if t:
            avg_c = sum(e['outcome_close_pct'] for e in t) / len(t)
            avg_h = sum(e['outcome_high_pct'] for e in t) / len(t)
            avg_l = sum(e['outcome_low_pct'] for e in t) / len(t)
            wr = sum(1 for e in t if e['outcome_close_pct'] > 0) / len(t) * 100
            ranked.append((fp, len(group), len(t), avg_c, avg_h, avg_l, wr))
        else:
            ranked.append((fp, len(group), 0, None, None, None, None))

    ranked.sort(key=lambda x: x[3] if x[3] is not None else -999, reverse=True)

    for fp, total, tag_count, avg_c, avg_h, avg_l, wr in ranked:
        icon = "+" if avg_c and avg_c > 0 else "-" if avg_c and avg_c < 0 else "?"
        print(f"  [{icon}] {fp}")
        print(f"      Entries: {total} | Tagged: {tag_count}")
        if avg_c is not None:
            print(f"      Avg close: {avg_c:+.2f}% | Avg high: {avg_h:+.2f}% | Avg low: {avg_l:+.2f}%")
            print(f"      Long WR: {wr:.0f}%")
            convictions = [e.get('conviction', 0) for e in fp_data[fp]]
            print(f"      Conviction range: {min(convictions):.1f} - {max(convictions):.1f}")
        else:
            print(f"      No tagged outcomes yet")
        print()

    # Conviction distribution
    print(f"{'─'*60}")
    print(f"  CONVICTION DISTRIBUTION")
    print(f"{'─'*60}\n")
    conv_counts = defaultdict(int)
    for e in entries:
        conv_counts[e.get('conviction', 0)] += 1
    for c in sorted(conv_counts.keys()):
        bar = "#" * (conv_counts[c] // 5)
        print(f"    {c:>4.1f}: {conv_counts[c]:>4} {bar}")

    # Daily summary
    print(f"\n{'─'*60}")
    print(f"  DAILY SUMMARY")
    print(f"{'─'*60}\n")
    daily = defaultdict(list)
    for e in entries:
        daily[e['timestamp'][:10]].append(e)
    for date in sorted(daily.keys()):
        group = daily[date]
        avg_price = sum(e.get('price', 0) for e in group) / len(group)
        fg_vals = [parse_fg(e.get('fear_greed')) for e in group if parse_fg(e.get('fear_greed')) is not None]
        avg_fg = sum(fg_vals) / len(fg_vals) if fg_vals else None
        fps_today = set(get_fingerprint(e) for e in group)
        fg_str = f"F&G {avg_fg:.0f}" if avg_fg else "F&G n/a"
        print(f"    {date} | {len(group):>3} entries | avg ${avg_price:.2f} | {fg_str} | {len(fps_today)} fingerprints")

    print(f"\n{'='*60}\n")

def save_xlsx(entries):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        print("  openpyxl not installed — run: pip install openpyxl")
        print("  Skipping XLSX export.\n")
        return

    wb = openpyxl.Workbook()
    hdr_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    hdr_fill = PatternFill("solid", fgColor="2F5496")
    data_font = Font(name="Arial", size=10)

    # Sheet 1: Raw data
    ws = wb.active
    ws.title = "Raw Decisions"
    headers = ['timestamp', 'date', 'hour', 'action', 'conviction', 'regime', 'price',
               'div_4h', 'div_1h', 'macd_cross', 'macd_hist', 'ema_status', 'fib_closest',
               'rsi_4h', 'rsi_1h', 'atr', 'btc_change', 'fear_greed', 'funding_rate',
               'open_interest_m', 'outcome_high_pct', 'outcome_low_pct', 'outcome_close_pct',
               'outcome_tagged', 'fingerprint']
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = hdr_font
        cell.fill = hdr_fill
        cell.alignment = Alignment(horizontal='center')

    for row_idx, e in enumerate(entries, 2):
        ts = e.get('timestamp', '')
        funding, oi = parse_funding_oi(e.get('funding_oi'))
        fg = parse_fg(e.get('fear_greed'))
        row = [ts[:19], ts[:10], ts[11:16], e.get('action'), e.get('conviction'),
               e.get('regime'), e.get('price'), e.get('div_4h'), e.get('div_1h'),
               e.get('macd_cross'), e.get('macd_hist'), e.get('ema_status'),
               e.get('fib_closest'), e.get('rsi_4h'), e.get('rsi_1h'), e.get('atr'),
               e.get('btc_change'), fg, funding, oi, e.get('outcome_high_pct'),
               e.get('outcome_low_pct'), e.get('outcome_close_pct'),
               e.get('outcome_tagged', False), get_fingerprint(e)]
        for col, val in enumerate(row, 1):
            ws.cell(row=row_idx, column=col, value=val).font = data_font

    widths = {'A':20,'B':12,'C':7,'D':8,'E':10,'F':7,'G':9,'H':18,'I':18,'J':15,
              'K':10,'L':22,'M':10,'N':8,'O':8,'P':8,'Q':10,'R':8,'S':10,'T':10,
              'U':12,'V':12,'W':12,'X':12,'Y':35}
    for c, w in widths.items():
        ws.column_dimensions[c].width = w
    ws.auto_filter.ref = f"A1:Y{len(entries)+1}"
    ws.freeze_panes = "A2"

    # Sheet 2: Fingerprint summary
    ws2 = wb.create_sheet("Fingerprint Summary")
    fp_headers = ['fingerprint','total','tagged','avg_close','avg_high','avg_low',
                  'long_wr','avg_conviction','avg_rsi_4h','avg_atr']
    for col, h in enumerate(fp_headers, 1):
        cell = ws2.cell(row=1, column=col, value=h)
        cell.font = hdr_font
        cell.fill = hdr_fill

    fp_groups = defaultdict(list)
    for e in entries:
        fp_groups[get_fingerprint(e)].append(e)

    row = 2
    for fp in sorted(fp_groups.keys()):
        g = fp_groups[fp]
        t = [e for e in g if e.get('outcome_tagged')]
        avg_c = round(sum(e['outcome_close_pct'] for e in t)/len(t),2) if t else None
        avg_h = round(sum(e['outcome_high_pct'] for e in t)/len(t),2) if t else None
        avg_l = round(sum(e['outcome_low_pct'] for e in t)/len(t),2) if t else None
        wr = round(sum(1 for e in t if e['outcome_close_pct']>0)/len(t)*100,1) if t else None
        avg_conv = round(sum(e.get('conviction',0) for e in g)/len(g),1)
        rsi_vals = [e.get('rsi_4h',0) for e in g if e.get('rsi_4h')]
        avg_rsi = round(sum(rsi_vals)/len(rsi_vals),1) if rsi_vals else None
        atr_vals = [e.get('atr',0) for e in g if e.get('atr')]
        avg_atr = round(sum(atr_vals)/len(atr_vals),2) if atr_vals else None
        vals = [fp,len(g),len(t),avg_c,avg_h,avg_l,wr,avg_conv,avg_rsi,avg_atr]
        for col, val in enumerate(vals, 1):
            cell = ws2.cell(row=row, column=col, value=val)
            cell.font = data_font
            if col == 4 and val is not None:
                cell.font = Font(name="Arial", size=10, color="006100" if val > 0 else "9C0006")
        row += 1

    fp_widths = {'A':40,'B':10,'C':10,'D':12,'E':12,'F':12,'G':12,'H':14,'I':10,'J':10}
    for c, w in fp_widths.items():
        ws2.column_dimensions[c].width = w
    ws2.auto_filter.ref = f"A1:J{row-1}"
    ws2.freeze_panes = "A2"

    wb.save(OUTPUT_XLSX)
    print(f"  Saved: {OUTPUT_XLSX}\n")

if __name__ == "__main__":
    if not os.path.exists(DECISIONS_FILE):
        print(f"  {DECISIONS_FILE} not found — run from your SOLPY folder.")
        sys.exit(1)

    with open(DECISIONS_FILE) as f:
        entries = [json.loads(l.strip()) for l in f if l.strip()]

    print_report(entries)

    if "--xlsx" in sys.argv:
        save_xlsx(entries)
    else:
        print(f"  (add --xlsx flag to export spreadsheet)\n")
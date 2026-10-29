"""Second case scan: SLGT-or-lower Day 1 (1630Z) outlooks, with SPC filtered report counts.

    python find_surprises.py 2024-03-01 2026-10-27

Surprise day: top category <= MRGL (or SLGT with 2% tor) yet >= 5 tornado or >= 60 total reports.
Quiet MRGL day: MRGL top category, <= 2 total reports, Mar-Jul (when the global models
would most likely paint a false signal).
Writes cases/surprise_scan.csv.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from config import CASES_DIR

ARCH = "https://www.spc.noaa.gov/products/outlook/archive"
RANK = {"TSTM": 1, "MRGL": 2, "SLGT": 3, "ENH": 4, "MDT": 5, "HIGH": 6}
S = requests.Session()
S.headers["User-Agent"] = "pogs-research (hunterlhurley case study)"


def get(url, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.ok:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


def geo(d, kind):
    r = get(f"{ARCH}/{d:%Y}/day1otlk_{d:%Y%m%d}_1630_{kind}.lyr.geojson")
    return r.json().get("features", []) if r is not None else None


def max_prob(d, kind):
    best = 0.0
    for f in geo(d, kind) or []:
        try:
            best = max(best, float(f["properties"].get("LABEL", 0)))
        except (TypeError, ValueError):
            pass
    return best


def report_counts(d):
    """Counts per hazard from the combined filtered file (three header-separated sections)."""
    r = get(f"https://www.spc.noaa.gov/climo/reports/{d:%y%m%d}_rpts_filtered.csv")
    n = {"torn": 0, "wind": 0, "hail": 0}
    if r is None:
        return None
    sec = None
    for line in r.text.splitlines():
        if line.startswith("Time,"):
            sec = {"F_Scale": "torn", "Speed": "wind", "Size": "hail"}.get(line.split(",")[1])
        elif sec and line.strip():
            n[sec] += 1
    return n


def scan_day(d):
    try:
        return _scan_day(d)
    except Exception as e:  # one bad product must not stop the scan
        print(f"  {d}: skipped ({e})", flush=True)
        return None


def _scan_day(d):
    cat = geo(d, "cat")
    if cat is None:
        return None
    top = max((RANK.get(str(f["properties"].get("LABEL")), 0) for f in cat), default=0)
    if top >= RANK["ENH"]:
        return None
    n = report_counts(d)
    if n is None:
        return None
    row = {"date": d.isoformat(), "category": next((k for k, v in RANK.items() if v == top), "NONE"),
           "torn_max": max_prob(d, "torn") if top >= RANK["MRGL"] else 0.0, **n, "total": sum(n.values())}
    low_risk = top <= RANK["MRGL"] or (top == RANK["SLGT"] and row["torn_max"] <= 0.02 + 1e-9)
    if low_risk and (n["torn"] >= 5 or row["total"] >= 60):
        row["kind"] = "SURPRISE"
    elif top == RANK["MRGL"] and row["total"] <= 2 and 3 <= d.month <= 7:
        row["kind"] = "QUIET_MRGL"
    else:
        row["kind"] = ""
    return row


def main():
    d0, d1 = (date.fromisoformat(x) for x in sys.argv[1:3])
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(scan_day, days) if r]
    df = pd.DataFrame(rows).sort_values("date")
    out = os.path.join(CASES_DIR, "surprise_scan.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} SLGT-or-lower days scanned -> {out}")
    pd.set_option("display.width", 200)
    for kind in ("SURPRISE", "QUIET_MRGL"):
        sub = df[df.kind == kind]
        print(f"\n{kind}: {len(sub)} days")
        print(sub[["date", "category", "torn_max", "torn", "wind", "hail", "total"]].to_string(index=False))


if __name__ == "__main__":
    main()

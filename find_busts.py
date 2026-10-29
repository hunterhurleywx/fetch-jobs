"""Find SPC bust days: ENH+ Day 1 (1630Z) outlooks whose elevated hazards under-verified.

    python find_busts.py 2024-03-01 2026-10-27

A hazard counts only if SPC had high odds for it (Hunter's rule):
  tornado >= 10%  -> bust if <= 2 tornado reports inside the 5%+ tornado area
  wind    >= 30%  -> bust if < 10 wind reports inside the 15%+ wind area
  hail    >= 30%  -> bust if < 10 hail reports inside the 15%+ hail area
Day class: FULL bust = every elevated hazard busted; PARTIAL = at least one; none otherwise.
Writes cases/bust_scan.csv.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests
from shapely.geometry import Point, shape

from config import CASES_DIR

ARCH = "https://www.spc.noaa.gov/products/outlook/archive"
RANK = {"TSTM": 1, "MRGL": 2, "SLGT": 3, "ENH": 4, "MDT": 5, "HIGH": 6}
RULE = {"torn": (0.10, 0.05, lambda n: n <= 2), "wind": (0.30, 0.15, lambda n: n < 10),
        "hail": (0.30, 0.15, lambda n: n < 10)}
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


def reports(d, haz):
    r = get(f"https://www.spc.noaa.gov/climo/reports/{d:%y%m%d}_rpts_filtered_{haz}.csv")
    rows = []
    if r is not None:
        for line in r.text.splitlines()[1:]:  # comments may contain unquoted commas
            f = line.split(",", 7)
            try:
                rows.append((float(f[5]), float(f[6])))
            except (IndexError, ValueError):
                pass
    return pd.DataFrame(rows, columns=["Lat", "Lon"])


def level(props):
    lab = str(props.get("LABEL", props.get("label", "")))
    try:
        return float(lab)
    except ValueError:
        return None


def scan_day(d):
    try:
        return _scan_day(d)
    except Exception as e:  # one bad product must not stop the scan
        print(f"  {d}: skipped ({e})", flush=True)
        return None


def _scan_day(d):
    cat = geo(d, "cat")
    if not cat:
        return None
    top = max((RANK.get(str(f["properties"].get("LABEL")), 0) for f in cat), default=0)
    if top < RANK["ENH"]:
        return None
    row = {"date": d.isoformat(), "category": [k for k, v in RANK.items() if v == top][0]}
    busted, elevated = [], []
    for haz, (hi, area_lvl, is_bust) in RULE.items():
        feats = geo(d, haz) or []
        levels = [(level(f["properties"]), shape(f["geometry"])) for f in feats]
        levels = [(lv, g) for lv, g in levels if lv is not None]
        mx = max((lv for lv, _ in levels), default=0.0)
        rep = reports(d, haz)
        area = [g for lv, g in levels if lv >= area_lvl - 1e-9]
        n_in = sum(any(g.contains(Point(lo, la)) for g in area) for la, lo in zip(rep.Lat, rep.Lon)) if area else 0
        row.update({f"{haz}_max": mx, f"{haz}_reports": len(rep), f"{haz}_in_area": n_in})
        if mx >= hi - 1e-9:
            elevated.append(haz)
            if is_bust(n_in):
                busted.append(haz)
    row["elevated"] = ",".join(elevated)
    row["busted"] = ",".join(busted)
    row["bust_class"] = ("FULL" if elevated and len(busted) == len(elevated) else "PARTIAL" if busted else "")
    return row


def main():
    d0, d1 = (date.fromisoformat(x) for x in sys.argv[1:3])
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(6) as ex:
        rows = [r for r in ex.map(scan_day, days) if r]
    df = pd.DataFrame(rows).sort_values("date")
    out = os.path.join(CASES_DIR, "bust_scan.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} ENH+ days scanned -> {out}")
    pd.set_option("display.width", 220)
    print(df[df.bust_class != ""][["date", "category", "elevated", "busted", "bust_class", "torn_max", "torn_in_area",
                                    "wind_max", "wind_in_area", "hail_max", "hail_in_area"]].to_string(index=False))


if __name__ == "__main__":
    main()

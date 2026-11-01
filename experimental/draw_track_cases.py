"""EXPERIMENTAL: past-case maps with the track-record blend (Hunter, 1 Nov 2026).

    python experimental/draw_track_cases.py [--cases ...] [--procs 3]

Reads cases/<date>/output/consistency_raw_d<L>_track.npz (consistency_blend.py --mode track --tag _track) for every lead,
calibrates it like the real-time maps (overall: 14-case any-severe; hazards capped at overall), and draws the 2x2 panel
with SPC's outline (Days 8-3: 15% any-severe; Days 2-1: Slight+ and each hazard's own lines) and the day's reports.
Writes pages/pasttrack/maps/<date>_d<L>.png and pages/pasttrack/cases.json (weights, maxima, report hits).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate_hazards as ch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import redraw_latest as rl  # noqa: E402
from config import CASES_DIR, PROJECT_DIR  # noqa: E402

OUT = os.environ.get("POGS_TRACK_OUT", os.path.join(PROJECT_DIR, "pages", "pasttrack"))
MODELS = ["eps", "gefs", "hres", "gfs", "nam", "hrrr", "refs", "nam12", "rap", "rrfs"]
HZ = (("tor", "Tornado"), ("wind", "Wind"), ("hail", "Hail"))


def extent_for(rep, sev):
    """24 x 17 deg box centred on the reports (or our 5% area when there are few)."""
    pts = []
    if rep is not None and len(rep):
        pts = list(zip(rep["lat"], rep["lon"]))
    if len(pts) < 10:
        m = sev >= 5
        pts += list(zip(plots.LAT2[m], plots.LON2[m]))
    if not pts:
        return (-108, -86, 26, 42)
    la, lo = np.median([p[0] for p in pts]), np.median([p[1] for p in pts])
    lo = float(np.clip(lo, -113, -79))
    la = float(np.clip(la, 33.5, 41.5))
    return (lo - 12, lo + 12, la - 8.5, la + 8.5)


def hits(rep, field, thr, types=None):
    """Share of reports whose grid cell has field >= thr (reports of the given types)."""
    if rep is None or not len(rep):
        return None
    d = rep if types is None else rep[rep["type"].isin(types)]
    if not len(d):
        return None
    i = np.clip(np.round((d["lat"].to_numpy() - 20) / 0.5).astype(int), 0, field.shape[0] - 1)
    j = np.clip(np.round((d["lon"].to_numpy() + 130) / 0.5).astype(int), 0, field.shape[1] - 1)
    return round(float(np.mean(field[i, j] >= thr)), 2)


def do_case(date):
    out = os.path.join(CASES_DIR, date, "output")
    conv = datetime.strptime(date, "%Y-%m-%d")
    hc = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
    any14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
    us = plot_hhwx.us_mask()
    rep = plots.load_reports(date, conv)
    res = np.load(os.path.join(out, "results.npz"))
    summary = {"date": date, "n_reports": int(len(rep)) if rep is not None else 0,
               "n_tor": int((rep["type"] == "tornado").sum()) if rep is not None else 0, "leads": {}}
    for L in range(8, 0, -1):
        p = os.path.join(out, f"consistency_raw_d{L}_track.npz")
        if not os.path.exists(p):
            continue
        r = np.load(p)
        sev = np.where(us, ch.calibrated(r["sev"], any14), 0)
        maps = [("Overall severe", sev)] + [(nm, np.where(us, np.minimum(ch.calibrated(r[k], hc[k]), sev), 0)) for k, nm in HZ]
        spc = (rl.spc_hazard_panels(date, L) if L in (1, 2) else None) or plots.spc_lead_polys(date, conv, L)[0]
        w = {m: round(float(r[f"w_{m}"]), 3) for m in MODELS if f"w_{m}" in r.files}
        used = [m for m in MODELS if w.get(m, 0) > 0]
        db03 = (conv - timedelta(days=1)).strftime("%Y%m%d") + "03"
        runs = rl.runs_text(res, L, used, db03)
        png = os.path.join(OUT, "maps", f"{date}_d{L}.png")
        rl.draw_panel(maps, f"HHwx Calibrated Guidance Day-{L}: {conv:%a %d %b %Y}", png, spc, L,
                      extent=extent_for(rep, sev), runs=runs, reports=rep, dpi=90)
        tor = maps[1][1]
        summary["leads"][L] = {
            "weights": w, "max": {nm: round(float(f.max()), 1) for nm, f in maps},
            "hit15": hits(rep, sev, 15), "hit5": hits(rep, sev, 5), "tor_hit2": hits(rep, tor, 2, ["tornado"]),
            "tor_hit5": hits(rep, tor, 5, ["tornado"])}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    cases = a.cases or sorted(d for d in os.listdir(CASES_DIR) if d[:2] == "20" and len(d) == 10 and d != "2026-11-02"
                              and os.path.exists(os.path.join(CASES_DIR, d, "output", "consistency_raw_d4_track.npz")))
    with Pool(a.procs) as pool:
        out = pool.map(do_case, cases)
    jp = os.path.join(OUT, "cases.json")
    old = json.load(open(jp)) if os.path.exists(jp) and a.cases else []
    keep = {c["date"]: c for c in old}
    keep.update({c["date"]: c for c in out})
    json.dump(sorted(keep.values(), key=lambda c: c["date"]), open(jp, "w"), indent=1)
    print(f"{len(out)} cases -> {jp}")


if __name__ == "__main__":
    main()

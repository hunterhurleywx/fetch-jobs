"""EXPERIMENTAL: does giving the global models a share of the Day 2-1 track blend help? (Hunter, 1 Nov 2026)

    python experimental/global_share_test.py 2025-05-16 [--shares 0 0.15 0.25 0.35]

For Days 2 and 1, re-blends with consistency_blend.py --mode track --global-share S for each share S (0 = the live
rule: regional models only), calibrates like the maps (overall capped at tornado + wind + hail), and scores every map
against the day's reports within 40 km (25 mi) over the US: Brier sums, event counts and grid counts per hazard, so
the cases can be pooled into a Brier skill score later. Also the share of reports inside our 15% / 5% overall areas.
Writes <POGS_TRACK_OUT>/gstest_<date>.json.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import calibrate_hazards as ch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import redraw_latest as rl  # noqa: E402
import verify_case as vc  # noqa: E402
from config import CASES_DIR, PROJECT_DIR  # noqa: E402

HZ = (("tor", "Tornado", "tornado"), ("wind", "Wind", "wind"), ("hail", "Hail", "hail"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--shares", nargs="+", type=float, default=[0.0, 0.15, 0.25, 0.35])
    a = ap.parse_args()
    out = os.environ.get("POGS_TRACK_OUT", os.path.join(PROJECT_DIR, "pages", "pasttrack"))
    os.makedirs(out, exist_ok=True)
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    rep = plots.load_reports(a.date, conv)
    if rep is None:
        print("no reports")
        return
    us = plot_hhwx.us_mask() & vc.DOM
    obs = {"Overall severe": vc.report_mask(rep)}
    for _, nm, typ in HZ:
        obs[nm] = vc.report_mask(rep[rep["type"] == typ])
    hc = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
    any14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
    res = {"date": a.date, "n_reports": int(len(rep)), "leads": {}}
    for L in (2, 1):
        res["leads"][L] = {}
        for s in a.shares:
            tag = f"_gs{int(round(s * 100)):02d}"
            r = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "consistency_blend.py"), a.date, "--lead", str(L),
                                "--hazards", "--mode", "track", "--no-draw", "--tag", tag, "--global-share", str(s)],
                               capture_output=True, text=True)
            p = os.path.join(CASES_DIR, a.date, "output", f"consistency_raw_d{L}{tag}.npz")
            if r.returncode != 0 or not os.path.exists(p):
                print(f"lead {L} share {s}: blend failed: {r.stderr.strip()[-200:]}")
                continue
            z = np.load(p)
            sev = np.where(us, ch.calibrated(z["sev"], any14), 0)
            maps = rl.union_cap([("Overall severe", sev)] + [(nm, np.where(us, np.minimum(ch.calibrated(z[k], hc[k]), sev), 0))
                                                           for k, nm, _ in HZ])
            row = {"weights": {k[2:]: round(float(z[k]), 3) for k in z.files if k.startswith("w_")}}
            for nm, f in maps:
                pr, o = f[us] / 100.0, obs[nm][us].astype(float)
                row[nm] = {"bs_sum": float(((pr - o) ** 2).sum()), "n": int(us.sum()), "events": float(o.sum()),
                           "max": round(float(f.max()), 1)}
            sv = maps[0][1]
            i = np.clip(np.round((rep["lat"].to_numpy() - 20) / 0.5).astype(int), 0, sv.shape[0] - 1)
            j = np.clip(np.round((rep["lon"].to_numpy() + 130) / 0.5).astype(int), 0, sv.shape[1] - 1)
            row["hit15"], row["hit5"] = round(float(np.mean(sv[i, j] >= 15)), 3), round(float(np.mean(sv[i, j] >= 5)), 3)
            res["leads"][L][str(s)] = row
            print(f"lead {L} share {s}: overall max {row['Overall severe']['max']} hit15 {row['hit15']}")
    json.dump(res, open(os.path.join(out, f"gstest_{a.date}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Bust-case summary: how strongly did we signal a big day, compared with SPC and with what happened?

    python bust_summary.py 2025-09-23 2024-03-25 ...

Per case and lead, from ci_layers.npz (env x CI) with the stored provisional calibration and the
base-rate correction (calibration_provisional.json + odds factor from base_rate.py):
  ours_max    our calibrated max over CONUS (%)
  ours_in_spc our mean / max inside SPC's area for that lead (Day 4-3: any-severe 15%; Day 2-1: SLGT+)
  spc         what SPC had at that lead
  hits        share of reports inside our >= 15% / >= 5% area (corrected)
Writes cases/bust_summary.csv and prints a table.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep

import plots
import verify_case as vc
from config import CASES_DIR, GRID_LATS, GRID_LONS

P_CLIM, P_TRAIN = 0.0097, 0.0282  # base_rate.py, 2025 climatology vs training cases
SHIFT = np.log(P_CLIM / (1 - P_CLIM)) - np.log(P_TRAIN / (1 - P_TRAIN))
LON2, LAT2 = np.meshgrid(GRID_LONS, GRID_LATS)


def calibrated(f, cal, corrected=True):
    p = np.clip(np.interp(f, cal["index"], cal["prob"]), 1e-6, 1 - 1e-6)
    if corrected:
        p = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + SHIFT)))
    return 100 * p


def spc_mask(case, conv_day, lead):
    polys, cat = plots.spc_lead_polys(case, conv_day, lead)
    if not polys:
        return None, "none"
    keep = [g for g, p in polys if (str(p.get("level")) in ("SLGT", "ENH", "MDT", "HIGH") if cat
                                     else str(p.get("level")) in ("0.15", "0.30"))]
    labels = sorted({str(p.get("level")) for g, p in polys})
    if not keep:
        return None, ",".join(labels)
    pg = [prep(g) for g in keep]
    m = np.array([[any(g.contains(Point(lo, la)) for g in pg) for lo in GRID_LONS] for la in GRID_LATS])
    return m, ",".join(labels)


def main():
    cal = json.load(open(os.path.join(CASES_DIR, "calibration_provisional.json")))["env_ci"]
    rows = []
    for case in sys.argv[1:]:
        conv_day = datetime.strptime(case, "%Y-%m-%d")
        z = np.load(os.path.join(CASES_DIR, case, "output", "ci_layers.npz"))
        rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
        rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
        ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
        rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
        ntor = int(rep.type.str.startswith("torn").sum())
        for lead in range(8, 0, -1):
            k = f"{lead}__env*ci"
            if k not in z.files:
                continue
            p = calibrated(z[k], cal)
            dom = np.where(vc.DOM, p, 0)
            m, labels = spc_mask(case, conv_day, lead)
            i, j = np.unravel_index(np.argmax(dom), dom.shape)
            rows.append(dict(case=case, lead=lead, reports=len(rep), tor=ntor, spc=labels,
                             ours_max=round(float(dom.max()), 1), at=f"{GRID_LATS[i]:.1f}N {-GRID_LONS[j]:.1f}W",
                             ours_in_spc_mean=round(float(p[m].mean()), 1) if m is not None and m.any() else None,
                             rep_in_15=round(float((p[ri, rj] >= 15).mean()), 2) if len(rep) else None,
                             rep_in_5=round(float((p[ri, rj] >= 5).mean()), 2) if len(rep) else None))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CASES_DIR, "bust_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

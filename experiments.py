"""Score-variant experiments across cached cases (no downloads).

    python experiments.py

Each variant re-scores the blend for every case/lead and reports AUC, the share of
tornado / all reports in the top-5% area, and the distance of the maximum from the
tornado-report centroid. Variants are hypotheses from the case diagnostics
(research/05_case_2025-04-27/03_diagnosis.md); none is adopted by running this.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

CASES = {"2025-05-16": (7, 6, 5), "2025-06-16": (5, 4, 3, 2, 1), "2025-04-27": (5, 4, 3, 2, 1)}

# variants = scoring.PROFILES names
VARIANTS = ["v2", "highplains", "shear30", "liftstrong", "msl", "mcs", "msl_mcs"]


def run(args):
    case, leads, vname = args
    import run_case, scoring
    import verify_case as vc
    from config import CASES_DIR, GRID_LATS, GRID_LONS
    scoring.set_profile(vname)
    scoring.score_current = lambda ing, fh: np.zeros_like(ing["shear"])  # skip: not under test
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    res, _, _ = run_case.compute(conv_day, list(leads), log=lambda s: None)
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    tor = rep.type.str.startswith("torn").values
    ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
    rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
    label = vc.report_mask(rep)
    tl, tn = rep.lat[tor].mean(), rep.lon[tor].mean()
    rows = []
    for lead in leads:
        f = res[f"v2|{lead}|blend|day"]
        t5 = vc.top_area_mask(f, 0.05)[ri, rj]
        j = np.unravel_index(np.argmax(np.where(vc.DOM, f, -1)), f.shape)
        wind = rep.type.str.startswith("wind").values
        rows.append(dict(case=case, variant=vname, lead=lead, auc=vc.auc(f, label, vc.W),
                         top5_tor=t5[tor].mean(), top5_wind=t5[wind].mean(), top5_all=t5.mean(),
                         dmax_tor_km=vc.hav_km(GRID_LATS[j[0]], GRID_LONS[j[1]], tl, tn)))
    return rows


def main():
    jobs = [(c, l, v) for c, l in CASES.items() for v in VARIANTS]
    with ProcessPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(run, jobs) for r in rs]
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases", "experiments_variants.csv")
    df.to_csv(out, index=False)
    s = df.groupby(["case", "variant"])[["auc", "top5_tor", "top5_wind", "top5_all", "dmax_tor_km"]].mean().round(3)
    pd.set_option("display.width", 200)
    print(s.to_string())
    print("\nMean over the three cases:")
    print(df.groupby("variant")[["auc", "top5_tor", "top5_wind", "top5_all", "dmax_tor_km"]].mean().round(3).sort_values("auc").to_string())


if __name__ == "__main__":
    main()

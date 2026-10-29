"""Backtest: does weighting cycles by cross-model agreement (and recency) beat the equal-weight pool?

    python agreement_test.py 2025-05-16 2025-06-16 2025-04-27 2026-04-13 [--scheme v2]

Uses the per-cycle model maps already in results.npz ("{scheme}|{lead}|{model}|cyc{YYYYMMDDHH}").
For each lead, per model:
  equal      mean of the model's cycle maps (baseline, ~ the official pool)
  newest2    mean of the newest two cycles only
  recency    weights 0.5^(age/12 h)
  agree      weight = pattern correlation of that cycle's map with the OTHER family's newest
             cycle (GFS/GEFS vs ECMWF HRES/EPS), floored at 0.05
  agree_rec  agree x recency
Models are then blended with scoring.BLEND_WEIGHTS. Scored with area-weighted AUC (report
within 40 km) and top-5% area report capture. Writes cases/agreement_test.csv.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import consensus
import scoring
import verify_case as vc
from config import CASES_DIR, MODELS

FAMILY = {"gfs": "ec", "gefs": "ec", "hres": "ncep", "eps": "ncep"}  # -> the OTHER family
MEMBERS = {"ncep": ("gfs", "gefs"), "ec": ("hres", "eps")}


def corr(a, b):
    m = vc.DOM & ((a > 0.05) | (b > 0.05))
    if m.sum() < 20:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def blend(fields):
    return scoring.blend({m: f for m, f in fields.items() if f is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--scheme", default="v2")
    a = ap.parse_args()
    rows = []
    for case in a.cases:
        conv_day = datetime.strptime(case, "%Y-%m-%d")
        res = np.load(os.path.join(CASES_DIR, case, "output", "results.npz"))
        rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
        rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
        lab = vc.report_mask(rep)
        ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
        rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
        leads = sorted({int(k.split("|")[1]) for k in res.files if k.startswith(f"{a.scheme}|") and "|cyc" in k})
        for lead in leads:
            cyc = {}  # model -> {init: field}
            for m in MODELS:
                pre = f"{a.scheme}|{lead}|{m}|cyc"
                cyc[m] = {datetime.strptime(k[len(pre):], "%Y%m%d%H"): res[k] for k in res.files if k.startswith(pre)}
            newest = max(i for m in MODELS for i in cyc[m])
            # the other family's newest-cycle map (blend of its two models)
            fam_new = {}
            for fam, ms in MEMBERS.items():
                fam_new[fam] = blend({m: cyc[m][max(cyc[m])] if cyc[m] else None for m in ms})
            variants = {k: {} for k in ("equal", "newest2", "recency", "agree", "agree_rec")}
            variants["scenario"] = consensus.scenario_pool(cyc)
            variants["scenario_rec"] = consensus.scenario_pool(cyc, half_life=12.0)
            for m in MODELS:
                if not cyc[m]:
                    continue
                inits = sorted(cyc[m])
                F = np.stack([cyc[m][i] for i in inits])
                age = np.array([(newest - i).total_seconds() / 3600 for i in inits])
                other = fam_new[FAMILY[m]]
                ag = np.array([max(corr(cyc[m][i], other), 0.05) if other is not None else 1.0 for i in inits])
                rec = 0.5 ** (age / 12.0)
                for name, w in (("equal", np.ones(len(inits))), ("newest2", (np.argsort(np.argsort(-age)) >= len(inits) - 2) * 1.0),
                                ("recency", rec), ("agree", ag), ("agree_rec", ag * rec)):
                    variants[name][m] = np.tensordot(w / w.sum(), F, axes=1)
            gated = {"gate_equal": ("equal", 0.4), "gate_scen_rec": ("scenario_rec", 0.4),
                     "gate_scen_rec_strong": ("scenario_rec", 0.7)}
            for name, fields in list(variants.items()) + [(g, None) for g in gated]:
                if fields is None:
                    base, k = gated[name]
                    f = consensus.agreement_gate(blend(variants[base]), cyc, k)
                else:
                    f = blend(fields)
                t5 = vc.top_area_mask(f, 0.05)[ri, rj]
                rows.append(dict(case=case, lead=lead, variant=name, auc=vc.auc(f, lab, vc.W), top5=t5.mean()))
        print(f"{case} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CASES_DIR, "agreement_test.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nMean over cases and leads:")
    print(df.groupby("variant")[["auc", "top5"]].mean().round(3).to_string())
    print("\nAUC by case (mean over leads):")
    print(df.pivot_table(index="case", columns="variant", values="auc").round(3).to_string())
    print("\nAUC difference vs equal, by lead (mean over cases):")
    p = df.pivot_table(index="lead", columns="variant", values="auc")
    print(p.sub(p["equal"], axis=0).round(3).to_string())


if __name__ == "__main__":
    main()

"""Tune the CI lift / sustain layers on training cases, then test on a held-out case.

    python ci_tune.py --train 2025-05-16:7,6,5 2025-06-16:5,3,1 2025-04-27:5,3,1 --test 2026-04-13:5,3,1

Raw per-member quantities (max over the 18-06Z valid times) are computed once, then a
small grid of ramp settings is applied:
  ascent ramp (Pa/s), convergence ramp (s^-1), weight of P_model in P_CI,
  sustain ramps (mid-level RH %, CAPE J/kg), and whether S multiplies the final.
Selection: highest mean AUC over training case-leads, ties broken by top-5% capture.
"""
import argparse
import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

import ci
import run_case
import scoring
import verify_case as vc
from config import CASES_DIR, MODELS, case_setup, fh_available

ASC = [(0.02, 0.15), (0.05, 0.25), (0.08, 0.35)]
CONV = [(0.5e-5, 3e-5), (1e-5, 5e-5), (2e-5, 8e-5)]
WMODEL = [0.6, 0.8, 1.0]
SUS = [((30, 65), (1000, 2500)), ((45, 75), (2000, 3500))]
USE_S = [False, True]


def raw_member(model, init, mem, valid):
    main, extra = {}, {}
    for v in valid:
        fh = int((v - init).total_seconds() // 3600)
        if fh_available(model, init, fh):
            a, b = ci.load(model, init, fh, mem), ci.load(f"{model}_x", init, fh, mem)
            if a is not None and b is not None:
                main[v], extra[v] = a, b
    if len(main) < 2:
        return None
    r = ci.member_layers(model, init, mem, valid)  # reuse the fired / precip logic
    if r is None:
        return None
    fired = r[0]
    asc, conv, cin, rh, cape = [], [], [], [], []
    for v in sorted(main):
        f, x = main[v], extra[v]
        asc.append(-scoring.gauss(0.5 * (f["w500"] + f["w700"]), 150))
        conv.append(np.maximum(scoring.gauss(ci.convergence(f["u10"], f["v10"]), 100),
                               scoring.gauss(ci.convergence(x["u850"], x["v850"]), 100)))
        c = f.get("cin90", f.get("cin180"))
        cin.append(np.full_like(asc[-1], np.nan) if c is None else np.nan_to_num(c))
        rh.append(0.5 * (x["rh700"] + x["rh500"]))
        cape.append(np.nan_to_num(f.get("cape90", f.get("cape180", f.get("capemu")))))
    return np.stack([fired, np.max(asc, 0), np.max(conv, 0), np.max(cin, 0), np.max(rh, 0), np.max(cape, 0)]).astype(np.float32)


def raw_case(case, leads):
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    valid, pools = case_setup(conv_day, leads)
    out = {}
    for lead in leads:
        out[lead] = {}
        for m in MODELS:
            mem = [raw_member(m, i, k, valid) for i in pools[lead] for k in run_case.members_of(m)]
            mem = [x for x in mem if x is not None]
            out[lead][m] = np.stack(mem) if mem else None
        print(f"  raw {case} D{lead}", flush=True)
    return out


def apply(raw, asc_r, conv_r, wm, sus_r, use_s):
    per = {}
    for m, R in raw.items():
        if R is None:
            per[m] = None
            continue
        fired, asc, conv, cin, rh, cape = (R[:, k] for k in range(6))
        a = ci.ramp(asc, *asc_r)
        cin_ok = np.where(np.isnan(cin), 1.0, ci.ramp(cin, -150, -25))
        lift = np.maximum(a, ci.ramp(conv, *conv_r)) * np.maximum(cin_ok, 0.7 * a)
        s = 0.4 + 0.6 * np.maximum(ci.ramp(rh, *sus_r[0]), ci.ramp(cape, *sus_r[1]))
        p_ci = wm * fired.mean(0) + (1 - wm) * lift.mean(0)
        per[m] = (p_ci, s.mean(0))
    p_ci = scoring.blend({m: v[0] if v else None for m, v in per.items()})
    s = scoring.blend({m: v[1] if v else None for m, v in per.items()})
    return p_ci * ((0.5 + 0.5 * s) if use_s else 1.0)


def case_eval(case, lead_fields, env_key="v2"):
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    res = dict(np.load(os.path.join(CASES_DIR, case, "output", "results.npz")))
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    tor = rep.type.str.startswith("torn").values
    ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
    rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
    label = vc.report_mask(rep)
    rows = []
    for lead, f in lead_fields.items():
        env = res[f"{env_key}|{lead}|blend|day"]
        for name, fld in (("env", env), ("final", env * f)):
            t5 = vc.top_area_mask(fld, 0.05)[ri, rj]
            rows.append(dict(case=case, lead=lead, what=name, auc=vc.auc(fld, label, vc.W),
                             top5_tor=t5[tor].mean() if tor.any() else np.nan, top5_all=t5.mean()))
    return rows


def parse(specs):
    return [(s.split(":")[0], [int(x) for x in s.split(":")[1].split(",")]) for s in specs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    a = ap.parse_args()
    train, test = parse(a.train), parse(a.test)
    raws = {c: raw_case(c, l) for c, l in train + test}
    grid = list(itertools.product(ASC, CONV, WMODEL, SUS, USE_S))
    scores = []
    for g in grid:
        rows = []
        for c, leads in train:
            rows += case_eval(c, {ld: apply(raws[c][ld], *g) for ld in leads})
        d = pd.DataFrame(rows)
        fin = d[d.what == "final"]
        scores.append(dict(asc=g[0], conv=g[1], wmodel=g[2], sus=g[3], use_s=g[4],
                           auc=fin.auc.mean(), top5_tor=fin.top5_tor.mean(), top5_all=fin.top5_all.mean()))
    s = pd.DataFrame(scores).sort_values(["auc", "top5_all"], ascending=False)
    base = pd.DataFrame([r for c, l in train for r in case_eval(c, {ld: np.ones((71, 141)) for ld in l})])
    base = base[base.what == "env"]
    pd.set_option("display.width", 220)
    print(f"\nTRAIN env-only baseline: AUC {base.auc.mean():.3f}  top5 tor {base.top5_tor.mean():.3f}  top5 all {base.top5_all.mean():.3f}")
    print("Top 8 settings on training cases:")
    print(s.head(8).round(3).to_string(index=False))
    s.to_csv(os.path.join(CASES_DIR, "ci_tune_grid.csv"), index=False)
    best = s.iloc[0]
    g = (best.asc, best.conv, best.wmodel, best.sus, best.use_s)
    orig = (ASC[0], CONV[0], 0.6, SUS[0], False)
    for c, leads in test:
        for label, gg in (("original", orig), ("tuned", g)):
            d = pd.DataFrame(case_eval(c, {ld: apply(raws[c][ld], *gg) for ld in leads}))
            for what in ("env", "final"):
                x = d[d.what == what]
                print(f"TEST {c} [{label:8s}] {what:5s}: AUC {x.auc.mean():.3f}  top5 tor {x.top5_tor.mean():.3f}  top5 all {x.top5_all.mean():.3f}")


if __name__ == "__main__":
    main()

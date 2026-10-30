"""Score a cached case and write result arrays + figures.

    python run_case.py 2025-05-16 [--leads 7 6 5]

Run download_case.py first. Outputs go to cases/<date>/output/.
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np

from config import CAM_MODELS, CASES_DIR, EPS_MEMBERS, GEFS_MEMBERS, MODELS, cam_pools, case_setup, fh_available
import fetch
from fetch import cache_path
import scoring

SCHEMES = ("current", "v2", "v3") + scoring.HAZARDS  # v3, tor/hail/wind = experimental (scoring.py)
DIAG_KEYS = ("cape_v2", "shear", "omega", "lr75", "q2", "pwat", "z500")


def members_of(model):
    return {"gefs": GEFS_MEMBERS, "eps": EPS_MEMBERS}.get(model, ["det"])  # deterministic: GFS, HRES, CAMs


def load(model, init, fh, member):
    return fetch.read_cached(model, init, fh, member)  # CAM winds rotated to earth-relative


def height_fall(model, init, mem, t0, t1):
    """12-h 500 hPa height change t0 -> t1 (m; negative = falls) for one member, or None."""
    src = "eps_z" if model == "eps" else model  # EPS geopotential is fetched separately
    z = []
    for t in (t0, t1):
        fh = int((t - init).total_seconds() // 3600)
        f = load(src, init, fh, mem) if fh_available(model, init, fh) else None
        if f is None or "z500" not in f:
            return None
        z.append(f["z500"])
    return z[1] - z[0]


def score_pool(model, inits, valid):
    """Score every member of every run in a pool.

    Returns per scheme: {'vt': {vt: [member fields]}, 'day': [member day-max],
    'cyc': {init: [member day-max]}} plus mean diagnostics at the 00Z valid time.
    """
    out = {s: {"vt": {v: [] for v in valid}, "day": [], "cyc": {}} for s in SCHEMES}
    diag_sum, diag_n = {}, {}
    v00 = valid[2]
    fhof = lambda init, v: int((v - init).total_seconds() // 3600)
    for init in inits:
        for mem in members_of(model):
            dz12 = height_fall(model, init, mem, valid[0], valid[-1])
            per_vt = {s: [] for s in SCHEMES}
            for v in valid:
                fh = fhof(init, v)
                if not fh_available(model, init, fh):
                    continue
                f = load(model, init, fh, mem)
                if f is None:
                    continue
                if model in MODELS:  # 850 hPa wind (v3, hazards) and 700/500 hPa RH (wind); v2 uses neither
                    x = load(f"{model}_x", init, fh, mem)
                    if x is not None:
                        f.update({k: x[k] for k in ("u850", "v850", "rh700", "rh500") if k in x})
                ing = scoring.ingredients(model, f)
                sc = {"current": scoring.score_current(ing, fh), "v2": scoring.score_v2(ing, fh),
                      "v3": scoring.score_v3(ing, fh, dz12)}
                sc.update(scoring.hazard_scores(ing, f, sc["v2"], dz12))
                for s in SCHEMES:
                    out[s]["vt"][v].append(sc[s])
                    per_vt[s].append(sc[s])
                if v == v00:
                    for k in DIAG_KEYS:
                        if ing[k] is not None:
                            diag_sum[k] = diag_sum.get(k, 0) + np.nan_to_num(ing[k])
                            diag_n[k] = diag_n.get(k, 0) + 1
            for s in SCHEMES:
                if per_vt[s]:
                    dmax = np.max(per_vt[s], axis=0)
                    out[s]["day"].append(dmax)
                    out[s]["cyc"].setdefault(init, []).append(dmax)
    diag = {k: v / diag_n[k] for k, v in diag_sum.items()}
    return out, diag


def combine(lst, scheme, model):
    if not lst:
        return None
    return scoring.pool_combine(np.stack(lst), scheme, model not in ("gefs", "eps"))


def compute(conv_day, leads, log=print):
    valid, pools = case_setup(conv_day, leads)
    res = {}
    for lead, inits in pools.items():
        log(f"Day {lead}: cycles {inits[0]:%m/%d %HZ} .. {inits[-1]:%m/%d %HZ}")
        by_model = {s: {"day": {}, "vt": {v: {} for v in valid}, "cyc": {i: {} for i in inits}} for s in SCHEMES}
        cams = cam_pools(conv_day, lead)
        for model in MODELS + [m for m in CAM_MODELS if m in cams]:
            t0 = time.time()
            minits = cams.get(model, inits)
            out, diag = score_pool(model, minits, valid)
            if model in cams and not out["v2"]["day"]:
                log(f"  {model:5s}    0 member-runs (not downloaded)")
                continue
            for k, v in diag.items():
                res[f"diag|{lead}|{model}|{k}"] = v
            for s in SCHEMES:
                d = combine(out[s]["day"], s, model)
                by_model[s]["day"][model] = d
                res[f"n|{s}|{lead}|{model}"] = np.array(len(out[s]["day"]))
                for v in valid:
                    by_model[s]["vt"][v][model] = combine(out[s]["vt"][v], s, model)
                for i in inits:
                    by_model[s]["cyc"][i][model] = combine(out[s]["cyc"].get(i, []), s, model)
            log(f"  {model:5s} {len(out['v2']['day']):4d} member-runs  {time.time() - t0:5.1f}s")
        for s in SCHEMES:
            def put(tag, fields):
                for m, f in fields.items():
                    if f is not None:
                        res[f"{s}|{lead}|{m}|{tag}"] = f
                b = scoring.blend(fields, lead)
                if b is not None:
                    res[f"{s}|{lead}|blend|{tag}"] = b
            put("day", by_model[s]["day"])
            for v in valid:
                put(f"vt{v:%Y%m%d%H}", by_model[s]["vt"][v])
            for i in inits:
                put(f"cyc{i:%Y%m%d%H}", by_model[s]["cyc"][i])
    return res, valid, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[7, 6, 5])
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved results.npz")
    a = ap.parse_args()
    conv_day = datetime.strptime(a.date, "%Y-%m-%d")
    outdir = os.path.join(CASES_DIR, a.date, "output")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    if a.replot:
        res = dict(np.load(os.path.join(outdir, "results.npz")))
        valid, pools = case_setup(conv_day, a.leads)
    else:
        res, valid, pools = compute(conv_day, a.leads)
        np.savez_compressed(os.path.join(outdir, "results.npz"), **res)
        print(f"scored in {time.time() - t0:.0f}s -> {outdir}")
    if not a.no_plots:
        import plots
        t1 = time.time()
        plots.make_all(a.date, res, valid, pools, outdir)
        print(f"plotted in {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()

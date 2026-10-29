"""Does using the full sounding change the outcome? (deterministic GFS + ECMWF HRES)

    python snd_experiment.py 2025-06-16 --leads 5 4 3 2 1

1. Computes sounding features for every cached gfs_snd / hres_snd step (cached to
   <cache>/features/...).
2. Scores them with sounding-based variants and compares, for the same models, with the
   single-level proxy scores already in results.npz -- then also inside the full blend
   (GEFS/EPS unchanged).
Variants (raw sounding quantities only, no composite indices):
  snd_v2   -- v2 structure with sounding ingredients: own ML CAPE/CIN, true 0-6 km AGL shear,
              ML mixing ratio
  snd_msl  -- as snd_v2 with the msl profile constants
  snd_full -- snd_msl plus whole-profile information: low-level hodograph (0-3 km SRH),
              mid-level dryness at low CAPE, an elevated (MU) path for night-time,
              and an MCS/wind path using DCAPE
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from config import CACHE_DIR, CASES_DIR, case_setup, fh_available

VARIANTS = ("snd_v2", "snd_msl", "snd_full")
LIFT_X = [-0.20, -0.08, -0.03, 0.00, 0.05, 0.20]


def feat_path(model, init, fh):
    return os.path.join(CACHE_DIR, "features", model, f"{init:%Y%m%d%H}", f"f{fh:03d}.npz")


def make_features(args):
    model, init, fh = args
    import fetch
    import sounding
    out = feat_path(model, init, fh)
    if os.path.exists(out):
        return out
    src = fetch.cache_path(model, init, fh, "det")
    if not os.path.exists(src):
        return None
    with np.load(src) as z:
        f = {k: z[k] for k in z.files}
    F = sounding.features(model, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **F)
    return out


def ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def g25(x, km):
    s = km / (0.25 * 111.2)
    return gaussian_filter(np.nan_to_num(x), sigma=(s, s / np.cos(np.deg2rad(37.5))), mode="nearest")


def score(F, variant, valid_hour):
    msl = variant != "snd_v2"
    shear_hi, lneu, mr = (30.0, 0.45, (5.0, 10.0)) if msl else (20.0, 0.65, (7.0, 12.0))
    lift_y = [1.10, 1.00, 0.80, lneu, lneu * 0.75, 0.20]
    lift = np.interp(np.clip(g25(F["omega"], 150), -0.3, 0.3), LIFT_X, lift_y)
    lr_mult = 1 + 0.5 * ramp(F["lr75"], 6.0, 7.5)
    moist = 0.5 + 0.5 * ramp(F["ml_r"], *mr)
    inst = np.clip(np.clip(np.nan_to_num(F["ml_cape"]) / 1500.0, 0, 1) ** 0.6 * lr_mult, 0, 1)
    cin = ramp(200 + np.nan_to_num(F["ml_cin"]), 0, 150)
    shear = ramp(F["shr06"], 10, shear_hi)
    raw = inst * shear * lift * cin * moist
    if variant == "snd_full":
        # low-level hodograph: organisation / tornado potential (bounded modifier)
        raw = raw * (0.85 + 0.30 * ramp(F["srh03"], 50, 250))
        # mid-level dryness only hurts weak-CAPE updrafts (entrainment)
        dry = np.where(F["ml_cape"] < 1000, 0.75 + 0.25 * ramp(F["rh75"], 25, 55), 1.0)
        raw = raw * dry
        # elevated path after dark: MU parcel when the surface layer is capped
        if valid_hour in (3, 6):
            inst_mu = np.clip(np.clip(np.nan_to_num(F["mu_cape"]) / 2000.0, 0, 1) ** 0.6 * lr_mult, 0, 1)
            cin_mu = ramp(200 + np.nan_to_num(F["mu_cin"]), 0, 150)
            raw = np.maximum(raw, 0.7 * inst_mu * shear * lift * cin_mu * moist)
        # MCS / damaging-wind path: weak deep shear is fine if downdraft potential is large
        wind = 0.6 * inst * ramp(F["shr06"], 5, 12) * lift * cin * moist * (0.6 + 0.4 * ramp(F["dcape"], 600, 1300))
        raw = np.maximum(raw, wind)
    return g25(np.clip(raw, 0, 1), 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[5, 4, 3, 2, 1])
    a = ap.parse_args()
    case, conv_day = a.date, datetime.strptime(a.date, "%Y-%m-%d")
    valid, pools = case_setup(conv_day, a.leads)
    steps = sorted({(m, i, int((v - i).total_seconds() // 3600)) for m in ("gfs_snd", "hres_snd")
                    for p in pools.values() for i in p for v in valid
                    if fh_available(m, i, int((v - i).total_seconds() // 3600))})
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        done = list(ex.map(make_features, steps))
    print(f"features for {sum(d is not None for d in done)}/{len(steps)} steps in {time.time() - t0:.0f}s")

    import scoring
    import verify_case as vc
    outdir = os.path.join(CASES_DIR, case, "output")
    res = dict(np.load(os.path.join(outdir, "results.npz")))
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    tor, wind = rep.type.str.startswith("torn").values, rep.type.str.startswith("wind").values
    ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
    rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
    label = vc.report_mask(rep)

    def metrics(f):
        t5 = vc.top_area_mask(f, 0.05)[ri, rj]
        return dict(auc=vc.auc(f, label, vc.W), top5_tor=t5[tor].mean(), top5_wind=t5[wind].mean(), top5_all=t5.mean())

    rows, fields = [], {}
    for lead in a.leads:
        det = {}
        for m in ("gfs", "hres"):
            per_var = {v: [] for v in VARIANTS}
            for init in pools[lead]:
                runs = {v: [] for v in VARIANTS}
                for vt in valid:
                    fh = int((vt - init).total_seconds() // 3600)
                    p = feat_path(f"{m}_snd", init, fh)
                    if not (fh_available(m, init, fh) and os.path.exists(p)):
                        continue
                    with np.load(p) as z:
                        F = {k: z[k] for k in z.files}
                    for v in VARIANTS:
                        runs[v].append(score(F, v, vt.hour))
                for v in VARIANTS:
                    if runs[v]:
                        per_var[v].append(np.max(runs[v], axis=0))
            for v in VARIANTS:
                det[(m, v)] = np.median(per_var[v], axis=0)[::2, ::2] if per_var[v] else None
        w = scoring.BLEND_WEIGHTS
        combos = {"proxy v2": {m: res.get(f"v2|{lead}|{m}|day") for m in ("gfs", "hres")}}
        for v in VARIANTS:
            combos[v] = {m: det[(m, v)] for m in ("gfs", "hres")}
        for name, d in combos.items():
            detb = (w["gfs"] * d["gfs"] + w["hres"] * d["hres"]) / (w["gfs"] + w["hres"])
            full = sum(w[m] * (d[m] if m in d else res[f"v2|{lead}|{m}|day"]) for m in w) / sum(w.values())
            for scope, f in (("GFS+HRES only", detb), ("full blend (GEFS/EPS proxy)", full)):
                rows.append(dict(lead=lead, scope=scope, method=name, **metrics(f)))
                fields[f"{scope}|{name}|{lead}"] = f
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "snd_experiment.csv"), index=False)
    np.savez_compressed(os.path.join(outdir, "snd_experiment_fields.npz"),
                        **{k.replace("|", "__"): v for k, v in fields.items()})
    pd.set_option("display.width", 200)
    print(df.groupby(["scope", "method"])[["auc", "top5_tor", "top5_wind", "top5_all"]].mean().round(3).to_string())
    print()
    print(df.pivot_table(index=["scope", "lead"], columns="method", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()

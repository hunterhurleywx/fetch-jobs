"""Convective-initiation (CI) and sustained-CI layers, combined with the environment score.

    python ci.py 2025-06-16 --leads 8 7 6 5 4 3 2 1

Layers (per member of every run in the lead's pool, then pooled and blended like the
environment score):
  P_model  -- did the model's own convection fire near the point? Fraction of members/runs
              with 18Z-06Z convective precipitation >= 1 mm (GFS/GEFS ACPCP) or total
              precipitation >= 2 mm with CAPE >= 250 J/kg (ECMWF tp; cp is not in open data
              before Oct 2026), within one 0.5-deg grid cell (~55 km).
  P_lift   -- physical lift able to start storms: max of synoptic ascent (700/500 hPa w),
              10 m and 850 hPa convergence, times a CIN-removal term (strong ascent can erode
              a cap). Max over the 18-06Z valid times.
  S        -- sustained CI: mid-level moisture (700/500 hPa RH; dry air entrains weak
              updrafts, Marquis et al. 2023) unless CAPE is large enough to overwhelm it.
  P_CI     = 0.6 P_model + 0.4 P_lift
Final candidates (all uncalibrated; compared against reports):
  env              -- environment score (v2 blend) alone
  env*ci           -- env x P_CI
  env*(.3+.7ci)    -- env x (0.3 + 0.7 P_CI)
  env*ci*S         -- env x (0.3 + 0.7 P_CI) x (0.5 + 0.5 S)
  sqrt(env*ci)     -- geometric mean of env and P_CI
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

import run_case
import scoring
from config import CASES_DIR, GRID_LATS, MODELS, case_setup, fh_available
from fetch import cache_path

DY = 0.5 * 111.2e3
DX = DY * np.cos(np.deg2rad(GRID_LATS))[:, None]


def load(model, init, fh, mem):
    p = cache_path(model, init, fh, mem)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def convergence(u, v):
    du = np.gradient(np.nan_to_num(u), axis=1) / DX
    dv = np.gradient(np.nan_to_num(v), axis=0) / DY
    return -(du + dv)


def ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def member_layers(model, init, mem, valid):
    """P_model indicator, lift and sustain maps for one member of one run (None if incomplete)."""
    main, extra = {}, {}
    for v in valid:
        fh = int((v - init).total_seconds() // 3600)
        if not fh_available(model, init, fh):
            continue
        a, b = load(model, init, fh, mem), load(f"{model}_x", init, fh, mem)
        if a is not None and b is not None:
            main[v], extra[v] = a, b
    if len(main) < 2:
        return None
    vt = sorted(main)
    # --- precipitation in the 18Z-06Z window
    first, last = vt[0], vt[-1]
    cape_any = np.max([np.nan_to_num(main[v].get("cape90", main[v].get("cape180", main[v].get("capemu"))))
                       for v in vt], axis=0)
    if model == "gfs" and "acpcp" in extra[first] and "acpcp" in extra[last]:
        pr = extra[last]["acpcp"] - extra[first]["acpcp"]
        fired = pr >= 1.0
    elif model == "gefs":
        buckets = [extra[v]["acpcpb"] for v in vt if v.hour in (0, 6) and "acpcpb" in extra[v]]
        if not buckets:
            return None
        fired = np.sum(buckets, axis=0) >= 1.0
    elif model in ("hres", "eps") and "cp" in extra[first] and "cp" in extra[last]:
        fired = 1000.0 * (extra[last]["cp"] - extra[first]["cp"]) >= 1.0  # convective precip (since Oct 2026)
    elif model in ("hres", "eps") and "tp" in extra[first] and "tp" in extra[last]:
        pr = 1000.0 * (extra[last]["tp"] - extra[first]["tp"])
        fired = (pr >= 2.0) & (cape_any >= 250)
    else:
        return None
    p_model = maximum_filter(fired.astype(float), size=3)
    # --- physical lift and sustainment
    lift, sus = [], []
    for v in vt:
        f, x = main[v], extra[v]
        asc = ramp(-scoring.gauss(0.5 * (f["w500"] + f["w700"]), 150), 0.02, 0.15)
        c10 = ramp(scoring.gauss(convergence(f["u10"], f["v10"]), 100), 0.5e-5, 3e-5)
        c85 = ramp(scoring.gauss(convergence(x["u850"], x["v850"]), 100), 0.5e-5, 3e-5)
        cin = f.get("cin90", f.get("cin180"))
        cin_ok = np.ones_like(asc) if cin is None else ramp(np.nan_to_num(cin), -150, -25)
        lift.append(np.maximum(asc, np.maximum(c10, c85)) * np.maximum(cin_ok, 0.7 * asc))
        cape = np.nan_to_num(f.get("cape90", f.get("cape180", f.get("capemu"))))
        rh = 0.5 * (x["rh700"] + x["rh500"])
        sus.append(0.4 + 0.6 * np.maximum(ramp(rh, 30, 65), ramp(cape, 1000, 2500)))
    return p_model, np.max(lift, axis=0), np.max(sus, axis=0)


def layers_for_lead(inits, valid):
    out = {}
    for m in MODELS:
        acc = [[], [], []]
        for init in inits:
            for mem in run_case.members_of(m):
                r = member_layers(m, init, mem, valid)
                if r is not None:
                    for k in range(3):
                        acc[k].append(r[k])
        out[m] = [np.mean(a, axis=0) if a else None for a in acc]
    blend = []
    for k in range(3):
        blend.append(scoring.blend({m: out[m][k] for m in MODELS}))
    return out, blend


def finals(env, p_model, p_lift, sus):
    p_ci = 0.6 * p_model + 0.4 * p_lift
    return p_ci, {
        "env": env,
        "env*ci": env * p_ci,
        "env*(.3+.7ci)": env * (0.3 + 0.7 * p_ci),
        "env*ci*S": env * (0.3 + 0.7 * p_ci) * (0.5 + 0.5 * sus),
        "sqrt(env*ci)": np.sqrt(env * p_ci),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[8, 7, 6, 5, 4, 3, 2, 1])
    ap.add_argument("--env", default="v2", help="environment scheme key in results.npz")
    a = ap.parse_args()
    import verify_case as vc
    case, conv_day = a.date, datetime.strptime(a.date, "%Y-%m-%d")
    outdir = os.path.join(CASES_DIR, case, "output")
    res = dict(np.load(os.path.join(outdir, "results.npz")))
    valid, pools = case_setup(conv_day, a.leads)
    rpath = os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv")
    have_reports = os.path.exists(rpath)  # real-time cases: no reports yet
    if have_reports:
        rep = pd.read_csv(rpath)
        rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
        tor, wind = rep.type.str.startswith("torn").values, rep.type.str.startswith("wind").values
        ri = np.rint((rep.lat.values - 20) / 0.5).astype(int)
        rj = np.rint((rep.lon.values + 130) / 0.5).astype(int)
        label = vc.report_mask(rep)
    rows, save = [], {}
    for lead in a.leads:
        per_model, (pm, pl, s) = layers_for_lead(pools[lead], valid)
        env = res[f"{a.env}|{lead}|blend|day"]
        p_ci, fin = finals(env, pm, pl, s)
        save.update({f"{lead}|p_model": pm, f"{lead}|p_lift": pl, f"{lead}|sustain": s, f"{lead}|p_ci": p_ci})
        for name, f in fin.items():
            save[f"{lead}|{name}"] = f
            if not have_reports:
                continue
            t5 = vc.top_area_mask(f, 0.05)[ri, rj]
            rows.append(dict(lead=lead, method=name, auc=vc.auc(f, label, vc.W),
                             top5_tor=t5[tor].mean() if tor.any() else np.nan,
                             top5_wind=t5[wind].mean() if wind.any() else np.nan, top5_all=t5.mean()))
        print(f"D{lead} done", flush=True)
    np.savez_compressed(os.path.join(outdir, "ci_layers.npz"), **{k.replace("|", "__"): v for k, v in save.items()})
    if not have_reports:
        print("no reports yet: layers saved, verification skipped")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "ci_experiment.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("method")[["auc", "top5_tor", "top5_wind", "top5_all"]].mean().round(3).to_string())
    print(df.pivot_table(index="lead", columns="method", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()


def plot_layers(case, leads, view="central"):
    """Environment, CI layers and the final product side by side for each lead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    outdir = os.path.join(CASES_DIR, case, "output")
    z = np.load(os.path.join(outdir, "ci_layers.npz"))
    rep = plots.load_reports(case, conv_day)
    panels = [("env", "Favourable environment (v2)"), ("p_model", "P(model convection fires)"),
              ("p_lift", "Physical lift / CIN removal"), ("sustain", "Sustained CI (mid-level RH / CAPE)"),
              ("p_ci", "P(CI) = 0.6 model + 0.4 lift"), ("env*ci", "FINAL: environment x P(CI)")]
    for lead in leads:
        fig, axes = plt.subplots(2, 3, figsize=(19, 11), subplot_kw={"projection": plots.PROJ})
        polys, iscat = plots.spc_lead_polys(case, conv_day, lead)
        for ax, (k, title) in zip(axes.flat, panels):
            plots.base_ax(ax, view)
            plots.draw_prob(ax, z[f"{lead}__{k}"])
            plots.draw_spc(ax, polys, iscat)
            plots.draw_reports(ax, rep, 4)
            ax.set_title(title, fontsize=11)
        cf = axes.flat[0].collections[-1] if axes.flat[0].collections else None
        fig.suptitle(f"{case} Day {lead}: environment x convective initiation (uncalibrated)", fontsize=13)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.04, hspace=0.12)
        fig.savefig(os.path.join(outdir, f"ci_layers_D{lead}_{view}.png"), dpi=100)
        plt.close(fig)

"""Skew-T + hodograph at report clusters and the score maximum, from cached model columns.

    python plot_soundings.py 2025-06-16 --model gfs_snd --init 2025061600 --valid 2025061700

Writes cases/<date>/output/soundings/<model>_<init>_v<valid>_<n>.png. Profiles, parcel
(mixed-layer) path and features come from sounding.py, so the plots show exactly what
the scoring sees.
"""
import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metpy.plots import Hodograph, SkewT
from metpy.units import units

import fetch
import plots
import sounding as snd
from config import CASES_DIR, SND_LATS, SND_LONS
from diagnose_case import clusters

FEATS = [("ml_cape", "MLCAPE", "J/kg"), ("ml_cin", "MLCIN", "J/kg"), ("mu_cape", "MUCAPE", "J/kg"),
         ("ml_lcl", "ML LCL", "m"), ("ml_lfc", "ML LFC", "m"), ("ml_cape3", "ML 0-3 km CAPE", "J/kg"),
         ("ml_cape_hgz", "ML CAPE -10..-30C", "J/kg"), ("dcape", "DCAPE", "J/kg"),
         ("lr03", "LR 0-3 km", "C/km"), ("lr75", "LR 700-500", "C/km"), ("eml_lr", "max 2-km LR 1-6 km", "C/km"),
         ("cap_dt", "cap strength", "K"), ("ml_r", "ML mixing ratio", "g/kg"), ("pw", "PW", "mm"),
         ("moist_depth", "moist depth (RH≥60%)", "m"), ("rh75", "RH 700-500", "%"),
         ("shr01", "shear 0-1 km", "m/s"), ("shr03", "shear 0-3 km", "m/s"), ("shr06", "shear 0-6 km", "m/s"),
         ("srh01", "SRH 0-1 km", "m2/s2"), ("srh03", "SRH 0-3 km", "m2/s2"), ("srw01", "SR inflow 0-1 km", "m/s"),
         ("crit_angle", "critical angle", "deg"), ("veer03", "veering 0-3 km", "deg"),
         ("backing15", "backing 1-5 km", "deg"), ("llj_max", "LLJ max 0-2 km", "m/s"), ("mean_wind06", "mean wind 0-6", "m/s")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--model", default="gfs_snd")
    ap.add_argument("--init", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--score-field", default=None, help="optional npz key in snd_experiment_fields for the max")
    a = ap.parse_args()
    case, conv_day = a.date, datetime.strptime(a.date, "%Y-%m-%d")
    init, valid = datetime.strptime(a.init, "%Y%m%d%H"), datetime.strptime(a.valid, "%Y%m%d%H")
    fh = int((valid - init).total_seconds() // 3600)
    with np.load(fetch.cache_path(a.model, init, fh, "det")) as z:
        f = {k: z[k] for k in z.files}
    pr = snd.build_profiles(a.model, f)
    F = snd.features(a.model, f)
    flat = np.flatnonzero(snd.BOX_MASK)
    rep = plots.load_reports(case, conv_day)
    pts = [(lab, la, lo) for lab, la, lo, n, _ in clusters(rep)[:6]]
    if a.score_field:
        fld = np.load(os.path.join(CASES_DIR, case, "output", "snd_experiment_fields.npz"))[a.score_field]
        j = np.unravel_index(np.nanargmax(fld), fld.shape)
        pts.insert(0, ("score max", 20 + 0.5 * j[0], -130 + 0.5 * j[1]))
    out = os.path.join(CASES_DIR, case, "output", "soundings")
    os.makedirs(out, exist_ok=True)
    for n, (lab, la, lo) in enumerate(pts):
        i, j = int(round((la - SND_LATS[0]) / 0.25)), int(round((lo - SND_LONS[0]) / 0.25))
        col = np.searchsorted(flat, i * SND_LONS.size + j)
        if col >= flat.size or flat[col] != i * SND_LONS.size + j:
            continue
        T, P, R, U, V = (pr[k][col] for k in ("T", "P", "R", "U", "V"))
        e = R * P / (snd.EPS + R)
        a_ = np.log(np.maximum(e, 1e-3) / 6.112)
        Td = 243.5 * a_ / (17.67 - a_)
        top = P > 100
        # mixed-layer parcel path (same construction as the scoring)
        ml = P >= P[0] - 100
        th = T * (1000.0 / P) ** snd.KAPPA
        t_ml = (th[ml].mean()) * (P[0] / 1000.0) ** snd.KAPPA
        r_ml = R[ml].mean()
        tvp, _ = snd.lift(T[None], P[None], np.array([0]), np.array([t_ml]), np.array([r_ml]), np.array([P[0]]))
        tp = tvp[0] * (1 + r_ml) / (1 + r_ml / snd.EPS)  # approx. back to temperature for display
        fig = plt.figure(figsize=(14, 9))
        skew = SkewT(fig, rect=(0.05, 0.07, 0.55, 0.86), rotation=45)
        p_u = P[top] * units.hPa
        skew.plot(p_u, (T[top] - 273.15) * units.degC, "r", lw=2)
        skew.plot(p_u, Td[top] * units.degC, "g", lw=2)
        skew.plot(p_u, (tp[top] - 273.15) * units.degC, "k--", lw=1.2, label="ML parcel")
        b = np.arange(0, top.sum(), 5)
        skew.plot_barbs(p_u[b], (U[top][b] * units("m/s")).to("kt"), (V[top][b] * units("m/s")).to("kt"))
        skew.plot_dry_adiabats(alpha=0.2)
        skew.plot_moist_adiabats(alpha=0.2)
        skew.plot_mixing_lines(alpha=0.2)
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 50)
        skew.ax.legend(loc="upper left")
        hax = fig.add_axes((0.62, 0.55, 0.35, 0.38))
        h = Hodograph(hax, component_range=40)
        h.add_grid(increment=10)
        km = snd.H / 1000.0
        for lo_, hi_, c in ((0, 1, "m"), (1, 3, "r"), (3, 6, "g"), (6, 9, "y")):
            s = (km >= lo_) & (km <= hi_)
            hax.plot(U[s], V[s], color=c, lw=2, label=f"{lo_}-{hi_} km")
        hax.plot(F["rm_u"][i, j], F["rm_v"][i, j], "ko", ms=6, label="Bunkers RM")
        hax.legend(fontsize=7, loc="lower right")
        txt = "\n".join(f"{nm:<22s}{F[k][i, j]:8.0f} {un}" if abs(F[k][i, j]) >= 20 or np.isnan(F[k][i, j])
                        else f"{nm:<22s}{F[k][i, j]:8.1f} {un}" for k, nm, un in FEATS)
        fig.text(0.62, 0.05, txt, family="monospace", fontsize=7.2, va="bottom")
        fig.suptitle(f"{case} | {a.model.replace('_snd', '').upper()} {init:%d/%HZ} f{fh:03d} valid {valid:%d/%HZ} | "
                     f"{lab} ({la:.2f}N {lo:.2f})", fontsize=12)
        fig.savefig(os.path.join(out, f"{a.model}_{init:%Y%m%d%H}_v{valid:%Y%m%d%H}_{n}.png"), dpi=100)
        plt.close(fig)
        print("saved", lab)


if __name__ == "__main__":
    main()

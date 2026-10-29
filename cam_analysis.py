"""Run-by-run HRRR comparison for one area: did each run make storms there, and what cap did it have?

    python cam_analysis.py 2026-04-13 --box 37.5 39.8 -97.2 -94.0 --point TOP --obs KTOP_202604140000

Writes to cases/<date>/output/cam/:
  hrrr_runs_box.csv        per run x valid hour: max REFC / UH 2-5 km / MAXUVV in the box, % area REFC >= 40 dBZ
  hrrr_runs_point.csv      per run x valid hour at the point: SB/ML CAPE and CIN, SRH 0-1 km, 2 m T/Td
  hrrr_snd_<point>.csv     per run at sounding hours: T850, T700, strongest inversion, SB/ML CAPE/CIN from the profile
  hrrr_refc_<hh>z.png      REFC (fill) + UH >= 75 m2/s2 (contour) for every run at one valid hour, reports overlaid
  hrrr_uhmax.png           run-total max UH over the valid window, every run
  hrrr_cap_<point>.png     skew-T of the cap layer at the point: each run vs the observed RAOB
"""
import argparse
import glob
import json
import os
import zipfile
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from metpy.plots import SkewT
from metpy.units import units

from config import CACHE_DIR, CASES_DIR

HDIR = os.path.join(CACHE_DIR, "hrrr")
REFC_LEVELS = [20, 30, 40, 50, 60, 75]
REFC_COLORS = ["#c6e8c0", "#6cc165", "#f2d74e", "#ef8a3b", "#d7301f", "#7a0177"]


def runs_for(day):
    out = []
    for d in sorted(glob.glob(os.path.join(HDIR, f"{day:%Y%m%d}??"))) + sorted(glob.glob(os.path.join(HDIR, f"{day - timedelta(days=1):%Y%m%d}??"))):
        out.append(datetime.strptime(os.path.basename(d), "%Y%m%d%H"))
    return sorted(set(out))


def load(init, fh, kind):
    p = os.path.join(HDIR, f"{init:%Y%m%d%H}", f"{kind}_f{fh:02d}.npz")
    if not os.path.exists(p):
        return None
    try:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):  # still being written
        return None


def profile_stats(snd, sfc, point):
    """Cap and parcel numbers from the HRRR pressure-level profile plus 2 m / surface-pressure values."""
    ps = float(sfc[f"pt_{point}_psfc"]) / 100
    lev = snd["levels"]
    T = snd[f"{point}_tmp"] - 273.15
    Td = snd[f"{point}_dpt"] - 273.15
    m = (lev < ps - 1) & np.isfinite(T) & np.isfinite(Td)
    P = np.concatenate([[ps], lev[m]])
    T = np.concatenate([[float(sfc[f"pt_{point}_t2"]) - 273.15], T[m]])
    Td = np.concatenate([[float(sfc[f"pt_{point}_td2"]) - 273.15], Td[m]])
    sb = mpcalc.surface_based_cape_cin(P * units.hPa, T * units.degC, Td * units.degC)
    ml = mpcalc.mixed_layer_cape_cin(P * units.hPa, T * units.degC, Td * units.degC, depth=100 * units.hPa)
    low = P >= 600
    best = (0.0, np.nan, np.nan)
    for i in np.where(low)[0]:
        for j in np.where(low)[0]:
            if j > i and T[j] - T[i] > best[0]:
                best = (T[j] - T[i], P[i], P[j])
    return dict(psfc=ps, t2=T[0], td2=Td[0], t850=np.interp(850, P[::-1], T[::-1]),
                t700=np.interp(700, P[::-1], T[::-1]), inv=best[0], inv_base=best[1], inv_top=best[2],
                sbcape=sb[0].m, sbcin=sb[1].m, mlcape=ml[0].m, mlcin=ml[1].m), (P, T, Td)


def obs_profile(path):
    prof = json.load(open(path))["profiles"][0]["profile"]
    a = np.array([(r["pres"], r["tmpc"], r["dwpc"]) for r in prof
                  if r.get("pres") and r.get("tmpc") is not None and r.get("dwpc") is not None], dtype=float)
    a = a[np.argsort(-a[:, 0])]
    return a[:, 0], a[:, 1], a[:, 2]


def reports(case, day):
    p = os.path.join(CASES_DIR, case, "spc", f"reports_{day:%Y%m%d}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def map_panels(fig_title, panels, grid, box, rep, out, levels=REFC_LEVELS):
    n = len(panels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=39)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), subplot_kw={"projection": proj})
    axs = np.atleast_1d(axs).ravel()
    pad = 2.5
    for ax, (title, refc, uh) in zip(axs, panels):
        ax.set_extent([box[2] - pad, box[3] + pad, box[0] - pad, box[1] + pad], ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="#555")
        if refc is not None:
            ax.contourf(grid["lon"], grid["lat"], refc, levels=levels, colors=REFC_COLORS,
                        transform=ccrs.PlateCarree(), extend="max")
        if uh is not None:
            ax.contour(grid["lon"], grid["lat"], uh, levels=[75], colors="k", linewidths=0.9,
                       transform=ccrs.PlateCarree())
        if rep is not None:
            t = rep[rep.type == "tornado"]
            ax.scatter(t.lon, t.lat, s=14, c="red", edgecolors="k", linewidths=0.4, transform=ccrs.PlateCarree(), zorder=5)
        ax.plot([box[2], box[3], box[3], box[2], box[2]], [box[0], box[0], box[1], box[1], box[0]],
                color="#1f4e9c", lw=0.8, ls="--", transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9)
    for ax in axs[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--box", type=float, nargs=4, default=[37.5, 39.8, -97.2, -94.0], help="lat0 lat1 lon0 lon1")
    ap.add_argument("--point", default="TOP")
    ap.add_argument("--obs", default="KTOP_202604140000")
    ap.add_argument("--maps", type=int, nargs="+", default=[22, 0, 1])
    a = ap.parse_args()
    day = datetime.fromisoformat(a.date)
    out_dir = os.path.join(CASES_DIR, a.date, "output", "cam")
    os.makedirs(out_dir, exist_ok=True)
    with np.load(os.path.join(HDIR, "grid_crop.npz")) as z:
        grid = {"lat": z["lat"], "lon": z["lon"]}
    la0, la1, lo0, lo1 = a.box
    inbox = (grid["lat"] >= la0) & (grid["lat"] <= la1) & (grid["lon"] >= lo0) & (grid["lon"] <= lo1)
    rep = reports(a.date, day)

    box_rows, pt_rows, snd_rows, profiles = [], [], [], {}
    uhmax = {}
    by_vt = {h: [] for h in a.maps}
    for init in runs_for(day):
        acc_uh = None
        for fh in range(1, 49):
            vt = init + timedelta(hours=fh)
            s = load(init, fh, "sfc")
            if s is None:
                continue
            refc, uh = s["refc"], s["uh25"]
            box_rows.append(dict(run=f"{init:%d/%HZ}", valid=f"{vt:%d/%HZ}", fh=fh,
                                 refc_max=float(np.nanmax(refc[inbox])), uh_max=float(np.nanmax(uh[inbox])),
                                 maxuvv=float(np.nanmax(s["maxuvv"][inbox])),
                                 pct_ge40=float(100 * np.mean(refc[inbox] >= 40))))
            pt = a.point
            pt_rows.append(dict(run=f"{init:%d/%HZ}", valid=f"{vt:%d/%HZ}", fh=fh,
                                **{k: float(s[f"pt_{pt}_{k}"]) for k in ("sbcape", "sbcin", "mlcape", "mlcin", "srh1", "t2", "td2")}))
            acc_uh = uh if acc_uh is None else np.fmax(acc_uh, uh)
            if vt.hour in by_vt:
                by_vt[vt.hour].append((f"HRRR {init:%HZ} run (f{fh:02d})", refc, uh))
            snd = load(init, fh, "snd")
            if snd is not None:
                st, prof = profile_stats(snd, s, pt)
                snd_rows.append(dict(run=f"{init:%d/%HZ}", valid=f"{vt:%d/%HZ}", fh=fh, **st))
                if vt.hour == 0:
                    profiles[init] = prof
        if acc_uh is not None:
            uhmax[init] = acc_uh

    pd.DataFrame(box_rows).to_csv(os.path.join(out_dir, "hrrr_runs_box.csv"), index=False)
    pd.DataFrame(pt_rows).to_csv(os.path.join(out_dir, f"hrrr_runs_{a.point}.csv"), index=False)
    pd.DataFrame(snd_rows).to_csv(os.path.join(out_dir, f"hrrr_snd_{a.point}.csv"), index=False)

    for h, panels in by_vt.items():
        map_panels(f"{a.date} HRRR composite reflectivity (fill, dBZ) and UH 2-5 km >= 75 (black), valid {h:02d}Z; "
                   f"red = tornado reports", panels, grid, a.box, rep, os.path.join(out_dir, f"hrrr_refc_{h:02d}z.png"))
    map_panels(f"{a.date} HRRR max UH 2-5 km over each run's 19Z-05Z window (m2/s2: 25/50/75/100/150/200); "
               f"red = tornado reports", [(f"HRRR {i:%HZ} run", u, None) for i, u in uhmax.items()], grid, a.box,
               rep, os.path.join(out_dir, "hrrr_uhmax.png"), levels=[25, 50, 75, 100, 150, 200])

    # cap-layer skew-T: every run's 00Z profile vs the observed RAOB
    obs = glob.glob(os.path.join(CASES_DIR, a.date, "obs", f"raob_{a.obs}.json"))
    fig = plt.figure(figsize=(8, 8))
    skew = SkewT(fig, rotation=45)
    cmap = plt.get_cmap("viridis", max(len(profiles), 2))
    for k, (init, (P, T, Td)) in enumerate(sorted(profiles.items())):
        lw, col = (2.4, "#d7301f") if init.hour == 18 else (1.0, cmap(k))
        skew.plot(P, T, color=col, lw=lw, label=f"HRRR {init:%HZ}")
        skew.plot(P, Td, color=col, lw=lw * 0.7, ls="--")
    if obs:
        P, T, Td = obs_profile(obs[0])
        skew.plot(P, T, color="k", lw=2.6, label=f"Observed {a.obs}")
        skew.plot(P, Td, color="k", lw=1.8, ls="--")
    skew.ax.set_ylim(1000, 500)
    skew.ax.set_xlim(-5, 40)
    skew.plot_dry_adiabats(alpha=0.2)
    skew.plot_moist_adiabats(alpha=0.2)
    skew.ax.legend(fontsize=8, loc="upper left")
    skew.ax.set_title(f"{a.point} valid 00Z: every HRRR run vs observed (cap layer)")
    fig.savefig(os.path.join(out_dir, f"hrrr_cap_{a.point}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

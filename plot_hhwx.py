"""Clean 'HHWX' calibrated maps for one lead: the blend and each model, from ci_layers.npz.

    python plot_hhwx.py 2026-11-02 --lead 4 [--label "runs thru 12Z 29 Oct"]

Uses the stored provisional calibration and the base-rate correction (bust_summary.calibrated);
single-model maps borrow the blend's calibration (sharper models, HRES especially, read hot).
The open Atlantic is masked. Writes cases/<date>/output/hhwx_d<lead>_<blend|model>.png
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import bust_summary as bs
import calibrate
import plots
from config import CASES_DIR, MODEL_LABELS


_US = None


def us_mask(buffer_deg=0.25):
    """Grid points over (or within ~25 km of) the contiguous US, from Natural Earth country outlines."""
    global _US
    if _US is None:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.prepared import prep
        rec = next(r for r in shpreader.Reader(shpreader.natural_earth("50m", "cultural", "admin_0_countries")).records()
                   if r.attributes.get("ADM0_A3") == "USA")
        us = prep(rec.geometry.buffer(buffer_deg))
        _US = np.array([[us.contains(Point(lo, la)) for lo, la in zip(rlo, rla)]
                        for rlo, rla in zip(plots.LON2, plots.LAT2)]) & (plots.LAT2 < 50)
    return _US


def draw(p, title, out, rep=None, spc=None):
    lv, cm = calibrate.SPC_LEVELS, calibrate.SPC_CMAP
    fig = plt.figure(figsize=(11, 7.4))
    ax = fig.add_axes([0.01, 0.1, 0.98, 0.84], projection=plots.PROJ)
    plots.base_ax(ax, "conus")
    cf = ax.contourf(plots.LON2, plots.LAT2, p, levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), transform=plots.PC)
    cl = ax.contour(plots.LON2, plots.LAT2, p, levels=lv[:-1], colors="#1f3b8a", linewidths=0.7, transform=plots.PC)
    ax.clabel(cl, fontsize=8, fmt=lambda v: f"{v:g}%")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="center")
    if rep is not None:
        plots.draw_reports(ax, rep, 9)
    if spc:  # SPC outline at the same lead, as a black dashed line so it reads against the fill
        for geom, _ in spc:
            ax.add_geometries([geom], crs=plots.PC, facecolor="none", edgecolor="black", linewidth=2.0, linestyle="--", zorder=8)
        ax.plot([], [], color="black", lw=2, ls="--", label="SPC 15%")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.022])
    fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lv[:-1]).ax.tick_params(labelsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--valid", nargs="*", default=[], help="draw blend maps valid at these times (YYYYMMDDHH)")
    ap.add_argument("--hazards", action="store_true", help="draw any-severe / tornado / hail / wind maps")
    ap.add_argument("--hcal", default="calibration_hazards.json", help="hazard calibration file in cases/")
    ap.add_argument("--reports", action="store_true", help="overlay SPC reports (each hazard map: its own type)")
    ap.add_argument("--spc", action="store_true", help="outline SPC's 15%% area for this lead (blend and model maps)")
    a = ap.parse_args()
    out_dir = os.path.join(CASES_DIR, a.date, "output")
    z = np.load(os.path.join(out_dir, "ci_layers.npz"))
    cal = json.load(open(os.path.join(CASES_DIR, "calibration_provisional.json")))["env_ci"]
    at = lambda p, la, lo: p[int(round((la - 20) / 0.5)), int(round((lo + 130) / 0.5))]
    if a.hazards:
        # EXPERIMENTAL: each hazard's env x the day's CI layer, calibrated against its own report type
        import ci
        import calibrate_hazards as ch
        res = np.load(os.path.join(out_dir, "results.npz"))
        hcal = json.load(open(os.path.join(CASES_DIR, a.hcal)))
        L = a.lead
        rep = plots.load_reports(a.date, datetime.strptime(a.date, "%Y-%m-%d")) if a.reports else None
        sub = lambda t: None if rep is None else rep[rep["type"].str.startswith(t)]
        line2 = "\n" + a.label if a.label else ""
        if "any" in hcal:  # any-severe fitted on the same cases as the hazards
            p_sev = ch.calibrated(z[f"{L}__env*ci"], hcal["any"])
        else:  # older hazard files: the 14-case any-severe fit
            p_sev = ch.calibrated(z[f"{L}__env*ci"], json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json"))))
        maps = [("any", "Severe", np.where(us_mask(), p_sev, 0), rep)]
        for hz, name, t in (("tor", "Tornado", "torn"), ("hail", "Hail", "hail"), ("wind", "Wind", "wind")):
            field = ci.finals(res[f"{hz}|{L}|blend|day"], z[f"{L}__p_model"], z[f"{L}__p_lift"], z[f"{L}__sustain"])[1]
            # each hazard's reports are a subset of all severe reports: never above the any-severe probability
            maps.append((hz, name, np.where(us_mask(), np.minimum(ch.calibrated(field["env*ci"], hcal[hz]), p_sev), 0), sub(t)))
        for hz, name, p, r in maps:
            out = os.path.join(out_dir, f"hhwx_d{L}_{'sev' if hz == 'any' else hz}{'_reports' if rep is not None else ''}.png")
            draw(p, f"HHwx Prob {name} (Calibrated) Day-{L} (Model-Blend){line2}", out, r)
            i, j = np.unravel_index(np.argmax(p), p.shape)
            print(f"{hz:5s} max {p[i, j]:5.1f}% at {plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}  -> {out}")
        return
    if a.valid:
        # the environment at one valid time (instead of the 18-06Z max) times the day's CI layer
        import ci
        res = np.load(os.path.join(out_dir, "results.npz"))
        L = a.lead
        for vt in a.valid:
            env = res[f"v2|{L}|blend|vt{vt}"]
            field = ci.finals(env, z[f"{L}__p_model"], z[f"{L}__p_lift"], z[f"{L}__sustain"])[1]["env*ci"]
            p = np.where(us_mask(), bs.calibrated(field, cal), 0)
            title = f"HHwx Prob Severe (Calibrated) Day-{L} (Model-Blend) Valid {vt[4:6]}/{vt[6:8]}/{vt[:4]} {vt[8:]} UTC"
            out = os.path.join(out_dir, f"hhwx_d{L}_blend_vt{vt}.png")
            draw(p, title, out)
            i, j = np.unravel_index(np.argmax(p), p.shape)
            print(f"vt {vt} max {p[i, j]:5.1f}% at {plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}  "
                  f"OKC {at(p, 35.5, -97.5):4.1f}  SPS {at(p, 34.0, -98.5):4.1f}  DFW {at(p, 33.0, -97.0):4.1f}  "
                  f"Ardmore {at(p, 34.0, -97.0):4.1f}  -> {out}")
        return
    models = ("gfs", "gefs", "hres", "eps", "hrrr", "nam", "rap", "nam12", "rrfs")
    for name, key in [("blend", f"{a.lead}__env*ci")] + [(m, f"{a.lead}__{m}__env*ci") for m in models]:
        if key not in z.files:
            print(f"{name}: no data")
            continue
        # US only: the product verifies against US reports, so Mexico, Canada and the oceans are masked
        p = np.where(us_mask(), bs.calibrated(z[key], cal), 0)
        if name == "blend":
            used = "Model-Blend"
        else:
            used = {"hrrr": "HRRR", "nam": "NAM 3km", "rap": "RAP", "nam12": "NAM 12km", "rrfs": "RRFS"}.get(name, MODEL_LABELS.get(name, name))
        title = f"HHwx Prob Severe (Calibrated) Day-{a.lead} ({used})"
        out = os.path.join(out_dir, f"hhwx_d{a.lead}_{name}{'_spc' if a.spc else ''}.png")
        spc = plots.spc_lead_polys(a.date, datetime.strptime(a.date, "%Y-%m-%d"), a.lead)[0] if a.spc else None
        draw(p, title, out, spc=spc)
        i, j = np.unravel_index(np.argmax(p), p.shape)
        print(f"{name:5s} max {p[i, j]:5.1f}% at {plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}  "
              f"OKC {at(p, 35.5, -97.5):4.1f}  SPS {at(p, 34.0, -98.5):4.1f}  DFW {at(p, 33.0, -97.0):4.1f}  -> {out}")


if __name__ == "__main__":
    main()

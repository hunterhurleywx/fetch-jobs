"""Redraw the 0-3 km CAPE composite from its saved file (cape3_<date>.npz from cape3_composite.py): no model data,
maps only. Hunter, 1-2 Nov 2026: title, tighter zoom on a target.

    python experimental/cape3_redraw.py D:/pogs_cases/2026-11-02/cape3ml/cape3_2026-11-02.npz \
        --target 36.42153 -98.88867 --extent -101.3 -96.5 34.9 37.9
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import plots  # noqa: E402
from cape3_composite import COLORS, LEVELS, mark  # noqa: E402

TITLE = "0-3km CAPE Median Composite using HRRR, RAP, RRFS - Varying Timeframes"
CITIES = {"OKC": (35.47, -97.52), "Enid": (36.40, -97.88), "Woodward": (36.43, -99.39), "Stillwater": (36.12, -97.06),
          "Ponca City": (36.71, -97.09), "Alva": (36.80, -98.67), "Clinton": (35.52, -98.97), "Elk City": (35.41, -99.40),
          "Guymon": (36.68, -101.48), "Wichita": (37.69, -97.34), "Dodge City": (37.75, -100.02), "Medicine Lodge": (37.28, -98.58),
          "Lawton": (34.60, -98.40), "Tulsa": (36.15, -95.99), "Kingfisher": (35.86, -97.93)}
HOURS = (("20", "20Z (2 PM CST)"), ("21", "21Z (3 PM CST)"), ("22", "22Z (4 PM CST)"), ("23", "23Z (5 PM CST)"),
         ("00", "00Z (6 PM CST)"))


VORT_LEVELS = [1, 2, 3, 4, 6]  # x 1e-4 /s, positive (cyclonic) surface vorticity
VORT40_LEVELS = [0.5, 1, 1.5, 2, 3]  # 40 km boxes give weaker values
SMOOTH_CELLS = 3  # Gaussian sigma on the 0.03 deg grid (~10 km) before differencing: 3 km winds are noisy


def vorticity(u, v, lat):
    """Relative vertical vorticity dv/dx - du/dy (1/s) on the regular 0.03 deg grid."""
    ok = np.isfinite(u) & np.isfinite(v)
    us = ndimage.gaussian_filter(np.where(ok, u, 0), SMOOTH_CELLS) / np.maximum(ndimage.gaussian_filter(ok * 1.0, SMOOTH_CELLS), 1e-3)
    vs = ndimage.gaussian_filter(np.where(ok, v, 0), SMOOTH_CELLS) / np.maximum(ndimage.gaussian_filter(ok * 1.0, SMOOTH_CELLS), 1e-3)
    dy = 0.03 * 111.2e3
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    z = np.gradient(vs, axis=1) / dx - np.gradient(us, axis=0) / dy
    return np.where(ok, z, np.nan)


def vorticity_40km(u, v, lat, lon, block=12):
    """SPC-mesoanalysis-like surface vorticity: 10 m winds averaged to ~40 km boxes (12 x 0.03 deg), then centred
    differences on that 40 km grid (as the SPC hourly objective analysis grid). Returns zeta, lat, lon (coarse)."""
    ny, nx = (u.shape[0] // block) * block, (u.shape[1] // block) * block

    def coarse(a):
        a = a[:ny, :nx].reshape(ny // block, block, nx // block, block)
        return np.nanmean(np.nanmean(a, axis=3), axis=1)
    uc, vc = coarse(u), coarse(v)
    lac = lat[:ny].reshape(-1, block).mean(1)
    loc = lon[:nx].reshape(-1, block).mean(1)
    dy = block * 0.03 * 111.2e3
    dx = dy * np.cos(np.deg2rad(lac))[:, None]
    return np.gradient(vc, axis=1) / dx - np.gradient(uc, axis=0) / dy, lac, loc


# PivotalWeather CAPE palette, as sampled for the HHwx model archive (hhwx.me model_ondemand.py "cape" ramp)
PW_CAPE = ["#ffffff", "#f0f0f0", "#d2d2d2", "#969696", "#787878", "#436075", "#5c7a8b", "#7594a2", "#9bbbc3",
           "#e9dd96", "#e6b865", "#e5a044", "#e27b12", "#dc4110", "#c12e25", "#af2234", "#940f49", "#73088a",
           "#9538a9", "#ac59be", "#c379d2", "#e9bec3", "#d795a0", "#ca7a89", "#be5f71", "#a97178", "#b6868b"]


def recipe_f(z, m, h):
    """SPC-mesoanalysis match found on 2 Nov 2026 (experimental/spc_match.py vs SPC 00Z, Southwest sector):
    100 mb ML 3CAPE smoothed ~30 km; vorticity from 10 m winds averaged to ~40 km boxes, then smoothed ~60 km.
    Returns (cape on the 0.03 deg grid, vorticity x1e-5 on the 40 km grid, coarse lat, coarse lon)."""
    from scipy.ndimage import gaussian_filter
    c = z[f"{m}_{h}"].astype(float)
    ok = np.isfinite(c)
    cs = gaussian_filter(np.where(ok, c, 0), 30 / 3.33) / np.maximum(gaussian_filter(ok * 1.0, 30 / 3.33), 1e-3)
    cs = np.where(ok, cs, np.nan)
    zeta, lac, loc = vorticity_40km(z[f"{m}_u_{h}"].astype(float), z[f"{m}_v_{h}"].astype(float), z["lat"], z["lon"])
    zeta = gaussian_filter(np.nan_to_num(zeta * 1e5), 60 / 40)
    return cs, zeta, lac, loc


REC_VORT = [3, 6, 10, 15, 20, 30, 40]  # x1e-5 /s, the HHwx archive's HRRR levels


def recommended(z, m, h):
    """Hunter's pick, 2 Nov 2026: our 100 mb ML 3CAPE (light ~10 km smoothing, SPC definition) + HHwx-archive surface
    vorticity (19 km uniform window, x1e-5) from earth-relative 10 m winds, on the common 0.03 deg grid."""
    from scipy.ndimage import gaussian_filter, uniform_filter
    c = z[f"{m}_{h}"].astype(float)
    ok = np.isfinite(c)
    cs = gaussian_filter(np.where(ok, c, 0), 3) / np.maximum(gaussian_filter(ok * 1.0, 3), 1e-3)
    u, v = z[f"{m}_u_{h}"].astype(float), z[f"{m}_v_{h}"].astype(float)
    okw = np.isfinite(u) & np.isfinite(v)
    dy = 0.03 * 111320.0
    dx = dy * np.cos(np.deg2rad(z["lat"]))[:, None]
    zeta = (np.gradient(np.where(okw, v, 0), axis=1) / dx - np.gradient(np.where(okw, u, 0), axis=0) / dy) * 1e5
    zeta = uniform_filter(zeta, size=int(round(19.0 / 3.34)), mode="nearest")
    return np.where(ok, cs, np.nan), np.where(okw, zeta, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--target", type=float, nargs=2, default=None)
    ap.add_argument("--extent", type=float, nargs=4, default=[-101.3, -96.5, 34.9, 37.9], help="lon0 lon1 lat0 lat1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=["median", "hrrr", "rap", "rrfs"], default="median")
    ap.add_argument("--vort", action="store_true", help="overlay surface (10 m) relative vorticity contours")
    ap.add_argument("--spc40", action="store_true", help="vorticity on ~40 km boxes like the SPC mesoanalysis grid")
    ap.add_argument("--recommended", action="store_true", help="ML 3CAPE (~10 km) + HHwx 19 km surface vorticity, per model")
    ap.add_argument("--recipe-f", action="store_true", help="SPC-matched smoothing (ML 3CAPE 30 km, vorticity 40 km grid + 60 km) and PW colours")
    a = ap.parse_args()
    z = np.load(a.npz)
    out = a.out or os.path.dirname(a.npz)
    LO, LA = np.meshgrid(z["lon"], z["lat"])
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS + [10000], cmap.N)
    key = a.model
    names = {"hrrr": "HRRR", "rap": "RAP", "rrfs": "RRFS"}
    if a.model == "median":
        title, tag = TITLE, ""
    else:
        run = str(z[f"{a.model}_run"])
        title = f"0-3km CAPE - {names[a.model]} ({run[6:8]}/{run[8:10]}Z run) - Varying Timeframes"
        tag = f"_{a.model}"
    if a.vort:
        title = title.replace("0-3km CAPE", "0-3km CAPE + Sfc Vorticity")
        tag += "_vort40" if a.spc40 else "_vort"
        if a.spc40:
            title = title.replace("Sfc Vorticity", "Sfc Vorticity (40 km, SPC-style)")
    stats = []
    if a.recommended:
        a.recipe_f = True  # same colours and layout; different smoothing below
    if a.recipe_f:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pwcape", PW_CAPE)
        flv = list(range(0, 351, 10))
        norm = BoundaryNorm(flv + [10000], cmap.N)
        nm = {"median": "Median Composite using HRRR, RAP, RRFS", "hrrr": "HRRR", "rap": "RAP", "rrfs": "RRFS"}[a.model]
        run = "" if a.model == "median" else f" ({str(z[f'{a.model}_run'])[6:8]}/{str(z[f'{a.model}_run'])[8:10]}Z run)"
        title = f"0-3km MLCAPE + Sfc Vorticity (SPC-style) - {nm}{run} - Varying Timeframes"
        tag = f"_{a.model}_spcf"
        if a.recommended:
            title = f"0-3km MLCAPE + Sfc Vorticity - {nm}{run} - Varying Timeframes"
            tag = f"_{a.model}"
    panels = [(h, lab) for h, lab in HOURS if f"{key}_{h}" in z.files or (a.model == "median" and f"hrrr_{h}" in z.files)]

    def draw(ax, h, lab):
        plots.base_ax(ax, "conus")
        ax.set_extent(a.extent, plots.PC)
        if a.recipe_f:
            ms = ["hrrr", "rap", "rrfs"] if a.model == "median" else [a.model]
            if a.recommended:
                got = [recommended(z, mm, h) for mm in ms if f"{mm}_u_{h}" in z.files]
                m = np.nanmedian(np.stack([g[0] for g in got]), axis=0)
                zeta = np.nanmedian(np.stack([g[1] for g in got]), axis=0)
                cf = ax.pcolormesh(LO, LA, m, cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
                ax.contour(LO, LA, np.nan_to_num(zeta), levels=REC_VORT, colors="#111111", linewidths=0.8,
                           transform=plots.PC, zorder=7)
            else:
                got = [recipe_f(z, mm, h) for mm in ms if f"{mm}_u_{h}" in z.files]
                m = np.nanmedian(np.stack([g[0] for g in got]), axis=0)
                zeta = np.nanmedian(np.stack([g[1] for g in got]), axis=0)
                cf = ax.pcolormesh(LO, LA, m, cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
                CLO, CLA = np.meshgrid(got[0][3], got[0][2])
                cs = ax.contour(CLO, CLA, zeta, levels=list(range(1, 21)), colors="#1f8fd0", linewidths=0.9, transform=plots.PC, zorder=7)
                ax.clabel(cs, fontsize=7, fmt="%d")
            stats.append((h, float(np.nanmax(zeta)), float(np.nanmax(m))))
        else:
            m = z[f"{key}_{h}"].astype(float)
            cf = ax.pcolormesh(LO, LA, np.ma.masked_less(m, LEVELS[0]), cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
        if a.vort and not a.recipe_f:
            ms = ["hrrr", "rap", "rrfs"] if a.model == "median" else [a.model]
            if a.spc40:
                got = [vorticity_40km(z[f"{mm}_u_{h}"].astype(float), z[f"{mm}_v_{h}"].astype(float), z["lat"], z["lon"])
                       for mm in ms if f"{mm}_u_{h}" in z.files]
                zs = [g[0] for g in got]
                CLO, CLA = (np.meshgrid(got[0][2], got[0][1]) if got else (LO, LA))
            else:
                zs = [vorticity(z[f"{mm}_u_{h}"].astype(float), z[f"{mm}_v_{h}"].astype(float), z["lat"])
                      for mm in ms if f"{mm}_u_{h}" in z.files]
                CLO, CLA = LO, LA
            if zs:
                zeta = 1e4 * np.nanmedian(np.stack(zs), axis=0)
                cs = ax.contour(CLO, CLA, np.nan_to_num(zeta), levels=VORT40_LEVELS if a.spc40 else VORT_LEVELS, colors="black",
                                linewidths=[0.8, 1.1, 1.4, 1.7, 2.0], transform=plots.PC, zorder=7)
                ax.clabel(cs, fontsize=7, fmt="%g")
                stats.append((h, float(np.nanmax(zeta))))
        mark(ax, a.target)
        for name, (la, lo) in CITIES.items():
            if a.extent[0] < lo < a.extent[1] and a.extent[2] < la < a.extent[3]:
                ax.plot(lo, la, "o", ms=3.5, color="black", transform=plots.PC, zorder=8)
                ax.text(lo + 0.04, la + 0.04, name, fontsize=8, transform=plots.PC, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
        ax.set_title(lab, fontsize=12, fontweight="bold")
        return cf

    for h, lab in panels:
        fig = plt.figure(figsize=(11, 8.2))
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.8], projection=plots.PROJ)
        cf = draw(ax, h, f"{title}\nValid {lab}")
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.022])
        lab_cb = (("0-3 km MLCAPE (J/kg); black contours: surface relative vorticity, x1e-5 /s (3, 6, 10, 15, 20, 30, 40)" if a.recommended else "0-3 km MLCAPE (J/kg); blue contours: surface relative vorticity, x1e-5 /s (every 1)") if a.recipe_f else None) or "0-3 km CAPE (J/kg)" + (f"; contours: surface relative vorticity, x1e-4 /s ({', '.join(f'{x:g}' for x in (VORT40_LEVELS if a.spc40 else VORT_LEVELS))})" if a.vort else "")
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=list(range(0, 351, 50)) if a.recipe_f else LEVELS,
                     label=lab_cb).ax.tick_params(labelsize=9)
        fig.savefig(os.path.join(out, f"cape3_zoom{tag}_{h}z.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), subplot_kw={"projection": plots.PROJ})
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (h, lab) in zip(axes.ravel(), panels):
        cf = draw(ax, h, lab)
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
    fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=list(range(0, 351, 50)) if a.recipe_f else LEVELS,
                 label=(("0-3 km MLCAPE (J/kg); black contours: surface relative vorticity, x1e-5 /s (3, 6, 10, 15, 20, 30, 40)" if a.recommended else "0-3 km MLCAPE (J/kg); blue contours: surface relative vorticity, x1e-5 /s (every 1)") if a.recipe_f else "0-3 km CAPE (J/kg)" + "") + (f"; contours: surface relative vorticity, x1e-4 /s ({', '.join(f'{x:g}' for x in (VORT40_LEVELS if a.spc40 else VORT_LEVELS))})" if a.vort else ""))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    p = os.path.join(out, f"cape3_zoom{tag}_panel.png")
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("->", p)
    if stats:
        print("by hour (max vorticity, max 3CAPE):", ", ".join(f"{t[0]}Z " + "/".join(f"{x:.1f}" for x in t[1:]) for t in stats[: len(panels)]))


if __name__ == "__main__":
    main()

"""Figures for a case: per-model maps, timing, run-to-run trend, lead-day summary, ingredients."""
import glob
import json
import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import shape

from config import CASES_DIR, GRID_LATS, GRID_LONS, MODEL_LABELS, MODELS, VIEWS

LEVELS = [1, 5, 10, 25, 50, 75, 90, 100]
CMAP = ListedColormap(["#fff3b0", "#fdd17a", "#f9a24d", "#ef6a32", "#d7301f", "#a50f15", "#5c0a0f"])
NORM = BoundaryNorm(LEVELS, CMAP.N)
PROJ = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
PC = ccrs.PlateCarree()
LON2, LAT2 = np.meshgrid(GRID_LONS, GRID_LATS)
SCHEME_LABEL = {"current": "Current scoring (gfs-ensemble.py)", "v2": "Prototype v2 scoring (research-based)"}
SPC_STYLE = {"0.15": ("#DDAA00", "15%"), "0.30": ("#CC0000", "30%"), "0.05": ("#8B4513", "5%"),
             "0.45": ("#FF00FF", "45%"), "0.60": ("#912CEE", "60%")}
CAT_STYLE = {"TSTM": "#55BB55", "MRGL": "#005500", "SLGT": "#DDAA00", "ENH": "#FF6600",
             "MDT": "#CC0000", "HIGH": "#FF00FF"}


# ---------------------------------------------------------------------------
# SPC data
# ---------------------------------------------------------------------------
def spc_dir(case):
    return os.path.join(CASES_DIR, case, "spc")


def load_geo(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [(shape(ft["geometry"]), ft["properties"]) for ft in json.load(f).get("features", [])]


def spc_lead_polys(case, conv_day, lead):
    """SPC outlook matching a lead day, issued near our ~08Z product time.

    Returns (polys, is_categorical). Day 4-8 and Day 3 are any-severe probabilities;
    Day 2 (06Z) and Day 1 (06Z) have no any-severe field, so their categorical
    outlook is used.
    """
    issue = conv_day - timedelta(days=lead - 1)
    d = spc_dir(case)
    if lead >= 4:
        pats, cat = [f"day{lead}_issued_{issue:%Y%m%d}.geojson"], False
    elif lead == 3:
        pats, cat = [f"day3_0730z_issued_{issue:%Y%m%d}_prob.geojson", "day3_0730z_*_prob.geojson"], False
    elif lead == 2:
        pats, cat = [f"day2_0600z_issued_{issue:%Y%m%d}_cat.geojson", "day2_0600z_*_cat.geojson"], True
    else:
        pats, cat = ["day1_0600z_cat.geojson", "day1_0600z_*_cat.geojson"], True
    for p in pats:
        hits = sorted(glob.glob(os.path.join(d, p)))
        if hits:
            return load_geo(hits[0]), cat
    return None, cat


def load_reports(case, conv_day):
    p = os.path.join(spc_dir(case), f"reports_{conv_day:%Y%m%d}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["type"] = df["type"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def base_ax(ax, view):
    ax.set_extent(VIEWS[view], crs=PC)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, edgecolor="#555555")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)


def draw_prob(ax, field, z500=None):
    if z500 is not None:
        cs = ax.contour(LON2, LAT2, z500 / 10.0, range(510, 600, 6), colors="#444444",
                        linewidths=0.6, alpha=0.5, transform=PC)
        ax.clabel(cs, fontsize=6, fmt="%d")
    pct = 100.0 * np.clip(field, 0, 1)
    cf = ax.contourf(LON2, LAT2, pct, levels=LEVELS, cmap=CMAP, norm=NORM, transform=PC, extend="neither")
    cl = ax.contour(LON2, LAT2, pct, levels=LEVELS[:-1], colors="#1f3b8a", linewidths=0.6, transform=PC)
    ax.clabel(cl, fontsize=6, fmt=lambda x: f"{x:g}%")
    return cf


def draw_spc(ax, polys, cat=False):
    if not polys:
        return
    for geom, props in polys:
        lvl = str(props.get("level"))
        color = CAT_STYLE.get(lvl, "#000000") if cat else SPC_STYLE.get(lvl, ("#000000", lvl))[0]
        ax.add_geometries([geom], crs=PC, facecolor="none", edgecolor=color,
                          linewidth=2.0, linestyle="-" if cat else "--", zorder=6)


def draw_reports(ax, rep, size=6):
    if rep is None:
        return
    for typ, mk, col in (("wind", "o", "#1f5fd6"), ("hail", "o", "#1a9e3a"), ("tornado", "^", "#d00000")):
        d = rep[rep["type"].str.startswith(typ[:4])]
        ax.scatter(d["lon"], d["lat"], s=size if typ != "tornado" else size * 2.2, marker=mk, c=col,
                   edgecolors="black" if typ == "tornado" else "none", linewidths=0.3,
                   transform=PC, zorder=7, alpha=0.8)


def spc_note(ax, polys, lead, cat=False):
    if polys is None:
        txt = f"SPC Day {lead}: file not found"
    elif len(polys) == 0:
        txt = f"SPC Day {lead}: no 15% area"
    elif cat:
        txt = f"SPC Day {lead} 06Z categorical (solid)"
    else:
        txt = f"SPC Day {lead}: " + ", ".join(sorted({SPC_STYLE.get(str(p['level']), ('', str(p['level'])))[1]
                                                      for _, p in polys})) + " (dashed)"
    ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"), zorder=8)


def colorbar(fig, cf, label="Ingredient index [%] (uncalibrated)"):
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS)
    cb.set_label(label, fontsize=9)


def legend_reports(fig):
    fig.text(0.99, 0.01, "Reports: ▲ tornado  ● wind (blue)  ● hail (green)", ha="right", fontsize=8)


def get(res, key):
    return res.get(key)


def z500_of(res, lead, model):
    if model == "blend":
        zs = [res[f"diag|{lead}|{m}|z500"] for m in MODELS if f"diag|{lead}|{m}|z500" in res]
        return np.mean(zs, axis=0) if zs else None
    return res.get(f"diag|{lead}|{model}|z500")


def nruns(res, scheme, lead, model):
    k = f"n|{scheme}|{lead}|{model}"
    return int(res[k]) if k in res else 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_models(case, res, conv_day, lead, pools, scheme, view, outdir, rep):
    polys, iscat = spc_lead_polys(case, conv_day, lead)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), subplot_kw={"projection": PROJ})
    cf = None
    for ax, m in zip(axes.flat, MODELS + ["blend"]):
        base_ax(ax, view)
        f = get(res, f"{scheme}|{lead}|{m}|day")
        n = "" if m == "blend" else f"  (n={nruns(res, scheme, lead, m)})"
        ax.set_title(f"{MODEL_LABELS[m]}{n}", fontsize=11)
        if f is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        cf = draw_prob(ax, f, z500_of(res, lead, m))
        draw_spc(ax, polys, iscat)
        draw_reports(ax, rep, 4)
        spc_note(ax, polys, lead, iscat)
    ax = axes.flat[5]
    base_ax(ax, view)
    ax.set_title("SPC Day 1 (1630Z) categorical + reports", fontsize=11)
    draw_spc(ax, load_geo(os.path.join(spc_dir(case), "day1_1630z_cat.geojson")), cat=True)
    draw_reports(ax, rep, 7)
    inits = pools[lead]
    fig.suptitle(f"{case}  Day {lead} (SPC numbering, issued {conv_day - timedelta(days=lead - 1):%d %b})  |  "
                 f"{SCHEME_LABEL[scheme]}\nConvective-day max of 18/21/00/03/06Z  |  pool: "
                 f"{inits[0]:%d/%HZ}–{inits[-1]:%d/%HZ} ({len(inits)} cycles)", fontsize=13)
    if cf is not None:
        colorbar(fig, cf)
    legend_reports(fig)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08, wspace=0.04, hspace=0.12)
    fig.savefig(os.path.join(outdir, f"models_{scheme}_D{lead}_{view}.png"), dpi=110)
    plt.close(fig)


def fig_timing(case, res, conv_day, lead, valid, scheme, view, outdir, rep):
    polys, iscat = spc_lead_polys(case, conv_day, lead)
    fig, axes = plt.subplots(1, len(valid), figsize=(4.2 * len(valid), 4.6), subplot_kw={"projection": PROJ})
    cf = None
    for ax, v in zip(axes, valid):
        base_ax(ax, view)
        ax.set_title(f"Blend valid {v:%d/%HZ}", fontsize=10)
        f = get(res, f"{scheme}|{lead}|blend|vt{v:%Y%m%d%H}")
        if f is not None:
            cf = draw_prob(ax, f)
            draw_spc(ax, polys, iscat)
            # reports within +/-90 min of this valid time
            if rep is not None:
                t = pd.to_datetime(rep["date_utc"] + " " + rep["time_utc"].astype(str).str.zfill(4),
                                   format="%Y-%m-%d %H%M")
                draw_reports(ax, rep[(t - v).abs() <= pd.Timedelta(minutes=90)], 7)
    fig.suptitle(f"{case}  Day {lead}  |  {SCHEME_LABEL[scheme]}  |  blend by valid time "
                 f"(reports ±90 min)", fontsize=12)
    if cf is not None:
        cax = fig.add_axes([0.3, 0.06, 0.4, 0.03])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.12, wspace=0.03)
    fig.savefig(os.path.join(outdir, f"timing_{scheme}_D{lead}_{view}.png"), dpi=110)
    plt.close(fig)


def fig_trend(case, res, conv_day, lead, pools, scheme, view, outdir, rep):
    polys, iscat = spc_lead_polys(case, conv_day, lead)
    inits = pools[lead]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), subplot_kw={"projection": PROJ})
    cf = None
    for ax, i in zip(axes.flat, inits):
        base_ax(ax, view)
        f = get(res, f"{scheme}|{lead}|blend|cyc{i:%Y%m%d%H}")
        have = [MODEL_LABELS[m] for m in MODELS if f"{scheme}|{lead}|{m}|cyc{i:%Y%m%d%H}" in res]
        ax.set_title(f"Cycle {i:%d %b %HZ}\n{', '.join(have) or 'no data'}", fontsize=9)
        if f is not None:
            cf = draw_prob(ax, f)
            draw_spc(ax, polys, iscat)
    fig.suptitle(f"{case}  Day {lead}  |  {SCHEME_LABEL[scheme]}  |  run-to-run trend "
                 f"(blend of models available from each cycle)", fontsize=12)
    if cf is not None:
        colorbar(fig, cf)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08, wspace=0.04, hspace=0.15)
    fig.savefig(os.path.join(outdir, f"trend_{scheme}_D{lead}_{view}.png"), dpi=110)
    plt.close(fig)


def fig_summary(case, res, conv_day, leads, view, outdir, rep):
    """One figure per scheme: the blend at every lead day, plus SPC Day 1 1630Z + reports."""
    n = len(leads) + 1
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    for scheme in ("current", "v2"):
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow + 1.2),
                                 subplot_kw={"projection": PROJ})
        axes = np.atleast_1d(axes).ravel()
        cf = None
        for ax, lead in zip(axes, leads):
            base_ax(ax, view)
            polys, iscat = spc_lead_polys(case, conv_day, lead)
            f = get(res, f"{scheme}|{lead}|blend|day")
            issue = conv_day - timedelta(days=lead - 1)
            ax.set_title(f"Day {lead} (issued {issue:%d %b}) blend", fontsize=11)
            if f is not None:
                cf = draw_prob(ax, f, z500_of(res, lead, "blend"))
            draw_spc(ax, polys, iscat)
            draw_reports(ax, rep, 4)
            spc_note(ax, polys, lead, iscat)
        ax = axes[len(leads)]
        base_ax(ax, view)
        ax.set_title("SPC Day 1 1630Z categorical + reports", fontsize=11)
        draw_spc(ax, load_geo(os.path.join(spc_dir(case), "day1_1630z_cat.geojson")), cat=True)
        draw_reports(ax, rep, 6)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"{case}: Day {leads[0]} → Day {leads[-1]} | {SCHEME_LABEL[scheme]}\n"
                     f"blend = ECMWF 60% / NCEP 40% (global models only); uncalibrated index", fontsize=13)
        if cf is not None:
            colorbar(fig, cf)
        legend_reports(fig)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.03, hspace=0.12)
        fig.savefig(os.path.join(outdir, f"summary_{scheme}_{view}.png"), dpi=100)
        plt.close(fig)


def fig_ingredients(case, res, conv_day, lead, view, outdir, rep):
    """Model-mean ingredients at 00Z (weighted like the blend)."""
    from scoring import BLEND_WEIGHTS
    def wmean(k):
        num, den = 0, 0
        for m in MODELS:
            key = f"diag|{lead}|{m}|{k}"
            if key in res:
                num, den = num + BLEND_WEIGHTS[m] * res[key], den + BLEND_WEIGHTS[m]
        return num / den if den else None
    panels = [("cape_v2", "CAPE used by v2 [J/kg] (GFS ML / GEFS best / EC MU)", "YlOrRd", 0, 3000),
              ("shear", "10 m–500 hPa bulk shear [m/s]", "BuPu", 0, 35),
              ("omega", "700/500 hPa mean ω [Pa/s] (neg = ascent)", "RdBu_r", -0.3, 0.3),
              ("lr75", "700–500 hPa lapse rate [K/km]", "RdYlBu_r", 5, 8.5),
              ("q2", "2 m specific humidity [g/kg]", "GnBu", 4, 18)]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), subplot_kw={"projection": PROJ})
    polys, iscat = spc_lead_polys(case, conv_day, lead)
    for ax, (k, title, cmap, lo, hi) in zip(axes.flat, panels):
        base_ax(ax, view)
        f = wmean(k)
        ax.set_title(title, fontsize=10)
        if f is None:
            continue
        pm = ax.pcolormesh(LON2, LAT2, f, cmap=cmap, vmin=lo, vmax=hi, transform=PC, shading="auto")
        fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
        draw_spc(ax, polys, iscat)
        draw_reports(ax, rep, 3)
    ax = axes.flat[5]
    base_ax(ax, view)
    ax.set_title("v2 blend valid 00Z", fontsize=10)
    v00 = datetime(conv_day.year, conv_day.month, conv_day.day) + timedelta(days=1)
    f = res.get(f"v2|{lead}|blend|vt{v00:%Y%m%d%H}")
    if f is not None:
        draw_prob(ax, f, wmean("z500"))
        draw_spc(ax, polys, iscat)
    fig.suptitle(f"{case}  Day {lead}: pooled-model mean ingredients at 00Z (blend weights)", fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.05, hspace=0.15)
    fig.savefig(os.path.join(outdir, f"ingredients_D{lead}_{view}.png"), dpi=110)
    plt.close(fig)


def make_all(case, res, valid, pools, outdir):
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    rep = load_reports(case, conv_day)
    leads = sorted(pools, reverse=True)
    for view in VIEWS:
        fig_summary(case, res, conv_day, leads, view, outdir, rep)
        for lead in leads:
            fig_ingredients(case, res, conv_day, lead, view, outdir, rep)
            for scheme in ("current", "v2"):
                fig_models(case, res, conv_day, lead, pools, scheme, view, outdir, rep)
                fig_timing(case, res, conv_day, lead, valid, scheme, view, outdir, rep)
                fig_trend(case, res, conv_day, lead, pools, scheme, view, outdir, rep)

"""Real-time map set for one convective day: calibrated + uncalibrated blend, per-model, and single-run maps.

    python redraw_latest.py 2026-11-02 --pool 3 --day 4 --spc-lead 4 [--single hres:2026103006]

--pool      pipeline lead whose run pool to draw (SPC Day-N pool: 3 = the newest runs before tomorrow's SPC Day 3)
--day       real-time label: days from TODAY (Day 1 = today), e.g. Monday seen on Friday = 4 (see day-numbering rule)
--spc-lead  which SPC outlook to outline (black dashed 15%); omit for none
--single    also draw one run alone: model:YYYYMMDDHH (overall / tornado / wind / hail)

Needs consistency_blend.py run first for the pool (it writes consistency_raw_d<pool>.npz):
    python consistency_blend.py 2026-11-02 --lead 3 --hazards
Writes cases/<date>/output/hhwx_d<day>latest_*.png, hhwx_d<day>_<model><cycle>_*.png.
Calibrations: overall = 14-case any-severe (calibration_anysevere_14case.json) everywhere; hazards = calibration_hazards.json
capped at the 14-case any-severe (calibration_anysevere_14case.json).
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

import bust_summary as bs
import calibrate
import calibrate_hazards as ch
import plot_hhwx
import scoring
import plots
from config import CASES_DIR, POOL_CYCLES

LV = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
CM = ListedColormap(["#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"])
NAMES = {"eps": "ECMWF EPS", "hres": "ECMWF HRES", "gfs": "GFS", "gefs": "GEFS", "nam12": "NAM 12 km", "rrfs": "RRFS", "nam": "NAM 3 km", "hrrr": "HRRR", "rap": "RAP", "refs": "REFS (5 members)"}
HZ = (("tor", "Tornado"), ("wind", "Wind"), ("hail", "Hail"))


PANEL_NAMES = {"Overall severe": "Overall Severe", "Tornado": "Tornado", "Wind": "Wind", "Hail": "Hail"}


def runs_text(res, L, models, day_before_03z=""):
    """Centered block of which runs feed the map: 'ECMWF EPS' / '29/18Z, 30/00Z 06Z 12Z 18Z' per model."""
    lines = []
    for m in models:
        cyc = sorted({k.split("cyc")[1] for k in res.files if k.startswith(f"v2|{L}|{m}|cyc")})
        if m not in ("nam12", "rrfs", "nam", "hrrr", "rap", "refs"):  # globals: the pool's newest POOL_CYCLES (as consistency_blend.py)
            cyc = cyc[-POOL_CYCLES:]
        if m == "rap" and day_before_03z:  # track-record mode uses RAP only from the 03Z day-before run
            cyc = [c for c in cyc if c >= day_before_03z]
        if not cyc:
            continue
        days = {}
        for c in cyc:
            days.setdefault(f"{c[4:6]}/{c[6:8]}", []).append(f"{c[8:]}z")
        lines.append((NAMES.get(m, m), chr(10).join(f"{d} - {', '.join(h)}" for d, h in days.items())))
    return lines


def draw_panel(maps, title, out, spc=None, spc_lead=None, at=None, extent=(-108, -86, 26, 42), runs=None, fine=False,
               reports=None, dpi=110):
    """Hunter's default layout: one 2x2 image (overall, tornado, wind, hail), calibrated, SPC outline on each panel.

    Each panel title is '<Hazard>: Max. <v>%' (max inside the mapped area) with a black crosshair at that point.
    The middle of the figure carries the legend (SPC outline, max marker) and, if given, the runs used per model
    (runs: list of (model name, runs string), e.g. from runs_text()). `at` is unused (kept for old callers).
    """
    lv, cm = calibrate.SPC_LEVELS, calibrate.SPC_CMAP
    if fine:  # 3-h blocks: probabilities are small, so add 0.5 and 1% shades below the usual 2% start
        lv = [0.5, 1] + list(lv)
        cm = ListedColormap(["#fffbe6", "#fff8cc"] + list(calibrate.SPC_CMAP.colors))
    # SPC outline: the 15% any-severe area (Days 3-8). Days 2-1 have no any-severe field: pass a dict
    # {panel name: [(geom, props)]} (spc_hazard_panels) to put Slight+ on the overall panel and each hazard's own
    # probability lines (plus the hatched significant area) on its panel.
    per_panel = isinstance(spc, dict)
    if per_panel:
        spc_txt = "(levels in titles)"
    else:
        spc = [(g, lab) for g, lab in (spc or []) if lab.get("LABEL") in ("0.15", "SLGT")]
        spc_txt = "Slight risk (15% wind/hail equivalent)" if any(lab["LABEL"] == "SLGT" for _, lab in spc) else "15%"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), subplot_kw={"projection": plots.PROJ})
    fig.subplots_adjust(wspace=0.42, hspace=0.18)
    inside = ((plots.LON2 >= extent[0]) & (plots.LON2 <= extent[1]) & (plots.LAT2 >= extent[2]) & (plots.LAT2 <= extent[3]))
    cf = None
    for ax, (nm, p) in zip(axes.flat, maps):
        plots.base_ax(ax, "conus")
        ax.set_extent(list(extent), plots.PC)
        cf = ax.contourf(plots.LON2, plots.LAT2, p, levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), transform=plots.PC)
        ax.clabel(ax.contour(plots.LON2, plots.LAT2, p, levels=lv[:-1], colors="#1f3b8a", linewidths=0.6, transform=plots.PC),
                  fontsize=8, fmt=lambda v: f"{v:g}%")
        polys = spc.get(nm, []) if per_panel else spc
        levels = []
        for geom, lab in polys:
            parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")]
            lbl = str(lab.get("LABEL"))
            if lbl.startswith("CIG") or lbl == "SIGN":  # significant / conditional-intensity area: hatched
                ax.add_geometries(parts, crs=plots.PC, facecolor="none", edgecolor="#444444", hatch="///", linewidth=0.6, zorder=7)
                continue
            ax.add_geometries(parts, crs=plots.PC, facecolor="none", edgecolor="black", linewidth=1.8, linestyle="--", zorder=8)
            levels.append(lbl)
        pin = np.where(inside, p, -1.0)
        i, j = np.unravel_index(np.argmax(pin), pin.shape)
        vmax = float(max(pin[i, j], 0.0))
        if vmax > 0:
            ax.plot(plots.LON2[i, j], plots.LAT2[i, j], marker="+", ms=9, mec="black", mew=1.8, transform=plots.PC, zorder=12)
        extra = ""
        if per_panel and levels:
            extra = "  (SPC " + ("/".join(f"{float(v) * 100:g}" for v in levels if v.replace(".", "").isdigit()) + "%)"
                                 if nm != "Overall severe" else "Slight+)")
        if reports is not None and len(reports):  # past cases: the reports that verify each panel
            sel = {"Tornado": ("tornado",), "Wind": ("wind",), "Hail": ("hail",)}.get(nm, ("tornado", "wind", "hail"))
            for typ, col, mk, sz in (("wind", "#1f5fd6", "o", 7), ("hail", "#1a9e3a", "o", 7), ("tornado", "#d00000", "v", 16)):
                if typ in sel:
                    d = reports[reports["type"] == typ]
                    ax.scatter(d["lon"], d["lat"], s=sz, marker=mk, c=col, edgecolors="black", linewidths=0.25,
                               transform=plots.PC, zorder=11, alpha=0.85)
        ax.set_title(f"{PANEL_NAMES.get(nm, nm)}: Max. {vmax:.1f}%{extra}", fontsize=13)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.03, pad=0.04, ticks=lv[:-1])
    cb.set_label("%")
    # middle column: centred on the real gap between the map columns (map axes shrink to keep their aspect,
    # so the gap centre is not the figure centre)
    for ax in axes.flat:
        ax.apply_aspect()
    gx = 0.5 * (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="black", lw=1.6, ls="--"),
               Line2D([], [], color="black", marker="+", ms=9, mew=1.8, lw=0)]
    labels = [f"SPC Day {spc_lead} {spc_txt}" if spc_lead and spc else "SPC outline (none issued)", "Highest Probability"]
    if reports is not None and len(reports):
        handles += [Line2D([], [], color="#d00000", marker="v", lw=0, mec="black", mew=0.3, ms=6),
                    Line2D([], [], color="#1f5fd6", marker="o", lw=0, mec="black", mew=0.3, ms=4),
                    Line2D([], [], color="#1a9e3a", marker="o", lw=0, mec="black", mew=0.3, ms=4)]
        labels += ["Tornado report", "Wind report", "Hail report"]
    if per_panel and any(str(lab.get("LABEL")).startswith("CIG") for v in spc.values() for _, lab in v):
        handles.append(Patch(facecolor="none", edgecolor="#444444", hatch="///"))
        labels.append("SPC cond. intensity")
    extra_rows = max(len(handles) - 2, 0)  # legend grows: shift it up and the model list down
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(gx, 0.80 + 0.012 * extra_rows), frameon=True, edgecolor="#bbbbbb",
               fancybox=False, fontsize=10, handlelength=2.2, borderpad=0.7, labelspacing=0.7)
    if runs:
        y = 0.71 - 0.022 * extra_rows
        fig.text(gx, y, "Models Used", ha="center", va="center", fontsize=11.5, fontweight="bold")
        fig.add_artist(Line2D([gx - 0.045, gx + 0.045], [y - 0.016, y - 0.016], transform=fig.transFigure, color="black", lw=1.0))
        y -= 0.008
        # squeeze the spacing (never stretch it) so the list ends above the colorbar
        need = sum(0.046 + 0.024 * len(rs.split(chr(10))) for _, rs in runs)
        s = min(1.0, (y - cb.ax.get_position().y1 - 0.03) / need)
        for name, rs in runs:
            y -= 0.04 * s
            fig.text(gx, y, name, ha="center", va="center", fontsize=10.5, fontweight="bold")
            for line in rs.split(chr(10)):
                y -= 0.024 * s
                fig.text(gx, y, line, ha="center", va="center", fontsize=9.5, color="#333333")
            y -= 0.006 * s
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def spc_hazard_panels(date, lead):
    """Day 2 / Day 1 SPC outlines per panel: Slight-or-higher categorical on Overall, each hazard's probabilities on its
    own panel. None if the hazard files are missing."""
    conv = datetime.strptime(date, "%Y-%m-%d")
    issue = conv - timedelta(days=lead - 1)
    d = plots.spc_dir(date)
    stem = f"day2_0600z_issued_{issue:%Y%m%d}" if lead == 2 else "day1_0600z"
    if lead == 2 and os.environ.get("POGS_SPC_D2"):
        stem = f"day2_{os.environ['POGS_SPC_D2']}z_issued_{issue:%Y%m%d}"
    load = lambda k: plots.load_geo(os.path.join(d, f"{stem}_{k}.geojson"))
    cat, tor, wind, hail = load("cat"), load("torn"), load("wind"), load("hail")
    if cat is None or tor is None:
        return None
    return {"Overall severe": [(g, p) for g, p in cat if p.get("LABEL") in ("SLGT", "ENH", "MDT", "HIGH")],
            "Tornado": tor or [], "Wind": wind or [], "Hail": hail or []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--pool", type=int, required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--spc-lead", type=int, default=None)
    ap.add_argument("--single", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="models left out of the blend (consistency_blend.py --exclude)")
    ap.add_argument("--tag", default="", help="draw only the blend panel from consistency_raw_d<L><tag>.npz")
    a = ap.parse_args()
    O = f"{CASES_DIR}/{a.date}/output/"
    L, tag = a.pool, f"Day-{a.day}"  # runs are listed on the map (Models Used), so no "(Latest Runs)"
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    spc = plots.spc_lead_polys(a.date, conv, a.spc_lead)[0] if a.spc_lead else None
    spc_pan = (spc_hazard_panels(a.date, a.spc_lead) if a.spc_lead in (1, 2) else None) or spc  # 2x2 panels
    hc = json.load(open(f"{CASES_DIR}/calibration_hazards.json"))
    any14 = json.load(open(f"{CASES_DIR}/calibration_anysevere_14case.json"))
    cal = json.load(open(f"{CASES_DIR}/calibration_provisional.json"))["env_ci"]
    us = plot_hhwx.us_mask()
    at = lambda x, la, lo: float(x[int(round((la - 20) / .5)), int(round((lo + 130) / .5))])
    # consistency-weighted blend: calibrated and uncalibrated
    r = np.load(O + f"consistency_raw_d{L}{a.tag}.npz")
    psev_c = ch.calibrated(r["sev"], any14)
    # overall = the same 14-case any-severe calibration the hazards are capped at (Hunter, 31 Oct 2026: wind 15% was
    # larger than overall 15% while overall used the old 5-case fit; P(any severe) must be >= P(any hazard))
    sev = np.where(us, psev_c, 0)
    plot_hhwx.draw(sev, f"HHwx Prob Severe (Calibrated) {tag} (Consistency-Weighted Blend)", O + f"hhwx_d{a.day}latest_sev_cal.png", spc=spc)
    print(f"blend severe max {sev.max():.1f} OKC {at(sev, 35.5, -97.5):.1f}")
    bmaps = [("Overall severe", sev)]
    for k, nm in HZ:
        p = np.where(us, np.minimum(ch.calibrated(r[k], hc[k]), psev_c), 0)
        plot_hhwx.draw(p, f"HHwx Prob {nm} (Calibrated) {tag} (Consistency-Weighted Blend)", O + f"hhwx_d{a.day}latest_{k}_cal.png", spc=spc)
        bmaps.append((nm, p))
        print(f"blend {k} max {p.max():.1f} OKC {at(p, 35.5, -97.5):.1f}")
    res = np.load(O + "results.npz")
    draw_panel(bmaps, f"HHwx Calibrated Guidance {tag}" + {"_cam": ": CAMs + NAM 12 km", "_cam_nonam12": ": CAMs"}.get(a.tag, ""),
               O + f"hhwx_d{a.day}latest_panel_blend{a.tag}.png",
               spc_pan, a.spc_lead, at, runs=runs_text(res, L, [m for m in ["eps", "gefs", "hres", "gfs", "nam", "hrrr", "refs", "nam12", "rap", "rrfs"] if m not in a.exclude
                                                and (float(r[f"w_{m}"]) > 0 if f"w_{m}" in r.files else scoring.weights(L).get(m, 0) > 0)],
                                                (conv - timedelta(days=1)).strftime("%Y%m%d") + "03" if "mode" in r.files and str(r["mode"]) == "track" else ""))
    if a.tag:
        return
    for k, nm in (("sev", "Severe"),) + HZ:
        fig = plt.figure(figsize=(11, 7.4))
        ax = fig.add_axes([0.01, 0.1, 0.98, 0.84], projection=plots.PROJ)
        plots.base_ax(ax, "conus")
        cf = ax.contourf(plots.LON2, plots.LAT2, r[k], levels=LV, cmap=CM, norm=BoundaryNorm(LV, CM.N), transform=plots.PC)
        ax.clabel(ax.contour(plots.LON2, plots.LAT2, r[k], levels=LV[:-1], colors="#2d004b", linewidths=0.6, transform=plots.PC),
                  fontsize=8, fmt=lambda v: f"{v:g}")
        ax.set_title(f"HHwx {nm} Index (Uncalibrated) {tag} (Consistency-Weighted Blend)", fontsize=14, fontweight="bold")
        fig.colorbar(cf, cax=fig.add_axes([0.3, 0.05, 0.4, 0.022]), orientation="horizontal", ticks=LV[:-1]).set_label(
            "raw index (environment x storm initiation), 0-1; NOT a probability", fontsize=9)
        fig.savefig(O + f"hhwx_d{a.day}latest_{k}_index.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    # per model (pooled runs): severe and tornado
    res, z = np.load(O + "results.npz"), np.load(O + "ci_layers.npz")
    for m in NAMES:
        if f"{L}__{m}__p_model" not in z.files:
            continue
        lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
        p_ci = 0.6 * lay[0] + 0.4 * lay[1]
        psev = np.where(us, ch.calibrated(z[f"{L}__{m}__env*ci"], any14), 0)
        pt = np.where(us, np.minimum(ch.calibrated(res[f"tor|{L}|{m}|day"] * p_ci, hc["tor"]), psev), 0)
        plot_hhwx.draw(pt, f"HHwx Prob Tornado (Calibrated) {tag} ({NAMES[m]})", O + f"hhwx_d{a.day}latest_tor_{m}.png", spc=spc)
        plot_hhwx.draw(psev, f"HHwx Prob Severe (Calibrated) {tag} ({NAMES[m]})", O + f"hhwx_d{a.day}latest_sev_{m}.png", spc=spc)
        # one 2x2 image per model: overall, tornado, wind, hail (all calibrated, SPC outline on each)
        maps = [("Overall severe", psev)] + [(nm, np.where(us, np.minimum(ch.calibrated(res[f"{k}|{L}|{m}|day"] * p_ci, hc[k]), psev), 0))
                                             for k, nm in HZ]
        draw_panel(maps, f"HHwx Calibrated Guidance {tag}: {NAMES[m]}", O + f"hhwx_d{a.day}latest_panel_{m}.png", spc_pan, a.spc_lead, at,
                   runs=runs_text(res, L, [m]))
        print(f"{m}: severe max {psev.max():.1f} OKC {at(psev, 35.5, -97.5):.1f} | tornado max {pt.max():.1f} OKC {at(pt, 35.5, -97.5):.1f}"
              + "".join(f" | {nm} max {p.max():.1f}" for nm, p in maps[2:]))
    # one run alone
    if a.single:
        m, c = a.single.split(":")
        lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
        p_ci = 0.6 * lay[0] + 0.4 * lay[1]
        psev = np.where(us, ch.calibrated(res[f"v2|{L}|{m}|cyc{c}"] * p_ci, any14), 0)
        t1 = f"Day-{a.day} ({NAMES[m]} {c[8:]}Z, single run)"
        plot_hhwx.draw(psev, f"HHwx Prob Severe (Calibrated) {t1}", O + f"hhwx_d{a.day}_{m}{c[8:]}z_sev.png", spc=spc)
        smaps = [("Overall severe", psev)]
        for k, nm in HZ:
            p = np.where(us, np.minimum(ch.calibrated(res[f"{k}|{L}|{m}|cyc{c}"] * p_ci, hc[k]), psev), 0)
            plot_hhwx.draw(p, f"HHwx Prob {nm} (Calibrated) {t1}", O + f"hhwx_d{a.day}_{m}{c[8:]}z_{k}.png", spc=spc)
            smaps.append((nm, p))
            print(f"single {m} {c} {k} max {p.max():.1f} OKC {at(p, 35.5, -97.5):.1f}")
        draw_panel(smaps, f"HHwx Calibrated Guidance {t1}", O + f"hhwx_d{a.day}_{m}{c[8:]}z_panel.png", spc_pan, a.spc_lead, at,
                   runs=[(NAMES[m], f"{c[4:6]}/{c[6:8]} {c[8:]}Z")])


if __name__ == "__main__":
    main()

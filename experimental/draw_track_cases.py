"""EXPERIMENTAL: past-case maps with the track-record blend (Hunter, 1 Nov 2026).

    python experimental/draw_track_cases.py [--cases ...] [--procs 3]

Reads cases/<date>/output/consistency_raw_d<L>_track.npz (consistency_blend.py --mode track --tag _track) for every lead,
calibrates it like the real-time maps (overall: 14-case any-severe; hazards capped at overall), and draws the 2x2 panel
with SPC's outline (Days 8-3: 15% any-severe; Days 2-1: Slight+ and each hazard's own lines) and the day's reports.
Writes pages/pasttrack/maps/<date>_d<L>.png and pages/pasttrack/cases.json (weights, maxima, report hits).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate_hazards as ch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import redraw_latest as rl  # noqa: E402
from config import CASES_DIR, PROJECT_DIR  # noqa: E402

OUT = os.environ.get("POGS_TRACK_OUT", os.path.join(PROJECT_DIR, "pages", "pasttrack"))
MODELS = ["eps", "gefs", "hres", "gfs", "nam", "hrrr", "refs", "nam12", "rap", "rrfs"]
HZ = (("tor", "Tornado"), ("wind", "Wind"), ("hail", "Hail"))


def extent_for(rep, sev):
    """24 x 17 deg box centred on the reports (or our 5% area when there are few)."""
    pts = []
    if rep is not None and len(rep):
        pts = list(zip(rep["lat"], rep["lon"]))
    if len(pts) < 10:
        m = sev >= 5
        pts += list(zip(plots.LAT2[m], plots.LON2[m]))
    if not pts:
        return (-108, -86, 26, 42)
    m = sev >= 15  # our high area joins the reports, so a split day keeps both halves (2025-04-19 lost Texas)
    pts += list(zip(plots.LAT2[m], plots.LON2[m]))
    la_all, lo_all = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    la0, la1 = np.percentile(la_all, [3, 97])
    lo0, lo1 = np.percentile(lo_all, [3, 97])
    w = max(24.0, (lo1 - lo0) + 4.0)
    h = max(17.0, (la1 - la0) + 3.0, w * 17.0 / 24.0)
    w = max(w, h * 24.0 / 17.0)  # keep the 24:17 shape
    lo, la = 0.5 * (lo0 + lo1), 0.5 * (la0 + la1)
    lo = float(np.clip(lo, -126 + w / 2, -66 - w / 2)) if w < 60 else -96.0
    la = float(np.clip(la, 24 + h / 2, 50 - h / 2)) if h < 26 else 37.0
    return (lo - w / 2, lo + w / 2, la - h / 2, la + h / 2)


def hits(rep, field, thr, types=None):
    """Share of reports whose grid cell has field >= thr (reports of the given types)."""
    if rep is None or not len(rep):
        return None
    d = rep if types is None else rep[rep["type"].isin(types)]
    if not len(d):
        return None
    i = np.clip(np.round((d["lat"].to_numpy() - 20) / 0.5).astype(int), 0, field.shape[0] - 1)
    j = np.clip(np.round((d["lon"].to_numpy() + 130) / 0.5).astype(int), 0, field.shape[1] - 1)
    return round(float(np.mean(field[i, j] >= thr)), 2)


NAMES = {"eps": "ECMWF EPS", "gefs": "GEFS", "hres": "ECMWF HRES", "gfs": "GFS", "nam": "NAM 3 km", "hrrr": "HRRR",
         "refs": "REFS", "nam12": "NAM 12 km", "rap": "RAP", "rrfs": "RRFS"}
PANELS = ("Overall severe", "Tornado", "Wind", "Hail")


def draw_grid(rows, title, out, spc, extent, rep, dpi=72):
    """One tall image: row 0 = blend, then one row per model; columns overall / tornado / wind / hail, each with SPC's
    lines for that lead and the day's reports. Saved as a reduced-palette PNG (fits ~400 maps in one artifact)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    from PIL import Image
    import calibrate
    lv, cm = calibrate.SPC_LEVELS, calibrate.SPC_CMAP
    per_panel = isinstance(spc, dict)
    n = len(rows)
    H = 3.25 * n + 1.8
    fig, axes = plt.subplots(n, 4, figsize=(16, H), subplot_kw={"projection": plots.PROJ}, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 1.05 / H, bottom=0.9 / H, wspace=0.04, hspace=0.16)
    cf = None
    for r, (label, maps) in enumerate(rows):
        for c, (nm, p) in enumerate(maps):
            ax = axes[r, c]
            plots.base_ax(ax, "conus")
            ax.set_extent(list(extent), plots.PC)
            cf = ax.contourf(plots.LON2, plots.LAT2, p, levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), transform=plots.PC)
            polys = spc.get(nm, []) if per_panel else [(g, lab) for g, lab in (spc or []) if lab.get("LABEL") in ("0.15", "SLGT")]
            for geom, lab in polys:
                parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")]
                if str(lab.get("LABEL")).startswith("CIG") or lab.get("LABEL") == "SIGN":
                    ax.add_geometries(parts, crs=plots.PC, facecolor="none", edgecolor="#444444", hatch="///", linewidth=0.5, zorder=7)
                else:
                    ax.add_geometries(parts, crs=plots.PC, facecolor="none", edgecolor="black", linewidth=1.3, linestyle="--", zorder=8)
            if rep is not None and len(rep):
                sel = {"Tornado": ("tornado",), "Wind": ("wind",), "Hail": ("hail",)}.get(nm, ("tornado", "wind", "hail"))
                for typ, col, mk, sz in (("wind", "#1f5fd6", "o", 5), ("hail", "#1a9e3a", "o", 5), ("tornado", "#d00000", "v", 12)):
                    if typ in sel:
                        d = rep[rep["type"] == typ]
                        ax.scatter(d["lon"], d["lat"], s=sz, marker=mk, c=col, edgecolors="black", linewidths=0.2, transform=plots.PC, zorder=11)
            ins = (plots.LON2 >= extent[0]) & (plots.LON2 <= extent[1]) & (plots.LAT2 >= extent[2]) & (plots.LAT2 <= extent[3])
            v = float(np.where(ins, p, 0).max())
            ax.set_title(f"{nm.replace(' severe', '')}: {v:.0f}%" if v >= 1 else f"{nm.replace(' severe', '')}: <1%",
                         fontsize=12, fontweight="bold" if r == 0 else "normal")
        axes[r, 0].text(-0.04, 0.5, label, rotation=90, transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=12, fontweight="bold" if r == 0 else "normal")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1 - 0.3 / H)
    cax = fig.add_axes([0.30, 0.25 / H, 0.40, 0.18 / H])
    fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lv[:-1]).set_label("%")
    hd = [Line2D([], [], color="black", lw=1.4, ls="--"), Line2D([], [], color="#d00000", marker="v", lw=0, mec="black", ms=6),
          Line2D([], [], color="#1f5fd6", marker="o", lw=0, mec="black", ms=4), Line2D([], [], color="#1a9e3a", marker="o", lw=0, mec="black", ms=4)]
    fig.legend(hd, ["SPC outlook lines for this lead", "Tornado report", "Wind report", "Hail report"], loc="lower left",
               bbox_to_anchor=(0.01, 0.05 / H), ncol=2, fontsize=10, frameon=False)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    Image.open(out).convert("RGB").quantize(colors=96).save(out, optimize=True)


def model_maps(z, res, L, m, us, any14, hc):
    """One model's pooled calibrated maps (as redraw_latest's per-model panels)."""
    lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
    p_ci = 0.6 * lay[0] + 0.4 * lay[1]
    psev = np.where(us, ch.calibrated(z[f"{L}__{m}__env*ci"], any14), 0)
    return rl.union_cap([("Overall severe", psev)] + [(nm, np.where(us, np.minimum(ch.calibrated(res[f"{k}|{L}|{m}|day"] * p_ci, hc[k]), psev), 0))
                                                    for k, nm in HZ])


def do_case(date):
    out = os.path.join(CASES_DIR, date, "output")
    conv = datetime.strptime(date, "%Y-%m-%d")
    hc = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
    any14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
    us = plot_hhwx.us_mask()
    rep = plots.load_reports(date, conv)
    res = np.load(os.path.join(out, "results.npz"))
    z = np.load(os.path.join(out, "ci_layers.npz"))
    summary = {"date": date, "n_reports": int(len(rep)) if rep is not None else 0,
               "n_tor": int((rep["type"] == "tornado").sum()) if rep is not None else 0, "leads": {}}
    for L in range(8, 0, -1):
        p = os.path.join(out, f"consistency_raw_d{L}_track.npz")
        if not os.path.exists(p):
            continue
        r = np.load(p)
        sev = np.where(us, ch.calibrated(r["sev"], any14), 0)
        maps = rl.union_cap([("Overall severe", sev)] + [(nm, np.where(us, np.minimum(ch.calibrated(r[k], hc[k]), sev), 0)) for k, nm in HZ])
        sev = maps[0][1]
        spc = (rl.spc_hazard_panels(date, L) if L in (1, 2) else None) or plots.spc_lead_polys(date, conv, L)[0]
        w = {m: round(float(r[f"w_{m}"]), 3) for m in MODELS if f"w_{m}" in r.files}
        used = [m for m in MODELS if w.get(m, 0) > 0]
        db03 = (conv - timedelta(days=1)).strftime("%Y%m%d") + "03"
        runs = rl.runs_text(res, L, used, db03)
        png = os.path.join(OUT, "maps", f"{date}_d{L}.png")
        mods = sorted((m for m in MODELS if f"{L}__{m}__p_model" in z.files), key=lambda m: (-w.get(m, 0), MODELS.index(m)))
        nrun = {nm: rs.count(",") + rs.count(chr(10)) + 1 for nm, rs in rl.runs_text(res, L, mods, db03)}
        rows = [("Blend (track-record)", maps)]
        for m in mods:
            tag = f"weight {w[m]:.2f}" if w.get(m, 0) > 0 else "shown only"
            rows.append((f"{NAMES[m]}\n{tag}", model_maps(z, res, L, m, us, any14, hc)))
        draw_grid(rows, f"HHwx Day-{L} guidance for {conv:%a %d %b %Y}: blend and each model", png, spc,
                  extent_for(rep, sev), rep)
        tor = maps[1][1]
        summary["leads"][L] = {
            "weights": w, "max": {nm: round(float(f.max()), 1) for nm, f in maps},
            "hit15": hits(rep, sev, 15), "hit5": hits(rep, sev, 5), "tor_hit2": hits(rep, tor, 2, ["tornado"]),
            "tor_hit5": hits(rep, tor, 5, ["tornado"])}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    cases = a.cases or sorted(d for d in os.listdir(CASES_DIR) if d[:2] == "20" and len(d) == 10 and d != "2026-11-02"
                              and os.path.exists(os.path.join(CASES_DIR, d, "output", "consistency_raw_d4_track.npz")))
    with Pool(a.procs) as pool:
        out = pool.map(do_case, cases)
    jp = os.path.join(OUT, "cases.json")
    old = json.load(open(jp)) if os.path.exists(jp) and a.cases else []
    keep = {c["date"]: c for c in old}
    keep.update({c["date"]: c for c in out})
    json.dump(sorted(keep.values(), key=lambda c: c["date"]), open(jp, "w"), indent=1)
    print(f"{len(out)} cases -> {jp}")


if __name__ == "__main__":
    main()

"""Why did the score peak where it did? Ingredient-term diagnostics for a cached case.

    python diagnose_case.py 2025-06-16 --leads 3 2 1 [--profile v2] [--hour 0]

For each lead, every member in the pool is scored at one valid time (default 00Z),
and the v2 terms (instability, shear, lift, CIN, moisture) are averaged with the blend
weights. Writes to cases/<date>/output/diagnostics/:
  terms_D{lead}_{view}.png -- score, each term, and a "limiting ingredient" map
  points_D{lead}.md        -- term values at report clusters and at the score maximum
"""
import argparse
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import plots
import run_case
import scoring
from config import CASES_DIR, GRID_LATS, GRID_LONS, MODELS, VIEWS, case_setup, fh_available

TERMS = ["inst", "shear", "lift", "cin", "moist"]
TERM_LABEL = {"inst": "Instability", "shear": "Shear", "lift": "Lift", "cin": "CIN", "moist": "Moisture"}
TERM_COLORS = ["#d62728", "#9467bd", "#1f77b4", "#7f7f7f", "#2ca02c"]


def pooled_terms(lead_inits, vt):
    """Blend-weighted mean of score and term maps over all members of all models."""
    num = {k: 0.0 for k in ["score"] + TERMS}
    den = 0.0
    for m in MODELS:
        acc = {k: 0.0 for k in num}
        n = 0
        for init in lead_inits:
            fh = int((vt - init).total_seconds() // 3600)
            if not fh_available(m, init, fh):
                continue
            for mem in run_case.members_of(m):
                f = run_case.load(m, init, fh, mem)
                if f is None:
                    continue
                ing = scoring.ingredients(m, f)
                s, parts = scoring.score_v2(ing, fh, parts=True)
                acc["score"] += s
                for t in TERMS:
                    acc[t] += parts[t] if not np.isscalar(parts[t]) else np.full_like(s, parts[t])
                n += 1
        if n:
            w = scoring.BLEND_WEIGHTS[m]
            for k in num:
                num[k] = num[k] + w * acc[k] / n
            den += w
    return {k: v / den for k, v in num.items()}


def clusters(rep, min_n=3):
    """Report clusters on a 2-degree grid: (label, lat, lon, n, types)."""
    out = []
    rep = rep.assign(bi=(rep.lat // 2).astype(int), bj=(rep.lon // 2).astype(int))
    for (bi, bj), g in rep.groupby(["bi", "bj"]):
        if len(g) >= min_n:
            types = g.type.str[:4].value_counts().to_dict()
            st = g.state.mode().iat[0] if "state" in g and len(g.state.mode()) else ""
            out.append((f"{st} {g.lat.mean():.1f}N {g.lon.mean():.1f}", g.lat.mean(), g.lon.mean(), len(g), types))
    return sorted(out, key=lambda c: -c[3])


def value_at(field, la, lo):
    return float(field[int(round((la - GRID_LATS[0]) / 0.5)), int(round((lo - GRID_LONS[0]) / 0.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 2, 1])
    ap.add_argument("--profile", default="v2")
    ap.add_argument("--hour", type=int, default=0, help="valid hour (UTC) on the night of the convective day")
    a = ap.parse_args()
    scoring.set_profile(a.profile)
    case, conv_day = a.date, datetime.strptime(a.date, "%Y-%m-%d")
    out = os.path.join(CASES_DIR, case, "output", "diagnostics")
    os.makedirs(out, exist_ok=True)
    vt = datetime(conv_day.year, conv_day.month, conv_day.day) + timedelta(hours=24 + a.hour if a.hour < 12 else a.hour)
    rep = plots.load_reports(case, conv_day)
    cl = clusters(rep)
    _, pools = case_setup(conv_day, a.leads)
    for lead in a.leads:
        T = pooled_terms(pools[lead], vt)
        jmax = np.unravel_index(np.argmax(T["score"]), T["score"].shape)
        pts = [(f"Score maximum", GRID_LATS[jmax[0]], GRID_LONS[jmax[1]], 0, {})] + cl
        lines = [f"# {case} Day {lead}: v2 terms at {vt:%d/%HZ} (profile {a.profile})", "",
                 "Blend-weighted means over all members. Terms multiply (CIN = 1 for ECMWF, no CIN in open data).", "",
                 "| Location | reports | score | " + " | ".join(TERM_LABEL[t] for t in TERMS) + " | limiting |",
                 "|---|---|---|" + "---|" * len(TERMS) + "---|"]
        for lab, la, lo, n, types in pts:
            vals = {t: value_at(T[t], la, lo) for t in TERMS}
            lim = min(vals, key=lambda t: vals[t] / (1.1 if t == "lift" else 1.0))
            rtxt = ", ".join(f"{v} {k}" for k, v in types.items()) if n else "—"
            lines.append(f"| {lab} | {rtxt} | {value_at(T['score'], la, lo):.2f} | "
                         + " | ".join(f"{vals[t]:.2f}" for t in TERMS) + f" | {TERM_LABEL[lim]} |")
        with open(os.path.join(out, f"points_D{lead}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines) + "\n")

        # limiting term: smallest term (lift scaled to its 1.1 max) where the score is meaningful
        stack = np.stack([T[t] / (1.1 if t == "lift" else 1.0) for t in TERMS])
        lim = np.argmin(stack, axis=0).astype(float)
        lim[T["score"] < 0.05] = np.nan
        for view in VIEWS:
            fig, axes = plt.subplots(2, 4, figsize=(24, 10.5), subplot_kw={"projection": plots.PROJ})
            ax = axes[0, 0]
            plots.base_ax(ax, view)
            plots.draw_prob(ax, T["score"])
            plots.draw_reports(ax, rep, 5)
            ax.set_title(f"v2 score at {vt:%d/%HZ} (pooled, {a.profile})", fontsize=10)
            for ax, t in zip(list(axes[0, 1:]) + list(axes[1, :2]), TERMS):
                plots.base_ax(ax, view)
                pm = ax.pcolormesh(plots.LON2, plots.LAT2, T[t], cmap="viridis", vmin=0, vmax=1.1 if t == "lift" else 1,
                                   transform=plots.PC, shading="auto")
                plots.draw_reports(ax, rep, 4)
                ax.set_title(f"{TERM_LABEL[t]} term", fontsize=10)
                fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
            ax = axes[1, 2]
            plots.base_ax(ax, view)
            ax.pcolormesh(plots.LON2, plots.LAT2, lim, cmap=ListedColormap(TERM_COLORS), vmin=-0.5, vmax=4.5,
                          transform=plots.PC, shading="auto")
            plots.draw_reports(ax, rep, 4)
            ax.set_title("Limiting ingredient (where score ≥ 5%)", fontsize=10)
            for i, t in enumerate(TERMS):
                ax.text(0.01, 0.95 - 0.06 * i, TERM_LABEL[t], color=TERM_COLORS[i], transform=ax.transAxes,
                        fontsize=9, fontweight="bold", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            axes[1, 3].set_visible(False)
            fig.suptitle(f"{case} Day {lead}: what drives and limits the score (valid {vt:%d/%HZ})", fontsize=13)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04, wspace=0.05, hspace=0.12)
            fig.savefig(os.path.join(out, f"terms_D{lead}_{view}.png"), dpi=100)
            plt.close(fig)


if __name__ == "__main__":
    main()

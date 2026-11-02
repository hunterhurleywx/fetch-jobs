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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--target", type=float, nargs=2, default=None)
    ap.add_argument("--extent", type=float, nargs=4, default=[-101.3, -96.5, 34.9, 37.9], help="lon0 lon1 lat0 lat1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.npz)
    out = a.out or os.path.dirname(a.npz)
    LO, LA = np.meshgrid(z["lon"], z["lat"])
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS + [10000], cmap.N)
    panels = [(h, lab) for h, lab in HOURS if f"median_{h}" in z.files]

    def draw(ax, h, lab):
        plots.base_ax(ax, "conus")
        ax.set_extent(a.extent, plots.PC)
        m = z[f"median_{h}"]
        cf = ax.pcolormesh(LO, LA, np.ma.masked_less(m, LEVELS[0]), cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
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
        cf = draw(ax, h, f"{TITLE}\nValid {lab}")
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.022])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS, label="J/kg").ax.tick_params(labelsize=9)
        fig.savefig(os.path.join(out, f"cape3_zoom_{h}z.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), subplot_kw={"projection": plots.PROJ})
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (h, lab) in zip(axes.ravel(), panels):
        cf = draw(ax, h, lab)
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
    fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS, label="J/kg")
    fig.suptitle(TITLE, fontsize=15, fontweight="bold")
    p = os.path.join(out, "cape3_zoom_panel.png")
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()

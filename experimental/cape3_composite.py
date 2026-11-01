"""0-3 km CAPE composite from the storm-scale / rapid-refresh models (Hunter, 1 Nov 2026). Cloud step (reads model
GRIB byte ranges); display only.

    python experimental/cape3_composite.py 2026-11-02 --runs hrrr:2026110118 rap:2026110121 rrfs:2026110118 \
        --hours 20 21 22 23 0

Field: CAPE "0-3000 m above ground" from HRRR (wrfprs), RAP (awp130, 13 km) and RRFS (2dfld). REFS members carry only
surface CAPE (checked 1 Nov), so REFS cannot enter. Each model is read on its own grid, cropped to the box, and put on
a common 0.03 deg latitude/longitude grid by nearest neighbour. Composite = MEDIAN of the models at each point: with
three models the median keeps realistic magnitudes and is not pulled by one model's outlier maximum, where a mean
would smear it (the mean is saved too). Colours: Pivotal Weather-style 3CAPE scale (approximate, after
experimental/cape_palette.json). Writes <POGS_CAPE3_OUT>/cape3_<HH>z.png per hour, a 5-panel, and cape3_<date>.npz.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import eccodes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch  # noqa: E402
import plots  # noqa: E402

SPEC = {"cape3": ("CAPE", "0-3000 m above ground", "inst")}
LEVELS = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500]
COLORS = ["#d9e3ee", "#a9c6e6", "#6fa2dd", "#3f78c8", "#2fa36b", "#64c44d", "#b5dd41", "#f2e93b", "#f7c23a",
          "#f39335", "#e95e2c", "#d22a2a", "#b0173f", "#d23a90"]
NAMES = {"hrrr": "HRRR", "rap": "RAP", "rrfs": "RRFS"}


def url_of(m, init, fh):
    return fetch.RRFS_2D_URL(init, fh) if m == "rrfs" else fetch.CAM_URL[m](init, fh)


def read(m, init, fh, box):
    url = url_of(m, init, fh)
    (s, e, (_, sub)), = fetch.cam_ranges(fetch.http_get(url + ".idx").decode(), SPEC, fh)
    h = fetch._cam_handle(fetch.http_get(url, (s, e)), sub)
    try:
        nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        v = eccodes.codes_get_values(h).astype(np.float32).reshape(ny, nx)
        lat = eccodes.codes_get_array(h, "latitudes").reshape(ny, nx)
        lon = eccodes.codes_get_array(h, "longitudes").reshape(ny, nx)
    finally:
        eccodes.codes_release(h)
    lon = np.where(lon > 180, lon - 360, lon)
    inside = (lat >= box[0] - 0.5) & (lat <= box[1] + 0.5) & (lon >= box[2] - 0.5) & (lon <= box[3] + 0.5)
    return v[inside], lat[inside], lon[inside]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--runs", nargs="+", required=True, help="model:YYYYMMDDHH, e.g. hrrr:2026110118")
    ap.add_argument("--hours", type=int, nargs="+", default=[20, 21, 22, 23, 0])
    ap.add_argument("--box", type=float, nargs=4, default=[30.0, 41.0, -104.0, -90.0])
    a = ap.parse_args()
    out = os.environ.get("POGS_CAPE3_OUT", "cape3_out")
    os.makedirs(out, exist_ok=True)
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    runs = [(r.split(":")[0], datetime.strptime(r.split(":")[1], "%Y%m%d%H")) for r in a.runs]
    valid = [conv + timedelta(hours=h + (24 if h < 12 else 0)) for h in a.hours]
    glat = np.arange(a.box[0], a.box[1] + 1e-6, 0.03)
    glon = np.arange(a.box[2], a.box[3] + 1e-6, 0.03)
    GLO, GLA = np.meshgrid(glon, glat)

    def one(job):
        m, init, v = job
        fh = int((v - init).total_seconds() // 3600)
        try:
            vals, la, lo = read(m, init, fh, a.box)
        except Exception as e:
            print(f"skip {m} {init:%d/%HZ} f{fh:02d}: {str(e)[:100]}", flush=True)
            return job, None
        d, i = cKDTree(np.c_[la, lo * np.cos(np.deg2rad(35))]).query(np.c_[GLA.ravel(), GLO.ravel() * np.cos(np.deg2rad(35))])
        g = vals[i].reshape(GLA.shape)
        g[d.reshape(GLA.shape) > (0.15 if m == "rap" else 0.06)] = np.nan
        return job, g

    jobs = [(m, i, v) for m, i in runs for v in valid]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, jobs))
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS + [10000], cmap.N)
    save = {"lat": glat, "lon": glon}
    panels = []
    for v in valid:
        fields = [(m, res[(m, i, v)]) for m, i in runs if res.get((m, i, v)) is not None]
        if not fields:
            continue
        stack = np.stack([f for _, f in fields])
        med, mean = np.nanmedian(stack, axis=0), np.nanmean(stack, axis=0)
        save[f"median_{v:%H}"], save[f"mean_{v:%H}"] = med.astype(np.float32), mean.astype(np.float32)
        used = ", ".join(f"{NAMES[m]} {i:%d/%HZ} f{int((v - i).total_seconds() // 3600):02d}" for m, i in runs if res.get((m, i, v)) is not None)
        panels.append((v, med, used))
        fig = plt.figure(figsize=(11, 8.2))
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.82], projection=plots.PROJ)
        plots.base_ax(ax, "conus")
        ax.set_extent([a.box[2], a.box[3], a.box[0], a.box[1]], plots.PC)
        cf = ax.pcolormesh(GLO, GLA, np.ma.masked_less(med, LEVELS[0]), cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
        ax.contour(GLO, GLA, np.nan_to_num(med), levels=[100, 200], colors="black", linewidths=0.6, transform=plots.PC)
        ax.set_title(f"0-3 km CAPE (J/kg), median of {len(fields)} models | valid {v:%a %d %b %HZ} ({(v.hour - 6) % 24:d}:00 CST)\n{used}",
                     fontsize=11, loc="left")
        cax = fig.add_axes([0.15, 0.05, 0.7, 0.022])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS).ax.tick_params(labelsize=9)
        fig.savefig(os.path.join(out, f"cape3_{v:%H}z.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        i0, j0 = int(round((35.5 - a.box[0]) / 0.03)), int(round((-97.5 - a.box[2]) / 0.03))
        print(f"{v:%HZ}: {used} | max median {np.nanmax(med):.0f} J/kg, OKC median {med[i0, j0]:.0f} mean {mean[i0, j0]:.0f}", flush=True)
    if panels:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={"projection": plots.PROJ})
        for ax in axes.ravel()[len(panels):]:
            ax.set_visible(False)
        for ax, (v, med, used) in zip(axes.ravel(), panels):
            plots.base_ax(ax, "conus")
            ax.set_extent([a.box[2], a.box[3], a.box[0], a.box[1]], plots.PC)
            cf = ax.pcolormesh(GLO, GLA, np.ma.masked_less(med, LEVELS[0]), cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
            ax.set_title(f"{v:%HZ} ({(v.hour - 6) % 24:d}:00 CST)", fontsize=12, fontweight="bold")
        cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS)
        fig.suptitle(f"0-3 km CAPE (J/kg), median of HRRR, RAP and RRFS, {conv:%a %d %b %Y} | runs: "
                     + ", ".join(f"{NAMES[m]} {i:%d/%HZ}" for m, i in runs), fontsize=13)
        fig.savefig(os.path.join(out, "cape3_panel.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    np.savez_compressed(os.path.join(out, f"cape3_{a.date}.npz"), **save)


if __name__ == "__main__":
    main()

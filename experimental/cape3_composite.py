"""0-3 km CAPE composite from the storm-scale / rapid-refresh models (Hunter, 1 Nov 2026). Cloud step (reads model
GRIB byte ranges); display only.

    python experimental/cape3_composite.py 2026-11-02 --runs hrrr:2026110118 rap:2026110121 rrfs:2026110118 \
        --hours 20 21 22 23 0

Field (default --source ml, Hunter 1 Nov: "like mesoanalysis uses"): SPC mesoanalysis 3CAPE = CAPE below 3 km AGL for
the 100 mb mixed-layer parcel, computed here from each model's own profiles (surface PRES/HGT/2 m T/2 m q + TMP, HGT,
SPFH or RH every 25 mb from 1000 to 500 mb) with sounding.py's parcel code (lowest-100-hPa mean theta and mixing
ratio, pseudo-adiabatic ascent, virtual-temperature buoyancy, positive area between LFC and EL below 3 km).
--source grib uses the models' own "CAPE 0-3000 m above ground" field instead (parcel definition not documented).
Models: HRRR (wrfprs), RAP (awp130, 13 km), RRFS (prslev + 2dfld). REFS members have no 0-3 km CAPE and no full
humidity profiles in their files (checked 1 Nov), so REFS cannot enter. Each model is read on its own grid, cropped to the box, and put on
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
LEVELS = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350]  # SPC/Pivotal-style 3CAPE range (to 350+)
COLORS = ["#d9e3ee", "#a9c6e6", "#6fa2dd", "#3f78c8", "#2fa36b", "#64c44d", "#b5dd41", "#f2e93b", "#f7c23a",
          "#f39335", "#e95e2c", "#d22a2a", "#b0173f", "#d23a90"]
NAMES = {"hrrr": "HRRR", "rap": "RAP", "rrfs": "RRFS"}


def url_of(m, init, fh):
    return fetch.RRFS_2D_URL(init, fh) if m == "rrfs" else fetch.CAM_URL[m](init, fh)


PLEV = list(range(1000, 499, -25))


def _native(blobs, box):
    """Decode GRIB messages {key: (bytes, sub)} on one grid -> {key: values in box}, lat, lon."""
    out, sel, lat, lon = {}, None, None, None
    for k, (b, sub) in blobs.items():
        h = fetch._cam_handle(b, sub)
        try:
            v = eccodes.codes_get_values(h).astype(np.float32)
            if eccodes.codes_get(h, "bitmapPresent"):
                v[v == eccodes.codes_get(h, "missingValue")] = np.nan
            if sel is None:
                la = eccodes.codes_get_array(h, "latitudes")
                lo = eccodes.codes_get_array(h, "longitudes")
                lo = np.where(lo > 180, lo - 360, lo)
                sel = (la >= box[0] - 0.5) & (la <= box[1] + 0.5) & (lo >= box[2] - 0.5) & (lo <= box[3] + 0.5)
                lat, lon = la[sel], lo[sel]
        finally:
            eccodes.codes_release(h)
        out[k] = v[sel]
    return out, lat, lon


def _get(url, spec, fh, optional):
    wanted = fetch.cam_ranges(fetch.http_get(url + ".idx").decode(), spec, fh, optional)
    rngs = sorted({(s_, e_) for s_, e_, _ in wanted})
    with ThreadPoolExecutor(8) as ex:
        data = dict(zip(rngs, ex.map(lambda r: fetch.http_get(url, r), rngs)))
    return {k: (data[(s_, e_)], sub) for s_, e_, (k, sub) in wanted}


def read_ml(m, init, fh, box):
    """100 mb mixed-layer CAPE below 3 km AGL (SPC mesoanalysis 3CAPE) from the model's own profiles."""
    import sounding as snd
    hum = "RH" if m == "rap" else "SPFH"
    sfc = {"ps": ("PRES", "surface", "inst"), "zs": ("HGT", "surface", "inst"),
           "t2": ("TMP", "2 m above ground", "inst"), "q2": ("SPFH", "2 m above ground", "inst")}
    lev = {f"{v}{p}": (g, f"{p} mb", "inst") for p in PLEV for v, g in (("t", "TMP"), ("z", "HGT"), ("h", hum))}
    if m == "rrfs":
        blobs = _get(fetch.RRFS_2D_URL(init, fh), sfc, fh, set())
        f2, lat, lon = _native(blobs, box)
        f3, la3, lo3 = _native(_get(fetch.CAM_URL["rrfs"](init, fh), lev, fh, set(lev)), box)
        if la3.size != lat.size:
            raise ValueError("RRFS prslev and 2dfld grids differ")
        f = {**f2, **f3}
    else:
        f, lat, lon = _native(_get(fetch.CAM_URL[m](init, fh), {**sfc, **lev}, fh, set(lev)), box)
    ps = f["ps"] / 100.0
    plev = [p for p in PLEV if f"t{p}" in f and f"z{p}" in f and f"h{p}" in f]
    P_ = np.array(plev, float)[:, None]
    T_ = np.stack([f[f"t{p}"] for p in plev])
    Z_ = np.stack([f[f"z{p}"] for p in plev]) - f["zs"][None]
    if hum == "RH":
        e = np.clip(np.stack([f[f"h{p}"] for p in plev]), 1, 100) / 100.0 * snd.es_hpa(T_)
        R_ = snd.EPS * e / (P_ - e)
    else:
        q = np.stack([f[f"h{p}"] for p in plev])
        R_ = q / (1 - q)
    r2 = f["q2"] / (1 - f["q2"])
    above = (P_ < ps[None]) & (Z_ > 20.0)
    zz = np.concatenate([np.zeros((1, ps.size)), np.where(above, Z_, 1e9)])
    tt = np.concatenate([f["t2"][None], T_])
    lp = np.concatenate([np.log(ps)[None], np.log(np.broadcast_to(P_, T_.shape))])
    rr = np.concatenate([r2[None], R_])
    order = np.argsort(zz, axis=0)
    zz, tt, lp, rr = (np.take_along_axis(x, order, 0) for x in (zz, tt, lp, rr))
    Hs = np.arange(0.0, 3501.0, 100.0)
    cols = np.arange(ps.size)
    prof = {}
    for name, arr in (("T", tt), ("LNP", lp), ("R", rr)):
        res = np.empty((ps.size, Hs.size))
        for k, h in enumerate(Hs):
            j = np.clip((zz <= h).sum(0), 1, zz.shape[0] - 1)
            z0, z1 = zz[j - 1, cols], zz[j, cols]
            w = np.clip((h - z0) / np.maximum(z1 - z0, 1e-3), 0, 1)
            res[:, k] = arr[j - 1, cols] + w * (arr[j, cols] - arr[j - 1, cols])
        prof[name] = res
    T, P, R = prof["T"], np.exp(prof["LNP"]), np.clip(prof["R"], 1e-6, None)
    snd.H, snd.DZ = Hs, 100.0  # parcel_stats integrates on the module's height grid
    ml = P >= (P[:, :1] - 100.0)
    th = T * (1000.0 / P) ** snd.KAPPA
    th_ml, r_ml = (th * ml).sum(1) / ml.sum(1), (R * ml).sum(1) / ml.sum(1)
    t_ml = th_ml * (P[:, 0] / 1000.0) ** snd.KAPPA
    st = snd.parcel_stats(T, P, R, np.zeros(ps.size, int), t_ml, r_ml, P[:, 0])
    return st["cape3"].astype(np.float32), lat, lon


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


def mark(ax, target):
    if target:
        ax.plot(target[1], target[0], marker="+", ms=22, mew=2.4, color="black", transform=plots.PC, zorder=9)
        ax.plot(target[1], target[0], marker="o", ms=16, mfc="none", mew=2.0, color="black", transform=plots.PC, zorder=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--runs", nargs="+", required=True, help="model:YYYYMMDDHH, e.g. hrrr:2026110118")
    ap.add_argument("--hours", type=int, nargs="+", default=[20, 21, 22, 23, 0])
    ap.add_argument("--box", type=float, nargs=4, default=[32.5, 38.5, -102.5, -94.0])
    ap.add_argument("--source", choices=["ml", "grib"], default="ml")
    ap.add_argument("--target", type=float, nargs=2, default=None, help="lat lon of a target marker")
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
            vals, la, lo = (read_ml if a.source == "ml" else read)(m, init, fh, a.box)
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
        for (m, i), f_ in zip([(m, i) for m, i in runs if res.get((m, i, v)) is not None], [f for _, f in fields]):
            save[f"{m}_{v:%H}"] = f_.astype(np.float16)
            save[f"{m}_run"] = np.array(f"{i:%Y%m%d%H}")
        used = ", ".join(f"{NAMES[m]} {i:%d/%HZ} f{int((v - i).total_seconds() // 3600):02d}" for m, i in runs if res.get((m, i, v)) is not None)
        panels.append((v, med, used))
        fig = plt.figure(figsize=(11, 8.2))
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.82], projection=plots.PROJ)
        plots.base_ax(ax, "conus")
        ax.set_extent([a.box[2], a.box[3], a.box[0], a.box[1]], plots.PC)
        cf = ax.pcolormesh(GLO, GLA, np.ma.masked_less(med, LEVELS[0]), cmap=cmap, norm=norm, transform=plots.PC, shading="auto")
        ax.contour(GLO, GLA, np.nan_to_num(med), levels=[100, 200], colors="black", linewidths=0.6, transform=plots.PC)
        mark(ax, a.target)
        ax.set_title(f"{"100 mb ML 0-3 km CAPE (SPC 3CAPE)" if a.source == "ml" else "0-3 km CAPE (model field)"} (J/kg), median of {len(fields)} models | valid {v:%a %d %b %HZ} ({(v.hour - 6) % 24:d}:00 CST)\n{used}",
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
            mark(ax, a.target)
        cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS)
        fig.suptitle(f"100 mb ML 0-3 km CAPE (J/kg, SPC 3CAPE), median of HRRR, RAP and RRFS, {conv:%a %d %b %Y} | runs: "
                     + ", ".join(f"{NAMES[m]} {i:%d/%HZ}" for m, i in runs), fontsize=13)
        fig.savefig(os.path.join(out, "cape3_panel.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    np.savez_compressed(os.path.join(out, f"cape3_{a.date}.npz"), **save)


if __name__ == "__main__":
    main()

"""Which RAP set-up reproduces SPC mesoanalysis "0-3 km MLCAPE and Surface Vorticity"? (Hunter, 2 Nov 2026; display
test, cloud step). SPC's hourly analysis starts from the RAP 1-h forecast on a 40 km grid, so we draw the RAP run that
is the first guess for the SPC hour, in SPC's style (red 3CAPE contours 25-150 J/kg, blue positive surface vorticity
every 1e-5 /s), four ways:
  A  RAP 13 km (awp130): its own "CAPE 0-3000 m" field; vorticity on the 13 km grid, no smoothing (HHwx archive, RAP)
  B  RAP 13 km: our 100 mb mixed-layer 3CAPE; vorticity on 13 km smoothed to ~40 km (3 x 3 cells)
  C  RAP 40 km (awp252): our 100 mb mixed-layer 3CAPE from its 25 mb profiles; vorticity on the 40 km grid (SPC-like)
  D  RAP 40 km vorticity with RAP 13 km's own "CAPE 0-3000 m" field
    python experimental/spc_match.py --init 2026110123 --fh 1 [--extent -124 -101 29.5 42.5]
Writes <POGS_SPCM_OUT>/spc_match_<valid>.png and .npz (each variant on a 0.1 deg grid).
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fetch  # noqa: E402
import cape3_composite as cc  # noqa: E402

fetch.CAM_URL["rap40"] = lambda i, fh: f"https://noaa-rap-pds.s3.amazonaws.com/rap.{i:%Y%m%d}/rap.t{i:%H}z.awp252pgrbf{fh:02d}.grib2"
fetch.CAM_LAMBERT["rap40"] = (25.0, 265.0)
CAPE_LV = [25, 50, 75, 100, 125, 150, 200, 250]
VORT_LV = list(range(1, 21))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--fh", type=int, default=1)
    ap.add_argument("--extent", type=float, nargs=4, default=[-124.0, -101.0, 29.5, 42.5])
    a = ap.parse_args()
    out = os.environ.get("POGS_SPCM_OUT", "spcm_out")
    os.makedirs(out, exist_ok=True)
    init = datetime.strptime(a.init, "%Y%m%d%H")
    valid = init + timedelta(hours=a.fh)
    box = [a.extent[2] - 1, a.extent[3] + 1, a.extent[0] - 1, a.extent[1] + 1]
    glat = np.arange(box[0], box[1] + 1e-6, 0.1)
    glon = np.arange(box[2], box[3] + 1e-6, 0.1)
    GLO, GLA = np.meshgrid(glon, glat)

    def regrid(vals, la, lo, maxd):
        d, i = cKDTree(np.c_[la, lo * np.cos(np.deg2rad(36))]).query(np.c_[GLA.ravel(), GLO.ravel() * np.cos(np.deg2rad(36))])
        g = vals[i].reshape(GLA.shape).astype(float)
        g[d.reshape(GLA.shape) > maxd] = np.nan
        return g

    f = {}
    v, la, lo = cc.read("rap", init, a.fh, box)
    f["cape_field13"] = regrid(v, la, lo, 0.2)
    r = cc.read_ml("rap", init, a.fh, box)
    f["cape_ml13"] = regrid(r[0], r[1], r[2], 0.2)
    r = cc.read_ml("rap40", init, a.fh, box)
    f["cape_ml40"] = regrid(r[0], r[1], r[2], 0.5)
    z, la, lo = cc.sfc_vort_native("rap", init, a.fh, box)  # 13 km, no smoothing (archive rule: 19 km / 13 km -> 1)
    f["vort13"] = regrid(z, la, lo, 0.2)
    f["vort13s"] = uniform_filter(np.nan_to_num(f["vort13"]), size=4, mode="nearest")  # 4 x 0.1 deg ~ 40 km
    z, la, lo = cc.sfc_vort_native("rap40", init, a.fh, box)
    f["vort40"] = regrid(z, la, lo, 0.5)
    np.savez_compressed(os.path.join(out, f"spc_match_{valid:%Y%m%d%H}.npz"), lat=glat, lon=glon,
                        **{k: x.astype(np.float32) for k, x in f.items()})
    variants = [("A", "RAP 13 km: own 0-3km CAPE field; 13 km vorticity (HHwx archive)", "cape_field13", "vort13"),
                ("B", "RAP 13 km: 100mb ML 3CAPE; 13 km vorticity smoothed to ~40 km", "cape_ml13", "vort13s"),
                ("C", "RAP 40 km: 100mb ML 3CAPE; 40 km vorticity (SPC grid)", "cape_ml40", "vort40"),
                ("D", "RAP 40 km vorticity + RAP 13 km own 0-3km CAPE field", "cape_field13", "vort40")]
    proj = ccrs.LambertConformal(central_longitude=-112, standard_parallels=(33, 45))
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5), subplot_kw={"projection": proj})
    for ax, (tag, lab, kc, kv) in zip(axes.ravel(), variants):
        ax.set_extent(a.extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#999", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#999", linewidth=0.7)
        cv = ax.contour(GLO, GLA, np.nan_to_num(f[kv]), levels=VORT_LV, colors="#29a8e0", linewidths=0.7,
                        transform=ccrs.PlateCarree())
        ax.clabel(cv, fontsize=6, fmt="%d")
        ccape = ax.contour(GLO, GLA, np.nan_to_num(f[kc]), levels=CAPE_LV, colors="#d01010",
                           linewidths=[0.6, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4], transform=ccrs.PlateCarree())
        ax.clabel(ccape, fontsize=7, fmt="%d")
        ax.set_title(f"{tag}. {lab}", fontsize=11)
        print(f"{tag}: 3CAPE max {np.nanmax(f[kc]):.0f} J/kg, vorticity max {np.nanmax(f[kv]):.1f} e-5/s", flush=True)
    fig.suptitle(f"RAP {init:%d/%HZ} f{a.fh:02d} valid {valid:%Y-%m-%d %HZ}: 0-3 km CAPE (red, J/kg) and surface "
                 f"vorticity (blue, x1e-5 /s), SPC mesoanalysis style", fontsize=14)
    p = os.path.join(out, f"spc_match_{valid:%Y%m%d%H}.png")
    fig.savefig(p, dpi=90, bbox_inches="tight")
    print("->", p)


if __name__ == "__main__":
    main()

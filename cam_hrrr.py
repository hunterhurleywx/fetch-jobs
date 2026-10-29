"""HRRR (3 km, hourly runs) for a case: 2-D storm fields on a regional crop + point soundings.

    python cam_hrrr.py 2026-04-13 --runs 10 11 12 13 14 15 16 17 18 19 20 --valid 19-05 --snd 21 0 3

Byte-range downloads via the .idx files from the AWS archive (noaa-hrrr-bdp-pds).
Cache: <POGS_CACHE>/hrrr/<YYYYMMDDHH>/
  sfc_f{fh:02d}.npz  crop of REFC, MXUPHL 2-5 km, MAXUVV, CAPE/CIN (sfc, ML 90-0, MU 255-0),
                     HLCY 0-1/0-3 km, 0-1/0-6 km shear, storm motion, 10 m wind, 2 m T/Td, sfc pressure, 1-h precip
  snd_f{fh:02d}.npz  profiles (T, Td, u, v, z on 25 hPa levels + surface values) at POINTS
  grid.npz           crop latitude/longitude
The crop is the case's region (CROP), not CONUS: the analysis looks at one area.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta

import eccodes
import numpy as np

from config import CACHE_DIR
from fetch import Missing, _merge, http_get

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
CROP = (34.0, 47.0, -101.0, -86.0)  # lat0, lat1, lon0, lon1
POINTS = {  # name: (lat, lon)
    "TOP": (39.07, -95.63),            # Topeka RAOB site
    "Ottawa": (38.62, -95.27),         # EF2 0023Z
    "CottonwoodFalls": (38.37, -96.54),  # first KS report 2155Z (dryline initiation)
    "SE_MN": (44.0, -93.0),
}
SFC = {  # key: (var, level, kind)  kind: inst | max (fh-1..fh window) | acc1 (1-h bucket)
    "refc": ("REFC", "entire atmosphere", "inst"),
    "uh25": ("MXUPHL", "5000-2000 m above ground", "max"),
    "maxuvv": ("MAXUVV", "100-1000 mb above ground", "max"),
    "sbcape": ("CAPE", "surface", "inst"), "sbcin": ("CIN", "surface", "inst"),
    "mlcape": ("CAPE", "90-0 mb above ground", "inst"), "mlcin": ("CIN", "90-0 mb above ground", "inst"),
    "mucape": ("CAPE", "255-0 mb above ground", "inst"),
    "srh1": ("HLCY", "1000-0 m above ground", "inst"), "srh3": ("HLCY", "3000-0 m above ground", "inst"),
    "ush1": ("VUCSH", "0-1000 m above ground", "inst"), "vsh1": ("VVCSH", "0-1000 m above ground", "inst"),
    "ush6": ("VUCSH", "0-6000 m above ground", "inst"), "vsh6": ("VVCSH", "0-6000 m above ground", "inst"),
    "ustm": ("USTM", "0-6000 m above ground", "inst"), "vstm": ("VSTM", "0-6000 m above ground", "inst"),
    "u10": ("UGRD", "10 m above ground", "inst"), "v10": ("VGRD", "10 m above ground", "inst"),
    "t2": ("TMP", "2 m above ground", "inst"), "td2": ("DPT", "2 m above ground", "inst"),
    "psfc": ("PRES", "surface", "inst"), "apcp1": ("APCP", "surface", "acc1"),
}
PRS_LEVELS = list(range(1000, 99, -25))
PRS = {f"{v.lower()}{p}": (v, f"{p} mb", "inst") for v in ("TMP", "DPT", "UGRD", "VGRD", "HGT") for p in PRS_LEVELS}

_crop = {}


def url(init, fh, kind):
    return f"{BUCKET}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrf{kind}f{fh:02d}.grib2"


def idx_ranges(idx_text, spec, fh):
    lines = [l.split(":") for l in idx_text.strip().split("\n")]
    offs = [int(p[1]) for p in lines]
    want = {"inst": "anl" if fh == 0 else f"{fh} hour fcst", "max": f"{fh - 1}-{fh} hour max fcst",
            "acc1": f"{fh - 1}-{fh} hour acc fcst"}
    found = {}
    for i, p in enumerate(lines):
        end = offs[i + 1] - 1 if i + 1 < len(lines) else None
        for key, (var, lev, kind) in spec.items():
            if p[3] == var and p[4] == lev and p[5] == want[kind] and key not in found:
                found[key] = (offs[i], end, key)
    return [v for v in found.values() if v[1] is not None]


def _decode(buf, need_latlon=False):
    """Full Lambert field (nj, ni), plus its lat/lon when asked (only the first message)."""
    h = eccodes.codes_new_from_message(buf)
    try:
        ni, nj = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        vals = eccodes.codes_get_values(h).astype(np.float32).reshape(nj, ni)
        latlon = None
        if need_latlon:
            lat = eccodes.codes_get_array(h, "latitudes").reshape(nj, ni)
            lon = eccodes.codes_get_array(h, "longitudes").reshape(nj, ni)
            latlon = (lat, np.where(lon > 180, lon - 360, lon))
    finally:
        eccodes.codes_release(h)
    return vals, latlon


def _setup_crop(lat, lon):
    la0, la1, lo0, lo1 = CROP
    inside = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    rows, cols = np.where(inside)
    _crop["sl"] = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    _crop["lat"], _crop["lon"] = lat[_crop["sl"]], lon[_crop["sl"]]
    _crop["pts"] = {n: np.unravel_index(np.argmin((lat - a) ** 2 + ((lon - o) * np.cos(np.radians(a))) ** 2),
                                        lat.shape) for n, (a, o) in POINTS.items()}


def grab(init, fh, kind, spec, pool):
    """Download the wanted messages; return {key: full field}."""
    u = url(init, fh, kind)
    wanted = idx_ranges(http_get(u + ".idx").decode(), spec, fh)
    merged = _merge(wanted)
    with ThreadPoolExecutor(8) as ex:  # S3 is happy with many concurrent range requests
        blobs = list(ex.map(lambda m: http_get(u, (m[0], m[1])), merged))
    out = {}
    for (start, end, parts), blob in zip(merged, blobs):
        futs = {key: pool.submit(_decode, blob[s - start:e - start + 1], not _crop and i == 0)
                for i, (s, e, key) in enumerate(parts)}
        for key, f in futs.items():
            vals, latlon = f.result()
            if latlon is not None and not _crop:
                _setup_crop(*latlon)
            out[key] = vals
    return out


def run_dir(init):
    d = os.path.join(CACHE_DIR, "hrrr", f"{init:%Y%m%d%H}")
    os.makedirs(d, exist_ok=True)
    return d


def save_atomic(p, **arrays):
    """Write next to the target and rename, so a half-written file never looks cached."""
    tmp = p[:-4] + ".part.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


def do_sfc(init, fh, pool):
    p = os.path.join(run_dir(init), f"sfc_f{fh:02d}.npz")
    if os.path.exists(p):
        return "cached"
    f = grab(init, fh, "sfc", SFC, pool)
    save_atomic(p, **{k: v[_crop["sl"]] for k, v in f.items()},
                        **{f"pt_{n}_{k}": v[ij] for n, ij in _crop["pts"].items() for k, v in f.items()})
    return f"{len(f)} fields"


def do_snd(init, fh, pool):
    p = os.path.join(run_dir(init), f"snd_f{fh:02d}.npz")
    if os.path.exists(p):
        return "cached"
    f = grab(init, fh, "prs", PRS, pool)
    out = {"levels": np.array(PRS_LEVELS, dtype=np.float32)}
    for n, ij in _crop["pts"].items():
        for v in ("tmp", "dpt", "ugrd", "vgrd", "hgt"):
            out[f"{n}_{v}"] = np.array([f[f"{v}{lv}"][ij] if f"{v}{lv}" in f else np.nan for lv in PRS_LEVELS],
                                       dtype=np.float32)
    save_atomic(p, **out)
    return f"{len(f)} msgs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(10, 21)))
    ap.add_argument("--valid", default="19-05", help="valid-hour window HH-HH (crosses midnight)")
    ap.add_argument("--snd", type=int, nargs="+", default=[21, 0, 3], help="valid hours for soundings")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    day = datetime.fromisoformat(a.date)
    conv_hour = lambda h: h if h >= 12 else h + 24  # convective day: 00-11Z belong to the next date
    v0, v1 = (conv_hour(int(x)) for x in a.valid.split("-"))
    vts = [day + timedelta(hours=h) for h in range(v0, v1 + 1)]
    snd_vts = {day + timedelta(hours=conv_hour(h)) for h in a.snd}
    jobs = []
    for r in a.runs:
        init = day + timedelta(hours=r)
        for vt in vts:
            fh = int((vt - init).total_seconds() // 3600)
            if 1 <= fh <= 18 or (init.hour % 6 == 0 and fh <= 48):
                jobs.append(("sfc", init, fh))
                if vt in snd_vts:
                    jobs.append(("snd", init, fh))
    print(f"{len(jobs)} files to fetch", flush=True)
    with ProcessPoolExecutor(4) as pool:
        # one file first so the crop/point indices exist before the threads start
        k, init, fh = jobs[0]
        print(f"  {init:%d/%HZ} f{fh:02d} {k}: {(do_sfc if k == 'sfc' else do_snd)(init, fh, pool)}", flush=True)
        if not _crop:  # first file was cached: build the crop from a fresh REFC message
            grab(init, fh, "sfc", {"refc": SFC["refc"]}, pool)
        np.savez_compressed(os.path.join(CACHE_DIR, "hrrr", "grid_crop.npz"), lat=_crop["lat"], lon=_crop["lon"],
                            **{f"pt_{n}": np.array(ij) for n, ij in _crop["pts"].items()})

        def one(job):
            k, init, fh = job
            try:
                msg = (do_sfc if k == "sfc" else do_snd)(init, fh, pool)
            except Missing as e:
                msg = f"missing ({e})"
            print(f"  {init:%d/%HZ} f{fh:02d} {k}: {msg}", flush=True)

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, jobs[1:]))
    print("done", flush=True)


if __name__ == "__main__":
    main()

"""Native-resolution storm snapshots from the convection-allowing models, for judging storm mode (display only).

    python cam_snap.py 2026-11-02 [--models nam rrfs] [--box 31 39 -104 -93] [--hours 18 19 ... 6]

The cached CAM storm fields (<model>_ci) are cell maxima on the 0.5 deg grid: enough to say where storms are, not
whether they are discrete cells or a line. This step re-reads REFC (composite reflectivity) and the 1-h max 2-5 km
updraft helicity from each CAM run in POGS_CAM_RUNS (or the lead-3 pool), on the model's own ~3 km grid, cropped to a
box (default southern Plains), at every hour of the convective day's 18-06Z window the run reaches.
Writes output/cam_snap.npz: "<model>|<run>|<valid>|refc" / "|uh25" (float16), plus "<model>|lat" / "|lon" per grid.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import eccodes
import numpy as np

import fetch
from config import CAM_CI_FIELDS, CASES_DIR, cam_pools, fh_available

_crop = {}


def _decode_native(buf, sub, box):
    h = fetch._cam_handle(buf, sub)
    try:
        nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        vals = eccodes.codes_get_values(h).astype(np.float32)
        if eccodes.codes_get(h, "bitmapPresent"):
            vals[vals == eccodes.codes_get(h, "missingValue")] = np.nan
        key = (nx, ny, eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"))
        if key not in _crop:
            lat = eccodes.codes_get_array(h, "latitudes").reshape(ny, nx)
            lon = eccodes.codes_get_array(h, "longitudes").reshape(ny, nx)
            lon = np.where(lon > 180, lon - 360, lon)
            inside = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
            r, c = np.where(inside)
            sl = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1))
            _crop[key] = (sl, lat[sl].astype(np.float32), lon[sl].astype(np.float32))
    finally:
        eccodes.codes_release(h)
    sl, lat, lon = _crop[key]
    return vals.reshape(ny, nx)[sl], lat, lon


def snap(model, init, fh, box):
    url = fetch.RRFS_2D_URL(init, fh) if model == "rrfs" else fetch.CAM_URL[model](init, fh)
    wanted = fetch.cam_ranges(fetch.http_get(url + ".idx").decode(), CAM_CI_FIELDS, fh, {"uh25"})
    out = {}
    for s, e, (k, sub) in wanted:
        out[k], lat, lon = _decode_native(fetch.http_get(url, (s, e)), sub, box)
    return out, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--models", nargs="+", default=["nam", "rrfs", "hrrr"])
    ap.add_argument("--box", type=float, nargs=4, default=[31.0, 39.0, -104.0, -93.0])
    ap.add_argument("--hours", type=int, nargs="+", default=list(range(18, 24)) + list(range(0, 7)))
    a = ap.parse_args()
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    pools = cam_pools(conv, 3)
    valid = [datetime(conv.year, conv.month, conv.day) + timedelta(days=h < 12, hours=h) for h in a.hours]
    jobs = [(m, i, v) for m in a.models for i in pools.get(m, []) for v in valid
            if fh_available(m, i, int((v - i).total_seconds() // 3600))]
    print(f"{len(jobs)} snapshots: " + ", ".join(f"{m} {i:%d/%HZ}" for m in a.models for i in pools.get(m, [])))

    def one(j):
        m, i, v = j
        try:
            return j, snap(m, i, int((v - i).total_seconds() // 3600), a.box)
        except Exception as e:  # a run still uploading, or a missing field: skip that hour
            print(f"  skip {m} {i:%d/%HZ} {v:%d/%HZ}: {str(e)[:120]}")
            return j, None

    res = {}
    with ThreadPoolExecutor(16) as ex:
        for (m, i, v), r in ex.map(one, jobs):
            if r is None:
                continue
            f, lat, lon = r
            res[f"{m}|lat"], res[f"{m}|lon"] = lat, lon
            for k, x in f.items():
                res[f"{m}|{i:%Y%m%d%H}|{v:%Y%m%d%H}|{k}"] = x.astype(np.float16)
    out = os.path.join(CASES_DIR, a.date, "output")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "cam_snap.npz"), **res)
    print(f"wrote {len(res)} arrays to output/cam_snap.npz")


if __name__ == "__main__":
    main()

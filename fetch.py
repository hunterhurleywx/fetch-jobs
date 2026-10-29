"""Byte-range download of GFS / GEFS / ECMWF fields, cropped to the common grid and cached."""
import json
from datetime import datetime
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import eccodes
import numpy as np
import requests

from config import (CACHE_DIR, EC_FIELDS, EC_FIELDS_EPS, EC_SND_FIELDS, EPS_MEMBERS, GEFS_FIELDS_A,
                    GEFS_FIELDS_B, GEFS_MEMBERS, GFS_FIELDS, GFS_SND_FIELDS, GRID_LATS, GRID_LONS,
                    SND_LATS, SND_LONS, GFS_X_FIELDS, GEFS_X_FIELDS_A, GEFS_X_FIELDS_B, EC_X_FIELDS, EC_Z_FIELDS,
                    OPTIONAL_X, CAM_FIELDS, CAM_CI_FIELDS, CAM_OPTIONAL, CAM_MAX_KEYS)

GRIDS = {"main": (GRID_LATS, GRID_LONS), "snd": (SND_LATS, SND_LONS)}

GFS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEFS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
# Google Cloud mirror of ECMWF open data (same paths). The AWS eu-central-1 bucket
# throttles hard (503 SlowDown) under parallel byte-range load.
EC_BUCKET = "https://storage.googleapis.com/ecmwf-open-data"

_local = threading.local()
STATS = {"bytes": 0, "requests": 0, "retries": 0}
_stats_lock = threading.Lock()


class Missing(Exception):
    """The requested file or field does not exist in the archive."""


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def http_get(url, rng=None, max_tries=8):
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    for attempt in range(max_tries):
        try:
            r = _session().get(url, headers=headers, timeout=60)
            if r.status_code in (403, 404):
                raise Missing(url)
            if r.status_code in (200, 206):
                with _stats_lock:
                    STATS["bytes"] += len(r.content)
                    STATS["requests"] += 1
                return r.content
            # 503 SlowDown / 5xx: back off and retry
        except (requests.ConnectionError, requests.Timeout):
            pass
        with _stats_lock:
            STATS["retries"] += 1
        time.sleep(min(30, 1.5 * 2 ** attempt))
    raise RuntimeError(f"gave up after {max_tries} tries: {url}")


# ---------------------------------------------------------------------------
# Decoding and regridding
# ---------------------------------------------------------------------------
_grid_cache = {}


def _grid_index(lat1, lon1, dlat, dlon, nj, ni, south_to_north, grid="main"):
    """Row/col indices into the source grid for every target-grid point."""
    key = (lat1, lon1, dlat, dlon, nj, ni, south_to_north, grid)
    if key not in _grid_cache:
        lats, lons = GRIDS[grid]
        sgn = 1 if south_to_north else -1
        rows = np.rint((lats - lat1) / (sgn * dlat)).astype(int)
        cols = np.rint(((lons - lon1) % 360) / dlon).astype(int) % ni
        assert rows.min() >= 0 and rows.max() < nj, "domain outside source grid"
        _grid_cache[key] = (rows, cols)
    return _grid_cache[key]


_decode_lock = threading.Lock()  # eccodes' definition parser is not thread-safe
DECODE_POOL = None  # optional ProcessPoolExecutor: one eccodes instance per process


def init_decode_pool(n):
    global DECODE_POOL
    from concurrent.futures import ProcessPoolExecutor
    DECODE_POOL = ProcessPoolExecutor(n)


def decode_to_grid(buf, grid="main"):
    """Decode one GRIB2 message (regular lat/lon) onto a target grid ('main' 0.5, 'snd' 0.25 deg)."""
    if DECODE_POOL is not None:
        return DECODE_POOL.submit(_decode, buf, grid).result()
    with _decode_lock:
        return _decode(buf, grid)


def _decode(buf, grid="main"):
    h = eccodes.codes_new_from_message(buf)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        s2n = eccodes.codes_get(h, "jScansPositively") == 1
        vals = eccodes.codes_get_values(h).astype(np.float32)
        if eccodes.codes_get(h, "bitmapPresent"):
            vals[vals == eccodes.codes_get(h, "missingValue")] = np.nan
    finally:
        eccodes.codes_release(h)
    rows, cols = _grid_index(lat1, lon1, dlat, dlon, nj, ni, s2n, grid)
    return vals.reshape(nj, ni)[np.ix_(rows, cols)]


def _merge(ranges):
    """Merge byte ranges that touch so contiguous messages come in one request."""
    ranges = sorted(ranges)
    out = []
    for start, end, key in ranges:
        if out and start == out[-1][1] + 1:
            out[-1][1] = end
            out[-1][2].append((start, end, key))
        else:
            out.append([start, end, [(start, end, key)]])
    return out


def fetch_ranges(url, wanted, grid="main"):
    """wanted: list of (start, end, key). Returns {key: grid array}."""
    fields = {}
    for start, end, parts in _merge(wanted):
        blob = http_get(url, (start, end))
        for s, e, key in parts:
            fields[key] = decode_to_grid(blob[s - start:e - start + 1], grid)
    return fields


# ---------------------------------------------------------------------------
# NCEP (GFS / GEFS) text .idx
# ---------------------------------------------------------------------------
def ncep_ranges(idx_text, spec, fh):
    lines = [l.split(":") for l in idx_text.strip().split("\n")]
    offsets = [int(p[1]) for p in lines]
    fcst = "anl" if fh == 0 else f"{fh} hour fcst"
    lookup, acc0, accb = {}, {}, {}
    for i, p in enumerate(lines):
        end = offsets[i + 1] - 1 if i + 1 < len(lines) else None
        if p[5] == fcst:
            lookup.setdefault((p[3], p[4]), (offsets[i], end))
        elif p[5].endswith("acc fcst") and "-" in p[5]:
            start = p[5].split("-")[0]
            if start == "0":
                acc0.setdefault((p[3], p[4]), (offsets[i], end))
            if p[5].endswith(f"-{fh} hour acc fcst") and start.isdigit():
                prev = accb.get((p[3], p[4]))
                if prev is None or int(start) > prev[0]:  # shortest bucket ending at fh
                    accb[(p[3], p[4])] = (int(start), offsets[i], end)
    wanted = []
    for key, sv in spec.items():
        var, lev, kind = (sv + ("inst",))[:3]
        if kind == "acc0":
            hit = acc0.get((var, lev))
        elif kind == "accb":
            hit = accb.get((var, lev))
            hit = hit[1:] if hit else None
        else:
            hit = lookup.get((var, lev))
        if hit is None or hit[1] is None:
            if key in OPTIONAL_X:
                continue
            raise Missing(f"{var} {lev} f{fh}")
        wanted.append((*hit, key))
    return wanted


def gfs_url(init, fh):
    return (f"{GFS_BUCKET}/gfs.{init:%Y%m%d}/{init:%H}/atmos/"
            f"gfs.t{init:%H}z.pgrb2.0p25.f{fh:03d}")


def gefs_url(init, fh, mem, ab):
    return (f"{GEFS_BUCKET}/gefs.{init:%Y%m%d}/{init:%H}/atmos/pgrb2{ab}p5/"
            f"ge{mem}.t{init:%H}z.pgrb2{ab}.0p50.f{fh:03d}")


def get_gfs(init, fh):
    url = gfs_url(init, fh)
    spec = dict(GFS_FIELDS)
    idx = http_get(url + ".idx").decode()
    if ":CAPE:90-0 mb above ground:" not in idx:  # pre-v16 (before 22 Mar 2021)
        spec.pop("cape90"), spec.pop("cin90")
    return {"det": fetch_ranges(url, ncep_ranges(idx, spec, fh))}


def get_gfs_snd(init, fh):
    """Full GFS column (21 levels + surface) on the 0.25 deg sounding grid."""
    url = gfs_url(init, fh)
    idx = http_get(url + ".idx").decode()
    return {"det": fetch_ranges(url, ncep_ranges(idx, GFS_SND_FIELDS, fh), grid="snd")}


def get_gfs_x(init, fh):
    url = gfs_url(init, fh)
    idx = http_get(url + ".idx").decode()
    return {"det": fetch_ranges(url, ncep_ranges(idx, GFS_X_FIELDS, fh))}


def get_gefs_member(init, fh, mem, specs=None):
    out = {}
    for ab, spec in specs or (("a", GEFS_FIELDS_A), ("b", GEFS_FIELDS_B)):
        url = gefs_url(init, fh, mem, ab)
        idx = http_get(url + ".idx").decode()
        out.update(fetch_ranges(url, ncep_ranges(idx, spec, fh)))
    return out


# ---------------------------------------------------------------------------
# ECMWF open data (JSON-lines .index)
# ---------------------------------------------------------------------------
# Open-data parameters that only exist from some date, and what to use before then.
# "mucape" arrived with Cy49r1 (~12 Nov 2024); earlier files carry "cape" (param 59), which
# ECMWF computes from the most unstable parcel departing below 350 hPa, so it's MU-type too.
EC_ALIASES = {"mucape": "cape"}


def ec_base(init, fh, model):
    if model in ("hres", "hres_snd", "hres_x"):
        # Cy50r1 (12 May 2026 06Z) moved the 06/18Z runs from "scda" into "oper"
        post50r1 = init >= datetime(2026, 5, 12, 6)
        stream, typ = ("oper", "fc") if (init.hour in (0, 12) or post50r1) else ("scda", "fc")
    else:
        stream, typ = "enfo", "ef"
    return (f"{EC_BUCKET}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/{stream}/"
            f"{init:%Y%m%d%H}0000-{fh}h-{stream}-{typ}")


def get_ecmwf(init, fh, model, pool):
    """Returns {member: fields}. HRES -> {'det': ...}; EPS -> {'1'..'50': ...}."""
    base = ec_base(init, fh, model)
    rows = [json.loads(l) for l in http_get(base + ".index").decode().splitlines() if l.strip()]
    spec = {"eps": EC_FIELDS_EPS, "hres_snd": EC_SND_FIELDS, "hres_x": EC_X_FIELDS,
            "eps_x": EC_X_FIELDS, "eps_z": EC_Z_FIELDS}.get(model, EC_FIELDS)
    have = {r["param"] for r in rows}
    spec = {k: ((EC_ALIASES[p] if p not in have and p in EC_ALIASES else p), lt, lv)
            for k, (p, lt, lv) in spec.items()}
    grid = "snd" if model.endswith("_snd") else "main"
    ens = model.startswith("eps")
    members = EPS_MEMBERS if ens else ["det"]
    want = {}
    for r in rows:
        if ens and r.get("type") != "pf":
            continue
        mem = r.get("number", "det") if ens else "det"
        if mem not in members:
            continue
        for key, (param, levtype, lev) in spec.items():
            if r["param"] == param and r["levtype"] == levtype and r.get("levelist") == lev:
                o = r["_offset"]
                want.setdefault(mem, []).append((o, o + r["_length"] - 1, key))
    need = len([k for k in spec if k not in OPTIONAL_X])
    for m in members:
        if len(want.get(m, [])) < need:
            raise Missing(f"{model} {init} f{fh} member {m} incomplete")
    futs = {m: pool.submit(fetch_ranges, base + ".grib2", want[m], grid) for m in members}
    return {m: f.result() for m, f in futs.items()}


# ---------------------------------------------------------------------------
# Convection-allowing / rapid-refresh models on Lambert grids (HRRR, NAM 3 km nest, RAP)
# ---------------------------------------------------------------------------
CAM_URL = {
    "hrrr": lambda i, fh: f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{i:%Y%m%d}/conus/hrrr.t{i:%H}z.wrfprsf{fh:02d}.grib2",
    "nam": lambda i, fh: f"https://noaa-nam-pds.s3.amazonaws.com/nam.{i:%Y%m%d}/nam.t{i:%H}z.conusnest.hiresf{fh:02d}.tm00.grib2",
    "rap": lambda i, fh: f"https://noaa-rap-pds.s3.amazonaws.com/rap.{i:%Y%m%d}/rap.t{i:%H}z.awp130pgrbf{fh:02d}.grib2",
}
_lam_map = {}  # per process: grid key -> flat target-cell index of every source point (-1 = outside)


def _cam_handle(buf, sub):
    """Handle on field `sub` (1-based) of a GRIB message; NAM packs u/v pairs into one message."""
    if sub == 1:
        return eccodes.codes_new_from_message(buf)
    eccodes.codes_grib_multi_support_on()
    fd, name = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf)
        with open(name, "rb") as f:
            h = None
            for _ in range(sub):
                if h is not None:
                    eccodes.codes_release(h)
                h = eccodes.codes_grib_new_from_file(f)
            eccodes.codes_grib_multi_support_reset_file(f)
        return h
    finally:
        try:
            os.remove(name)
        except OSError:  # Windows may still hold it; it is in the temp dir
            pass


def _decode_cam(buf, use_max, sub=1):
    """Lambert field -> 0.5 deg grid: mean of the points in each cell, or max (REFC / UH)."""
    h = _cam_handle(buf, sub)
    try:
        nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        key = (nx, ny, eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
               eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"))
        vals = eccodes.codes_get_values(h).astype(np.float64)
        if eccodes.codes_get(h, "bitmapPresent"):
            vals[vals == eccodes.codes_get(h, "missingValue")] = np.nan
        if key not in _lam_map:
            lat = eccodes.codes_get_array(h, "latitudes")
            lon = eccodes.codes_get_array(h, "longitudes")
            lon = np.where(lon > 180, lon - 360, lon)
            ti = np.rint((lat - GRID_LATS[0]) / 0.5).astype(int)
            tj = np.rint((lon - GRID_LONS[0]) / 0.5).astype(int)
            ok = (ti >= 0) & (ti < len(GRID_LATS)) & (tj >= 0) & (tj < len(GRID_LONS))
            _lam_map[key] = np.where(ok, ti * len(GRID_LONS) + tj, -1)
    finally:
        eccodes.codes_release(h)
    t = _lam_map[key]
    ncell = len(GRID_LATS) * len(GRID_LONS)
    ok = (t >= 0) & np.isfinite(vals)
    if use_max:
        out = np.full(ncell, -np.inf)
        np.maximum.at(out, t[ok], vals[ok])
        out[np.isinf(out)] = np.nan
    else:
        n = np.bincount(t[ok], minlength=ncell)
        out = np.bincount(t[ok], vals[ok], minlength=ncell) / np.where(n > 0, n, np.nan)
    return out.reshape(len(GRID_LATS), len(GRID_LONS)).astype(np.float32)


def cam_ranges(idx_text, spec, fh, optional=()):
    lines = [l.split(":") for l in idx_text.strip().split("\n")]
    offs = [int(p[1]) for p in lines]
    want = {"inst": f"{fh} hour fcst", "max": f"{fh - 1}-{fh} hour max fcst"}
    found = {}
    for i, p in enumerate(lines):
        # sub-fields ("303.1", "303.2") share one message: it ends where the next offset differs
        nxt = next((o for o in offs[i + 1:] if o != offs[i]), None)
        end = nxt - 1 if nxt is not None else None
        sub = int(p[0].split(".")[1]) if "." in p[0] else 1
        for key, (var, lev, kind) in spec.items():
            if key not in found and p[3] == var and p[4].startswith(lev) and p[5] == want[kind] and end is not None:
                found[key] = (offs[i], end, (key, sub))
    missing = [k for k in spec if k not in found and k not in optional]
    if missing:
        raise Missing(f"fields {missing} f{fh}")
    # one request per distinct byte range (u and v of a packed pair share it)
    return sorted(found.values(), key=lambda r: (r[0], r[2][1]))


def get_cam(model, init, fh):
    """model: hrrr | nam | rap (environment fields) or hrrr_ci | nam_ci | rap_ci (hourly REFC / UH)."""
    base, ci = model.replace("_ci", ""), model.endswith("_ci")
    url = CAM_URL[base](init, fh)
    spec = CAM_CI_FIELDS if ci else CAM_FIELDS
    wanted = cam_ranges(http_get(url + ".idx").decode(), spec, fh, CAM_OPTIONAL)
    blobs = {}
    for start, end in sorted({(s, e) for s, e, _ in wanted}):
        blobs[(start, end)] = http_get(url, (start, end))
    fields = {}
    for s, e, (key, sub) in wanted:
        b, use_max = blobs[(s, e)], key in CAM_MAX_KEYS
        fields[key] = (DECODE_POOL.submit(_decode_cam, b, use_max, sub).result() if DECODE_POOL is not None
                       else _decode_cam(b, use_max, sub))
    return {"det": fields}


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
def cache_path(model, init, fh, member):
    return os.path.join(CACHE_DIR, model, f"{init:%Y%m%d%H}", f"f{fh:03d}", f"{member}.npz")


def missing_marker(model, init, fh):
    return os.path.join(CACHE_DIR, model, f"{init:%Y%m%d%H}", f"f{fh:03d}", "_MISSING")


def is_cached(model, init, fh):
    members = {"gefs": GEFS_MEMBERS, "eps": EPS_MEMBERS, "gefs_x": GEFS_MEMBERS,
               "eps_x": EPS_MEMBERS, "eps_z": EPS_MEMBERS}.get(model, ["det"])
    if os.path.exists(missing_marker(model, init, fh)):
        return True
    return all(os.path.exists(cache_path(model, init, fh, m)) for m in members)


def _save(model, init, fh, member_fields):
    for m, fields in member_fields.items():
        p = cache_path(model, init, fh, m)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p + ".tmp.npz", **fields)
        os.replace(p + ".tmp.npz", p)


def _mark_missing(model, init, fh, why):
    p = missing_marker(model, init, fh)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(str(why))


def download_all(tasks, ncep_workers=24, ec_workers=8, log=print):
    """tasks: list of (model, init, fh). Downloads everything not already cached."""
    todo = [t for t in tasks if not is_cached(*t)]
    log(f"{len(tasks)} model-steps requested, {len(tasks) - len(todo)} cached, {len(todo)} to fetch")
    t0 = time.time()
    ncep_pool = ThreadPoolExecutor(ncep_workers)
    ec_pool = ThreadPoolExecutor(ec_workers)
    ec_member_pool = ThreadPoolExecutor(ec_workers)
    futs = {}
    for model, init, fh in todo:
        if model == "gfs":
            futs[ncep_pool.submit(get_gfs, init, fh)] = (model, init, fh, None)
        elif model == "gfs_snd":
            futs[ncep_pool.submit(get_gfs_snd, init, fh)] = (model, init, fh, None)
        elif model == "gfs_x":
            futs[ncep_pool.submit(get_gfs_x, init, fh)] = (model, init, fh, None)
        elif model == "gefs_x":
            for m in GEFS_MEMBERS:
                if not os.path.exists(cache_path(model, init, fh, m)):
                    futs[ncep_pool.submit(get_gefs_member, init, fh, m,
                                          (("a", GEFS_X_FIELDS_A), ("b", GEFS_X_FIELDS_B)))] = (model, init, fh, m)
        elif model == "gefs":
            for m in GEFS_MEMBERS:
                if not os.path.exists(cache_path(model, init, fh, m)):
                    futs[ncep_pool.submit(get_gefs_member, init, fh, m)] = (model, init, fh, m)
        elif model.replace("_ci", "") in CAM_URL:
            futs[ncep_pool.submit(get_cam, model, init, fh)] = (model, init, fh, None)
        else:
            futs[ec_pool.submit(get_ecmwf, init, fh, model, ec_member_pool)] = (model, init, fh, None)
    done = 0
    for fut in as_completed(futs):
        model, init, fh, mem = futs[fut]
        done += 1
        try:
            res = fut.result()
            _save(model, init, fh, {mem: res} if mem else res)
        except Missing as e:
            _mark_missing(model, init, fh, e)
            log(f"  missing: {model} {init:%Y%m%d%H} f{fh:03d} ({e})")
        except Exception as e:  # keep going; a rerun retries anything not cached
            log(f"  FAILED: {model} {init:%Y%m%d%H} f{fh:03d} {mem or ''}: {e}")
        if done % 50 == 0 or done == len(futs):
            el = time.time() - t0
            log(f"  {done}/{len(futs)} jobs  {STATS['bytes'] / 1e9:.2f} GB  "
                f"{STATS['bytes'] / 1e6 / max(el, 1):.1f} MB/s  retries={STATS['retries']}  {el / 60:.1f} min")
    for p in (ncep_pool, ec_pool, ec_member_pool):
        p.shutdown()
    return time.time() - t0

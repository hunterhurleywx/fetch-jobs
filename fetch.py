"""Byte-range download of GFS / GEFS / ECMWF fields, cropped to the common grid and cached."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import eccodes
import numpy as np
import requests

from config import (CACHE_DIR, EC_FIELDS, EC_FIELDS_EPS, EPS_MEMBERS, GEFS_FIELDS_A,
                    GEFS_FIELDS_B, GEFS_MEMBERS, GFS_FIELDS, GRID_LATS, GRID_LONS)

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


def _grid_index(lat1, lon1, dlat, dlon, nj, ni, south_to_north):
    """Row/col indices into the source grid for every common-grid point."""
    key = (lat1, lon1, dlat, dlon, nj, ni, south_to_north)
    if key not in _grid_cache:
        sgn = 1 if south_to_north else -1
        rows = np.rint((GRID_LATS - lat1) / (sgn * dlat)).astype(int)
        cols = np.rint(((GRID_LONS - lon1) % 360) / dlon).astype(int) % ni
        assert rows.min() >= 0 and rows.max() < nj, "domain outside source grid"
        _grid_cache[key] = (rows, cols)
    return _grid_cache[key]


_decode_lock = threading.Lock()  # eccodes' definition parser is not thread-safe
DECODE_POOL = None  # optional ProcessPoolExecutor: one eccodes instance per process


def init_decode_pool(n):
    global DECODE_POOL
    from concurrent.futures import ProcessPoolExecutor
    DECODE_POOL = ProcessPoolExecutor(n)


def decode_to_grid(buf):
    """Decode one GRIB2 message (regular lat/lon) onto the common 0.5 deg grid."""
    if DECODE_POOL is not None:
        return DECODE_POOL.submit(_decode, buf).result()
    with _decode_lock:
        return _decode(buf)


def _decode(buf):
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
    rows, cols = _grid_index(lat1, lon1, dlat, dlon, nj, ni, s2n)
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


def fetch_ranges(url, wanted):
    """wanted: list of (start, end, key). Returns {key: grid array}."""
    fields = {}
    for start, end, parts in _merge(wanted):
        blob = http_get(url, (start, end))
        for s, e, key in parts:
            fields[key] = decode_to_grid(blob[s - start:e - start + 1])
    return fields


# ---------------------------------------------------------------------------
# NCEP (GFS / GEFS) text .idx
# ---------------------------------------------------------------------------
def ncep_ranges(idx_text, spec, fh):
    lines = [l.split(":") for l in idx_text.strip().split("\n")]
    offsets = [int(p[1]) for p in lines]
    fcst = "anl" if fh == 0 else f"{fh} hour fcst"
    lookup = {}
    for i, p in enumerate(lines):
        if p[5] == fcst:
            end = offsets[i + 1] - 1 if i + 1 < len(lines) else None
            lookup.setdefault((p[3], p[4]), (offsets[i], end))
    wanted = []
    for key, varlev in spec.items():
        if varlev not in lookup or lookup[varlev][1] is None:
            raise Missing(f"{varlev} f{fh}")
        wanted.append((*lookup[varlev], key))
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


def get_gefs_member(init, fh, mem):
    out = {}
    for ab, spec in (("a", GEFS_FIELDS_A), ("b", GEFS_FIELDS_B)):
        url = gefs_url(init, fh, mem, ab)
        idx = http_get(url + ".idx").decode()
        out.update(fetch_ranges(url, ncep_ranges(idx, spec, fh)))
    return out


# ---------------------------------------------------------------------------
# ECMWF open data (JSON-lines .index)
# ---------------------------------------------------------------------------
def ec_base(init, fh, model):
    if model == "hres":
        stream, typ = ("oper", "fc") if init.hour in (0, 12) else ("scda", "fc")
    else:
        stream, typ = "enfo", "ef"
    return (f"{EC_BUCKET}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/{stream}/"
            f"{init:%Y%m%d%H}0000-{fh}h-{stream}-{typ}")


def get_ecmwf(init, fh, model, pool):
    """Returns {member: fields}. HRES -> {'det': ...}; EPS -> {'1'..'50': ...}."""
    base = ec_base(init, fh, model)
    rows = [json.loads(l) for l in http_get(base + ".index").decode().splitlines() if l.strip()]
    spec = EC_FIELDS_EPS if model == "eps" else EC_FIELDS
    members = EPS_MEMBERS if model == "eps" else ["det"]
    want = {}
    for r in rows:
        if model == "eps" and r.get("type") != "pf":
            continue
        mem = r.get("number", "det") if model == "eps" else "det"
        if mem not in members:
            continue
        for key, (param, levtype, lev) in spec.items():
            if r["param"] == param and r["levtype"] == levtype and r.get("levelist") == lev:
                o = r["_offset"]
                want.setdefault(mem, []).append((o, o + r["_length"] - 1, key))
    for m in members:
        if len(want.get(m, [])) != len(spec):
            raise Missing(f"{model} {init} f{fh} member {m} incomplete")
    futs = {m: pool.submit(fetch_ranges, base + ".grib2", want[m]) for m in members}
    return {m: f.result() for m, f in futs.items()}


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
def cache_path(model, init, fh, member):
    return os.path.join(CACHE_DIR, model, f"{init:%Y%m%d%H}", f"f{fh:03d}", f"{member}.npz")


def missing_marker(model, init, fh):
    return os.path.join(CACHE_DIR, model, f"{init:%Y%m%d%H}", f"f{fh:03d}", "_MISSING")


def is_cached(model, init, fh):
    members = {"gfs": ["det"], "hres": ["det"], "gefs": GEFS_MEMBERS, "eps": EPS_MEMBERS}[model]
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
        elif model == "gefs":
            for m in GEFS_MEMBERS:
                if not os.path.exists(cache_path(model, init, fh, m)):
                    futs[ncep_pool.submit(get_gefs_member, init, fh, m)] = (model, init, fh, m)
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

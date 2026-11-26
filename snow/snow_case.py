"""Snowfall-total test run (experimental, cloud only): every model's runs before an event -> 24-h snowfall on a
0.1 deg box, with several snow-to-liquid ratio (SLR) methods, plus NBM / NOHRSC for comparison.

  python snow/snow_case.py fetch  --out DIR --start 2026112518 --end 2026112618 --last-ncep 2026112512 --last-ec 2026112506
  python snow/snow_case.py finish --out DIR          (after snow/slr_rf.py has written slr_rf.npz)

Per model / run / member, for each 3-h interval inside the window:
  QPF (mm), snow fraction (CSNOW, or ECMWF sf/tp), snow water = QPF x fraction,
  SLR: 10:1, Kuchera (warmest T below 500 hPa), MaxTAloft (NBM polynomial, warmest T 610 m AGL..400 hPa),
       850-700 hPa thickness (NBM), Pletcher 2026 RF / LR (features written here, predicted by slr_rf.py),
  native snowfall: HRRR / RAP / RRFS / REFS ASNOW (model's own variable-density snow).
Snowfall (in) = sum over intervals of snow water x SLR / 25.4. Outputs (small): snow_members.npz, snow_meta.json.
"""
import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import requests

# ---------------------------------------------------------------------------------------------- target grid (0.1 deg)
LAT0, LAT1, LON0, LON1, D = 35.0, 42.5, -96.0, -84.0, 0.1
LATS = np.round(np.arange(LAT0, LAT1 + 1e-6, D), 2)
LONS = np.round(np.arange(LON0, LON1 + 1e-6, D), 2)
NY, NX = len(LATS), len(LONS)
LAT2, LON2 = np.meshgrid(LATS, LONS, indexing="ij")
AGL = [300, 600, 900, 1200, 1500, 1800, 2100, 2400]          # Pletcher 2026 feature levels
PRS_FULL = [1000, 950, 925, 900, 850, 800, 750, 700]        # T, Z, RH, U, V
PRS_T = [600, 500]                                          # T only (Kuchera / MaxTAloft)

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
EC = "https://storage.googleapis.com/ecmwf-open-data"
EC_DIRECT = "https://data.ecmwf.int/forecasts"
HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
RAP = "https://noaa-rap-pds.s3.amazonaws.com"
RRFS = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
NBM = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
NOHRSC = "https://www.nohrsc.noaa.gov/snowfall_v2/data"


class Missing(Exception):
    pass


_local = threading.local()


def get(url, rng=None, tries=8):
    s = getattr(_local, "s", None) or requests.Session()
    _local.s = s
    h = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    for a in range(tries):
        try:
            r = s.get(url, headers=h, timeout=90)
            if r.status_code in (403, 404):
                raise Missing(url)
            if r.status_code in (200, 206):
                return r.content
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(min(30, 1.5 * 2 ** a))
    raise RuntimeError("gave up: " + url)


# ---------------------------------------------------------------------------------------------- decoding / regridding
_maps = {}


def _decode(buf, sub=1):
    """Any GRIB2 message -> (NY, NX) float32 on the 0.1 deg box. Regular lat/lon: bilinear. Lambert: mean of source
    points in each cell, nearest point where a cell has none."""
    import eccodes
    if sub > 1:  # packed sub-field (not expected for these models, kept for safety)
        eccodes.codes_grib_multi_support_on()
    h = eccodes.codes_new_from_message(buf)
    try:
        gt = eccodes.codes_get(h, "gridType")
        vals = eccodes.codes_get_values(h).astype(np.float64)
        if eccodes.codes_get(h, "bitmapPresent"):
            vals[vals == eccodes.codes_get(h, "missingValue")] = np.nan
        if gt == "regular_ll":
            ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            la1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            lo1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dla = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
            dlo = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
            s2n = eccodes.codes_get(h, "jScansPositively") == 1
            f = vals.reshape(nj, ni)
            fi = (LAT2 - la1) / (dla if s2n else -dla)
            fj = ((LON2 - lo1) % 360) / dlo
            i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
            wi, wj = fi - i0, fj - j0
            i1, j1 = np.clip(i0 + 1, 0, nj - 1), (j0 + 1) % ni
            out = (f[i0, j0] * (1 - wi) * (1 - wj) + f[i1, j0] * wi * (1 - wj) + f[i0, j1] * (1 - wi) * wj
                   + f[i1, j1] * wi * wj)
            return out.astype(np.float32)
        key = (gt, eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny"),
               round(eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), 4))
        if key not in _maps:
            lat = eccodes.codes_get_array(h, "latitudes")
            lon = eccodes.codes_get_array(h, "longitudes")
            lon = np.where(lon > 180, lon - 360, lon)
            ti = np.rint((lat - LAT0) / D).astype(int)
            tj = np.rint((lon - LON0) / D).astype(int)
            ok = (ti >= 0) & (ti < NY) & (tj >= 0) & (tj < NX)
            cell = np.where(ok, ti * NX + tj, -1)
            near = (lat > LAT0 - 1) & (lat < LAT1 + 1) & (lon > LON0 - 1) & (lon < LON1 + 1)
            from scipy.spatial import cKDTree
            idx_near = np.where(near)[0]
            tree = cKDTree(np.c_[lat[idx_near], lon[idx_near] * np.cos(np.deg2rad(38.5))])
            _, nn = tree.query(np.c_[LAT2.ravel(), LON2.ravel() * np.cos(np.deg2rad(38.5))])
            _maps[key] = (cell, idx_near[nn])
    finally:
        eccodes.codes_release(h)
    cell, nn = _maps[key]
    ok = (cell >= 0) & np.isfinite(vals)
    n = np.bincount(cell[ok], minlength=NY * NX)
    s = np.bincount(cell[ok], vals[ok], minlength=NY * NX)
    out = np.where(n > 0, s / np.maximum(n, 1), vals[nn])
    return out.reshape(NY, NX).astype(np.float32)


# ---------------------------------------------------------------------------------------------- index parsing
def ncep_lines(idx_text):
    rows = [l.split(":") for l in idx_text.strip().split("\n") if l.strip()]
    offs = [int(r[1]) for r in rows]
    out = []
    for i, r in enumerate(rows):
        nxt = next((o for o in offs[i + 1:] if o != offs[i]), None)
        out.append((r[3], r[4], r[5], r[6] if len(r) > 6 else "", offs[i], (nxt - 1) if nxt else offs[i] + 30_000_000))
    return out


def pick_ncep(lines, fh, want_prs, want_t, ens_tag=None):
    """Byte ranges for one NCEP file at hour fh: accumulations (APCP/ASNOW, any a-b), CSNOW (inst or ave), levels."""
    inst = f"{fh} hour fcst"
    sel = {}
    for var, lev, desc, extra, s, e in lines:
        if ens_tag is not None and extra and not extra.startswith(ens_tag):
            continue
        m = re.match(r"(\d+)-(\d+) hour acc fcst$", desc)
        if var in ("APCP", "ASNOW") and lev == "surface" and m and int(m.group(2)) == fh:
            sel.setdefault(f"acc|{var}|{m.group(1)}|{fh}", (s, e))
        elif var == "CSNOW" and lev == "surface":
            if desc == inst:
                sel.setdefault("csnow_inst", (s, e))
            elif re.match(rf"\d+-{fh} hour ave fcst$", desc):
                a = int(desc.split("-")[0])
                if "csnow_ave" not in sel or a > sel["csnow_ave_a"][0]:
                    sel["csnow_ave"] = (s, e)
                    sel["csnow_ave_a"] = (a, a)
        elif desc == inst and lev.endswith(" mb"):
            p = int(lev.split()[0])
            if (var in ("TMP", "HGT", "RH", "UGRD", "VGRD") and p in want_prs) or (var == "TMP" and p in want_t):
                sel.setdefault(f"{var}{p}", (s, e))
        elif desc == inst and var == "HGT" and lev == "surface":
            sel.setdefault("orog", (s, e))
    sel.pop("csnow_ave_a", None)
    return sel


def fetch_ncep(url, fh, want_prs=PRS_FULL, want_t=PRS_T, ens_tag=None, extra_urls=()):
    fields = {}
    for u in (url, *extra_urls):
        try:
            lines = ncep_lines(get(u + ".idx").decode())
        except Missing:
            if u == url:
                raise
            continue
        sel = pick_ncep(lines, fh, want_prs, want_t, ens_tag)
        for k, (s, e) in sel.items():
            if k in fields:
                continue
            fields[k] = _decode(get(u, (s, e)))
    return fields


def fetch_ec(base, fh, members, params_sfc=("tp", "sf"), prs=(1000, 925, 850, 700), tonly=(600, 500)):
    """ECMWF open data (HRES / ENS / AIFS): {member: fields}. tp/sf accumulated from 0 h (m)."""
    try:
        idx = get(base + ".index")
    except Missing:
        base = base.replace(EC, EC_DIRECT)
        idx = get(base + ".index")
    rows = [json.loads(l) for l in idx.decode().splitlines() if l.strip()]
    want = {}
    for r in rows:
        mem = str(r.get("number")) if r.get("type") == "pf" else ("det" if r.get("type") == "fc" else None)
        if mem is None or mem not in members:
            continue
        p, lt, lv = r["param"], r["levtype"], r.get("levelist")
        key = None
        if lt == "sfc" and p in params_sfc:
            key = p
        elif lt == "pl" and lv and int(lv) in prs and p in ("t", "gh", "r", "u", "v", "q"):
            key = f"{p}{lv}"
        elif lt == "pl" and lv and int(lv) in tonly and p == "t":
            key = f"t{lv}"
        if key:
            want.setdefault(mem, {})[key] = (r["_offset"], r["_offset"] + r["_length"] - 1)
    # merge touching byte ranges (ENS members of one field sit side by side) -> far fewer requests
    flat = sorted((s, e, mem, k) for mem, kv in want.items() for k, (s, e) in kv.items())
    groups = []
    for s, e, mem, k in flat:
        if groups and s == groups[-1][1] + 1 and groups[-1][1] - groups[-1][0] < 64_000_000:
            groups[-1][1] = e
            groups[-1][2].append((s, e, mem, k))
        else:
            groups.append([s, e, [(s, e, mem, k)]])
    out = {}
    for gs, ge, parts in groups:
        blob = get(base + ".grib2", (gs, ge))
        for s, e, mem, k in parts:
            out.setdefault(mem, {})[k] = _decode(blob[s - gs:e - gs + 1])
    return out


# ---------------------------------------------------------------------------------------------- task worker
def task(model, init_s, fh, mem):
    """One file-hour (one member). Returns (model, init, mem, fh, fields)."""
    init = datetime.strptime(init_s, "%Y%m%d%H")
    d, H = f"{init:%Y%m%d}", f"{init:%H}"
    try:
        if model == "gfs":
            f = fetch_ncep(f"{GFS}/gfs.{d}/{H}/atmos/gfs.t{H}z.pgrb2.0p25.f{fh:03d}", fh)
            return [(model, init_s, "det", fh, f)]
        if model == "gefs":
            a = f"{GEFS}/gefs.{d}/{H}/atmos/pgrb2ap5/ge{mem}.t{H}z.pgrb2a.0p50.f{fh:03d}"
            f = fetch_ncep(a, fh, want_prs=(1000, 925, 850, 700), want_t=(500,))
            return [(model, init_s, mem, fh, f)]
        if model == "hrrr":
            sfc = f"{HRRR}/hrrr.{d}/conus/hrrr.t{H}z.wrfsfcf{fh:02d}.grib2"
            prs = f"{HRRR}/hrrr.{d}/conus/hrrr.t{H}z.wrfprsf{fh:02d}.grib2"
            f = fetch_ncep(sfc, fh, want_prs=(), want_t=())
            f.update({k: v for k, v in fetch_ncep(prs, fh).items() if not k.startswith(("acc", "csnow"))})
            return [(model, init_s, "det", fh, f)]
        if model == "rap":
            f = fetch_ncep(f"{RAP}/rap.{d}/rap.t{H}z.awp130pgrbf{fh:02d}.grib2", fh)
            return [(model, init_s, "det", fh, f)]
        if model == "rrfs":
            two = f"{RRFS}/rrfs.{d}/{H}/rrfs.t{H}z.2dfld.3km.f{fh:03d}.conus.grib2"
            prs = f"{RRFS}/rrfs.{d}/{H}/rrfs.t{H}z.prslev.3km.f{fh:03d}.conus.grib2"
            f = fetch_ncep(two, fh, want_prs=(), want_t=())
            f.update({k: v for k, v in fetch_ncep(prs, fh).items() if not k.startswith(("acc", "csnow"))})
            return [(model, init_s, "det", fh, f)]
        if model == "refs":
            two = f"{RRFS}/rrfsens.{d}/{H}/{mem}/rrfs.t{H}z.{mem}.2dfldnomads.3km.f{fh:03d}.conus.grib2"
            prs = f"{RRFS}/rrfsens.{d}/{H}/{mem}/rrfs.t{H}z.{mem}.prslevnomads.3km.f{fh:03d}.conus.grib2"
            f = fetch_ncep(two, fh, want_prs=(), want_t=())
            f.update({k: v for k, v in fetch_ncep(prs, fh).items() if not k.startswith(("acc", "csnow"))})
            return [(model, init_s, mem, fh, f)]
        if model in ("hres", "eps", "aifs"):
            if model == "aifs":
                base = f"{EC}/{d}/{H}z/aifs-single/0p25/oper/{init_s}0000-{fh}h-oper-fc"
                mems = ["det"]
            elif model == "hres":
                base = f"{EC}/{d}/{H}z/ifs/0p25/oper/{init_s}0000-{fh}h-oper-fc"
                mems = ["det"]
            else:
                base = f"{EC}/{d}/{H}z/ifs/0p25/enfo/{init_s}0000-{fh}h-enfo-ef"
                mems = [str(i) for i in range(1, 51)]
            res = fetch_ec(base, fh, set(mems))
            return [(model, init_s, m, fh, v) for m, v in res.items()]
    except Missing as e:
        return [(model, init_s, mem or "det", fh, {"_missing": str(e)})]
    raise ValueError(model)


# ---------------------------------------------------------------------------------------------- SLR methods
def rh_from_q(q, t, p_hpa):
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return np.clip(100 * e / es, 0, 100)


def profile(f, prs):
    """Stack levels that exist: returns (p list, T, Z, RH, SPD) arrays (L, NY, NX); RH from q if needed (AIFS)."""
    keys = {"T": ("TMP", "t"), "Z": ("HGT", "gh"), "R": ("RH", "r"), "U": ("UGRD", "u"), "V": ("VGRD", "v")}
    ps, T, Z, R, S = [], [], [], [], []
    for p in sorted(prs, reverse=True):
        g = lambda k: f.get(f"{keys[k][0]}{p}", f.get(f"{keys[k][1]}{p}"))
        t, z, u, v = g("T"), g("Z"), g("U"), g("V")
        r = g("R")
        if r is None and f.get(f"q{p}") is not None and t is not None:
            r = rh_from_q(f[f"q{p}"], t, p)
        if any(x is None for x in (t, z, r, u, v)):
            continue
        ps.append(p); T.append(t); Z.append(z); R.append(r); S.append(np.hypot(u, v))
    if len(ps) < 3:
        return None
    return ps, np.array(T), np.array(Z), np.array(R), np.array(S)


def agl_interp(Z, X, orog, agl):
    """Linear interpolation in height of X (L, NY, NX) to orog + agl; below the lowest level -> lowest level."""
    tgt = orog + agl
    out = X[0].copy()
    for k in range(len(Z) - 1):
        z0, z1 = Z[k], Z[k + 1]
        m = (tgt >= z0) & (tgt < z1)
        w = (tgt - z0) / np.where(z1 - z0 == 0, 1, z1 - z0)
        out = np.where(m, X[k] + w * (X[k + 1] - X[k]), out)
    out = np.where(tgt >= Z[-1], X[-1], out)
    return out


def tmax_layer(f, Z, T, orog, bottom_agl, top_hpa):
    """Warmest T (K) from bottom_agl above ground up to top_hpa, using every T level available."""
    lv = sorted({int(k[1:] if k.startswith("t") else k[3:]) for k in f
                 if (k.startswith("TMP") or (k.startswith("t") and k[1:].isdigit()))})
    best = np.full((NY, NX), -np.inf)
    for p in lv:
        if p < top_hpa:
            continue
        t = f.get(f"TMP{p}", f.get(f"t{p}"))
        z = f.get(f"HGT{p}", f.get(f"gh{p}"))
        if z is None:  # T-only level (600/500 hPa): always above the ground here
            best = np.maximum(best, t)
        else:
            best = np.where(z >= orog + bottom_agl, np.maximum(best, t), best)
    return np.where(np.isinf(best), np.nan, best)


def slr_methods(f, orog):
    """Returns dict of SLR arrays plus Pletcher features (27, NY, NX) or None."""
    out = {"s10": np.full((NY, NX), 10.0, np.float32)}
    prs_all = PRS_FULL
    pr = profile(f, prs_all)
    tk = tmax_layer(f, None, None, orog, 0, 500)
    if np.isfinite(tk).any():
        out["kuch"] = np.where(tk > 271.16, 12 + 2 * (271.16 - tk), 12 + (271.16 - tk)).clip(0, 40)
    tm = tmax_layer(f, None, None, orog, 610, 400) - 273.15
    if np.isfinite(tm).any():
        T = tm
        s = 0.0000045 * T**5 + 0.0004432 * T**4 + 0.0130903 * T**3 + 0.0585968 * T**2 - 1.8150809 * T + 5.9805722
        out["mta"] = np.clip(s, 0, 40)
    z85, z70 = f.get("HGT850", f.get("gh850")), f.get("HGT700", f.get("gh700"))
    if z85 is not None and z70 is not None:
        out["thk"] = np.clip(-0.16559 * (z70 - z85) + 263.35, 0, 40)
    feats = None
    if pr is not None:
        ps, T, Z, R, S = pr
        cols = []
        for X in (S, T, R):  # model_keys order: SPD03K..SPD24K, T03K..T24K, R03K..R24K, elev, lat, lon
            for a in AGL:
                cols.append(agl_interp(Z, X, orog, a))
        cols += [orog, LAT2, LON2]
        feats = np.array(cols, np.float32)
    return out, feats


# ---------------------------------------------------------------------------------------------- run plan
def plan(start, end, last_ncep, last_ec):
    """Runs before the event: synoptic cycles >= 6 h apart (severe-system rule), last 4 per model."""
    def cycles(last, every, n, offset=0):
        c, out = last, []
        while len(out) < n:
            if (c.hour - offset) % every == 0:
                out.append(c)
            c -= timedelta(hours=1)
        return out[::-1]
    span = int((end - start).total_seconds() // 3600)
    p = {
        "gfs": cycles(last_ncep, 6, 4), "gefs": cycles(last_ncep, 6, 4),
        "hrrr": cycles(last_ncep, 6, 4), "rrfs": cycles(last_ncep, 6, 4), "refs": cycles(last_ncep, 6, 4),
        "rap": cycles(last_ncep - timedelta(hours=3), 6, 4, offset=3),   # 51-h RAP runs: 03/09/15/21Z
        "hres": cycles(last_ec, 6, 4), "eps": cycles(last_ec, 6, 4), "aifs": cycles(last_ec, 6, 4),
    }
    mems = {"gefs": ["c00"] + [f"p{i:02d}" for i in range(1, 31)], "refs": [f"m{i:03d}" for i in range(1, 6)]}
    tasks = []
    for m, runs in p.items():
        for i in runs:
            h0 = int((start - i).total_seconds() // 3600)
            for fh in range(h0, h0 + span + 1, 3):
                for mem in mems.get(m, [None]):
                    tasks.append((m, f"{i:%Y%m%d%H}", fh, mem))
    return p, tasks


# ---------------------------------------------------------------------------------------------- assembly
def increments(acc, hours, var):
    """3-h increments of an accumulation from messages acc[(a, b)] (any bucket convention)."""
    inc = []
    for h0, h1 in zip(hours[:-1], hours[1:]):
        v = acc.get((h0, h1))
        if v is None:
            for (a, b), x in acc.items():
                if b == h1 and (a, h0) in acc:
                    v = x - acc[(a, h0)]
                    break
                if b == h1 and a == h0:
                    v = x
        if v is None and (0, h1) in acc and h0 == 0:
            v = acc[(0, h1)]
        inc.append(None if v is None else np.clip(v, 0, None))
    return inc


def fetch_main(a):
    start, end = [datetime.strptime(x, "%Y%m%d%H") for x in (a.start, a.end)]
    runs, tasks = plan(start, end, datetime.strptime(a.last_ncep, "%Y%m%d%H"), datetime.strptime(a.last_ec, "%Y%m%d%H"))
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(tasks)} file-hour tasks; runs:", {m: [f"{r:%d/%HZ}" for r in v] for m, v in runs.items()}, flush=True)
    store = {}
    t0 = time.time()
    nproc = a.procs or max(8, 3 * os.cpu_count())
    done = 0
    with ProcessPoolExecutor(nproc) as ex:
        futs = [ex.submit(task, *t) for t in tasks]
        for fu in as_completed(futs):
            try:
                res = fu.result()
            except Exception as e:  # keep going; the member is dropped later if incomplete
                print("task error:", repr(e)[:300], flush=True)
                continue
            for model, init_s, mem, fh, f in res:
                store.setdefault((model, init_s, mem), {})[fh] = f
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
    print(f"downloads done in {time.time() - t0:.0f} s", flush=True)

    # one orography for every model (GFS surface height) so AGL levels are consistent
    orog = None
    for (m, i, mem), byh in store.items():
        if m == "gfs":
            for f in byh.values():
                if "orog" in f:
                    orog = f["orog"]; break
        if orog is not None:
            break
    orog = np.zeros((NY, NX), np.float32) if orog is None else orog

    span = int((end - start).total_seconds() // 3600)
    members, sw_store, keys, tabs, ids = {}, {}, [], [], []
    meta = {"runs": {m: [f"{r:%Y%m%d%H}" for r in v] for m, v in runs.items()}, "missing": [],
            "grid": [LAT0, LAT1, LON0, LON1, D], "window": [a.start, a.end]}
    for (m, i, mem), byh in sorted(store.items()):
        init = datetime.strptime(i, "%Y%m%d%H")
        h0 = int((start - init).total_seconds() // 3600)
        hours = list(range(h0, h0 + span + 1, 3))
        if any(h not in byh or "_missing" in byh[h] for h in hours):
            meta["missing"].append(f"{m}|{i}|{mem}")
            continue
        if m in ("hres", "eps", "aifs"):
            tp = {(0, h): byh[h]["tp"] * 1000 for h in hours if "tp" in byh[h]}
            sf = {(0, h): byh[h]["sf"] * 1000 for h in hours if "sf" in byh[h]}
            q = increments(tp, hours, "tp")
            s = increments(sf, hours, "sf")
            if any(x is None for x in q + s):
                meta["missing"].append(f"{m}|{i}|{mem}|acc"); continue
            swe = [np.minimum(si, qi) for si, qi in zip(s, q)]
            native = None
        else:
            acc = {(int(k.split("|")[2]), int(k.split("|")[3])): v for h in hours for k, v in byh[h].items()
                   if k.startswith("acc|APCP")}
            q = increments(acc, hours, "APCP")
            if any(x is None for x in q):
                meta["missing"].append(f"{m}|{i}|{mem}|apcp"); continue
            fr = []
            for h0_, h1_ in zip(hours[:-1], hours[1:]):
                f0, f1 = byh[h0_], byh[h1_]
                if "csnow_inst" in f1 and "csnow_inst" in f0:
                    fr.append(0.5 * (f0["csnow_inst"] + f1["csnow_inst"]))
                elif "csnow_ave" in f1:
                    fr.append(f1["csnow_ave"])
                elif "csnow_inst" in f1:
                    fr.append(f1["csnow_inst"])
                else:
                    fr.append(None)
            if any(x is None for x in fr):
                meta["missing"].append(f"{m}|{i}|{mem}|csnow"); continue
            swe = [qi * np.clip(fi, 0, 1) for qi, fi in zip(q, fr)]
            asn = {(int(k.split("|")[2]), int(k.split("|")[3])): v for h in hours for k, v in byh[h].items()
                   if k.startswith("acc|ASNOW")}
            ai = increments(asn, hours, "ASNOW") if asn else None
            native = None if (ai is None or any(x is None for x in ai)) else sum(ai) * 39.3701  # m -> in
        tot = {"qpf": sum(q) / 25.4, "swe": sum(swe) / 25.4}
        acc_m = {}
        fe = []
        for k, h1_ in enumerate(hours[1:]):
            sl, ft = slr_methods(byh[h1_], orog)
            for name, arr in sl.items():
                acc_m[name] = acc_m.get(name, 0) + swe[k] * arr / 25.4
            if ft is not None:
                fe.append(ft)
        tot.update({f"snow_{k}": v for k, v in acc_m.items()})
        if native is not None:
            tot["snow_native"] = native
        if m in ("hres", "eps", "aifs"):
            tot["snow_native"] = sum(swe) * 10 / 25.4  # ECMWF sf at 10:1 (its own snow water, no density given)
        key = f"{m}|{i}|{mem}"
        for k, v in tot.items():
            members[f"{key}|{k}"] = np.asarray(v, np.float32)
        sw_store[key] = np.array(swe, np.float32)  # (K, NY, NX) mm, to apply the RF / LR SLR later
        if len(fe) == len(hours) - 1:  # Pletcher features only where an interval has snow water > 0.01 mm
            kid = len(keys)
            keys.append(key)
            for k in range(len(fe)):
                ii = np.flatnonzero(swe[k].ravel() > 0.01)
                if len(ii):
                    tabs.append(fe[k].reshape(27, -1)[:, ii].T.astype(np.float32))
                    ids.append(np.c_[np.full(len(ii), kid), np.full(len(ii), k), ii].astype(np.int32))
        byh.clear()  # free memory as we go
    np.savez_compressed(os.path.join(a.out, "snow_members.npz"), **members)
    np.savez_compressed(os.path.join(a.out, "snow_water.npz"), **sw_store)
    if tabs:
        np.savez_compressed(os.path.join(a.out, "slr_features.npz"), X=np.concatenate(tabs), ids=np.concatenate(ids),
                            keys=np.array(keys))
    meta["n_members"] = len({k.rsplit("|", 1)[0] for k in members})
    meta["feature_rows"] = int(sum(len(t) for t in tabs)) if tabs else 0
    json.dump(meta, open(os.path.join(a.out, "snow_meta.json"), "w"), indent=1)
    print(f"members {meta['n_members']}, missing {len(meta['missing'])}, feature rows {meta['feature_rows']}", flush=True)
    np.save(os.path.join(a.out, "orog.npy"), orog)


def extras(a):
    """NBM v5 24-h snowfall (deterministic + 50th percentile, from the last run before the window) and NOHRSC."""
    start, end = [datetime.strptime(x, "%Y%m%d%H") for x in (a.start, a.end)]
    out = {}
    nbm_init = datetime.strptime(a.last_ncep, "%Y%m%d%H")
    tot, ok = 0, True
    for h in range(int((start - nbm_init).total_seconds() // 3600) + 6, int((end - nbm_init).total_seconds() // 3600) + 1, 6):
        url = f"{NBM}/blend.{nbm_init:%Y%m%d}/{nbm_init:%H}/core/blend.t{nbm_init:%H}z.core.f{h:03d}.co.grib2"
        try:
            lines = ncep_lines(get(url + ".idx").decode())
        except Missing:
            ok = False; break
        hit = [(s, e) for var, lev, desc, extra, s, e in lines
               if var == "ASNOW" and desc == f"{h - 6}-{h} hour acc fcst" and extra == ""]
        if not hit:
            ok = False; break
        tot = tot + _decode(get(url, hit[0]))
    if ok:
        out["nbm_det"] = tot * 39.3701  # m -> in
    print("NBM:", "ok" if ok else "missing", flush=True)
    # NOHRSC 6-h analyses ending each 6 h through the window (NetCDF, inches? -> metres in v2 files)
    got = []
    for h in range(6, int((end - start).total_seconds() // 3600) + 1, 6):
        t = start + timedelta(hours=h)
        url = f"{NOHRSC}/{t:%Y%m}/sfav2_CONUS_6h_{t:%Y%m%d%H}.nc"
        try:
            got.append((t, get(url)))
        except Missing:
            pass
    print(f"NOHRSC 6-h files: {len(got)}", flush=True)
    for t, blob in got:
        open(os.path.join(a.out, f"nohrsc_6h_{t:%Y%m%d%H}.nc"), "wb").write(blob)
    if out:
        np.savez_compressed(os.path.join(a.out, "snow_benchmarks.npz"), **out)


def finish(a):
    """Apply the RF / LR SLR (slr_rf.npz) to the stored per-interval snow water."""
    p = os.path.join(a.out, "slr_rf.npz")
    if not os.path.exists(p):
        print("no slr_rf.npz: Pletcher methods skipped", flush=True)
        return
    r = np.load(p)
    sw = np.load(os.path.join(a.out, "snow_water.npz"))
    feats = np.load(os.path.join(a.out, "slr_features.npz"))
    keys, ids = list(feats["keys"]), feats["ids"]
    mem = dict(np.load(os.path.join(a.out, "snow_members.npz")))
    for meth in ("rf", "lr"):
        if meth not in r.files:
            continue
        slr = r[meth]
        acc = {}
        for kid, k, cell, s in zip(ids[:, 0], ids[:, 1], ids[:, 2], slr):
            key = keys[kid]
            if key not in acc:
                acc[key] = np.zeros(NY * NX, np.float32)
            acc[key][cell] += sw[key][k].ravel()[cell] * s / 25.4
        for key, v in acc.items():
            mem[f"{key}|snow_p{meth}"] = v.reshape(NY, NX)
        print(f"Pletcher {meth}: {len(acc)} members", flush=True)
    np.savez_compressed(os.path.join(a.out, "snow_members.npz"), **mem)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "finish"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--last-ncep")
    ap.add_argument("--last-ec")
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "fetch":
        extras(a)
        fetch_main(a)
    else:
        finish(a)

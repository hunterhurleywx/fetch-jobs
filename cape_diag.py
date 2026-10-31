"""Surface-based and mixed-layer CAPE maps per model (ECMWF HRES, EPS member median / mean, NAM 12 km), for display.

    python cape_diag.py 2026-11-02 --leads 3 [--hours 18 21 0 3]

For each model, the newest cached run in the lead's pool (CAM pools follow POGS_CAM_RUNS) at the convective day's
18/21/00/03Z. All three use the same parcel code (coarse_parcel.py, virtual temperature), so they compare like for like:
  * SB: 2 m parcel, raw 2 m temperature (the night-time surface inversion is kept, so SB CAPE falls off after dark)
  * ML: lowest 100 hPa mixed layer (the SPC / Pivotal convention), raw 2 m point
  * HRES: hres_snd, 12 levels; EPS: eps_pl, 7 levels + sp from the main file, 50 members (median, mean, 90th pct);
    NAM 12 km: the lean 15-level column from nam12_add (1000-200 hPa) + 2 m T / q2 / surface pressure.
Display only (scoring uses cin90 / mlcape90 / bpcape from derive_parcel.py). Writes output/cape_diag.npz.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np

import coarse_parcel as cp
import fetch
from config import CASES_DIR, EPS_MEMBERS, LEAN_LEVELS, add_key, cam_pools, case_setup, fh_available

KW = dict(ml_depth=100.0, parcels=("sb", "ml"), sfc="2m")


def hres(init, fh):
    s = fetch.read_cached("hres_snd", init, fh, "det")
    return None if s is None else cp.from_hres_snd(s, levels=cp.HRES_LEVELS, **KW)


def eps_member(init, fh, mem):
    p, m = fetch.read_cached("eps_pl", init, fh, mem), fetch.read_cached("eps", init, fh, mem)
    if p is None or m is None or "sp" not in m:
        return None
    return cp.from_eps_pl(p, sp=m["sp"], **KW)


def nam12(init, fh):
    a, m = fetch.read_cached("nam12_add", init, fh, "det"), fetch.read_cached("nam12", init, fh, "det")
    if a is None or m is None:
        return None
    lv = [l for l in LEAN_LEVELS if add_key("TMP", f"{l} mb") in a and add_key("RH", f"{l} mb") in a]
    sp = a.get(add_key("PRES", "surface"))
    t2 = a.get(add_key("TMP", "2 m above ground"))
    if len(lv) < 8 or sp is None or t2 is None or "q2" not in m:
        return None
    t = np.stack([a[add_key("TMP", f"{l} mb")] for l in lv]).astype(float)
    r = np.stack([cp.r_from_rh(a[add_key("RH", f"{l} mb")], a[add_key("TMP", f"{l} mb")], l) for l in lv])
    sp, t2, r2 = np.asarray(sp, float) / 100.0, np.asarray(t2, float), cp.r_from_q(np.asarray(m["q2"], float))
    ok = np.isfinite(sp) & np.isfinite(t2) & np.isfinite(r2) & np.isfinite(t).all(0) & np.isfinite(r).all(0)
    o = cp.coarse_cape_cin(lv, t[:, ok], r[:, ok], sp[ok], t2[ok], r2[ok], **KW)  # NAM grid ends inside our domain
    return {k: _fill(ok, v) for k, v in o.items()}


def _fill(ok, v):
    full = np.full(ok.shape, np.nan, np.float32)
    full[ok] = v
    return full


def newest(model, inits, valid, probe):
    """Newest init whose steps at the display hours are cached (probe(init, fh) -> fields or None)."""
    for init in sorted(inits, reverse=True):
        fh = int((valid[-1] - init).total_seconds() // 3600)
        if fh_available(model, init, fh) and probe(init, fh) is not None:
            return init
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[3])
    ap.add_argument("--hours", type=int, nargs="+", default=[18, 21, 0, 3])
    a = ap.parse_args()
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    lead = min(a.leads)
    _, pools = case_setup(conv, [lead])
    day18 = datetime(conv.year, conv.month, conv.day, 18)
    valid = [day18 + timedelta(hours=(h - 18) % 24) for h in a.hours]
    out = {}
    sources = (("hres", pools[lead], hres), ("nam12", cam_pools(conv, lead).get("nam12", []), nam12))
    for model, inits, fn in sources:
        init = newest(model, inits, valid, fn)
        if init is None:
            print(f"{model}: nothing cached for {valid[0]:%m/%d %HZ}-{valid[-1]:%HZ}")
            continue
        for v in valid:
            fh = int((v - init).total_seconds() // 3600)
            o = fn(init, fh) if fh_available(model, init, fh) else None
            if o is not None:
                for k in ("sb", "ml"):
                    out[f"{model}|{init:%Y%m%d%H}|{v:%Y%m%d%H}|{k}"] = o[f"{k}_cape"].astype(np.float32)
        print(f"{model} {init:%Y%m%d%H}: {sum(1 for k in out if k.startswith(model + '|'))} fields", flush=True)
    init = newest("eps", pools[lead], valid, lambda i, fh: eps_member(i, fh, EPS_MEMBERS[0]))
    if init is not None:
        for v in valid:
            fh = int((v - init).total_seconds() // 3600)
            if not fh_available("eps", init, fh):
                continue
            mem = [o for m in EPS_MEMBERS if (o := eps_member(init, fh, m)) is not None]
            for k in ("sb", "ml"):
                st = np.stack([o[f"{k}_cape"] for o in mem])
                base = f"eps|{init:%Y%m%d%H}|{v:%Y%m%d%H}|{k}"
                out[base + "_med"] = np.median(st, axis=0).astype(np.float32)
                out[base + "_mean"] = st.mean(axis=0).astype(np.float32)
                out[base + "_p90"] = np.percentile(st, 90, axis=0).astype(np.float32)
            out[f"eps|{init:%Y%m%d%H}|{v:%Y%m%d%H}|n"] = np.array(len(mem))
        print(f"eps {init:%Y%m%d%H}: {len(mem)} members", flush=True)
    dst = os.path.join(CASES_DIR, a.date, "output", "cape_diag.npz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)  # runs before scoring, which creates output/
    np.savez_compressed(dst, **out)
    print(f"-> {dst} ({len(out)} arrays)")


if __name__ == "__main__":
    sys.exit(main())

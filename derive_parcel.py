"""ECMWF CIN and surface-based share from the cached upper-air sets (coarse_parcel.py), cached as hres_pcl / eps_pcl.

    python derive_parcel.py 2026-11-02 --leads 3 [--procs 2]

ECMWF open data has CAPE (MU) but no CIN, so before 31 Oct 2026 HRES and EPS got no CIN penalty and no
surface-based-share check while GFS / GEFS / CAMs did (research/11_qlcs_vs_supercell/05_ecmwf_cin_triple_check.md).
For every HRES / EPS step in the case's pools this writes, on the 0.5 deg main grid:
  cin90     90-hPa mixed-layer CIN (J/kg; the NCEP UPP cin90 definition: lfc="top", 2 m inversion stripped)
  mlcape90  90-hPa mixed-layer CAPE
  bpcape    best-parcel CAPE (highest-theta-e 30-hPa layer in the lowest 180 hPa, like UPP cape180)
HRES uses all 12 hres_snd levels; EPS uses its 7 eps_pl levels plus surface pressure from the main EPS file.
Scoring merges these into the ECMWF field dicts (run_case.score_pool, ci.member_layers).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

import coarse_parcel as cp
import fetch
from config import EPS_MEMBERS, case_setup, fh_available

KEYS = {"ml_cin": "cin90", "ml_cape": "mlcape90", "bl6_cape": "bpcape"}


def derive(model, init, fh, mem):
    """Parcel fields for one HRES / EPS member-step, or None if its inputs are not cached."""
    kw = dict(ml_depth=90.0, parcels=("ml", "bl6"))
    if model == "hres":
        s = fetch.read_cached("hres_snd", init, fh, "det")
        if s is None:
            return None
        o = cp.from_hres_snd(s, levels=cp.HRES_LEVELS, **kw)
    else:
        p, m = fetch.read_cached("eps_pl", init, fh, mem), fetch.read_cached("eps", init, fh, mem)
        if p is None or m is None or "sp" not in m:
            return None
        o = cp.from_eps_pl(p, sp=m["sp"], **kw)
    return {new: o[old].astype(np.float32) for old, new in KEYS.items()}


def one(task):
    model, init, fh, mem = task
    f = derive(model, init, fh, mem)
    if f is None:
        return task, False
    fetch._save(f"{model}_pcl", init, fh, {mem: f})
    return task, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[3])
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    valid, pools = case_setup(datetime.strptime(a.date, "%Y-%m-%d"), a.leads)
    inits = sorted({i for p in pools.values() for i in p})
    tasks = []
    for model, mems in (("hres", ["det"]), ("eps", EPS_MEMBERS)):
        for init in inits:
            for v in valid:
                fh = int((v - init).total_seconds() // 3600)
                if not fh_available(model, init, fh):
                    continue
                for mem in mems:
                    if not os.path.exists(fetch.cache_path(f"{model}_pcl", init, fh, mem)):
                        tasks.append((model, init, fh, mem))
    print(f"{len(tasks)} member-steps to derive", flush=True)
    done = {"hres": [0, 0], "eps": [0, 0]}
    with ProcessPoolExecutor(a.procs) as ex:
        for (model, init, fh, mem), ok in ex.map(one, tasks, chunksize=8):
            done[model][0 if ok else 1] += 1
    for m, (ok, miss) in done.items():
        print(f"  {m}_pcl: {ok} written, {miss} skipped (inputs not cached)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

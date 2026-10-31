"""Convective signal vs forcing for the NAM 12 km and NAM 3 km runs in the lead-3 pool (POGS_CAM_RUNS), for display.

    python nam_signal.py 2026-11-02 [--hours 18 21 22 23 0 3]

From the cached 0.5 deg fields, per model / run / valid hour:
  * signal: NAM 12 km 3-h convective precip (acpcpb, nam12_x); NAM 3 km 1-h max composite reflectivity and 2-5 km UH
    (nam_ci; cell maxima, so "where", not storm mode: cam_snap.npz has the native grid)
  * forcing: 500 mb height, wind (-> relative vorticity in the plot), 700 / 500 mb omega (Pa/s, negative = rising)
  * environment: 2 m specific humidity (dryline = its sharp west-east gradient), 90 mb ML CAPE / CIN, 10 m, 850 mb
    and 500 mb winds (deep shear)
Writes output/nam_signal.npz: "<model>|<run>|<valid>|<field>" (float16). Display only.
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np

import fetch
from config import CASES_DIR, cam_pools, fh_available

ENV = ("z500", "u500", "v500", "w500", "w700", "q2", "cape90", "cin90", "u10", "v10", "u850", "v850")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--hours", type=int, nargs="+", default=[18, 21, 22, 23, 0, 3])
    a = ap.parse_args()
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    pools = cam_pools(conv, 3)
    valid = [conv + timedelta(days=h < 12, hours=h) for h in a.hours]
    out = {}
    for m in ("nam12", "nam"):
        for init in pools.get(m, []):
            for v in valid:
                fh = int((v - init).total_seconds() // 3600)
                if not fh_available(m, init, fh):
                    continue
                base = f"{m}|{init:%Y%m%d%H}|{v:%Y%m%d%H}"
                env = fetch.read_cached(m, init, fh, "det")
                if env is not None:
                    for k in ENV:
                        if k in env:
                            out[f"{base}|{k}"] = np.asarray(env[k], dtype=np.float16)
                sig = fetch.read_cached("nam12_x" if m == "nam12" else "nam_ci", init, fh, "det")
                if sig is not None:
                    for k in (("acpcpb",) if m == "nam12" else ("refc", "uh25")):
                        if k in sig:
                            out[f"{base}|{k}"] = np.asarray(sig[k], dtype=np.float16)
    o = os.path.join(CASES_DIR, a.date, "output")
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(os.path.join(o, "nam_signal.npz"), **out)
    print(f"wrote {len(out)} arrays: " + ", ".join(sorted({'|'.join(k.split('|')[:2]) for k in out})))


if __name__ == "__main__":
    main()

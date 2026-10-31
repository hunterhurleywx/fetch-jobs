"""Surface fields per model for surface analysis maps (display only): 2 m T / Td, 10 m wind, sea-level pressure.

    python sfc_fields.py 2026-11-02 [--hours 18 21 0 3 6]

For each model's newest run in the lead-3 pool (CAM runs follow POGS_CAM_RUNS) that reaches the hour: GFS, GEFS
(member mean), ECMWF HRES, EPS (member mean), NAM 12 km, NAM 3 km, RRFS. T / Td / wind from the main and upper-air
sets (as diag_upper.py); sea-level pressure from the add-on sets (GFS / GEFS PRMSL, ECMWF msl, NAM MSLET); NAM 2 m T
from its add-on set. RRFS has no cached 2 m T or pressure, so it carries Td (from q2 and terrain pressure) and wind.
Writes output/sfc_fields.npz: "<model>|<run>|<valid>|<t2m|td2m|u10|v10|mslp>" (deg C, m/s, hPa; float16).
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np

import diag_upper as du
import run_case as rc
import scoring
from config import CASES_DIR, cam_pools, case_setup, fh_available


def find(d, *names):
    """First key in d whose name starts with one of names (add-on keys carry level suffixes)."""
    if d is None:
        return None
    for n in names:
        for k in d:
            if k == n or k.startswith(n + "_") or k == n.lower():
                return d[k]
    return None


def mslp_of(model, init, fh, mem):
    a = rc.load(f"{model}_add", init, fh, mem)
    x = find(a, "prmsl", "msl", "mslet", "mslma")
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    return x / 100.0 if np.nanmax(x) > 2000 else x


def p_terrain():
    return 1013.25 * np.exp(-scoring.orography() / 8400.0) * 100.0  # Pa, rough surface pressure from terrain


def one(model, init, fh):
    if model in ("gfs", "hres"):
        f = du.member_fields(model, init, fh, "det")
        if f is None:
            return None
        out = {k: f[k] for k in ("t2m", "td2m", "u10", "v10")}
        p = mslp_of(model, init, fh, "det")
        if p is not None:
            out["mslp"] = p
        return out
    if model in ("gefs", "eps"):
        acc, n, pacc, pn = {}, 0, 0.0, 0
        for mem in rc.members_of(model):
            f = du.member_fields(model, init, fh, mem)
            if f is None:
                continue
            for k in ("t2m", "td2m", "u10", "v10"):
                acc[k] = acc.get(k, 0.0) + np.asarray(f[k], dtype=float)
            n += 1
            p = mslp_of(model, init, fh, mem)
            if p is not None:
                pacc, pn = pacc + p, pn + 1
        if not n:
            return None
        out = {k: v / n for k, v in acc.items()}
        if pn:
            out["mslp"] = pacc / pn
        return out
    m = rc.load(model, init, fh, "det")  # CAMs and NAM 12 km (winds earth-relative)
    if m is None:
        return None
    a = rc.load(f"{model}_add", init, fh, "det")
    sp = find(a, "pres_surface", "pres")
    sp = np.asarray(sp, dtype=float) if sp is not None else p_terrain()
    out = {"u10": m["u10"], "v10": m["v10"], "td2m": du.td_from_q(m["q2"], sp)}
    t2 = find(a, "tmp_2_m", "tmp_2")
    if t2 is not None:
        out["t2m"] = np.asarray(t2, dtype=float) - 273.15
    p = find(a, "mslet", "prmsl", "mslma")
    if p is not None:
        p = np.asarray(p, dtype=float)
        out["mslp"] = p / 100.0 if np.nanmax(p) > 2000 else p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--hours", type=int, nargs="+", default=[18, 21, 0, 3, 6])
    a = ap.parse_args()
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    _, pools = case_setup(conv, [3])
    cams = cam_pools(conv, 3)
    valid = [conv + timedelta(days=h < 12, hours=h) for h in a.hours]
    res = {}
    for model in ("gfs", "gefs", "hres", "eps", "nam12", "nam", "rrfs"):
        inits = cams.get(model, []) if model in ("nam12", "nam", "rrfs") else pools[3]
        for v in valid:
            for init in sorted(inits, reverse=True):  # newest run that has this hour
                fh = int((v - init).total_seconds() // 3600)
                if not fh_available(model, init, fh):
                    continue
                f = one(model, init, fh)
                if f is None:
                    continue
                for k, x in f.items():
                    res[f"{model}|{init:%Y%m%d%H}|{v:%Y%m%d%H}|{k}"] = np.asarray(x, dtype=np.float16)
                break
    out = os.path.join(CASES_DIR, a.date, "output")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "sfc_fields.npz"), **res)
    print(f"wrote {len(res)} arrays: " + ", ".join(sorted({'|'.join(k.split('|')[:2]) for k in res})))


if __name__ == "__main__":
    main()

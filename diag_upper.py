"""Upper-air diagnostics for a case: per model, cycle and valid time, the member-mean upper-air fields.

    python diag_upper.py 2026-11-02 --leads 4 3

Runs in the cloud after scoring, where the raw member fields are cached. It reads:
- GFS and HRES full columns (gfs_snd, hres_snd; 0.25 deg, thinned to the 0.5 deg grid);
- GEFS and EPS key levels (gefs_pl, eps_pl), plus 10 m wind from the main files.

It writes cases/<date>/output/upper_diag.npz (float16), keyed "<model>|<cycle YYYYMMDDHH>|<valid YYYYMMDDHH>|<var>", with:
- u/v at 10 m, 850, 700, 500 and 300 hPa;
- t at 850, 700 and 500 hPa;
- t2m and td2m (deg C);
- z500.

Maps and boundary-relative diagnostics are drawn locally from this file (plot_upper.py).
"""
import argparse
import os
from datetime import datetime

import numpy as np

import run_case as rc
from config import CASES_DIR, case_setup, fh_available

VARS = ["u10", "v10", "u850", "v850", "u700", "v700", "u500", "v500", "u300", "v300",
        "t850", "t700", "t500", "t2m", "td2m", "z500"]


def td_from_rh(t_k, rh):
    """Dewpoint (deg C) from temperature (K) and RH (%) - Magnus."""
    tc = t_k - 273.15
    g = np.log(np.clip(rh, 1, 100) / 100.0) + 17.62 * tc / (243.12 + tc)
    return 243.12 * g / (17.62 - g)


def td_from_q(q, p_pa):
    e = q * p_pa / (0.622 + 0.378 * q) / 100.0  # hPa
    g = np.log(np.maximum(e, 1e-3) / 6.112)
    return 243.12 * g / (17.62 - g)


def member_fields(model, init, fh, mem):
    """The common variable set for one member, on the 0.5 deg grid, or None."""
    if model in ("gfs", "hres"):
        s = rc.load(f"{model}_snd", init, fh, "det")
        if s is None:
            return None
        s = {k: v[::2, ::2] for k, v in s.items()}  # 0.25 -> 0.5 deg (same origin)
        out = {k: s.get(k) for k in VARS if k in s}
        out["t2m"] = s["t2m"] - 273.15
        out["td2m"] = (s["d2m"] - 273.15) if "d2m" in s else td_from_q(s["q2"], s["sp"])
        out["z500"] = s.get("z500")
        return out
    p = rc.load(f"{model}_pl", init, fh, mem)
    m = rc.load(model, init, fh, mem)
    if p is None or m is None:
        return None
    out = {k: p[k] for k in VARS if k in p}
    out["u10"], out["v10"] = m["u10"], m["v10"]
    out["t2m"] = p["t2m"] - 273.15
    out["td2m"] = (p["d2m"] - 273.15) if "d2m" in p else td_from_rh(p["t2m"], p["rh2m"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[4])
    a = ap.parse_args()
    conv_day = datetime.strptime(a.date, "%Y-%m-%d")
    valid, pools = case_setup(conv_day, a.leads)
    inits = sorted({i for p in pools.values() for i in p})
    save = {}
    for model in ("gfs", "gefs", "hres", "eps"):
        for init in inits:
            for v in valid:
                fh = int((v - init).total_seconds() // 3600)
                if not fh_available(model, init, fh):
                    continue
                acc, n = {}, 0
                for mem in rc.members_of(model):
                    f = member_fields(model, init, fh, mem)
                    if f is None:
                        continue
                    for k, x in f.items():
                        if x is not None:
                            acc[k] = acc.get(k, 0) + np.nan_to_num(x.astype(np.float64))
                    n += 1
                if n:
                    for k, x in acc.items():
                        save[f"{model}|{init:%Y%m%d%H}|{v:%Y%m%d%H}|{k}"] = (x / n).astype(np.float16)
                    save[f"{model}|{init:%Y%m%d%H}|{v:%Y%m%d%H}|n"] = np.array(n)
        print(f"{model}: {sum(1 for k in save if k.startswith(model + '|') and k.endswith('|n'))} cycle-times", flush=True)
    out = os.path.join(CASES_DIR, a.date, "output", "upper_diag.npz")
    np.savez_compressed(out, **save)
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

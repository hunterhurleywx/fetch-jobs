"""EXPERIMENTAL: score the AI models (ECMWF AIFS, AIFS ENS, NOAA AIGFS) for one convective day, like a physics model.

    python experimental/ai_models.py 2026-11-02 --day 4 --spc-lead 4 \
        --runs aifs:2026103000,2026103006,2026103012 aifs_ens:2026103006,2026103012 aigfs:2026103000,2026103006,2026103012

AI models output only the basic state (T, q, u, v, w, z on 13 pressure levels + a few surface fields), so CAPE/CIN are
computed here from those levels with experimental/coarse_parcel.py (ML -> the cape90/cin90 slot, MU -> cape180/cin180)
and the fields go through scoring.ingredients on the NCEP (CIN-aware) branch. Valid times 18Z, 00Z, 06Z (6-hourly
output). Storm signal: AIFS convective precip (>= 1 mm over the window, like the ECMWF); AIGFS has only total precip
(>= 2 mm and CAPE >= 250, the old HRES fallback). Lift and sustain as ci.member_layers. Calibrations are the physics-model
ones (14-case any-severe; hazards capped at overall), NOT fitted for AI models. Writes
cases/<date>/output/hhwx_d<day>_ai_<model>_panel.png and ai_models.npz.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_hazards as ch  # noqa: E402
import ci  # noqa: E402
import coarse_parcel as cp  # noqa: E402
import fetch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import scoring  # noqa: E402
from config import CASES_DIR  # noqa: E402
from redraw_latest import HZ, draw_panel  # noqa: E402

LEV = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
G0 = 9.80665
NAMES = {"aifs": "ECMWF AIFS (AI)", "aifs_ens": "ECMWF AIFS ENS (AI)", "aigfs": "NOAA AIGFS (AI)"}


def ec_rows(base):
    return [json.loads(l) for l in fetch.http_get(base + ".index").decode().splitlines() if l.strip()]


def aifs_member_fields(init, fh, ens):
    """{member: {key: grid}} for AIFS single (member 'det') or AIFS ENS (members '1'..'50')."""
    stream, typ, prod = ("enfo", "pf", "aifs-ens") if ens else ("oper", "fc", "aifs-single")
    base = f"{fetch.EC_BUCKET}/{init:%Y%m%d}/{init:%H}z/{prod}/0p25/{stream}/{init:%Y%m%d%H}0000-{fh}h-{stream}-{typ}"
    want = {}
    for r in ec_rows(base):
        p, lt, lv = r["param"], r["levtype"], r.get("levelist")
        if lt == "pl" and p in ("t", "q", "u", "v", "w", "z") and lv and int(lv) in LEV:
            key = f"{p}{lv}"
        elif lt == "sfc" and p in ("2t", "2d", "sp", "10u", "10v", "100u", "100v", "cp", "tp", "msl"):
            key = p
        else:
            continue
        m = str(r.get("number", "det")) if ens else "det"
        want.setdefault(m, []).append((r["_offset"], r["_offset"] + r["_length"] - 1, key))
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(want, ex.map(lambda m: fetch.fetch_ranges(base + ".grib2", want[m]), want)))
    out = {}
    for m, f in res.items():
        lv = [l for l in LEV if f"t{l}" in f and f"q{l}" in f]
        out[m] = dict(levels=lv, t=np.stack([f[f"t{l}"] for l in lv]), q=np.stack([f[f"q{l}"] for l in lv]),
                      u=np.stack([f[f"u{l}"] for l in lv]), v=np.stack([f[f"v{l}"] for l in lv]),
                      w=np.stack([f[f"w{l}"] for l in lv]), z=np.stack([f[f"z{l}"] / G0 for l in lv]),
                      sp=f["sp"] / 100.0, t2=f["2t"], td2=f["2d"], u10=f["10u"], v10=f["10v"],
                      cp=f.get("cp"), tp=f.get("tp"), msl=f.get("msl"))
    return out


LOW_BOX = (34.0, 40.5, -102.5, -93.5)  # Kansas / Oklahoma: SPC's 1 Nov meso-low corridor


def find_low(msl):
    """Lowest sea-level pressure in LOW_BOX (hPa), its position, and whether it is a closed low (every point on a ring
    1.5 deg away at least 1 hPa higher). None if no field."""
    if msl is None:
        return None
    from scipy.ndimage import gaussian_filter
    lat, lon = plots.LAT2, plots.LON2
    p = gaussian_filter(np.nan_to_num(msl / 100.0, nan=1030.0), 0.7)
    box = (lat >= LOW_BOX[0]) & (lat <= LOW_BOX[1]) & (lon >= LOW_BOX[2]) & (lon <= LOW_BOX[3])
    q = np.where(box, p, 9999.0)
    i, j = np.unravel_index(np.argmin(q), q.shape)
    ring = [p[int(round(i + 3 * np.sin(a))), int(round(j + 3 * np.cos(a)))] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    on_edge = lat[i, j] in (LOW_BOX[0], LOW_BOX[1]) or lon[i, j] in (LOW_BOX[2], LOW_BOX[3])
    return {"lat": float(lat[i, j]), "lon": float(lon[i, j]), "hpa": round(float(p[i, j]), 1),
            "depth": round(float(np.mean(ring) - p[i, j]), 1), "closed": bool(min(ring) - p[i, j] >= 1.0 and not on_edge)}


def aigfs_fields(init, fh):
    base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{init:%Y%m%d}/{init:%H}/model/atmos/grib2/aigfs.t{init:%H}z"
    out = {}
    for kind in ("pres", "sfc"):
        url = f"{base}.{kind}.f{fh:03d}.grib2"
        lines = fetch.http_get(url + ".idx").decode().strip().splitlines()
        offs = [int(l.split(":")[1]) for l in lines]
        want = []
        for i, l in enumerate(lines):
            p = l.split(":")
            end = offs[i + 1] - 1 if i + 1 < len(offs) else offs[i] + 20_000_000
            if kind == "pres" and p[4].endswith(" mb") and int(p[4].split()[0]) in LEV:
                want.append((offs[i], end, f"{p[3]}{p[4].split()[0]}"))
            elif kind == "sfc" and (p[3], p[4]) in (("TMP", "2 m above ground"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")):
                want.append((offs[i], end, p[3]))
            elif kind == "sfc" and p[3] == "APCP" and p[5].startswith("0-"):
                want.append((offs[i], end, "APCP"))
        out.update(fetch.fetch_ranges(url, want))
    lv = [l for l in LEV if f"TMP{l}" in out]
    t = np.stack([out[f"TMP{l}"] for l in lv])
    z = np.stack([out[f"HGT{l}"] for l in lv])
    zs = scoring.orography()
    sp = cp.sp_from_heights(lv, z, t, out["TMP"], zs)
    q = np.stack([out[f"SPFH{l}"] for l in lv])
    # no 2 m humidity in AIGFS: use the specific humidity of the lowest level above the ground
    j = np.argmax(np.array(lv)[:, None, None] <= sp[None] - 5.0, axis=0)
    q2 = np.take_along_axis(q, j[None], 0)[0]
    return {"det": dict(levels=lv, t=t, q=q, u=np.stack([out[f"UGRD{l}"] for l in lv]), v=np.stack([out[f"VGRD{l}"] for l in lv]),
                        w=np.stack([out[f"VVEL{l}"] for l in lv]), z=z, sp=sp, t2=out["TMP"], q2=q2,
                        u10=out["UGRD"], v10=out["VGRD"], cp=None, tp=out.get("APCP"))}


def to_scoring(m):
    """Basic-state dict -> the field dict scoring.ingredients expects on its NCEP branch."""
    lv, sp = m["levels"], m["sp"]
    idx = {l: i for i, l in enumerate(lv)}
    r = cp.r_from_q(m["q"])
    if "td2" in m:
        r2 = cp.r_from_td(m["td2"], sp)
        q2 = r2 / (1 + r2)
    else:
        q2 = m["q2"]
        r2 = cp.r_from_q(q2)
    par = cp.coarse_cape_cin(lv, m["t"], r, sp, m["t2"], r2, parcels=("ml", "mu"))
    p = np.array(lv, float)[:, None, None]
    above = p <= sp[None] - 5.0
    dp = np.abs(np.diff(np.concatenate([sp[None], np.broadcast_to(p, m["q"].shape)], 0), axis=0))
    pwat = np.sum(np.where(above, m["q"], 0) * dp * 100.0 / G0, axis=0)  # mm
    rh = lambda l: 100.0 * np.clip(r[idx[l]] / cp.rsat(m["t"][idx[l]], float(l)), 0, 1)
    return {"cape90": par["ml_cape"], "cin90": -np.abs(par["ml_cin"]), "cape180": par["mu_cape"], "cin180": -np.abs(par["mu_cin"]),
            "z500": m["z"][idx[500]], "z700": m["z"][idx[700]], "t500": m["t"][idx[500]], "t700": m["t"][idx[700]],
            "w500": m["w"][idx[500]], "w700": m["w"][idx[700]], "u500": m["u"][idx[500]], "v500": m["v"][idx[500]],
            "u850": m["u"][idx[850]], "v850": m["v"][idx[850]], "u10": m["u10"], "v10": m["v10"], "pwat": pwat, "q2": q2,
            "sp": sp * 100.0, "rh700": rh(700), "rh500": rh(500)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date"); ap.add_argument("--day", type=int, required=True); ap.add_argument("--spc-lead", type=int)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    valid = [conv + timedelta(hours=h) for h in (18, 24, 30)]
    O = os.path.join(CASES_DIR, a.date, "output")
    any14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
    hc = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
    us = plot_hhwx.us_mask()
    spc = plots.spc_lead_polys(a.date, conv, a.spc_lead)[0] if a.spc_lead else None
    if a.spc_lead in (1, 2):
        from redraw_latest import spc_hazard_panels
        spc = spc_hazard_panels(a.date, a.spc_lead) or spc
    lows = {}  # model -> cycle -> valid -> [member low dicts]
    at = lambda x, la, lo: float(x[int(round((la - 20) / .5)), int(round((lo + 130) / .5))])
    save = {}
    for spec in a.runs:
        model, cycles = spec.split(":")
        mem_env, mem_ci = {}, {}  # (init, member) -> {scheme: day-max env} / (p_model, lift, sus)
        for c in cycles.split(","):
            init = datetime.strptime(c, "%Y%m%d%H")
            per = {}
            for v in valid:
                fh = int((v - init).total_seconds() // 3600)
                try:
                    got = aigfs_fields(init, fh) if model == "aigfs" else aifs_member_fields(init, fh, model == "aifs_ens")
                except Exception as e:
                    print(f"  {model} {c} f{fh}: missing ({str(e)[:70]})"); continue
                for mem, m in got.items():
                    lw = find_low(m.get("msl"))
                    if lw:
                        lows.setdefault(model, {}).setdefault(c, {}).setdefault(f"{v:%d%H}Z", []).append(lw)
                    f = to_scoring(m)
                    ing = scoring.ingredients("gfs", f)
                    v2 = scoring.score_v2(ing, fh)
                    sc = {"v2": v2, **scoring.hazard_scores(ing, f, v2)}
                    asc = ci.ramp(-scoring.gauss(0.5 * (f["w500"] + f["w700"]), 150), 0.02, 0.15)
                    c10 = ci.ramp(scoring.gauss(ci.convergence(f["u10"], f["v10"]), 100), 0.5e-5, 3e-5)
                    c85 = ci.ramp(scoring.gauss(ci.convergence(f["u850"], f["v850"]), 100), 0.5e-5, 3e-5)
                    cin_ok = ci.ramp(np.nan_to_num(f["cin90"]), -150, -25)
                    lift = np.maximum(asc, np.maximum(c10, c85)) * np.maximum(cin_ok, 0.7 * asc)
                    sus = 0.4 + 0.6 * np.maximum(ci.ramp(0.5 * (f["rh700"] + f["rh500"]), 30, 65), ci.ramp(f["cape90"], 1000, 2500))
                    d = per.setdefault(mem, {"sc": [], "lift": [], "sus": [], "pr": [], "cape": []})
                    d["sc"].append(sc); d["lift"].append(lift); d["sus"].append(sus)
                    d["pr"].append(m["cp"] if m["cp"] is not None else m["tp"]); d["cape"].append(f["cape90"])
                print(f"  {model} {c} f{fh}: {len(got)} member(s)", flush=True)
            for mem, d in per.items():
                if len(d["sc"]) < 2:
                    continue
                mem_env[(c, mem)] = {k: np.max([s[k] for s in d["sc"]], axis=0) for k in d["sc"][0]}
                acc = 1000.0 * (d["pr"][-1] - d["pr"][0]) if model != "aigfs" else d["pr"][-1] - d["pr"][0]
                fired = acc >= 1.0 if model != "aigfs" else (acc >= 2.0) & (np.max(d["cape"], axis=0) >= 250)
                mem_ci[(c, mem)] = (maximum_filter(fired.astype(float), size=3), np.max(d["lift"], 0), np.max(d["sus"], 0))
        if not mem_env:
            print(f"{model}: no data"); continue
        keys = list(mem_env)
        stack = lambda k: np.stack([mem_env[x][k] for x in keys])
        comb = (lambda s: scoring.trimmed_mean(s)) if model == "aifs_ens" else (lambda s: s.mean(0))
        layers = [np.mean([mem_ci[x][i] for x in keys], axis=0) for i in range(3)]
        prod = lambda k: ci.finals(comb(stack(k)), *layers)[1]["env*ci"]
        psev = np.where(us, ch.calibrated(prod("v2"), any14), 0)
        maps = [("Overall severe", psev)] + [(nm, np.where(us, np.minimum(ch.calibrated(prod(k), hc[k]), psev), 0)) for k, nm in HZ]
        out = os.path.join(O, f"hhwx_d{a.day}_ai_{model}_panel.png")
        draw_panel(maps, f"HHwx Calibrated Guidance Day-{a.day}: {NAMES[model]}, {cycles.replace(',', ' / ')} (EXPERIMENTAL)",
                   out, spc, a.spc_lead, at)
        save.update({f"{model}_{nm}": p for nm, p in maps})
        for c in cycles.split(","):  # each run on its own (same calibration), for run-to-run consistency
            ck = [x for x in keys if x[0] == c]
            if not ck:
                continue
            st = lambda k: np.stack([mem_env[x][k] for x in ck])
            lay_c = [np.mean([mem_ci[x][i] for x in ck], axis=0) for i in range(3)]
            raw_c = ci.finals(comb(st("v2")), *lay_c)[1]["env*ci"]
            save[f"{model}|cyc{c}|raw"] = raw_c
            save[f"{model}|cyc{c}|Overall severe"] = np.where(us, ch.calibrated(raw_c, any14), 0)
            save[f"{model}|cyc{c}|Tornado"] = np.where(us, np.minimum(ch.calibrated(ci.finals(comb(st("tor")), *lay_c)[1]["env*ci"], hc["tor"]),
                                                                      save[f"{model}|cyc{c}|Overall severe"]), 0)
        print(f"{model}: {len(keys)} member-runs | " + " | ".join(f"{nm} max {p.max():.1f} OKC {at(p, 35.5, -97.5):.1f}" for nm, p in maps))
    np.savez_compressed(os.path.join(O, "ai_models.npz"), **save)
    json.dump(lows, open(os.path.join(O, "ai_lows.json"), "w"), indent=1)
    for model, by_c in lows.items():
        for c, by_v in sorted(by_c.items()):
            for v, ls in sorted(by_v.items()):
                closed = [l for l in ls if l["closed"]]
                print(f"LOW {model} {c} valid {v}: {len(closed)}/{len(ls)} closed lows in KS/OK"
                      + (f", mean {np.mean([l['lat'] for l in closed]):.1f}N {np.mean([l['lon'] for l in closed]):.1f} "
                         f"{np.mean([l['hpa'] for l in closed]):.0f} hPa" if closed else ""))


if __name__ == "__main__":
    main()

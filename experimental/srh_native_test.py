"""EXPERIMENTAL: native CAM helicity vs the 2-point proxy in the Day 2-1 tornado map (Hunter, 1 Nov 2026).

    python experimental/srh_native_test.py 2025-05-16 [--weights-flag ...]

Needs output/results_2pt.npz (scored with POGS_SRH=2pt) and output/results_native.npz (POGS_SRH=native) from the same
run, plus ci_layers.npz (P(CI) does not depend on helicity). For Days 2 and 1 and each variant: the live track-record
blend (consistency_blend.py --mode track), calibrated like the maps, scored against the day's tornado reports within
40 km (25 mi) over the US. Writes <POGS_SRH_OUT>/srhtest_<date>.json (Brier sums, events, placement) and
srhtest_<date>.npz (raw blend tornado index per lead and variant + report masks, for a pooled leave-one-case-out
recalibration later: helicity changes the index range, so the live calibration alone would not be a fair judge).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import calibrate_hazards as ch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import verify_case as vc  # noqa: E402
from config import CASES_DIR, PROJECT_DIR  # noqa: E402

VARIANTS = ("2pt", "native")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--blend-args", nargs="*", default=[], help="extra consistency_blend.py flags, e.g. --all-models")
    a = ap.parse_args()
    out = os.environ.get("POGS_SRH_OUT", os.path.join(PROJECT_DIR, "research", "29_native_srh"))
    os.makedirs(out, exist_ok=True)
    O = os.path.join(CASES_DIR, a.date, "output")
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    rep = plots.load_reports(a.date, conv)
    if rep is None:
        print("no reports")
        return
    tor_rep = rep[rep["type"] == "tornado"]
    us = plot_hhwx.us_mask() & vc.DOM
    obs = vc.report_mask(tor_rep)
    hc = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
    any14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
    live = os.path.join(O, "results.npz")
    keep = live + ".keep"
    if os.path.exists(live):
        shutil.copy2(live, keep)
    res = {"date": a.date, "n_tor": int(len(tor_rep)), "leads": {}}
    arrays = {"us": us, "obs_tor": obs}
    try:
        for L in (2, 1):
            res["leads"][L] = {}
            for v in VARIANTS:
                src = os.path.join(O, f"results_{v}.npz")
                if not os.path.exists(src):
                    print(f"missing {src}")
                    continue
                shutil.copy2(src, live)
                tag = f"_srh{v}"
                r = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "consistency_blend.py"), a.date,
                                    "--lead", str(L), "--hazards", "--mode", "track", "--no-draw", "--tag", tag, *a.blend_args],
                                   capture_output=True, text=True)
                p = os.path.join(O, f"consistency_raw_d{L}{tag}.npz")
                if r.returncode != 0 or not os.path.exists(p):
                    print(f"lead {L} {v}: blend failed: {r.stderr.strip()[-200:]}")
                    continue
                z = np.load(p)
                sev = np.where(us, ch.calibrated(z["sev"], any14), 0)
                tor = np.where(us, np.minimum(ch.calibrated(z["tor"], hc["tor"]), sev), 0)
                pr, o = tor[us] / 100.0, obs[us].astype(float)
                row = {"weights": {k[2:]: round(float(z[k]), 3) for k in z.files if k.startswith("w_")},
                       "bs_sum": float(((pr - o) ** 2).sum()), "n": int(us.sum()), "events": float(o.sum()),
                       "max": round(float(tor.max()), 1)}
                i0, j0 = np.unravel_index(int(tor.argmax()), tor.shape)
                row["max_at"] = [float(plots.LAT2[i0, j0]), float(plots.LON2[i0, j0])]
                if len(tor_rep):
                    la, lo = tor_rep["lat"].to_numpy(), tor_rep["lon"].to_numpy()
                    d = 6371 * np.hypot(np.deg2rad(la - row["max_at"][0]),
                                        np.deg2rad(lo - row["max_at"][1]) * np.cos(np.deg2rad(la)))
                    row["max_to_nearest_tor_km"] = round(float(d.min()), 0)
                    i = np.clip(np.round((la - 20) / 0.5).astype(int), 0, tor.shape[0] - 1)
                    j = np.clip(np.round((lo + 130) / 0.5).astype(int), 0, tor.shape[1] - 1)
                    row["hit2"], row["hit5"] = round(float(np.mean(tor[i, j] >= 2)), 3), round(float(np.mean(tor[i, j] >= 5)), 3)
                res["leads"][L][v] = row
                arrays[f"raw_tor_d{L}_{v}"] = z["tor"].astype(np.float32)
                arrays[f"raw_sev_d{L}_{v}"] = z["sev"].astype(np.float32)
                print(f"lead {L} {v}: tor max {row['max']} at {row['max_at']} bs {row['bs_sum']:.2f} "
                      f"nearest tor {row.get('max_to_nearest_tor_km')} km")
    finally:
        if os.path.exists(keep):
            shutil.move(keep, live)
    json.dump(res, open(os.path.join(out, f"srhtest_{a.date}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, f"srhtest_{a.date}.npz"), **arrays)


if __name__ == "__main__":
    main()

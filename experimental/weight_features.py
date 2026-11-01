"""EXPERIMENTAL weighting test, per-case step (roadmap Phase 2, 1 Nov 2026): save what the pooled weighting test needs.

    python experimental/weight_features.py 2025-05-16

For every lead and model in results.npz / ci_layers.npz: the model's own pooled product per hazard (overall = v2,
tor, wind, hail; env x its CI layer, each run counting equally, as the track blend does), its run-to-run consistency
(consistency_blend.run_consistency on the overall product: score, core IoU, move km, peak swing) and its run count.
Pool rules as live: globals use their newest POOL_CYCLES runs; RAP at Days 2-1 only from the 03Z day-before run.
Plus the day's report grids (any, tornado, wind, hail; 40 km) and the US mask. Raw index, float16, no calibration:
the pooled script fits per-model calibrations and weights leave-one-case-out.
Writes <POGS_WT_OUT>/wfeat_<date>.npz (a few MB).
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import calibrate_hazards as ch  # noqa: E402
import ci  # noqa: E402
import consistency_blend as cb  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import verify_case as vc  # noqa: E402
from config import CASES_DIR, POOL_CYCLES, PROJECT_DIR  # noqa: E402

MODELS = ("gfs", "gefs", "hres", "eps", "nam12", "rrfs", "nam", "hrrr", "rap", "refs")
SCHEME = {"sev": "v2", "tor": "tor", "wind": "wind", "hail": "hail"}
REGIONAL = {"nam12", "rrfs", "nam", "hrrr", "rap", "refs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    a = ap.parse_args()
    out_dir = os.environ.get("POGS_WT_OUT", os.path.join(PROJECT_DIR, "research", "30_weighting", "features"))
    os.makedirs(out_dir, exist_ok=True)
    O = os.path.join(CASES_DIR, a.date, "output")
    res, z = np.load(os.path.join(O, "results.npz")), np.load(os.path.join(O, "ci_layers.npz"))
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    us = plot_hhwx.us_mask()
    W = np.cos(np.deg2rad(plots.LAT2))
    save = {"us": us, "dom": vc.DOM}
    for hz in (None, "tor", "wind", "hail"):
        save[f"obs_{hz or 'sev'}"] = vc.report_mask(ch.reports(a.date, hz))
    leads = sorted({int(k.split("__")[0]) for k in z.files if k.endswith("__p_model") and k.count("__") == 1})
    for L in leads:
        P = {}
        for m in MODELS:
            if f"{L}__{m}__p_model" not in z.files:
                continue
            lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
            cyc = sorted(k[-10:] for k in res.files if k.startswith(f"v2|{L}|{m}|cyc"))
            if m not in REGIONAL:
                cyc = cyc[-POOL_CYCLES:]
            if m == "rap" and L <= 2:
                first = f"{conv - timedelta(days=1) + timedelta(hours=3):%Y%m%d%H}"
                cyc = [c for c in cyc if c >= first]
            cyc = [c for c in cyc if all(f"{s}|{L}|{m}|cyc{c}" in res.files for s in SCHEME.values())]
            if not cyc:
                continue
            P[m] = {hz: [(c, ci.finals(res[f"{s}|{L}|{m}|cyc{c}"], *lay)[1]["env*ci"]) for c in cyc]
                    for hz, s in SCHEME.items()}
        if not P:
            continue
        act = us & (np.max([p for v in P.values() for _, p in v["sev"]], axis=0) > 0.05)
        for m, v in P.items():
            for hz in SCHEME:
                save[f"{L}|{m}|{hz}"] = np.where(us, np.mean([p for _, p in v[hz]], axis=0), 0).astype(np.float16)
            st = cb.run_consistency(v["sev"], act, W)
            save[f"{L}|{m}|cons"] = np.array(st if st else (np.nan,) * 4, dtype=np.float32)
            save[f"{L}|{m}|nruns"] = np.int16(len(v["sev"]))
        print(f"lead {L}: " + ", ".join(f"{m}({len(v['sev'])})" for m, v in P.items()))
    np.savez_compressed(os.path.join(out_dir, f"wfeat_{a.date}.npz"), **save)
    print("->", os.path.join(out_dir, f"wfeat_{a.date}.npz"))


if __name__ == "__main__":
    main()

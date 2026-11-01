"""EXPERIMENTAL: calibrated guidance for one 3-h block (b21 = 18-21Z, b00 = 21-00Z, b03 = 00-03Z, b06 = 03-06Z).

    python experimental/block_maps.py 2026-11-02 --block b21 [--lead 3] [--spc-lead 3]

Per model: environment at the block's two valid times (max) x the block's storm-start layer (ci_layers.npz
<lead>__<model>__<block>__p_model / p_lift), calibrated with cases/calibration_blocks.json (event: a report within
25 mi during the block). Hazards are capped at the block's overall severe. Blend: the page's consistency weights
(consistency_raw_d<lead>.npz w_<model>; in track mode Days 2-1 are regional only) applied to the raw products; only
models with w > 0 enter. Overall is capped at tor + wind + hail (redraw_latest.union_cap).
Writes output/hhwx_d<lead>_<block>_panel_{blend,<model>}.png and prints maxima / OKC values.

    python experimental/block_maps.py 2026-11-02 --combine b00 b03 --lead 2 --spc-lead 2

--combine: P(report in any of the blocks) = 1 - prod(1 - p_block) per hazard (blocks treated as independent),
written as output/hhwx_d<lead>_<first>-<last>_panel_blend.png.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate_hazards as ch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import redraw_latest as rl  # noqa: E402
from config import CASES_DIR  # noqa: E402

BLOCKS = {"b21": (18, 21, "21Z", "18-21Z (noon-3 pm CST)"), "b00": (21, 24, "00Z", "21-00Z (3-6 pm CST)"),
          "b03": (24, 27, "03Z", "00-03Z (6-9 pm CST)"), "b06": (27, 30, "06Z", "03-06Z (9 pm-midnight CST)")}
SCHEME = {"any": "v2", "tor": "tor", "wind": "wind", "hail": "hail"}
HZ = (("tor", "Tornado"), ("wind", "Wind"), ("hail", "Hail"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--block", choices=list(BLOCKS))
    ap.add_argument("--combine", nargs="+", choices=list(BLOCKS))
    ap.add_argument("--lead", type=int, default=3)
    ap.add_argument("--spc-lead", type=int, default=3)
    ap.add_argument("--cal", default="calibration_blocks.json")
    ap.add_argument("--weights-tag", default="", help="read consistency_raw_d<lead><tag>.npz (e.g. _all)")
    ap.add_argument("--out-tag", default="", help="suffix for the output PNG / summary names")
    a = ap.parse_args()
    if bool(a.block) == bool(a.combine):
        ap.error("give --block or --combine")
    O = os.path.join(CASES_DIR, a.date, "output") + os.sep
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    res, z = np.load(O + "results.npz"), np.load(O + "ci_layers.npz")
    cal = json.load(open(os.path.join(CASES_DIR, a.cal)))
    raw = np.load(O + f"consistency_raw_d{a.lead}{a.weights_tag}.npz")
    W = {k[2:]: float(raw[k]) for k in raw.files if k.startswith("w_") and float(raw[k]) > 0}
    track = "mode" in raw.files and str(raw["mode"]) == "track"
    spc = plots.spc_lead_polys(a.date, conv, a.spc_lead)[0] if a.spc_lead else None
    spc = (rl.spc_hazard_panels(a.date, a.spc_lead) if a.spc_lead in (1, 2) else None) or spc
    d03 = f"{conv - timedelta(days=1):%Y%m%d}03" if track and a.lead <= 2 else ""
    us = plot_hhwx.us_mask()
    at = lambda x, la, lo: float(x[int(round((la - 20) / .5)), int(round((lo + 130) / .5))])
    L = a.lead
    models = ("eps", "gefs", "hres", "gfs", "nam12", "rrfs", "nam", "hrrr", "rap", "refs")

    def block_maps(b):
        h0, h1, name, span = BLOCKS[b]
        t0, t1 = conv + timedelta(hours=h0), conv + timedelta(hours=h1)

        def raw_product(m, hz):
            ks = [f"{SCHEME[hz]}|{L}|{m}|vt{t:%Y%m%d%H}" for t in (t0, t1)]
            pm, pl = f"{L}__{m}__{b}__p_model", f"{L}__{m}__{b}__p_lift"
            if not all(k in res.files for k in ks) or pm not in z.files or pl not in z.files:
                return None
            return np.maximum(res[ks[0]], res[ks[1]]) * (0.6 * z[pm] + 0.4 * z[pl])

        def calibrated(field, hz):
            c = cal[f"{b}|{hz}"]
            return np.where(us, ch.calibrated(field, c), 0)

        def maps_for(prod):
            sev = calibrated(prod["any"], "any")
            out = [("Overall severe", sev)]
            for hz, nm in HZ:
                out.append((nm, np.minimum(calibrated(prod[hz], hz), sev)))
            return out

        per = {m: {hz: raw_product(m, hz) for hz in SCHEME} for m in models}
        per = {m: v for m, v in per.items() if all(x is not None for x in v.values())}
        blend = {}
        for hz in SCHEME:
            acc, ws = 0.0, 0.0
            for m, w in W.items():
                if m in per:
                    acc, ws = acc + w * per[m][hz], ws + w
            blend[hz] = acc / ws if ws else None
        return span, rl.union_cap(maps_for(blend)), {m: rl.union_cap(maps_for(per[m])) for m in per}

    def drawn(maps, label, out, mods):
        rl.draw_panel(maps, f"HHwx Calibrated Guidance Day-{L}, {label}", out, spc, a.spc_lead, at,
                      runs=rl.runs_text(res, L, mods, d03), fine=True)
        out_ = {}
        for nm, p in maps:
            i, j = np.unravel_index(int(p.argmax()), p.shape)
            out_[nm] = (round(float(p.max()), 1), round(at(p, 35.5, -97.5), 1), f"{plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}")
        return out_

    summary = {}
    blended = [m for m in W if m in models]
    if a.block:
        b = a.block
        span, bm, per = block_maps(b)
        summary["blend"] = drawn(bm, f"{span} block", O + f"hhwx_d{L}_{b}_panel_blend{a.out_tag}.png", blended)
        for m, mm in per.items():
            summary[m] = drawn(mm, f"{span} block: {rl.NAMES.get(m, m)}", O + f"hhwx_d{L}_{b}_panel_{m}.png", [m])
        tag = b
    else:
        parts = [block_maps(b) for b in a.combine]
        comb = []
        for k, (nm, _) in enumerate(parts[0][1]):
            q = np.ones_like(parts[0][1][k][1])
            for _, bm, _ in parts:
                q = q * (1 - bm[k][1] / 100)
            comb.append((nm, 100 * (1 - q)))
        comb = rl.union_cap(comb)
        h0, h1 = BLOCKS[a.combine[0]][0] % 24, BLOCKS[a.combine[-1]][1] % 24
        tag = f"{a.combine[0]}-{a.combine[-1]}"
        np.savez_compressed(O + f"block_{tag}_d{L}{a.out_tag}.npz", **{nm: p for nm, p in comb})
        summary["blend"] = drawn(comb, f"{h0:02d}-{h1:02d}Z", O + f"hhwx_d{L}_{tag}_panel_blend{a.out_tag}.png", blended)
    for k, v in summary.items():
        print(k, "  ".join(f"{nm} max {x} at {w} OKC {o}" for nm, (x, o, w) in v.items()))
    json.dump(summary, open(O + f"block_summary_{tag}{a.out_tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()

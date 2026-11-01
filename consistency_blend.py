"""EXPERIMENTAL consistency/convergence-weighted blend (Hunter's rule: trends and consistency first, no flat composite).

    python consistency_blend.py 2026-11-02 --lead 4 [--spc]

Inputs, per model and cycle in the lead's pool:
- P[m][c]: that cycle's pooled environment (results.npz v2|L|m|cyc...) x the model's CI layer (ci_layers.npz).
  This is the same raw index as the maps.
- "active area": US points where any model-cycle P > 0.05. All correlations are area-weighted over it.

Weights (all shown in the printout):
1. Within a model, cycle c gets w_c = max(corr(P_c, L3), 0)^2, where L3 is the mean of that model's last 3 cycles.
   Runs that agree with the model's settled recent solution count; flips away from it count little.
   A flip-flopping model gives roughly even weights, so this is not blind recency.
2. Stability k_m = 0.5 + 0.5 ramp(s, 0.80, 0.98), where s is the mean run-to-run correlation of the last 3 cycles.
3. Cross-model support a_m = 0.5 + 0.5 ramp(r, 0.5, 0.9), where r = corr(model's L3, mean L3 of the other models).
   A model whose current solution the others do not share counts less; this applies to either family.
4. W_m = base blend weight x k_m x a_m, renormalised.

Blend = sum_m W_m sum_c w_c P_c / sum_c w_c. It is calibrated like the main map (provisional calibration + base-rate
shift, bust_summary.calibrated).

Caveat: the calibration was fitted on the flat blend, a trimmed mean over all member-runs. A "flat" version built by
this same method (all weights equal per run, base model weights) is printed so the comparison is like for like.
Not backtested yet.
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np

import bust_summary as bs
import ci
import plot_hhwx
import plots
import scoring
from config import CASES_DIR, POOL_CYCLES

ramp = scoring.ramp


def wcorr(a, b, m, w):
    a, b, w = a[m], b[m], w[m]
    a, b = a - (w * a).sum() / w.sum(), b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


CAM_FAMILY = {"nam", "hrrr", "rap", "refs"}
FAMILY_SHARE = {1: 0.80, 2: 0.65}  # CAM family's share by lead (Hunter: CAMs get hefty weight from the start)
TRACK_SHOWN_ONLY = {"rrfs"}  # Hunter: "RRFS is just goofy"
MAX_W = 0.40  # largest share any one model may take within its group
GLOBALS = {"gfs", "gefs", "hres", "eps"}  # Hunter (30 Oct, again 1 Nov): Days 2-1 blend no global models; shown only


def run_consistency(v, act, W):
    """How consistent a model's own runs have been: successive-run core overlap (IoU of each run's top cells, the
    size of a Slight-risk-like area), core movement (km between core centroids) and swings in its peak (95th pct).
    Returns (score 0-1, iou, move_km, swing) or None with fewer than 2 runs."""
    # judge runs at least 6 h apart: hourly HRRR / RAP runs look "steady" just because they are 1 h apart (1 Nov review)
    kept, last = [], None
    for c, p in v:
        t = datetime.strptime(c[-10:], "%Y%m%d%H")
        if last is None or (t - last).total_seconds() >= 6 * 3600:
            kept.append((c, p))
            last = t
    v = kept
    if len(v) < 2:
        return None
    n = max(int(0.12 * act.sum()), 20)
    lat, lon = plots.LAT2, plots.LON2
    cores, peaks = [], []
    for _, p in v:
        q = np.where(act, p, -1.0)
        top = np.zeros(p.shape, bool)
        top.ravel()[np.argsort(q.ravel())[::-1][:n]] = True
        cores.append(top)
        peaks.append(np.percentile(p[act], 95))
    ious, moves, swings = [], [], []
    for a_, b_, pa, pb in zip(cores[:-1], cores[1:], peaks[:-1], peaks[1:]):
        ious.append((a_ & b_).sum() / (a_ | b_).sum())
        ca = (lat[a_].mean(), lon[a_].mean())
        cb = (lat[b_].mean(), lon[b_].mean())
        moves.append(111.0 * np.hypot(ca[0] - cb[0], (ca[1] - cb[1]) * np.cos(np.deg2rad(0.5 * (ca[0] + cb[0])))))
        swings.append(abs(pb - pa) / max(0.5 * (pa + pb), 1e-6))
    iou, mv, sw = float(np.mean(ious)), float(np.mean(moves)), float(np.mean(swings))
    score = (iou * np.exp(-mv / 150.0) * np.exp(-sw / 0.3)) ** (1 / 3)
    return score, iou, mv, sw


def track_weights(P, rows, act, W, L, one_family=False, global_share=0.0, all_models=False):
    """Model weight = (own-run consistency)^2 within its family (CAMs vs the rest), family shares by lead; every run of
    a model counts equally (no recency). A model with one run has no track record yet: it gets its family's LOWEST score."""
    stats = {m: run_consistency(v, act, W) for m, v in P.items()}
    share = FAMILY_SHARE.get(L, 0.0)
    fam = (lambda m: "cam") if one_family else (lambda m: "cam" if m in CAM_FAMILY else "other")
    shown_only = TRACK_SHOWN_ONLY | (GLOBALS if L <= 2 and global_share <= 0 else set())
    if all_models:  # Hunter, 1 Nov 2026 (Monday view): every model incl. RRFS and the globals, one group, consistency only
        shown_only, one_family, fam = set(), True, (lambda m: "cam")
    elif L <= 2 and global_share <= 0:  # regional models only (NAM 3 km, HRRR, REFS, RAP, NAM 12 km): one group
        one_family = True
        fam = lambda m: "cam"
    elif L <= 2:  # test: globals get global_share, the regional models (incl. NAM 12 km) the rest
        fam = lambda m: "other" if m in GLOBALS else "cam"
        share = 1.0 - global_share
    used = [m for m in P if m not in shown_only]
    present = {fam(m) for m in used}
    shares = {"cam": share if "other" in present else 1.0, "other": (1 - share) if "cam" in present else 1.0}
    if L >= 3:
        shares = {"cam": 0.0, "other": 1.0} if "other" in present else shares
    raw = {}
    for f in ("cam", "other"):
        ms = [m for m in used if fam(m) == f]
        known = [stats[m][0] for m in ms if stats[m]]
        low = float(np.min(known)) if known else 0.5
        for m in ms:
            raw[m] = (stats[m][0] if stats[m] else low) ** 2
        tot = sum(raw[m] for m in ms) or 1.0
        for m in ms:
            raw[m] = raw[m] / tot
        for _ in range(10):  # no single model above MAX_W of its group (1 Nov review: NAM family 0.75-0.99 at Day 2)
            over = [m for m in ms if raw[m] > MAX_W + 1e-9]
            if not over or len(ms) * MAX_W < 1:
                break
            extra = sum(raw[m] - MAX_W for m in over)
            free = [m for m in ms if raw[m] < MAX_W - 1e-9]
            for m in over:
                raw[m] = MAX_W
            fs = sum(raw[m] for m in free) or 1.0
            for m in free:
                raw[m] += extra * raw[m] / fs
        for m in ms:
            raw[m] = shares[f] * raw[m]
    print("\nTRACK-RECORD WEIGHTS (no default model weights; every run counts equally)")
    print(f"{'model':5s} family runs  core IoU  move km  peak swing  consistency  weight")
    blend, wsum = 0.0, 0.0
    for m, v in P.items():
        st = stats[m]
        w = raw.get(m, 0.0)
        txt = f"{st[1]:.2f}      {st[2]:5.0f}    {st[3]:.2f}        {st[0]:.2f}" if st else "  -          -        -        (1 run)"
        print(f"{m:5s} {fam(m):6s} {len(v):3d}   {txt}       {w:.2f}{'  (shown only)' if m in shown_only else ''}")
        n = len(v)
        rows[m]["wc"] = np.full(n, 1.0 / n)
        rows[m]["Wm"] = w
        if w > 0:
            blend = blend + w * np.mean([p for _, p in v], axis=0)
            wsum += w
    return blend / wsum, wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--lead", type=int, default=4)
    ap.add_argument("--spc", action="store_true")
    ap.add_argument("--hazards", action="store_true", help="also tornado / hail / wind maps with the same weights")
    ap.add_argument("--exclude", nargs="*", default=[], help="models to leave out, e.g. rrfs (with/without comparisons)")
    ap.add_argument("--no-draw", action="store_true", help="skip the single-map PNGs (batch re-weighting)")
    ap.add_argument("--tag", default="", help="suffix for the saved blend file (side versions, e.g. _cam)")
    ap.add_argument("--global-share", type=float, default=0.0,
                    help="Days 2-1 test only: give the global models this share of the blend (0 = live rule: none)")
    ap.add_argument("--all-models", action="store_true", help="track mode: every model (RRFS and globals too) in one group")
    ap.add_argument("--one-family", action="store_true", help="track mode: no CAM/other split; consistency alone")
    ap.add_argument("--mode", choices=["base", "track"], default="base",
                    help="track (Hunter, 1 Nov 2026; Monday test): no default model weights; each model weighted by how "
                         "consistent its own runs have been (core overlap, core movement, peak swings), CAM family share "
                         "by day, RAP only from the 03Z day-before run, RRFS shown not blended")
    a = ap.parse_args()
    out = os.path.join(CASES_DIR, a.date, "output")
    res, z = np.load(os.path.join(out, "results.npz")), np.load(os.path.join(out, "ci_layers.npz"))
    cal = json.load(open(os.path.join(CASES_DIR, "calibration_provisional.json")))["env_ci"]
    us, L = plot_hhwx.us_mask(), a.lead
    W = np.cos(np.deg2rad(plots.LAT2))
    base = scoring.weights(L)
    P = {}
    for m in ("gfs", "gefs", "hres", "eps", "nam12", "rrfs", "nam", "hrrr", "rap", "refs"):
        if f"{L}__{m}__p_model" not in z.files or m in a.exclude:
            continue
        lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
        ks = sorted(k for k in res.files if k.startswith(f"v2|{L}|{m}|cyc"))
        if m not in ("nam12", "rrfs", "nam", "hrrr", "rap", "refs"):  # globals: the pool's newest POOL_CYCLES (a --pool-end run adds newer cycles)
            ks = ks[-POOL_CYCLES:]
        if a.mode == "track" and m == "rap":  # RAP only "gets its act together" from the 03Z day-before run (Hunter)
            first = datetime.strptime(a.date, "%Y-%m-%d") - timedelta(days=1) + timedelta(hours=3)
            ks = [k for k in ks if datetime.strptime(k[-10:], "%Y%m%d%H") >= first]
            if not ks:
                continue
        P[m] = [(k[-10:], ci.finals(res[k], *lay)[1]["env*ci"]) for k in ks]
    act = us & (np.max([p for v in P.values() for _, p in v], axis=0) > 0.05)
    L3 = {m: np.mean([p for _, p in v[-3:]], axis=0) for m, v in P.items()}
    rows, wsum, flat, flatw = {}, 0.0, 0.0, 0.0
    blend = np.zeros_like(act, float)
    for m, v in P.items():
        wc = np.array([max(wcorr(p, L3[m], act, W), 0) ** 2 for _, p in v])
        pairs = [wcorr(v[i][1], v[i + 1][1], act, W) for i in range(max(len(v) - 3, 0), len(v) - 1)]
        s = np.mean(pairs) if pairs else 0.98  # a model with one run so far has no stability record: neutral
        k = 0.5 + 0.5 * ramp(s, 0.80, 0.98)
        others = np.mean([L3[o] for o in P if o != m], axis=0)
        r = wcorr(L3[m], others, act, W)
        al = 0.5 + 0.5 * ramp(r, 0.5, 0.9)
        Wm = base.get(m, 0) * k * al
        rows[m] = dict(cyc=[c for c, _ in v], wc=wc / wc.sum(), s=s, k=float(k), r=r, a=float(al), Wm=Wm)
        blend += Wm * np.tensordot(wc / wc.sum(), np.stack([p for _, p in v]), 1)
        wsum += Wm
        flat += base.get(m, 0) * np.mean([p for _, p in v], axis=0)
        flatw += base.get(m, 0)
    blend /= wsum
    flat /= flatw
    if a.mode == "track":
        blend, wsum = track_weights(P, rows, act, W, L, a.one_family, a.global_share, a.all_models)
    lab = lambda c: f"{c[6:8]}/{c[8:10]}Z"
    print(f"Day {L} pool cycles: {', '.join(lab(c) for c in rows[next(iter(rows))]['cyc'])}  (active area {int(act.sum())} cells)")
    print(f"{'model':5s}  {'cycle weights (oldest -> newest)':42s} stability  k     support  a     base -> final weight")
    for m, r in rows.items():
        print(f"{m:5s}  {' '.join(f'{x:.2f}' for x in r['wc']):42s} {r['s']:.3f}    {r['k']:.2f}  {r['r']:.2f}     {r['a']:.2f}  "
              f"{base.get(m, 0):.2f} -> {r['Wm'] / wsum:.2f}")
    at = lambda p, la, lo: p[int(round((la - 20) / .5)), int(round((lo + 130) / .5))]
    spc_mask, polys = None, None
    if a.spc:
        polys = plots.spc_lead_polys(a.date, datetime.strptime(a.date, "%Y-%m-%d"), L)[0]
        from shapely.prepared import prep
        from shapely.geometry import Point
        g = [prep(p) for p, _ in polys]
        spc_mask = np.array([[any(x.contains(Point(lo, la)) for x in g) for lo, la in zip(r1, r2)]
                             for r1, r2 in zip(plots.LON2, plots.LAT2)])
    print("\nmap                         max     where           OKC   Wichita Falls  Ardmore  DFW   Paris  15% cells  IoU vs SPC (same-size core)")
    for name, f in (("official blend (flat)", z[f"{L}__env*ci"]), ("same-method flat", flat), ("consistency-weighted", blend)):
        p = np.where(us, bs.calibrated(f, cal), 0)
        i, j = np.unravel_index(np.argmax(p), p.shape)
        iou = ""
        if spc_mask is not None:
            n = int(spc_mask.sum())
            top = np.zeros(p.shape, bool)
            top.ravel()[np.argsort(p.ravel())[::-1][:n]] = True
            iou = f"{(top & spc_mask).sum() / (top | spc_mask).sum():.2f}"
        print(f"{name:26s} {p.max():5.1f}%  {plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}   {at(p, 35.5, -97.5):4.1f}  {at(p, 34, -98.5):5.1f}"
              f"          {at(p, 34, -97):4.1f}    {at(p, 33, -97):4.1f}  {at(p, 33.5, -95.5):4.1f}   {int((p >= 15).sum()):4d}       {iou}")
        if name == "consistency-weighted":
            dst = os.path.join(out, f"hhwx_d{L}_blend_consistency{'_spc' if a.spc else ''}.png")
            if not a.no_draw:
                plot_hhwx.draw(p, f"HHwx Prob Severe (Calibrated) Day-{L} (Consistency-Weighted Blend)", dst, spc=polys)
            print("->", dst)
    if a.hazards:
        # same model and cycle weights (judged on the overall solution) applied to each hazard's per-cycle fields;
        # hazard calibrations from calibrate_hazards.py, each capped at the any-severe fitted on the same 14 cases
        import calibrate_hazards as ch
        hcal = json.load(open(os.path.join(CASES_DIR, "calibration_hazards.json")))
        sev14 = json.load(open(os.path.join(CASES_DIR, "calibration_anysevere_14case.json")))
        def wblend(scheme):
            acc = 0.0
            for m, r in rows.items():
                lay = [z[f"{L}__{m}__{k}"] for k in ("p_model", "p_lift", "sustain")]
                fs = np.stack([ci.finals(res[f"{scheme}|{L}|{m}|cyc{c}"], *lay)[1]["env*ci"] for c in r["cyc"]])
                acc = acc + (r["Wm"] / wsum) * np.tensordot(r["wc"], fs, 1)
            return acc
        p_sev = ch.calibrated(blend, sev14)
        raw = {"sev": blend}
        print()
        print(f"hazards (consistency-weighted)  any-severe on the 14-case fit: max {np.where(us, p_sev, 0).max():.1f}%")
        for hz, nm in (("tor", "Tornado"), ("hail", "Hail"), ("wind", "Wind")):
            raw[hz] = wblend(hz)
            p = np.where(us, np.minimum(ch.calibrated(raw[hz], hcal[hz]), p_sev), 0)
            i, j = np.unravel_index(np.argmax(p), p.shape)
            print(f"  {hz:5s} max {p.max():5.1f}% at {plots.LAT2[i, j]:.1f}N {plots.LON2[i, j]:.1f}  OKC {at(p, 35.5, -97.5):4.1f}  "
                  f"Wichita Falls {at(p, 34, -98.5):4.1f}  Ardmore {at(p, 34, -97):4.1f}  DFW {at(p, 33, -97):4.1f}  Paris {at(p, 33.5, -95.5):4.1f}")
            dst = os.path.join(out, f"hhwx_d{L}_{hz}_consistency{'_spc' if a.spc else ''}.png")
            if not a.no_draw:
                plot_hhwx.draw(p, f"HHwx Prob {nm} (Calibrated) Day-{L} (Consistency-Weighted Blend)", dst, spc=polys)
        # uncalibrated index fields (env x CI, 0-1): saved for side-by-side maps; never labelled as a probability
        np.savez_compressed(os.path.join(out, f"consistency_raw_d{L}{a.tag}.npz"), **{k: np.where(us, v, 0) for k, v in raw.items()},
                            **{f"w_{m}": np.float64(r["Wm"] / wsum) for m, r in rows.items()}, mode=np.array(a.mode))  # final model weights (block maps)


if __name__ == "__main__":
    main()

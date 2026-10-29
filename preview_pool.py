"""Off-schedule preview: score a convective day from the newest cycles, outside the SPC-day pools.

    python preview_pool.py 2026-11-02 --end 2026102906 --cycles 6 --gfs-extra 2026102912 [--compare-lead 5]

Downloads whatever the pool is missing (main + CI-extra fields), scores v2 + CI layers with the
same code as run_case/ci, applies the stored provisional calibration, and writes
cases/<date>/output/preview_<end>/ (npz + maps). The official results.npz / ci_layers.npz are not touched.
--gfs-extra adds newer GFS cycles that are out before the other models.
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import calibrate
import ci
import fetch
import plots
import run_case
import scoring
from config import CACHE_DIR, CASES_DIR, MODELS, MODEL_LABELS, VALID_OFFSETS, fh_available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--end", required=True, help="newest cycle in the pool, YYYYMMDDHH")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--gfs-extra", nargs="*", default=[], help="extra GFS-only cycles, YYYYMMDDHH")
    ap.add_argument("--compare-lead", type=int, default=5)
    ap.add_argument("--view", default="central")
    a = ap.parse_args()
    conv_day = datetime.strptime(a.date, "%Y-%m-%d")
    end = datetime.strptime(a.end, "%Y%m%d%H")
    inits = [end - timedelta(hours=6 * k) for k in range(a.cycles)][::-1]
    model_inits = {m: list(inits) for m in MODELS}
    model_inits["gfs"] += [datetime.strptime(x, "%Y%m%d%H") for x in a.gfs_extra]
    valid = [conv_day.replace(hour=12) + timedelta(hours=h) for h in VALID_OFFSETS]
    out = os.path.join(CASES_DIR, a.date, "output", f"preview_{a.end}")
    os.makedirs(out, exist_ok=True)

    # 1. download what is missing
    tasks = [(m + sfx, i, int((v - i).total_seconds() // 3600)) for m in MODELS for sfx in ("", "_x")
             for i in model_inits[m] for v in valid if fh_available(m, i, int((v - i).total_seconds() // 3600))]
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetch.init_decode_pool(8)
    secs = fetch.download_all(tasks, 24, 8, log=lambda s: print(s, flush=True))
    print(f"download {secs / 60:.1f} min, {fetch.STATS['bytes'] / 1e9:.2f} GB", flush=True)

    # 2. environment (v2) per model and blend
    scoring.set_profile("v2")
    env = {}
    for m in MODELS:
        o, _ = run_case.score_pool(m, model_inits[m], valid)
        env[m] = run_case.combine(o["v2"]["day"], "v2", m)
        print(f"  {m}: {len(o['v2']['day'])} member-runs", flush=True)
    env_blend = scoring.blend(env)

    # 3. CI layers (same as ci.layers_for_lead, with per-model cycle lists)
    acc = {m: [[], [], []] for m in MODELS}
    for m in MODELS:
        for i in model_inits[m]:
            for mem in run_case.members_of(m):
                r = ci.member_layers(m, i, mem, valid)
                if r is not None:
                    for k in range(3):
                        acc[m][k].append(r[k])
    lay = [scoring.blend({m: (np.mean(acc[m][k], axis=0) if acc[m][k] else None) for m in MODELS}) for k in range(3)]
    p_ci, fin = ci.finals(env_blend, *lay)
    final = fin["env*ci"]

    # 4. provisional calibration table
    cal = json.load(open(os.path.join(CASES_DIR, "calibration_provisional.json")))["env_ci"]
    to_pct = lambda f: 100 * np.interp(f, cal["index"], cal["prob"])
    prev = to_pct(final)
    off = np.load(os.path.join(CASES_DIR, a.date, "output", "ci_layers.npz"))
    official = to_pct(off[f"{a.compare_lead}__env*ci"])
    np.savez_compressed(os.path.join(out, "preview.npz"), env=env_blend, p_ci=p_ci, final=final, prob=prev,
                        **{f"env_{m}": f for m, f in env.items() if f is not None})

    # 5. maps: official vs preview (calibrated), then per-model environment
    def panel(ax, field, title, levels, cmap, norm, label_fmt):
        plots.base_ax(ax, a.view)
        cf = ax.contourf(plots.LON2, plots.LAT2, field, levels=levels, cmap=cmap, norm=norm, transform=plots.PC)
        cl = ax.contour(plots.LON2, plots.LAT2, field, levels=levels[:-1], colors="#1f3b8a", linewidths=0.6,
                        transform=plots.PC)
        ax.clabel(cl, fontsize=6, fmt=label_fmt)
        ax.set_title(title, fontsize=10)
        return cf

    lv, cm = calibrate.SPC_LEVELS, calibrate.SPC_CMAP
    nm = BoundaryNorm(lv, cm.N)
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), subplot_kw={"projection": plots.PROJ})
    axs = axs.ravel()
    cf = panel(axs[0], official, f"OFFICIAL Day {a.compare_lead} (cycles to "
               f"{datetime.strptime(a.date, '%Y-%m-%d') - timedelta(days=a.compare_lead - 1):%m/%d} 00Z)  max {official.max():.0f}%",
               lv, cm, nm, lambda v: f"{v:g}%")
    panel(axs[1], prev, f"PREVIEW (cycles {inits[0]:%m/%d %HZ}-{inits[-1]:%m/%d %HZ}"
          f"{' + GFS ' + ','.join(x[-2:] + 'Z' for x in a.gfs_extra) if a.gfs_extra else ''})  max {prev.max():.0f}%",
          lv, cm, nm, lambda v: f"{v:g}%")
    for ax, m in zip(axs[2:], MODELS):
        if env[m] is not None:
            panel(ax, 100 * env[m], f"{MODEL_LABELS[m]} environment index (raw, not %)", plots.LEVELS, plots.CMAP,
                  BoundaryNorm(plots.LEVELS, plots.CMAP.N), lambda v: f"{v:g}")
    fig.colorbar(cf, ax=axs[:2], orientation="horizontal", fraction=0.04, pad=0.04,
                 ticks=lv[:-1]).set_label("calibrated P(severe within 25 mi) [%], PROVISIONAL (biased high)")
    fig.suptitle(f"{a.date}: off-schedule preview with the newest cycles vs the official Day {a.compare_lead}", fontsize=13)
    p = os.path.join(out, f"preview_{a.view}.png")
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()

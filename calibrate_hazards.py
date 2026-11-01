"""EXPERIMENTAL per-hazard calibration: tornado, hail and wind probabilities within 25 mi.

    python calibrate_hazards.py --train 2025-03-14 2025-03-15 ... [--apply 2026-11-02 --lead 4]

For each hazard, the product is that hazard's environment score (scoring.hazard_factors x v2, saved by
run_case.py as tor|L|blend|day etc.) times the day's CI layer (ci_layers.npz), exactly as the
any-severe product. It is fitted against reports of that type only (logistic on logit(index),
area-weighted, every lead of every training case), the same method as calibrate.py.

Base-rate correction: training cases are hand-picked active days, so each hazard's calibrated odds
are shifted by its own climatology / training ratio (2025 per-hazard area fractions from base_rate.py),
as bust_summary.SHIFT does for any-severe.
Writes cases/calibration_hazards.json; prints leave-one-case-out BSS and reliability per hazard.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import ci
import verify_case as vc
from calibrate import Calib
from config import CASES_DIR

REPORT_TYPE = {"tor": "tornado", "hail": "hail", "wind": "wind", "tor2": "tornado", "hail2": "hail", "wind2": "wind"}


def product(case, hz, lead):
    out = os.path.join(CASES_DIR, case, "output")
    res, z = np.load(os.path.join(out, "results.npz")), np.load(os.path.join(out, "ci_layers.npz"))
    k = f"{hz}|{lead}|blend|day"
    if k not in res.files or f"{lead}__p_model" not in z.files:
        return None
    return ci.finals(res[k], z[f"{lead}__p_model"], z[f"{lead}__p_lift"], z[f"{lead}__sustain"])[1]["env*ci"]


def leads(case):
    z = np.load(os.path.join(CASES_DIR, case, "output", "ci_layers.npz"))
    return sorted({int(k.split("__")[0]) for k in z.files if k.endswith("__p_model") and k.count("__") == 1})


def reports(case, hz):
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{case.replace('-', '')}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    return rep if hz is None else rep[rep.type == REPORT_TYPE[hz]]


def pairs(case, hz):
    lab = vc.report_mask(reports(case, hz))
    xs, ys, ws = [], [], []
    for L in leads(case):
        f = product(case, hz, L)
        if f is not None:
            xs.append(f[vc.DOM]); ys.append(lab[vc.DOM]); ws.append(vc.W[vc.DOM])
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys).astype(float), np.concatenate(ws)


def fit(data):
    x, y, w = (np.concatenate(a) for a in zip(*data))
    return Calib(LogisticRegression(C=1.0).fit(Calib.feat(x), y, sample_weight=w))


def loco(data):
    """Leave-one-case-out Brier skill and reliability table."""
    rows = []
    for held in data:
        cal = fit([d for c, d in data.items() if c != held])
        x, y, w = data[held]
        rows.append(pd.DataFrame(dict(p=cal.predict(x), y=y, w=w)))
    d = pd.concat(rows)
    d["bin"] = pd.cut(100 * d.p, [0, 2, 5, 10, 15, 30, 45, 101], right=False)
    t = d.groupby("bin", observed=True).apply(
        lambda g: pd.Series(dict(forecast=100 * (g.p * g.w).sum() / g.w.sum(),
                                 observed=100 * (g.y * g.w).sum() / g.w.sum(), area=g.w.sum())),
        include_groups=False)
    t["area"] = 100 * t.area / t.area.sum()
    clim = (d.y * d.w).sum() / d.w.sum()
    bs = (d.w * (d.p - d.y) ** 2).sum() / d.w.sum()
    return t, 1 - bs / ((d.w * (clim - d.y) ** 2).sum() / d.w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", default="calibration_hazards.json", help="file name in cases/")
    a = ap.parse_args()
    clim = pd.read_csv(os.path.join(CASES_DIR, "base_rate_2025-01-01_2025-12-31.csv"))
    store = dict(trained_on=a.train, note="EXPERIMENTAL per-hazard logistic calibration (calibrate_hazards.py)")
    for hz in [h for h in REPORT_TYPE if f"{h}|1|blend|day" in np.load(os.path.join(CASES_DIR, a.train[0], "output", "results.npz")).files]:
        data = {c: d for c in a.train if (d := pairs(c, hz)) is not None}
        t, bss = loco(data)
        print(f"\n[{hz}] {len(data)} cases, leave-one-case-out BSS {bss:.3f}")
        print(t.round(2).to_string())
        cal = fit(list(data.values()))
        xs = np.linspace(0, 1, 101)
        # base-rate (prior) correction: climatological hazard area fraction vs the training cases'
        p_clim = clim[f"frac_{REPORT_TYPE[hz]}"].mean()
        p_train = np.mean([(vc.W * vc.report_mask(reports(c, hz)))[vc.DOM].sum() / vc.W[vc.DOM].sum() for c in data])
        shift = np.log(p_clim / (1 - p_clim)) - np.log(p_train / (1 - p_train))
        store[hz] = dict(index=xs.round(2).tolist(), prob=cal.predict(xs).round(5).tolist(),
                         p_clim=round(float(p_clim), 5), p_train=round(float(p_train), 5), shift=round(float(shift), 3))
        print(f"  base rate: climatology {100 * p_clim:.3f}%  training {100 * p_train:.3f}%  logit shift {shift:+.2f}")
    # any-severe on the same cases (same method), so each hazard can be capped consistently
    import calibrate
    cal = calibrate.fit(a.train, "env_ci")
    fr = [(vc.W * vc.report_mask(reports(c, None)))[vc.DOM].sum() / vc.W[vc.DOM].sum() for c in a.train]
    p_clim, p_train = clim["frac"].mean(), float(np.mean(fr))
    shift = np.log(p_clim / (1 - p_clim)) - np.log(p_train / (1 - p_train))
    store["any"] = dict(index=xs.round(2).tolist(), prob=cal.predict(xs).round(5).tolist(),
                        p_clim=round(float(p_clim), 5), p_train=round(p_train, 5), shift=round(float(shift), 3))
    with open(os.path.join(CASES_DIR, a.out), "w") as f:
        json.dump(store, f, indent=1)
    print(f"-> cases/{a.out}")


def calibrated(field, cal):
    """Calibrated, base-rate-corrected probability (%) for one hazard's env x CI field."""
    p = np.clip(np.interp(field, cal["index"], cal["prob"]), 1e-6, 1 - 1e-6)
    return 100 / (1 + np.exp(-(np.log(p / (1 - p)) + cal["shift"])))


if __name__ == "__main__":
    main()

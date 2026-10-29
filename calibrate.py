"""Provisional calibration: raw index -> P(severe report within 25 mi / 40 km), SPC's definition.

    python calibrate.py --train 2025-05-16 2025-06-16 2025-04-27 2026-04-13 --apply 2026-11-02

Logistic (Platt) calibration on logit(index), area-weighted, over every grid point / lead of
the verified cases. It beat isotonic regression in leave-one-case-out tests on 3 cases
(BSS 0.115 vs 0.080 for env x CI; isotonic over-fitted the top bins). Leave-one-case-out (LOCO) reliability is printed so over/under-forecasting is visible.
CAVEAT: training cases are all active severe days, so the map is biased high until quiet
and bust days are added. The raw index must never be labelled "%"; only this output is a
probability.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.linear_model import LogisticRegression

import plots
import verify_case as vc
from config import CASES_DIR

PRODUCTS = {"env": ("results.npz", "v2|{L}|blend|day"), "env_ci": ("ci_layers.npz", "{L}__env*ci")}
SPC_LEVELS = [2, 5, 10, 15, 30, 45, 60, 100]
TRAIN_NOTE = "active severe days only, likely biased high"
SPC_CMAP = ListedColormap(["#fff3b0", "#fdd17a", "#f9a24d", "#ef6a32", "#d7301f", "#a50f15", "#5c0a0f"])


def load_field(case, prod, lead):
    fname, key = PRODUCTS[prod]
    p = os.path.join(CASES_DIR, case, "output", fname)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    k = key.format(L=lead)
    return z[k] if k in z.files else None


def leads_of(case, prod):
    fname, key = PRODUCTS[prod]
    z = np.load(os.path.join(CASES_DIR, case, "output", fname))
    return sorted({int(k.split("|")[1]) if "|" in k else int(k.split("__")[0]) for k in z.files
                   if (k.startswith("v2|") and k.endswith("|blend|day")) or k.endswith("__env*ci")})


def pairs(case, prod):
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    lab = vc.report_mask(rep)
    xs, ys, ws = [], [], []
    for L in leads_of(case, prod):
        f = load_field(case, prod, L)
        if f is not None:
            xs.append(f[vc.DOM]); ys.append(lab[vc.DOM]); ws.append(vc.W[vc.DOM])
    return np.concatenate(xs), np.concatenate(ys).astype(float), np.concatenate(ws)


class Calib:
    """Monotone logistic map on logit(index)."""
    def __init__(self, lr):
        self.lr = lr

    @staticmethod
    def feat(x):
        x = np.clip(np.asarray(x, float), 1e-3, 1 - 1e-3)
        return np.log(x / (1 - x)).reshape(-1, 1)

    def predict(self, x):
        return self.lr.predict_proba(self.feat(x))[:, 1]


def fit(cases, prod):
    x, y, w = (np.concatenate(a) for a in zip(*[pairs(c, prod) for c in cases]))
    return Calib(LogisticRegression(C=1.0).fit(Calib.feat(x), y, sample_weight=w))


def reliability(cases, prod):
    rows = []
    for held in cases:
        iso = fit([c for c in cases if c != held], prod)
        x, y, w = pairs(held, prod)
        p = iso.predict(x)
        rows.append(pd.DataFrame(dict(p=p, y=y, w=w, case=held)))
    d = pd.concat(rows)
    d["bin"] = pd.cut(100 * d.p, [0, 2, 5, 10, 15, 30, 45, 60, 101], right=False)
    t = d.groupby("bin", observed=True).apply(
        lambda g: pd.Series(dict(forecast=100 * (g.p * g.w).sum() / g.w.sum(),
                                 observed=100 * (g.y * g.w).sum() / g.w.sum(), area_share=g.w.sum())),
        include_groups=False)
    t["area_share"] = 100 * t.area_share / t.area_share.sum()
    clim = (d.y * d.w).sum() / d.w.sum()
    bs = (d.w * (d.p - d.y) ** 2).sum() / d.w.sum()
    bs_ref = (d.w * (clim - d.y) ** 2).sum() / d.w.sum()
    return t, 1 - bs / bs_ref


def plot_calibrated(case, prod, iso, view="conus", ntrain=3):
    conv_day = datetime.strptime(case, "%Y-%m-%d")
    rep = plots.load_reports(case, conv_day)
    leads = leads_of(case, prod)[::-1]
    n = len(leads)
    ncol = 4 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow + 1.2), subplot_kw={"projection": plots.PROJ})
    axes = np.atleast_1d(axes).ravel()
    cf = None
    for ax, L in zip(axes, leads):
        plots.base_ax(ax, view)
        p = 100 * iso.predict(load_field(case, prod, L).ravel()).reshape(71, 141)
        cf = ax.contourf(plots.LON2, plots.LAT2, p, levels=SPC_LEVELS, cmap=SPC_CMAP,
                         norm=BoundaryNorm(SPC_LEVELS, SPC_CMAP.N), transform=plots.PC)
        cl = ax.contour(plots.LON2, plots.LAT2, p, levels=SPC_LEVELS[:-1], colors="#1f3b8a", linewidths=0.6, transform=plots.PC)
        ax.clabel(cl, fontsize=6, fmt=lambda v: f"{v:g}%")
        polys, iscat = plots.spc_lead_polys(case, conv_day, L)
        plots.draw_spc(ax, polys, iscat)
        plots.spc_note(ax, polys, L, iscat)
        plots.draw_reports(ax, rep, 4)
        ax.set_title(f"Day {L}  (max {p.max():.0f}%)", fontsize=11)
    for ax in axes[n:]:
        ax.set_visible(False)
    name = {"env": "environment", "env_ci": "environment x initiation"}[prod]
    fig.suptitle(f"{case}: calibrated probability of severe within 25 mi ({name})\n"
                 f"PROVISIONAL: logistic fit on {ntrain} case days ({TRAIN_NOTE})", fontsize=12)
    if cf is not None:
        cax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=SPC_LEVELS[:-1]).set_label(
            "P(severe report within 25 mi) [%]: SPC definition", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.03, hspace=0.12)
    out = os.path.join(CASES_DIR, case, "output", f"calibrated_{prod}_{view}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--note", default="active severe days only, likely biased high",
                    help="what the training set is, shown on the maps")
    a = ap.parse_args()
    global TRAIN_NOTE
    TRAIN_NOTE = a.note
    store = {}
    for prod in PRODUCTS:
        t, bss = reliability(a.train, prod)
        print(f"\n[{prod}] leave-one-case-out reliability (calibrated forecast vs observed), BSS {bss:.3f}")
        print(t.round(1).to_string())
        iso = fit(a.train, prod)
        xs = np.linspace(0, 1, 101)
        store[prod] = dict(index=xs.round(2).tolist(), prob=iso.predict(xs).round(4).tolist())
        for c in a.apply + a.train:
            print("  ", plot_calibrated(c, prod, iso, ntrain=len(a.train)))
    with open(os.path.join(CASES_DIR, "calibration_provisional.json"), "w") as f:
        json.dump(dict(trained_on=a.train, note=f"logistic on logit(index), area-weighted; {a.note}", **store), f, indent=1)
    for prod in PRODUCTS:
        m = dict(zip(store[prod]["index"], store[prod]["prob"]))
        print(f"{prod}: index 10/25/50/75 -> " + ", ".join(f"{100 * m[k]:.0f}%" for k in (0.1, 0.25, 0.5, 0.75)))


if __name__ == "__main__":
    main()

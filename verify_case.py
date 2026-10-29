"""Verify a scored case against SPC storm reports and SPC outlooks.

    python verify_case.py 2025-06-16

Metrics per lead day, scheme and model (convective-day field):
  * AUC   -- ROC area of the index for "report within 40 km" grid points (CONUS box).
             0.5 = no skill, 1.0 = perfect ranking. Insensitive to calibration.
  * top5  -- share of all / tornado reports inside the highest-index 5% of the CONUS box.
  * spcA  -- area-matched capture: our top-N area where N = the SPC lead polygon's area,
             next to the SPC polygon's own capture (Day 8-3 any-severe 15% only).
  * dmax  -- distance (km) from our maximum to the report-density centroid.
Writes cases/<date>/output/verification.csv and verification.md.
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

from config import CASES_DIR, GRID_LATS, GRID_LONS, MODELS

LA, LO = np.meshgrid(GRID_LATS, GRID_LONS, indexing="ij")
DOM = (LA >= 25) & (LA <= 49) & (LO >= -125) & (LO <= -67)
W = np.cos(np.deg2rad(LA))  # area weight


def hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def report_mask(rep, radius_km=40.0):
    m = np.zeros(LA.shape, bool)
    for la, lo in zip(rep.lat, rep.lon):
        near = (np.abs(LA - la) < 1.0) & (np.abs(LO - lo) < 1.3)
        m[near] |= hav_km(la, lo, LA[near], LO[near]) <= radius_km
    return m & DOM


def auc(score, label, weight):
    """Area-weighted ROC AUC via the rank-sum identity."""
    s, y, w = score[DOM], label[DOM], weight[DOM]
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    wp, wn = (w * y).sum(), (w * ~y).sum()
    if wp == 0 or wn == 0:
        return np.nan
    cum_neg = np.cumsum(w * ~y) - (w * ~y)  # negative weight strictly below each point
    # ties: give half credit
    _, first = np.unique(s, return_index=True)
    tie_neg = np.zeros_like(s)
    groups = np.split(np.arange(len(s)), first[1:])
    for g in groups:
        tie_neg[g] = (w[g] * ~y[g]).sum()
        cum_neg[g] = cum_neg[g[0]]
    return float(((cum_neg + 0.5 * tie_neg) * w * y).sum() / (wp * wn))


def top_area_mask(field, frac):
    vals, wts = field[DOM], W[DOM]
    order = np.argsort(-vals)
    cw = np.cumsum(wts[order]) / wts.sum()
    thr = vals[order][min(np.searchsorted(cw, frac), len(vals) - 1)]
    return (field >= thr) & DOM


def spc_polygon(case, conv_day, lead):
    if lead < 3:
        return None
    issue = conv_day - timedelta(days=lead - 1)
    name = (f"day{lead}_issued_{issue:%Y%m%d}.geojson" if lead >= 4
            else f"day3_0730z_issued_{issue:%Y%m%d}_prob.geojson")
    p = os.path.join(CASES_DIR, case, "spc", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        feats = json.load(f).get("features", [])
    geoms = [shape(ft["geometry"]) for ft in feats if str(ft["properties"].get("level")) == "0.15"]
    return geoms


def in_geoms(geoms, lats, lons):
    return np.array([any(g.contains(Point(lo, la)) for g in geoms) for la, lo in zip(lats, lons)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    a = ap.parse_args()
    case, conv_day = a.date, datetime.strptime(a.date, "%Y-%m-%d")
    out = os.path.join(CASES_DIR, case, "output")
    res = dict(np.load(os.path.join(out, "results.npz")))
    rep = pd.read_csv(os.path.join(CASES_DIR, case, "spc", f"reports_{conv_day:%Y%m%d}.csv"))
    rep = rep[(rep.lat > 24) & (rep.lat < 50) & (rep.lon > -126) & (rep.lon < -66)]
    tor = rep.type.str.startswith("torn").values
    ri = np.rint((rep.lat.values - GRID_LATS[0]) / 0.5).astype(int)
    rj = np.rint((rep.lon.values - GRID_LONS[0]) / 0.5).astype(int)
    label = report_mask(rep)
    cen_lat, cen_lon = rep.lat.mean(), rep.lon.mean()

    leads = sorted({int(k.split("|")[1]) for k in res if k.split("|")[0] in ("current", "v2")}, reverse=True)
    rows = []
    for lead in leads:
        geoms = spc_polygon(case, conv_day, lead)
        spc = {}
        if geoms:
            grid_in = in_geoms(geoms, LA[DOM], LO[DOM])
            frac = float((W[DOM] * grid_in).sum() / W[DOM].sum())
            rin = in_geoms(geoms, rep.lat.values, rep.lon.values)
            spc = dict(spc_frac=frac, spc_all=rin.mean(), spc_tor=rin[tor].mean() if tor.any() else np.nan)
        for scheme in ("current", "v2"):
            for m in MODELS + ["blend"]:
                f = res.get(f"{scheme}|{lead}|{m}|day")
                if f is None:
                    continue
                t5 = top_area_mask(f, 0.05)[ri, rj]
                jmax = np.unravel_index(np.argmax(np.where(DOM, f, -1)), f.shape)
                row = dict(lead=lead, scheme=scheme, model=m, auc=auc(f, label, W),
                           top5_all=t5.mean(), top5_tor=t5[tor].mean() if tor.any() else np.nan,
                           dmax_km=hav_km(GRID_LATS[jmax[0]], GRID_LONS[jmax[1]], cen_lat, cen_lon),
                           max_lat=GRID_LATS[jmax[0]], max_lon=GRID_LONS[jmax[1]])
                if spc:
                    hit = top_area_mask(f, spc["spc_frac"])[ri, rj]
                    row.update(spcA_all=hit.mean(), spcA_tor=hit[tor].mean() if tor.any() else np.nan, **spc)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "verification.csv"), index=False)

    lines = [f"# Verification: {case}", "",
             f"Reports in CONUS box: {len(rep)} ({tor.sum()} tornado). Report centroid {cen_lat:.1f}N {cen_lon:.1f}.",
             "AUC uses 'report within 40 km' grid points. top5 = share of reports inside our top-5% area. "
             "spcA = share inside our top-N area where N = SPC 15% polygon area (Day 8-3), next to SPC's own share.", ""]
    blend = df[df.model == "blend"]
    lines += ["## Blend by lead day", "",
              "| Lead | Scheme | AUC | top5 all | top5 tor | max at | km from report centroid | SPC 15% area | ours same area (all/tor) | SPC (all/tor) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in blend.iterrows():
        spc_cols = (f"{100 * r.spc_frac:.1f}% | {100 * r.spcA_all:.0f}% / {100 * r.spcA_tor:.0f}% | "
                    f"{100 * r.spc_all:.0f}% / {100 * r.spc_tor:.0f}%") if "spc_frac" in r and pd.notna(r.get("spc_frac")) else "none | — | —"
        lines.append(f"| D{r.lead} | {r.scheme} | {r.auc:.3f} | {100 * r.top5_all:.0f}% | {100 * r.top5_tor:.0f}% | "
                     f"{r.max_lat:.1f}N {r.max_lon:.1f} | {r.dmax_km:.0f} | {spc_cols} |")
    lines += ["", "## AUC by model (v2 / current)", "", "| Lead | " + " | ".join(MODELS + ["blend"]) + " |",
              "|---" * (len(MODELS) + 2) + "|"]
    for lead in leads:
        cells = []
        for m in MODELS + ["blend"]:
            v = df[(df.lead == lead) & (df.model == m)].set_index("scheme").auc
            cells.append(f"{v.get('v2', np.nan):.2f} / {v.get('current', np.nan):.2f}")
        lines.append(f"| D{lead} | " + " | ".join(cells) + " |")
    with open(os.path.join(out, "verification.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

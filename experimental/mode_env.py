"""EXPERIMENTAL: storm mode from the ENVIRONMENT (Hunter, 1 Nov 2026: mode from upper air, shear vectors and soundings,
not from model convection). Cloud step; reads the cached decoded fields. Display only until verified against
observed mode (research/20_boundary_orientation/01_boundary_relative_shear_and_mode.md section 4.0 point 6).

    python experimental/mode_env.py 2026-11-02 --leads 2

Follows the pre-registered designs in research/20_boundary_orientation/01 section 4 (thresholds from the literature,
fixed beforehand, not tuned):
  BO0 (minimum viable): moisture boundary per 0.5 deg row 28-42N, 105-92W: smoothed 2 m q (sigma 0.75 cell), the
      longitude of max |grad q| where >= 2 g/kg per 100 km and q rises eastward; rows linked if |dlon| <= 1 deg,
      segments >= 6 rows. Normal n points to the moist side (least-squares axis over +-2 rows). Boundary speed c_n
      from the line position at t-3 h and t+3 h (invalid if |c_n| > 20 m/s or missing). Band weight G_w: 1 on
      0-100 km warm side, tapering to 0 at 150 km, and to 0 at 50 km on the dry side. Thermal fronts are not detected
      yet (known gap: after a front overtakes a dryline the moisture line can lag).
  BO1 Dial's primary control, cloud-layer mean wind normal to the boundary, boundary-relative:
      Ur = Vcl.n - c_n, D1 = ramp(Ur, 3, 9) (Dial split 6 m/s); without c_n: ground-relative ramp(Vcl.n, 9, 15).
  BO2b normal deep shear (pressure-level arm): D2 = 0.5 ramp(|S5.n|, 9, 16) + 0.5 ramp(|(V300 - V700).n|, 5, 11).
  D = 0.6 D1 + 0.4 D2 (F-0351 weights). Vcl = mean(V700, V500, V300); fallback mean(V850, V500) where a model has
  no 700/300 hPa winds (RRFS, REFS), flagged in the output.
Organised (supercell-capable) share O = ramp(|V500 - V10|, 10, 18) m/s (0-5 km shear, the 0-6 km 12-20 m/s range
depth-scaled). Classes per cell and time:
  discrete supercell O.max(0, 2D - 1); semi-discrete O.(1 - |2D - 1|); linear / QLCS O.max(0, 1 - 2D); disorganized 1 - O.
Day value = time mean over 18, 21, 00, 03, 06Z weighted by the model's own storm-start probability for the block
(ci_layers <L>__<m>__<block>__p_model), then members and the lead's pooled runs averaged (same pool rules as the
live blend). Also saved: line-normal 0-3 km shear proxy |(V700 - V10).n| (QLCS tornado potential, BO6 idea) and
the 00Z line points of each model's newest run.
Writes <POGS_MODE_OUT or output>/mode_env_<date>.npz.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch  # noqa: E402
import plot_hhwx  # noqa: E402
import plots  # noqa: E402
import scoring  # noqa: E402
from config import CASES_DIR, EPS_MEMBERS, GEFS_MEMBERS, POOL_CYCLES, REFS_MEMBERS, UA_SET, fh_available  # noqa: E402

ramp = scoring.ramp
MODELS = ("gfs", "gefs", "hres", "eps", "nam12", "rrfs", "nam", "hrrr", "rap", "refs")
REGIONAL = {"nam12", "rrfs", "nam", "hrrr", "rap", "refs"}
CAM_ADD = {"hrrr": "hrrr_add", "nam": "nam_add", "rap": "rap_add", "nam12": "nam12_add"}
HOURS = (18, 21, 24, 27, 30)
BLOCK_OF = {18: ("b21",), 21: ("b21", "b00"), 24: ("b00", "b03"), 27: ("b03", "b06"), 30: ("b06",)}
LAT, LON = plots.LAT2, plots.LON2
DY = 55.5  # km per 0.5 deg of latitude


def members(m):
    return {"gefs": GEFS_MEMBERS, "eps": EPS_MEMBERS, "refs": REFS_MEMBERS}.get(m, ["det"])


def load_fields(m, init, fh, mem):
    """dict with u10 v10 u850 v850 u500 v500 q2 (g/kg) and, where cached, u700 v700 u300 v300; or None."""
    f = fetch.read_cached(m, init, fh, mem)
    if f is None:
        return None
    x = fetch.read_cached(f"{m}_x", init, fh, mem) if m in ("gfs", "gefs", "hres", "eps") else None
    if x is not None:
        f.update({k: x[k] for k in ("u850", "v850") if k in x})
    if not all(k in f for k in ("u10", "v10", "u850", "v850", "u500", "v500")):
        return None
    q = 1000.0 * f["q2"] if "q2" in f else (scoring.q_from_td(f["d2m"], f["sp"]) if "d2m" in f and "sp" in f else None)
    if q is None:
        return None
    out = {k: f[k].astype(float) for k in ("u10", "v10", "u850", "v850", "u500", "v500")}
    out["q2"] = np.asarray(q, float)
    shp = out["u10"].shape
    if m in UA_SET:
        ua = fetch.read_cached(UA_SET[m], init, fh, mem if m in ("gefs", "eps") else "det")
        if ua is not None:
            for k in ("u700", "v700", "u300", "v300"):
                if k in ua:
                    a = ua[k] if ua[k].shape == shp else ua[k][::2, ::2]
                    if a.shape == shp:
                        out[k] = a.astype(float)
    elif m in CAM_ADD:
        add = fetch.read_cached(CAM_ADD[m], init, fh, "det")
        if add is not None:
            pairs = {f"{c}{p}": add.get(f"{c}grd_{p}_mb") for c in "uv" for p in (700, 300)}
            if all(v is not None for v in pairs.values()):
                rot = fetch.earth_relative(m, dict(pairs))  # add-on CAM winds are cached grid-relative
                out.update({k: v.astype(float) for k, v in rot.items()})
    return out


def find_line(q):
    """Moisture boundary (BO0 minimum viable): {row: col} of line points, linked segments >= 6 rows."""
    qs = ndimage.gaussian_filter(np.nan_to_num(q, nan=0.0), 0.75)
    gy, gx = np.gradient(qs)
    dx = DY * np.cos(np.deg2rad(LAT))
    G = 100.0 * np.hypot(gy / DY, gx / dx)  # g/kg per 100 km
    rows = [i for i in range(LAT.shape[0]) if 28 <= LAT[i, 0] <= 42]
    cols = np.where((LON[0] >= -105) & (LON[0] <= -92))[0]
    pts = {}
    for i in rows:
        g = np.where(gx[i, cols] > 0, G[i, cols], 0)
        j = int(np.argmax(g))
        if g[j] >= 2.0:
            pts[i] = int(cols[j])
    keep, seg = {}, []
    for i in sorted(pts):
        if seg and (i != seg[-1] + 1 or abs(pts[i] - pts[seg[-1]]) > 2):
            if len(seg) >= 6:
                keep.update({r: pts[r] for r in seg})
            seg = []
        seg.append(i)
    if len(seg) >= 6:
        keep.update({r: pts[r] for r in seg})
    return keep, G


def normals(line):
    """Unit normal (east, north) per line row, pointing east-ish (moist side), from a +-2 row least-squares axis."""
    out = {}
    rows = sorted(line)
    for i in rows:
        nb = [r for r in rows if abs(r - i) <= 2]
        if len(nb) < 3:
            continue
        y = np.array([LAT[r, 0] for r in nb]) * DY / 0.5
        x = np.array([LON[0, line[r]] for r in nb]) * DY / 0.5 * np.cos(np.deg2rad(LAT[i, 0]))
        slope = np.polyfit(y, x, 1)[0]  # dx/dy along the line
        t = np.array([slope, 1.0]) / np.hypot(slope, 1.0)  # (east, north) tangent
        n = np.array([t[1], -t[0]])
        out[i] = n if n[0] >= 0 else -n
    return out


def mode_at(fl, line, nrm, cn):
    """Class shares (4, ny, nx) and diagnostics for one model-member-time; None without a usable line."""
    if not nrm:
        return None
    ny, nx = LAT.shape
    mask = np.zeros((ny, nx), bool)
    for i in nrm:
        mask[i, line[i]] = True
    _, (ii, jj) = ndimage.distance_transform_edt(~mask, return_indices=True)
    nE = np.zeros((ny, nx)); nN = np.zeros((ny, nx)); c = np.full((ny, nx), np.nan)
    for i, n in nrm.items():
        sel = (ii == i) & (jj == line[i])
        nE[sel], nN[sel] = n
        c[sel] = cn.get(i, np.nan)
    dxkm = (LON - LON[ii, jj]) * DY / 0.5 * np.cos(np.deg2rad(LAT))
    dykm = (LAT - LAT[ii, jj]) * DY / 0.5
    d = dxkm * nE + dykm * nN  # signed distance, + on the moist side
    Gw = np.where((d >= 0) & (d <= 100), 1.0, 0.0)
    Gw = np.where((d > 100) & (d <= 150), 1 - (d - 100) / 50, Gw)
    Gw = np.where((d >= -50) & (d < 0), 1 + d / 50, Gw)
    far = np.hypot(dxkm, dykm) > 300
    has_cl = "u700" in fl and "u300" in fl
    if has_cl:
        ucl, vcl = (fl["u700"] + fl["u500"] + fl["u300"]) / 3, (fl["v700"] + fl["v500"] + fl["v300"]) / 3
    else:
        ucl, vcl = (fl["u850"] + fl["u500"]) / 2, (fl["v850"] + fl["v500"]) / 2
    vn = ucl * nE + vcl * nN
    D1 = np.where(np.isfinite(c), ramp(vn - np.nan_to_num(c), 3, 9), ramp(vn, 9, 15))
    s5u, s5v = fl["u500"] - fl["u10"], fl["v500"] - fl["v10"]
    if has_cl:
        D2 = 0.5 * ramp(np.abs(s5u * nE + s5v * nN), 9, 16) + 0.5 * ramp(
            np.abs((fl["u300"] - fl["u700"]) * nE + (fl["v300"] - fl["v700"]) * nN), 5, 11)
    else:
        D2 = ramp(np.abs(s5u * nE + s5v * nN), 9, 16)
    D = np.where(far, 0.5, 0.6 * D1 + 0.4 * D2)
    O = ramp(np.hypot(s5u, s5v), 10, 18)
    disc, lin = O * np.maximum(0, 2 * D - 1), O * np.maximum(0, 1 - 2 * D)
    shares = np.stack([disc, O - disc - lin, lin, 1 - O])
    s3 = fl.get("u700", fl["u850"]) - fl["u10"], fl.get("v700", fl["v850"]) - fl["v10"]
    qlcs_n = np.abs(s3[0] * nE + s3[1] * nN)
    return shares, dict(D=D, Gw=Gw, qlcs_n=qlcs_n, has_cl=has_cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[2])
    a = ap.parse_args()
    O = os.path.join(CASES_DIR, a.date, "output")
    out_dir = os.environ.get("POGS_MODE_OUT", O)
    os.makedirs(out_dir, exist_ok=True)
    res = np.load(os.path.join(O, "results.npz"))
    z = np.load(os.path.join(O, "ci_layers.npz"))
    conv = datetime.strptime(a.date, "%Y-%m-%d")
    us = plot_hhwx.us_mask()
    save = {}
    for L in a.leads:
        for m in MODELS:
            if f"{L}__{m}__p_model" not in z.files:
                continue
            cyc = sorted(k[-10:] for k in res.files if k.startswith(f"v2|{L}|{m}|cyc"))
            if m not in REGIONAL:
                cyc = cyc[-POOL_CYCLES:]
            if m == "rap" and L <= 2:
                cyc = [c for c in cyc if c >= f"{conv - timedelta(days=1) + timedelta(hours=3):%Y%m%d%H}"]
            wblk = {b: z[f"{L}__{m}__{b}__p_model"] for b in ("b21", "b00", "b03", "b06") if f"{L}__{m}__{b}__p_model" in z.files}
            acc, acc_w, accD, accQ, n_used, n_cl = 0.0, 0.0, 0.0, 0.0, 0, 0
            for c in cyc:
                init = datetime.strptime(c, "%Y%m%d%H")
                for mem in members(m):
                    seq = {}
                    for h in (15,) + HOURS + (33,):  # t-3 h / t+3 h neighbours for the boundary speed
                        t = conv + timedelta(hours=h)
                        fh = int((t - init).total_seconds() // 3600)
                        if fh_available(m, init, fh):
                            fl = load_fields(m, init, fh, mem)
                            if fl is not None:
                                seq[h] = (fl, *find_line(fl["q2"]))
                    for h in HOURS:
                        if h not in seq:
                            continue
                        fl, line, _ = seq[h]
                        nrm = normals(line)
                        cn = {}
                        if (h - 3) in seq and (h + 3) in seq:
                            lp, ln_ = seq[h - 3][1], seq[h + 3][1]
                            for i, n in nrm.items():
                                if i in lp and i in ln_:
                                    dx = (LON[0, ln_[i]] - LON[0, lp[i]]) * DY / 0.5 * np.cos(np.deg2rad(LAT[i, 0]))
                                    v = dx * 1000 * n[0] / (6 * 3600)
                                    if abs(v) <= 20:
                                        cn[i] = v
                            if len(cn) >= 5:  # 5-point running mean along the line
                                rs = sorted(cn)
                                cn = {r: float(np.mean([cn[x] for x in rs if abs(x - r) <= 2])) for r in rs}
                        r = mode_at(fl, line, nrm, cn)
                        if r is None:
                            continue
                        shares, dg = r
                        bl = [wblk[b] for b in BLOCK_OF[h] if b in wblk]
                        w = np.mean(bl, axis=0) if bl else np.ones(LAT.shape)
                        acc = acc + w * shares
                        acc_w = acc_w + w
                        accD = accD + w * dg["D"]
                        accQ = accQ + w * dg["qlcs_n"]
                        n_used += 1
                        n_cl += int(dg["has_cl"])
                        if c == cyc[-1] and mem == members(m)[0] and h == 24:
                            save[f"{L}|{m}|line00z"] = np.array([[LAT[i, 0], LON[0, j]] for i, j in sorted(line.items())], np.float32)
            if n_used == 0:
                print(f"lead {L} {m}: no usable boundary")
                continue
            sh = acc / np.maximum(acc_w, 1e-6)
            save[f"{L}|{m}|shares"] = np.where(us, sh, 0).astype(np.float16)
            save[f"{L}|{m}|D"] = np.where(us, accD / np.maximum(acc_w, 1e-6), np.nan).astype(np.float16)
            save[f"{L}|{m}|qlcs_n"] = np.where(us, accQ / np.maximum(acc_w, 1e-6), np.nan).astype(np.float16)
            save[f"{L}|{m}|n"] = np.array([n_used, n_cl], np.int32)
            i, j = int(round((35.5 - 20) / .5)), int(round((-97.5 + 130) / .5))
            print(f"lead {L} {m}: {n_used} member-times ({n_cl} with 700/300 hPa winds); OKC shares "
                  f"disc {sh[0, i, j]:.2f} semi {sh[1, i, j]:.2f} lin {sh[2, i, j]:.2f} disorg {sh[3, i, j]:.2f}", flush=True)
    p = os.path.join(out_dir, f"mode_env_{a.date}.npz")
    np.savez_compressed(p, **save)
    print("->", p)


if __name__ == "__main__":
    main()

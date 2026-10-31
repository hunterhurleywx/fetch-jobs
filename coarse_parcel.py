"""Surface-based / mixed-layer / most-unstable CAPE and CIN from a few pressure levels.

research/11_qlcs_vs_supercell/02_overnight_hslc_and_ecmwf_cin.md, Part B; checked against MetPy, GFS UPP and
SHARPpy in 05_ecmwf_cin_triple_check.md. In the pipeline since 31 Oct 2026: derive_parcel.py turns the ECMWF
upper-air sets (hres_snd, eps_pl) into hres_pcl / eps_pcl (ML90 CIN and CAPE, best-parcel CAPE), so ECMWF gets
the same CIN and surface-based-share terms as GFS.

Purpose: give ECMWF HRES (`hres_snd`, 12 levels) and EPS (`eps_pl`, 7 levels) the same CIN and
surface-based-share information that GFS/GEFS/CAMs get from NCEP UPP, without composite indices.

Method (vectorised over columns, no per-column Python loop):
  * nodes   = the 2 m point (surface pressure, T2m, r from Td2m or q2; by default T2m is floored at the
              dry-adiabatic extension of the lowest level, i.e. the shallow surface inversion is removed)
              + every pressure level at least
              `gap` hPa above the ground; levels below ground (p >= sp - gap) are dropped (they are
              extrapolated by the model and would invent a cold or warm layer).
  * profile = T and mixing ratio interpolated linearly in ln p onto a terrain-following grid
              p_k = sp - k*dp (dp = 10 hPa), up to the top level.
  * parcels = SB (2 m), ML (mean theta and r of the lowest `ml_depth` hPa, launched from the surface),
              MU (max theta-e of the fine grid in the lowest 300 hPa, launched from its level).
  * ascent  = dry adiabat to the Bolton LCL, then pseudo-adiabatic RK2 steps of dp; no condensate loading.
  * buoyancy uses virtual temperature for parcel and environment (Doswell & Rasmussen 1994);
    integration in ln p: CAPE/CIN = Rd * sum (Tv_p - Tv_e) dln p.
    CAPE = positive area between LFC (first positive level above the LCL) and EL (last positive level);
    CIN  = negative area from the launch level to the LFC; both 0 when there is no LFC.
    This is the convention of prototype/sounding.py (validated against NCEP ML CAPE, r = 0.99).

Inputs are arrays shaped (L, ...) for levels and (...) for surface fields; any trailing shape works.
"""
import numpy as np

RD, RV, CP, LV, G = 287.04, 461.5, 1005.7, 2.501e6, 9.80665
EPS, KAPPA = RD / RV, RD / CP


def es_hpa(t):
    return 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def rsat(t, p):
    e = es_hpa(t)
    return EPS * e / np.maximum(p - e, 1e-3)


def r_from_q(q):
    return q / (1.0 - q)


def r_from_td(td, p):
    e = es_hpa(td)
    return EPS * e / np.maximum(p - e, 1e-3)


def r_from_rh(rh, t, p):
    """Mixing ratio from RH (%) w.r.t. water (ECMWF 'r' is w.r.t. ice below -23 C; small effect aloft)."""
    return np.clip(rh, 0, 100) / 100.0 * rsat(t, p)


def tv(t, r, virtual=True):
    return t * (1 + r / EPS) / (1 + r) if virtual else t


def lcl(t0, r0, p0):
    e = np.maximum(r0 * p0 / (EPS + r0), 1e-3)
    a = np.log(e / 6.112)
    td = 243.5 * a / (17.67 - a) + 273.15
    tl = 1.0 / (1.0 / (np.minimum(td, t0) - 56.0) + np.log(t0 / np.minimum(td, t0)) / 800.0) + 56.0
    return tl, p0 * (tl / t0) ** (1.0 / KAPPA)


def theta_e(t, r, p):
    tl, _ = lcl(t, r, p)
    return t * (1000.0 / p) ** (0.2854 * (1 - 0.28 * r)) * np.exp((3.376 / tl - 0.00254) * 1000 * r * (1 + 0.81 * r))


def _dtdp(t, p):
    rs = rsat(t, p)
    return (RD * t + LV * rs) / (p * (CP + LV * LV * rs * EPS / (RD * t * t)))


def _moist_step(t, p0, p1):
    k1 = _dtdp(t, p0)
    tm = t + 0.5 * k1 * (p1 - p0)
    return t + _dtdp(tm, 0.5 * (p0 + p1)) * (p1 - p0)


def fine_profile(levels, t, r, sp, t2, r2, dp=10.0, gap=5.0, ptop=None):
    """Environment on p_k = sp - k*dp. levels (L,) hPa descending in p; t, r (L, n); sp, t2, r2 (n,) hPa/K/kg/kg.

    Returns P, T, R as (K, n); points above the top level are NaN."""
    levels = np.asarray(levels, float)
    L, n = t.shape
    above = levels[:, None] < (sp[None, :] - gap)  # (L, n) levels genuinely above ground
    # node arrays: surface first, then levels; below-ground levels collapse onto the surface node
    lp = np.vstack([np.log(sp)[None], np.where(above, np.log(levels)[:, None], np.log(sp)[None])])
    nt = np.vstack([t2[None], np.where(above, t, t2[None])])
    nr = np.vstack([r2[None], np.where(above, r, r2[None])])
    ptop = levels.min() if ptop is None else ptop
    K = int(np.ceil((sp.max() - ptop) / dp)) + 1
    P = sp[None, :] - dp * np.arange(K)[:, None]
    P = np.where(P >= ptop, P, np.nan)
    lpf = np.log(np.where(np.isfinite(P), P, ptop))
    # upper node index: number of nodes with ln p >= ln pf (nodes are non-increasing in p)
    cnt = (lp[None, :, :] >= lpf[:, None, :] - 1e-9).sum(axis=1)  # (K, n)
    hi = np.clip(cnt, 1, L)
    lo = hi - 1
    take = lambda a, i: np.take_along_axis(a, i, axis=0)
    l0, l1 = take(lp, lo), take(lp, hi)
    w = np.clip((lpf - l0) / np.where(np.abs(l1 - l0) < 1e-9, -1e-9, l1 - l0), 0, 1)
    T = take(nt, lo) + w * (take(nt, hi) - take(nt, lo))
    R = take(nr, lo) + w * (take(nr, hi) - take(nr, lo))
    bad = ~np.isfinite(P)
    T[bad], R[bad] = np.nan, np.nan
    return P, T, np.clip(R, 1e-6, None)


def _parcel(P, T, R, k0, t0, r0, p0, virtual=True, lfc="top"):
    """CAPE, CIN (J/kg) and LCL pressure of a parcel launched at fine index k0 (n,).

    lfc="bottom": LFC = first positive level above the LCL (MetPy's default). A shallow positive layer under a
    cap then hides the cap's negative area from CIN.
    lfc="top": LFC = base of the last positive layer below the EL (SHARPpy's parcelx re-sets the LFC at every
    negative-to-positive crossing), so the cap counts as CIN; CAPE is still all positive area from that LFC up."""
    K, n = P.shape
    tl, pl = lcl(t0, r0, p0)
    th0 = t0 * (1000.0 / p0) ** KAPPA
    tm, pm = tl.copy(), pl.copy()
    b = np.full((K, n), np.nan)
    for k in range(K):
        p = P[k]
        act = (k >= k0) & np.isfinite(p)
        dry = act & (p >= pl)
        moist = act & (p < pl)
        tp = np.full(n, np.nan)
        rp = np.full(n, np.nan)
        tp[dry] = th0[dry] * (p[dry] / 1000.0) ** KAPPA
        rp[dry] = r0[dry]
        if moist.any():
            tn = _moist_step(tm[moist], pm[moist], p[moist])
            tm[moist], pm[moist] = tn, p[moist]
            tp[moist], rp[moist] = tn, rsat(tn, p[moist])
        b[k] = tv(tp, rp, virtual) - tv(T[k], R[k], virtual)
    lnp = np.log(P)
    dl = np.vstack([np.zeros((1, n)), -np.diff(lnp, axis=0)])  # ln p decrement into level k
    seg = np.where(np.isfinite(dl), dl, 0.0)
    bm = 0.5 * (b + np.vstack([b[:1], b[:-1]]))  # mean buoyancy of the layer ending at k
    pos = np.isfinite(b) & (b > 0) & (P <= pl[None]) & (np.arange(K)[:, None] > k0[None])
    has = pos.any(0)
    klfc = np.where(has, np.argmax(pos, axis=0), K)
    kel = np.where(has, K - 1 - np.argmax(pos[::-1], axis=0), -1)
    if lfc == "top":  # base of the positive run that ends at the EL
        idx0 = np.arange(K)[:, None]
        notpos = ~pos & (idx0 < kel[None]) & (idx0 > k0[None])
        last_neg = np.where(notpos.any(0), K - 1 - np.argmax(notpos[::-1], axis=0), -1)
        klfc = np.where(has, np.maximum(klfc, last_neg + 1), K)
    idx = np.arange(K)[:, None]
    inside = (idx > klfc[None]) & (idx <= kel[None])
    below = (idx > k0[None]) & (idx <= klfc[None])
    bmf = np.nan_to_num(bm)
    cape = RD * (np.where(inside & (bmf > 0), bmf, 0) * seg).sum(0)
    cin = RD * (np.where(below & (bmf < 0), bmf, 0) * seg).sum(0)
    return np.where(has, cape, 0.0), np.where(has, cin, 0.0), pl


def strip_surface_inversion(levels, t, sp, t2, gap=5.0):
    """2 m temperature not allowed below the dry-adiabatic extension of the lowest level above ground.

    Removes only the shallow radiational surface inversion (tens of metres) that NCEP UPP's native-level
    parcels do not see. With it, 7-level ML90 CIN matches GFS cin90 at all hours (02_... .md Part B)."""
    lv = np.asarray(levels, float)
    above = lv[:, None] < (sp[None, :] - gap)
    j = np.argmax(above, axis=0)
    cols = np.arange(sp.size)
    return np.maximum(t2, t[j, cols] * (sp / lv[j]) ** KAPPA)


def coarse_cape_cin(levels, t, r, sp, t2, r2, ml_depth=100.0, dp=10.0, gap=5.0, ptop=None,
                    virtual=True, parcels=("sb", "ml", "mu"), mu_depth=300.0, sfc="no_inversion", lfc="top"):
    """SB / ML / MU / BL6 CAPE and CIN from pressure levels. Shapes: t, r (L, *S); sp (hPa), t2, r2 (*S).

    sfc: "no_inversion" (default; see strip_surface_inversion) or "2m" (raw 2 m temperature).
    "bl6" = UPP-style best parcel: the highest-theta-e of six 30-hPa layer means in the lowest 180 hPa.
    Returns {sb_cape, sb_cin, ml_cape, ml_cin, mu_cape, mu_cin, bl6_cape, bl6_cin, ...} each shaped S."""
    S = np.shape(sp)
    L = len(levels)
    t, r = np.reshape(t, (L, -1)).astype(float), np.reshape(r, (L, -1)).astype(float)
    sp, t2, r2 = (np.ravel(a).astype(float) for a in (sp, t2, r2))
    if sfc == "no_inversion":
        t2 = strip_surface_inversion(levels, t, sp, t2, gap)
    r2 = np.minimum(r2, rsat(t2, sp))  # no supersaturated 2 m air
    P, T, R = fine_profile(levels, t, r, sp, t2, r2, dp, gap, ptop)
    n = sp.size
    k0 = np.zeros(n, int)
    out = {}
    if "sb" in parcels:
        out["sb_cape"], out["sb_cin"], out["sb_lclp"] = _parcel(P, T, R, k0, t2, r2, sp, virtual, lfc)
    if "ml" in parcels:
        ml = np.isfinite(P) & (P >= sp[None] - ml_depth)
        th = T * (1000.0 / P) ** KAPPA
        th_ml = np.nansum(np.where(ml, th, 0), 0) / ml.sum(0)
        r_ml = np.nansum(np.where(ml, R, 0), 0) / ml.sum(0)
        out["ml_cape"], out["ml_cin"], out["ml_lclp"] = _parcel(
            P, T, R, k0, th_ml * (sp / 1000.0) ** KAPPA, r_ml, sp, virtual, lfc)
    if "mu" in parcels:
        lay = np.isfinite(P) & (P >= sp[None] - mu_depth)
        the = np.where(lay, theta_e(T, R, P), -np.inf)
        kmu = np.argmax(the, axis=0)
        cols = np.arange(n)
        out["mu_cape"], out["mu_cin"], _ = _parcel(P, T, R, kmu, T[kmu, cols], R[kmu, cols], P[kmu, cols], virtual, lfc)
        out["mu_p"] = P[kmu, cols]
    if "bl6" in parcels:  # pick the highest-theta-e 30-hPa layer first, then lift once
        th = T * (1000.0 / P) ** KAPPA
        cols = np.arange(n)
        the6, t6, r6, p6, k6 = (np.empty((6, n)) for _ in range(5))
        for j in range(6):
            m = np.isfinite(P) & (P <= sp[None] - 30.0 * j + 1e-6) & (P >= sp[None] - 30.0 * (j + 1) - 1e-6)
            k6[j] = int(round((30.0 * j + 15.0) / dp))
            p6[j] = P[k6[j].astype(int), cols]
            t6[j] = np.nansum(np.where(m, th, 0), 0) / m.sum(0) * (p6[j] / 1000.0) ** KAPPA
            r6[j] = np.nansum(np.where(m, R, 0), 0) / m.sum(0)
            the6[j] = theta_e(t6[j], r6[j], p6[j])
        jb = np.nan_to_num(the6, nan=-np.inf).argmax(0)
        pick = lambda a: a[jb, cols]
        out["bl6_cape"], out["bl6_cin"], _ = _parcel(P, T, R, pick(k6).astype(int), pick(t6), pick(r6), pick(p6), virtual, lfc)
        out["bl6_p"] = pick(p6)
    return {k: v.reshape(S) for k, v in out.items()}


def sp_from_heights(levels, z, t, t2, zs):
    """Surface pressure (hPa) by the hypsometric equation from the lowest level whose height is above the
    terrain zs (m). For GEFS, whose `gefs_pl` / main files carry no surface pressure. z, t (L, ...)."""
    lv = np.asarray(levels, float)
    z, t = np.asarray(z, float), np.asarray(t, float)
    above = z > zs[None]
    j = np.argmax(above, axis=0)
    zj = np.take_along_axis(z, j[None], 0)[0]
    tj = np.take_along_axis(t, j[None], 0)[0]
    return lv[j] * np.exp(G * (zj - zs) / (RD * 0.5 * (tj + t2)))


# ---------------------------------------------------------------------------
# Convenience wrappers for the cached field dicts
# ---------------------------------------------------------------------------
HRES_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
EPS_LEVELS = [1000, 925, 850, 700, 500, 300, 250]


def from_hres_snd(f, levels=None, main_grid=True, **kw):
    """f: hres_snd dict (0.25 deg; t/q at HRES_LEVELS, sp Pa, t2m, d2m). By default sampled onto the 0.5 deg main
    grid and thinned to EPS_LEVELS, so HRES is treated exactly like EPS / GEFS / GFS; levels=HRES_LEVELS keeps all 12."""
    levels = EPS_LEVELS if levels is None else levels
    sub = (lambda a: np.asarray(a, float)[::2, ::2]) if main_grid else (lambda a: np.asarray(a, float))
    sp = sub(f["sp"]) / 100.0
    t = np.stack([sub(f[f"t{l}"]) for l in levels])
    r = r_from_q(np.stack([sub(f[f"q{l}"]) for l in levels]))
    return coarse_cape_cin(levels, t, r, sp, sub(f["t2m"]), r_from_td(sub(f["d2m"]), sp), **kw)


def from_eps_pl(f, sp=None, **kw):
    """f: eps_pl dict (t/rh at EPS_LEVELS, t2m, d2m). eps_pl has no surface pressure: pass `sp` (Pa) from
    the main EPS file (EC_FIELDS 'sp'), which is the same member and step."""
    sp = (f["sp"] if sp is None else sp) / 100.0
    t = np.stack([f[f"t{l}"] for l in EPS_LEVELS])
    r = np.stack([r_from_rh(f[f"rh{l}"], f[f"t{l}"], l) for l in EPS_LEVELS])
    return coarse_cape_cin(EPS_LEVELS, t, r, sp, f["t2m"], r_from_td(f["d2m"], sp), **kw)


def from_gefs_pl(f, zs, **kw):
    """f: gefs_pl dict (z/t/rh at EPS_LEVELS, t2m, rh2m); zs: terrain height (m) on the same grid
    (scoring.orography() is the GFS cell mean). Surface pressure is hypsometric (sp_from_heights)."""
    t = np.stack([f[f"t{l}"] for l in EPS_LEVELS])
    sp = sp_from_heights(EPS_LEVELS, np.stack([f[f"z{l}"] for l in EPS_LEVELS]), t, f["t2m"], zs)
    r = np.stack([r_from_rh(f[f"rh{l}"], f[f"t{l}"], l) for l in EPS_LEVELS])
    return coarse_cape_cin(EPS_LEVELS, t, r, sp, f["t2m"], r_from_rh(f["rh2m"], f["t2m"], sp), **kw)


def from_gfs_snd(f, **kw):
    """f: gfs_snd dict (0.25 deg). Sampled onto the 0.5 deg main grid and thinned to EPS_LEVELS so GFS is
    treated exactly like the other three globals."""
    sub = lambda a: np.asarray(a, float)[::2, ::2]
    sp = sub(f["sp"]) / 100.0
    t = np.stack([sub(f[f"t{l}"]) for l in EPS_LEVELS])
    r = r_from_q(np.stack([sub(f[f"q{l}"]) for l in EPS_LEVELS]))
    return coarse_cape_cin(EPS_LEVELS, t, r, sp, sub(f["t2m"]), r_from_q(sub(f["q2"])), **kw)

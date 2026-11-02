"""Full-profile (Skew-T + hodograph) analysis of model columns on the 0.25 deg grid.

Every column inside the CONUS box is rebuilt on a 100 m height-AGL grid (0-16 km) from
the surface fields plus all pressure levels above ground. From that profile we derive raw,
physically interpretable features of the whole sounding and hodograph -- no composite
indices. Parcel code is our own, vectorised over columns:
  * pseudo-adiabatic ascent (RK2 in pressure), virtual-temperature buoyancy
  * SB / ML (lowest 100 hPa) / MU (max theta-e in lowest 300 hPa) parcels
  * DCAPE from the minimum-theta-e level in the lowest 400 hPa (Normand descent)

Features (all 2-D maps):
  thermo : sb/ml/mu CAPE & CIN, ML LCL/LFC/EL heights, ML 0-3 km CAPE, ML hail-growth-zone
           CAPE (-10 to -30 C), DCAPE, lapse rates 0-1/0-3/3-6 km and 700-500 hPa, strongest
           low-level inversion (cap) and its base, max 2-km lapse rate 1-6 km (EML), ML mixing
           ratio, PW, moisture depth (RH >= 60%), mean RH 700-500 hPa, min RH 1-6 km,
           freezing-level height
  kinem  : bulk shear 0-0.5/1/3/6/8 km, 0-6 km mean wind, Bunkers right-mover, SRH
           0-0.5/1/3 km, storm-relative wind 0-1 km / 4-6 km / 9-11 km, critical angle,
           veering 0-1 / 0-3 km, backing within 1-5 km (veer-back), hodograph length 0-6 km,
           low-level jet max (0-2 km) and its height
"""
import numpy as np

from config import EC_SND_LEVELS, GFS_SND_LEVELS, SND_LATS, SND_LONS

RD, RV, CP, LV, G = 287.04, 461.5, 1005.7, 2.501e6, 9.80665
EPS, KAPPA = RD / RV, RD / CP
H = np.arange(0.0, 16001.0, 100.0)  # height AGL grid (m)
DZ = 100.0
BOX = (24.0, 50.0, -126.0, -66.0)
_LA, _LO = np.meshgrid(SND_LATS, SND_LONS, indexing="ij")
BOX_MASK = (_LA >= BOX[0]) & (_LA <= BOX[1]) & (_LO >= BOX[2]) & (_LO <= BOX[3])


def es_hpa(t):
    return 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def rsat(t, p):
    e = es_hpa(t)
    return EPS * e / np.maximum(p - e, 1e-3)


def tv(t, r):
    return t * (1 + r / EPS) / (1 + r)


def lcl(t0, r0, p0):
    e = np.maximum(r0 * p0 / (EPS + r0), 1e-3)
    a = np.log(e / 6.112)
    td = 243.5 * a / (17.67 - a) + 273.15
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t0 / td) / 800.0) + 56.0
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


# ---------------------------------------------------------------------------
# Profile reconstruction
# ---------------------------------------------------------------------------
def build_profiles(model, f):
    """Columns inside BOX on the H grid. Returns dict of (n, K) arrays + surface vectors."""
    levels = np.array(GFS_SND_LEVELS if model.startswith("gfs") else EC_SND_LEVELS, float)
    sel = BOX_MASK
    st = lambda k: np.stack([f[f"{k}{int(l)}"][sel] for l in levels])  # (L, n)
    z, t, q, u, v = st("z"), st("t"), st("q"), st("u"), st("v")
    sp = f["sp"][sel] / 100.0
    t2 = f["t2m"][sel]
    if "q2" in f:
        q2 = f["q2"][sel]
    else:  # ECMWF: 2 m dewpoint
        e = es_hpa(f["d2m"][sel])
        q2 = EPS * e / (sp - (1 - EPS) * e)
    u10, v10 = f["u10"][sel], f["v10"][sel]
    if "zs" in f:
        zs = f["zs"][sel]
    else:  # ECMWF open data has no orography: hypsometric from the lowest level above ground
        above = levels[:, None] < sp[None, :]
        j = np.argmax(above, axis=0)
        cols = np.arange(sp.size)
        tl, zl, pl = t[j, cols], z[j, cols], levels[j]
        zs = zl - RD * 0.5 * (tl + t2) / G * np.log(sp / pl)
    n, K = sp.size, H.size
    T, LNP, R, U, V = (np.empty((n, K)) for _ in range(5))
    r = q / (1 - q)
    r2 = q2 / (1 - q2)
    for i in range(n):
        ok = levels < sp[i]
        za = z[ok, i] - zs[i]
        good = za > 20.0
        za = za[good]
        hz = np.concatenate(([0.0], za))
        T[i] = np.interp(H, hz, np.concatenate(([t2[i]], t[ok, i][good])))
        LNP[i] = np.interp(H, hz, np.concatenate(([np.log(sp[i])], np.log(levels[ok][good]))))
        R[i] = np.interp(H, hz, np.concatenate(([r2[i]], r[ok, i][good])))
        hw = np.concatenate(([10.0], za))
        U[i] = np.interp(H, hw, np.concatenate(([u10[i]], u[ok, i][good])))
        V[i] = np.interp(H, hw, np.concatenate(([v10[i]], v[ok, i][good])))
    P = np.exp(LNP)
    R = np.clip(R, 1e-6, None)
    return dict(T=T, P=P, R=R, U=U, V=V, sp=sp, n=n)


# ---------------------------------------------------------------------------
# Parcels
# ---------------------------------------------------------------------------
def lift(T, P, k0, t0, r0, p0):
    """Parcel virtual temperature on the H grid from start index k0 (per column)."""
    n, K = T.shape
    tl, pl = lcl(t0, r0, p0)
    th0 = t0 * (1000.0 / p0) ** KAPPA
    tvp = np.full((n, K), np.nan)
    tprev, pprev = tl.copy(), pl.copy()
    started = np.zeros(n, bool)
    for k in range(K):
        p = P[:, k]
        act = k >= k0
        dry = act & (p >= pl)
        moist = act & (p < pl)
        if dry.any():
            td = th0[dry] * (p[dry] / 1000.0) ** KAPPA
            tvp[dry, k] = tv(td, r0[dry])
        if moist.any():
            tn = _moist_step(tprev[moist], pprev[moist], p[moist])
            tprev[moist], pprev[moist] = tn, p[moist]
            started[moist] = True
            tvp[moist, k] = tv(tn, rsat(tn, p[moist]))
    return tvp, pl


def parcel_stats(T, P, R, k0, t0, r0, p0):
    tve = tv(T, R)
    tvp, pl = lift(T, P, k0, t0, r0, p0)
    b = G * (tvp - tve) / tve
    valid = np.isfinite(b) & (P > 100.0)
    pos = valid & (b > 0) & (P <= pl[:, None])
    has = pos.any(axis=1)
    K = H.size
    klfc = np.where(has, np.argmax(pos, axis=1), K - 1)
    kel = np.where(has, K - 1 - np.argmax(pos[:, ::-1], axis=1), 0)
    idx = np.arange(K)[None, :]
    above = (idx >= klfc[:, None]) & (idx <= kel[:, None]) & valid
    below = (idx < klfc[:, None]) & valid
    bpos = np.where(above & (b > 0), b, 0.0)
    cape = np.where(has, bpos.sum(1) * DZ, 0.0)
    cin = np.where(has, np.where(below & (b < 0), b, 0.0).sum(1) * DZ, 0.0)
    cape3 = np.where(has, (bpos * (H[None, :] <= 3000)).sum(1) * DZ, 0.0)
    hgz = (T <= 263.15) & (T >= 243.15)
    cape_hgz = np.where(has, (bpos * hgz).sum(1) * DZ, 0.0)
    lcl_h = H[np.clip(np.argmax(P <= pl[:, None], axis=1), 0, K - 1)]
    return dict(cape=cape, cin=cin, cape3=cape3, cape_hgz=cape_hgz, lcl=lcl_h,
                lfc=np.where(has, H[klfc], np.nan), el=np.where(has, H[kel], np.nan))


def dcape(T, P, R):
    """Saturated descent from the minimum theta-e level in the lowest 400 hPa."""
    n, K = T.shape
    the = theta_e(T, R, P)
    lay = (P >= P[:, :1] - 400.0) & (np.arange(K)[None, :] > 0)
    ks = np.argmin(np.where(lay, the, np.inf), axis=1)
    cols = np.arange(n)
    tl, pl = lcl(T[cols, ks], R[cols, ks], P[cols, ks])
    tcur, pcur = tl.copy(), pl.copy()
    tve = tv(T, R)
    out = np.zeros(n)
    for k in range(K - 1, -1, -1):
        m = k <= ks
        if not m.any():
            continue
        tn = _moist_step(tcur[m], pcur[m], P[m, k])
        tcur[m], pcur[m] = tn, P[m, k]
        neg = (tve[m, k] - tv(tn, rsat(tn, P[m, k]))) / tve[m, k] * G
        out[m] += np.clip(neg, 0, None) * DZ
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _at(A, h):
    return A[:, int(round(h / DZ))]


def _lapse(T, h1, h2):
    return (_at(T, h1) - _at(T, h2)) / ((h2 - h1) / 1000.0)


def _hgt_of(mask):
    """Height of the first True along the column (nan if none)."""
    has = mask.any(axis=1)
    return np.where(has, H[np.argmax(mask, axis=1)], np.nan)


def features(model, f):
    pr = build_profiles(model, f)
    T, P, R, U, V = pr["T"], pr["P"], pr["R"], pr["U"], pr["V"]
    n, K = T.shape
    out = {}
    # --- parcels
    out.update({f"sb_{k}": v for k, v in parcel_stats(T, P, R, np.zeros(n, int), T[:, 0], R[:, 0], P[:, 0]).items()})
    ml = P >= (P[:, :1] - 100.0)
    th = T * (1000.0 / P) ** KAPPA
    th_ml = (th * ml).sum(1) / ml.sum(1)
    r_ml = (R * ml).sum(1) / ml.sum(1)
    t_ml = th_ml * (P[:, 0] / 1000.0) ** KAPPA
    out.update({f"ml_{k}": v for k, v in parcel_stats(T, P, R, np.zeros(n, int), t_ml, r_ml, P[:, 0]).items()})
    the = theta_e(T, R, P)
    mu_lay = P >= (P[:, :1] - 300.0)
    kmu = np.argmax(np.where(mu_lay, the, -np.inf), axis=1)
    cols = np.arange(n)
    mu = parcel_stats(T, P, R, kmu, T[cols, kmu], R[cols, kmu], P[cols, kmu])
    out.update({"mu_cape": mu["cape"], "mu_cin": mu["cin"], "mu_start_hgt": H[kmu]})
    out["dcape"] = dcape(T, P, R)
    # --- thermodynamic structure
    out["lr01"], out["lr03"], out["lr36"] = _lapse(T, 0, 1000), _lapse(T, 0, 3000), _lapse(T, 3000, 6000)
    win = 15  # strongest warming within any 1.5 km layer starting 0.3-4 km AGL: the cap / inversion
    k1, k2 = 3, 40
    inv = np.stack([np.max(T[:, k:k + win], axis=1) - T[:, k] for k in range(k1, k2)], axis=1)
    out["cap_dt"] = inv.max(1)
    out["cap_base"] = H[k1 + np.argmax(inv, axis=1)]
    lr2 = np.stack([(T[:, k] - T[:, k + 20]) / 2.0 for k in range(10, 41)], axis=1)  # 2-km layers, base 1-4 km
    out["eml_lr"] = lr2.max(1)
    out["eml_base"] = H[10 + np.argmax(lr2, axis=1)]
    rh = np.clip(R / rsat(T, P), 0, 1.2)
    out["ml_r"] = 1000.0 * r_ml
    q = R / (1 + R)
    out["pw"] = np.sum(0.5 * (q[:, 1:] + q[:, :-1]) * (P[:, :-1] - P[:, 1:]) * 100.0, axis=1) / G
    out["moist_depth"] = np.where(np.isnan(_hgt_of(rh < 0.6)), 4000.0, np.minimum(_hgt_of(rh < 0.6), 4000.0))
    mid = (P <= 700) & (P >= 500)
    out["rh75"] = 100.0 * np.where(mid, rh, 0).sum(1) / np.maximum(mid.sum(1), 1)
    out["rh_min16"] = 100.0 * np.min(rh[:, 10:61], axis=1)
    out["frz_hgt"] = _hgt_of(T < 273.15)
    # --- hodograph
    shr = lambda a, b: np.hypot(_at(U, b) - _at(U, a), _at(V, b) - _at(V, a))
    for a, b, nm in ((0, 500, "shr005"), (0, 1000, "shr01"), (0, 3000, "shr03"), (0, 6000, "shr06"), (0, 8000, "shr08")):
        out[nm] = shr(a, b)
    um, vm = U[:, :61].mean(1), V[:, :61].mean(1)
    su = U[:, 55:61].mean(1) - U[:, :6].mean(1)
    sv = V[:, 55:61].mean(1) - V[:, :6].mean(1)
    sm = np.maximum(np.hypot(su, sv), 1e-3)
    cu, cv = um + 7.5 * sv / sm, vm - 7.5 * su / sm  # Bunkers right mover
    out["mean_wind06"] = np.hypot(um, vm)
    out["rm_u"], out["rm_v"] = cu, cv

    def srh(top):
        k = int(top / DZ)
        uu, vv = U[:, :k + 1] - cu[:, None], V[:, :k + 1] - cv[:, None]
        return np.sum(uu[:, 1:] * vv[:, :-1] - uu[:, :-1] * vv[:, 1:], axis=1)
    out["srh005"], out["srh01"], out["srh03"] = srh(500), srh(1000), srh(3000)
    srw = lambda a, b: np.hypot(U[:, a:b + 1].mean(1) - cu, V[:, a:b + 1].mean(1) - cv)
    out["srw01"], out["srw46"], out["srw911"] = srw(0, 10), srw(40, 60), srw(90, 110)
    inflow_u, inflow_v = U[:, 0] - cu, V[:, 0] - cv
    sh_u, sh_v = _at(U, 500) - U[:, 0], _at(V, 500) - V[:, 0]
    cosang = (inflow_u * sh_u + inflow_v * sh_v) / np.maximum(np.hypot(inflow_u, inflow_v) * np.hypot(sh_u, sh_v), 1e-6)
    out["crit_angle"] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ang = np.arctan2(V, U)
    dang = np.angle(np.exp(1j * np.diff(ang, axis=1)))  # +ccw (backing), -cw (veering)
    out["veer01"] = -np.degrees(dang[:, :10].sum(1))
    out["veer03"] = -np.degrees(dang[:, :30].sum(1))
    seg = -np.degrees(np.add.reduceat(dang[:, 10:50], np.arange(0, 40, 5), axis=1))  # 500-m segments, 1-5 km
    spd15 = np.hypot(U[:, 10:50], V[:, 10:50]).reshape(n, 8, 5).min(2)
    out["backing15"] = np.where(spd15 >= 7.5, np.clip(-seg, 0, None), 0).sum(1)  # ignore light/variable winds
    out["hodo_len06"] = np.hypot(np.diff(U[:, :61], axis=1), np.diff(V[:, :61], axis=1)).sum(1)
    spd = np.hypot(U[:, :21], V[:, :21])
    out["llj_max"], out["llj_hgt"] = spd.max(1), H[np.argmax(spd, axis=1)]
    # --- carried through for scoring / maps
    out["lr75"] = (f["t700"][BOX_MASK] - f["t500"][BOX_MASK]) / ((f["z500"][BOX_MASK] - f["z700"][BOX_MASK]) / 1000.0)
    out["omega"] = 0.5 * (f["w500"][BOX_MASK] + f["w700"][BOX_MASK])
    out["z500"] = f["z500"][BOX_MASK]
    maps = {}
    for k, v in out.items():
        m = np.full(BOX_MASK.shape, np.nan, np.float32)
        m[BOX_MASK] = v
        maps[k] = m
    return maps

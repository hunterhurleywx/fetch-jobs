"""Ingredient scoring on the common 0.5 deg grid.

Two schemes are implemented side by side:

* ``current`` -- a faithful port of GFS/gfs-ensemble.py (score_grid + its
  250/300/150 km smoothing chain), applied to every model.
* ``v2`` -- the research-based prototype (research/01_severe_ingredients/00_synthesis.md
  section 0.4, Tier 2). It is uncalibrated; outputs are an index, not a probability.
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from config import CAM_BLEND, GRID_LATS

_DY_KM = 0.5 * 111.2
_DX_KM = 0.5 * 111.2 * np.cos(np.deg2rad(37.5))  # grid spacing at mid-domain


def gauss(field, km):
    if km <= 0:
        return field
    f = np.nan_to_num(field, nan=0.0)
    return gaussian_filter(f, sigma=(km / _DY_KM, km / _DX_KM), mode="nearest")


def ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ingredient extraction (model-specific field names -> common ingredients)
# ---------------------------------------------------------------------------
def q_from_td(td_k, p_pa):
    td = td_k - 273.15
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))  # hPa
    p = p_pa / 100.0
    return 1000.0 * 0.622 * e / (p - 0.378 * e)  # g/kg


_RD_G = 287.04 / 9.80665


def ingredients(model, f):
    """Return a dict of common ingredients for one member/run at one valid time."""
    if "z500" in f:
        dz = f["z500"] - f["z700"]
    else:  # hypsometric 700-500 hPa thickness (EPS members skip geopotential)
        dz = _RD_G * 0.5 * (f["t500"] + f["t700"]) * np.log(700.0 / 500.0)
    ing = {
        "shear": np.hypot(f["u500"] - f["u10"], f["v500"] - f["v10"]),
        "lr75": (f["t700"] - f["t500"]) / (dz / 1000.0),
        "omega": 0.5 * (f["w500"] + f["w700"]),  # Pa/s, negative = ascent
        "pwat": f["pwat"],
        "z500": f.get("z500"),
        "dz75": dz,
    }
    if "u850" in f:  # low-level shear / LLJ (CAM main files; merged from *_x for global models)
        ing["llshear"] = np.hypot(f["u850"] - f["u10"], f["v850"] - f["v10"])
        ing["v850"] = np.hypot(f["u850"], f["v850"])
    if model in ("gfs", "gefs", "hrrr", "nam", "rap", "nam12", "rrfs"):  # NCEP UPP fields (CAMs carry the same set)
        ing["q2"] = 1000.0 * f["q2"]
        ing["cape_cur"], ing["cin_cur"] = f["cape180"], f["cin180"]
        if "cape90" in f:  # GFS v16+: true 90-hPa mixed layer
            ing["cape_v2"], ing["cin_v2"], ing["cape_scale"] = f["cape90"], f["cin90"], 1500.0
        else:  # GEFS has no ML CAPE; 180-0 is a most-unstable-type "best" parcel
            ing["cape_v2"], ing["cin_v2"], ing["cape_scale"] = f["cape180"], f["cin180"], 2000.0
    else:  # ECMWF open data: MUCAPE only, no CIN
        ing["q2"] = q_from_td(f["d2m"], f["sp"])
        cape = np.nan_to_num(f["capemu"], nan=0.0)
        ing["cape_cur"], ing["cin_cur"] = cape, None
        ing["cape_v2"], ing["cin_v2"], ing["cape_scale"] = cape, None, 2500.0
    return ing


# ---------------------------------------------------------------------------
# Current scheme (port of GFS/gfs-ensemble.py)
# ---------------------------------------------------------------------------
def score_current(ing, fh):
    cape = np.nan_to_num(ing["cape_cur"], nan=0.0)
    inst = np.clip(np.clip(cape / 2000.0, 0, 1) * (1.0 + np.clip((ing["lr75"] - 5.5) / 2.0, 0, 1)), 0, 1)
    shear = np.clip((ing["shear"] - 12.0) / 15.0, 0, 1)
    omega = gauss(ing["omega"], 250)
    lift = np.interp(-omega, [-0.20, 0.00, 0.03, 0.08, 0.20], [0.10, 0.20, 0.55, 0.85, 1.15])
    if ing["cin_cur"] is None:
        cin = 1.0
    else:
        cin_base = np.clip((200.0 + np.nan_to_num(ing["cin_cur"], nan=0.0)) / 150.0, 0, 1)
        cin = np.power(cin_base, np.clip(1.0 + 0.5 * (fh - 120) / 120.0, 1.0, 1.5))
    moist = np.maximum(np.clip((ing["pwat"] - 10.0) / 15.0, 0, 1), 0.5)
    score = np.clip(inst * shear * lift * cin * moist, 0, 1)
    return gauss(score, 300)  # per-run pre-smooth; pool mean is post-smoothed 150 km


# ---------------------------------------------------------------------------
# v2 prototype (research-based, uncalibrated)
# ---------------------------------------------------------------------------
_LIFT_X = [-0.20, -0.08, -0.03, 0.00, 0.05, 0.20]
_LIFT_Y = [1.10, 1.00, 0.80, 0.65, 0.50, 0.20]

# Optional v2 profiles: background tools from the case diagnostics, not defaults.
# research/05_case_2025-04-27/03_diagnosis.md and research/04_case_2025-06-16/03_diagnosis.md
#   moist: (q2 lo, q2 hi) g/kg ramp; shear_hi: m/s for full credit; lift_neutral: score at w = 0
PROFILES = {
    "v2": dict(moist=(8.0, 13.0), shear_hi=20.0, lift_neutral=0.65),
    "highplains": dict(moist=(6.0, 11.0), shear_hi=20.0, lift_neutral=0.65),  # dry, high-based setups
    "shear30": dict(moist=(8.0, 13.0), shear_hi=30.0, lift_neutral=0.65),      # strongly sheared outbreaks
    "liftstrong": dict(moist=(8.0, 13.0), shear_hi=20.0, lift_neutral=0.45),   # forcing-dominated days
    "msl": dict(moist=(6.0, 11.0), shear_hi=30.0, lift_neutral=0.45),          # all three combined
    # Warm-season MCS / damaging-wind pathway: weakly sheared, high-CAPE organised clusters
    # (Evans & Doswell 2001). Scored in parallel; final = max(supercell path, MCS path).
    "mcs": dict(moist=(8.0, 13.0), shear_hi=20.0, lift_neutral=0.65, mcs=(5.0, 12.0, 0.6)),
    "msl_mcs": dict(moist=(6.0, 11.0), shear_hi=30.0, lift_neutral=0.45, mcs=(5.0, 12.0, 0.6)),
}
PROFILE = PROFILES["v2"]


def set_profile(name):
    global PROFILE
    PROFILE = PROFILES[name]


def score_v2(ing, fh, parts=False):
    p = PROFILE
    cape = np.clip(np.nan_to_num(ing["cape_v2"], nan=0.0), 0, None)
    c_term = np.clip(cape / ing["cape_scale"], 0, 1) ** 0.6
    inst = np.clip(c_term * (1.0 + 0.5 * ramp(ing["lr75"], 6.0, 7.5)), 0, 1)
    shear = ramp(ing["shear"], 10.0, p["shear_hi"])
    omega = np.clip(gauss(ing["omega"], 150), -0.3, 0.3)
    ln = p["lift_neutral"]
    lift = np.interp(omega, _LIFT_X, [1.10, 1.00, 0.80, ln, ln * 0.75, 0.20])
    cin = 1.0 if ing["cin_v2"] is None else ramp(200.0 + np.nan_to_num(ing["cin_v2"], nan=0.0), 0, 150)
    moist = 0.5 + 0.5 * ramp(ing["q2"], *p["moist"])
    raw = inst * shear * lift * cin * moist
    if "mcs" in p:
        lo, hi, cap = p["mcs"]
        raw = np.maximum(raw, cap * inst * ramp(ing["shear"], lo, hi) * lift * cin * moist)
    score = gauss(np.clip(raw, 0, 1), 100)
    if parts:
        return score, {"inst": inst, "shear": shear, "lift": lift, "cin": cin, "moist": moist}
    return score


def score_v3(ing, fh, dz12=None):
    """EXPERIMENTAL (29 Oct 2026, from SPC's 14-15 Mar 2025 Day 1 reasoning; not the product).

    v2 with three changes:
      inst   CAPE credit saturates at ~1000 J/kg ("SBCAPE should exceed 1000 J/kg" is enough with
             strong forcing/shear), so extra Gulf CAPE stops dominating;
      ll     low-level shear / LLJ: max(ramp(|V850-V10|, 8, 18 m/s), ramp(|V850|, 15, 28 m/s)),
             factor 0.6 (none) .. 1.0;
      force  12-h 500 hPa height fall (18Z->06Z) scales lift: 0.75 at < 30 m .. 1.25 at > 150 m falls
             (non-saturating forcing; 1.0 when heights are unavailable).
    """
    p = PROFILES["v2"]
    cape = np.clip(np.nan_to_num(ing["cape_v2"], nan=0.0), 0, None)
    inst = np.clip(ramp(cape, 0, 1000.0) ** 0.5 * (1.0 + 0.5 * ramp(ing["lr75"], 6.0, 7.5)), 0, 1)
    shear = ramp(ing["shear"], 10.0, p["shear_hi"])
    omega = np.clip(gauss(ing["omega"], 150), -0.3, 0.3)
    ln = p["lift_neutral"]
    lift = np.interp(omega, _LIFT_X, [1.10, 1.00, 0.80, ln, ln * 0.75, 0.20])
    if dz12 is not None:
        lift = lift * (0.75 + 0.5 * ramp(-gauss(dz12, 150), 30.0, 150.0))
    cin = 1.0 if ing["cin_v2"] is None else ramp(200.0 + np.nan_to_num(ing["cin_v2"], nan=0.0), 0, 150)
    moist = 0.5 + 0.5 * ramp(ing["q2"], *p["moist"])
    ll = 1.0
    if "llshear" in ing:
        ll = 0.6 + 0.4 * np.maximum(ramp(ing["llshear"], 8.0, 18.0), ramp(ing["v850"], 15.0, 28.0))
    return gauss(np.clip(inst * shear * lift * cin * moist * ll, 0, 1), 100)


HAZARDS = ("tor", "hail", "wind", "tor2", "hail2", "wind2")  # *2 = variants from the full-text paper review (30 Oct)
_OROG = None


def orography():
    """Static terrain height (m) on the 0.5 deg grid: GFS HGT:surface averaged over each cell."""
    global _OROG
    if _OROG is None:
        _OROG = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orog_0p5.npy")).astype(float)
    return _OROG


def lift_term(ing, dz12=None):
    """v2's omega lift term, scaled by v3's 12-h 500 hPa height-fall forcing when available."""
    omega = np.clip(gauss(ing["omega"], 150), -0.3, 0.3)
    ln = PROFILES["v2"]["lift_neutral"]
    lift = np.interp(omega, _LIFT_X, [1.10, 1.00, 0.80, ln, ln * 0.75, 0.20])
    if dz12 is not None:
        lift = lift * (0.75 + 0.5 * ramp(-gauss(dz12, 150), 30.0, 150.0))
    return lift


def freezing_level(f, ing):
    """0 C height above ground (m) from 700/500 hPa T and z (research/10_hazard_forecasting/02_hail.md 4.2)."""
    t7, t5 = f["t700"] - 273.15, f["t500"] - 273.15
    lr = np.maximum(ing["lr75"], 3.0)
    if "z700" in f:
        z7 = f["z700"]
    else:  # EPS members skip geopotential: hypsometric 700 hPa height from surface pressure
        z7 = orography() + _RD_G * (f["t700"] + 6.0) * np.log(np.maximum(f["sp"], 70100.0) / 70000.0)
    z5 = z7 + ing["dz75"]
    dt = np.where(np.abs(t7 - t5) < 1e-3, 1e-3, t7 - t5)
    z0 = np.where(t7 < 0, z7 + 1000.0 * t7 / lr, np.where(t5 < 0, z7 + (z5 - z7) * t7 / dt, z5 + 1000.0 * t5 / lr))
    return np.maximum(z0 - orography(), 0.0)


def tornado_env(ing, f, dz12=None, v=1):
    """EXPERIMENTAL tornado environment, version T0 (research/10_hazard_forecasting/01_tornado.md 4.1-4.2).

    Own base environment (CAPE credit saturating at 1000 J/kg, no lapse-rate bonus, deep shear 10->20 m/s,
    v3 lift, any-severe CIN) times the tornado factor T0 = K x M x C_tor x E_sb:
      K      low-level kinematics: max(supercell path = SRH(10 m-850 hPa, Bunkers-type motion) + |V850-V10|,
             0.8 x LLJ/QLCS path = |V850| x |V850-V10|)
      M      low-level moisture (cloud-base proxy) 0.2 + 0.8 ramp(q2, 9, 13 g/kg)
      C_tor  tornado CIN 0.4 + 0.6 ramp(CIN + 100, 0, 75) (NCEP; 1 for ECMWF)
      E_sb   surface-based share ramp(CAPE90 / CAPE180, 0.3, 0.7) where both exist (GFS, CAMs; else 1)
    """
    cape = np.clip(np.nan_to_num(ing["cape_v2"], nan=0.0), 0, None)
    cin = ing["cin_v2"]
    cin_sev = 1.0 if cin is None else ramp(200.0 + np.nan_to_num(cin, nan=0.0), 0, 150)
    s6 = ramp(ing["shear"], 10.0, 20.0)
    if v == 2:  # outbreaks run stronger deep shear (Anderson-Frey et al. 2018): keep adding credit to 30 m/s
        s6 = s6 * (0.8 + 0.2 * ramp(ing["shear"], 20.0, 30.0))
    base = ramp(cape, 0, 1000.0) ** 0.5 * s6 * lift_term(ing, dz12) * cin_sev
    if "u850" in f:
        u10, v10, u8, v8, u5, v5 = f["u10"], f["v10"], f["u850"], f["v850"], f["u500"], f["v500"]
        shu, shv = u5 - u10, v5 - v10
        sh = np.hypot(shu, shv) + 1e-6
        cu = (u10 + u8 + u5) / 3.0 + 7.5 * shv / sh  # 7.5 m/s to the right of the 10 m-500 hPa shear
        cv = (v10 + v8 + v5) / 3.0 - 7.5 * shu / sh
        srh = (u8 - cu) * (v10 - cv) - (u10 - cu) * (v8 - cv)
        if v == 2:  # second stage: outbreak SRH01 ~350 vs isolated ~160 (Anderson-Frey 2018); HSLC shear > 17 m/s is common (King 2017)
            k_sc = (0.5 * (0.75 * ramp(srh, 40.0, 220.0) + 0.25 * ramp(srh, 220.0, 400.0))
                    + 0.5 * (0.75 * ramp(ing["llshear"], 7.5, 17.5) + 0.25 * ramp(ing["llshear"], 17.5, 27.5)))
        else:
            k_sc = 0.5 * ramp(srh, 40.0, 220.0) + 0.5 * ramp(ing["llshear"], 7.5, 17.5)
        k_q = ramp(ing["v850"], 15.0, 25.0) * ramp(ing["llshear"], 10.0, 20.0)
        k = 0.1 + 0.9 * np.maximum(k_sc, 0.8 * k_q)
    else:
        k = 0.4
    m = 0.2 + 0.8 * ramp(ing["q2"], 9.0, 13.0)
    if cin is None:
        c_tor = 1.0
    elif v == 2:  # tornado yield per warned supercell by MLCIN (Davies 2004): 43 / 27 / 25 / 2 %
        c_tor = np.interp(-np.nan_to_num(cin, nan=0.0), [0, 25, 100, 140, 200], [1.0, 1.0, 0.6, 0.5, 0.1])
    else:
        c_tor = 0.4 + 0.6 * ramp(np.nan_to_num(cin, nan=0.0) + 100.0, 0, 75.0)
    e_sb = 1.0
    if "cape90" in f and "cape180" in f:
        c180 = np.nan_to_num(f["cape180"], nan=0.0)
        ratio = np.nan_to_num(f["cape90"], nan=0.0) / np.maximum(c180, 1.0)
        e_sb = np.where(c180 >= 100.0, 0.25 + 0.75 * ramp(ratio, 0.3, 0.7), 1.0)
    return gauss(np.clip(base * k * m * c_tor * e_sb, 0, 1), 100)


def hail_env(ing, f, v2, v=1):
    """EXPERIMENTAL hail environment, version A (research/10_hazard_forecasting/02_hail.md 4.1-4.3).

    max(v2 x H, 0.6 x E):
      H  = L x C x S x M_lo x M_hi, with
           L     mid-level lapse rate 0.2 + 0.8 ramp(LR75, 5.5, 7.5 C/km)
           C     most-unstable CAPE ramp(CAPE, 0.1 s, 0.75 s)^0.5 (s = 2000 NCEP 180-0, 2500 ECMWF MU)
           S     deep shear 0.35 + 0.65 ramp(|V500-V10|, 12, 25 m/s)
           M_lo  shallow / cold column 0.5 + 0.5 ramp(FZL, 1500, 2500 m AGL)
           M_hi  melting in warm, moist columns 1 - 0.5 ramp(FZL, 4200, 5200 m) ramp(PW, 30, 50 mm)
      E  = elevated hail north of fronts (GFS / CAMs: CAPE180 - CAPE90 buoyancy above the surface layer) x L x C
           x ramp(|V500-V850|, 12, 22 m/s) x M_lo x lift; no surface CIN or moisture term
    """
    s = 2000.0 if "cape180" in f else 2500.0
    cape_mu = np.clip(np.nan_to_num(ing["cape_cur"], nan=0.0), 0, None)
    lr = 0.2 + 0.8 * ramp(ing["lr75"], 5.5, 7.5)
    c = ramp(cape_mu, 0.1 * s, 0.75 * s) ** 0.5
    fzl = freezing_level(f, ing)
    m_lo = 0.5 + 0.5 * ramp(fzl, 1500.0, 2500.0)
    m_hi = 1.0 - 0.5 * ramp(fzl, 4200.0, 5200.0) * ramp(ing["pwat"], 30.0, 50.0)
    s_hi = 30.0 if v == 2 else 25.0  # v2: deep shear keeps helping larger hail to 30 m/s (Dennis & Kumjian 2017; tentative)
    h = gauss(np.clip(lr * c * (0.35 + 0.65 * ramp(ing["shear"], 12.0, s_hi)) * m_lo * m_hi, 0, 1), 100)
    env = v2 * h
    if "cape90" in f and "u850" in f:
        elev = ramp(np.nan_to_num(f["cape180"], nan=0.0) - np.nan_to_num(f["cape90"], nan=0.0), 250.0, 1000.0)
        s_el = ramp(np.hypot(f["u500"] - f["u850"], f["v500"] - f["v850"]), 12.0, 22.0)
        e = gauss(np.clip(elev * lr * c * s_el * m_lo * lift_term(ing), 0, 1), 100)
        env = np.maximum(env, 0.6 * e)
    return env


def wind_env(ing, f, dz12=None, v=1):
    """EXPERIMENTAL damaging-wind environment, version 1 (research/10_hazard_forecasting/03_wind.md 4.2, 4.5).

    Own base environment Ew = instability (CAPE credit saturating ~1000 J/kg, lapse-rate bonus) x lift x CIN
    x moisture -- no deep-shear cut-off, so weakly sheared derechos and pulse days are not zeroed -- times
    W1 = (0.2 + 0.8 max(A, B)) x Morg x Selev:
      A      strong flow brought down: max(ramp(|V850|, 12.5, 25), ramp(|mean(V10, V850, V500)|, 10, 20))
             x coupling 0.5 + 0.5 ramp(V10 . V850 / |V850|^2, 0.1, 0.4) (weak / opposed surface wind = decoupled)
      B      cold pool: ramp(CAPE, 500, 2500 ML | 750, 3000 best/MU)^0.5 x downdraft potential
             0.5 ramp(LR75, 6, 7.5) + 0.5 ramp(70 - mean RH700/500, 0, 40)
      Morg   organisation 0.6 + 0.4 ramp(|V500 - V10|, 8, 20)
      Selev  elevated storms 1 - 0.5 ramp(1 - CAPE90/CAPE180, 0.5, 0.9) where both exist (GFS, CAMs)
    """
    cape = np.clip(np.nan_to_num(ing["cape_v2"], nan=0.0), 0, None)
    inst = np.clip(ramp(cape, 0, 1000.0) ** 0.5 * (1.0 + 0.5 * ramp(ing["lr75"], 6.0, 7.5)), 0, 1)
    cin = 1.0 if ing["cin_v2"] is None else ramp(200.0 + np.nan_to_num(ing["cin_v2"], nan=0.0), 0, 150)
    moist = 0.5 + 0.5 * ramp(ing["q2"], *PROFILES["v2"]["moist"])
    ew = inst * lift_term(ing, dz12) * cin * moist
    if "u850" in f:
        u10, v10, u8, v8 = f["u10"], f["v10"], f["u850"], f["v850"]
        um, vm = (u10 + u8 + f["u500"]) / 3.0, (v10 + v8 + f["v500"]) / 3.0
        r10 = (u10 * u8 + v10 * v8) / np.maximum(u8 ** 2 + v8 ** 2, 1.0)
        a = np.maximum(ramp(ing["v850"], 12.5, 25.0), ramp(np.hypot(um, vm), 10.0, 20.0)) * (0.5 + 0.5 * ramp(r10, 0.1, 0.4))
        if v == 2:  # big limited-instability wind days: moist low levels under strong 500 hPa ascent (Campbell et al. 2024)
            a = a * (0.6 + 0.4 * ramp(ing["q2"], 10.0, 14.0) * ramp(-gauss(f["w500"], 150), 0.1, 0.3))
    else:
        a = 0.0
    rhm = 0.5 * (f["rh700"] + f["rh500"]) if "rh700" in f and "rh500" in f else 50.0
    d = 0.5 * ramp(ing["lr75"], 6.0, 7.5) + 0.5 * ramp(70.0 - rhm, 0, 40.0)
    lo, hi = (500.0, 2500.0) if "cape90" in f else (750.0, 3000.0)
    b = ramp(cape, lo, hi) ** 0.5 * d
    morg = 0.6 + 0.4 * ramp(ing["shear"], 8.0, 20.0)
    selev = 1.0
    if "cape90" in f and "cape180" in f:
        c180 = np.nan_to_num(f["cape180"], nan=0.0)
        frac = 1.0 - np.nan_to_num(f["cape90"], nan=0.0) / np.maximum(c180, 1.0)
        selev = np.where(c180 >= 250.0, 1.0 - 0.5 * ramp(frac, 0.5, 0.9), 1.0)
    w = (0.2 + 0.8 * np.maximum(a, b)) * morg * selev
    return gauss(np.clip(ew * w, 0, 1), 100)


def hazard_scores(ing, f, v2, dz12=None):
    """Tornado / hail / wind environment scores for one member and valid time (calibrate_hazards.py)."""
    return {"tor": tornado_env(ing, f, dz12), "hail": hail_env(ing, f, v2), "wind": wind_env(ing, f, dz12),
            "tor2": tornado_env(ing, f, dz12, 2), "hail2": hail_env(ing, f, v2, 2), "wind2": wind_env(ing, f, dz12, 2)}


def trimmed_mean(stack, frac=0.1):
    """Mean after dropping the top and bottom `frac` of members at each point."""
    n = stack.shape[0]
    k = int(np.floor(frac * n))
    if k == 0:
        return stack.mean(axis=0)
    s = np.sort(stack, axis=0)
    return s[k:n - k].mean(axis=0)


def pool_combine(stack, scheme, deterministic):
    """Combine member/run scores from a pool into one field."""
    if scheme == "current":
        return gauss(stack.mean(axis=0), 150)
    # v2: robust combination -- median of runs for deterministic models,
    # 10% trimmed mean of all members for ensembles
    return np.median(stack, axis=0) if deterministic else trimmed_mean(stack, 0.1)


# Prior blend weights (research/02_models/00_synthesis.md 0.4): ECMWF family 0.6, NCEP 0.4;
# deterministic runs are minor members within each family.
BLEND_WEIGHTS = {"hres": 0.12, "eps": 0.48, "gfs": 0.08, "gefs": 0.32}


def weights(lead=None):
    """Global weights, plus CAM weights at Day 2-1 (globals scaled down; config.CAM_BLEND)."""
    if lead not in CAM_BLEND:
        return dict(BLEND_WEIGHTS)
    g, cam = CAM_BLEND[lead]
    w = {m: g * v for m, v in BLEND_WEIGHTS.items()}
    w.update(cam)
    return w


def blend(fields, lead=None):
    """fields: {model: 2-D array or None}. Renormalises over available models."""
    w = weights(lead)
    use = {m: f for m, f in fields.items() if f is not None and m in w}
    tot = sum(w[m] for m in use)
    if tot == 0:
        return None
    return sum(w[m] * f for m, f in use.items()) / tot


_ = GRID_LATS  # grid defined in config; imported for clarity of dependency

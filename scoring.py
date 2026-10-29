"""Ingredient scoring on the common 0.5 deg grid.

Two schemes are implemented side by side:

* ``current`` -- a faithful port of GFS/gfs-ensemble.py (score_grid + its
  250/300/150 km smoothing chain), applied to every model.
* ``v2`` -- the research-based prototype (research/01_severe_ingredients/00_synthesis.md
  section 0.4, Tier 2). It is uncalibrated; outputs are an index, not a probability.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from config import GRID_LATS

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
    }
    if model in ("gfs", "gefs"):
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


def score_v2(ing, fh, parts=False):
    cape = np.clip(np.nan_to_num(ing["cape_v2"], nan=0.0), 0, None)
    c_term = np.clip(cape / ing["cape_scale"], 0, 1) ** 0.6
    inst = np.clip(c_term * (1.0 + 0.5 * ramp(ing["lr75"], 6.0, 7.5)), 0, 1)
    shear = ramp(ing["shear"], 10.0, 20.0)
    omega = np.clip(gauss(ing["omega"], 150), -0.3, 0.3)
    lift = np.interp(omega, _LIFT_X, _LIFT_Y)
    cin = 1.0 if ing["cin_v2"] is None else ramp(200.0 + np.nan_to_num(ing["cin_v2"], nan=0.0), 0, 150)
    moist = 0.5 + 0.5 * ramp(ing["q2"], 8.0, 13.0)
    score = gauss(np.clip(inst * shear * lift * cin * moist, 0, 1), 100)
    if parts:
        return score, {"inst": inst, "shear": shear, "lift": lift, "cin": cin, "moist": moist}
    return score


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


def blend(fields):
    """fields: {model: 2-D array or None}. Renormalises over available models."""
    tot = sum(BLEND_WEIGHTS[m] for m, f in fields.items() if f is not None)
    if tot == 0:
        return None
    return sum(BLEND_WEIGHTS[m] * f for m, f in fields.items() if f is not None) / tot


_ = GRID_LATS  # grid defined in config; imported for clarity of dependency

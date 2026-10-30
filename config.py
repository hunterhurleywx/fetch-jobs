"""Case configuration: models, fields, run pools and the common analysis grid."""
import os
from datetime import datetime, timedelta

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Field cache lives on D: (outside OneDrive, 240 GB free). Raw GRIB is never kept:
# each message is decoded, mapped to the 0.5 deg grid and saved as compressed npz.
CACHE_DIR = os.environ.get("POGS_CACHE", r"D:\pogs_cache")
CASES_DIR = os.environ.get("POGS_CASES", os.path.join(PROJECT_DIR, "cases"))

# Common analysis grid (0.5 deg) that every model is mapped onto
GRID_LATS = np.arange(20.0, 55.0 + 0.01, 0.5)
GRID_LONS = np.arange(-130.0, -60.0 + 0.01, 0.5)

VIEWS = {
    "conus": [-125, -66, 23, 50],
    "central": [-104, -78, 29, 46],  # Plains to Ohio Valley
}

# Valid-time offsets (h) from 12Z of the convective day: 18Z, 21Z, 00Z, 03Z, 06Z
VALID_OFFSETS = (6, 9, 12, 15, 18)
POOL_CYCLES = 6  # ~36 h of 6-hourly cycles per lead day

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
GFS_FIELDS = {
    "cape180": ("CAPE", "180-0 mb above ground"), "cin180": ("CIN", "180-0 mb above ground"),
    "cape90": ("CAPE", "90-0 mb above ground"), "cin90": ("CIN", "90-0 mb above ground"),
    "z500": ("HGT", "500 mb"), "t500": ("TMP", "500 mb"), "w500": ("VVEL", "500 mb"),
    "u500": ("UGRD", "500 mb"), "v500": ("VGRD", "500 mb"),
    "z700": ("HGT", "700 mb"), "t700": ("TMP", "700 mb"), "w700": ("VVEL", "700 mb"),
    "u10": ("UGRD", "10 m above ground"), "v10": ("VGRD", "10 m above ground"),
    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"),
    "q2": ("SPFH", "2 m above ground"),
}
GEFS_FIELDS_A = {
    "cape180": ("CAPE", "180-0 mb above ground"), "cin180": ("CIN", "180-0 mb above ground"),
    "z500": ("HGT", "500 mb"), "t500": ("TMP", "500 mb"),
    "u500": ("UGRD", "500 mb"), "v500": ("VGRD", "500 mb"),
    "z700": ("HGT", "700 mb"), "t700": ("TMP", "700 mb"),
    "u10": ("UGRD", "10 m above ground"), "v10": ("VGRD", "10 m above ground"),
    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"),
}
GEFS_FIELDS_B = {
    "w500": ("VVEL", "500 mb"), "w700": ("VVEL", "700 mb"), "q2": ("SPFH", "2 m above ground"),
}
# ECMWF open data: key -> (param, levtype, levelist)
EC_FIELDS = {
    "capemu": ("mucape", "sfc", None), "d2m": ("2d", "sfc", None), "sp": ("sp", "sfc", None),
    "u10": ("10u", "sfc", None), "v10": ("10v", "sfc", None), "pwat": ("tcwv", "sfc", None),
    "z500": ("gh", "pl", "500"), "t500": ("t", "pl", "500"), "w500": ("w", "pl", "500"),
    "u500": ("u", "pl", "500"), "v500": ("v", "pl", "500"),
    "z700": ("gh", "pl", "700"), "t700": ("t", "pl", "700"), "w700": ("w", "pl", "700"),
}

# ---------------------------------------------------------------------------
# Full-column "sounding" variants of the deterministic models (0.25 deg grid)
# ---------------------------------------------------------------------------
SND_LATS = np.arange(20.0, 55.0 + 0.01, 0.25)
SND_LONS = np.arange(-130.0, -60.0 + 0.01, 0.25)
GFS_SND_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
                  450, 400, 350, 300, 250, 200, 150, 100]
EC_SND_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
GFS_SND_FIELDS = {f"{k}{lev}": (v, f"{lev} mb") for lev in GFS_SND_LEVELS
                  for k, v in (("z", "HGT"), ("t", "TMP"), ("q", "SPFH"), ("u", "UGRD"), ("v", "VGRD"), ("w", "VVEL"))}
GFS_SND_FIELDS.update({
    "sp": ("PRES", "surface"), "zs": ("HGT", "surface"), "t2m": ("TMP", "2 m above ground"),
    "q2": ("SPFH", "2 m above ground"), "u10": ("UGRD", "10 m above ground"), "v10": ("VGRD", "10 m above ground"),
})
EC_SND_FIELDS = {f"{k}{lev}": (p, "pl", str(lev)) for lev in EC_SND_LEVELS
                 for k, p in (("z", "gh"), ("t", "t"), ("q", "q"), ("u", "u"), ("v", "v"), ("w", "w"))}
EC_SND_FIELDS.update({
    "sp": ("sp", "sfc", None), "t2m": ("2t", "sfc", None), "d2m": ("2d", "sfc", None),
    "u10": ("10u", "sfc", None), "v10": ("10v", "sfc", None),
})

# ---------------------------------------------------------------------------
# "Extras" for the convective-initiation layer (separate cache: <model>_x)
#   NCEP accumulations: "acc0" = run total (0-fh), "accb" = shortest bucket ending at fh
# ---------------------------------------------------------------------------
GFS_X_FIELDS = {"apcp": ("APCP", "surface", "acc0"), "acpcp": ("ACPCP", "surface", "acc0"),
                "u850": ("UGRD", "850 mb"), "v850": ("VGRD", "850 mb"),
                "rh700": ("RH", "700 mb"), "rh500": ("RH", "500 mb")}
GEFS_X_FIELDS_A = {"apcpb": ("APCP", "surface", "accb"), "u850": ("UGRD", "850 mb"), "v850": ("VGRD", "850 mb"),
                   "rh700": ("RH", "700 mb"), "rh500": ("RH", "500 mb")}
GEFS_X_FIELDS_B = {"acpcpb": ("ACPCP", "surface", "accb")}
EC_X_FIELDS = {"tp": ("tp", "sfc", None), "cp": ("cp", "sfc", None), "u850": ("u", "pl", "850"), "v850": ("v", "pl", "850"),
               "rh700": ("r", "pl", "700"), "rh500": ("r", "pl", "500")}
# EPS 500/700 hPa geopotential, fetched separately ("eps_z") when height maps are needed
EC_Z_FIELDS = {"z500": ("gh", "pl", "500"), "z700": ("gh", "pl", "700")}
# Upper-air winds (separate cache "<model>_ua", so already-cached cycles backfill cleanly):
# 300 hPa jet / diffluence and 700 hPa steering and mid-level flow
NCEP_UA_FIELDS = {"u300": ("UGRD", "300 mb"), "v300": ("VGRD", "300 mb"), "u700": ("UGRD", "700 mb"), "v700": ("VGRD", "700 mb")}
EC_UA_FIELDS = {"u300": ("u", "pl", "300"), "v300": ("v", "pl", "300"), "u700": ("u", "pl", "700"), "v700": ("v", "pl", "700")}
# Ensemble upper air (separate cache "<model>_pl"): height, temperature, RH and wind at 7 key levels, plus
# 2 m temperature / moisture for cloud base. GFS and HRES carry the full column instead ("gfs_snd", "hres_snd").
ENS_PL_LEVELS = [1000, 925, 850, 700, 500, 300, 250]
GEFS_PL_FIELDS_A = {f"{k}{lev}": (v, f"{lev} mb") for lev in ENS_PL_LEVELS
                    for k, v in (("z", "HGT"), ("t", "TMP"), ("rh", "RH"), ("u", "UGRD"), ("v", "VGRD"))
                    if not (lev == 300 and k in ("t", "rh"))}  # 300 mb T and RH live in the b file
GEFS_PL_FIELDS_A.update({"t2m": ("TMP", "2 m above ground"), "rh2m": ("RH", "2 m above ground")})
GEFS_PL_FIELDS_B = {"t300": ("TMP", "300 mb"), "rh300": ("RH", "300 mb")}
EC_PL_FIELDS = {f"{k}{lev}": (p, "pl", str(lev)) for lev in ENS_PL_LEVELS
                for k, p in (("z", "gh"), ("t", "t"), ("rh", "r"), ("u", "u"), ("v", "v"))}
EC_PL_FIELDS.update({"t2m": ("2t", "sfc", None), "d2m": ("2d", "sfc", None)})
OPTIONAL_X = {"apcp", "acpcp", "apcpb", "acpcpb", "tp", "cp"}  # cp: ECMWF open data from 20 Oct 2026  # absent at some steps (e.g. f000)

MODELS = ["gfs", "gefs", "hres", "eps"]
MODEL_LABELS = {"gfs": "GFS", "gefs": "GEFS", "hres": "ECMWF HRES", "eps": "ECMWF EPS", "blend": "Blend"}
GEFS_MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]
# Perturbed members only (the control is the HRES configuration). Full 50 by default now
# that downloads run in the cloud (remote_fetch.py); POGS_EPS_N=25 halves a home download.
EPS_MEMBERS = [str(i) for i in range(1, int(os.environ.get("POGS_EPS_N", 50)) + 1)]
# EPS members skip geopotential: the 700-500 hPa thickness is computed hypsometrically
EC_FIELDS_EPS = {k: v for k, v in EC_FIELDS.items() if k not in ("z500", "z700")}


# ---------------------------------------------------------------------------
# Convection-allowing / rapid-refresh models (Day 2-1). Native Lambert grids are aggregated onto
# the 0.5 deg grid: mean for environment fields, max for REFC / UH (a storm anywhere in the cell).
# key -> (VAR, level prefix, kind); kind "inst" = "<fh> hour fcst", "max" = "<fh-1>-<fh> hour max fcst"
# ---------------------------------------------------------------------------
CAM_MODELS = ["hrrr", "nam", "rap", "nam12"]  # nam = NAM 3 km nest; nam12 = NAM 12 km (awphys, to 84 h)
CAM_FIELDS = {
    "cape90": ("CAPE", "90-0 mb above ground", "inst"), "cin90": ("CIN", "90-0 mb above ground", "inst"),
    "cape180": ("CAPE", "180-0 mb above ground", "inst"), "cin180": ("CIN", "180-0 mb above ground", "inst"),
    "z500": ("HGT", "500 mb", "inst"), "z700": ("HGT", "700 mb", "inst"),
    "t500": ("TMP", "500 mb", "inst"), "t700": ("TMP", "700 mb", "inst"),
    "w500": ("VVEL", "500 mb", "inst"), "w700": ("VVEL", "700 mb", "inst"),
    "u500": ("UGRD", "500 mb", "inst"), "v500": ("VGRD", "500 mb", "inst"),
    "u10": ("UGRD", "10 m above ground", "inst"), "v10": ("VGRD", "10 m above ground", "inst"),
    "u850": ("UGRD", "850 mb", "inst"), "v850": ("VGRD", "850 mb", "inst"),
    "pwat": ("PWAT", "entire atmosphere", "inst"), "q2": ("SPFH", "2 m above ground", "inst"),
    "rh700": ("RH", "700 mb", "inst"), "rh500": ("RH", "500 mb", "inst"),
}
# hourly storm signal, 18Z-06Z
CAM_CI_FIELDS = {"refc": ("REFC", "entire atmosphere", "inst"), "uh25": ("MXUPHL", "5000-2000 m above ground", "max")}
CAM_OPTIONAL = {"uh25"}  # RAP has no UH
CAM_MAX_KEYS = {"refc", "uh25"}
# Day 2-1 blend: globals keep their relative weights, scaled down; CAM weights added.
# Hunter, 30 Oct 2026: Days 2 and 1 drop the global models; only the CAMs and the NAM 12 km.
# Day 3 pool (the newest runs before SPC's Day 3): globals plus a small NAM 12 km share (Hunter, 30 Oct: the 12Z NAM
# runs bullish; raise its weight only once the 18Z / 00Z runs show it is any good).
CAM_BLEND = {3: (0.95, {"nam12": 0.05}),
             2: (0.0, {"nam": 0.35, "hrrr": 0.25, "nam12": 0.25, "rap": 0.15}),
             1: (0.0, {"hrrr": 0.40, "nam": 0.25, "rap": 0.20, "nam12": 0.15})}


def cam_pools(conv_day, lead):
    """CAM runs available by SPC's ~06Z issuance (cutoff 00Z of the issue day) that reach the window."""
    cut = datetime(conv_day.year, conv_day.month, conv_day.day) - timedelta(days=lead - 1)
    h = lambda k: cut - timedelta(hours=k)
    if lead == 1:
        pools = {"hrrr": [h(18), h(12), h(6), h(0)], "nam": [h(18), h(12), h(6), h(0)],
                 "rap": [h(21), h(15), h(9), h(3)], "nam12": [h(18), h(12), h(6), h(0)]}
        # Real-time override, e.g. POGS_CAM_RUNS="hrrr:2026102919,2026102920;rap:2026102923"
        # replaces those models' Day 1 runs with the newest ones (nowcast-style)
        for part in filter(None, os.environ.get("POGS_CAM_RUNS", "").split(";")):
            m, runs = part.split(":")
            pools[m.strip()] = [datetime.strptime(r.strip(), "%Y%m%d%H") for r in runs.split(",")]
        return pools
    if lead == 2:
        return {"hrrr": [h(6), h(0)], "nam": [h(12), h(6), h(0)], "rap": [h(9), h(3)], "nam12": [h(12), h(6), h(0)]}
    if lead == 3:  # NAM 12 km runs to 84 h, so its 12Z / 18Z / 00Z runs before Day 3 issuance reach the window
        return {"nam12": [h(12), h(6), h(0)]}
    return {}


def fh_available(model, init, fh):
    """Whether a model run has output at forecast hour fh."""
    if fh <= 0:
        return False
    model = model.replace("_snd", "").replace("_x", "").replace("_z", "").replace("_ci", "").replace("_ua", "").replace("_pl", "")
    if model == "hrrr":
        return fh <= 18 or (init.hour % 6 == 0 and fh <= 48)
    if model == "nam":
        return fh <= 60
    if model == "nam12":  # awphys: hourly to 36 h, then 3-hourly to 84 h
        return fh <= 84 and (fh <= 36 or fh % 3 == 0)
    if model == "rap":
        return fh <= 21 or (init.hour % 6 == 3 and fh <= 51)
    if model == "gfs":
        return fh <= 384 and (fh <= 120 or fh % 3 == 0)
    if model == "gefs":
        return fh <= 384 and fh % 3 == 0 and (fh <= 240 or fh % 6 == 0)
    if model in ("hres", "eps"):
        max_fh = 360 if init.hour in (0, 12) else 144
        return fh <= max_fh and fh % 3 == 0 and (fh <= 144 or fh % 6 == 0)
    raise ValueError(model)


def case_setup(conv_day, leads=(7, 6, 5)):
    """Valid times and SPC-numbered run pools for a convective day.

    SPC Day N for conv_day is issued N-1 days earlier (~08Z). The newest cycle
    available then is 00Z of the issue day; the pool is the POOL_CYCLES cycles
    ending there.
    """
    day12 = datetime(conv_day.year, conv_day.month, conv_day.day, 12)
    valid = [day12 + timedelta(hours=h) for h in VALID_OFFSETS]
    pools = {}
    for d in leads:
        issue = datetime(conv_day.year, conv_day.month, conv_day.day) - timedelta(days=d - 1)
        pools[d] = [issue - timedelta(hours=6 * k) for k in range(POOL_CYCLES)][::-1]
    return valid, pools

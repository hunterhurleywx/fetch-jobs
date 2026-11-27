"""Pletcher et al. (2026) CONUS SLR random forest + linear regression applied to our feature table (cloud only).

  python snow/slr_rf.py --repo /opt/utahrfslr --out DIR

Runs in its own venv (aws/snow_pipeline.sh picks a scikit-learn version the RF pickle loads with). Reads DIR/slr_features.npz (X: rows x 27 in
model_keys order SPD03K..24K, T03K..24K, R03K..24K, elev, lat, lon; our T in K, RH %, wind m/s, lon negative), checks
the training units against the repo's HRRR testing set, converts if needed, writes DIR/slr_rf.npz {rf, lr}.
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

f = np.load(os.path.join(a.out, "slr_features.npz"))
X = f["X"].astype(np.float64)
print("feature rows:", X.shape, flush=True)
keys = list(np.load(os.path.join(a.repo, "models/rf/rf_slr_model_keys.npy"), allow_pickle=True))
ours = [f"SPD{h:02d}K" for h in range(3, 25, 3)] + [f"T{h:02d}K" for h in range(3, 25, 3)] + \
       [f"R{h:02d}K" for h in range(3, 25, 3)] + ["elev", "lat", "lon"]
assert keys == ours, f"key order differs: {keys}"
df = pd.DataFrame(X, columns=ours)

# Units / conventions from the repo's own HRRR testing set (fall back to K, %, negative lon)
info = {}
try:
    t = pd.read_pickle(os.path.join(a.repo, "data/testing/HRRR_testing_dataset_clean.pickle"))
    for c in ("T03K", "T24K", "R03K", "SPD03K", "lon", "lat", "elev"):
        if c in t:
            info[c] = [float(np.nanmin(t[c])), float(np.nanmean(t[c])), float(np.nanmax(t[c]))]
    print("testing-set ranges:", json.dumps(info), flush=True)
except Exception as e:  # pragma: no cover
    print("could not read testing set:", repr(e)[:200], flush=True)
conv = []
if "T03K" in info and info["T03K"][1] < 100:
    for c in ours:
        if c.startswith("T"):
            df[c] = df[c] - 273.15
    conv.append("T K->C")
if "R03K" in info and info["R03K"][2] <= 1.5:
    for c in ours:
        if c.startswith("R"):
            df[c] = df[c] / 100.0
    conv.append("RH %->fraction")
if "lon" in info and info["lon"][0] > 0:
    df["lon"] = df["lon"] % 360
    conv.append("lon -> 0..360")
print("conversions:", conv or "none", flush=True)
print("our ranges:", {c: [round(float(df[c].min()), 2), round(float(df[c].max()), 2)] for c in ("T03K", "R03K", "SPD03K", "lon", "elev")}, flush=True)

out = {}
for meth in ("lr", "rf"):
    try:
        mdl = pickle.load(open(os.path.join(a.repo, f"models/{meth}/{meth}_slr_model.pkl"), "rb"))
        if hasattr(mdl, "n_jobs"):
            mdl.n_jobs = -1
        kk = list(np.load(os.path.join(a.repo, f"models/{meth}/{meth}_slr_model_keys.npy"), allow_pickle=True))
        pred = np.empty(len(df), np.float32)
        for s in range(0, len(df), 1_000_000):
            pred[s:s + 1_000_000] = mdl.predict(df.iloc[s:s + 1_000_000][kk])
        out[meth] = np.clip(pred, 1, 40)
        print(f"{meth}: SLR mean {pred.mean():.1f}, p10 {np.percentile(pred, 10):.1f}, p90 {np.percentile(pred, 90):.1f}", flush=True)
    except Exception as e:
        print(f"{meth} failed:", repr(e)[:300], flush=True)
np.savez_compressed(os.path.join(a.out, "slr_rf.npz"), **out)
json.dump({"testing_ranges": info, "conversions": conv}, open(os.path.join(a.out, "slr_rf_info.json"), "w"), indent=1)

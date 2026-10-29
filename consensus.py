"""Scenario-consistency pooling: weight each model-cycle map, point by point, by whether its
solution is shared by the OTHER model family and whether it has persisted.

Hunter's rule (research/09_run_weighting/01_agreement_and_convergence.md): don't highlight on
one good run; trust a solution once several runs AND both families (NCEP: GFS/GEFS,
ECMWF: HRES/EPS) settle on it, and watch for one family converging on the other.

At every grid point x, for model m and cycle c (soft "favourable" mask s = ramp(F, LO, HI)):
  O(x)      other family's support: mean of s over its models' newest NEWEST cycles
  A_c(x)    agreement of this run with the other family: 1 - |s_c(x) - O(x)|
  flip_c(x) area this run newly made favourable: s_c(x) * (1 - s_prev(x))
  w_c(x)    = rec_c * (FLOOR + (1 - FLOOR) * A_c(x)) * (1 - 0.5 * flip_c(x) * (1 - O(x)))
The model's pooled map is sum_c w_c F_c / sum_c w_c. When the families agree everywhere this
reduces to the recency-weighted (or equal) pool; it only acts where they split.
"""
import numpy as np

FAMILIES = {"ncep": ("gfs", "gefs"), "ec": ("hres", "eps")}
OTHER = {"gfs": "ec", "gefs": "ec", "hres": "ncep", "eps": "ncep"}
LO, HI = 0.15, 0.35   # index range over which a point becomes "favourable" in a run
FLOOR = 0.2           # weight kept by a run that fully disagrees with the other family
NEWEST = 2            # cycles of the other family that define its current solution


def soft(F):
    return np.clip((F - LO) / (HI - LO), 0.0, 1.0)


def family_support(cyc, fam):
    """Mean favourable-mask over the family's newest NEWEST cycles, all its models."""
    masks = []
    for m in FAMILIES[fam]:
        for i in sorted(cyc.get(m, {}))[-NEWEST:]:
            masks.append(soft(cyc[m][i]))
    return np.mean(masks, axis=0) if masks else None


def scenario_pool(cyc, half_life=None):
    """cyc: {model: {init datetime: field}} -> {model: pooled field} with point-wise weights.

    half_life: hours for recency weighting (None = no recency)."""
    newest = max(i for m in cyc for i in cyc[m]) if any(cyc.values()) else None
    support = {fam: family_support(cyc, fam) for fam in FAMILIES}
    out = {}
    for m, runs in cyc.items():
        if not runs:
            out[m] = None
            continue
        inits = sorted(runs)
        O = support[OTHER[m]]
        num = den = 0.0
        prev = None
        for i in inits:
            F = runs[i]
            s = soft(F)
            rec = 1.0 if half_life is None else 0.5 ** ((newest - i).total_seconds() / 3600 / half_life)
            if O is None:
                w = rec * np.ones_like(F)
            else:
                agree = 1.0 - np.abs(s - O)
                flip = s * (1.0 - prev) if prev is not None else np.zeros_like(F)
                w = rec * (FLOOR + (1 - FLOOR) * agree) * (1.0 - 0.5 * flip * (1.0 - O))
            num = num + w * F
            den = den + w
            prev = s
        out[m] = num / np.maximum(den, 1e-9)
    return out


def agreement_gate(field, cyc, strength=0.4):
    """Damp the blended field where the families split on the favourable area.

    A(x) = 1 - |S_ncep(x) - S_ec(x)| (newest-cycle support). Where both agree (both favourable
    or both not) the field is unchanged; where one family alone supports it, it is scaled by
    (1 - strength). The "predictability too low" behaviour, as a map."""
    s = support_summary(cyc)
    if "agreement" not in s:
        return field
    return field * (1.0 - strength * (1.0 - s["agreement"]))


def support_summary(cyc):
    """Per-family support maps (newest cycles), and where they agree: for plots/diagnostics."""
    s = {fam: family_support(cyc, fam) for fam in FAMILIES}
    if s["ncep"] is not None and s["ec"] is not None:
        s["agreement"] = 1.0 - np.abs(s["ncep"] - s["ec"])
    return s

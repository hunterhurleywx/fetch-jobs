"""Shared velocity unfolding for RadarHub sweep loops (wind_loop.py, rh_mob_case.py).

Unfolds a whole LOOP, not each frame on its own (2026-11-19: frame-by-frame
unfolding let a cut-off echo region land on a different fold from one frame to
the next, so the loop flashed between inbound and outbound -- Hunter deleted the
7/5 case for it; rapid flashing is also a photosensitivity risk).

  1. every sweep: Py-ART region-based dealiasing (noise / texture masked)
  2. a SEED frame (the one with the most valid velocity gates, i.e. the best
     connected echo) has each connected region re-folded toward a uniform
     reference wind (the echo motion)
  3. walking forward and backward from the seed, each region of the next frame
     takes the whole-fold shift that best matches the frame before it (median
     over the gates both frames have); a region with no overlap falls back to
     the reference wind
  4. flicker(): share of overlapping gates whose value jumps by > 1.5 Nyquist
     between consecutive frames -- reported per frame pair; publish only when
     the worst pair is under FLICKER_MAX
"""
import os
import numpy as np
os.environ.setdefault('PYART_QUIET', '1')
import pyart
from scipy import ndimage

FLICKER_MAX = 0.03          # worst consecutive-frame jump share allowed before publishing


def dealias(Z, V, ghz):
    """Py-ART region-based unfolding of one sweep pair -> (masked field, nyquist)."""
    rays, gates, gw, el, az = Z['rays'], Z['gates'], Z['gw'], Z['el'], Z['az']
    ref = np.ma.masked_equal(Z['u8'], 0).astype(np.float32); ref = (ref - 64.0) / 2.0
    vel = np.ma.masked_equal(V['u8'], 0).astype(np.float32); vel = (vel - 128.0) / 2.0
    va = V['info']['prf'] * (299792458.0 / (ghz * 1e9)) / 4.0
    radar = pyart.testing.make_empty_ppi_radar(gates, rays, 1)
    radar.range['data'] = (np.arange(gates) + 0.5) * gw * 1000.0
    radar.azimuth['data'] = az.astype(np.float32)
    radar.elevation['data'] = np.full(rays, el, np.float32)
    radar.fixed_angle['data'] = np.array([el], np.float32)
    radar.latitude['data'] = np.array([Z['lat0']]); radar.longitude['data'] = np.array([Z['lon0']])
    radar.add_field('velocity', {'data': vel.copy(), 'units': 'm/s'})
    radar.add_field('reflectivity', {'data': ref.copy(), 'units': 'dBZ'})
    radar.instrument_parameters = {'nyquist_velocity': {'data': np.full(rays, va, np.float32)}}
    radar.add_field('tex', pyart.retrieve.calculate_velocity_texture(radar, vel_field='velocity', nyq=va))
    gf = pyart.filters.GateFilter(radar)
    if os.environ.get('RH_VEL_MASK') == 'loose':
        # WSR-88D (Preloaded Data): keep EVERY gate NOAA sends (like RadarScope) and only unfold them.
        # The X-band mask below (and even a loose texture cut) blanked the tornado's own core, where
        # the shear makes the texture highest (Elkhorn 2024, Hunter's RadarScope comparison 2026-11-22).
        gf.exclude_masked('velocity')
    else:
        gf.exclude_masked('velocity'); gf.exclude_masked('reflectivity')
        gf.exclude_below('reflectivity', 5.0); gf.exclude_above('tex', max(3.0, 0.4 * va))
        gf = pyart.correct.despeckle_field(radar, 'velocity', gatefilter=gf, size=20)
    dv = pyart.correct.dealias_region_based(radar, vel_field='velocity', nyquist_vel=va, gatefilter=gf,
                                            keep_original=False, centered=True)
    unf = np.ma.masked_where(gf.gate_excluded | np.ma.getmaskarray(dv['data']), dv['data'])
    return unf, va


def _regions(unf):
    valid = ~np.ma.getmaskarray(unf)
    lab, _ = ndimage.label(valid, structure=np.ones((3, 3)))
    for g in np.nonzero(valid[0] & valid[-1])[0]:            # join across the 0/360 seam
        l0, l1 = lab[0, g], lab[-1, g]
        if l0 != l1: lab[lab == l1] = l0
    return lab


def _refold(unf, va, target, min_overlap=150):
    """Shift each connected region by whole folds toward `target` (array, NaN = unknown)."""
    data = unf.filled(np.nan); itv = 2 * va
    lab = _regions(unf)
    for l in np.unique(lab[lab > 0]):
        m = lab == l
        if m.sum() < 30: continue
        both = m & np.isfinite(target)
        if both.sum() < min_overlap: continue
        k = int(np.round(np.nanmedian(target[both] - data[both]) / itv))
        if k: data[m] += k * itv
    return np.ma.masked_invalid(data)


def _same_grid(a, b):
    return (a['rays'] == b['rays'] and a['gates'] == b['gates'] and abs(a['gw'] - b['gw']) < 1e-6
            and abs(a['lat0'] - b['lat0']) < 0.005 and abs(a['lon0'] - b['lon0']) < 0.005)


def _ref_field(Z, ref_uv):
    azr = np.radians(Z['az'])[:, None]
    return (np.cos(np.radians(Z['el'])) * (ref_uv[0] * np.sin(azr) + ref_uv[1] * np.cos(azr))
            * np.ones((1, Z['gates']))).astype(np.float64)


def _on_grid(src_unf, src, dst):
    """src field resampled onto dst's ray order (nearest azimuth); NaN where unknown."""
    if not _same_grid(src, dst): return None
    o = np.argsort(src['az'])
    idx = np.searchsorted(src['az'][o], dst['az']) % len(o)
    return src_unf.filled(np.nan)[o][idx]


def _smooth_nan(a, size=9):
    """NaN-aware box mean (a reference field must be smooth: shear lines move between frames)."""
    v = np.isfinite(a)
    s = ndimage.uniform_filter(np.where(v, a, 0.0), size=size, mode='wrap')
    w = ndimage.uniform_filter(v.astype(float), size=size, mode='wrap')
    out = np.full(a.shape, np.nan)
    ok = w > 0.15
    out[ok] = s[ok] / w[ok]
    return out


def _advect(prev_field, P, C, ref_uv, dt_s):
    """prev frame's field sampled at each gate of frame C, moved along the echo motion."""
    if prev_field is None: return None
    if abs(P['gw'] - C['gw']) > 1e-6 or abs(P['lat0'] - C['lat0']) > 0.005 or abs(P['lon0'] - C['lon0']) > 0.005:
        return None
    rc = (np.arange(C['gates']) + 0.5) * C['gw'] * np.cos(np.radians(C['el']))
    A, R = np.meshgrid(np.radians(C['az']), rc, indexing='ij')
    x = R * np.sin(A) - ref_uv[0] * dt_s / 1000.0          # where this air was dt_s ago (km)
    y = R * np.cos(A) - ref_uv[1] * dt_s / 1000.0
    o = np.argsort(P['az']); pa = P['az'][o]
    az_p = np.degrees(np.arctan2(x, y)) % 360.0
    r_p = np.hypot(x, y) / np.cos(np.radians(P['el']))
    ri = np.floor(np.searchsorted(pa, az_p)).astype(int) % len(o)
    gi = np.floor(r_p / P['gw']).astype(int)
    out = np.full(A.shape, np.nan)
    ok = (gi >= 0) & (gi < P['gates'])
    src = prev_field[o]
    out[ok] = src[ri[ok], gi[ok]]
    return out


def _gate_unfold(raw, va, ref):
    """each gate: the fold of `raw` closest to `ref` (NaN ref -> NaN)."""
    itv = 2 * va
    k = np.round((ref - raw) / itv)
    return raw + k * itv


def _region_fix(u, va, passes=2, win=(9, 151)):
    """Optional (env RH_REGION_FIX=1): split the field at fold jumps (> 1 Nyquist between
    neighbouring gates), then shift each piece by whole folds toward the smoothed field
    of everything AROUND it. Catches strips that are wrong in every frame (so they never
    flicker) -- e.g. 2026-11-03 Okarche, where a band beyond the -Nyquist fold stayed
    folded in the seed frame and was carried through the loop."""
    data = u.filled(np.nan).astype(np.float64); itv = 2 * va
    hr, hg = win[0] // 2 + 1, win[1] // 2 + 1
    for _ in range(passes):
        v = np.isfinite(data)
        E = np.zeros(data.shape, bool)
        dg = np.abs(np.diff(data, axis=1)) > va; dg &= np.isfinite(np.diff(data, axis=1))
        da = np.abs(np.diff(data, axis=0)) > va; da &= np.isfinite(np.diff(data, axis=0))
        E[:, 1:] |= dg; E[:, :-1] |= dg; E[1:, :] |= da; E[:-1, :] |= da
        lab, n = ndimage.label(v & ~E)
        if n < 2: break
        sizes = ndimage.sum(np.ones(data.shape), lab, index=np.arange(1, n + 1))
        order = np.argsort(sizes)[::-1]
        for j in order[1:80]:
            if sizes[j] < 100: break
            l = j + 1; m = lab == l
            rr, gg = np.nonzero(m)
            r0, r1 = max(rr.min() - hr, 0), min(rr.max() + hr + 1, data.shape[0])
            g0, g1 = max(gg.min() - hg, 0), min(gg.max() + hg + 1, data.shape[1])
            sub = data[r0:r1, g0:g1].copy(); ms = m[r0:r1, g0:g1]
            sub[ms] = np.nan
            tgt = _smooth_nan(sub, size=win)
            both = ms & np.isfinite(tgt)
            if both.sum() < 50: continue
            k = int(np.round(np.nanmedian(tgt[both] - data[r0:r1, g0:g1][both]) / itv))
            if k: data[m] += k * itv
    loc = _smooth_nan(data, size=7)
    fix = np.isfinite(data) & np.isfinite(loc) & (np.abs(data - loc) > va)
    data[fix] = _gate_unfold(data[fix], va, loc[fix])
    return np.ma.masked_invalid(data)


def unfold_loop(items, ghz, ref_uv):
    """items: [(Z, V or None)] in time order -> list of unfolded fields (None where no V).

    Seed = the frame with the most valid velocity: Py-ART region-based, regions
    re-folded toward the echo-motion wind. Every other frame is unfolded GATE BY GATE
    against the neighbouring (already unfolded) frame, advected by the echo motion and
    smoothed; gates the neighbour cannot reach take Py-ART's value shifted to agree
    with their already-unfolded surroundings, then the echo-motion wind."""
    out = [None] * len(items)
    Zs = [it[0] for it in items]
    raws, vas = [], []
    for Z, V in items:
        if V is None:
            raws.append(None); vas.append(None); continue
        v = np.ma.masked_equal(V['u8'], 0).astype(np.float64); v = (v - 128.0) / 2.0
        raws.append(v); vas.append(V['info']['prf'] * (299792458.0 / (ghz * 1e9)) / 4.0)
    good = [i for i, r in enumerate(raws) if r is not None]
    if not good: return out
    seed = max(good, key=lambda i: int((~np.ma.getmaskarray(raws[i])).sum()))
    u0, va0 = dealias(Zs[seed], items[seed][1], ghz)
    out[seed] = _refold(u0, va0, _ref_field(Zs[seed], ref_uv), min_overlap=30)
    if os.environ.get('RH_REGION_FIX') == 'seed':     # fix the seed only; the loop carries it
        out[seed] = _region_fix(out[seed], va0)
    for order in (range(seed + 1, len(items)), range(seed - 1, -1, -1)):
        prev = seed
        for i in order:
            if raws[i] is None: continue
            va = vas[i]
            dt_s = Zs[i]['t'] - Zs[prev]['t']
            ref = _advect(_smooth_nan(out[prev].filled(np.nan)), Zs[prev], Zs[i], ref_uv, dt_s)
            raw = raws[i].filled(np.nan)
            # noise mask from Py-ART's own gate filter (texture, despeckle) keeps junk out
            base, _ = dealias(Zs[i], items[i][1], ghz)
            keep = ~np.ma.getmaskarray(base)
            res = np.full(raw.shape, np.nan)
            if ref is not None:
                hit = keep & np.isfinite(ref)
                res[hit] = _gate_unfold(raw[hit], va, ref[hit])
            # gates the neighbour could not reach: Py-ART's value, refolded to the local
            # surroundings already unfolded, else to the echo-motion wind
            rest = keep & ~np.isfinite(res)
            if rest.any():
                loc = _smooth_nan(res, size=15)
                fill = base.filled(np.nan)
                lr = rest & np.isfinite(loc)
                res[lr] = _gate_unfold(fill[lr], va, loc[lr])
                wr = rest & ~np.isfinite(loc)
                if wr.any():
                    res[wr] = _gate_unfold(fill[wr], va, _ref_field(Zs[i], ref_uv)[wr])
            # last pass: an isolated gate a fold away from its neighbours follows them
            loc = _smooth_nan(res, size=7)
            fix = np.isfinite(res) & np.isfinite(loc) & (np.abs(res - loc) > va)
            res[fix] = _gate_unfold(res[fix], va, loc[fix])
            out[i] = np.ma.masked_invalid(res)
            prev = i
    if os.environ.get('RH_REGION_FIX') == '1':
        for i in good:
            if out[i] is not None: out[i] = _region_fix(out[i], vas[i])
    return out


def flicker(items, unf, ref_uv=(0.0, 0.0), ghz=9.6):
    """Share of shared gates jumping > 1.5 Nyquist between consecutive frames, with the
    earlier frame moved along the echo motion (so storm motion is not counted) and both
    lightly smoothed (so speckle is not counted): what is left is fold flipping."""
    res = []
    for i in range(1, len(items)):
        a, b = unf[i - 1], unf[i]
        if a is None or b is None or items[i][1] is None: res.append(None); continue
        P, C = items[i - 1][0], items[i][0]
        A = _advect(_smooth_nan(a.filled(np.nan), 5), P, C, ref_uv, C['t'] - P['t'])
        if A is None: res.append(None); continue
        va = items[i][1]['info']['prf'] * (299792458.0 / (ghz * 1e9)) / 4.0   # the radar's own band (was hard-coded X band: WSR-88D limit came out 3.4x too small)
        B = _smooth_nan(b.filled(np.nan), 5); B[np.ma.getmaskarray(b)] = np.nan
        both = np.isfinite(A) & np.isfinite(B)
        res.append(float((np.abs(A[both] - B[both]) > 1.5 * va).mean()) if both.sum() > 500 else None)
    return res

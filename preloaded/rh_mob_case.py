"""RadarHub sweep folder (PX-1000 / RaXPol, Z + V pairs) -> Mobile Radar Finder case.

Each sweep becomes one finder scan: REF and unfolded VEL encoded exactly like
mobcore.convert (u8, ENC ranges, 360 azimuth rows), packed with mobcore.pack and
gzip'd into <cache>/<dep_id>/<n>.base.bin.gz (REF) and <n>.vel.bin.gz (VEL) -- the
same files mobile_service.py serves and the GitHub runners deliver. Velocity is
unfolded like wind_loop.py: Py-ART region-based, then each connected echo region
re-folded toward the echo-motion reference. Sweeps more than GAP_MIN apart start a
new deployment.

usage: rh_mob_case.py <folder> <case.json> <cache_dir> <out_deployments.json>
case.json: {"id": "radarhub-px1000-20260705", "radar": "PX-1000", "ghz": 9.55,
            "place": "Washington", "state": "OK", "placeKm": 2, "note": "...",
            "tilts": [2.0] (optional: only these elevations)}
"""
import datetime as dt, glob, gzip, json, os, struct, sys
import numpy as np
os.environ.setdefault('PYART_QUIET', '1')
import pyart
from scipy import ndimage
sys.path.insert(0, '/opt/mobile_radar')
import mobcore

folder, case_path, cache_dir, out_path = sys.argv[1:5]
CASE = json.load(open(case_path))
GHZ = CASE.get('ghz', 9.55)
GAP_MIN = 12
NROWS = int(CASE.get('nrows', 360))     # 720 for WSR-88D super-res (Preloaded Data)
# the page draws row k centred on k * 360/nrows; super-res beams are centred on x.25 / x.75 deg and are
# stored by floor() -> tell the page to draw them a quarter degree on (true beam azimuths)
AZOFF = 0.25 if NROWS == 720 else 0.0
ENC = mobcore.ENC


def load(path):
    b = open(path, 'rb').read()
    h = struct.unpack('<hhhhddddffffffff', b[:72])
    rays, gates, ilen, attr, t, lat0, lon0 = h[:7]
    info = json.loads(b[72:72 + ilen]); p = 72 + ilen
    az = np.frombuffer(b, '<u2', rays, p + 2 * rays) / 65536.0 * 360.0
    u8 = np.frombuffer(b, 'u1', rays * gates, p + 4 * rays).reshape(rays, gates)
    return dict(rays=rays, gates=gates, t=t, lat0=lat0, lon0=lon0, el=h[12], gw=h[15], info=info, az=az, u8=u8)


def zgrid(d):
    z = np.ma.masked_equal(d['u8'], 0).astype(np.float32); z = (z - 64.0) / 2.0
    rc = (np.arange(d['gates']) + 0.5) * d['gw'] * np.cos(np.radians(d['el']))
    A, R = np.meshgrid(np.radians(d['az']), rc, indexing='ij')
    x, y = R * np.sin(A), R * np.cos(A)
    w = np.ma.filled(np.clip(z - 30.0, 0, None), 0.0); w[R < 20.0] = 0.0
    H, _, _ = np.histogram2d(y.ravel(), x.ravel(), bins=200, range=[[-100, 100], [-100, 100]], weights=w.ravel())
    N, _, _ = np.histogram2d(y.ravel(), x.ravel(), bins=200, range=[[-100, 100], [-100, 100]])
    return np.where(N > 0, H / np.maximum(N, 1), 0.0)


def shift(a, b):
    a = a - a.mean(); b = b - b.mean()
    r = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    if iy > 100: iy -= 200
    if ix > 100: ix -= 200
    return -float(ix), -float(iy)


def encode_grid(vals, az, prod):
    """mobcore's u8 encoding + one row per degree of azimuth (gaps of one row closed)."""
    lo, hi = ENC[prod]
    a = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(vals, dtype=np.float32)), np.nan)
    enc = np.zeros(a.shape, np.uint8); ok = np.isfinite(a)
    enc[ok] = np.clip(np.round((a[ok] - lo) * (254.0 / (hi - lo))) + 1, 1, 255).astype(np.uint8)
    nrows = NROWS
    if nrows == 360:
        rows = np.round(az / (360.0 / nrows)).astype(int) % nrows
    else:                                      # super-res rays sit at k*0.5 + 0.25 deg: floor keeps them 1:1
        rows = np.floor(az / (360.0 / nrows)).astype(int) % nrows
    grid = np.zeros((nrows, enc.shape[1]), np.uint8); seen = np.zeros(nrows, bool)
    for r, row in zip(rows, enc):
        if not seen[r]: grid[r] = row; seen[r] = True
    for r in np.where(~seen)[0]:
        p, q = (r - 1) % nrows, (r + 1) % nrows
        if seen[p] and seen[q]: grid[r] = grid[p]
    return grid


# ---------- sweeps ----------
items = []
for zf in sorted(glob.glob(os.path.join(folder, '*-Z.bin'))):
    vf = zf[:-6] + '-V.bin'
    Z = load(zf)
    if CASE.get('tilts') and not any(abs(Z['el'] - t) < 0.05 for t in CASE['tilts']):
        continue
    V = load(vf) if os.path.exists(vf) else None
    if V is not None and (V['t'] != Z['t'] or V['rays'] != Z['rays'] or V['gates'] != Z['gates']):
        V = None
    Z['extra'] = {}                              # optional (Preloaded Data / l2_to_rhbins.py): products from the
    for suf, prod in (('-R.bin', 'REF'), ('-D.bin', 'ZDR'), ('-C.bin', 'CC')):   # surveillance half, each on its
        xf = zf[:-6] + suf                                                          # OWN beams and with its own time
        if os.path.exists(xf):
            X = load(xf)
            if X['gates'] == Z['gates']:
                Z['extra'][prod] = X
    items.append((Z, V))
items.sort(key=lambda it: it[0]['t'])
assert items, 'no sweeps'
# RadarHub mixes misfiled sweeps from other sites into some days (Colorado / Norman sweeps inside the
# Florida week): keep only sweeps within 2 km of the case's median truck position
_la = float(np.median([Z['lat0'] for Z, _ in items])); _lo = float(np.median([Z['lon0'] for Z, _ in items]))
_n0 = len(items)
items = [it for it in items if abs(it[0]['lat0'] - _la) * 111 < 2 and abs(it[0]['lon0'] - _lo) * 111 * np.cos(np.radians(_la)) < 2]
if len(items) != _n0: print(f'  dropped {_n0 - len(items)} misfiled sweep(s) far from the truck position')

# echo motion reference (frames 6-15 min apart)
grids = [(Z['t'], zgrid(Z)) for Z, _ in items]
us, vs = [], []
for i in range(len(grids)):
    for j in range(i + 1, len(grids)):
        dts = grids[j][0] - grids[i][0]
        if 360 <= dts <= 900 and grids[i][1].sum() and grids[j][1].sum():
            dx, dy = shift(grids[i][1], grids[j][1])
            u, v = dx * 1000 / dts, dy * 1000 / dts
            if np.hypot(u, v) < 40: us.append(u); vs.append(v)
REF_UV = CASE.get('ref_uv') or ([float(np.median(us)), float(np.median(vs))] if us else [0.0, 0.0])
print(f"{CASE['id']}: echo motion u {REF_UV[0]:.1f} v {REF_UV[1]:.1f} m/s from {len(us)} pairs")

# ---------- convert ----------
# the whole loop is unfolded together (no frame-to-frame fold flicker), see rh_unfold.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_unfold
UNF = rh_unfold.unfold_loop(items, GHZ, REF_UV)
FL = [f for f in rh_unfold.flicker(items, UNF, REF_UV, GHZ) if f is not None]
worst = max(FL) if FL else 0.0
print(f"  flicker: worst frame pair {100 * worst:.2f}% of gates jump > 1.5 Nyquist (limit {100 * rh_unfold.FLICKER_MAX:.0f}%), mean {100 * (sum(FL) / len(FL) if FL else 0):.2f}%")
if worst > rh_unfold.FLICKER_MAX and '--force' not in sys.argv:
    sys.exit(f"  NOT PUBLISHED: velocity flickers too much between frames ({100 * worst:.1f}%); fix the unfolding or pass --force")
scans = []
for (Z, V), unf in zip(items, UNF):
    rays, gates, gw, el, az = Z['rays'], Z['gates'], Z['gw'], Z['el'], Z['az']
    ref = np.ma.masked_equal(Z['u8'], 0).astype(np.float32); ref = (ref - 64.0) / 2.0
    ts = int(round(Z['t'] * 1000))
    angle = round(float(el), 2)
    common = {'ts': ts, 'lat': Z['lat0'], 'lon': Z['lon0']}
    geo = {'nrows': NROWS, 'gates': int(gates), 'firstGateM': 0.0, 'gateSizeM': float(gw * 1000.0), **({'azOffDeg': AZOFF} if AZOFF else {})}
    n = len(scans)
    out = os.path.join(cache_dir, CASE['id'])  # dep id fixed below when split
    xg, xts = {}, {}
    for prod, X in Z.get('extra', {}).items():
        xv = np.ma.masked_equal(X['u8'], 0).astype(np.float32)
        xv = (xv - 128.0) / 16.0 if prod == 'ZDR' else (xv / 200.0 if prod == 'CC' else (xv - 64.0) / 2.0)
        xg[prod] = encode_grid(xv, X['az'], prod)
        xts[prod] = int(round(X['t'] * 1000))
    zdisp = xg.pop('REF', None)                  # REF as measured in the surveillance half, when there is one
    zts = xts.pop('REF', ts)
    scans.append({'Z': zdisp if zdisp is not None else encode_grid(ref, az, 'REF'),
                  'V': encode_grid(unf, az, 'VEL') if unf is not None else None, 'X': xg, 'xts': xts,
                  'zts': zts, 'vts': ts,
                  'ts': zts, 'angle': angle, 'lat': Z['lat0'], 'lon': Z['lon0'], 'geo': geo,
                  'rangeKm': gates * gw})

# ---------- split into deployments, write cache files ----------
groups, cur = [], [scans[0]]
for s in scans[1:]:
    if s['ts'] - cur[-1]['ts'] > GAP_MIN * 60e3:
        groups.append(cur); cur = [s]
    else:
        cur.append(s)
groups.append(cur)
deps = []
for gi, g in enumerate(groups, 1):
    dep_id = f"{CASE['id']}-{gi}"
    ddir = os.path.join(cache_dir, dep_id); os.makedirs(ddir, exist_ok=True)
    idx_scans = []
    for n, s in enumerate(g):
        for grp, prod, grid in (('base', 'REF', s['Z']), ('vel', 'VEL', s['V'])):
            if grid is None:
                grid = np.zeros_like(s['Z'])          # keeps the scan 'ready'; draws nothing
            raw = grid.tobytes()
            # each product carries the time ITS data was collected (surveillance vs Doppler half)
            prods = {prod: {**s['geo'], 'off': 0, 'len': len(raw), 'ts': s['zts'] if grp == 'base' else s['vts']}}
            if grp == 'base':                          # ZDR / CC ride in the base file with REF
                for xp, xgrid in s.get('X', {}).items():
                    xb = xgrid.tobytes(); prods[xp] = {**s['geo'], 'off': len(raw), 'len': len(xb), 'ts': s['xts'].get(xp, s['zts'])}; raw += xb
            hdr = {'ts': s['ts'], 'lat': s['lat'], 'lon': s['lon'],
                   'tilts': [{'angle': s['angle'], 'ts': s['ts'], 'products': prods}]}
            with open(os.path.join(ddir, f'{n}.{grp}.bin.gz'), 'wb') as f:
                f.write(gzip.compress(mobcore.pack(hdr, raw), compresslevel=3))
        nxt = g[n + 1]['ts'] if n + 1 < len(g) else s['ts'] + 20000
        idx_scans.append({'t0': s['ts'], 't1': nxt, 'lat': s['lat'], 'lon': s['lon'],
                          'sweeps': [{'angle': s['angle'], 'ts': s['ts']}], 'ref': {'kind': 'pinned', 'member': f'{n}'}})
    angles = sorted({s['angle'] for s in g})
    dep = {'id': dep_id, 'source': 'radarhub-wind-2026', 'radar': CASE['radar'], 'band': 'X', 'operator': 'OU ARRC',
           'credit': f"OU ARRC {CASE['radar']} via RadarHub (shared with permission)", 'home': 'https://radarhub.arrc.ou.edu/',
           'terms': 'OU ARRC RadarHub data, downloaded with permission; BETA preview only.',
           'note': CASE.get('note', ''), 'flag': '', 'place': CASE['place'], 'state': CASE['state'],
           'placeKm': CASE.get('placeKm', 0), 'lat': g[0]['lat'], 'lon': g[0]['lon'],
           'start': g[0]['ts'], 'end': idx_scans[-1]['t1'], 'rangeKm': round(max(s['rangeKm'] for s in g), 1),
           'products': ['REF', 'VEL'] + [p for p in ('ZDR', 'CC') if any(p in s.get('X', {}) for s in g)], 'nscans': len(g), 'angles': angles, 'scans': idx_scans,
           'pinned': True, 'beta': True}
    dep['rev'] = int(dt.datetime.now(dt.timezone.utc).timestamp())   # new on every rebuild: the page appends it to scan URLs (cache-buster)
    dep.update(CASE.get('dep', {}))          # e.g. Preloaded Data (WSR-88D): source/operator/credit/preloaded
    deps.append(dep)
    print(f"  {dep_id}: {len(g)} scans {dt.datetime.utcfromtimestamp(g[0]['ts']/1000):%Y-%m-%d %H:%M:%S}-"
          f"{dt.datetime.utcfromtimestamp(g[-1]['ts']/1000):%H:%M:%S}Z angles {angles}")
json.dump(deps, open(out_path, 'w'))

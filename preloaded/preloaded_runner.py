#!/usr/bin/env python3
"""hhwx.me Radar Archive > Preloaded Data: build ONE WSR-88D lowest-tilt case on a GitHub runner.

CASE (env, JSON): {"id": "preloaded-<site>-YYYYMMDD-HHMM", "site": "KXXX", "w0": ms, "w1": ms,
                   "place", "state", "placeKm", "note", "tor": {...}, optional "ref_uv": [u, v]}
1. lists + downloads the Level II volumes in [w0 - 10 min, w1] from NOAA's AWS bucket
2. l2_to_rhbins.py: lowest tilt only, each half on its own beams with its own time (true data)
3. rh_mob_case.py: whole-loop velocity unfolding (every gate kept), flicker gate, finder scans
   -> out/cache/<dep>/<n>.{base,vel}.bin.gz + out/case.json
Raw volumes are deleted once converted. mobile/mobile_push.py then pushes out/cache to hhwx.me;
the main session installs out/case.json into the finder index (install_preloaded_json.py).
mobcore.py comes from the server (token-gated /api/mobile/src), like mobile_shard.py.
"""
import datetime as dt, glob, json, os, re, shutil, subprocess, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.environ.get('MOBILE_SRV', 'https://hhwx.me/api/mobile')
TOK = os.environ.get('MOBILE_SRC_TOKEN', '').strip()
L2 = 'https://unidata-nexrad-level2.s3.amazonaws.com'
c = json.loads(os.environ['CASE'])
os.makedirs('lib', exist_ok=True); os.makedirs('out/cache', exist_ok=True)
urllib.request.urlretrieve(f'{SRV}/src/{TOK}/mobcore.py', 'lib/mobcore.py')
sys.path.insert(0, HERE)
import l2_to_rhbins


def list_keys(site, w0, w1):
    keys = []
    t = dt.datetime.fromtimestamp((w0 - 15 * 60000) / 1000, dt.timezone.utc).replace(minute=0, second=0)
    end = dt.datetime.fromtimestamp(w1 / 1000, dt.timezone.utc)
    while t <= end:
        x = urllib.request.urlopen(f'{L2}/?list-type=2&prefix={t:%Y/%m/%d}/{site}/{site}{t:%Y%m%d_%H}', timeout=60).read().decode()
        keys += re.findall(r'<Key>([^<]+)</Key>', x)
        t += dt.timedelta(hours=1)
    out = []
    for k in sorted(set(keys)):
        m = re.search(r'(\d{8})_(\d{6})(_V0\d)?(\.gz)?$', k)
        if not m or k.endswith('_MDM'):
            continue
        ts = dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        if w0 - 10 * 60000 <= ts <= w1:
            out.append(k)
    return out


L3 = 'https://unidata-nexrad-level3.s3.amazonaws.com'
GCS = 'https://storage.googleapis.com/gcp-public-data-nexrad-l3'
AWS_L3_FROM = 1585612800000                       # 2020-03-31


def nws_products(site, w0, w1):
    """NWS's own unfolded lowest-tilt velocity (Level III N0G, else N0U) for every cut in the window,
    keyed by the cut's start (NWS stamps every low cut, SAILS rescans included, with the start of its surveillance
    half -- the same time l2_to_rhbins.py stores): {epoch s: {az, vel, gate_km}}.
    2020-03-31 on: AWS Level III keys; before: Google's daily Level III tarball streamed here (never stored)."""
    from metpy.io import Level3File
    os.makedirs('l3', exist_ok=True)
    files = []
    if w0 >= AWS_L3_FROM:
        t = dt.datetime.fromtimestamp((w0 - 15 * 60000) / 1000, dt.timezone.utc).replace(minute=0, second=0)
        end = dt.datetime.fromtimestamp(w1 / 1000, dt.timezone.utc)
        while t <= end:
            for prod in ('N0G', 'N0U'):          # super-res N0G when the archive has it (not 2020-21), else 1-degree N0U
                x = urllib.request.urlopen(f'{L3}/?list-type=2&prefix={site[1:]}_{prod}_{t:%Y_%m_%d_%H}', timeout=60).read().decode()
                ks = re.findall(r'<Key>([^<]+)</Key>', x)
                for k in ks:
                    p = os.path.join('l3', k); urllib.request.urlretrieve(f'{L3}/{k}', p); files.append(p)
                if ks:
                    break
            t += dt.timedelta(hours=1)
    else:
        import tarfile
        d0 = dt.datetime.fromtimestamp((w0 - 15 * 60000) / 1000, dt.timezone.utc).date()
        d1 = dt.datetime.fromtimestamp(w1 / 1000, dt.timezone.utc).date()
        for k in range((d1 - d0).days + 1):
            day = d0 + dt.timedelta(days=k)
            q = json.load(urllib.request.urlopen(f'https://storage.googleapis.com/storage/v1/b/gcp-public-data-nexrad-l3/o?prefix={day:%Y/%m/%d}/{site}/&maxResults=5', timeout=60))
            names = [x['name'] for x in q.get('items', []) if x['name'].endswith(('.tar.gz', '.tar.Z', '.tgz'))]
            if not names:
                continue
            got = {'N0G': [], 'N0U': [], 'N0V': []}
            pr = subprocess.Popen(['bash', '-c', f'curl -s --fail "{GCS}/{names[0]}" | gzip -dc'], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=pr.stdout, mode='r|') as tf:
                for m in tf:
                    b = os.path.basename(m.name)
                    for prod in got:
                        if m.isfile() and f'_{prod}' in b:
                            p = os.path.join('l3', b); open(p, 'wb').write(tf.extractfile(m).read()); got[prod].append(p)
            pr.wait()
            files += got['N0G'] or got['N0U'] or got['N0V']   # super-res N0G when the day has it, else 1-degree N0U, else old N0V (1 km)
    vols = {}
    for p in files:
        try:
            f = Level3File(p)
            blk = f.sym_block[0][0]
            a0 = np.asarray(blk['start_az'], float); a1 = np.asarray(blk['end_az'], float)
            az = (a0 + ((a1 - a0) % 360) / 2) % 360
            vel = np.asarray(f.map_data(blk['data']), float)
            vt = f.metadata['vol_time'].replace(tzinfo=dt.timezone.utc).timestamp()
            gk = 1.0 if f.header.code == 27 else 0.25           # N0V (code 27) 1 km bins; N0U (99) / N0G (154) 0.25 km
            if f.header.code == 27:      # old 16-level N0V: each value is a bin EDGE in knots -> bin middle in m/s
                mid = {-64: -64, -50: -57, -36: -43, -26: -31, -20: -23, -10: -15, -1: -5.5,
                       0: 5, 10: 15, 20: 23, 26: 31, 36: 43, 50: 57, 64: 64}
                vel = np.vectorize(lambda x: mid.get(int(x), np.nan) * 0.514444 if np.isfinite(x) else np.nan)(vel)
            vols.setdefault(round(vt), {'az': az, 'vel': vel, 'gate_km': gk})
        except Exception as e:
            print('  skip L3', os.path.basename(p), type(e).__name__, e, flush=True)
    shutil.rmtree('l3', ignore_errors=True)
    print(f'  NWS velocity products: {len(vols)} cuts', flush=True)
    return vols


cid = c['id']
NWS = nws_products(c['site'], c['w0'], c['w1'])


def nws_for(t):
    best = min(NWS, key=lambda v: abs(v - t), default=None)
    return NWS[best] if best is not None and abs(best - t) <= 3 else None


keys = list_keys(c['site'], c['w0'], c['w1'])
print(f'{cid}: {len(keys)} Level II volumes', flush=True)
if not keys:
    sys.exit(f'{cid}: no Level II files')
os.makedirs('raw', exist_ok=True); bins = os.path.join('bins', cid); os.makedirs(bins, exist_ok=True)
paths = []
for k in keys:
    p = os.path.join('raw', os.path.basename(k))
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(f'{L2}/{k}', p); break
        except Exception as e:
            print('retry', k, e, flush=True); time.sleep(5 * (attempt + 1))
    if os.path.exists(p):
        paths.append(p)
n = l2_to_rhbins.convert(paths, bins, float(c.get('crop_km', 150.0)), log=lambda *a: print(*a, flush=True), nws=nws_for,
                        site=(c['lat'], c['lon']) if c.get('lat') is not None else None)
shutil.rmtree('raw', ignore_errors=True)
for f in glob.glob(os.path.join(bins, '*.bin')):          # frames inside the window only
    ts = dt.datetime.strptime(os.path.basename(f)[:15], '%Y%m%d-%H%M%S').replace(tzinfo=dt.timezone.utc).timestamp() * 1000
    if not (c['w0'] <= ts <= c['w1']):
        os.remove(f)
nz = len(glob.glob(os.path.join(bins, '*-Z.bin')))
print(f'{cid}: {n} frames converted, {nz} in the window', flush=True)
if nz < 3:
    sys.exit(f'{cid}: only {nz} frames')
mob = {'id': cid, 'radar': c['site'], 'ghz': 2.8, 'nrows': 720, 'place': c['place'], 'state': c['state'],
       'placeKm': c.get('placeKm', 0), 'note': c['note'],
       'dep': {'source': 'preloaded-wsr88d', 'band': 'S', 'operator': 'NOAA NWS', 'preloaded': True, 'beta': False,
               'credit': f"NOAA NEXRAD Level II ({c['site']}) via NOAA Open Data on AWS",
               'home': 'https://registry.opendata.aws/noaa-nexrad/', 'terms': 'NOAA NEXRAD Level II, public domain.',
               **({'tor': c['tor']} if c.get('tor') else {})}}
if c.get('ref_uv'):
    mob['ref_uv'] = c['ref_uv']
json.dump(mob, open('mob.json', 'w'))
env = {**os.environ, 'PYART_QUIET': '1', 'RH_VEL_MASK': 'loose', 'MPLBACKEND': 'Agg',
       'PYTHONPATH': os.path.abspath('lib') + os.pathsep + os.environ.get('PYTHONPATH', '')}
pr = subprocess.run([sys.executable, os.path.join(HERE, 'rh_mob_case.py'), bins, 'mob.json', 'out/cache', 'out/case.json'],
                    capture_output=True, text=True, env=env)
print('\n'.join(l for l in (pr.stdout + pr.stderr).splitlines() if 'Deprecat' not in l and 'warn' not in l.lower()), flush=True)
if not os.path.exists('out/case.json'):
    sys.exit(f'{cid}: not built (see above)')
print(f'{cid}: built', flush=True)

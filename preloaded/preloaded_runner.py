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


cid = c['id']
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
n = l2_to_rhbins.convert(paths, bins, 150.0, log=lambda *a: print(*a, flush=True))
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

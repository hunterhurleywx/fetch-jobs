#!/usr/bin/env python3
"""Mobile Radar Finder (hhwx.me Radar Archive): convert one shard of every
indexed mobile-radar scan into the viewer's cached byte grids, and check each
result decodes.

Every shard computes the same plan from the server's index.json, so the shards
split the work without talking to each other. Converted files go to
out/cache/<deployment>/<n>.<base|vel>.bin.gz (the server cache layout) and one
JSON line per scan to out/report.jsonl.

Needs MOBILE_SRC_TOKEN (GitHub secret). mobcore.py comes from the server, so
the conversion is exactly the one the live service runs.
"""
import argparse, gzip, json, os, sys, time, traceback, urllib.parse, zipfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests

SRV = os.environ.get('MOBILE_SRV', 'https://hhwx.me/api/mobile')
TOK = os.environ.get('MOBILE_SRC_TOKEN', '').strip()
GROUPS = {'base': ('REF', 'ZDR', 'CC'), 'vel': ('VEL',)}
PRODS = ('REF', 'VEL', 'ZDR', 'CC')
UNIT = 120          # scans per work unit (big deployments spread over shards)


def src_url(name):
    return f'{SRV}/src/{TOK}/' + urllib.parse.quote(name)


def cost(dep):
    """Rough seconds per scan, for balancing shards."""
    i = dep['id']
    if i.startswith('noxp-2011'):
        return 80.0          # 15-tilt SIGMET volumes, slow to unfold
    if i.startswith('px1000'):
        return 3.0
    if i.startswith('mpar'):
        return 3.0
    return 8.0               # DOW / RaXPol / COW Level II volumes


def plan(deps, shards):
    units = []
    for d in deps:
        n = d['nscans']
        for a in range(0, n, UNIT):
            b = min(n, a + UNIT)
            units.append((cost(d) * (b - a), d['id'], a, b))
    units.sort(reverse=True)
    load = [0.0] * shards
    out = [[] for _ in range(shards)]
    for c, dep, a, b in units:
        k = min(range(shards), key=lambda j: load[j])
        load[k] += c
        out[k].append((dep, a, b))
    return out, load


# ---- worker process --------------------------------------------------------
_zips = {}
mobcore = None


def _init(here):
    global mobcore
    os.environ.setdefault('PYART_QUIET', '1')
    sys.path.insert(0, here)
    import mobcore as m
    mobcore = m


def _fetch(ref):
    if ref['kind'] != 'localzip':
        return mobcore.fetch_member(ref)
    name = os.path.basename(ref['path'])
    z = _zips.get(name)
    if z is None:
        z = _zips[name] = zipfile.ZipFile(mobcore.HttpFile(src_url(name), block=4 << 20))
    try:
        return z.read(ref['name'])
    finally:
        z.fp.cache.clear()


def _check(hdr, blob, expect_prods):
    """Problems with one packed result (empty list = good)."""
    probs = []
    for k, t in enumerate(hdr.get('tilts', [])):
        for P, d in t['products'].items():
            if d['off'] + d['len'] > len(blob):
                probs.append(f'tilt {k} {P}: data past end')
            elif d['len'] != d['nrows'] * d['gates']:
                probs.append(f'tilt {k} {P}: size {d["len"]} != {d["nrows"]}x{d["gates"]}')
    return probs


def job(dep_id, n, ref, nsweeps, outdir):
    t0 = time.time()
    rec = {'dep': dep_id, 'n': n, 'expect_tilts': nsweeps}
    try:
        for attempt in range(3):
            try:
                buf = _fetch(ref)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        rec['raw_mb'] = round(len(buf) / 1e6, 2)
        hdr, blob = mobcore.convert(buf, want=PRODS, dealias=not ref.get('nodealias'),
                                    az_offset=float(ref.get('azOffset', 0.0)))
        del buf
        warn = []
        rec['tilts'] = len(hdr['tilts'])
        have = sorted({P for t in hdr['tilts'] for P in t['products']})
        rec['products'] = have
        if rec['tilts'] != nsweeps:
            warn.append(f'{rec["tilts"]} tilts, index says {nsweeps}')
        for g, ps in GROUPS.items():
            h, b = mobcore.subset(hdr, blob, set(ps))
            data = gzip.compress(mobcore.pack(h, b), compresslevel=3)
            h2, b2 = mobcore.unpack(gzip.decompress(data))          # round trip
            probs = _check(h2, b2, ps)
            if probs:
                raise ValueError(f'{g}: ' + '; '.join(probs[:3]))
            rec[f'{g}_tilts'] = len(h2['tilts'])
            rec[f'{g}_kb'] = round(len(data) / 1024)
            path = os.path.join(outdir, 'cache', dep_id, f'{n}.{g}.bin.gz')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
        if 'VEL' in have and rec['vel_tilts'] != rec['base_tilts']:
            warn.append(f'VEL on {rec["vel_tilts"]} of {rec["base_tilts"]} tilts (tilt order may shift)')
        if 'REF' not in have:
            warn.append('no reflectivity')
        else:
            ref_px = 0
            for t in hdr['tilts']:
                d = t['products'].get('REF')
                if d:
                    seg = blob[d['off']:d['off'] + d['len']]
                    ref_px += len(seg) - seg.count(0)
            rec['ref_px'] = ref_px
            if ref_px == 0:
                warn.append('reflectivity is empty')
        rec['ok'] = True
        rec['warn'] = warn
    except Exception as e:
        rec['ok'] = False
        rec['err'] = f'{type(e).__name__}: {e}'[:400]
        rec['trace'] = traceback.format_exc()[-1500:]
    rec['sec'] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--of', type=int, default=20)
    ap.add_argument('--only', default='', help='comma list of deployment ids')
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--out', default='out')
    ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args()
    if not TOK:
        sys.exit('MOBILE_SRC_TOKEN not set')

    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('index.json', 'mobcore.py'):
        r = requests.get(src_url(name), timeout=60)
        r.raise_for_status()
        open(os.path.join(here, name), 'wb').write(r.content)
    deps = json.load(open(os.path.join(here, 'index.json')))['deployments']
    if a.only:
        keep = {s.strip() for s in a.only.split(',') if s.strip()}
        deps = [d for d in deps if d['id'] in keep]
    shards, load = plan(deps, a.of)
    mine = shards[a.shard]
    print(f'shard {a.shard}/{a.of}: {sum(b - x for _, x, b in mine)} scans, est {load[a.shard] / a.procs / 60:.1f} min '
          f'on {a.procs} procs; units: ' + ', '.join(f'{d}[{x}:{b}]' for d, x, b in mine), flush=True)
    if a.plan_only:
        return
    byid = {d['id']: d for d in deps}
    os.makedirs(a.out, exist_ok=True)
    rep = open(os.path.join(a.out, 'report.jsonl'), 'w')
    ok = bad = 0
    t0 = time.time()
    ctx = mp.get_context('spawn')                     # HDF5 is not fork/thread safe
    # No max_tasks_per_child: recycling spawn workers hangs the pool on 3.12
    # (every shard froze at 4 x 40 = 160 scans, run 43970937446).
    with ProcessPoolExecutor(a.procs, mp_context=ctx, initializer=_init, initargs=(here,)) as pool:
        futs = []
        for dep_id, x, b in mine:
            d = byid[dep_id]
            for n in range(x, b):
                s = d['scans'][n]
                futs.append(pool.submit(job, dep_id, n, s['ref'], len(s['sweeps']), a.out))
        total = len(futs)
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            rep.write(json.dumps(rec) + '\n')
            rep.flush()
            if rec['ok']:
                ok += 1
            else:
                bad += 1
                print(f'FAIL {rec["dep"]} {rec["n"]}: {rec["err"]}', flush=True)
            if i % 50 == 0 or i == total:
                print(f'{i}/{total} done ({ok} ok, {bad} failed) {time.time() - t0:.0f}s', flush=True)
    rep.close()
    print(f'shard {a.shard} finished: {ok} ok, {bad} failed in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

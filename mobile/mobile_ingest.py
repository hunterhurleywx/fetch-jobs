#!/usr/bin/env python3
"""Mobile Radar Finder (hhwx.me): build one runner-side ("remote") source end to end.

For a source in the server's remote_sources.json: download each archive (several
ranged connections), read every scan in it, build the finder's index entries with
the server's own build_index.group() (same deployment split, place names, fields),
convert every scan with the server's mobcore.convert(), check each result, and
lay it out as out/cache/<deployment>/<n>.<base|vel>.bin.gz plus
out/index_<source>.json. The workflow then stores it in S3, pushes the cache to
hhwx.me and finally posts the index entries (the server refuses them unless every
scan is cached).

Needs MOBILE_SRC_TOKEN.
"""
import argparse, gzip, io, json, os, re, shutil, sys, tarfile, threading, time, traceback, urllib.parse, zipfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import requests

SRV = os.environ.get('MOBILE_SRV', 'https://hhwx.me/api/mobile')
TOK = os.environ.get('MOBILE_SRC_TOKEN', '').strip()
UA = {'User-Agent': 'hhwx-radar-archive/1.0 (mobile radar viewer)'}
GROUPS = {'base': ('REF', 'ZDR', 'CC'), 'vel': ('VEL',)}
PRODS = ('REF', 'VEL', 'ZDR', 'CC')
JUNK = re.compile(r'(^|/)(\._|\.|__MACOSX)|\.(txt|md|pdf|png|jpg|gif|csv|json|xml|log|py|npy|mat)$|(^|/)(README|corrections)', re.I)


def src_url(name):
    return f'{SRV}/src/{TOK}/' + urllib.parse.quote(name)


def download(url, path, lanes=12, chunk=16 << 20):
    """Parallel ranged download (one Zenodo connection runs ~1 MB/s)."""
    r = requests.head(url, headers=UA, allow_redirects=True, timeout=60)
    size = int(r.headers.get('Content-Length') or 0)
    final = r.url
    if not size or r.headers.get('Accept-Ranges', '').lower() != 'bytes':
        with requests.get(final, headers=UA, stream=True, timeout=(20, 300)) as g, open(path, 'wb') as f:
            g.raise_for_status()
            shutil.copyfileobj(g.raw, f, 1 << 20)
        return os.path.getsize(path)
    with open(path, 'wb') as f:
        f.truncate(size)
    lock = threading.Lock()

    def part(a):
        b = min(size, a + chunk) - 1
        for attempt in range(8):
            try:
                g = requests.get(final, headers={**UA, 'Range': f'bytes={a}-{b}'}, timeout=(20, 120))
                if g.status_code != 206 or len(g.content) != b - a + 1:
                    raise IOError(f'HTTP {g.status_code} len {len(g.content)}')
                with lock, open(path, 'r+b') as f:
                    f.seek(a); f.write(g.content)
                return
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(3 * (attempt + 1))
    t0 = time.time()
    with ThreadPoolExecutor(lanes) as ex:
        list(ex.map(part, range(0, size, chunk)))
    print(f'  downloaded {size / 1e9:.2f} GB in {time.time() - t0:.0f}s ({size / 1e6 / max(1, time.time() - t0):.1f} MB/s)', flush=True)
    return size


def zip_sequential(path):
    """Read a zip front to back by its local headers, ignoring the central directory
    (OU's 7 GB El Reno zip has 32-bit offsets that overflow, so ZipFile seeks to junk)."""
    import struct, zlib
    # Sizes from the central directory (its 32-bit offsets overflow, its sizes don't):
    # streamed entries carry no sizes in their local headers.
    cdsize = {}
    with open(path, 'rb') as f:
        f.seek(0, 2); end = f.tell()
        f.seek(max(0, end - (1 << 20))); tail = f.read()
        i = tail.rfind(b'PK\x05\x06')
        if i >= 0:
            size_cd = struct.unpack('<I', tail[i + 12:i + 16])[0]
            eocd_pos = end - len(tail) + i
            f.seek(eocd_pos - size_cd); cd = f.read(size_cd)
            p = 0
            while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
                csz = struct.unpack('<I', cd[p + 20:p + 24])[0]
                nl, xl, cl = struct.unpack('<HHH', cd[p + 28:p + 34])
                cdsize[cd[p + 46:p + 46 + nl].decode('utf-8', 'replace')] = csz
                p += 46 + nl + xl + cl
    with open(path, 'rb') as f:
        while True:
            h = f.read(30)
            if len(h) < 30 or h[:4] != b'PK\x03\x04':
                return
            flags, method = struct.unpack('<HH', h[6:10])
            csize, usize = struct.unpack('<II', h[18:26])
            nlen, xlen = struct.unpack('<HH', h[26:30])
            name = f.read(nlen).decode('utf-8', 'replace')
            extra = f.read(xlen)
            if 0xFFFFFFFF in (csize, usize):          # zip64 sizes live in extra field 0x0001
                p = 0
                while p + 4 <= len(extra):
                    tag, sz = struct.unpack('<HH', extra[p:p + 4])
                    if tag == 1:
                        vals, q = [], p + 4
                        while q + 8 <= p + 4 + sz:
                            vals.append(struct.unpack('<Q', extra[q:q + 8])[0]); q += 8
                        if usize == 0xFFFFFFFF and vals:
                            usize = vals.pop(0)
                        if csize == 0xFFFFFFFF and vals:
                            csize = vals.pop(0)
                    p += 4 + sz
            if flags & 8 and csize == 0:
                if name not in cdsize:
                    raise IOError(f'{name}: streamed zip entry without sizes')
                csize = cdsize[name]
            data = f.read(csize)
            if flags & 8:                             # data descriptor after the data
                sig = f.read(4)
                f.read(12 if sig == b'PK\x07\x08' else 8)
            if name.endswith('/'):
                continue
            yield name, (zlib.decompress(data, -15) if method == 8 else data)


def members(path):
    """(name, bytes) for every file in an archive (tar any compression, zip) or the file itself."""
    if zipfile.is_zipfile(path):
        done = set()
        try:
            with zipfile.ZipFile(path) as z:
                for i in z.infolist():
                    if not i.is_dir():
                        data = z.read(i)
                        done.add(i.filename)
                        yield i.filename, data
        except zipfile.BadZipFile as e:
            print(f'  zip index unusable after {len(done)} files ({e}); reading it front to back', flush=True)
            for name, data in zip_sequential(path):
                if name not in done:
                    yield name, data
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, 'r:*') as t:
            for m in t:
                if m.isfile():
                    yield m.name, t.extractfile(m).read()
    else:
        yield os.path.basename(path), open(path, 'rb').read()


# ---- worker ----------------------------------------------------------------
mobcore = None


def _init(here):
    global mobcore
    os.environ.setdefault('PYART_QUIET', '1')
    sys.path.insert(0, here)
    import mobcore as m
    mobcore = m


def work(name, buf, flags, outdir, k):
    rec = {'name': name, 'k': k}
    t0 = time.time()
    try:
        if isinstance(buf, list):                 # one volume of DORADE sweep files
            import dorade
            buf = dorade.volume_to_cfradial(buf)
            name = name.rstrip('/') + '.nc'
        if buf[:2] == b'\x1f\x8b':
            buf = gzip.decompress(buf)
        meta = mobcore.scan_meta(buf, name)
        if not meta['sweeps'] or meta['t0'] is None:
            rec['skip'] = 'no sweeps'
            return rec
        hdr, blob = mobcore.convert(buf, name, want=PRODS, dealias=not flags.get('nodealias'),
                                    az_offset=float(flags.get('azOffset', 0.0)))
        shift = int(flags.get('timeShiftMs') or 0)
        if shift:
            for key in ('t0', 't1'):
                if meta.get(key) is not None:
                    meta[key] += shift
            for sw in meta['sweeps']:
                sw['ts'] += shift
            if hdr.get('ts'):
                hdr['ts'] += shift
            for t in hdr['tilts']:
                t['ts'] += shift
        warn = []
        if len(hdr['tilts']) != len(meta['sweeps']):
            # keep the index in step with what the data really holds
            got = [round(t['angle'], 2) for t in hdr['tilts']]
            keep, i = [], 0
            for sw in meta['sweeps']:
                if i < len(got) and abs(round(sw['angle'], 2) - got[i]) < 0.05:
                    keep.append(sw); i += 1
            meta['sweeps'] = keep
            warn.append(f'{len(got)} tilts with data')
        if not hdr['tilts']:
            rec['skip'] = 'no usable tilts'
            return rec
        for g, ps in GROUPS.items():
            h, b = mobcore.subset(hdr, blob, set(ps))
            data = gzip.compress(mobcore.pack(h, b), compresslevel=3)
            h2, b2 = mobcore.unpack(gzip.decompress(data))
            for kk, t in enumerate(h2['tilts']):
                for P, d in t['products'].items():
                    if d['off'] + d['len'] > len(b2) or d['len'] != d['nrows'] * d['gates']:
                        raise ValueError(f'{g} tilt {kk} {P}: bad grid')
            with open(os.path.join(outdir, f'{k}.{g}.bin.gz'), 'wb') as f:
                f.write(data)
            rec[f'{g}_kb'] = round(len(data) / 1024)
        ref_px = 0
        for t in hdr['tilts']:
            d = t['products'].get('REF')
            if d:
                seg = blob[d['off']:d['off'] + d['len']]
                ref_px += len(seg) - seg.count(0)
        if ref_px == 0:
            warn.append('reflectivity is empty')
        rec.update(ok=True, meta=meta, warn=warn, tilts=len(hdr['tilts']))
    except Exception as e:
        rec.update(ok=False, err=f'{type(e).__name__}: {e}'[:400], trace=traceback.format_exc()[-1200:])
    rec['sec'] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--out', default='out')
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--work', default='work')
    ap.add_argument('--part', type=int, default=0, help='convert only every --of-th scan, starting at this one')
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--merge', default='', help='directory of part outputs (part_*/scans.json + tmp/) to group and publish')
    a = ap.parse_args()
    if not TOK:
        sys.exit('MOBILE_SRC_TOKEN not set')
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('mobcore.py', 'build_index.py', 'remote_sources.json', 'places.json'):
        r = requests.get(src_url(name), timeout=120)
        r.raise_for_status()
        open(os.path.join(here, name), 'wb').write(r.content)
    sys.path.insert(0, here)
    os.environ.setdefault('PYART_QUIET', '1')
    import numpy as np
    import build_index as bi
    src = next(s for s in json.load(open(os.path.join(here, 'remote_sources.json'))) if s['id'] == a.source)
    P = json.load(open(os.path.join(here, 'places.json')))
    P = {'n': P['n'], 'y': np.asarray(P['y'], float) / 1e4, 'x': np.asarray(P['x'], float) / 1e4,
         'p': np.asarray(P.get('p', [0] * len(P['n'])), float)}
    flags = {k: src[k] for k in ('nodealias', 'azOffset', 'timeShiftMs') if src.get(k)}
    tmp = os.path.join(a.out, 'tmp')
    os.makedirs(tmp, exist_ok=True)
    if a.merge:
        return merge_parts(a, src, P, bi)
    os.makedirs(a.work, exist_ok=True)
    rep = open(os.path.join(a.out, 'report.jsonl'), 'w')
    scans, k, fails = [], 0, 0
    seen_swp = set()
    t_all = time.time()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(a.procs, mp_context=ctx, initializer=_init, initargs=(here,)) as pool:
        for ui, url in enumerate(src['urls']):
            path = os.path.join(a.work, f'archive_{ui}')
            print(f'[{ui + 1}/{len(src["urls"])}] {url.rsplit("/", 2)[-2 if url.endswith("/content") else -1]}', flush=True)
            download(url, path)
            inflight = []

            def drain(limit):
                nonlocal fails
                while len(inflight) > limit:
                    fut, u, nm = inflight.pop(0)
                    rec = fut.result()
                    rec['url'] = u
                    rep.write(json.dumps({x: v for x, v in rec.items() if x != 'meta'}) + '\n')
                    if rec.get('ok'):
                        m = rec['meta']
                        if src.get('pos'):
                            m['lat'], m['lon'] = src['pos']
                        elif not (abs(m['lat']) > 0.01 and abs(m['lon']) > 0.01 and -90 <= m['lat'] <= 90):
                            print('  no usable position, skipped', nm, flush=True)
                            continue
                        ref = {'kind': 'pinned', 'url': u, 'member': nm, **flags}
                        scans.append({'ref': ref, 'name': nm, 'k': rec['k'], 'tag': radar_tag(nm), **m})
                    elif not rec.get('skip'):
                        fails += 1
                        print(f'  FAIL {nm}: {rec.get("err")}', flush=True)
            vol = {'dir': None, 'bufs': []}       # DORADE: one file per sweep, a directory per volume

            def submit(nm, buf):
                nonlocal k
                if a.of > 1 and k % a.of != a.part:     # another part's scan
                    k += 1
                    return
                inflight.append((pool.submit(work, nm, buf, flags, tmp, k), url, nm))
                k += 1
                drain(a.procs * 2)
                if k % 100 == 0:
                    print(f'  {k} scans read, {len(scans)} converted, {fails} failed, {time.time() - t_all:.0f}s', flush=True)

            def flush_vol():
                if vol['bufs']:
                    submit(vol['dir'], vol['bufs'])
                vol['dir'], vol['bufs'] = None, []
            for nm, buf in members(path):
                base = os.path.basename(nm)
                if src.get('include') and not re.search(src['include'], nm):
                    continue                          # e.g. only the DOW folders, not the NEXRAD/code in the same zip
                if src.get('exclude') and re.search(src['exclude'], nm):
                    continue
                if base.startswith('swp.'):
                    tg = radar_tag(nm)
                    if re.fullmatch(r'K[A-Z]{3}', tg or ''):
                        continue                      # WSR-88D sweeps bundled with the trucks
                    if (tg, base) in seen_swp:
                        continue                      # the same sweep kept in several folders (edited/by-elevation copies)
                    seen_swp.add((tg, base))
                if base.startswith('swp.'):
                    if 'RHI' in base.upper():
                        continue
                    # one volume = same folder + radar + volume number (swp.<time>.<radar>.<n>.<angle>_PPI_v<vol>);
                    # some archives keep every sweep of the day in one folder
                    mv = re.search(r'^swp\.\d+\.([^.]+)\..*_v(\d+)$', base)
                    d = os.path.dirname(nm) + (f'/{mv.group(1)}_v{mv.group(2)}' if mv else '')
                    if d != vol['dir']:
                        flush_vol()
                        vol['dir'] = d
                    vol['bufs'].append(buf)
                    continue
                if JUNK.search(nm) or len(buf) < 1024:
                    continue
                submit(nm, buf)
                del buf
            flush_vol()
            drain(0)
            os.remove(path)
    rep.close()
    if a.of > 1:                                     # a part: hand the scans to the merge job
        json.dump(scans, open(os.path.join(a.out, 'scans.json'), 'w'))
        print(f'{a.source} part {a.part}/{a.of}: {len(scans)} scans, {fails} failed, {time.time() - t_all:.0f}s', flush=True)
        return
    if not scans:
        sys.exit(f'{a.source}: no scans converted ({fails} failed)')
    scans, deps = group_all(src, scans, P, bi)
    i = 0
    for d in deps:
        d['pinned'] = True
        os.makedirs(os.path.join(a.out, 'cache', d['id']), exist_ok=True)
        for n in range(len(d['scans'])):
            kk = scans[i]['k']; i += 1
            for g in GROUPS:
                os.replace(os.path.join(tmp, f'{kk}.{g}.bin.gz'), os.path.join(a.out, 'cache', d['id'], f'{n}.{g}.bin.gz'))
    assert i == len(scans)
    json.dump({'source': a.source, 'deployments': deps}, open(os.path.join(a.out, f'index_{a.source}.json'), 'w'))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'{a.source}: {len(scans)} scans in {len(deps)} deployments, {fails} failed, {time.time() - t_all:.0f}s', flush=True)
    for d in deps:
        print(f"  {d['id']}: {d['radar']} near {d['place']} {time.strftime('%Y-%m-%d %H:%MZ', time.gmtime(d['start'] / 1000))}"
              f" {d['nscans']} scans, {len(d['angles'])} angles, {d['products']}", flush=True)


def radar_tag(nm):
    """Radar named in a DORADE sweep file name (swp.<time>.<RADAR>.<n>...), else ''."""
    m = re.match(r'swp\.\d+\.([^.]+)\.', os.path.basename(nm))
    if not m:                                   # a merged volume: <folder>/<RADAR>_v<n>[.nc]
        m = re.search(r'([A-Za-z0-9]+)_v\d+(\.nc)?$', nm)
    return m.group(1).upper() if m else ''


def group_all(src, scans, P, bi):
    """Deployments for a source: one radar (src['radar']), or, with src['radars'] =
    {tag: display name}, one set per radar tag found in the files. Returns the scans
    in the order the deployments list them, so cache files are renamed in step."""
    if not src.get('radars'):
        scans.sort(key=lambda s: s['t0'])
        return scans, bi.group(src, scans, P)
    order, deps = [], []
    for tag in sorted({s.get('tag') or '' for s in scans}):
        name = src['radars'].get(tag)
        if not name:
            print(f'  radar tag {tag!r} not in this source, skipped', flush=True)
            continue
        ss = sorted((s for s in scans if (s.get('tag') or '') == tag), key=lambda s: s['t0'])
        sub = {**src, 'id': f"{src['id']}-{name.lower().replace(' ', '').replace('-', '')}", 'radar': name}
        for d in bi.group(sub, ss, P):
            d['source'] = src['id']
            deps.append(d)
        order += ss
    return order, deps


def merge_parts(a, src, P, bi):
    import glob
    scans, where = [], {}
    for sj in sorted(glob.glob(os.path.join(a.merge, '*', 'scans.json'))):
        d = os.path.dirname(sj)
        for s in json.load(open(sj)):
            where[s['k']] = os.path.join(d, 'tmp')
            scans.append(s)
    if not scans:
        sys.exit(f'{a.source}: no part output found in {a.merge}')
    scans, deps = group_all(src, scans, P, bi)
    i = 0
    for d in deps:
        d['pinned'] = True
        os.makedirs(os.path.join(a.out, 'cache', d['id']), exist_ok=True)
        for n in range(len(d['scans'])):
            kk = scans[i]['k']; i += 1
            for g in GROUPS:
                shutil.move(os.path.join(where[kk], f'{kk}.{g}.bin.gz'), os.path.join(a.out, 'cache', d['id'], f'{n}.{g}.bin.gz'))
    json.dump({'source': a.source, 'deployments': deps}, open(os.path.join(a.out, f'index_{a.source}.json'), 'w'))
    print(f'{a.source}: merged {len(scans)} scans into {len(deps)} deployments', flush=True)


if __name__ == '__main__':
    main()

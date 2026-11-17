"""Minimal DORADE sweep-file reader (NCAR/EOL "swp.*" files, one sweep per file)
for the Mobile Radar Finder. volume_to_cfradial([bytes, ...]) merges one volume's
sweep files into a Py-ART Radar and returns it written as CfRadial bytes, which the
existing mobcore pipeline reads like any other CfRadial file.

Blocks used: VOLD (date), RADD (position, Nyquist, compression), PARM (field
names, scale/bias, bad value), CELV (gate ranges), SWIB (fixed angle), RYIB (ray
time/azimuth/elevation), ASIB (moving-platform position), RDAT (ray data, HRD
run-length compressed when RADD.data_compress == 1).
"""
import datetime as dt, os, struct, tempfile
import numpy as np

NAMES = {   # DORADE field name -> CfRadial standard name used by mobcore.FIELDS
    'reflectivity': ('DBZ', 'DZ', 'DBZH', 'ZH', 'DBZC', 'DCZ', 'DB_DBZ', 'DB_DBZ2'),
    'velocity': ('VEL', 'VR', 'VE', 'VG', 'DB_VEL', 'VT'),
    'differential_reflectivity': ('ZDR', 'DR', 'DB_ZDR'),
    'cross_correlation_ratio': ('RHOHV', 'RH', 'RHO', 'DB_RHOHV'),
}
LOOKUP = {d: std for std, ds in NAMES.items() for d in ds}


def _blocks(buf):
    """Yield (id, offset, length). Detects byte order from the first block length."""
    end = '>' if 0 < struct.unpack('>i', buf[4:8])[0] < 1 << 24 else '<'
    p = 0
    while p + 8 <= len(buf):
        bid = buf[p:p + 4].decode('latin1')
        n = struct.unpack(end + 'i', buf[p + 4:p + 8])[0]
        if n < 8 or p + n > len(buf):
            break
        yield bid, p, n, end
        p += n


def _hrd(words, ncells, bad):
    """HRD run-length decompression of 16-bit words."""
    out = np.full(ncells, bad, np.int32)
    i = j = 0
    while i < len(words) and j < ncells:
        w = int(words[i]) & 0xFFFF
        i += 1
        if w == 1 or w == 0:
            break
        n = w & 0x7FFF
        if w & 0x8000:
            out[j:j + n] = words[i:i + n][:max(0, ncells - j)]
            i += n
        j += n
    return out


def read_sweep(buf):
    year = None; radd = {}; parms = []; cells = None; fixed = None
    rays = []; cur = None
    for bid, p, n, e in _blocks(buf):
        b = buf[p:p + n]
        if bid == 'VOLD':
            year = struct.unpack(e + 'h', b[36:38])[0]
        elif bid == 'RADD':
            radd = {'compress': struct.unpack(e + 'h', b[68:70])[0],
                    'lon': struct.unpack(e + 'f', b[80:84])[0], 'lat': struct.unpack(e + 'f', b[84:88])[0],
                    'alt': struct.unpack(e + 'f', b[88:92])[0], 'nyq': struct.unpack(e + 'f', b[92:96])[0]}
        elif bid == 'PARM':
            parms.append({'name': b[8:16].decode('latin1').strip('\x00 '),
                          'fmt': struct.unpack(e + 'h', b[78:80])[0],
                          'scale': struct.unpack(e + 'f', b[92:96])[0], 'bias': struct.unpack(e + 'f', b[96:100])[0],
                          'bad': struct.unpack(e + 'i', b[100:104])[0]})
        elif bid == 'CELV':
            nc = struct.unpack(e + 'i', b[8:12])[0]
            cells = np.frombuffer(b[12:12 + 4 * nc], e + 'f4').astype(float)
        elif bid == 'CSFD' and cells is None:
            # cell spacing as segments: first-gate distance, then (spacing, count) pairs
            nseg = struct.unpack(e + 'i', b[8:12])[0]
            first = struct.unpack(e + 'f', b[12:16])[0]
            sp = struct.unpack(e + '8f', b[16:48])
            cnt = struct.unpack(e + '8h', b[48:64])
            d, out = first, []
            for k in range(max(0, min(nseg, 8))):
                for _ in range(cnt[k]):
                    out.append(d); d += sp[k]
            cells = np.array(out, float)
        elif bid == 'SWIB':
            fixed = struct.unpack(e + 'f', b[32:36])[0]
        elif bid == 'RYIB':
            jd, hh, mm, ss, ms = struct.unpack(e + 'ihhhh', b[12:24])
            az, el = struct.unpack(e + 'ff', b[24:32])
            cur = {'jd': jd, 't': hh * 3600 + mm * 60 + ss + ms / 1000.0, 'az': az, 'el': el, 'data': {}}
            rays.append(cur)
        elif bid == 'ASIB' and cur is not None:
            lon, lat, alt = struct.unpack(e + 'fff', b[8:20])
            cur['pos'] = (lat, lon, alt)
        elif bid == 'RDAT' and cur is not None:
            cur['data'][b[8:16].decode('latin1').strip('\x00 ')] = b[16:]
    if year is None or cells is None or not rays:
        raise ValueError('not a complete DORADE sweep')
    nc = len(cells)
    fields = {}
    for pm in parms:
        std = LOOKUP.get(pm['name'].upper())
        if not std or std in fields:
            continue
        dtype = {1: 'i1', 2: 'i2', 3: 'i4', 4: 'f4'}.get(pm['fmt'], 'i2')
        arr = np.ma.masked_all((len(rays), nc), np.float32)
        for r, ray in enumerate(rays):
            raw = ray['data'].get(pm['name'])
            if raw is None:
                continue
            e = '>' if buf[4:8] and 0 < struct.unpack('>i', buf[4:8])[0] < 1 << 24 else '<'
            if radd.get('compress') == 1 and dtype == 'i2':
                vals = _hrd(np.frombuffer(raw[:len(raw) // 2 * 2], e + 'i2'), nc, pm['bad'])
            else:
                vals = np.frombuffer(raw[:nc * np.dtype(dtype).itemsize], e + dtype).astype(np.float64)
                if len(vals) < nc:
                    vals = np.concatenate([vals, np.full(nc - len(vals), pm['bad'])])
            v = np.ma.masked_equal(np.asarray(vals, np.float64), pm['bad'])
            arr[r] = ((v - pm['bias']) / (pm['scale'] or 1.0)).astype(np.float32)
        fields[std] = arr
    base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    t = np.array([(base + dt.timedelta(days=ray['jd'] - 1, seconds=ray['t'])).timestamp() for ray in rays])
    pos = [ray.get('pos') for ray in rays if ray.get('pos') and abs(ray['pos'][0]) > 0.01]
    lat, lon = (np.median([p[0] for p in pos]), np.median([p[1] for p in pos])) if pos else (radd['lat'], radd['lon'])
    return {'t': t, 'az': np.array([r['az'] for r in rays]) % 360, 'el': np.array([r['el'] for r in rays]),
            'range': cells, 'fields': fields, 'fixed': fixed, 'lat': float(lat), 'lon': float(lon),
            'alt': float(radd.get('alt', 0)) * 1000.0, 'nyq': float(radd.get('nyq') or 0)}


def volume_radar(bufs):
    import pyart
    sw = sorted((read_sweep(b) for b in bufs), key=lambda s: s['t'][0])
    ng = max(len(s['range']) for s in sw)
    rng = max((s['range'] for s in sw), key=len)
    names = sorted({f for s in sw for f in s['fields']})
    nr = sum(len(s['t']) for s in sw)
    t0 = sw[0]['t'][0]
    fields = {}
    for f in names:
        a = np.ma.masked_all((nr, ng), np.float32)
        r = 0
        for s in sw:
            k = len(s['t'])
            if f in s['fields']:
                a[r:r + k, :s['fields'][f].shape[1]] = s['fields'][f]
            r += k
        fields[f] = {'data': a, **pyart.config.get_metadata(f)}
    starts, ends, r = [], [], 0
    for s in sw:
        starts.append(r); r += len(s['t']); ends.append(r - 1)
    def d(x, kind=None):
        m = pyart.config.get_metadata(kind) if kind else {}
        return {**m, 'data': np.asarray(x)}
    tsec = np.concatenate([s['t'] for s in sw]) - t0
    iso = dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    time = {'data': tsec, 'units': f'seconds since {iso}', 'standard_name': 'time', 'calendar': 'gregorian'}
    nyq = np.concatenate([np.full(len(s['t']), s['nyq'], np.float32) for s in sw])
    radar = pyart.core.Radar(
        time, d(rng, 'range'), fields, {'instrument_name': 'NOXP'}, 'ppi',
        d([sw[0]['lat']], 'latitude'), d([sw[0]['lon']], 'longitude'), d([sw[0]['alt']], 'altitude'),
        d(np.arange(len(sw), dtype=np.int32), 'sweep_number'), d(np.array([b'azimuth_surveillance'] * len(sw)), 'sweep_mode'),
        d(np.array([s['fixed'] for s in sw], np.float32), 'fixed_angle'),
        d(np.array(starts, np.int32), 'sweep_start_ray_index'), d(np.array(ends, np.int32), 'sweep_end_ray_index'),
        d(np.concatenate([s['az'] for s in sw]).astype(np.float32), 'azimuth'),
        d(np.concatenate([s['el'] for s in sw]).astype(np.float32), 'elevation'),
        instrument_parameters={'nyquist_velocity': {'data': nyq, 'units': 'meters_per_second'}})
    return radar


def volume_to_cfradial(bufs):
    import pyart
    radar = volume_radar(bufs)
    fd, path = tempfile.mkstemp(suffix='.nc')
    os.close(fd)
    try:
        pyart.io.write_cfradial(path, radar, format='NETCDF4')
        return open(path, 'rb').read()
    finally:
        os.remove(path)

"""NEXRAD Level II volumes -> RadarHub-format sweep files so the RadarHub pipeline
(rh_unfold / rh_mob_case / wind_loop) can build Preloaded Data cases from WSR-88D data.

Per frame (one per Doppler cut at the volume's LOWEST tilt; SAILS / MESO-SAILS rescans give
several per volume):
  <YYYYMMDD-HHMMSS>-E<el>-Z.bin  reflectivity  (from the paired surveillance cut when there is one:
                                               no range folding; else the Doppler cut's own)
                         -V.bin  velocity      (Doppler cut)
                         -D.bin  ZDR           (surveillance cut)   u8 = zdr * 16 + 128
                         -C.bin  CC            (surveillance cut)   u8 = cc * 200
All four share the Doppler cut's rays (super-res 720) and gates; gates are padded so gate i sits
at (i + 0.5) * gate width (RadarHub convention, firstGateM 0) and cropped to <crop_km>.

usage (server, Py-ART): python3 l2_to_rhbins.py <out_dir> <crop_km> <L2 file> [<L2 file> ...]
   or import and call convert(paths, out_dir, crop_km) -> number of frames
"""
import datetime as dt, json, os, struct, sys
import numpy as np
os.environ.setdefault('PYART_QUIET', '1')
import pyart

GHZ = 2.8                                     # WSR-88D (S band); case json must use the same ghz
LAM = 299792458.0 / (GHZ * 1e9)
FIELDS = {'REF': 'reflectivity', 'VEL': 'velocity', 'ZDR': 'differential_reflectivity', 'CC': 'cross_correlation_ratio'}


def _write(path, rays, gates, t, lat, lon, el, gw_km, info, az, u8):
    ib = json.dumps(info).encode()
    h = struct.pack('<hhhhddddffffffff', rays, gates, len(ib), 0, t, lat, lon, 0.0,
                    0.0, 0.0, 0.0, 0.0, el, 0.0, 0.0, gw_km)
    els = np.full(rays, int(round(el * 100)), '<i2').tobytes()
    azs = (np.round(np.asarray(az) / 360.0 * 65536) % 65536).astype('<u2').tobytes()
    open(path, 'wb').write(h + ib + els + azs + u8.astype('u1').tobytes())


def _has(r, sw, key):
    f = FIELDS[key]
    return f in r.fields and np.ma.count(r.fields[f]['data'][sw]) > 0


def convert(paths, out_dir, crop_km=150.0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in paths:
        try:
            r = pyart.io.read_nexrad_archive(f)
        except Exception as e:
            log(f'  skip {os.path.basename(f)}: {type(e).__name__}: {e}')
            continue
        lat = float(r.latitude['data'][0]); lon = float(r.longitude['data'][0])
        rng = np.asarray(r.range['data'], float)
        gw = float(rng[1] - rng[0])
        pad = int(round((rng[0] - gw / 2) / gw))
        keep = int(crop_km * 1000 // gw)
        nyq_all = r.instrument_parameters['nyquist_velocity']['data']
        fa = np.asarray(r.fixed_angle['data'], float)
        amin = float(fa.min())
        if amin >= 0.8:
            continue
        base = dt.datetime.strptime(r.time['units'].split('since ')[1].replace('Z', ''), '%Y-%m-%dT%H:%M:%S')
        low = [i for i, a in enumerate(fa) if abs(a - amin) <= 0.05]
        for i in low:
            sw = r.get_slice(i)
            if not _has(r, sw, 'VEL'):
                continue
            # the surveillance cut that goes with this Doppler cut: the nearest earlier low sweep with ZDR
            cs = next((j for j in sorted((j for j in low if j < i), reverse=True) if _has(r, r.get_slice(j), 'ZDR')), None)
            az = np.asarray(r.azimuth['data'][sw], float)
            rays = len(az)
            el = round(float(fa[i]), 1)        # the cut's nominal angle: measured elevations wobble (0.48/0.53) and would split one loop into two "tilts"
            t_of = lambda swx: (base + dt.timedelta(seconds=float(np.min(r.time['data'][swx])))).replace(tzinfo=dt.timezone.utc).timestamp()
            t_cd = t_of(sw)                                    # Doppler half: velocity was measured from here
            nyq = float(np.median(nyq_all[sw]))

            def grid(vals, nrays, scale, off):
                g = np.zeros((nrays, pad + vals.shape[1]), np.uint8)
                a2 = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(vals, np.float32)), np.nan)
                ok = np.isfinite(a2)
                sub = np.zeros(a2.shape, np.uint8)
                sub[ok] = np.clip(np.round(a2[ok] * scale + off), 1, 255).astype(np.uint8)
                g[:, pad:] = sub
                return g[:, :keep]

            info = {'wf': 'wsr88d', 'prf': 4.0 * nyq / LAM, 'nyquist': nyq, 'src': os.path.basename(f)}
            # Z + V: the Doppler half on its own beams (Z here is only the unfolding partner of V)
            files = {'Z': (az, t_cd, grid(r.fields['reflectivity']['data'][sw], rays, 2.0, 64.0)),
                     'V': (az, t_cd, grid(r.fields['velocity']['data'][sw], rays, 2.0, 128.0))}
            name_t = t_cd
            if cs is not None:
                # R / D / C: REF, ZDR, CC exactly as measured in the surveillance half -- its OWN beams and its OWN
                # start time (the time NWS stamps on the product); nothing is moved onto the Doppler half's beams
                ssw = r.get_slice(cs)
                saz = np.asarray(r.azimuth['data'][ssw], float); t_cs = t_of(ssw); nr = len(saz)
                files['R'] = (saz, t_cs, grid(r.fields['reflectivity']['data'][ssw], nr, 2.0, 64.0))
                files['D'] = (saz, t_cs, grid(r.fields['differential_reflectivity']['data'][ssw], nr, 16.0, 128.0))
                files['C'] = (saz, t_cs, grid(r.fields['cross_correlation_ratio']['data'][ssw], nr, 200.0, 0.0))
                name_t = t_cs
            name = dt.datetime.fromtimestamp(name_t, dt.timezone.utc).strftime('%Y%m%d-%H%M%S') + f'-E{round(float(fa[i]), 1)}'
            for k, (kaz, kt, g) in files.items():
                _write(os.path.join(out_dir, f'{name}-{k}.bin'), len(kaz), g.shape[1], kt, lat, lon, el, gw / 1000.0, info, kaz, g)
            n += 1
    return n


if __name__ == '__main__':
    out_dir, crop_km = sys.argv[1], float(sys.argv[2])
    print(convert(sys.argv[3:], out_dir, crop_km), 'frames')

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


def convert(paths, out_dir, crop_km=150.0, log=print, nws=None):
    """nws (optional): callable(cut_start_epoch_s) -> NWS's lowest-tilt velocity product for that cut or None,
    {'az': ray centre azimuths (deg), 'vel': (rays x bins) m/s with NaN, 'gate_km': bin size}. NWS stamps each
    low cut (SAILS rescans included) with the start of its SURVEILLANCE half, so that is the key. It is resampled
    onto the Doppler half's beams as '-N.bin' (u8 = round(v + 128), 1 m/s steps): NWS's own unfolding decision,
    used per gate by rh_mob_case.py."""
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
        # some volumes have no VCP message (e.g. KCLX 2020-04-13 10:10Z): Py-ART then reports 0.0 as the fixed angle of
        # sweeps of every height -> use that sweep's own measured elevation (median), to 0.1 deg, instead
        for k in np.where(~np.isfinite(fa) | (fa < 0.1))[0]:
            fa[k] = round(float(np.median(r.elevation['data'][r.get_slice(k)])), 1)
        amin = float(fa.min())
        if amin >= 0.8:
            continue
        base = dt.datetime.strptime(r.time['units'].split('since ')[1].replace('Z', ''), '%Y-%m-%dT%H:%M:%S')
        # legacy (pre-2008 super-res) volumes label the pair of halves 0.5 / 0.4: one tilt, so allow 0.15
        low = [i for i, a in enumerate(fa) if abs(a - amin) <= 0.15]
        for i in low:
            sw = r.get_slice(i)
            if not _has(r, sw, 'VEL'):
                continue
            # the surveillance cut that goes with this Doppler cut: the nearest earlier low sweep with ZDR
            cs = next((j for j in sorted((j for j in low if j < i), reverse=True) if _has(r, r.get_slice(j), 'ZDR')), None)
            if cs is None:                     # before dual-pol (2011-13): the surveillance half has REF only (no VEL)
                cs = next((j for j in sorted((j for j in low if j < i), reverse=True)
                           if _has(r, r.get_slice(j), 'REF') and not _has(r, r.get_slice(j), 'VEL')), None)
            az = np.asarray(r.azimuth['data'][sw], float)
            rays = len(az)
            el = round(float(fa[cs] if cs is not None else fa[i]), 1)        # the cut's nominal angle: measured elevations wobble (0.48/0.53) and would split one loop into two "tilts"
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
            zsrc = r.fields['reflectivity']['data'][sw]
            if not _has(r, sw, 'REF') and cs is not None:   # legacy Doppler halves carry no REF: the surveillance half's, nearest beam
                saz0 = np.asarray(r.azimuth['data'][r.get_slice(cs)], float)
                k = np.argmin(np.abs(((az[:, None] - saz0[None, :]) + 180) % 360 - 180), axis=1)
                zsrc = r.fields['reflectivity']['data'][r.get_slice(cs)][k]
            files = {'Z': (az, t_cd, grid(zsrc, rays, 2.0, 64.0)),
                     'V': (az, t_cd, grid(r.fields['velocity']['data'][sw], rays, 2.0, 128.0))}
            name_t = t_cd
            if cs is not None:
                # R / D / C: REF, ZDR, CC exactly as measured in the surveillance half -- its OWN beams and its OWN
                # start time (the time NWS stamps on the product); nothing is moved onto the Doppler half's beams
                ssw = r.get_slice(cs)
                saz = np.asarray(r.azimuth['data'][ssw], float); t_cs = t_of(ssw); nr = len(saz)
                files['R'] = (saz, t_cs, grid(r.fields['reflectivity']['data'][ssw], nr, 2.0, 64.0))
                if _has(r, ssw, 'ZDR'):
                    files['D'] = (saz, t_cs, grid(r.fields['differential_reflectivity']['data'][ssw], nr, 16.0, 128.0))
                if _has(r, ssw, 'CC'):
                    files['C'] = (saz, t_cs, grid(r.fields['cross_correlation_ratio']['data'][ssw], nr, 200.0, 0.0))
                name_t = t_cs
            P = nws(name_t) if nws else None
            if P is not None:                          # NWS's unfolded velocity for this very cut, on this cut's beams
                pa = np.asarray(P['az'], float); o = np.argsort(pa)
                d = np.abs(((az[:, None] - pa[o][None, :]) + 180) % 360 - 180)
                rows = np.asarray(P['vel'], float)[o][np.argmin(d, axis=1)]
                step = max(1, int(round(gw / 1000.0 / P['gate_km'])))
                ng = (pad + r.fields['velocity']['data'][sw].shape[1])
                nv = np.full((rays, ng), np.nan)
                src = rows[:, ::step]                      # NWS bin j centred at (j + 0.5) * gate, same as our padded gates
                rep = int(round(P['gate_km'] / (gw / 1000.0)))
                if rep > 1:                                # coarser NWS bins (old 1 km N0V): each covers rep of our gates
                    src = np.repeat(rows, rep, axis=1)
                m = min(ng, src.shape[1]); nv[:, :m] = src[:, :m]
                gN = np.zeros((rays, ng), np.uint8); ok = np.isfinite(nv)
                gN[ok] = np.clip(np.round(nv[ok] + 128.0), 1, 255).astype(np.uint8)
                files['N'] = (az, t_cd, gN[:, :keep])
            name = dt.datetime.fromtimestamp(name_t, dt.timezone.utc).strftime('%Y%m%d-%H%M%S') + f'-E{el}'
            for k, (kaz, kt, g) in files.items():
                _write(os.path.join(out_dir, f'{name}-{k}.bin'), len(kaz), g.shape[1], kt, lat, lon, el, gw / 1000.0, info, kaz, g)
            n += 1
    return n


if __name__ == '__main__':
    out_dir, crop_km = sys.argv[1], float(sys.argv[2])
    print(convert(sys.argv[3:], out_dir, crop_km), 'frames')

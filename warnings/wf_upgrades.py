#!/usr/bin/env python3
"""Warning Finder: every tornado / severe thunderstorm warning's HIGHEST tags across ALL its statements, per UTC day.

IEM's bulk sources keep tornadotag / damagetag only on a warning's issuance polygon; the tags of each SVS update
exist only in IEM's time snapshots (geojson/sbw.geojson?ts=). For one UTC day:
  1. api/1/vtec/sbw_interval (only_new=false, TO + SV) lists every polygon with its [polygon_begin, polygon_end)
  2. the fewest snapshot times that land inside every UPDATE polygon (greedy interval stabbing)
  3. each snapshot's TO / SV polygons -> per warning (wfo|ph|etn|year): max damage tag (1 considerable,
     2 destructive, 3 catastrophic), tornado observed, emergency
Writes /var/www/html/radar_archive/wf_upg/YYYY-MM-DD.json (static: the page and the desktop app read it directly).
Days before yesterday never change; today/yesterday are rebuilt when older than 30 min.

usage: wf_upgrades.py YYYY-MM-DD [--force]      one day
       wf_upgrades.py --range D0 D1              every day D0..D1 not yet built (runners)
       wf_upgrades.py --backfill [minutes]      newest missing days first, for at most `minutes` (cron)
"""
import datetime as dt, json, os, sys, time, urllib.request

OUT = os.environ.get('WF_UPG_OUT', '/var/www/html/radar_archive/wf_upg')   # runners: WF_UPG_OUT=out/wf_upg
IEM = 'https://mesonet.agron.iastate.edu'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder statement tags)'}
PAUSE = float(os.environ.get('WF_PAUSE', 0.6))                      # seconds between IEM requests (polite); --fast (a user is waiting): 0.25
FIRST = dt.date(2008, 1, 1)
DMG = {'CONSIDERABLE': 1, 'DESTRUCTIVE': 2, 'CATASTROPHIC': 3}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def ms(s):
    return int(dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() * 1000)


def day_file(day):
    return os.path.join(OUT, f'{day:%Y-%m-%d}.json')


def build(day):
    a = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    b = a + dt.timedelta(days=1)
    iso = lambda t: t.strftime('%Y-%m-%dT%H:%MZ')
    iv = get(f'{IEM}/api/1/vtec/sbw_interval.geojson?begints={iso(a)}&endts={iso(b)}&only_new=false&ph=TO&ph=SV')
    time.sleep(PAUSE)
    upd = []                                         # update polygons: [begin, end, product_id]
    for f in iv.get('features', []):
        p = f['properties']
        if p.get('status') in ('NEW', 'EXP', 'CAN') or not p.get('utc_polygon_begin') or not p.get('utc_polygon_end'):
            continue
        t0, t1 = ms(p['utc_polygon_begin']), ms(p['utc_polygon_end'])
        if t1 > t0:
            upd.append((t0, t1, p.get('product_id')))
    # greedy stabbing: sorted by end, one snapshot 1 min before the earliest uncovered end covers all that span it
    upd.sort(key=lambda x: x[1])
    pts, last = [], None
    for t0, t1, _ in upd:
        if last is not None and t0 <= last < t1:
            continue
        last = max(t0, t1 - 60000)
        pts.append(last)
    ev, seen = {}, set()
    for t in pts:
        gj = get(f'{IEM}/geojson/sbw.geojson?ts=' + dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'))
        time.sleep(PAUSE)
        for f in gj.get('features', []):
            p = f['properties']
            if p.get('phenomena') not in ('TO', 'SV'):
                continue
            seen.add(p.get('product_id'))
            k = f"{p['wfo']}|{p['phenomena']}|{int(p['eventid'])}|{p.get('year')}"
            e = ev.get(k, [0, 0, 0])
            e[0] = max(e[0], DMG.get((p.get('damagetag') or '').upper(), 0))
            if (p.get('tornadotag') or '').upper() == 'OBSERVED': e[1] = 1
            if p.get('is_emergency'): e[2] = 1
            ev[k] = e
    ev = {k: v for k, v in ev.items() if any(v)}     # only warnings with a tag to report
    covered = sum(1 for _, _, pid in upd if pid in seen)
    out = {'day': f'{day:%Y-%m-%d}', 'built': int(time.time()), 'updates': len(upd), 'covered': covered,
           'snapshots': len(pts), 'events': ev}
    os.makedirs(OUT, exist_ok=True)
    tmp = day_file(day) + '.tmp'
    json.dump(out, open(tmp, 'w'), separators=(',', ':'))
    os.replace(tmp, day_file(day))
    try:
        import shutil; shutil.chown(day_file(day), 'www-data', 'www-data')
    except Exception:
        pass
    return out


def fresh(day):
    f = day_file(day)
    if not os.path.exists(f):
        return False
    if day >= dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1):
        return time.time() - os.path.getmtime(f) < 1800
    return True


if __name__ == '__main__':
    if '--fast' in sys.argv:
        PAUSE = 0.25
    if sys.argv[1:2] == ['--range']:
        d, end = dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3])
        while d <= end:
            if not os.path.exists(day_file(d)):
                for k in range(3):
                    try:
                        o = build(d)
                        print(o['day'], o['updates'], 'updates', o['covered'], 'covered', o['snapshots'], 'snapshots', len(o['events']), 'tagged', flush=True)
                        break
                    except Exception as e:
                        print(d, 'retry', k, e, flush=True); time.sleep(60)
            d += dt.timedelta(days=1)
        sys.exit(0)
    if sys.argv[1:2] == ['--backfill']:
        stop = time.time() + 60 * float(sys.argv[2] if len(sys.argv) > 2 else 9)
        d = dt.datetime.now(dt.timezone.utc).date()
        while d >= FIRST and time.time() < stop:
            if not fresh(d):
                try:
                    o = build(d)
                    print(o['day'], o['updates'], 'updates', o['snapshots'], 'snapshots', len(o['events']), 'tagged', flush=True)
                except Exception as e:
                    print(d, 'FAILED', e, flush=True)
                    time.sleep(30)
            d -= dt.timedelta(days=1)
    else:
        day = dt.date.fromisoformat(sys.argv[1])
        if '--force' in sys.argv or not fresh(day):
            o = build(day)
            print(json.dumps({k: o[k] for k in ('day', 'updates', 'covered', 'snapshots')}), len(o['events']), 'tagged')
        else:
            print('fresh', day)

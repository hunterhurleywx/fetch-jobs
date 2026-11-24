#!/usr/bin/env python3
"""hhwx.me Radar Archive > Storm Report Finder: the local storm-report store, built on a runner (or the server cron).

Every IEM Local Storm Report the finder shows -- tornado, thunderstorm wind, hail, rain (not freezing rain, not
non-thunderstorm wind), 2004 on -- one file per month. IEM's lsr.geojson answers at most 10,000 reports per
request, so a span that comes back full is split in half and fetched again (down to an hour). IEM re-sends a
report in later summary LSRs: exact repeats (time, type, size, place) are dropped. Kept: what the finder draws and
its popup shows; coordinates to 0.001 deg (~100 m). Out: out/lsrstore/YYYY-MM.json (FeatureCollection).

usage: lsr_fetch.py FIRST_YEAR LAST_YEAR [MONTH]
"""
import datetime as dt, json, os, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/geojson/lsr.geojson'
UA = {'User-Agent': 'hhwx.me radar archive (storm report finder store)'}
OUT = os.environ.get('LSR_OUT', 'out/lsrstore')
PAUSE = float(os.environ.get('LSR_PAUSE', 1.0))
CAP = 10000
KEEP = ('valid', 'typetext', 'magnitude', 'unit', 'city', 'county', 'st', 'source', 'remark', 'wfo')


def category(t):
    t = (t or '').upper()
    if 'NON-TSTM' in t: return None
    if 'TORNADO' in t: return 'tornado'
    if 'HAIL' in t: return 'hail'
    if 'WND' in t or 'WIND' in t or 'GUST' in t: return 'wind'
    if 'RAIN' in t and 'FREEZING' not in t: return 'rain'
    return None


def get(a, b):
    url = f"{IEM}?sts={a:%Y-%m-%dT%H:%M:%SZ}&ets={b:%Y-%m-%dT%H:%M:%SZ}"
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return json.load(r).get('features', [])
        except Exception as e:
            print(f'  retry {a:%Y-%m-%d %H:%M} {k}: {e}', flush=True)
            time.sleep(20 * (k + 1))
    raise SystemExit(f'gave up on {a} - {b}')


def span(a, b):
    fs = get(a, b)
    time.sleep(PAUSE)
    if len(fs) >= CAP and (b - a).total_seconds() > 3600:
        mid = a + dt.timedelta(minutes=int((b - a).total_seconds() // 120))
        return span(a, mid) + span(mid, b)
    return fs


def month(y, m):
    a = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
    b = dt.datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=dt.timezone.utc)
    seen, feats = set(), []
    for f in span(a, b):
        p = f.get('properties') or {}
        c = (f.get('geometry') or {}).get('coordinates')
        if not c or not category(p.get('typetext')):
            continue
        k = (p.get('valid'), p.get('typetext'), p.get('magnitude'), c[0], c[1], p.get('city'))
        if k in seen:
            continue
        seen.add(k)
        feats.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(c[0], 3), round(c[1], 3)]},
                      'properties': {q: p[q] for q in KEEP if p.get(q) not in (None, '')}})
    os.makedirs(OUT, exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'month': f'{y}-{m:02d}', 'source': 'IEM LSR', 'features': feats},
              open(os.path.join(OUT, f'{y}-{m:02d}.json'), 'w'), separators=(',', ':'))
    return len(feats)


if __name__ == '__main__':
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    now = dt.datetime.now(dt.timezone.utc)
    total = 0
    for y in range(y0, y1 + 1):
        for m in ([int(sys.argv[3])] if len(sys.argv) > 3 else range(1, 13)):
            if (y, m) > (now.year, now.month):
                break
            n = month(y, m); total += n
            print(f'{y}-{m:02d}: {n} reports', flush=True)
    print('total', total)

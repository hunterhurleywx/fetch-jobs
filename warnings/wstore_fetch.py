#!/usr/bin/env python3
"""hhwx.me Radar Archive > Warning Finder: the local warnings store, built on a runner.

Every tornado / severe thunderstorm / flash flood warning IEM has (1991 on; storm-based polygons from Oct 2007,
the warned counties before that), one file per month: IEM geojson/sbw.geojson?sts&ets returns each warning's
ISSUANCE polygon (status NEW) with its tags and the event-lifetime maxima. Kept: what the Finder draws / filters /
shows, coordinates rounded to 0.001 deg (~100 m). Out: out/wstore/YYYY-MM.json (FeatureCollection).

usage: wstore_fetch.py FIRST_YEAR LAST_YEAR       (months after the current one are skipped)
"""
import datetime as dt, json, os, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/geojson/sbw.geojson'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder store)'}
OUT = os.environ.get('WSTORE_OUT', 'out/wstore')
PAUSE = float(os.environ.get('WSTORE_PAUSE', 1.0))
KEEP = ('wfo', 'phenomena', 'eventid', 'year', 'issue', 'expire', 'product_id', 'tornadotag', 'damagetag', 'windtag',
        'hailtag', 'is_pds', 'is_emergency', 'max_is_pds', 'max_is_emergency', 'floodtag_damage', 'max_floodtag_damage')


def rnd(c):
    return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 3), round(c[1], 3)]


def month(y, m):
    a = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
    b = dt.datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=dt.timezone.utc)
    url = f"{IEM}?sts={a:%Y-%m-%dT%H:%MZ}&ets={b:%Y-%m-%dT%H:%MZ}"
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                gj = json.load(r)
            break
        except Exception as e:
            print(f'{y}-{m:02d} retry {k}: {e}', flush=True)
            time.sleep(20 * (k + 1))
    else:
        raise SystemExit(f'{y}-{m:02d}: gave up')
    feats = []
    for f in gj.get('features', []):
        p = f.get('properties') or {}
        if p.get('phenomena') not in ('TO', 'SV', 'FF') or p.get('status') != 'NEW' or not f.get('geometry'):
            continue
        feats.append({'type': 'Feature',
                      'geometry': {'type': f['geometry']['type'], 'coordinates': rnd(f['geometry']['coordinates'])},
                      'properties': {k: p[k] for k in KEEP if p.get(k) not in (None, False, '')}})
    os.makedirs(OUT, exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'month': f'{y}-{m:02d}', 'features': feats},
              open(os.path.join(OUT, f'{y}-{m:02d}.json'), 'w'), separators=(',', ':'))
    return len(feats)


if __name__ == '__main__':
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    now = dt.datetime.now(dt.timezone.utc)
    total = 0
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            if (y, m) > (now.year, now.month):
                break
            n = month(y, m); total += n
            print(f'{y}-{m:02d}: {n} warnings', flush=True)
            time.sleep(PAUSE)
    print('total', total)

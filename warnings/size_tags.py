#!/usr/bin/env python3
"""hhwx.me Warning Finder > Advanced Search: each tornado / severe thunderstorm warning's LARGEST hail and wind tag
over its whole life -- issuance and every statement (a statement can raise HAIL...2.00IN to HAIL...3.00IN).

Tags in the text: "HAIL...3.00IN" (also "HAIL...<.75IN"), "WIND...70MPH" (also "...WIND...60 MPH"); TOR, SVR, SVS.
One IEM AFOS request per office, product and half-year.

Out: <out>/size_YYYY.json {"year", "events": {"WFO|PH|ETN|YEAR": [max hail in, max wind mph]}}  (0 = no tag)
usage: size_tags.py YEAR [--recent DAYS]
"""
import datetime as dt, json, os, re, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder hail/wind tags)'}
OUT = os.environ.get('SIZE_OUT', 'out/size')
PAUSE = float(os.environ.get('SIZE_PAUSE', 0.5))
WFOS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wfos.txt')).read().split()
VTEC = re.compile(r'/O\.(NEW|CON|EXT|EXA|EXB|COR|UPG)\.K?([A-Z]{3,4})\.(TO|SV)\.W\.(\d{4})\.(\d{6}T\d{4}Z)-')
HAIL = re.compile(r'^\s*(?:MAX\s+)?HAIL(?:\s+SIZE)?\.\.\.\s*[<>]?\s*(\d*\.\d+|\d+)\s*IN', re.M)
WIND = re.compile(r'^\s*(?:MAX\s+)?WIND(?:\s+GUST)?\.\.\.\s*[<>]?\s*(\d+)\s*MPH', re.M)


def fetch(pil, a, b):
    url = f'{IEM}?pil={pil}&sdate={a}&edate={b}&limit=9999&fmt=text'
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read().decode('latin-1')
        except Exception as e:
            print(f'  {pil} {a} retry {k}: {e}', flush=True)
            time.sleep(15 * (k + 1))
    raise SystemExit(f'gave up on {pil} {a}')


def parse(text, year, ev):
    for seg in text.split('$$'):
        vs = list(VTEC.finditer(seg))
        if not vs:
            continue
        up = seg.upper()
        h = max([float(x) for x in HAIL.findall(up)] or [0])
        w = max([int(x) for x in WIND.findall(up)] or [0])
        if not (h or w) or h > 10 or w > 250:
            continue
        for v in vs:
            yr = 2000 + int(v.group(5)[:2]) if v.group(5) != '000000T0000Z' else year
            k = f'{v.group(2)[-3:]}|{v.group(3)}|{int(v.group(4))}|{yr}'
            e = ev.get(k, [0, 0]); ev[k] = [max(e[0], h), max(e[1], w)]


if __name__ == '__main__':
    year = int(sys.argv[1])
    path = os.path.join(OUT, f'size_{year}.json')
    ev = {}
    if '--recent' in sys.argv:
        days = int(sys.argv[sys.argv.index('--recent') + 1])
        if os.path.exists(path):
            ev = json.load(open(path)).get('events', {})
        today = dt.datetime.now(dt.timezone.utc).date()
        spans = [(f'{max(today - dt.timedelta(days=days), dt.date(year, 1, 1))}', f'{today + dt.timedelta(days=1)}')]
    else:
        spans = [(f'{year}-01-01', f'{year}-07-01'), (f'{year}-07-01', f'{year + 1}-01-01')]
    for wfo in WFOS:
        for pil in ('TOR', 'SVR', 'SVS'):
            for a, b in spans:
                parse(fetch(f'{pil}{wfo}', a, b), year, ev)
                time.sleep(PAUSE)
        print(f'{year} {wfo}: {len(ev)} warnings with hail/wind tags so far', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({'year': year, 'events': ev}, open(path + '.tmp', 'w'), separators=(',', ':'))
    os.replace(path + '.tmp', path)
    print('done', year, len(ev), 'warnings with hail/wind tags')

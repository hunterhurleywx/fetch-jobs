#!/usr/bin/env python3
"""hhwx.me Warning Finder > Advanced Search: each tornado / severe thunderstorm warning's storm motion at issuance.

NWS warnings carry it only in their text: "TIME...MOT...LOC 2241Z 232DEG 50KT 4210 9258" (2008 on, with storm-based
warnings) -- the direction the storm is moving FROM, in degrees, and its speed in knots. IEM's AFOS archive returns a
whole year of one office's TOR or SVR products in one request. Each product's NEW VTEC (/O.NEW.KDMX.TO.W.0053...) ties
the motion to the warning.

Out: <out>/motion_YYYY.json  {"year", "events": {"WFO|PH|ETN|YEAR": [from_deg, knots]}}
usage: motion_tags.py YEAR [--recent DAYS]      (--recent: only the last DAYS days, merged into an existing file)
"""
import datetime as dt, json, os, re, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder storm motion)'}
OUT = os.environ.get('MOTION_OUT', 'out/motion')
PAUSE = float(os.environ.get('MOTION_PAUSE', 0.6))
WFOS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wfos.txt')).read().split()
NEW = re.compile(r'/O\.NEW\.K?([A-Z]{3,4})\.(TO|SV)\.W\.(\d{4})\.(\d{6}T\d{4}Z)-')
TML = re.compile(r'TIME\.\.\.MOT\.\.\.LOC\s+\d{4}Z\s+(\d{1,3})DEG\s+(\d{1,3})KT')


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


def parse(text, ev):
    for prod in re.split(r'\x01|\$\$', text):
        m, t = NEW.search(prod), TML.search(prod)
        if not m or not t:
            continue
        k = f'{m.group(1)[-3:]}|{m.group(2)}|{int(m.group(3))}|{2000 + int(m.group(4)[:2])}'
        ev.setdefault(k, [int(t.group(1)) % 360, int(t.group(2))])     # the issuance motion


if __name__ == '__main__':
    year = int(sys.argv[1])
    path = os.path.join(OUT, f'motion_{year}.json')
    ev = {}
    if '--recent' in sys.argv:
        days = int(sys.argv[sys.argv.index('--recent') + 1])
        if os.path.exists(path):
            ev = json.load(open(path)).get('events', {})
        today = dt.datetime.now(dt.timezone.utc).date()
        spans = [(f'{max(today - dt.timedelta(days=days), dt.date(year, 1, 1))}', f'{today + dt.timedelta(days=1)}')]
    else:
        spans = [(f'{year}-01-01', f'{year + 1}-01-01')]
    for wfo in WFOS:
        for pil in ('TOR', 'SVR'):
            for a, b in spans:
                parse(fetch(f'{pil}{wfo}', a, b), ev)
                time.sleep(PAUSE)
        print(f'{year} {wfo}: {len(ev)} warnings with a motion so far', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({'year': year, 'events': ev}, open(path + '.tmp', 'w'), separators=(',', ':'))
    os.replace(path + '.tmp', path)
    print('done', year, len(ev), 'warnings with a storm motion')

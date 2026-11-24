#!/usr/bin/env python3
"""hhwx.me Warning Finder > Advanced Search: was each warning RADAR INDICATED or OBSERVED (spotters, law enforcement,
broadcast media, emergency management, public -- any human confirmation)?

From the warning text, issuance AND every statement (a warning can turn observed later):
  tornado      TORNADO...OBSERVED | RADAR INDICATED                 (TOR, SVS)
  severe       HAIL THREAT... / WIND THREAT...OBSERVED | RADAR INDICATED   (SVR, SVS)
  flash flood  FLASH FLOOD...OBSERVED | RADAR INDICATED             (FFW, FFS)
One IEM AFOS request per office, product and half-year.

Out: <out>/obs_YYYY.json {"year", "events": {"WFO|PH|ETN|YEAR": 1 observed (at any point) | 0 radar indicated only}}
usage: obs_tags.py YEAR [--recent DAYS]
"""
import datetime as dt, json, os, re, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder observed tags)'}
OUT = os.environ.get('OBS_OUT', 'out/obs')
PAUSE = float(os.environ.get('OBS_PAUSE', 0.5))
WFOS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wfos.txt')).read().split()
VTEC = re.compile(r'/O\.(NEW|CON|EXT|EXA|EXB|COR|UPG)\.K?([A-Z]{3,4})\.(TO|SV|FF)\.W\.(\d{4})\.(\d{6}T\d{4}Z)-')
TAG = re.compile(r'(TORNADO|HAIL THREAT|WIND THREAT|FLASH FLOOD)\.\.\.(OBSERVED|RADAR INDICATED)')


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
    for seg in re.split(r'\$\$', text):
        vs = list(VTEC.finditer(seg))
        if not vs:
            continue
        tags = {(m.group(1), m.group(2)) for m in TAG.finditer(seg.upper())}
        for v in vs:
            ph = v.group(3)
            want = {'TO': ('TORNADO',), 'SV': ('HAIL THREAT', 'WIND THREAT'), 'FF': ('FLASH FLOOD',)}[ph]
            seen = [s for t, s in tags if t in want]
            if not seen:
                continue
            yr = 2000 + int(v.group(5)[:2]) if v.group(5) != '000000T0000Z' else year
            k = f'{v.group(2)[-3:]}|{ph}|{int(v.group(4))}|{yr}'
            ev[k] = max(ev.get(k, 0), 1 if 'OBSERVED' in seen else 0)


if __name__ == '__main__':
    year = int(sys.argv[1])
    path = os.path.join(OUT, f'obs_{year}.json')
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
        for pil in ('TOR', 'SVR', 'SVS', 'FFW', 'FFS'):
            for a, b in spans:
                parse(fetch(f'{pil}{wfo}', a, b), year, ev)
                time.sleep(PAUSE)
        print(f'{year} {wfo}: {len(ev)} warnings tagged so far', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({'year': year, 'events': ev}, open(path + '.tmp', 'w'), separators=(',', ':'))
    os.replace(path + '.tmp', path)
    print('done', year, len(ev), 'warnings,', sum(ev.values()), 'observed')

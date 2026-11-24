#!/usr/bin/env python3
"""hhwx.me Warning Finder: statement upgrades for one year, from the Severe Weather Statements' own text.

Every update to a tornado / severe thunderstorm warning is an SVS; its segments carry the warning's VTEC
(/O.CON.KDMX.TO.W.0037.../) and the tags as text: TORNADO...OBSERVED, TORNADO DAMAGE THREAT...CONSIDERABLE (PDS) /
CATASTROPHIC (emergency), THUNDERSTORM DAMAGE THREAT...CONSIDERABLE / DESTRUCTIVE, "TORNADO EMERGENCY". IEM's AFOS
archive returns a whole year of one office's SVS in one request (vs ~100k time snapshots). Matches the snapshot
method exactly once combined with the issuance tags (checked on DMX 2024-05-21/22).

Out: out/svs/svs_YYYY.json  {"year", "offices", "products", "events": {"WFO|PH|ETN|YEAR": [dmg 0-3, observed, emergency]}}
usage: svs_tags.py YEAR
"""
import json, os, re, sys, time, urllib.request

IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py'
UA = {'User-Agent': 'hhwx.me radar archive (warning finder statement tags)'}
OUT = os.environ.get('SVS_OUT', 'out/svs')
PAUSE = float(os.environ.get('SVS_PAUSE', 0.6))
WFOS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wfos.txt')).read().split()
VTEC = re.compile(r'/O\.(NEW|CON|EXT|EXA|EXB|CAN|EXP|COR|UPG)\.K?([A-Z]{3,4})\.(TO|SV)\.W\.(\d{4})\.(\d{6}T\d{4}Z)-(\d{6}T\d{4}Z)/')
DMG = {'CONSIDERABLE': 1, 'DESTRUCTIVE': 2, 'CATASTROPHIC': 3}


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
        vs = [m for m in VTEC.finditer(seg) if m.group(1) not in ('CAN', 'EXP')]
        if not vs:
            continue
        up = seg.upper()
        d = 0
        for m in re.finditer(r'(TORNADO|THUNDERSTORM) DAMAGE THREAT\.\.\.(CONSIDERABLE|DESTRUCTIVE|CATASTROPHIC)', up):
            d = max(d, DMG[m.group(2)])
        obs = 1 if re.search(r'TORNADO\.\.\.OBSERVED', up) else 0
        emer = 1 if 'TORNADO EMERGENCY' in up else 0
        if not (d or obs or emer):
            continue
        for m in vs:
            yr = 2000 + int(m.group(5)[:2]) if m.group(5) != '000000T0000Z' else year
            k = f'{m.group(2)[-3:]}|{m.group(3)}|{int(m.group(4))}|{yr}'
            e = ev.get(k, [0, 0, 0]); e[0] = max(e[0], d); e[1] |= obs; e[2] |= emer; ev[k] = e


if __name__ == '__main__':
    year = int(sys.argv[1])
    ev, prods = {}, 0
    for wfo in WFOS:
        # half a year per request keeps even the busiest office well under the 9,999-product limit
        for a, b in ((f'{year}-01-01', f'{year}-07-01'), (f'{year}-07-01', f'{year + 1}-01-01')):
            t = fetch(f'SVS{wfo}', a, b)
            n = t.count('\x01') or t.count('Severe Weather Statement')
            if n >= 9999:
                print(f'  WARNING {wfo} {a}: hit the product limit', flush=True)
            prods += n
            parse(t, year, ev)
            time.sleep(PAUSE)
        print(f'{year} {wfo}: {len(ev)} upgraded warnings so far', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({'year': year, 'offices': len(WFOS), 'products': prods, 'events': ev},
              open(os.path.join(OUT, f'svs_{year}.json'), 'w'), separators=(',', ':'))
    print('done', year, prods, 'statements,', len(ev), 'warnings with statement tags')

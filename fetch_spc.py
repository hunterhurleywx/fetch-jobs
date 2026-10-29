"""Fetch the SPC files a case needs into cases/<date>/spc/ (same names as the hand-built cases).

    python fetch_spc.py 2024-03-25 2024-09-12 ...

Per convective day:
  day{4..8}_issued_YYYYMMDD.geojson + .txt     Day 4-8 probabilities and discussion
  day3_0730z_issued_YYYYMMDD_{cat,prob}.geojson + .txt
  day2_0600z_issued_YYYYMMDD_{cat,torn,wind,hail}.geojson + .txt
  day1_{0600z,1300z,1630z,2000z}_{cat,torn,wind,hail}.geojson + .txt   (0600z = SPC's "_1200" file)
  reports_YYYYMMDD.csv     tidy filtered reports (type,date_utc,time_utc,magnitude,location,county,state,lat,lon,comments)
  _sources.json            file -> URL -> note
"""
import html
import json
import os
import re
import sys
import time
from datetime import date, timedelta

import requests

from config import CASES_DIR

SPC = "https://www.spc.noaa.gov"
S = requests.Session()
S.headers["User-Agent"] = "fetch-jobs severe-weather research (github.com/hunterhurleywx/fetch-jobs)"
HAZ = ("cat", "torn", "wind", "hail")


def get(url, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.ok:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


def pre_text(page_url):
    """Discussion text from an SPC archive page (the <pre> block)."""
    r = get(page_url)
    if r is None:
        return None
    m = re.search(r"<pre>(.*?)</pre>", r.text, re.S | re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() + "\n" if m else None


def save(out, name, url, payload, note, sources):
    if payload is None:
        sources.append([name, url, "MISSING (404 or unavailable)"])
        return
    with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
        fh.write(payload)
    sources.append([name, url, note])


def normalize(geojson_text):
    """SPC's raw .lyr.geojson -> the layout the readers use: properties["level"] = LABEL
    ("0.15", "SLGT", "SIGN", ...), and drop the empty placeholder features SPC writes for "no area"."""
    g = json.loads(geojson_text)
    keep = []
    for f in g.get("features", []):
        p = f.get("properties", {})
        lab = str(p.get("level", p.get("LABEL", ""))).strip()
        if not lab or not f.get("geometry"):
            continue
        p["level"] = lab
        keep.append(f)
    g["features"] = keep
    return json.dumps(g)


def labels(geojson_text):
    try:
        feats = json.loads(geojson_text).get("features", [])
    except ValueError:
        return "unparseable"
    return ", ".join(str(f["properties"].get("LABEL")) for f in feats) or "no areas"


WINTER_TIME = {"0730": "0830", "0600": "0700"}  # Day 3 / Day 2 issue an hour later (UTC) in standard time


def otlk(out, sources, day, issue, hhmm, name_stem, kinds, page):
    if day in (2, 3) and get(f"{SPC}/products/outlook/archive/{issue:%Y}/day{day}otlk_{issue:%Y%m%d}_{hhmm}_cat.lyr.geojson") is None \
            and hhmm in WINTER_TIME:
        alt = WINTER_TIME[hhmm]
        page = page.replace(f"_{hhmm}.html", f"_{alt}.html")
        hhmm = alt  # saved under the usual (summer-time) file names so the readers find it
    base = f"{SPC}/products/outlook/archive/{issue:%Y}/day{day}otlk_{issue:%Y%m%d}_{hhmm}"
    for k in kinds:
        url = f"{base}_{k}.lyr.geojson"
        r = get(url)
        save(out, f"{name_stem}_{k}.geojson", url, normalize(r.text) if r else None, labels(r.text) if r else "",
             sources)
    save(out, f"{name_stem}.txt", page, pre_text(page), "discussion", sources)


def reports(conv_day):
    """Tidy rows from the three filtered CSVs. Comments may hold unquoted commas, so split positionally."""
    rows, urls = [], []
    for haz, typ in (("torn", "tornado"), ("wind", "wind"), ("hail", "hail")):
        url = f"{SPC}/climo/reports/{conv_day:%y%m%d}_rpts_filtered_{haz}.csv"
        urls.append(url)
        r = get(url)
        if r is None:
            continue
        for line in r.text.splitlines()[1:]:
            f = line.split(",", 7)
            if len(f) < 7:
                continue
            t = f[0].strip().zfill(4)
            d = conv_day + timedelta(days=1) if int(t) < 1200 else conv_day
            comment = f[7].strip() if len(f) > 7 else ""
            vals = [typ, d.isoformat(), t, f[1], f[2], f[3], f[4], f[5], f[6], comment]
            rows.append(",".join(v if "," not in v and '"' not in v else '"' + v.replace('"', '""') + '"'
                                 for v in vals))
    head = "type,date_utc,time_utc,magnitude,location,county,state,lat,lon,comments"
    return "\n".join([head] + rows) + "\n", urls, len(rows)


def fetch_case(conv_day):
    out = os.path.join(CASES_DIR, conv_day.isoformat(), "spc")
    os.makedirs(out, exist_ok=True)
    sources = []
    for n in range(8, 3, -1):
        issue = conv_day - timedelta(days=n - 1)
        url = f"{SPC}/products/exper/day4-8/archive/{issue:%Y}/day{n}prob_{issue:%Y%m%d}.lyr.geojson"
        r = get(url)
        save(out, f"day{n}_issued_{issue:%Y%m%d}.geojson", url, normalize(r.text) if r else None,
             labels(r.text) if r else "", sources)
        page = f"{SPC}/products/exper/day4-8/archive/{issue:%Y}/day4-8_{issue:%Y%m%d}.html"
        save(out, f"day{n}_issued_{issue:%Y%m%d}.txt", page, pre_text(page), "Day 4-8 discussion", sources)
    d3 = conv_day - timedelta(days=2)
    otlk(out, sources, 3, d3, "0730", f"day3_0730z_issued_{d3:%Y%m%d}", ("cat", "prob"),
         f"{SPC}/products/outlook/archive/{d3:%Y}/day3otlk_{d3:%Y%m%d}_0730.html")
    d2 = conv_day - timedelta(days=1)
    otlk(out, sources, 2, d2, "0600", f"day2_0600z_issued_{d2:%Y%m%d}", HAZ,
         f"{SPC}/products/outlook/archive/{d2:%Y}/day2otlk_{d2:%Y%m%d}_0600.html")
    for hhmm, stem in (("1200", "day1_0600z"), ("1300", "day1_1300z"), ("1630", "day1_1630z"), ("2000", "day1_2000z")):
        otlk(out, sources, 1, conv_day, hhmm, stem, HAZ,
             f"{SPC}/products/outlook/archive/{conv_day:%Y}/day1otlk_{conv_day:%Y%m%d}_{hhmm}.html")
    text, urls, n = reports(conv_day)
    save(out, f"reports_{conv_day:%Y%m%d}.csv", " ; ".join(urls), text,
         f"{n} filtered reports; wind speed in mph; times 0000-1159 are the next UTC day", sources)
    with open(os.path.join(out, "_sources.json"), "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=1)
    missing = [s[0] for s in sources if s[2].startswith("MISSING")]
    print(f"{conv_day}: {len(sources) - len(missing)} files, {n} reports"
          + (f", missing: {', '.join(missing)}" if missing else ""), flush=True)


def renormalize_case(case):
    """Fix geojson files saved before normalize() existed (no re-download)."""
    d = os.path.join(CASES_DIR, case, "spc")
    n = 0
    for name in os.listdir(d):
        if name.endswith(".geojson"):
            p = os.path.join(d, name)
            with open(p, encoding="utf-8") as fh:
                txt = fh.read()
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(normalize(txt))
            n += 1
    print(f"{case}: normalized {n} geojson files")


if __name__ == "__main__":
    if sys.argv[1] == "--renormalize":
        for arg in sys.argv[2:]:
            renormalize_case(arg)
    else:
        for arg in sys.argv[1:]:
            fetch_case(date.fromisoformat(arg))

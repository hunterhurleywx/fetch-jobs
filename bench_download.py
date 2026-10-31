"""Download / decode throughput benchmark, to find where a case's download time goes before optimising.

    python bench_download.py --init 2026103100 --fh 72 --members 20

For one EPS step (eps_pl: 7 levels x 5 variables per member, from the Google mirror) and one GEFS step (main fields,
from AWS), times: (1) byte-range download only, at several thread counts; (2) decode only (eccodes, processes);
(3) both together as fetch.download_all does them. Prints MB/s, requests/s and seconds per request, and whether the
run is network-, request-latency- or CPU-bound. Writes nothing to the cache.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import fetch
from config import EC_PL_FIELDS, GEFS_FIELDS_A


def eps_ranges(init, fh, members):
    """(url, start, end) for every eps_pl field of the given members (the ECMWF .index JSON lines)."""
    base = fetch.ec_base(init, fh, "eps")
    import json
    idx = fetch.http_get(base + ".index").decode().splitlines()
    want = set(EC_PL_FIELDS.values())  # (param, levtype, levelist) as get_ecmwf matches them
    out = []
    for line in idx:
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("type") != "pf" or int(j.get("number", 0)) not in members:
            continue
        if (j["param"], j["levtype"], j.get("levelist")) in want:
            out.append((base + ".grib2", j["_offset"], j["_offset"] + j["_length"] - 1))
    return out


def gefs_ranges(init, fh, members):
    out = []
    for m in members:
        url = fetch.gefs_url(init, fh, m, "a")
        idx = fetch.http_get(url + ".idx").decode()
        for s, e, _ in fetch.ncep_ranges(idx, GEFS_FIELDS_A, fh):
            out.append((url, s, e))
    return out


def timed_download(ranges, threads):
    fetch.STATS.update(bytes=0, requests=0, retries=0)
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        blobs = list(ex.map(lambda r: fetch.http_get(r[0], (r[1], r[2])), ranges))
    dt = time.time() - t0
    mb = sum(len(b) for b in blobs) / 1e6
    return dt, mb, blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="2026103100")
    ap.add_argument("--fh", type=int, default=72)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--threads", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    init = datetime.strptime(a.init, "%Y%m%d%H")
    print(f"cores {os.cpu_count()}", flush=True)
    sets = {"EPS eps_pl (Google mirror)": eps_ranges(init, a.fh, set(range(1, a.members + 1))),
            "GEFS main a-file (AWS)": gefs_ranges(init, a.fh, fetch.GEFS_MEMBERS[:a.members])}
    for name, ranges in sets.items():
        print(f"\n== {name}: {len(ranges)} byte ranges", flush=True)
        blobs = None
        for th in a.threads:
            dt, mb, blobs = timed_download(ranges, th)
            print(f"  download {th:4d} threads: {dt:6.1f} s  {mb / dt:7.1f} MB/s  {len(ranges) / dt:6.1f} req/s  "
                  f"{dt * th / len(ranges):5.2f} s/request/thread  retries {fetch.STATS['retries']}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(a.procs) as ex:
            list(ex.map(fetch._decode, blobs, chunksize=8))
        dd = time.time() - t0
        print(f"  decode only ({a.procs} procs): {dd:6.1f} s  {len(blobs) / dd:6.1f} fields/s  "
              f"({dd * a.procs / len(blobs) * 1000:5.1f} ms CPU per field)", flush=True)


if __name__ == "__main__":
    main()

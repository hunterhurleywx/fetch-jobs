"""Download (and cache) every model step a case needs.

    python download_case.py 2025-05-16 [--leads 7 6 5] [--models gfs gefs hres eps]
"""
import argparse
import os
import sys
from datetime import datetime

from config import CACHE_DIR, MODELS, case_setup, fh_available
import fetch


def case_tasks(conv_day, leads, models):
    valid, pools = case_setup(conv_day, leads)
    inits = sorted({i for p in pools.values() for i in p})
    tasks = []
    for model in models:
        for init in inits:
            for v in valid:
                fh = int((v - init).total_seconds() // 3600)
                if fh_available(model, init, fh):
                    tasks.append((model, init, fh))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", default=[7, 6, 5])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--ncep-workers", type=int, default=24)
    ap.add_argument("--ec-workers", type=int, default=8)
    ap.add_argument("--decode-procs", type=int, default=8)
    a = ap.parse_args()
    fetch.init_decode_pool(a.decode_procs)
    conv_day = datetime.strptime(a.date, "%Y-%m-%d")
    tasks = case_tasks(conv_day, a.leads, a.models)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for m in a.models:
        print(f"  {m}: {sum(t[0] == m for t in tasks)} model-steps")
    secs = fetch.download_all(tasks, a.ncep_workers, a.ec_workers,
                              log=lambda s: print(s, flush=True))
    print(f"DONE in {secs / 60:.1f} min, {fetch.STATS['bytes'] / 1e9:.2f} GB, "
          f"{fetch.STATS['requests']} requests, {fetch.STATS['retries']} retries", flush=True)


if __name__ == "__main__":
    sys.exit(main())

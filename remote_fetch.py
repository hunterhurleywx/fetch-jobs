"""Fetch a case through GitHub Actions instead of the home connection.

    python remote_fetch.py 2025-05-16 [--leads 7 6 5] [--models gfs gefs hres eps]

Triggers the fetch-case workflow, waits for it, downloads the cropped-field release
assets, unpacks them into the local cache (existing files are kept) and deletes the
temporary release.
"""
import argparse
import glob
import json
import os
import subprocess
import tarfile
import tempfile
import time

from config import CACHE_DIR

REPO = "hunterhurleywx/pogs-fetch"


def gh(*args, capture=True):
    r = subprocess.run(["gh", *args], capture_output=capture, text=True, check=True)
    return r.stdout.strip() if capture else None


def trigger(date, leads, models):
    before = {r["databaseId"] for r in json.loads(gh("run", "list", "-R", REPO, "-w", "fetch.yml",
                                                       "--json", "databaseId", "-L", "50"))}
    gh("workflow", "run", "fetch.yml", "-R", REPO, "-f", f"date={date}",
       "-f", f"leads={' '.join(leads)}", "-f", f"models={' '.join(models)}")
    while True:  # the new run takes a few seconds to appear
        time.sleep(3)
        runs = json.loads(gh("run", "list", "-R", REPO, "-w", "fetch.yml", "--json", "databaseId", "-L", "10"))
        new = [r["databaseId"] for r in runs if r["databaseId"] not in before]
        if new:
            return new[0]


def collect(date, run_id, log=print):
    """Wait for a run, then download, unpack and delete its release."""
    t0 = time.time()
    subprocess.run(["gh", "run", "watch", str(run_id), "-R", REPO, "--exit-status", "-i", "30"],
                   check=True, stdout=subprocess.DEVNULL)
    log(f"[{date}] cloud fetch finished ({(time.time() - t0) / 60:.1f} min wait); downloading")
    tag = f"cache-{date}-{run_id}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(CACHE_DIR.rstrip("\\/")) or None) as tmp:
        t1 = time.time()
        gh("release", "download", tag, "-R", REPO, "-D", tmp, capture=False)
        parts = sorted(glob.glob(os.path.join(tmp, "cache.tar.part*")))
        tar_path = os.path.join(tmp, "cache.tar")
        with open(tar_path, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
                os.remove(p)
        log(f"[{date}] downloaded {os.path.getsize(tar_path) / 1e6:.0f} MB in {time.time() - t1:.0f}s")
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and not os.path.exists(os.path.join(CACHE_DIR, m.name))]
            tf.extractall(CACHE_DIR, members=members, filter="data")
        log(f"[{date}] added {len(members)} files to {CACHE_DIR}")
    gh("release", "delete", tag, "-R", REPO, "--yes", "--cleanup-tag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--leads", nargs="+", default=["7", "6", "5"])
    ap.add_argument("--models", nargs="+", default=["gfs", "gefs", "hres", "eps"])
    a = ap.parse_args()
    t0 = time.time()
    runs = [(d, trigger(d, a.leads, a.models)) for d in a.dates]  # cloud runs go in parallel
    for d, rid in runs:
        print(f"[{d}] run {rid} started", flush=True)
    for d, rid in runs:
        collect(d, rid, log=lambda s: print(s, flush=True))
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

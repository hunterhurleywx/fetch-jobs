"""Decoded-field cache in Cloudflare R2, so reruns skip the raw-GRIB download.

    python r2_cache.py pull 2025-03-14 --leads 8 7 6 5 4 3 2 1 --models gfs gefs ... [--manifest m.json]
    python r2_cache.py push 2025-03-14 --leads ...            --models ...  [--manifest m.json]

One object per model and cycle: cache/<model>/<YYYYMMDDHH>.tar, holding that cycle's decoded 0.5 deg
npz files (already compressed, so the tar is not). `pull` fetches the bundles this case needs that exist;
`push` uploads bundles that are new or have gained files since the pull (a cycle first used for Day 4 can
later gain Day 2 hours). _MISSING markers are never uploaded, so an unpublished real-time cycle is
retried later instead of being remembered as missing.
Credentials: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET (env). Without them it
does nothing, so runs without R2 still work.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import CACHE_DIR
from download_case import case_tasks

KEYS = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT")


def client():
    if not all(os.environ.get(k) for k in KEYS):
        return None
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                        config=Config(max_pool_connections=32, retries={"max_attempts": 5}))


def cycles(date, leads, models):
    conv_day = datetime.strptime(date, "%Y-%m-%d")
    return sorted({(m, i) for m, i, _ in case_tasks(conv_day, leads, models)})


def local_files(model, init):
    d = os.path.join(CACHE_DIR, model, f"{init:%Y%m%d%H}")
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.endswith(".npz") and not f.endswith(".tmp.npz")]
    return out


def key_of(model, init):
    return f"cache/{model}/{init:%Y%m%d%H}.tar"


def pull_one(s3, bucket, model, init, tries=5):
    for attempt in range(tries):  # a dropped stream (IncompleteRead) killed a whole case run on 31 Oct: retry it
        try:
            body = s3.get_object(Bucket=bucket, Key=key_of(model, init))["Body"].read()
            break
        except s3.exceptions.NoSuchKey:
            return 0
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    with tarfile.open(fileobj=io.BytesIO(body)) as tf:
        members = [m for m in tf.getmembers() if m.isfile()
                   and not os.path.exists(os.path.join(CACHE_DIR, m.name))]
        tf.extractall(CACHE_DIR, members=members, filter="data")
    return len(tarfile.open(fileobj=io.BytesIO(body)).getmembers())


def push_one(s3, bucket, model, init):
    files = local_files(model, init)
    if not files:
        return 0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in files:
            tf.add(p, arcname=os.path.relpath(p, CACHE_DIR).replace("\\", "/"))
    s3.put_object(Bucket=bucket, Key=key_of(model, init), Body=buf.getvalue())
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pull", "push"])
    ap.add_argument("date")
    ap.add_argument("--leads", type=int, nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default=os.path.join(CACHE_DIR, "r2_manifest.json"),
                    help="file counts per bundle at pull time (push uploads only what grew)")
    a = ap.parse_args()
    s3 = client()
    if s3 is None:
        print("R2 credentials not set: skipping")
        return 0
    bucket = os.environ.get("R2_BUCKET", "hhwx-model-cache")
    todo = cycles(a.date, a.leads, a.models)
    if a.action == "pull":
        with ThreadPoolExecutor(16) as ex:
            counts = list(ex.map(lambda mi: pull_one(s3, bucket, *mi), todo))
        manifest = {key_of(m, i): n for (m, i), n in zip(todo, counts)}
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(a.manifest, "w") as f:
            json.dump(manifest, f)
        hit = sum(1 for n in counts if n)
        print(f"pulled {hit}/{len(todo)} bundles ({sum(counts)} files) from r2://{bucket}")
    else:
        manifest = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {}
        grew = [(m, i) for m, i in todo if len(local_files(m, i)) > manifest.get(key_of(m, i), 0)]
        with ThreadPoolExecutor(16) as ex:
            counts = list(ex.map(lambda mi: push_one(s3, bucket, *mi), grew))
        print(f"pushed {len(grew)} new/grown bundles ({sum(counts)} files) to r2://{bucket}; "
              f"{len(todo) - len(grew)} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

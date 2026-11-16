#!/usr/bin/env python3
"""Push a shard's converted scans (out/cache/<dep>/<n>.<group>.bin.gz) to the
hhwx.me mobile-radar cache, a tar of up to 150 files per request."""
import io, os, sys, tarfile, time
import requests

SRV = os.environ.get('MOBILE_SRV', 'https://hhwx.me/api/mobile')
TOK = os.environ['MOBILE_SRC_TOKEN'].strip()
root = os.path.join(sys.argv[1] if len(sys.argv) > 1 else 'out', 'cache')
CHUNK = 150
stored = rejected = 0
if os.path.isdir(root):
    for dep in sorted(os.listdir(root)):
        files = sorted(os.listdir(os.path.join(root, dep)))
        for i in range(0, len(files), CHUNK):
            bio = io.BytesIO()
            with tarfile.open(fileobj=bio, mode='w') as tf:
                for name in files[i:i + CHUNK]:
                    tf.add(os.path.join(root, dep, name), arcname=name)
            body = bio.getvalue()
            for attempt in range(4):
                try:
                    r = requests.post(f'{SRV}/ingest/{TOK}/{dep}', data=body, timeout=600,
                                      headers={'Content-Type': 'application/x-tar'})
                    r.raise_for_status()
                    j = r.json()
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    print(f'retry {dep} chunk {i}: {e}', flush=True)
                    time.sleep(10 * (attempt + 1))
            stored += j['stored']
            rejected += j['rejected']
            print(f'{dep} [{i}:{i + CHUNK}] stored {j["stored"]} rejected {j["rejected"]}', flush=True)
print(f'pushed: {stored} stored, {rejected} rejected')
if rejected:
    sys.exit(1)

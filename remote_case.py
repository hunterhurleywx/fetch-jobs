"""Run whole cases in the cloud (run-case workflow) and bring home only the results.

    python remote_case.py 2025-03-14 2025-03-15 ... [--leads 8 7 6 5 4 3 2 1] [--max-parallel 20]

Each date is one GitHub Actions job (SPC files, downloads, scoring, verification, CI layer).
The job's artifact (the case folder, ~50 MB) is downloaded into cases/<date>/, overwriting the
files it contains. Raw model data never comes home. Maps are drawn locally afterwards
(plot_vs_spc.py etc.) from the small result files.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from config import CASES_DIR

REPO = "hunterhurleywx/fetch-jobs"
WORKFLOW = "case.yml"


def gh(*args, capture=True, check=True):
    r = subprocess.run(["gh", *args], capture_output=capture, text=True, check=check)
    return r.stdout.strip() if capture else None


def runs(limit=100):
    return json.loads(gh("run", "list", "-R", REPO, "-w", WORKFLOW, "--json",
                         "databaseId,displayTitle,status,conclusion", "-L", str(limit)))


def trigger(date, leads):
    """Dispatch one case and return its run id, matched by run-name."""
    title = f"case {date} | {' '.join(leads)}"
    before = {r["databaseId"] for r in runs()}
    gh("workflow", "run", WORKFLOW, "-R", REPO, "-f", f"date={date}", "-f", f"leads={' '.join(leads)}")
    while True:
        time.sleep(3)
        new = [r["databaseId"] for r in runs(30) if r["databaseId"] not in before and r["displayTitle"] == title]
        if new:
            return new[0]


def collect(date, run_id, log=print):
    """Download the run's artifact into cases/<date>/ (whatever the job managed to produce)."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["gh", "run", "download", str(run_id), "-R", REPO, "-n", f"case-{date}", "-D", tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(f"[{date}] no artifact: {r.stderr.strip()[:200]}")
            return False
        dest = os.path.join(CASES_DIR, date)
        n = 0
        for root, _, files in os.walk(tmp):
            for f in files:
                src = os.path.join(root, f)
                out = os.path.join(dest, os.path.relpath(src, tmp))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if f.endswith(".npz") and os.path.exists(out):
                    # merge, so a run with fewer leads (e.g. tonight's Day 4) keeps earlier leads' keys
                    with np.load(out) as old, np.load(src) as new:
                        merged = {k: old[k] for k in old.files}
                        merged.update({k: new[k] for k in new.files})
                    np.savez_compressed(out, **merged)
                else:
                    shutil.copy2(src, out)
                n += 1
        log(f"[{date}] {n} files -> {dest}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--leads", nargs="+", default=["8", "7", "6", "5", "4", "3", "2", "1"])
    ap.add_argument("--max-parallel", type=int, default=20, help="GitHub runs at most 20 jobs at once (Free plan)")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    queue, active = list(a.dates), {}
    while queue or active:
        while queue and len(active) < a.max_parallel:
            d = queue.pop(0)
            active[d] = trigger(d, a.leads)
            log(f"[{d}] run {active[d]} started")
        time.sleep(30)
        status = {r["databaseId"]: r for r in runs()}
        for d, rid in list(active.items()):
            r = status.get(rid)
            if r and r["status"] == "completed":
                log(f"[{d}] {r['conclusion']} after {(time.time() - t0) / 60:.0f} min")
                collect(d, rid, log)
                del active[d]
    log(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

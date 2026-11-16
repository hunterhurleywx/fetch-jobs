#!/usr/bin/env python3
"""Summarise every shard's report.jsonl: per deployment scans ok / failed /
warned, the failures with their errors, and a markdown table for the job page."""
import collections, glob, json, sys

recs = []
for p in glob.glob(f'{sys.argv[1] if len(sys.argv) > 1 else "reports"}/**/report.jsonl', recursive=True):
    recs += [json.loads(l) for l in open(p) if l.strip()]
by = collections.defaultdict(list)
for r in recs:
    by[r['dep']].append(r)
lines = ['| deployment | scans | ok | failed | warned | tilts | sec/scan |', '|---|---|---|---|---|---|---|']
out = {'deployments': {}, 'failures': [], 'warnings': []}
for dep in sorted(by):
    rs = by[dep]
    ok = [r for r in rs if r['ok']]
    bad = [r for r in rs if not r['ok']]
    warned = [r for r in ok if r.get('warn')]
    tilts = sorted({r.get('tilts') for r in ok})
    sec = sum(r['sec'] for r in rs) / max(1, len(rs))
    lines.append(f'| {dep} | {len(rs)} | {len(ok)} | {len(bad)} | {len(warned)} | {",".join(map(str, tilts[:4]))} | {sec:.1f} |')
    out['deployments'][dep] = {'scans': len(rs), 'ok': len(ok), 'failed': len(bad), 'warned': len(warned)}
    out['failures'] += [{'dep': dep, 'n': r['n'], 'err': r['err']} for r in bad]
    out['warnings'] += [{'dep': dep, 'n': r['n'], 'warn': r['warn']} for r in warned]
tot_ok = sum(v['ok'] for v in out['deployments'].values())
tot_bad = sum(v['failed'] for v in out['deployments'].values())
md = [f'## Mobile radar conversion: {tot_ok} scans ok, {tot_bad} failed, {len(out["warnings"])} with warnings', ''] + lines
if out['failures']:
    md += ['', '### Failures (first 40)'] + [f'- {f["dep"]} #{f["n"]}: {f["err"]}' for f in out['failures'][:40]]
if out['warnings']:
    cnt = collections.Counter((w['dep'], '; '.join(w['warn'])) for w in out['warnings'])
    md += ['', '### Warnings'] + [f'- {d}: {w} (x{c})' for (d, w), c in cnt.most_common(40)]
open('summary.json', 'w').write(json.dumps(out, indent=1))
open('summary.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))

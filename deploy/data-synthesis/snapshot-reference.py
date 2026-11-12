#!/usr/bin/env python3
"""Snapshot the coordinator's EXISTING dedup index without stopping writers.

Run on the coordinator host; stdlib only. This does not invent a baseline index.
The coordinator must ensure the input registry covers original training data and
previously accepted synthetic data. The output contains fingerprints, not videos.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'runtime'))
from coverage_sampling import compact

p = argparse.ArgumentParser()
p.add_argument('--registry', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--original-root', type=Path, help='LeRobot root containing task folders/meta/source_records.json')
p.add_argument('--manifests', type=Path, nargs='*', default=[], help='Accepted synthetic manifests, never evaluation outputs')
args = p.parse_args()
# Build compact scene coverage before creating an output directory. Videos,
# absolute data paths, prompts and evaluation seeds are not distributed.
coverage = []
if args.original_root:
    for source in sorted(args.original_root.glob('*/meta/source_records.json')):
        for row in json.loads(source.read_text()):
            r = compact(row)
            if r['task'] in ('pick_attribute', 'spatial_directions') and r['target_uuid']:
                coverage.append(r)
    if not coverage:
        raise ValueError('No color/spatial training source_records found')
for manifest in args.manifests:
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list):
        raise ValueError('Supply accepted candidate manifests, not status/evaluation files')
    for row in rows:
        if not row.get('raw_gate_pass') or not row.get('dedup_gate_pass'):
            raise ValueError('Coverage manifest includes an unaccepted candidate')
        r = compact(row)
        if r['task'] in ('pick_attribute', 'spatial_directions'):
            coverage.append(r)
args.output.mkdir(parents=True, exist_ok=False)
dst = args.output/'registry.sqlite3'
with sqlite3.connect(f'file:{args.registry.resolve()}?mode=ro', uri=True) as original:
    db = sqlite3.connect(dst)
    original.backup(db)
count = db.execute('SELECT count(*) FROM records').fetchone()[0]
if count == 0:
    raise RuntimeError('Empty baseline index; do not distribute it')
with db:
    db.execute('DELETE FROM decisions')
    # windows has a (digest, source) PK; add a temporary source index so this
    # operation is linear-ish rather than scanning all windows for every file.
    db.execute('CREATE INDEX hbot_snapshot_sources ON windows(source)')
    sources = [r[0] for r in db.execute('SELECT source FROM records')]
    for source in sources:
        opaque = 'reference:' + hashlib.sha256(source.encode()).hexdigest()
        db.execute('UPDATE windows SET source=? WHERE source=?', (opaque, source))
        db.execute('UPDATE records SET source=? WHERE source=?', (opaque, source))
    kinds = dict(db.execute('SELECT kind,count(*) FROM records GROUP BY kind'))
    db.execute('DROP INDEX hbot_snapshot_sources')
db.execute('VACUUM')
db.close()
h = hashlib.sha256()
with dst.open('rb') as f:
    for block in iter(lambda: f.read(4*1024*1024), b''):
        h.update(block)
(args.output/'reference.json').write_text(json.dumps(dict(
    records=count, kinds=kinds, sha256=h.hexdigest(),
    scope='snapshot_of_supplied_registry',
    warning='Coverage is only as complete as the supplied coordinator index.'), indent=2))
(args.output/'coverage.json').write_text(json.dumps(dict(schema=1, records=coverage,
    original_training_included=bool(args.original_root),
    synthetic_manifests=len(args.manifests)), indent=2))
print(args.output)

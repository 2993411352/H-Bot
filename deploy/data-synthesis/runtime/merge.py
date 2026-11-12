"""Read-only source audit and cross-worker dedup. Writes a NEW report only."""
import argparse
import json
from pathlib import Path
import sqlite3
from common import contained, require, sha, save, seed_for
from dedup_registry import describe_mcap, compare, insert


def reference_db(destination):
    source = Path('/reference/registry.sqlite3')
    require(source.is_file(), 'A shared baseline fingerprint index is required')
    with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as src:
        db = sqlite3.connect(str(destination))
        src.backup(db)
    return db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', nargs='+', type=int, required=True)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    require(len(set(args.workers)) == len(args.workers), 'Repeated worker ID')
    root = Path('/work')
    report = root/'merged'/args.name
    report.mkdir(parents=True, exist_ok=False)
    db = reference_db(report/'registry.sqlite3')
    accepted, rejected, seeds, files = [], [], set(), set()
    split_hash = None
    for worker in sorted(args.workers):
        run = root/f'worker_{worker:04d}'
        plan = json.loads((run/'plan.json').read_text())
        require(plan['worker'] == worker, 'Worker ID differs from directory')
        current_split = plan['inventory']['split_sha256']
        require(split_hash in (None, current_split), 'Workers used different seen asset splits')
        split_hash = current_split
        for row in json.loads((run/'manifest.json').read_text()):
            src = contained(run, row['mcap_relative'])
            require(sha(src) == row['mcap_sha256'], 'MCAP changed after audit')
            require(row.get('raw_gate_pass') and row.get('dedup_gate_pass'), 'Unqualified manifest entry')
            audit = json.loads(contained(run, row['audit_relative']).read_text())
            require(audit['raw_gate_pass'] and audit['terminal_stability_pass'] and
                    not audit['summary'].get('error'), 'Episode proof does not pass')
            identity = (row['task'], row['seed'])
            start = seed_for(worker, row['task'], 0)
            require(start <= row['seed'] < start + 250000, 'Seed outside assigned partition')
            require(identity not in seeds, 'Seed reused across submitted workers')
            seeds.add(identity)
            d = describe_mcap(src, kind='team', task=row['task'], seed=row['seed'])
            result = compare(db, d)
            if row['mcap_sha256'] in files:
                result = dict(pass_check=False, reason='identical_source_file')
            files.add(row['mcap_sha256'])
            item = dict(worker=worker, task=row['task'], seed=row['seed'],
                        source=str(src.relative_to(root)), sha256=row['mcap_sha256'],
                        dedup=result, training_ready=False, review_status='pending')
            if result['pass_check']:
                insert(db, d)
                db.commit()
                accepted.append(item)
            else:
                rejected.append(item)
    db.close()
    save(report/'accepted.json', accepted)
    save(report/'rejected.json', rejected)
    save(report/'summary.json', dict(accepted=len(accepted), rejected=len(rejected),
        baseline_sha256=sha('/reference/registry.sqlite3'), training_ready=False,
        note='Cross-run identity/scene dedup passed; conversion and visual review still required.'))
    print(f'Accepted={len(accepted)} rejected={len(rejected)}; originals untouched')


if __name__ == '__main__':
    main()

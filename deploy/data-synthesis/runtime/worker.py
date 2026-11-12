"""One selected GPU, one owned child, durable single-attempt accounting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from common import TASKS, LIMITS, require, seed_for, sha, save
from inventory import inventory
from targeting import definitions, allocate, choose_profile, specialize, make_config
from coverage_sampling import report as coverage_report


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--quota', type=int, default=400)
    p.add_argument('--max-attempts', type=int, default=2000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--disk-reserve-gib', type=int, default=100)
    args = p.parse_args()
    require(1 <= args.quota <= 10000 and args.quota <= args.max_attempts < 250000,
            'Invalid quota/attempt budget')
    seed_for(args.worker, TASKS[0], 0)
    root = Path('/work')
    os.environ['HBOT_RUN_ROOT'] = str(root)
    # Kernel lock is held until child exits, including parent crash.
    with (root / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / 'plan.json'
        inv = inventory()
        coverage_path = Path('/reference/coverage.json')
        require(coverage_path.is_file(), 'Missing coverage.json; regenerate the shared reference snapshot')
        coverage_reference = json.loads(coverage_path.read_text())
        require(coverage_reference.get('schema') == 1, 'Unsupported coverage reference schema')
        require(coverage_reference.get('original_training_included'),
                'Regenerate reference with --original-root so original demonstrations inform coverage')
        inv['coverage_baseline'] = coverage_reference['records']
        inv['selection_salt'] = args.worker
        code = {str(f.relative_to('/opt/hbot')): sha(f)
                for folder in ('runtime', 'configs', 'patches')
                for f in sorted((Path('/opt/hbot') / folder).rglob('*'))
                if f.is_file() and '__pycache__' not in str(f)}
        plan = dict(schema=1, worker=args.worker, quota=args.quota,
                    max_attempts=args.max_attempts, limits=LIMITS,
                    inventory=inv, code_sha256=code,
                    memory_profile=os.environ['HBOT_MEMORY_PROFILE'],
                    reference_sha256=sha('/reference/registry.sqlite3'),
                    coverage_reference_sha256=sha(coverage_path),
                    build=json.loads(Path('/opt/hbot/build.json').read_text()))
        if plan_path.exists():
            require(args.resume, 'Run exists. Inspect status before explicit --resume')
            require(json.loads(plan_path.read_text()) == plan, 'Frozen run plan changed')
        else:
            require(not args.resume, 'Cannot resume a run without a plan')
            require(not (root/'batches').exists(), 'Orphan output exists; do not overwrite')
            save(plan_path, plan)
            shutil.copytree('/opt/hbot/configs', root/'config_snapshot')
        from dedup_registry import DB, READY, connect, initialize
        DB.parent.mkdir(exist_ok=True)
        if not DB.exists():
            from merge import reference_db
            reference_db(DB).close()
            save(READY, dict(scope='shared_reference_plus_this_run',
                             reference_sha256=plan['reference_sha256']))
        require(READY.exists(), 'Partial registry initialization; inspect manually')
        statefile = root / 'status.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else dict(
            status='preparing', started_at=now(), batches=[], accepted=[])
        require(state['status'] != 'running',
                'Previous worker did not finish cleanly; inspect its container and output before recovery')
        if state['status'] == 'completed':
            print('Already completed; no duplicate collection')
            return
        require(state['status'] not in ('failed', 'attempt_limit'),
                'Failed run is preserved. Review before assigning a NEW worker ID; no automatic retry')
        profiles = {t: allocate(args.quota, definitions()[t]) for t in TASKS}
        for t in TASKS:
            for profile in profiles[t]:
                profile['task'] = t
        for task in TASKS:
            while sum(r['task'] == task for r in state['accepted']) < args.quota:
                if (root/'PAUSE').exists():
                    state.update(status='paused', updated_at=now())
                    save(statefile, state)
                    return
                require(shutil.disk_usage(root).free > args.disk_reserve_gib * 1024**3,
                        'Disk reserve reached; no new simulation started')
                prior = [b for b in state['batches'] if b['task'] == task]
                if len(prior) >= args.max_attempts:
                    state.update(status='attempt_limit', updated_at=now())
                    save(statefile, state)
                    raise RuntimeError('Attempt budget exhausted; see failed candidates')
                rows = [r for r in state['accepted'] if r['task'] == task]
                attempts = Counter(b['profile']['id'] for b in prior)
                profile = choose_profile(profiles[task], rows, [], {}, attempts)
                require(profile is not None, 'Quota accounting inconsistent')
                profile = specialize(profile, task, rows, state['batches'], inv)
                seed = seed_for(args.worker, task, len(prior))
                batch = root/'batches'/f'{task}_{seed}'
                batch.mkdir(parents=True, exist_ok=False)
                config = make_config(yaml.safe_load((Path('/opt/hbot/configs')/
                                     f'{task}.yaml').read_text()), task, profile)
                (batch/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
                save(batch/'profile.json', profile)
                record = dict(task=task, seed=seed, profile=profile, count=1,
                              path=str(batch.relative_to(root)), started=now())
                state['batches'].append(record)
                state.update(status='running', updated_at=now())
                save(statefile, state)  # Reserve the seed BEFORE spawning.
                cmd = [sys.executable, '/opt/hbot/runtime/collect_targeted.py',
                       '--task', task, '--config', str(batch/'config.yaml'),
                       '--profile', str(batch/'profile.json'), '--seed', str(seed),
                       '--episodes', '1', '--max-steps', str(LIMITS[task]),
                       '--output', str(batch/'raw')]
                try:
                    with (batch/'stdout.log').open('w') as log:
                        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                    require(rc == 0, f'Collector exit code {rc}; see {batch}/stdout.log')
                    episode = batch/'raw'/f'episode_{seed}.json'
                    require(episode.is_file(), 'Missing durable episode audit')
                    result = json.loads(episode.read_text())
                    require(not result['summary'].get('error'), str(result['summary'].get('error')))
                    record.update(finished=now(), raw_gate_pass=result['raw_gate_pass'])
                    if result['raw_gate_pass'] and result.get('dedup_gate_pass'):
                        a = result['audits'][0]
                        source = Path(a['source'])
                        a.update(task=task, seed=seed, mcap_relative=str(source.relative_to(root)),
                                 mcap_sha256=sha(source), audit_relative=str(episode.relative_to(root)),
                                 training_ready=False, review_status='pending',
                                 dedup_scope='shared_reference_plus_this_run')
                        state['accepted'].append(a)
                    state.update(status='between_episodes', updated_at=now())
                    save(statefile, state)
                    save(root/'manifest.json', state['accepted'])
                    save(root/'coverage_status.json', {t: coverage_report(t,
                        [r for r in state['accepted'] if r['task'] == t], inv)
                        for t in ('pick_attribute', 'spatial_directions')})
                    print(task, len([r for r in state['accepted'] if r['task']==task]),
                          '/', args.quota, 'accepted', flush=True)
                except BaseException as exc:
                    state.update(status='failed', error=repr(exc), updated_at=now())
                    save(statefile, state)
                    raise
        state.update(status='completed', finished_at=now())
        save(statefile, state)


if __name__ == '__main__':
    main()

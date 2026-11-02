#!/usr/bin/env python3
"""Verify actual episodes and videos, not just worker return codes or scores."""
import argparse
import json
from pathlib import Path

ALL_TASKS = 'pick_color,pick_spatial,place_a2b,stack_cubes,open_drawer,joint_direction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='Oldest to newest; newer task results take precedence')
    parser.add_argument('--tasks', default=ALL_TASKS)
    args = parser.parse_args()
    selected = {}
    for root in args.runs:
        summary = json.loads((root / 'summary.json').read_text())
        for name, item in summary['task_results'].items():
            selected[name] = (root, item)
    reports = []
    errors = []
    for name in args.tasks.split(','):
        if name not in selected:
            errors.append(f'{name}: no summary')
            continue
        root, item = selected[name]
        count = item['episode_count']
        if count < 1 or item['failed_episode_count']:
            errors.append(f'{name}: missing or failed episodes')
        episodes = []
        # The dispatcher also writes merged group/config copies. Read only
        # authoritative shard outputs to avoid counting each episode twice.
        for result in (root / name / 'shards').glob('*/eval_result.json'):
            episodes.extend(json.loads(result.read_text())['episode_results'])
        if len(episodes) != count or any(e['steps'] <= 0 or e['stop_reason'].startswith('episode_error') for e in episodes):
            errors.append(f'{name}: invalid action loop or episode_error')
        recordings = list((root / name).rglob('*.mcap'))
        if len(recordings) < count or any(p.stat().st_size == 0 for p in recordings):
            errors.append(f'{name}: missing recordings')
        previews = item.get('preview_episodes', [])
        if not previews:
            errors.append(f'{name}: missing video preview')
        for preview in previews:
            relative = preview.get('video')
            if not relative or not (root / relative).is_file() or (root / relative).stat().st_size == 0:
                errors.append(f'{name}: missing exported video')
        reports.append({'task': name, 'run': str(root), 'episodes': count,
                        'steps': [e['steps'] for e in episodes],
                        'stop_reasons': [e['stop_reason'] for e in episodes],
                        'score': item['score'], 'recordings': len(recordings)})
    print(json.dumps({'passed': not errors, 'tasks': reports, 'errors': errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()

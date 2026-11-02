#!/usr/bin/env python3
"""Read-only source audit; write an explicitly mixed-run report in the new run."""
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path('/study/wangmengfan/hbot')
SOURCE_RUN = 'local_pi05_full_99999_parallel_20261101_191358'
SOURCE = ROOT / 'outputs/sim' / SOURCE_RUN
RETAINED = ('open_drawer', 'stack_cubes', 'joint_direction')
RETEST = ('pick_color', 'pick_spatial', 'place_a2b')
ORDER = ('place_a2b', 'pick_color', 'pick_spatial', 'open_drawer',
         'stack_cubes', 'joint_direction')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def close(a, b):
    return isinstance(a, (int, float)) and math.isclose(a, b, abs_tol=1e-7)


def audit_task(root, task):
    """Require all planned shards, 50 unique nominal slots and valid raw results."""
    manifest = read(root / task / 'evaluation_manifest.json')
    planned = manifest.get('shards', [])
    problems, episodes, slots, paths = [], [], [], []
    shard_ids = [s['shard_id'] for s in planned]
    if len(set(shard_ids)) != len(shard_ids):
        problems.append('duplicate planned shard')
    actual = {p.parent.name for p in (root / task / 'shards').glob('*/eval_result.json')}
    if actual != set(shard_ids):
        problems.append('missing or unexpected result shards')
    for shard in planned:
        folder = root / task / 'shards' / shard['shard_id']
        result_path, summary_path = folder / 'eval_result.json', folder / 'shard_summary.json'
        result, summary = read(result_path), read(summary_path)
        items = result.get('episode_results', [])
        n = shard['episode_count']
        offset = shard['episode_offset']
        slots.extend(range(offset, offset + n))
        if (len(items) != n or result.get('episode_num') != n
                or summary.get('status') != 'ok' or summary.get('error')
                or summary.get('episodes') != n or summary.get('shard') != shard):
            problems.append(shard['shard_id'] + ': incomplete/failed shard')
        if items:
            scores = [e.get('progress') for e in items]
            if not all(isinstance(s, (int, float)) and math.isfinite(s) and 0 <= s <= 1 for s in scores):
                problems.append('invalid score')
            else:
                mean = sum(scores) / len(items)
                rate = sum(e.get('success') is True for e in items) / len(items)
                if not (close(result.get('average_progress'), mean)
                        and close(result.get('success_rate'), rate)
                        and close(summary.get('avg_progress'), mean)
                        and close(summary.get('success_rate'), rate)):
                    problems.append('raw/shard score mismatch')
            if any(e.get('stop_reason') not in ('success', 'max_steps')
                   or not isinstance(e.get('success'), bool)
                   or (e['stop_reason'] == 'success') != e['success'] for e in items):
                problems.append('invalid episode termination')
        episodes.extend(items)
        paths.extend([str(result_path), str(summary_path)])
    if sorted(slots) != list(range(50)) or len(episodes) != 50:
        problems.append('not exactly 50 unique planned episodes')
    valid_scores = [e['progress'] for e in episodes if isinstance(e.get('progress'), (int, float))]
    successes = sum(e.get('success') is True for e in episodes)
    stages = {}
    for e in episodes:
        for name, value in e.get('metrics', {}).get('criteria_reached', {}).items():
            stages[name] = stages.get(name, 0) + int(bool(value))
    return dict(task=task, path=str(root), complete=not problems,
                episodes=len(episodes), successes=successes,
                score=sum(valid_scores) / len(episodes) if episodes and len(valid_scores) == len(episodes) else None,
                success_rate=successes / len(episodes) if episodes else None,
                stage_counts=stages, problems=problems, raw_files=paths)


def audit_supplement(root):
    rows = {t: audit_task(root, t) for t in RETEST}
    summary = read(root / 'summary.json')
    final = summary.get('task_results', {})
    good = summary.get('status') == 'succeeded' and set(final) == set(RETEST)
    for task, row in rows.items():
        value = final.get(task, {})
        if not (row['complete'] and value.get('episode_count') == 50
                and value.get('failed_episode_count') == 0
                and close(value.get('score'), row['score'])
                and close(value.get('success_rate'), row['success_rate'])):
            good = False
    if good:
        good = (close(summary.get('eval_average_score'), sum(r['score'] for r in rows.values()) / 3)
                and close(summary.get('eval_average_success_rate'), sum(r['success_rate'] for r in rows.values()) / 3))
    return dict(complete=good, tasks=rows)


def generate(root, source=SOURCE, history=None):
    supplement = audit_supplement(root)
    rows = {t: audit_task(source, t) for t in RETAINED}
    rows.update(supplement['tasks'])
    complete = supplement['complete'] and all(r['complete'] for r in rows.values())
    payload = dict(complete=complete, supplement_complete=supplement['complete'],
                   source_run=str(source), supplement_run=str(root),
                   method='Mixed local evaluation: retained tasks 3 Clients; retested tasks 2 Clients. Not one uninterrupted 300-episode run.',
                   tasks=rows, episodes=sum(r['episodes'] for r in rows.values()),
                   successes=sum(r['successes'] for r in rows.values()),
                   mean_score=sum(r['score'] for r in rows.values()) / 6 if complete else None,
                   mean_success_rate=sum(r['success_rate'] for r in rows.values()) / 6 if complete else None)
    lines = ['# π₀.₅ 99999步双并发补测与历史对比', '',
             '本地分批合并结果：保留原轮三并发的 Open_Drawer、Stack_Cubes、Joint_Direction；',
             'Pick_Color、Pick_Spatial、Place_A2B 改为双并发，各重新测50回合。',
             '不计入原轮这三项未完整落盘的51回合；不修改原轮失败状态或原始结果。',
             '模型、种子、动作步数和任务预算沿用冻结配置，但会话RNG/场景重采样可能不同，非严格逐轨迹对照。', '',
             '完整性：' + ('六任务各50回合的原始分片与得分复核通过。' if complete else '尚未完成或审计未通过，不能作为六任务总成绩。'), '',
             '| 模型 | 平均得分 / 成功率 | Place_A2B | Pick_Color | Pick_Spatial | Open_Drawer | Stack_Cubes | Joint_Direction |',
             '|---|---|---|---|---|---|---|---|']
    if complete:
        cells = [f"{rows[t]['score']:.3f} / {rows[t]['success_rate']:.2%}" for t in ORDER]
        lines.append(f"| 99999步（3+2并发分批合并） | {payload['mean_score']:.4f} / {payload['mean_success_rate']:.2%} | " + ' | '.join(cells) + ' |')
    else:
        lines.append('| 99999步补测 | 未完成 | — | — | — | — | — | — |')
    references = []
    if history is not None:
        for label, name in reversed(history.REFERENCES):
            r = history.summarize(root.parent / name, label)
            references.append(r)
            if r['complete']:
                cells = [f"{r['tasks'][t]['score']:.3f} / {r['tasks'][t]['success_rate']:.2%}" for t in ORDER]
                lines.append(f"| {label} | {r['mean_score']:.4f} / {r['mean_success_rate']:.2%} | " + ' | '.join(cells) + ' |')
    payload['references'] = references
    lines += ['', '## 成绩来源与阶段指标', '', f'原轮：`{source}`', f'补测：`{root}`', '']
    for task in ORDER:
        r = rows[task]
        lines.append(f"- {task}：{r['successes']}/{r['episodes']}；来源 {'补测（2并发）' if task in RETEST else '原轮（3并发）'}；阶段计数 {r['stage_counts']}。")
    lines += ['', '原始成绩未复制改写；combined_results.json 保存每任务来源与原始文件路径。',
              '本目录 summary.json 仅是三项补测的150回合摘要，不是六任务总成绩。',
              '视频导出状态另见 run_status.json；config_snapshot 保存本轮配置与报告实现。', '']
    (root / 'combined_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'report_99999_supplement_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload


if __name__ == '__main__':
    import sys
    spec = importlib.util.spec_from_file_location('history', SOURCE / 'config_snapshot/report-pi05-99999-comparison.py')
    history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(history)
    print(json.dumps({'complete': generate(Path(sys.argv[1]), history=history)['complete']}))

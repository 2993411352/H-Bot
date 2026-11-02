#!/usr/bin/env python3
"""Consolidate completed 99999 tasks; archive both original runs without deletion.

prepare -> CPU aggregate -> CPU video export -> verify -> commit (rename originals).
No training, simulation, GPU settings, or existing result edits are performed.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess as sp
import sys

ROOT = Path('/study/wangmengfan/hbot')
INSIDE = Path('/home/users/wangmengfan')
ORIGINAL = 'local_pi05_full_99999_parallel_20261101_191358'
SUPPLEMENT = 'local_pi05_full_99999_supplement_20261102_112239'
MERGED = 'local_pi05_full_99999_merged_20261102'
RETAINED = ('open_drawer', 'stack_cubes', 'joint_direction')
ORDER = ('place_a2b', 'pick_color', 'pick_spatial', 'open_drawer', 'stack_cubes', 'joint_direction')
TASK_SOURCE = {t: ORIGINAL if t in RETAINED else SUPPLEMENT for t in ORDER}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise RuntimeError('Symlink in source, refusing unsafe relocation: ' + str(p))
        if p.is_file():
            stat = p.stat()
            row = dict(size=stat.st_size, inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns)
            # Large immutable recordings are checked by inode, size and mtime.
            if p.suffix not in ('.mcap', '.mp4'):
                row['sha256'] = hash_file(p)
            result[str(p.relative_to(root))] = row
    return result


def copy_artifact(source, destination):
    """Hard-link immutable recordings only; mutable metadata gets its own copy."""
    if Path(source).suffix in ('.mcap', '.mp4'):
        os.link(source, destination)
        return destination
    return shutil.copy2(source, destination)


def rewrite_paths(value):
    """Relocate active result paths; leave scores, run IDs and seeds unchanged."""
    if isinstance(value, list):
        return [rewrite_paths(x) for x in value]
    if isinstance(value, dict):
        return {k: rewrite_paths(v) for k, v in value.items()}
    if isinstance(value, str):
        for root in (ROOT, INSIDE):
            for name in (ORIGINAL, SUPPLEMENT):
                prefix = str(root / 'outputs/sim' / name)
                if value == prefix or value.startswith(prefix + '/'):
                    suffix = value[len(prefix):].lstrip('/')
                    first = suffix.split('/', 1)[0]
                    target = root / 'outputs/sim' / MERGED
                    if first in TASK_SOURCE and TASK_SOURCE[first] == name:
                        return str(target / suffix)
                    if name == SUPPLEMENT and first == 'videos':
                        return str(target / suffix)
                    return str(target / 'history' / name / suffix)
    return value


def assert_contained(path, parent):
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise RuntimeError('Unexpected source/target path: ' + str(path))


def source_audits(root):
    audit = load_module(root / 'scripts/report-pi05-99999-supplement.py', 'supplement_audit')
    sources = root / 'outputs/sim'
    supplement = read(sources / SUPPLEMENT / 'run_status.json')
    if not (supplement.get('state') == 'completed' and supplement.get('evaluation_exit_code') == 0
            and supplement.get('video_export_exit_code') == 0 and supplement.get('source_preserved') is True
            and supplement.get('complete_300_episode_combined_results') is True):
        raise RuntimeError('Supplement not fully completed')
    if read(sources / ORIGINAL / 'run_status.json').get('state') != 'failed':
        raise RuntimeError('Original status differs from the known interrupted run')
    rows = {t: audit.audit_task(sources / TASK_SOURCE[t], t) for t in ORDER}
    if not all(r['complete'] for r in rows.values()) or not audit.audit_supplement(sources / SUPPLEMENT)['complete']:
        raise RuntimeError('Source raw-result audit failed')
    return rows


def prepare(root=ROOT):
    parent, target = root / 'outputs/sim', root / 'outputs/sim' / MERGED
    assert_contained(target, parent)
    if target.exists():
        raise RuntimeError('Target already exists; inspect instead of overwriting')
    for name in (ORIGINAL, SUPPLEMENT):
        assert_contained(parent / name, parent)
    if json.loads(sp.check_output(['docker', 'exec', 'hbot-sim', 'python3',
            str(INSIDE / 'scripts/hbot-eval-control.py'), 'status'], text=True, timeout=30)):
        raise RuntimeError('An evaluation is running; refusing consolidation')
    rows = source_audits(root)
    inventories = {name: inventory(parent / name) for name in (ORIGINAL, SUPPLEMENT)}
    target.mkdir()
    (target / 'history').mkdir()
    plan = dict(created_at=now(), phase='preparing', canonical=str(target),
                source_inventories=inventories, task_sources=TASK_SOURCE,
                source_path_map={str(parent / name): str(target / 'history' / name) for name in (ORIGINAL, SUPPLEMENT)},
                source_audits=rows, immutable_media_storage='hard links, no duplicate media allocation')
    write(target / 'merge_manifest.json', plan)
    for task in ORDER:
        shutil.copytree(parent / TASK_SOURCE[task] / task, target / task, copy_function=copy_artifact)
        for p in (target / task).rglob('*.json'):
            data = read(p)
            moved = rewrite_paths(data)
            if moved != data:
                write(p, moved)
    shutil.copytree(parent / SUPPLEMENT / 'videos', target / 'videos', copy_function=copy_artifact)
    snapshot = target / 'config_snapshot'
    snapshot.mkdir()
    for name in (ORIGINAL, SUPPLEMENT):
        shutil.copytree(parent / name / 'config_snapshot', snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    plan['phase'] = 'prepared'
    write(target / 'merge_manifest.json', plan)
    print('PREPARED', target, flush=True)


def aggregate(root=INSIDE):
    """Use the installed repo's CPU-only merge helpers, never launch Isaac."""
    target = root / 'outputs/sim' / MERGED
    sys.path.insert(0, str(root / 'robo_orchard_sim'))
    from robo_orchard_sim.pipeline.evaluator.distributed.sharding import EvaluationShard
    entry = load_module(root / 'robo_orchard_sim/examples/manipulation-app/scripts/eval_policy.py', 'eval_entry_for_consolidation')
    config = entry.load_eval_config(target / 'config_snapshot' / ORIGINAL / (ORIGINAL + '.yaml'))
    task_results, task_payloads = {}, {}
    for name in ORDER:
        task = next(t for t in config.tasks if t.name == name)
        shards = [EvaluationShard(**v) for v in read(target / name / 'evaluation_manifest.json')['shards']]
        summary = entry._merge_task_shard_summaries(output_dir=target, task=task, shards=shards)
        row = entry._leaderboard_task_result(summary)
        if row['status'] != 'succeeded' or row['episode_count'] != 50 or row['failed_episode_count']:
            raise RuntimeError('Official aggregation failed for ' + name)
        row.pop('status')
        row['preview_episodes'] = row['preview_episodes'][:2]
        task_results[name] = row
        task_payloads[name] = summary.to_payload()
    mean_score = math.fsum(t['score'] for t in task_results.values()) / 6
    mean_success = math.fsum(t['success_rate'] for t in task_results.values()) / 6
    write(target / 'summary.json', dict(evaluation_id=MERGED, team_id=config.team_id, status='succeeded',
        eval_average_score=mean_score, eval_average_success_rate=mean_success, task_results=task_results,
        result_mode='consolidated completed tasks from a 3-client interrupted run and 2-client supplement'))
    write(target / 'distributed_summary.json', dict(tasks=task_payloads,
        overall=dict(status='ok', total_tasks=6, total_episodes=300,
                     success_episodes=sum(t['successes'] for t in task_payloads.values()),
                     avg_progress=mean_score, success_rate=mean_success),
        execution=dict(mode='mixed_runs', workers_per_gpu=None,
                       task_workers_per_gpu={t: 3 if t in RETAINED else 2 for t in ORDER}),
        elapsed_seconds=None, note='Not one uninterrupted run; source timings are preserved in history/'))
    print('AGGREGATED', mean_score, mean_success, flush=True)


def verify(root=ROOT, require_videos=True):
    target = root / 'outputs/sim' / MERGED
    report = load_module(root / 'scripts/report-pi05-99999-comparison.py', 'full_audit')
    audit = load_module(root / 'scripts/report-pi05-99999-supplement.py', 'task_audit')
    current = report.summarize(target, 'Full SFT 99999 (mixed 3/2 Clients)')
    if not current['complete'] or current['episodes'] != 300:
        raise RuntimeError('Canonical summary/raw audit failed: ' + str(current['problems']))
    expected = read(target / 'merge_manifest.json')['source_audits']
    for task in ORDER:
        row = audit.audit_task(target, task)
        before = expected[task]
        if not row['complete'] or row['successes'] != before['successes'] or not math.isclose(row['score'], before['score'], abs_tol=1e-10):
            raise RuntimeError('Scores changed during consolidation: ' + task)
        for p in (target / task).glob('shards/*/shard_summary.json'):
            for cfg in read(p)['configs']:
                path = Path(cfg['eval_result_json'].replace(str(INSIDE), str(ROOT), 1))
                if not path.is_file() or not path.is_relative_to(target):
                    raise RuntimeError('Broken active result path: ' + str(path))
        for p in (target / task).glob('shards/*/eval_result.json'):
            for episode in read(p)['episode_results']:
                rec = episode.get('metrics', {}).get('record_dir')
                if rec and not Path(rec.replace(str(INSIDE), str(ROOT), 1)).is_dir():
                    raise RuntimeError('Broken recording path: ' + rec)
    summary = read(target / 'summary.json')
    previews = [p for row in summary['task_results'].values() for p in row['preview_episodes']]
    if len(previews) != 12:
        raise RuntimeError('Expected exactly two previews per task')
    if require_videos:
        for preview in previews:
            video = preview.get('video', '')
            if not video or not (target / video).is_file():
                raise RuntimeError('Missing preview video')
            probe = json.loads(sp.check_output(['ffprobe', '-v', 'error', '-show_entries',
                'format=duration:stream=codec_type,width,height', '-of', 'json', str(target / video)], text=True, timeout=15))
            if float(probe['format']['duration']) <= 0 or not any(s.get('codec_type') == 'video' for s in probe['streams']):
                raise RuntimeError('Invalid video: ' + video)
    return current


def publish_report(root=ROOT):
    target = root / 'outputs/sim' / MERGED
    report = load_module(root / 'scripts/report-pi05-99999-comparison.py', 'history_report')
    current = report.summarize(target, 'Full SFT 99999 (3+2 Clients, consolidated)')
    refs = [report.summarize(target.parent / name, label) for label, name in report.REFERENCES]
    payload = dict(current=current, references=refs, source_runs={t: str(target / 'history' / TASK_SOURCE[t]) for t in ORDER},
                   caveat='Mixed 3/2-client local result, not a single uninterrupted evaluation or platform submission.')
    write(target / 'comparison_all_models.json', payload)
    lines = ['# π₀.₅ 99999步最终成绩（统一目录）', '',
        f"六任务各50回合，共300回合；成功{current['successes']}次。平均得分 **{current['mean_score']:.4f}**，成功率 **{current['mean_success_rate']:.2%}**。", '',
        '本地分批合并成绩：Open_Drawer、Stack_Cubes、Joint_Direction 来自三并发原轮；其余三项来自双并发补测。',
        '原轮显存保护停止的记录完整保留，不将其改称完整成功运行。未完成的51回合不计分。',
        '不同轮次并发数、会话RNG及场景重采样可能不同，不能视为严格逐轨迹对照。', '',
        '| 模型 | 平均得分 / 成功率 | Place_A2B | Pick_Color | Pick_Spatial | Open_Drawer | Stack_Cubes | Joint_Direction |',
        '|---|---|---|---|---|---|---|---|']
    for row in [current, *reversed(refs)]:
        if row['complete']:
            cells = [f"{row['tasks'][t]['score']:.3f} / {row['tasks'][t]['success_rate']:.2%}" for t in ORDER]
            lines.append(f"| {row['label']} | {row['mean_score']:.4f} / {row['mean_success_rate']:.2%} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 使用入口', '',
              '- `summary.json`：统一六任务300回合摘要。',
              '- `comparison_all_models.json`：本轮及历史完整数值。',
              '- 六个任务目录：有效分片成绩、逐回合结果、MCAP与日志。',
              '- `videos/`：每任务前2回合，共12个视频。',
              '- `config_snapshot/`：两批配置与汇总脚本。',
              '- `history/`：两批完整原始目录，含原始失败记录和未计分回合，归档文件保持原字节。',
              '- `merge_manifest.json`：旧路径映射、原始文件清单和完整性检查。', '',
              '只有顶层六个任务参与计分；不要递归统计 history/ 中的重复历史文件。',
              '大体积MCAP/已有视频采用硬链接共用磁盘空间；请视为只读原始素材。历史日志中的旧绝对路径仅作为原始记录保留。', '']
    (target / 'report_99999_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def commit(root=ROOT):
    parent, target = root / 'outputs/sim', root / 'outputs/sim' / MERGED
    manifest = read(target / 'merge_manifest.json')
    if manifest['phase'] != 'verified':
        raise RuntimeError('Not verified; cannot archive originals')
    for name in (ORIGINAL, SUPPLEMENT):
        source, dest = parent / name, target / 'history' / name
        assert_contained(source, parent)
        assert_contained(dest, target / 'history')
        if dest.exists() or inventory(source) != manifest['source_inventories'][name]:
            raise RuntimeError('Source changed or archive collision: ' + name)
    moved = []
    try:
        for name in (ORIGINAL, SUPPLEMENT):
            (parent / name).rename(target / 'history' / name)
            moved.append(name)
        for name in moved:
            if inventory(target / 'history' / name) != manifest['source_inventories'][name]:
                raise RuntimeError('Archive integrity mismatch')
    except BaseException:
        for name in reversed(moved):
            if not (parent / name).exists():
                (target / 'history' / name).rename(parent / name)
        raise
    manifest.update(phase='completed', finished_at=now(), archived_sources_verified=True)
    write(target / 'merge_manifest.json', manifest)
    write(target / 'run_status.json', dict(state='completed', completion_kind='consolidated_results',
        checkpoint=str(root / 'checkpoints/pi05_hbot_full/99999'), output=str(target),
        complete_300_episode_evaluation=True, single_uninterrupted_run=False,
        episodes=manifest['audited_episodes'], successes=manifest['audited_successes'], source_runs_preserved=True, preview_count=12,
        video_export_exit_code=0, finished_at=now(),
        note='Result collection completed; historical original run remains failed in history/.'))
    print('CONSOLIDATED', target, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'aggregate', 'export', 'probe', 'verify', 'commit'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'aggregate':
        aggregate()
    elif args.mode == 'export':
        target = ROOT / 'outputs/sim' / MERGED
        log = target / 'video_export.log'
        with log.open('x') as stream:
            result = sp.run(['docker', 'exec', '-e', 'OMP_NUM_THREADS=8',
                '-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim', 'python3',
                'tools/mcap_to_video.py', str(INSIDE / 'outputs/sim' / MERGED)],
                stdout=stream, stderr=sp.STDOUT, check=False)
        code = result.returncode
        if code == 0:
            probe = sp.run(['docker', 'exec', 'hbot-sim', 'python3',
                str(INSIDE / 'scripts' / Path(__file__).name), 'probe'], check=False)
            code = probe.returncode
        write(target / 'video_export_status.json', dict(exit_code=code, finished_at=now()))
        if code:
            raise RuntimeError('Video export or probing failed; sources not moved')
    elif args.mode == 'probe':
        target = INSIDE / 'outputs/sim' / MERGED
        previews = [p for row in read(target / 'summary.json')['task_results'].values()
                    for p in row['preview_episodes']]
        if len(previews) != 12:
            raise RuntimeError('Expected 12 previews')
        for p in previews:
            path = target / p['video']
            if not path.is_file():
                raise RuntimeError('Missing video: ' + str(path))
            probe = json.loads(sp.check_output(['ffprobe', '-v', 'error', '-show_entries',
                'format=duration:stream=codec_type,width,height', '-of', 'json', str(path)], text=True, timeout=15))
            if float(probe['format']['duration']) <= 0 or not any(s.get('codec_type') == 'video' for s in probe['streams']):
                raise RuntimeError('Invalid video: ' + str(path))
        print('VIDEO_PROBE_PASSED', len(previews))
    elif args.mode == 'verify':
        # ffprobe lives in the simulator image; validate its outputs separately.
        current = verify(require_videos=False)
        target = ROOT / 'outputs/sim' / MERGED
        if read(target / 'video_export_status.json')['exit_code'] != 0:
            raise RuntimeError('Video export not verified')
        summary = read(target / 'summary.json')
        for row in summary['task_results'].values():
            for preview in row['preview_episodes']:
                if not preview.get('video') or not (target / preview['video']).is_file():
                    raise RuntimeError('Missing preview after export')
        publish_report()
        manifest = read(target / 'merge_manifest.json')
        manifest.update(phase='verified', verified_at=now(), audited_episodes=current['episodes'],
                        audited_successes=current['successes'], mean_score=current['mean_score'],
                        mean_success_rate=current['mean_success_rate'])
        write(target / 'merge_manifest.json', manifest)
        print('VERIFIED', current['episodes'], current['successes'], current['mean_score'], current['mean_success_rate'])
    else:
        commit()


if __name__ == '__main__':
    main()

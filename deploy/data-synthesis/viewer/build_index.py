"""Index completed collector exports without changing any source files."""
import hashlib
import json
from pathlib import Path
import re

TASK_MAP = {'pick_attribute':'pick_color', 'spatial_directions':'spatial_pick',
            'place_a2b':'place_a2b', 'stack_cubes':'stack_cubes'}
CAMERAS = ('ext1_camera', 'ext2_camera', 'wrist_camera')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_file(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path.is_file(), f'Missing or outside source: {path}')
    return path


def read_json(root, path):
    return json.loads(checked_file(root, path).read_text(encoding='utf-8'))


def jsonlines(root, path):
    return [json.loads(line) for line in checked_file(root, path).read_text(encoding='utf-8').splitlines() if line.strip()]


def build_index(source, workers, export_name):
    import pyarrow.parquet as pq
    root = Path(source).resolve()
    require(re.fullmatch(r'[A-Za-z0-9_-]+', export_name), 'Invalid export name')
    require(workers and len(workers) == len(set(workers)), 'Choose distinct workers')
    require(all(type(w) is int and w > 0 for w in workers), 'Invalid worker number')
    rows = []
    for worker in sorted(workers):
        base = root/f'worker_{worker:04d}'
        export = base/'exports'/export_name
        report = read_json(root, export/'report.json')
        require(report.get('status') == 'format_verified', f'Worker {worker}: export is not complete')
        dataset = export/'lerobot_v21'
        info = read_json(root, dataset/'meta/info.json')
        require(info['codebase_version'] == 'v2.1' and info['fps'] == 30, 'Expected LeRobot v2.1 at 30 fps')
        require(info['chunks_size'] > 0, 'Invalid chunk size')
        episodes = sorted(jsonlines(root, dataset/'meta/episodes.jsonl'), key=lambda e:e['episode_index'])
        count = info['total_episodes']
        require(count > 0 and len(episodes) == count and [e['episode_index'] for e in episodes] == list(range(count)), 'Episode index coverage mismatch')
        require(report['accepted_episodes'] == count, 'Export report count mismatch')
        entries = read_json(root, export/'converter_manifest.json')
        frozen = export/'source_manifest.json'
        # Older exports only wrote converter_manifest. Validate its ordered mapping
        # before using the prefix of the worker manifest, which may have grown.
        manifest = read_json(root, frozen if frozen.exists() else base/'manifest.json')
        require(len(entries) == count and len(manifest) >= count, 'Source mapping count mismatch')
        if frozen.exists():
            require(len(manifest) == count, 'Frozen source manifest count mismatch')
        for i, ep in enumerate(episodes):
            raw, converted = manifest[i], entries[i]
            task = TASK_MAP.get(raw['task'])
            require(task is not None, 'Unsupported collector task')
            instruction = raw['metadata']['instruction']
            relative_mcap = Path(raw['mcap_relative']).as_posix()
            require(not Path(relative_mcap).is_absolute() and '..' not in Path(relative_mcap).parts, 'Invalid MCAP provenance')
            require(('seed' not in converted or converted['seed'] == raw['seed']) and converted['instruction'] == instruction and
                    converted['mcap_path'].replace('\\','/').endswith('/'+relative_mcap), 'Export/source mapping mismatch')
            require(ep.get('tasks') == [instruction], 'Export language instruction mismatch')
            digest = raw['mcap_sha256']
            require(re.fullmatch(r'[a-f0-9]{64}', digest), 'Invalid source hash')
            values = dict(episode_chunk=i//info['chunks_size'], episode_index=i)
            parquet = checked_file(root, dataset/info['data_path'].format(**values))
            require(pq.read_metadata(parquet).num_rows == ep['length'] > 0, 'Parquet frame count mismatch')
            for cam in CAMERAS:
                checked_file(root, dataset/info['video_path'].format(**values, video_key='observation.images.'+cam))
            rows.append(dict(id=f'{task}--w{worker:04d}-{digest}', task=task,
                worker=worker, seed=raw['seed'], episode_index=i, length=ep['length'],
                instruction=instruction, source_id=f'worker:{worker}:mcap:{digest}',
                dataset=dataset.relative_to(root).as_posix(), parquet=parquet.relative_to(root).as_posix(),
                parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(), mcap_sha256=digest,
                asset_id=str(raw.get('metadata',{}).get('asset_id','')),
                attribute=str(raw.get('targeted_profile',{}).get('relation','')) if isinstance(raw.get('targeted_profile'),dict) else '',
                events=[]))
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate source trajectory within a worker')
    return rows


def freeze_index(path, rows):
    """Never silently remap an existing review database onto a different dataset."""
    path = Path(path)
    content = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2)+'\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == content,
                'Review inputs changed. Choose a new --name; existing labels were not overwritten.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as f:
            f.write(content)
    return path

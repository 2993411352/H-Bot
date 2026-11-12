"""Tiny generated videos and tables; no project/user trajectories in tests."""
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def make_export(root, worker=101, tasks=('stack_cubes',), frames=30):
    root = Path(root)
    base = root/f'worker_{worker:04d}'
    export = base/'exports/export_v1'
    dataset = export/'lerobot_v21'
    info = dict(codebase_version='v2.1', fps=30, chunks_size=1000, total_episodes=len(tasks),
        data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    write_json(dataset/'meta/info.json', info)
    raw, entries, episodes = [], [], []
    for i, task in enumerate(tasks):
        instruction = f'{task}: 生成的界面测试视频，非机器人轨迹'
        mcap = f'batches/task_{i}/example.mcap'
        seed = worker*1000+i
        digest = hashlib.sha256(f'{worker}-{i}'.encode()).hexdigest()
        raw.append(dict(task=task, seed=seed, mcap_relative=mcap, mcap_sha256=digest,
            metadata={'instruction':instruction}, raw_gate_pass=True, dedup_gate_pass=True))
        entries.append(dict(instruction=instruction, mcap_path='/work/'+mcap))
        episodes.append(dict(episode_index=i, length=frames, tasks=[instruction]))
        values = dict(episode_chunk=0, episode_index=i)
        path = dataset/info['data_path'].format(**values)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = np.zeros((frames,8), dtype=np.float32)
        states[:,0] = np.linspace(0,1,frames)
        states[:,7] = np.linspace(0,.08,frames)
        actions = states.copy(); actions[:,7] = states[::-1,7]
        pq.write_table(pa.table({'observation.state':states.tolist(),'action':actions.tolist()}),path)
        for camera in ('ext1_camera','ext2_camera','wrist_camera'):
            path = dataset/info['video_path'].format(**values,video_key='observation.images.'+camera)
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path),'w') as out:
                stream = out.add_stream('libx264',rate=30)
                stream.width=160; stream.height=96; stream.pix_fmt='yuv420p'
                for n in range(frames):
                    rgb = np.full((96,160,3), n*5, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                        out.mux(packet)
                for packet in stream.encode(): out.mux(packet)
    (dataset/'meta/episodes.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in episodes)+'\n',encoding='utf-8')
    write_json(base/'manifest.json',raw)
    write_json(export/'source_manifest.json',raw)
    write_json(export/'converter_manifest.json',entries)
    write_json(export/'report.json',dict(status='format_verified',accepted_episodes=len(tasks)))
    return export

"""Panda Droid conversion and source-to-output numeric validation."""
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import is_dataclass, replace
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


BASE = OUTPUT = None
OBS_TOPIC = '/robot/joint_states'
ARM_TOPIC = '/action/robot_state/joint/joint_states'
GRIP_TOPIC = '/action/robot_state/gripper/joint_states'
from audit_raw import atomic_json

def convert_dataset(entries):
    # Panda Droid: eight observed values, seven arm actions plus absolute grip.
    from tools.mcap_to_lerobot.config_presets import EMBODIMENT_REGISTRY
    from tools.mcap_to_lerobot import convert

    def update(obj, **kwargs):
        if is_dataclass(obj):
            return replace(obj, **kwargs)
        for k,v in kwargs.items():
            setattr(obj,k,v)
        return obj

    cfg = copy.deepcopy(EMBODIMENT_REGISTRY['franka_panda'])
    arm = update(cfg.arms[0],
                 obs_joint_names=[f'panda_joint{i}' for i in range(1,9)],
                 action_arm_joint_names=[f'panda_joint{i}' for i in range(1,8)] + ['robotiq_gripper'],
                 gripper_joint_name='joint1', gripper_scale=1.0)
    cfg = update(cfg, name='panda_droid',arms=[arm],viz_gripper_scale=1.0,
                 urdf_path=None, viz_state_to_joints=[],
                 viz_ee_links=[], viz_drawn_links=[])
    EMBODIMENT_REGISTRY['panda_droid'] = cfg
    convert.EMBODIMENT_REGISTRY['panda_droid'] = cfg

    converter_manifest = BASE / 'converter_manifest.json'
    exact = [{'mcap_path':e['mcap_path'],'instruction':e['instruction']} for e in entries]
    if converter_manifest.exists():
        assert json.loads(converter_manifest.read_text())==exact, 'Manifest changed since last run'
    else:
        atomic_json(converter_manifest,exact)
    if OUTPUT.exists():
        raise RuntimeError('Output exists; preserve it and use a new export directory')
    print('CONVERT_START: please keep enough free disk; CPU-only',flush=True)
    rc = convert.main([
        '--embodiment','panda_droid', '--manifest',str(converter_manifest),
        '--out',str(OUTPUT),'--repo-id','hbot/synthetic-v1',
        '--num-workers','1'
    ])
    assert rc in (0,None), f'Converter RC={rc}'
    print('CONVERT_FINISHED',flush=True)


def verify_labels(entries):
    path = OUTPUT/'meta/episodes.jsonl'
    if not path.is_file():
        return False,'META_EPISODES_JSONL_MISSING'
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(lines)!=len(entries):
        return False, f'EPISODE_META_COUNT={len(lines)}'
    for i,(entry,row) in enumerate(zip(entries,lines)):
        if int(row.get('episode_index',-1))!=i:
            return False,f'EPISODE_INDEX_MISMATCH:{i}'
        tasks = row.get('tasks',[])
        if not isinstance(tasks,list) or entry['instruction'] not in tasks:
            return False,f'INSTRUCTION_MISMATCH:{i}: {tasks}'
    return True,'EXACT_SOURCE_INSTRUCTIONS_PRESERVED'


def raw_arrays(src):
    data = {t:[] for t in (OBS_TOPIC,ARM_TOPIC,GRIP_TOPIC)}
    times = {t:[] for t in data}
    with src.open('rb') as f:
        rd = make_reader(f,decoder_factories=[DecoderFactory()])
        for _,ch,_,msg in rd.iter_decoded_messages(topics=list(data)):
            data[ch.topic].append({s.name:float(s.position) for s in msg.states})
            times[ch.topic].append(msg.timestamp.seconds + msg.timestamp.nanos*1e-9)
    state = np.asarray([[r[f'panda_joint{i}'] for i in range(1,9)] for r in data[OBS_TOPIC]],dtype=float)
    action = np.asarray([[a[f'joint{i}'] for i in range(1,8)] + [g['joint1']]
                         for a,g in zip(data[ARM_TOPIC],data[GRIP_TOPIC])], dtype=float)
    return state,action,times


def audit_episode(entry,index):
    src = Path(entry['mcap_path'])
    state,action,times = raw_arrays(src)
    paths = list(OUTPUT.rglob(f'episode_{index:06d}.parquet'))
    assert len(paths)==1, f'Parquet count wrong: {index}: {paths}'
    t = pq.read_table(paths[0],columns=['observation.state','action','timestamp'])
    s=np.asarray(t['observation.state'].to_pylist(),dtype=float)
    a=np.asarray(t['action'].to_pylist(),dtype=float)
    ts=np.asarray(t['timestamp'].to_pylist(),dtype=float).reshape(-1)
    n=len(s)
    assert s.shape==a.shape==(n,8) and n>=30 and n<=len(state)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(a))
    assert len(ts)==n and np.all(np.isfinite(ts)) and np.all(np.diff(ts)>0)
    matches=[]
    for k in range(len(state)-n+1):
        es=float(np.max(np.abs(s-state[k:k+n])))
        ea=float(np.max(np.abs(a-action[k:k+n])))
        matches.append((es+ea,k,es,ea))
    _,offset,es,ea=min(matches)
    raw_t=np.asarray(times[OBS_TOPIC]); arm_t=np.asarray(times[ARM_TOPIC]); grip_t=np.asarray(times[GRIP_TOPIC])
    te=float(np.max(np.abs((ts-ts[0])-(raw_t[offset:offset+n]-raw_t[offset]))))
    sync=float(max(np.max(np.abs(raw_t[offset:offset+n]-arm_t[offset:offset+n])),
                   np.max(np.abs(arm_t[offset:offset+n]-grip_t[offset:offset+n]))))
    vids=list(OUTPUT.rglob(f'episode_{index:06d}.mp4'))
    samples=[]
    for video in vids:
        count=0; width=0; height=0; fps=0.; hashes=[]
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            fps=float(stream.average_rate) if stream.average_rate else 0.
            for frame in container.decode(video=0):
                if count in (0,n//2,n-1):
                    thumb=frame.reformat(width=32,height=18,format='gray')
                    hashes.append(hashlib.sha256(thumb.to_ndarray().tobytes()).hexdigest())
                count+=1
                width,height=frame.width,frame.height
        samples.append({'name':str(video.relative_to(OUTPUT)),'frames':count,
                        'width':width,'height':height,'fps':fps,'samples':hashes})
    good = (es<1e-5 and ea<1e-5 and te<0.02 and sync<0.02
            and len(vids)==3 and all(v['frames']==n and v['width']==1280
            and v['height']==720 and abs(v['fps']-30)<0.01 and len(v['samples'])==3 for v in samples))
    trajectory_fingerprint = hashlib.sha256(np.round(a,decimals=3).astype('float32').tobytes()).hexdigest()
    return {'episode_index':index,'seed':entry['seed'],'source':str(src),'frames':n,'raw_offset':offset,
            'state_max_error':es,'action_max_error':ea,'timestamp_max_error_s':te,
            'source_topic_sync_max_error_s':sync,'videos':samples,
            'action_trajectory_fingerprint':trajectory_fingerprint,'pass':bool(good)}


def loader_smoke():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds=LeRobotDataset(repo_id='hbot/synthetic-v1',root=str(OUTPUT),download_videos=False)
        assert len(ds)>0
        sample=ds[0]
        assert len(sample['observation.state'])==8 and len(sample['action'])==8
        return True, f'LE_ROBOT_LOCAL_READ_PASS; frames={len(ds)}'
    except Exception as exc:
        return False, f'LE_ROBOT_LOADER_CHECK_FAILED: {type(exc).__name__}: {exc}'

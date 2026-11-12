"""Cross-format trajectory identity and scene checks, shared by both GPU workers."""
import hashlib
import os
import json
from pathlib import Path
import sqlite3
import numpy as np
from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

BASE=Path(os.environ.get('HBOT_RUN_ROOT', '/work'))
DB=BASE/'dedup/registry.sqlite3'
READY=BASE/'dedup/local_scope.json'
OBS='/robot/joint_states'
ARM='/action/robot_state/joint/joint_states'
GRIP='/action/robot_state/gripper/joint_states'
ALIASES={'pick_color':'pick_attribute','spatial_pick':'spatial_directions'}


def connect(path=DB):
    c=sqlite3.connect(str(path),timeout=60)
    c.execute('PRAGMA busy_timeout=60000')
    return c


def initialize(c):
    c.executescript('''
    CREATE TABLE records(source TEXT PRIMARY KEY, kind TEXT NOT NULL, task TEXT NOT NULL,
      seed INTEGER, frames INTEGER, action_hash TEXT, scene_group TEXT, scene_json TEXT,
      windows INTEGER, file_size INTEGER, file_mtime_ns INTEGER);
    CREATE INDEX actions ON records(action_hash);
    CREATE INDEX scenes ON records(scene_group);
    CREATE TABLE windows(digest BLOB NOT NULL,source TEXT NOT NULL,
      PRIMARY KEY(digest,source));
    CREATE TABLE decisions(source TEXT PRIMARY KEY, result_json TEXT NOT NULL);
    ''')


def source_id(path):
    return str(Path(path).resolve())


def scene_descriptor(meta,task):
    actors=meta.get('actors',{});poses=meta.get('init_position',{})
    if not actors or not poses:return None,None
    rows=[]
    for name,actor in actors.items():
        if name not in poses or len(poses[name])!=7:return None,None
        ident=[actor.get('actor_uuid',''),actor.get('actor_category',''),actor.get('actor_type','')]
        rows.append((name,ident,poses[name]))
    rows.sort(key=lambda x:(x[1],x[2][:3]))
    indexes={r[0]:i for i,r in enumerate(rows)}
    def roles(v):
        if isinstance(v,dict):
            if 'scene_name' in v:return indexes.get(v['scene_name'],v['scene_name'])
            return {k:roles(w) for k,w in sorted(v.items())}
        if isinstance(v,list):return [roles(w) for w in v]
        return v
    bindings=roles(meta.get('role_bindings'))
    if bindings is None:bindings={'instruction':meta.get('instruction','').strip().lower()}
    group=json.dumps([task,[r[1] for r in rows],bindings],sort_keys=True,separators=(',',':'))
    return hashlib.sha256(group.encode()).hexdigest(),json.dumps([r[2] for r in rows])


def describe(source,kind,task,seed,states,actions,meta):
    task=ALIASES.get(task,task)
    states=np.asarray(states,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
    assert states.shape==actions.shape and states.ndim==2 and states.shape[1]==8
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    quant=np.rint(actions.astype(np.float64)*100000).astype('<i8')
    # Consecutive repeated commands and idle trimming do not create new data.
    keep=np.r_[True,np.any(quant[1:]!=quant[:-1],axis=1)]
    core=quant[keep]
    action_hash=hashlib.sha256(core.tobytes()).hexdigest() if len(core)>=10 else None
    joint=np.rint(np.concatenate([states,actions],axis=1).astype(np.float64)*100000).astype('<i8')
    windows=set()
    for i in range(len(joint)-14):
        # Exclude static home/posture holds, which are shared by many valid demos.
        if np.max(np.ptp(actions[i:i+15,:7],axis=0))<.005:continue
        windows.add(hashlib.sha256(joint[i:i+15].tobytes()).digest())
    group,scene=scene_descriptor(meta,task)
    p=Path(source);st=p.stat()
    return dict(source=source_id(p),kind=kind,task=task,seed=seed,frames=len(states),
        action_hash=action_hash,scene_group=group,scene_json=scene,
        windows=windows,file_size=st.st_size,file_mtime_ns=st.st_mtime_ns)


def describe_mcap(path,kind='current',task=None,seed=None):
    state=[];arm=[];grip=[];meta={}
    with Path(path).open('rb') as f:
        rd=make_reader(f,decoder_factories=[DecoderFactory()])
        assert rd.get_summary(),'Incomplete MCAP footer'
        for _,ch,_,m in rd.iter_decoded_messages(topics=[OBS,ARM,GRIP,'/meta_data']):
            if ch.topic=='/meta_data':meta=MessageToDict(m,preserving_proto_field_name=True);continue
            v={s.name:float(s.position) for s in m.states}
            if ch.topic==OBS:state.append([v[f'panda_joint{i}'] for i in range(1,9)])
            elif ch.topic==ARM:arm.append([v[f'joint{i}'] for i in range(1,8)])
            else:grip.append(v['joint1'])
    assert len(state)==len(arm)==len(grip) and state,'Incomplete action/state alignment'
    if task is None:
        s=str(path)
        task=next((t for token,t in [('Pick_Color','pick_attribute'),('Pick_Spatial','spatial_directions'),
              ('Place_A2B','place_a2b'),('Stack_Cubes','stack_cubes'),('_stack','stack_cubes'),
              ('_gpu1_spatial','spatial_directions')] if token in s),meta.get('task','unknown'))
    if seed is None:
        import re
        found=re.findall(r'seed_?(\d+)',str(path))
        seed=int(found[-1]) if found else int(meta.get('seed',-1))
    return describe(path,kind,task,seed,state,np.column_stack([arm,grip]),meta)


def insert(c,d):
    c.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)',(
        d['source'],d['kind'],d['task'],d['seed'],d['frames'],d['action_hash'],d['scene_group'],
        d['scene_json'],len(d['windows']),d['file_size'],d['file_mtime_ns']))
    c.executemany('INSERT OR IGNORE INTO windows VALUES (?,?)',[(x,d['source']) for x in d['windows']])


def compare(c,d):
    source=d['source'];hits=[]
    identity=c.execute('SELECT kind,file_size,file_mtime_ns FROM records WHERE source=?',(source,)).fetchone()
    if identity and (identity[1],identity[2])!=(d['file_size'],d['file_mtime_ns']):
        hits.append(dict(source=source,kind=identity[0],reason='previously_indexed_source_was_modified'))
    if d['action_hash']:
        for src,kind in c.execute('SELECT source,kind FROM records WHERE action_hash=? AND source<>?',(d['action_hash'],source)):
            hits.append(dict(source=src,kind=kind,reason='same_command_trajectory_after_idle_collapse'))
    if not hits and d['windows']:
        counts={};digests=list(d['windows'])
        for start in range(0,len(digests),300):
            part=digests[start:start+300]
            query='SELECT source,count(*) FROM windows WHERE digest IN ('+','.join('?'*len(part))+') AND source<>? GROUP BY source'
            for src,n in c.execute(query,[*part,source]):counts[src]=counts.get(src,0)+n
        for src,n in counts.items():
            old_n,kind=c.execute('SELECT windows,kind FROM records WHERE source=?',(src,)).fetchone()
            denom=min(len(digests),old_n)
            if n>=10 and denom and n/denom>=.9:
                hits.append(dict(source=src,kind=kind,reason='moving_state_action_sequence_overlap',
                    matched_windows=n,shorter_sequence_overlap=n/denom))
    if not hits and d['scene_group']:
        a=np.asarray(json.loads(d['scene_json']))
        for src,kind,scene in c.execute('SELECT source,kind,scene_json FROM records WHERE scene_group=? AND source<>?',(d['scene_group'],source)):
            b=np.asarray(json.loads(scene))
            if a.shape!=b.shape:continue
            pos=float(np.max(np.linalg.norm(a[:,:3]-b[:,:3],axis=1)))
            qa=a[:,3:]/np.linalg.norm(a[:,3:],axis=1,keepdims=True)
            qb=b[:,3:]/np.linalg.norm(b[:,3:],axis=1,keepdims=True)
            angle=float(np.max(2*np.arccos(np.clip(np.abs((qa*qb).sum(1)),0,1))))
            if pos<=.001 and angle<=np.deg2rad(1):
                hits.append(dict(source=src,kind=kind,reason='same_assets_roles_and_near_identical_initial_scene',position_max_m=pos,rotation_max_deg=float(np.rad2deg(angle))))
    existing=c.execute('SELECT kind FROM records WHERE source=?',(source,)).fetchone()
    return dict(pass_check=not hits,source_id=source,duplicate_matches=hits,
        same_source_already_indexed=bool(existing),existing_kind=existing[0] if existing else None,
        protocol='cross_dataset_v1',quantization=1e-5,window_length=15,
        scene_check_available=bool(d['scene_group']),
        note='One source ID is counted once. Asset/category/instruction reuse alone is allowed; this does not guarantee semantic novelty.')


def check(path,task,seed):
    assert READY.is_file(),'Local dedup registry not initialized'
    d=describe_mcap(path,task=task,seed=seed)
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        result=compare(c,d)
        if result['pass_check']:insert(c,d)
        c.execute('INSERT OR REPLACE INTO decisions VALUES (?,?)',(d['source'],json.dumps(result)))
    return result

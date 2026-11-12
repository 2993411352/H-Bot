"""Private, CPU-only H-Bot annotation workspace. Originals are read-only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from urllib.parse import urlsplit

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from flask import Flask, abort, jsonify, request, send_file

TASKS = {'stack_cubes': '堆叠方块', 'place_a2b': '物体放置', 'pick_color': '颜色拾取',
         'spatial_pick': '空间拾取', 'open_drawer': '抽屉操作', 'joint_direction': '关节方向'}
PHASES = {'reach': '接近目标', 'grasp': '抓取 / 夹持', 'transport': '搬运 / 移动',
          'align': '精细对齐', 'release': '释放物体', 'retreat': '撤离 / 回位',
          'contact': '建立接触', 'actuate': '推动 / 拉动', 'settle': '必要等待 / 稳定',
          'recovery': '纠偏 / 重抓', 'idle': '无效静止 / 重复', 'unknown': '无法判断'}
QUALITY = {'usable': '可用', 'uncertain': '待复核', 'exclude': '不用于正向模仿'}
TAGS = {'slip': '滑落', 'misalignment': '偏位', 'collision': '非预期碰撞',
        'wrong_target': '目标错误', 'occluded': '遮挡', 'contact_uncertain': '接触存疑'}
CAMERAS = ('ext1_camera', 'ext2_camera', 'wrist_camera')
decode_lock = threading.BoundedSemaphore(2)


def now():
    return datetime.now(timezone.utc).isoformat()


def representative_ids(rows):
    """Deterministic diverse suggestions, not a held-out evaluation split."""
    selected = set()
    for task in TASKS:
        group = sorted([r for r in rows if r['task'] == task], key=lambda r: (r['length'], r['source_id']))
        strata = defaultdict(list)
        for i, row in enumerate(group):
            # Cover asset/operation for articulation, duration quintiles otherwise.
            key = (row.get('asset_id'), row.get('attribute')) if task in ('open_drawer', 'joint_direction') else (min(4, i*5//max(1,len(group))),)
            strata[key].append(row)
        buckets = sorted(strata.values(), key=lambda g: (-len(g), g[0]['source_id']))
        for bucket in buckets:
            bucket.sort(key=lambda r: hashlib.sha256(('hbot-label-v1:'+r['source_id']).encode()).hexdigest())
        while len([r for r in group if r['id'] in selected]) < min(10, len(group)):
            for bucket in buckets:
                if bucket:
                    selected.add(bucket.pop(0)['id'])
                if len([r for r in group if r['id'] in selected]) >= min(10, len(group)):
                    break
    return selected


def validate_annotation(body, episode, submitted=False):
    if not isinstance(body, dict):
        raise ValueError('标注必须是对象')
    if body.get('status') not in ('draft', 'reviewed'):
        raise ValueError('状态必须为草稿或人工审核完成')
    if type(body.get('revision')) is not int or body['revision'] < 0:
        raise ValueError('缺少有效版本号')
    reviewer = body.get('reviewer', '').strip() if isinstance(body.get('reviewer'), str) else ''
    if not reviewer or len(reviewer) > 80:
        raise ValueError('请填写标注者姓名/代号（1–80字）')
    notes = body.get('notes', '')
    if not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError('备注过长')
    segments = body.get('segments')
    if not isinstance(segments, list) or len(segments) > episode['length']:
        raise ValueError('区间列表不合法')
    clean = []
    for s in segments:
        if not isinstance(s, dict):
            raise ValueError('区间格式错误')
        lo, hi = s.get('start'), s.get('end')
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < episode['length']:
            raise ValueError('区间必须使用有效的起止帧，含两端')
        if s.get('phase') not in PHASES or s.get('quality') not in QUALITY:
            raise ValueError('未知阶段或质量标签')
        cycle = s.get('cycle', 0)
        if type(cycle) is not int or cycle not in (0,1,2,3) or (episode['task'] != 'stack_cubes' and cycle != 0):
            raise ValueError('仅堆叠可标注第1–3次搬运循环；0表示不适用/未知')
        confidence = s.get('confidence', 'high')
        if confidence not in ('high', 'medium', 'low'):
            raise ValueError('置信度不合法')
        if s['phase'] == 'unknown' and s['quality'] != 'uncertain':
            raise ValueError('无法判断的阶段必须标为待复核')
        if type(s.get('critical', False)) is not bool:
            raise ValueError('关键阶段标记不合法')
        tags = s.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(t,str) or t not in TAGS for t in tags):
            raise ValueError('未知事件标签')
        note = s.get('note', '')
        if not isinstance(note, str) or len(note)>1000:
            raise ValueError('区间备注过长')
        clean.append(dict(start=lo,end=hi,phase=s['phase'],quality=s['quality'],cycle=cycle,
                          confidence=confidence,critical=s.get('critical',False),tags=sorted(set(tags)),note=note))
    clean.sort(key=lambda s:s['start'])
    edge = 0
    for s in clean:
        if s['start'] < edge:
            raise ValueError('标注区间不能重叠')
        if submitted and s['start'] != edge:
            raise ValueError(f'请先补齐第 {edge} 帧起的未标注区间')
        edge = s['end']+1
    if submitted and edge != episode['length']:
        raise ValueError(f'请覆盖完整轨迹（当前到 {edge-1} 帧，末帧为 {episode["length"]-1}）')
    return dict(status=body['status'],reviewer=reviewer,notes=notes,segments=clean)


class Workspace:
    def __init__(self, source, index, data):
        self.source, self.data = Path(source).resolve(), Path(data).resolve()
        if self.data == self.source or self.data.is_relative_to(self.source):
            raise ValueError('标签目录不能放在原始数据内部')
        self.data.mkdir(parents=True,exist_ok=True)
        self.rows = json.loads(Path(index).read_text(encoding='utf-8'))
        for r in self.rows:
            r.setdefault('id', f'{r["task"]}--{r["episode_index"]:06d}')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', r['id']):
                raise ValueError('Invalid episode ID')
            if r['task'] not in TASKS:
                raise ValueError('Unknown task in index')
        self.episodes = {r['id']:r for r in self.rows}
        if len(self.episodes) != len(self.rows):
            raise ValueError('Duplicate episode ID')
        self.selected = representative_ids(self.rows)
        self.index_sha = hashlib.sha256(Path(index).read_bytes()).hexdigest()
        self.db = self.data/'annotations.sqlite3'
        with self.connect() as c:
            c.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS labels (episode_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS history (episode_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(episode_id,revision));
            CREATE TABLE IF NOT EXISTS proposals (episode_id TEXT NOT NULL, batch_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(episode_id,batch_id));
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);''')
            stored = c.execute('SELECT value FROM metadata WHERE key=?',('index_sha',)).fetchone()
            if stored and stored[0] != self.index_sha:
                raise ValueError('Dataset index changed; use a new workspace')
            c.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('index_sha',self.index_sha))
        os.chmod(self.db,0o600)
        plan=self.data/'sample_plan.json'
        if not plan.exists():
            plan.write_text(json.dumps(dict(index_sha256=self.index_sha,episode_ids=sorted(self.selected),
                            purpose='optional representative review examples'),ensure_ascii=False,indent=2),encoding='utf-8')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.db),timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def episode(self,eid):
        if eid not in self.episodes:
            raise KeyError('不存在该轨迹')
        return self.episodes[eid]

    def safe_source(self,relative):
        path=(self.source/relative).resolve()
        if not path.is_relative_to(self.source) or not path.is_file():
            raise ValueError('源数据路径无效')
        return path

    def video(self,eid,camera):
        if camera not in CAMERAS:
            raise KeyError('未知相机')
        e=self.episode(eid)
        info=json.loads(self.safe_source(e['dataset']+'/meta/info.json').read_text(encoding='utf-8'))
        relative=e['dataset']+'/'+info['video_path'].format(episode_chunk=e['episode_index']//info['chunks_size'],
                  episode_index=e['episode_index'],video_key='observation.images.'+camera)
        return self.safe_source(relative)

    def label(self,eid):
        self.episode(eid)
        with self.connect() as c:
            row=c.execute('SELECT payload FROM labels WHERE episode_id=?',(eid,)).fetchone()
        return json.loads(row[0]) if row else dict(episode_id=eid,revision=0,status='new',reviewer='',notes='',segments=[])

    def save(self,eid,body):
        e=self.episode(eid)
        clean=validate_annotation(body,e,isinstance(body,dict) and body.get('status')=='reviewed')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT revision FROM labels WHERE episode_id=?',(eid,)).fetchone()
            revision=row[0] if row else 0
            if revision != body['revision']:
                raise RuntimeError('版本冲突：此轨迹已在其他窗口更新。请先重新载入，避免覆盖。')
            clean.update(episode_id=eid,revision=revision+1,updated_at=now(),origin='human',
                         source_id=e['source_id'],parquet_sha256=e['parquet_sha256'],schema_version=1)
            payload=json.dumps(clean,ensure_ascii=False,allow_nan=False)
            c.execute('INSERT OR REPLACE INTO labels VALUES (?,?,?)',(eid,revision+1,payload))
            c.execute('INSERT INTO history VALUES (?,?,?)',(eid,revision+1,payload))
        return clean

    def labels(self):
        with self.connect() as c:
            return {eid:json.loads(payload) for eid,payload in c.execute('SELECT episode_id,payload FROM labels')}

    def summary(self):
        labels=self.labels()
        tasks=[]
        for task,name in TASKS.items():
            rows=[r for r in self.rows if r['task']==task]
            if not rows:
                continue
            reviewed=[r for r in rows if labels.get(r['id'],{}).get('status')=='reviewed']
            uncertain=sum(any(s['quality']=='uncertain' for s in labels[r['id']]['segments']) for r in reviewed)
            tasks.append(dict(id=task,name=name,total=len(rows),reviewed=len(reviewed),target=len(rows),
                              uncertain_reviewed=uncertain,suggested=sum(r['id'] in self.selected for r in rows)))
        return dict(tasks=tasks,total=len(self.rows),total_reviewed=sum(t['reviewed'] for t in tasks),
                    ready_for_auto=bool(tasks) and all(t['reviewed']>=min(10,t['total']) for t in tasks),
                    note='按轨迹审核阶段和质量；待复核片段可以稍后继续处理。')

    def export(self):
        return dict(schema_version=1,exported_at=now(),index_sha256=self.index_sha,
                    frame_interval='zero-based inclusive [start,end]',fps=30,source_root=str(self.source),
                    summary=self.summary(),annotations=list(self.labels().values()),
                    provenance='human sidecars only; original videos/parquet unchanged')

    @lru_cache(maxsize=24)
    def signals(self,eid):
        import numpy as np
        import pyarrow.parquet as pq
        e=self.episode(eid)
        p=self.safe_source(e['parquet'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=e['parquet_sha256']:
            raise ValueError('原始表格内容与索引摘要不一致')
        table=pq.read_table(p,columns=['observation.state','action'])
        state=np.asarray(table['observation.state'].to_pylist())
        action=np.asarray(table['action'].to_pylist())
        if state.shape != (e['length'],8) or action.shape != state.shape or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('状态或动作维度、帧数、数值与索引不一致')
        movement=np.linalg.norm(np.diff(state[:,:7],axis=0,prepend=state[:1,:7]),axis=1)
        return dict(gripper_state=state[:,7].tolist(),gripper_action=action[:,7].tolist(),
                    joint_motion=movement.tolist(),events=e.get('events',[]),
                    warning='曲线/夹爪事件仅为运动学提示，不能证明接触、抓牢或成功。')

    @lru_cache(maxsize=96)
    def frame(self,eid,camera,index):
        import av
        e=self.episode(eid)
        if not 0<=index<e['length']:
            raise ValueError('帧号越界')
        with decode_lock, av.open(str(self.video(eid,camera))) as container:
            stream=container.streams.video[0]; stream.thread_count=1
            start=stream.start_time or 0
            container.seek(start+int(index/30/stream.time_base),stream=stream,backward=True,any_frame=False)
            found=None
            for f in container.decode(stream):
                n=round(float((f.pts-start)*stream.time_base)*30)
                if n==index:
                    found=f; break
                if n>index: break
            if found is None:
                # Fallback is exact sequential indexing, never return nearest keyframe.
                container.seek(start,stream=stream,backward=True)
                for n,f in enumerate(container.decode(stream)):
                    if n==index: found=f; break
            if found is None: raise ValueError('无法解码指定帧')
            out=io.BytesIO(); found.to_image().save(out,format='JPEG',quality=90)
            return out.getvalue()


def create_app(source,index,data,sandbox=False):
    app=Flask(__name__,static_folder='static')
    app.config['MAX_CONTENT_LENGTH']=512*1024
    app.config['TRUSTED_HOSTS']=['127.0.0.1','localhost','[::1]']
    store=Workspace(source,index,data)
    app.config['WORKSPACE']=store
    csrf=secrets.token_urlsafe(32)

    @app.before_request
    def guard():
        if request.method in ('POST','PUT','DELETE','PATCH'):
            origin=request.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=request.host: abort(403)
            if not secrets.compare_digest(request.headers.get('X-CSRF-Token',''),csrf): abort(403)
            if not request.is_json: abort(415)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        if request.path.startswith('/api/'): response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(ValueError)
    def bad(e): return jsonify(error=str(e)),400
    @app.errorhandler(KeyError)
    def missing(e): return jsonify(error=str(e)),404
    @app.errorhandler(RuntimeError)
    def conflict(e): return jsonify(error=str(e)),409

    @app.get('/')
    def home(): return send_file(Path(__file__).parent/'static/index.html')

    @app.get('/api/bootstrap')
    def bootstrap():
        labels=store.labels()
        rows=[dict(id=r['id'],task=r['task'],episode_index=r['episode_index'],length=r['length'],
                   instruction=r['instruction'],asset_id=r.get('asset_id',''),attribute=r.get('attribute',''),
                   worker=r.get('worker'),seed=r.get('seed'),
                   suggested=r['id'] in store.selected,status=labels.get(r['id'],{}).get('status','new')) for r in store.rows]
        return jsonify(csrf=csrf,episodes=rows,summary=store.summary(),phases=PHASES,quality=QUALITY,
                       tags=TAGS,cameras=CAMERAS,sandbox=sandbox,fps=30,schema_version=1,index_sha=store.index_sha)

    @app.get('/api/summary')
    def summary(): return jsonify(store.summary())

    @app.get('/api/episode/<eid>')
    def episode(eid):
        e=store.episode(eid)
        with store.connect() as c:
            row=c.execute('SELECT payload FROM proposals WHERE episode_id=? ORDER BY rowid DESC LIMIT 1',(eid,)).fetchone()
        return jsonify(episode=e,annotation=store.label(eid),signals=store.signals(eid),
                       proposal=json.loads(row[0]) if row else None)

    @app.post('/api/annotation/<eid>')
    def save(eid): return jsonify(store.save(eid,request.get_json()))

    @app.get('/api/history/<eid>')
    def history(eid):
        store.episode(eid)
        with store.connect() as c:
            return jsonify([json.loads(r[0]) for r in c.execute('SELECT payload FROM history WHERE episode_id=? ORDER BY revision DESC',(eid,))])

    @app.get('/api/export')
    def export():
        result=jsonify(store.export()); result.headers['Content-Disposition']='attachment; filename=hbot-human-annotations.json'
        return result

    @app.get('/media/<eid>/<camera>.mp4')
    def media(eid,camera):
        return send_file(store.video(eid,camera),mimetype='video/mp4',conditional=True,max_age=3600)

    @app.get('/frame/<eid>/<camera>/<int:frame>.jpg')
    def frame(eid,camera,frame):
        response=send_file(io.BytesIO(store.frame(eid,camera,frame)),mimetype='image/jpeg',max_age=3600)
        response.headers['X-Frame-Index']=str(frame)
        return response
    return app


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--port',type=int,default=18765)
    p.add_argument('--sandbox',action='store_true')
    args=p.parse_args()
    app=create_app(args.source,args.index,args.data,args.sandbox)
    from waitress import serve
    serve(app,host='127.0.0.1',port=args.port,threads=4)


if __name__=='__main__': main()

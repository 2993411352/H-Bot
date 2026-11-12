"""Independent raw-trajectory gate; no simulator or model mutation."""
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


@lru_cache(maxsize=1)
def seen_assets():
    import yaml
    split=Path('/assets/OBJECTS_v2/splits/hbot_panda_droid_split_seen.yaml')
    return {r['uuid'] for r in yaml.safe_load(split.read_text())['seen']}

OBS = '/robot/joint_states'
ARM = '/action/robot_state/joint/joint_states'
GRIP = '/action/robot_state/gripper/joint_states'
CAMS = {c: f'/observation/cameras/{c}/color_image/image_raw'
        for c in ('ext1_camera', 'ext2_camera', 'wrist_camera')}


def rotation(q):
    w,x,y,z = q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def audit(path, task, seed, contact_sheet=None, terminal_proof=None, allow_legacy_nonseen=False):
    path = Path(path)
    out = dict(source=str(path), task=task, seed=seed, raw_gate_pass=False,
               converted_loader_pass=False, training_ready=False)
    try:
        topics = [OBS, ARM, GRIP, *CAMS.values()]
        ts = {t: [] for t in topics}
        vals = {t: [] for t in (OBS, ARM, GRIP)}
        stats = {t: [] for t in CAMS.values()}
        thumbs = {t: {} for t in CAMS.values()}
        metadata = []
        object_frames = []
        with path.open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            assert summary and summary.statistics, 'Missing MCAP footer'
            counts = {c.topic: summary.statistics.channel_message_counts.get(i, 0)
                      for i, c in summary.channels.items()}
            n = counts[OBS]
            assert n >= 30 and all(counts.get(t) == n for t in topics), counts
            for _, ch, _, msg in reader.iter_decoded_messages(topics=topics + ['/meta_data','/observation/objects/tf']):
                t = ch.topic
                if t == '/observation/objects/tf':
                    object_frames.append({v.child_frame_id:
                        [v.translation.x,v.translation.y,v.translation.z,
                         v.rotation.w,v.rotation.x,v.rotation.y,v.rotation.z]
                        for v in msg.transforms})
                    continue
                if t == '/meta_data':
                    metadata.append(MessageToDict(msg, preserving_proto_field_name=True))
                    continue
                ts[t].append(msg.timestamp.seconds + msg.timestamp.nanos * 1e-9)
                if t in vals:
                    vals[t].append({s.name: float(s.position) for s in msg.states})
                else:
                    image = Image.open(io.BytesIO(msg.data)).convert('RGB')
                    assert image.size == (1280, 720), (t, image.size)
                    small = np.asarray(image.resize((64, 36)))
                    stats[t].append((float(small.std()), hashlib.sha256(small.tobytes()).hexdigest()))
                    k = len(ts[t]) - 1
                    if k in (0, n // 2, n - 1):
                        thumbs[t][k] = image.resize((426, 240))
        assert len(metadata) == 1, 'Expected exactly one final metadata record'
        meta = metadata[0]
        out['metadata'] = meta
        assert meta.get('task_success') == 1.0, 'Official task validator did not pass'
        assert meta.get('instruction', '').strip(), 'Missing instruction'
        unknown=[]
        if task!='stack_cubes':
            unknown=[name for name,a in meta['actors'].items() if a['actor_uuid'] not in seen_assets()]
            targets=[v['scene_name'] for v in meta['role_bindings'].values() if isinstance(v,dict)]
            out['all_scene_assets_seen']=not unknown
            out['outside_seen_scene_actors']=unknown
            out['bound_targets_seen']=not any(name in unknown for name in targets)
            assert out['bound_targets_seen'], 'A selected task target is outside the seen asset split'
            if not (task=='pick_attribute' and allow_legacy_nonseen):
                assert not unknown, 'A scene actor is outside the seen asset split'
        else:
            out['all_scene_assets_seen']=True
            out['asset_split_note']='Four task-native cubes, no external sampled assets'
        state = np.array([[v[f'panda_joint{i}'] for i in range(1, 9)] for v in vals[OBS]])
        actions = np.array([[a[f'joint{i}'] for i in range(1, 8)] + [g['joint1']]
                            for a, g in zip(vals[ARM], vals[GRIP])])
        assert state.shape == actions.shape == (n, 8)
        assert np.isfinite(state).all() and np.isfinite(actions).all()
        assert actions[:, 7].min() >= -1e-4 and actions[:, 7].max() <= 1.0001
        reference = np.array(ts[OBS])
        timing = {t: dict(sync_max_s=float(np.max(np.abs(np.array(v) - reference))),
                         dt_error_max_s=float(np.max(np.abs(np.diff(v) - 1 / 30))))
                  for t, v in ts.items()}
        assert all(v['sync_max_s'] < .002 and v['dt_error_max_s'] < .002 for v in timing.values()), timing
        camera_stats = {c: dict(frames=len(stats[t]), min_std=min(x[0] for x in stats[t]),
                                unique_thumbnails=len(set(x[1] for x in stats[t])))
                        for c, t in CAMS.items()}
        assert all(v['unique_thumbnails'] > 2 for v in camera_stats.values()), camera_stats
        frame_valid = np.ones(n,dtype=bool)
        for t in CAMS.values():
            frame_valid &= np.array([x[0] > 2 for x in stats[t]])
        # Preserve recorded timing. Conservative exclusions are a sampler mask,
        # never deletion/concatenation of frames around an occlusion.
        eligible = np.array([frame_valid[i:i+15].all() for i in range(n-14)])
        assert frame_valid.mean() >= .8 and eligible.sum() >= 30, 'Insufficient clear continuous windows'
        if task=='stack_cubes' and not object_frames:
            # Native cubes have final metadata but no /observation/objects/tf.
            # Require the online current-criterion proof; never invent TF frames.
            if terminal_proof is None:
                sidecar=path.parents[3]/f'episode_{seed}.json'
                if sidecar.exists():terminal_proof=json.loads(sidecar.read_text()).get('terminal_stability_pass')
            assert terminal_proof is True, 'Native cube recording needs online terminal stability proof'
            poses=[meta['final_position']]
        else:
            assert len(object_frames) == n, 'Object trajectory/observation length mismatch'
            poses = object_frames[-15:]
        roles = meta['role_bindings']
        terminal = dict(frames=len(poses), pass_check=True,
                        temporal_evidence='recorded object TF' if object_frames else 'online last-15-current-criteria sidecar',
                        online_terminal_stability_pass=terminal_proof)
        if task in ('spatial_directions','pick_attribute','place_a2b'):
            picked=roles['pick']['scene_name']
            xyz=np.array([f[picked][:3] for f in poses])
            motion=float(np.max(np.linalg.norm(xyz-xyz[-1],axis=1)))
            assert motion<.015, f'Target unstable at end: {motion}'
            terminal['target_max_motion_m']=motion
            if task == 'place_a2b':
                placed=roles['place']['scene_name']
                assert np.max(actions[-15:,7]) < .05, 'Gripper did not remain open after placement'
                target_box=meta['actors'][picked]['bbox']
                support_box=meta['actors'][placed]['bbox']
                for f in poses:
                    a,b=np.array(f[picked]),np.array(f[placed])
                    center=a[:3]+rotation(a[3:])@np.array(target_box['center'])
                    local=rotation(b[3:]).T@(center-b[:3])-np.array(support_box['center'])
                    assert np.all(np.abs(local[:2])<=np.array(support_box['size'])[:2]/2), 'Final target outside container footprint'
                    assert center[2]>=b[2]-np.array(support_box['size'])[2]/2-.01, 'Final target fell below container'
            else:
                assert np.min(actions[-15:,7]) > .5, 'Gripper did not remain closed after lift'
                assert np.min(xyz[:,2])-meta['init_position'][picked][2] > .03, 'Target not held above initial height'
        elif task == 'stack_cubes':
            names=[x['scene_name'] for x in roles['members']]
            assert len(names)==4 and np.max(actions[-15:,7])<.05
            for f in poses:
                for lower,upper in zip(names,names[1:]):
                    delta=np.array(f[upper][:3])-f[lower][:3]
                    assert np.all(np.abs(delta-np.array([0,0,.05]))<=np.array([.01,.01,.006])), 'Final stack alignment failed'
            for name in names:
                xyz=np.array([f[name][:3] for f in poses])
                assert np.max(np.linalg.norm(xyz-xyz[-1],axis=1)) < .005, 'Final stack still moving'
        spatial = None
        if task == 'spatial_directions':
            pick = meta['role_bindings']['pick']['scene_name']
            ref = meta['role_bindings']['ref']['scene_name']
            delta = np.array(meta['init_position'][pick][:3]) - meta['init_position'][ref][:3]
            relation = pick.rsplit('_', 1)[-1]
            phrase, axis, sign = {'left': ('to the left of', 1, 1),
                                  'right': ('to the right of', 1, -1),
                                  'front': ('in front of', 0, 1),
                                  'behind': ('behind', 0, -1)}[relation]
            assert phrase in meta['instruction'], 'Instruction/role relation mismatch'
            assert delta[axis] * sign > abs(delta[1-axis]), 'Initial geometry/relation mismatch'
            lifted = meta['final_position'][pick][2] - meta['init_position'][pick][2]
            assert lifted > .05, 'Selected target was not lifted'
            spatial = dict(target=pick, reference=ref, relation=relation,
                           delta_xyz=delta.tolist(), final_lift_m=lifted)
        out.update(raw_gate_pass=True, audit_version='terminal_and_frame_mask_v2', frames=n,
                   terminal_geometry=terminal,
                   excluded_frame_indices=np.flatnonzero(~frame_valid).tolist(),
                   valid_15_step_anchor_indices=np.flatnonzero(eligible).tolist(),
                   requires_frame_filter=bool(not frame_valid.all()),
                   valid_15_step_anchors=int(eligible.sum()),
                   instruction=meta['instruction'], timing=timing, cameras=camera_stats,
                   spatial_semantics=spatial,
                   action_fingerprint=hashlib.sha256(actions.astype('<f4').tobytes()).hexdigest(),
                   gripper_action_range=[float(actions[:,7].min()), float(actions[:,7].max())],
                   state_action_max_tracking_error=float(np.abs(state[:,:7]-actions[:,:7]).max()))
        if contact_sheet:
            sheet = Image.new('RGB', (1278, 3*264+48), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f'{task} seed={seed}: {meta["instruction"]}', fill='black')
            for row, (c, t) in enumerate(CAMS.items()):
                for col, (k, im) in enumerate(sorted(thumbs[t].items())):
                    x, y = col*426, row*264+48
                    sheet.paste(im, (x,y+24))
                    draw.text((x+6,y+4), f'{c} frame {k}/{n}', fill='black')
            Path(contact_sheet).parent.mkdir(parents=True, exist_ok=True)
            sheet.save(contact_sheet)
            out['contact_sheet'] = str(contact_sheet)
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out

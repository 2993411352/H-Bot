"""Seen-only instance, context and layout coverage. No simulator imports."""
import copy
import hashlib
import itertools
import math
from collections import Counter

VERSION = 'coverage_v5'
DIRECTIONS = ('front', 'right', 'left', 'behind')
ALIASES = {'pick_color': 'pick_attribute', 'spatial_pick': 'spatial_directions'}


def compact(row, task=None):
    """Only measured fields become coverage; missing source fields stay unknown."""
    if row.get('coverage_record'):
        return dict(row)
    meta = row.get('metadata', {})
    actors, roles = meta.get('actors', {}), meta.get('role_bindings', {})
    name = roles.get('pick', {}).get('scene_name')
    ref = roles.get('ref', {}).get('scene_name')
    place = roles.get('place', {}).get('scene_name')
    actor = actors.get(name, {})
    task = ALIASES.get(task or row.get('task', row.get('task_family')), task or row.get('task', row.get('task_family')))
    result = dict(coverage_record=True, task=task,
                  target_uuid=actor.get('actor_uuid'), category=actor.get('actor_category'),
                  relation=name.rsplit('_', 1)[-1] if task == 'spatial_directions' and name else None,
                  reference_uuid=actors.get(ref, {}).get('actor_uuid'),
                  reference_category=actors.get(ref, {}).get('actor_category'),
                  container_uuid=actors.get(place, {}).get('actor_uuid'),
                  container=actors.get(place, {}).get('actor_category'),
                  target_pose=meta.get('init_position', {}).get(name),
                  reference_pose=meta.get('init_position', {}).get(ref))
    if task == 'pick_attribute' and actors:
        result['distractor_uuids'] = sorted(a['actor_uuid'] for n, a in actors.items()
                                          if n != name and a.get('actor_type') == 'distractors')
    # Original LeRobot source_records include the manipulated instance, but not
    # distractor identities or initial object poses. Never invent those fields.
    if row.get('uuid'):
        result.update(target_uuid=row['uuid'], category=row.get('asset_id', '').rsplit('_', 1)[0])
        if task == 'spatial_directions':
            result['relation'] = {'front_of': 'front', 'left_of': 'left', 'right_of': 'right'}.get(row.get('attribute'), row.get('attribute'))
            text = row.get('instruction', '').rstrip('. ')
            for phrase in (' in front of the ', ' to the right of the ', ' to the left of the ', ' behind the '):
                if phrase in text:
                    result['reference_category'] = text.split(phrase, 1)[1].replace(' ', '_')
                    break
    return result


def catalog(rows):
    """Caller supplies only entries from the official seen split."""
    assets = [{k: r.get(k) for k in ('uuid', 'asset_id', 'category', 'tags', 'color', 'spec_type')}
              for r in rows]
    rigid = [a for a in assets if a['spec_type'] == 'usd.rigid_object']
    targets = [a for a in rigid if {'has_interactive_poses', 'is_graspable'} <= set(a['tags'] or [])]
    refs = [a for a in rigid if 'has_interactive_poses' in (a['tags'] or [])]
    peers = {}
    for a in targets:
        colors = set(a['color'] or [])
        peers[a['uuid']] = [b['uuid'] for b in rigid if b['category'] == a['category']
                           and b['uuid'] != a['uuid'] and colors and b['color']
                           and colors.isdisjoint(b['color'])]
    return dict(assets=assets, training_categories=sorted({a['category'] for a in targets}),
                coverage_pools=dict(pick_attribute=[a['uuid'] for a in targets if peers[a['uuid']]],
                                    spatial_directions=[a['uuid'] for a in targets],
                                    place_a2b=[a['uuid'] for a in targets],
                                    reference=[a['uuid'] for a in refs]),
                color_peers=peers,
                unsupported_target_uuids=[a['uuid'] for a in refs if a not in targets])


def tie(salt, *values):
    return hashlib.sha256(repr((salt, values)).encode()).hexdigest()


def records(rows, inventory, task):
    return [compact(r, task) for r in rows] + [r for r in inventory.get('coverage_baseline', []) if r['task'] == task]


def target_name(task, profile):
    return 'objects/pick_' + profile['relation'] if task == 'spatial_directions' else 'objects/pick_object'


def pose_bin(task, row):
    p = row.get('target_pose')
    if not p or len(p) != 7:
        return None
    if task == 'pick_attribute':
        w, x, y, z = p[3:]
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        quadrant = min(3, int((yaw+math.pi)/(math.pi/2)))
        return ('left' if p[1] >= 0 else 'right') + ':' + str(quadrant)
    ref = row.get('reference_pose')
    if task == 'spatial_directions' and ref and len(ref) == 7:
        distance = math.hypot(p[0]-ref[0], p[1]-ref[1])
        return ('left' if ref[1] >= 0 else 'right') + ':' + ('near' if distance < .275 else 'far')
    return None


def select(profile, task, rows, batches, inventory):
    """Select one candidate; pending candidates reserve actual identities."""
    p = copy.deepcopy(profile)
    if task not in ('pick_attribute', 'spatial_directions'):
        return p
    if not inventory.get('coverage_pools'):
        raise ValueError('Missing instance inventory; regenerate coverage inventory')
    assets = {a['uuid']: a for a in inventory['assets']}
    from targeting import THIN, COMPACT
    mode = p.get('category_pool', 'all')
    allowed = None if mode == 'all' else set(THIN if mode == 'thin' else COMPACT)
    pool = [u for u in inventory['coverage_pools'][task] if allowed is None or assets[u]['category'] in allowed]
    if not pool:
        raise ValueError('No eligible seen instances for requested profile')
    data = records(rows, inventory, task)
    history = [b for b in batches if b['task'] == task and 'profile' in b]
    active = [b for b in history if 'finished' not in b]
    count = Counter(r.get('target_uuid') for r in data)
    cats = Counter(r.get('category') for r in data)
    pairs = Counter((r.get('category'), r.get('relation')) for r in data)
    attempts = Counter()
    pair_attempts = Counter()
    pending = Counter()
    for b in history:
        q = b['profile']; attempts[q.get('target_uuid')] += b.get('count', 1)
        pair_attempts[q.get('category'), q.get('relation')] += b.get('count', 1)
    for b in active:
        q = b['profile']; n = b.get('count', 1)
        pending[q.get('target_uuid')] += n
        cats[q.get('category')] += n
        pairs[q.get('category'), q.get('relation')] += n
    salt = (inventory.get('selection_salt', 0), len(history))
    sizes = Counter(assets[u]['category'] for u in pool)
    def score(u):
        c = assets[u]['category']
        # Failed attempts affect retry ordering, never accepted coverage counts.
        effort = .35 * max(0, attempts[u] - count[u] - pending[u])
        instance = count[u] + pending[u] + effort
        if task == 'spatial_directions':
            pair = (c, p['relation'])
            pair_effort = .2 * max(0, pair_attempts[pair] - pairs[pair])
            return (min(1, pairs[pair] + pair_effort), instance, cats[c]/sizes[c], tie(salt, u))
        return (instance, cats[c]/sizes[c], tie(salt, u))
    uid = min(pool, key=score)
    p.update(coverage_version=VERSION, target_uuid=uid, category=assets[uid]['category'])
    if task == 'pick_attribute':
        peers = inventory['color_peers'][uid]
        combinations = list(itertools.combinations(sorted(peers), min(2, len(peers))))
        observed = Counter((r.get('target_uuid'), tuple(r['distractor_uuids'])) for r in data if r.get('distractor_uuids'))
        tried = Counter((b['profile'].get('target_uuid'), tuple(b['profile'].get('distractor_uuids', []))) for b in history)
        for b in active:
            q = b['profile']; observed[q.get('target_uuid'), tuple(q.get('distractor_uuids', []))] += b.get('count', 1)
        chosen = min(combinations, key=lambda c: (observed[uid, c], tried[uid, c], tie(salt, uid, c)))
        p['distractor_uuids'] = list(chosen)
    else:
        refs = [u for u in inventory['coverage_pools']['reference'] if assets[u]['category'] != p['category']]
        contexts = Counter((r.get('category'), r.get('relation'), r.get('reference_category')) for r in data)
        ref_counts = Counter(r.get('reference_uuid') for r in data)
        context_attempts = Counter()
        for b in history:
            q = b['profile']; key = (q.get('category'), q.get('relation'), q.get('reference_category'))
            context_attempts[key] += b.get('count', 1)
            if 'finished' not in b:
                contexts[key] += b.get('count', 1); ref_counts[q.get('reference_uuid')] += b.get('count', 1)
        ref = min(refs, key=lambda u: (contexts[p['category'], p['relation'], assets[u]['category']],
                                      context_attempts[p['category'], p['relation'], assets[u]['category']],
                                      ref_counts[u], tie(salt, uid, u)))
        p.update(reference_uuid=ref, reference_category=assets[ref]['category'])
    # One in five proposals retains the full original layout range.
    if len(history) % 5 == 0:
        p['layout_bin'] = 'full'
    else:
        bins = [s+':'+str(q) for s in ('left', 'right') for q in range(4)] if task == 'pick_attribute' else [s+':'+d for s in ('left', 'right') for d in ('near', 'far')]
        local = Counter(pose_bin(task, r) for r in data if r.get('target_uuid') == uid)
        category = Counter(pose_bin(task, r) for r in data if r.get('category') == p['category'])
        tried = Counter()
        for b in history:
            q = b['profile']
            if q.get('target_uuid') == uid:
                tried[q.get('layout_bin')] += b.get('count', 1)
                if 'finished' not in b:
                    local[q.get('layout_bin')] += b.get('count', 1)
        p['layout_bin'] = min(bins, key=lambda k: (local[k] + .35*tried[k], category[k], tie(salt, uid, k)))
    return p


def configure(cfg, task, p):
    if p.get('coverage_version') != VERSION:
        return cfg
    cfg = copy.deepcopy(cfg)
    assets = cfg['asset_configs']
    anchor = 'pick_front' if task == 'spatial_directions' else 'pick'
    assets[anchor]['uuid'] = p['target_uuid']
    assets[anchor]['split'] = 'seen'
    if task == 'pick_attribute':
        peers = p['distractor_uuids']
        assets['distractors'].setdefault('filter', {})['only_in'] = peers
        assets['distractors'].update(split='seen', min_count=len(peers), max_count=len(peers))
    else:
        assets['ref']['filter'] = dict(tags=['has_interactive_poses'], only_in=[p['reference_uuid']])
        assets['ref'].update(split='seen', min_count=1, max_count=1)
    layout = p['layout_bin']
    if layout != 'full':
        side, zone = layout.split(':')
        reset = cfg['task']['params']['pose_reset']
        if task == 'pick_attribute':
            reset['pose_range']['y'] = [0., .35] if side == 'left' else [-.35, 0.]
            q = int(zone)
            reset['pose_range']['yaw'] = [-math.pi+q*math.pi/2, -math.pi+(q+1)*math.pi/2]
        else:
            reset['anchor_range']['y'] = [0., .08] if side == 'left' else [-.08, 0.]
            reset['candidates']['pick_'+p['relation']]['radius'] = [.15, .275] if zone == 'near' else [.275, .4]
    return cfg


def verify(task, profile, meta):
    if profile.get('coverage_version') != VERSION:
        return []
    failures = []
    r = compact({'metadata': meta}, task)
    if r['target_uuid'] != profile['target_uuid']:
        failures.append('Actual target instance differs from requested UUID')
    if task == 'pick_attribute' and r.get('distractor_uuids') != sorted(profile['distractor_uuids']):
        failures.append('Actual color distractor combination differs from request')
    if task == 'spatial_directions':
        if r['reference_uuid'] != profile['reference_uuid'] or r['reference_category'] != profile['reference_category']:
            failures.append('Actual reference instance/category differs from request')
        candidates = [a.get('actor_uuid') for n, a in meta.get('actors', {}).items() if n in ['objects/pick_'+d for d in DIRECTIONS]]
        if len(candidates) != 4 or set(candidates) != {profile['target_uuid']}:
            failures.append('Spatial candidates must be four copies of the requested instance')
    # Settling can slightly move/rotate objects. Record the measured bin; do not
    # claim that the requested yaw is the actual yaw or discard valid successes.
    if not r.get('target_pose') or (task == 'spatial_directions' and not r.get('reference_pose')):
        failures.append('Missing measured initial object pose')
    return failures


def report(task, rows, inventory):
    rows = list(rows)
    data = records(rows, inventory, task)
    assets = {a['uuid']: a for a in inventory['assets']}
    pool = inventory['coverage_pools'][task]
    counts = Counter(r.get('target_uuid') for r in data)
    categories = sorted({assets[u]['category'] for u in pool})
    pairs = Counter((r.get('category'), r.get('relation')) for r in data)
    return dict(version=VERSION, accepted_synthetic=len(rows), baseline_records=len(data)-len(rows),
                eligible_instances=len(pool), covered_target_instances=sum(counts[u] > 0 for u in pool),
                eligible_categories=len(categories),
                missing_target_uuids=[u for u in pool if not counts[u]],
                category_counts=dict(sorted(Counter(r.get('category') for r in data if r.get('category')).items())),
                target_instance_counts={u: counts[u] for u in pool},
                category_direction_counts={c: {d: pairs[c, d] for d in DIRECTIONS} for c in categories} if task == 'spatial_directions' else {},
                measured_layout_counts=dict(Counter(pose_bin(task, r) for r in data if pose_bin(task, r))),
                known_color_combinations=len({(r.get('target_uuid'), tuple(r['distractor_uuids'])) for r in data if r.get('distractor_uuids')}),
                known_spatial_contexts=len({(r.get('category'), r.get('relation'), r['reference_category']) for r in data if r.get('reference_category')}))

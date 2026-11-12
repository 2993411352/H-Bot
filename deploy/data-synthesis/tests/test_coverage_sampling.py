import copy
import math
import unittest

from coverage_sampling import catalog, compact, configure, pose_bin, report, select, verify
from targeting import make_config, verify_metadata


def inv():
    rows = []
    for category in ('toothbrush', 'thermos', 'marker'):
        for i, color in enumerate(('red', 'blue', 'green', 'white')):
            rows.append(dict(uuid=category+str(i), asset_id=category+'_'+str(i),
                             category=category, color=[color],
                             spec_type='usd.rigid_object', tags=['is_graspable', 'has_interactive_poses']))
    return catalog(rows)


def row(uid, relation='front', reference='marker0'):
    category = uid[:-1]
    name = 'objects/pick_'+relation
    return dict(task='spatial_directions', metadata=dict(
        role_bindings=dict(pick=dict(scene_name=name), ref=dict(scene_name='objects/ref_0')),
        actors={**{'objects/pick_'+d: dict(actor_uuid=uid, actor_category=category) for d in ('front', 'right', 'left', 'behind')},
                'objects/ref_0': dict(actor_uuid=reference, actor_category=reference[:-1])},
        init_position={name: [.7, .04, .05, 1, 0, 0, 0], 'objects/ref_0': [.5, .04, .05, 1, 0, 0, 0]}))


class CoverageTests(unittest.TestCase):
    def test_original_source_coverage_does_not_invent_full_scenes(self):
        r = compact(dict(task_family='spatial_pick', uuid='thermos0', asset_id='thermos_001',
                         attribute='right_of', instruction='Pick up the thermos to the right of the marker.'))
        self.assertEqual((r['category'], r['relation'], r['reference_category']), ('thermos', 'right', 'marker'))
        self.assertIsNone(r['reference_uuid'])
        self.assertIsNone(r['target_pose'])
        self.assertNotIn('distractor_uuids', r)

    def test_instance_missing_from_original_and_new_data_selected(self):
        i = inv(); pool = i['coverage_pools']['pick_attribute']; missing = pool[-1]
        i['coverage_baseline'] = [dict(coverage_record=True, task='pick_attribute', target_uuid=u, category=u[:-1]) for u in pool if u != missing]
        p = select(dict(category_pool='all'), 'pick_attribute', [], [], i)
        self.assertEqual(p['target_uuid'], missing)
        self.assertEqual(len(p['distractor_uuids']), 2)
        self.assertNotIn(missing, p['distractor_uuids'])

    def test_pending_target_reservation_avoids_duplicate_parallel_selection(self):
        i = inv(); profile = dict(category_pool='all')
        a = select(profile, 'pick_attribute', [], [], i)
        b = select(profile, 'pick_attribute', [], [dict(task='pick_attribute', profile=a, count=1)], i)
        self.assertNotEqual(a['target_uuid'], b['target_uuid'])

    def test_spatial_targets_missing_category_direction_not_global_counts(self):
        i = inv()
        rows = [row('thermos'+str(n), 'left') for n in range(4)]
        rows += [row('toothbrush'+str(n), 'front') for n in range(4)]
        rows += [row('marker'+str(n), 'front', 'thermos0') for n in range(4)]
        p = select(dict(category_pool='all', relation='front'), 'spatial_directions', rows, [], i)
        self.assertEqual(p['category'], 'thermos')
        self.assertNotEqual(p['reference_category'], p['category'])

    def test_actual_reference_identity_and_four_clones_checked(self):
        i = inv(); p = select(dict(category_pool='all', relation='front'), 'spatial_directions', [], [], i)
        r = row(p['target_uuid'], 'front', p['reference_uuid'])
        self.assertEqual(verify('spatial_directions', p, r['metadata']), [])
        r['metadata']['actors']['objects/pick_left']['actor_uuid'] = 'bad'
        self.assertIn('Spatial candidates', verify('spatial_directions', p, r['metadata'])[0])

    def test_color_filter_keeps_seen_and_enforces_exact_peers(self):
        i = inv(); p = select(dict(category_pool='all'), 'pick_attribute', [], [], i)
        p['layout_bin'] = 'left:2'
        cfg = make_config(dict(asset_configs={'pick': {'filter': {}}, 'distractors': {'anchor': 'pick', 'match': ['category'], 'differ': ['color']}},
                               task={'params': {'pose_reset': {'pose_range': {'x': [.25, .55], 'y': [-.35, .35], 'yaw': [-math.pi, math.pi]}}}}), 'pick_attribute', p)
        self.assertEqual(cfg['asset_configs']['pick']['uuid'], p['target_uuid'])
        self.assertEqual(cfg['asset_configs']['distractors']['filter']['only_in'], p['distractor_uuids'])
        self.assertEqual(cfg['asset_configs']['distractors']['split'], 'seen')
        self.assertEqual(cfg['task']['params']['pose_reset']['pose_range']['y'], [0, .35])

    def test_no_compatible_color_peer_not_selected(self):
        i = inv(); assets = copy.deepcopy(i['assets'])
        assets.append(dict(uuid='solo', category='solo', asset_id='solo_001', tags=['is_graspable', 'has_interactive_poses'], color=['red'], spec_type='usd.rigid_object'))
        self.assertNotIn('solo', catalog(assets)['coverage_pools']['pick_attribute'])

    def test_failed_attempt_does_not_count_as_coverage_or_stick_forever(self):
        i = inv(); p = select(dict(category_pool='all'), 'pick_attribute', [], [], i)
        history = [dict(task='pick_attribute', profile=p, count=1, finished='now', raw_passed=0)] * 3
        other = select(dict(category_pool='all'), 'pick_attribute', [], history, i)
        self.assertNotEqual(p['target_uuid'], other['target_uuid'])
        self.assertEqual(report('pick_attribute', [], i)['covered_target_instances'], 0)

    def test_pose_report_uses_measurement_not_requested_bin(self):
        r = row('thermos0'); r['requested_layout_bin'] = 'right:far'
        self.assertEqual(pose_bin('spatial_directions', compact(r)), 'left:near')
        self.assertEqual(report('spatial_directions', [r], inv())['measured_layout_counts'], {'left:near': 1})

    def test_different_workers_break_ties_differently(self):
        chosen = set()
        for worker in range(10):
            i = inv(); i['selection_salt'] = worker
            chosen.add(select(dict(category_pool='all'), 'pick_attribute', [], [], i)['target_uuid'])
        self.assertGreater(len(chosen), 3)

    def test_historical_unprofiled_batches_are_preserved_without_crashing(self):
        p = select(dict(category_pool='all'), 'pick_attribute', [],
                   [dict(task='pick_attribute', seed=123, count=3, finished='past')], inv())
        self.assertTrue(p['target_uuid'])

    def test_wrong_distractor_rejected_even_when_task_successful(self):
        i = inv(); p = select(dict(category_pool='all'), 'pick_attribute', [], [], i)
        meta = dict(role_bindings={'pick': {'scene_name': 'objects/pick_object'}},
                    actors={'objects/pick_object': {'actor_uuid': p['target_uuid'], 'actor_category': p['category']},
                            'objects/distractor_0': {'actor_type': 'distractors', 'actor_uuid': 'other'}},
                    init_position={'objects/pick_object': [.4, .1, .1, 1, 0, 0, 0]})
        self.assertFalse(verify_metadata('pick_attribute', p, meta)['pass_check'])


if __name__ == '__main__':
    unittest.main()

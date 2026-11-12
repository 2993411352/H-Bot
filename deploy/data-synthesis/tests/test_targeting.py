"""Regression checks for task quotas and actual target validation."""
import unittest
from collections import Counter
from targeting import allocate, choose_profile, definitions, make_config, verify_metadata


class TargetingTest(unittest.TestCase):
    def test_choose_profile_failed_attempts_visits_untried_stratum(self):
        profiles = [dict(id='x', quota=10, weight=50),
                    dict(id='y', quota=10, weight=50)]
        self.assertEqual(choose_profile(
            profiles, [], [], {}, Counter({'x': 5}))['id'], 'y')

    def test_allocate_remaining_counts_preserves_four_hundred(self):
        for task, baseline in zip(definitions(), (73, 71, 71, 65)):
            with self.subTest(task=task):
                profiles = allocate(400 - baseline, definitions()[task])
                self.assertEqual(sum(p['quota'] for p in profiles) + baseline, 400)

    def test_choose_profile_reserved_last_slot_returns_none(self):
        profile = dict(id='x', quota=1, weight=100)
        self.assertIsNone(choose_profile(
            [profile], [], [dict(profile=profile, count=1)], {}))

    def test_verify_metadata_wrong_target_rejects_successful_distractor(self):
        meta = dict(role_bindings={'pick': {'scene_name': 'objects/pick_object'}},
                    actors={'objects/pick_object': {'actor_category': 'beverage_bottle'}})
        self.assertFalse(verify_metadata(
            'pick_attribute', dict(category='thermos'), meta)['pass_check'])

    def test_verify_metadata_wrong_direction_rejects_scene(self):
        meta = dict(role_bindings={'pick': {'scene_name': 'objects/pick_left'}},
                    actors={'objects/pick_left': {'actor_category': 'thermos'}})
        self.assertFalse(verify_metadata('spatial_directions',
            dict(category='thermos', relation='front'), meta)['pass_check'])

    def test_verify_metadata_requested_place_and_confusable_accepts_scene(self):
        meta = dict(role_bindings={'pick': {'scene_name': 'a'},
                                   'place': {'scene_name': 'b'}}, actors={
            'a': {'actor_category': 'thermos'}, 'b': {'actor_category': 'bowl'},
            'c': {'actor_category': 'beverage_bottle'}})
        self.assertTrue(verify_metadata('place_a2b', dict(
            category='thermos', container='bowl', confusable='beverage_bottle'),
            meta)['pass_check'])

    def test_make_config_place_preserves_six_pick_distractors(self):
        base = dict(asset_configs={
            'pick': dict(filter={}), 'place': dict(filter={}),
            'distractors_pick': dict(filter={}, min_count=6, max_count=6),
            'distractors_place': dict(min_count=1, max_count=1)})
        cfg = make_config(base, 'place_a2b', dict(category='thermos',
            container='bowl', confusable='beverage_bottle'))
        self.assertEqual(cfg['asset_configs']['distractors_pick']['max_count']
                         + cfg['asset_configs']['confusable_pick']['max_count'], 6)

    def test_verify_metadata_cube_region_mismatch_rejects_scene(self):
        names = [str(i) for i in range(4)]
        meta = dict(role_bindings={'members': [dict(scene_name=n) for n in names]},
                    init_position={n: [.4, -.1, 0, 1, 0, 0, 0] for n in names})
        self.assertFalse(verify_metadata('stack_cubes', dict(
            x_range=[.28, .52], y_range=[.08, .24]), meta)['pass_check'])


if __name__ == '__main__':
    unittest.main()

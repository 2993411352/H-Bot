"""Check coverage allocation and exact target validation behavior."""
import unittest
from collections import Counter

from targeting import (TASKS, choose_profile, profiles_for_remaining,
                       specialize, verify_metadata)


def row(category="toothbrush", relation="front", container="tray"):
    pick = "objects/pick_" + relation
    return {"metadata": {
        "role_bindings": {"pick": {"scene_name": pick},
                          "place": {"scene_name": "bowl"}},
        "actors": {pick: {"actor_category": category},
                   "bowl": {"actor_category": container}}}}


class DiversityTests(unittest.TestCase):
    def test_remaining_quota_keeps_all_prior_examples(self):
        for task in TASKS:
            baseline = [row() for _ in range(52)]
            profiles = profiles_for_remaining(task, baseline)
            self.assertEqual(sum(p["quota"] for p in profiles), 348)

    def test_spatial_corrects_existing_direction_imbalance(self):
        baseline = [row(relation=d) for d, n in
                    {"front": 52, "right": 38, "left": 25,
                     "behind": 26}.items() for _ in range(n)]
        p = profiles_for_remaining("spatial_directions", baseline)
        self.assertEqual({x["relation"]: x["quota"] for x in p},
                         {"front": 48, "right": 62, "left": 75,
                          "behind": 74})

    def test_active_reservation_blocks_last_slot(self):
        profile = dict(id="v3_x", quota=2, weight=100)
        self.assertIsNone(choose_profile(
            [profile], [], [dict(profile=profile, count=2)], {}, Counter()))

    def test_broad_pool_does_not_exclude_previous_hard_categories(self):
        p = specialize(dict(category_pool="all"), "pick_attribute", [], [],
                       dict(training_categories=["thermos"]))
        self.assertEqual(p["category"], "thermos")

    def test_coverage_selects_missing_category(self):
        p = specialize(dict(category_pool="all"), "pick_attribute",
                       [row("thermos")], [],
                       dict(training_categories=["thermos", "toothbrush"]))
        self.assertEqual(p["category"], "toothbrush")

    def test_contrast_pair_and_container_rotation(self):
        p = specialize(dict(category_pool="contrast"), "place_a2b",
                       [row(container="tray")], [],
                       dict(training_categories=["toothbrush", "toothpaste"]))
        self.assertEqual(p["category"], "toothpaste")
        self.assertEqual(p["confusable"], "toothbrush")
        self.assertNotEqual(p["container"], "tray")

    def test_requested_target_mismatch_is_rejected(self):
        self.assertFalse(verify_metadata(
            "place_a2b", dict(category="thermos", container="tray"),
            row()["metadata"])["pass_check"])

    def test_combination_profile_fills_missing_pair_before_category_total(self):
        baseline = [row("thermos", container=c) for c in
                    ("bowl", "tray", "basket", "plate")]
        baseline += [row("toothbrush", container="tray") for _ in range(8)]
        p = specialize(dict(id="v3_place_combinations", category_pool="all"),
                       "place_a2b", baseline, [],
                       dict(training_categories=["thermos", "toothbrush"]))
        self.assertEqual(p["category"], "toothbrush")
        self.assertNotEqual(p["container"], "tray")


if __name__ == "__main__":
    unittest.main()

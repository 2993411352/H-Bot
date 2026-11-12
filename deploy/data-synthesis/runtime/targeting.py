"""Select diverse seen training scenes without replaying evaluation scenes."""
import copy
from collections import Counter

from targeting_base import (  # Frozen, previously exercised implementation.
    TASKS, allocate, choose_profile, make_config as base_make_config, target_category,
    verify_metadata as base_verify_metadata,
)
import coverage_sampling

DIRECTIONS = ("front", "right", "left", "behind")
CONTAINERS = ("bowl", "basket", "tray", "plate")
THIN = ("toothbrush", "spoon", "fork", "chopsticks", "ballpoint_pen",
        "marker", "peeler", "spatula", "screwdriver", "fruit_knife")
COMPACT = ("pill_organizer", "measuring_tape", "earbud_case", "eraser",
           "charging_head", "dental_floss", "chocolate", "hair_clip")
CONTRAST_PAIRS = (
    ("toothbrush", "toothpaste"), ("toothpaste", "toothbrush"),
    ("spoon", "fork"), ("fork", "spoon"),
    ("ballpoint_pen", "marker"), ("marker", "ballpoint_pen"),
    ("thermos", "beverage_bottle"), ("beverage_bottle", "thermos"),
    ("peeler", "spoon"), ("chocolate", "hair_clip"),
    ("scissors", "wire_cutters"), ("wire_cutters", "scissors"),
)


def definitions():
    """Weights apply to remaining accepted slots, not existing examples."""
    return {
        "pick_attribute": [
            dict(id="v3_color_coverage", weight=65, category_pool="all"),
            dict(id="v3_color_thin", weight=20, category_pool="thin"),
            dict(id="v3_color_compact", weight=15, category_pool="compact"),
        ],
        "spatial_directions": [
            dict(id="v3_spatial_" + d, weight=25, relation=d,
                 category_pool="all") for d in DIRECTIONS
        ],
        "place_a2b": [
            dict(id="v3_place_coverage", weight=50, category_pool="all"),
            dict(id="v3_place_contrast", weight=35,
                 category_pool="contrast"),
            dict(id="v3_place_combinations", weight=15,
                 category_pool="all"),
        ],
        "stack_cubes": [
            dict(id="v3_stack_left", weight=20,
                 x_range=[0.28, 0.52], y_range=[0.08, 0.24]),
            dict(id="v3_stack_right", weight=20,
                 x_range=[0.28, 0.52], y_range=[-0.24, -0.08]),
            dict(id="v3_stack_far", weight=20,
                 x_range=[0.40, 0.52], y_range=[-0.24, 0.24]),
            dict(id="v3_stack_center", weight=20,
                 x_range=[0.31, 0.49], y_range=[-0.16, 0.16]),
            dict(id="v3_stack_full_workspace", weight=20,
                 x_range=[0.28, 0.52], y_range=[-0.24, 0.24]),
        ],
    }


def row_relation(row):
    name = row.get("metadata", {}).get("role_bindings", {}).get(
        "pick", {}).get("scene_name", "")
    return name.rsplit("_", 1)[-1]


def row_container(row):
    meta = row.get("metadata", {})
    name = meta.get("role_bindings", {}).get("place", {}).get("scene_name")
    return meta.get("actors", {}).get(name, {}).get("actor_category")


def profiles_for_remaining(task, rows):
    """Preserve all accepted examples; allocate only the unfilled quota."""
    rows = list(rows)
    remaining = 400 - len(rows)
    if task == "spatial_directions":
        counts = Counter(row_relation(r) for r in rows)
        if set(counts) - set(DIRECTIONS):
            raise ValueError("Unknown direction in accepted baseline")
        targets = {d: max(100, counts[d]) for d in DIRECTIONS}
        while sum(targets.values()) > 400:
            removable = [d for d in DIRECTIONS if targets[d] > counts[d]]
            if not removable:
                raise ValueError("Baseline exceeds target")
            d = max(removable, key=lambda d: (targets[d], d))
            targets[d] -= 1
        result = copy.deepcopy(definitions()[task])
        for p in result:
            p["quota"] = targets[p["relation"]] - counts[p["relation"]]
        assert sum(p["quota"] for p in result) == remaining
    else:
        result = allocate(remaining, definitions()[task])
    for p in result:
        p["task"] = task
    return result


def specialize(profile, task, rows, batches, inventory):
    """Prioritize sparse categories/combinations; randomize scenes via seeds."""
    profile = copy.deepcopy(profile)
    if task == "stack_cubes":
        return profile
    rows = list(rows)
    if inventory.get('coverage_pools') and task in ('pick_attribute', 'spatial_directions'):
        return coverage_sampling.select(profile, task, rows, batches, inventory)
    allowed = set(inventory["training_categories"])
    mode = profile["category_pool"]
    pool = sorted(allowed if mode == "all" else
                  allowed.intersection(THIN if mode == "thin" else
                                       COMPACT if mode == "compact" else
                                       [a for a, b in CONTRAST_PAIRS
                                        if b in allowed]))
    if not pool:
        raise ValueError("No permitted training categories for profile")
    counts = Counter(target_category(r) for r in rows)
    attempts = Counter()
    pending = Counter()
    for batch in batches:
        if batch["task"] != task:
            continue
        category = batch["profile"].get("category")
        attempts[category] += batch["count"]
        if "finished" not in batch:
            pending[category] += batch["count"]
    if task == "spatial_directions":
        relation = profile["relation"]
        pair_counts = Counter(target_category(r) for r in rows
                              if row_relation(r) == relation)
        category = min(pool, key=lambda c: (
            pair_counts[c] + pending[c], counts[c], attempts[c], c))
    else:
        category = min(pool, key=lambda c: (
            counts[c] + pending[c], attempts[c], c))
    profile["category"] = category
    if task == "place_a2b":
        containers = Counter(row_container(r) for r in rows)
        pairs = Counter((target_category(r), row_container(r)) for r in rows)
        for batch in batches:
            if batch["task"] == task and "finished" not in batch:
                bp = batch["profile"]
                pairs[bp.get("category"), bp.get("container")] += batch["count"]
                containers[bp.get("container")] += batch["count"]
        if profile.get("id") == "v3_place_combinations":
            category, _ = min(
                ((c, vessel) for c in pool for vessel in CONTAINERS),
                key=lambda pair: (pairs[pair],
                                  counts[pair[0]] + pending[pair[0]],
                                  containers[pair[1]], pair))
            profile["category"] = category
        profile["container"] = min(CONTAINERS, key=lambda c: (
            pairs[category, c], containers[c], c))
        if mode == "contrast":
            profile["confusable"] = next(b for a, b in CONTRAST_PAIRS
                                          if a == category and b in allowed)
    return profile


def make_config(base_config, task, profile):
    return coverage_sampling.configure(base_make_config(base_config, task, profile), task, profile)


def verify_metadata(task, profile, meta):
    result = base_verify_metadata(task, profile, meta)
    result['failures'] += coverage_sampling.verify(task, profile, meta)
    result['pass_check'] = not result['failures']
    if profile.get('coverage_version'):
        actual = coverage_sampling.compact({'metadata': meta}, task)
        result.update(requested_layout_bin=profile.get('layout_bin'),
                      measured_layout_bin=coverage_sampling.pose_bin(task, actual),
                      actual_scene=actual)
    return result

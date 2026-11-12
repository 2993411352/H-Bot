"""Explicit training-scene strata and quota scheduling."""
import copy
from collections import Counter

TASKS = ("pick_attribute", "spatial_directions", "place_a2b", "stack_cubes")
DIRECTIONS = ("front", "right", "left", "behind")


def definitions():
    return {
        "pick_attribute": [
            dict(id="color_carrot", weight=18, category="carrot"),
            dict(id="color_peeler", weight=18, category="peeler"),
            dict(id="color_chocolate", weight=18, category="chocolate"),
            dict(id="color_thermos", weight=16, category="thermos"),
            dict(id="color_coverage", weight=30, coverage=True),
        ],
        "spatial_directions": [
            dict(id="spatial_thermos_front", weight=20,
                 category="thermos", relation="front"),
            dict(id="spatial_peeler_front", weight=20,
                 category="peeler", relation="front"),
            dict(id="spatial_thermos_right", weight=15,
                 category="thermos", relation="right"),
            dict(id="spatial_peeler_right", weight=15,
                 category="peeler", relation="right"),
            dict(id="spatial_coverage", weight=30, coverage=True),
        ],
        "place_a2b": [
            dict(id="place_thermos_beverage_bowl", weight=20,
                 category="thermos", confusable="beverage_bottle",
                 container="bowl"),
            dict(id="place_chocolate_hairclip_tray", weight=20,
                 category="chocolate", confusable="hair_clip",
                 container="tray"),
            dict(id="place_carrot_peeler_tray", weight=15,
                 category="carrot", confusable="peeler", container="tray"),
            dict(id="place_peeler_spoon_basket", weight=15,
                 category="peeler", confusable="spoon", container="basket"),
            dict(id="place_coverage", weight=30, coverage=True),
        ],
        "stack_cubes": [
            dict(id="stack_left", weight=25,
                 x_range=[0.28, 0.52], y_range=[0.08, 0.24]),
            dict(id="stack_right", weight=25,
                 x_range=[0.28, 0.52], y_range=[-0.24, -0.08]),
            dict(id="stack_far", weight=20,
                 x_range=[0.40, 0.52], y_range=[-0.24, 0.24]),
            dict(id="stack_center", weight=30,
                 x_range=[0.31, 0.49], y_range=[-0.16, 0.16]),
        ],
    }


def allocate(remaining, profiles):
    if remaining < 0 or sum(p["weight"] for p in profiles) != 100:
        raise ValueError("Invalid remaining count or profile weights")
    result = copy.deepcopy(profiles)
    for p in result:
        p["quota"] = remaining * p["weight"] // 100
    extra = remaining - sum(p["quota"] for p in result)
    ordered = sorted(result, key=lambda p: (
        -(remaining * p["weight"] % 100), p["id"]))
    for p in ordered[:extra]:
        p["quota"] += 1
    return result


def target_category(row):
    meta = row.get("metadata", {})
    role = meta.get("role_bindings", {}).get("pick", {})
    return meta.get("actors", {}).get(role.get("scene_name"), {}).get(
        "actor_category")


def choose_profile(profiles, rows, active, blocked, attempts=None):
    attempts = attempts or Counter()
    accepted = Counter(r.get("targeted_profile") for r in rows)
    reserved = Counter()
    for b in active:
        reserved[b["profile"]["id"]] += b["count"]
    available = [p for p in profiles if p["id"] not in blocked
                 and accepted[p["id"]] + reserved[p["id"]] < p["quota"]]
    if not available:
        return None
    return copy.deepcopy(min(available, key=lambda p: (
        (accepted[p["id"]] + reserved[p["id"]]) / p["quota"],
        attempts[p["id"]] / p["quota"],
        bool(p.get("coverage") or p["id"] == "stack_center"),
        -p["weight"], p["id"])))


def specialize(profile, task, rows, batches, inventory):
    profile = copy.deepcopy(profile)
    if profile.get("coverage"):
        counts = Counter(target_category(r) for r in rows)
        attempts = Counter()
        for b in batches:
            if b["task"] == task:
                attempts[b["profile"].get("category")] += b["count"]
        excluded = {"carrot", "peeler", "chocolate", "thermos"}
        pool = [c for c in inventory["training_categories"]
                if c not in excluded]
        if not pool:
            raise ValueError("Empty seen-asset coverage pool")
        profile["category"] = min(pool, key=lambda c: (
            counts[c], attempts[c], c))
        if task == "spatial_directions":
            directions = Counter()
            for row in rows:
                meta = row.get("metadata", {})
                name = meta.get("role_bindings", {}).get("pick", {}).get(
                    "scene_name", "")
                directions[name.rsplit("_", 1)[-1]] += 1
            for b in batches:
                if b["task"] == task and "finished" not in b:
                    directions[b["profile"].get("relation")] += b["count"]
            profile["relation"] = min(DIRECTIONS, key=lambda d: directions[d])
    return profile


def make_config(base_config, task, profile):
    cfg = copy.deepcopy(base_config)
    assets = cfg["asset_configs"]
    if task == "stack_cubes":
        ranges = cfg["task"]["params"]["pose_reset"]["pose_range"]
        ranges["x"] = profile["x_range"]
        ranges["y"] = profile["y_range"]
        return cfg
    anchor = "pick_front" if task == "spatial_directions" else "pick"
    assets[anchor]["filter"]["category"] = profile["category"]
    if task == "place_a2b":
        if profile.get("container"):
            assets["place"]["filter"]["category"] = profile["container"]
        if profile.get("confusable"):
            assets["distractors_pick"]["min_count"] = 5
            assets["distractors_pick"]["max_count"] = 5
            assets["distractors_pick"]["filter"]["exclude_categories"] = [
                profile["confusable"]]
            cfg["asset_configs"] = {
                "pick": assets["pick"], "place": assets["place"],
                "confusable_pick": dict(
                    anchor="pick", split="seen", differ=["category"],
                    filter=dict(tags=["has_interactive_poses"],
                                category=profile["confusable"]),
                    min_count=1, max_count=1,
                    prim_name_prefix="hard_negative"),
                "distractors_pick": assets["distractors_pick"],
                "distractors_place": assets["distractors_place"],
            }
    return cfg


def verify_metadata(task, profile, meta):
    failures = []
    roles = meta.get("role_bindings", {})
    actors = meta.get("actors", {})
    if task == "stack_cubes":
        members = roles.get("members", [])
        if len(members) != 4:
            failures.append("Expected four bound stack members")
        for member in members:
            pos = meta.get("init_position", {}).get(member["scene_name"], [])
            if len(pos) != 7:
                failures.append("Missing initial cube pose")
                continue
            for axis, key in enumerate(("x_range", "y_range")):
                low, high = profile[key]
                if not low - 0.005 <= pos[axis] <= high + 0.005:
                    failures.append("Initial cube outside requested region")
    else:
        selected = roles.get("pick", {}).get("scene_name", "")
        if actors.get(selected, {}).get("actor_category") != profile["category"]:
            failures.append("Bound pick category differs from quota")
        if task == "spatial_directions":
            if selected != "objects/pick_" + profile["relation"]:
                failures.append("Bound direction differs from quota")
        if task == "place_a2b":
            placed = roles.get("place", {}).get("scene_name", "")
            if profile.get("container") and actors.get(placed, {}).get(
                    "actor_category") != profile["container"]:
                failures.append("Bound container differs from quota")
            if profile.get("confusable") and not any(
                name != selected and actor.get("actor_category") == profile["confusable"]
                for name, actor in actors.items()
            ):
                failures.append("Required confusable distractor absent")
    return dict(pass_check=not failures, failures=failures)

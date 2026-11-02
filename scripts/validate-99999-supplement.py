"""CPU-only check of exact original episode coverage for a two-worker retest."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path('/home/users/wangmengfan')
sys.path.insert(0, str(root / 'robo_orchard_sim'))
from robo_orchard_sim.pipeline.evaluator.distributed import EvaluationExecutionConfig, build_evaluation_shards
from robo_orchard_sim.pipeline.evaluator.distributed.scheduling import EvaluationWorkload, allocate_task_shards

source = root / 'outputs/sim/local_pi05_full_99999_parallel_20261101_191358'
source_config = source / 'config_snapshot' / (source.name + '.yaml')
spec = importlib.util.spec_from_file_location('eval_entry', root / 'robo_orchard_sim/examples/manipulation-app/scripts/eval_policy.py')
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)
old = entry.load_eval_config(source_config)
current = entry.load_eval_config(Path(sys.argv[1])) if len(sys.argv) > 1 else old
names = {'pick_color', 'pick_spatial', 'place_a2b'}
tasks = [t for t in current.tasks if t.name in names]
assert {t.name for t in tasks} == names
if len(sys.argv) > 1:
    assert len(current.tasks) == 3
workloads = [EvaluationWorkload(episodes_per_config=t.episode_num, max_steps=t.max_steps,
    config_count=1, episodes_per_scene=t.swap.swap_per_scene if t.swap.enabled else 1,
    shards=t.shards) for t in tasks]
counts = allocate_task_shards(workloads, 2)
rows = []
for task, count in zip(tasks, counts, strict=True):
    original = next(t for t in old.tasks if t.name == task.name)
    assert task == original
    plan = SimpleNamespace(task=task.task_type, episodes_per_config=task.episode_num,
        groups=[SimpleNamespace(group_id='group_0000', seed=task.seed, configs=[task.yaml])])
    shards = build_evaluation_shards(plan=plan, task_name=task.name,
        shards_per_config=count, episodes_per_scene=task.swap.swap_per_scene if task.swap.enabled else 1,
        scene_seeds=task.scene_seeds)
    original_shards = json.loads((source / task.name / 'evaluation_manifest.json').read_text())['shards']
    current_keys = [k for shard in shards for k in shard.episode_keys()]
    assert len(current_keys) == len(set(current_keys)) == 50
    fields = ('shard_id', 'episode_count', 'episode_offset', 'seed_start',
              'fallback_seed_start', 'fallback_seed_count', 'episodes_per_scene')
    assert len(shards) == len(original_shards)
    for shard, before in zip(shards, original_shards, strict=True):
        after = shard.to_payload()
        assert all(after[k] == before[k] for k in fields), (after, before)
    rows.append(dict(task=task.name, episodes=50, seed=task.seed,
                     shards=[s.episode_count for s in shards], max_steps=task.max_steps))
assert sum(r['episodes'] for r in rows) == 150
assert EvaluationExecutionConfig(gpus=('0',), workers_per_gpu=2).concurrency(sum(counts)) == 2
print(json.dumps(dict(coverage='PASS', concurrency=2, tasks=rows), indent=2))

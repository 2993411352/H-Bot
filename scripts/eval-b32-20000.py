"""Run-local heterogeneous slot adapter; original workers stay unmodified."""
import importlib.util
import json
import os
from pathlib import Path
import sys


class MixedExecution:
    """GPU0 has one simulator; GPU1 has two. Model resides on GPU0."""

    def __init__(self, gpus, workers_per_gpu=1):
        if tuple(gpus) != ('0', '1') or workers_per_gpu != 1:
            raise ValueError('This run requires GPUs 0,1 and slot multiplier 1')
        self.gpus = ('0', '1')
        self.workers_per_gpu = {'0': 1, '1': 2}

    def gpu_slots(self):
        return ('1', '1', '0')

    def concurrency(self, work_item_count):
        return min(max(work_item_count, 0), 3)


def load_entry():
    root = Path(__file__).resolve().parent.parent
    repo = root / 'robo_orchard_sim'
    sys.path.insert(0, str(repo))
    path = repo / 'examples/manipulation-app/scripts/eval_policy.py'
    spec = importlib.util.spec_from_file_location('mixed_eval_entry', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(path):
    entry = load_entry()
    from robo_orchard_sim.pipeline.evaluator.distributed import build_evaluation_shards
    from robo_orchard_sim.pipeline.evaluator.distributed.scheduling import EvaluationWorkload, allocate_task_shards
    cfg = entry.load_eval_config(path)
    plans = [entry._build_plan(t) for t in cfg.tasks]
    workloads = [EvaluationWorkload(
        episodes_per_config=p.episodes_per_config, max_steps=t.max_steps,
        config_count=sum(len(g.configs) for g in p.groups),
        episodes_per_scene=t.swap.swap_per_scene if t.swap.enabled else 1,
        shards=t.shards) for t, p in zip(cfg.tasks, plans, strict=True)]
    counts = allocate_task_shards(workloads, 3)
    rows = []
    for task, plan, count in zip(cfg.tasks, plans, counts, strict=True):
        options = dict(plan=plan, task_name=task.name,
            episodes_per_scene=task.swap.swap_per_scene if task.swap.enabled else 1,
            scene_seeds=task.scene_seeds)
        shards = build_evaluation_shards(shards_per_config=count, **options)
        keys = [k for s in shards for k in s.episode_keys()]
        baseline = [k for s in build_evaluation_shards(shards_per_config=1, **options) for k in s.episode_keys()]
        if keys != baseline or len(keys) != len(set(keys)) or len(keys) != 50 or task.seed != 100000:
            raise ValueError('Duplicate, missing or changed episode coverage: ' + task.name)
        expected = 4 if task.name == 'stack_cubes' else 2 if task.name == 'place_a2b' else 1
        if len(shards) != expected:
            raise ValueError('Unexpected shard count: ' + task.name)
        rows.append(dict(task=task.name, seed=task.seed, max_steps=task.max_steps,
            episodes=len(keys), shards=[s.episode_count for s in shards]))
    if len(rows) != 6 or sum(r['episodes'] for r in rows) != 300:
        raise ValueError('Expected six tasks, 300 episodes')
    return dict(coverage='PASS', concurrency=3, gpu_slots=['1', '1', '0'], tasks=rows)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--validate-config':
        print(json.dumps(validate(Path(sys.argv[2])), indent=2))
        return
    entry = load_entry()
    if '--_shard-id' in sys.argv:
        # CUDA ordinal 0 is the worker's masked physical GPU. Vulkan ignores
        # CUDA_VISIBLE_DEVICES, so pin its physical adapter separately.
        physical = os.environ.get('CUDA_VISIBLE_DEVICES')
        if physical not in ('0', '1'):
            raise ValueError('Expected exactly one physical GPU per worker')
        import robo_orchard_sim.launcher as launcher
        original = launcher.AppLauncher

        class PinnedAppLauncher(original):
            def _create_app(self):
                self._sim_app_config.update(active_gpu=int(physical),
                                            physics_gpu=0, multi_gpu=False)
                print('[device-pin] CUDA logical=0 physical=' + physical +
                      ' Vulkan adapter=' + physical + ' multi_gpu=False',
                      flush=True)
                super()._create_app()

        launcher.AppLauncher = PinnedAppLauncher
        entry.main()
        return
    import robo_orchard_sim.pipeline.evaluator.distributed as distributed
    # This binding changes ONLY this dispatcher process. Workers use the
    # official evaluator, with their launch device pinned by this wrapper.
    distributed.EvaluationExecutionConfig = MixedExecution
    build_command = entry._build_shard_worker_cmd

    def worker_command(**kwargs):
        command = build_command(**kwargs)
        command[1] = str(Path(__file__).resolve())
        return command

    entry._build_shard_worker_cmd = worker_command
    print('[mixed-slots] GPU0=1 GPU1=2 model=GPU0 concurrency=3', flush=True)
    entry.main()


if __name__ == '__main__':
    main()

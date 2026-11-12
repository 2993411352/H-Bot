"""Pinned headless simulation; durable per-episode results before app shutdown."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from audit_raw import audit, atomic_json
from dedup_registry import check as check_duplicate, READY as DEDUP_READY
from targeting import verify_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', required=True)
    args = p.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    assert profile['task'] == args.task
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    physical = os.environ['CUDA_VISIBLE_DEVICES']
    assert physical == '0', 'Expose only the selected GPU to this container'
    import robo_orchard_sim.launcher as launcher
    original = launcher.AppLauncher

    class Pinned(original):
        def _create_app(self):
            self._sim_app_config.update(active_gpu=0, physics_gpu=0, multi_gpu=False)
            print(f'DEVICE_PIN cuda_logical=0 physical={physical} vulkan={physical}', flush=True)
            super()._create_app()

    launcher.AppLauncher = Pinned
    from robo_orchard_sim.pipeline.data_synthesis.single_task import (
        LaunchConfig, TaskDataSynthesisCfg, TaskDataSynthesisRunner, data_synthesis_runtime)

    class AuditedRunner(TaskDataSynthesisRunner):
        @staticmethod
        def build_role_registry(*, runtime_task, seed, swap_enabled=False):
            registry = TaskDataSynthesisRunner.build_role_registry(
                runtime_task=runtime_task, seed=seed,
                swap_enabled=swap_enabled)
            requested = {}
            if args.task == 'spatial_directions':
                requested['pick'] = 'objects/pick_' + profile['relation']
            elif args.task in ('pick_attribute', 'place_a2b'):
                requested['pick'] = 'objects/pick_object'
            if args.task == 'place_a2b':
                requested['place'] = 'objects/place_object'
            for role, name in requested.items():
                matches = [candidate for candidate in
                           runtime_task.get_role_candidates(role, swap=True)
                           if candidate.scene_name == name]
                assert len(matches) == 1, (role, name, matches)
                registry.bind_one(0, role, matches[0])
            return registry

        def _run_atomic_action_loop(self, **kwargs):
            self._terminal_validator = kwargs['validator']
            self._terminal_checks = []
            steps, reason, output = super()._run_atomic_action_loop(**kwargs)
            if self._terminal_checks:
                output = self._terminal_output
            return steps, reason, output

        def wait_until_robot_stationary(self, env, sim_app):
            criterion = dict(pick_attribute='lift_pick', spatial_directions='lift_pick', place_a2b='place_within_xy',
                             stack_cubes='settled_stack')[args.task]
            for _ in range(30):
                if not sim_app.is_running():
                    break
                env.step()
                output = self._terminal_validator.evaluate(env, env_idx=0)
                current = output.metrics['criteria_met_now']
                assert criterion in current, (criterion, current)
                self._terminal_checks.append(bool(current[criterion]))
                self._terminal_output = output

        def run_episode(self, **kwargs):
            self._terminal_checks = []
            diagnostic = out / f'control_diagnostics_{kwargs["seed"]}.jsonl'
            os.environ['P2C_DIAGNOSTICS_PATH'] = str(diagnostic)
            os.environ['HBOT_SCENE_INIT_AUDIT_DIR']=str(out/f'scene_init_{kwargs["seed"]}')
            os.environ['HBOT_EPISODE_SEED']=str(kwargs['seed'])
            summary = super().run_episode(**kwargs)
            stable = len(self._terminal_checks) == 30 and all(self._terminal_checks[-15:])
            row = dict(summary=asdict(summary), audits=[], terminal_stability_pass=stable,
                       terminal_checks=self._terminal_checks,
                       collector_version='portable_targeted_v1',
                       targeted_profile=profile, diagnostics=str(diagnostic),
                       requested_config_sha256=hashlib.sha256(
                           Path(args.config).read_bytes()).hexdigest(),
                       collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            init_path=out/f'scene_init_{kwargs["seed"]}/scene_init_audit.json'
            if init_path.exists():
                import json
                row['scene_initialization']=json.loads(init_path.read_text())
            if summary.success and not summary.error:
                for i, path in enumerate(summary.mcap_paths):
                    row['audits'].append(audit(path, args.task, summary.seed,
                        out / f'preview_{summary.seed}_{i}.jpg' if kwargs['episode_index'] == 0 else None,
                        terminal_proof=stable))
            for audit_result in row['audits']:
                evidence = verify_metadata(
                    args.task, profile, audit_result.get('metadata', {}))
                audit_result.update(
                    targeted_profile=profile['id'],
                    targeted_category=profile.get('category'),
                    targeted_relation=profile.get('relation'),
                    targeted_evidence=evidence,
                    diagnostics=str(diagnostic))
                audit_result['raw_gate_pass'] = bool(
                    audit_result['raw_gate_pass'] and evidence['pass_check'])
            row['raw_gate_pass'] = bool(summary.success and not summary.error and stable
                                       and len(row['audits']) == 1
                                       and row['audits'][0]['raw_gate_pass'])
            if row['raw_gate_pass']:
                a=row['audits'][0]
                if DEDUP_READY.exists():
                    try:
                        result=check_duplicate(a['source'],args.task,summary.seed)
                        a['dedup']=result;a['dedup_gate_pass']=result['pass_check']
                        row['dedup_gate_pass']=result['pass_check']
                    except Exception as exc:
                        a['dedup_gate_pass']=False;row['dedup_gate_pass']=False
                        row['dedup_pending']=repr(exc)
                else:
                    a['dedup_gate_pass']=False
                    row['dedup_gate_pass']=False
                    row['dedup_pending']='Baseline index not ready; post-audit required before counting'
            atomic_json(out / f'episode_{summary.seed}.json', row)
            print('EPISODE_AUDIT', summary.seed, row['raw_gate_pass'], flush=True)
            return summary

    cfg = TaskDataSynthesisCfg(task=args.task, config=args.config,
        asset_root='/assets/OBJECTS_v2',
        splits_path=Path('/assets/OBJECTS_v2/splits/hbot_panda_droid_split_seen.yaml'),
        seed=args.seed, episode_num=args.episodes, max_steps=args.max_steps,
        task_save_root=str(out), swap_enabled=True, enable_recording=True,
        launch=LaunchConfig(headless=True, enable_cameras=True, virtual_display=True))
    with data_synthesis_runtime(cfg.launch) as runtime:
        from compact_ik import install
        install()
        from robo_orchard_sim.asset_manager.registry import AssetRegistry
        original_registry_init = AssetRegistry.__init__

        def frozen_registry(self, asset_root, *, index_path=None, auto_build_index=False):
            original_registry_init(self, asset_root, index_path=index_path,
                                   auto_build_index=False)

        AssetRegistry.__init__ = frozen_registry
        result = AuditedRunner(cfg).run_with_runtime(sim_app=runtime.sim_app)
        atomic_json(out / 'result.json', asdict(result))
        print('BATCH_RESULT_SAVED', result.success_count, result.total, flush=True)


if __name__ == '__main__':
    main()

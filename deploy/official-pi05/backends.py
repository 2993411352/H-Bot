"""Backend imports are deliberately lazy: JAX and LoongForge use separate venvs."""
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent


def enable_loongforge():
    root = PACKAGE / 'loongforge_backend'
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'vendor'))
    return root


def prepare_backend(name, checkpoint, tokenizer):
    if name == 'loongforge':
        enable_loongforge()
        from hbot_inference import Codec, SessionPolicy, example_observation
        from model_runtime import TorchEngine, check_checkpoint, verify_runtime
        verify_runtime()
        report = check_checkpoint(checkpoint)
        stats = checkpoint / 'norm_stats.json'
        codec = Codec(stats, checkpoint / 'hbot_contract.json', tokenizer)

        def check():
            print(json.dumps(dict(status='CPU_CHECKPOINT_PASSED', backend=name, **report,
                                  weights_loaded=False, gpu_inference_tested=False)), flush=True)

        def load():
            import torch
            torch.set_num_threads(8)
            return SessionPolicy(TorchEngine(checkpoint, denoise_steps=10), codec,
                                 action_steps=15, seed=20261106)

        return dict(check=check, load=load, observation=example_observation, stats=stats)

    if name != 'jax':
        raise ValueError('Unknown inference backend')
    import pyarrow.parquet  # Preserve working native-library import order.
    import yaml
    from legacy_server import checkpoint_metadata, cpu_check
    from pi05_server_config import get_config, make_adapter, example_observation
    settings = yaml.safe_load((PACKAGE / 'model.yaml').read_text())['hbot-full']
    cfg = get_config('pi05_hbot_full')
    shapes, stats = checkpoint_metadata(checkpoint, cfg.data.assets.asset_id, 'hbot-full')
    adapter = make_adapter('hbot-full', settings)

    def check():
        cpu_check(cfg, adapter, 'hbot-full', settings, shapes, stats)

    def load():
        from openpi.policies.policy_config import create_trained_policy
        from robo_orchard_server.policy.pi05.policy import Pi05Policy
        pipeline = create_trained_policy(cfg, checkpoint, sample_kwargs={'num_steps': 10})
        return Pi05Policy(pipeline, adapter, steps=15)

    return dict(check=check, load=load, observation=example_observation, stats=stats)

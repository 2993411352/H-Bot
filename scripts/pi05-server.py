#!/usr/bin/env python3
"""Local pi05 service: experimental base, matching H-Bot LoRA or full finetune.

--check is CPU-only: real tokenizer/transforms and abstract parameter shapes,
but no weight restore, GPU inference, socket listener or simulator launch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot"))
PROFILES = ("base-experimental", "hbot-lora", "hbot-full")


def checkpoint_metadata(checkpoint, asset_id, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    stats = checkpoint / "assets" / asset_id / "norm_stats.json"
    metadata = checkpoint / "params/_METADATA"
    if not stats.is_file() or not metadata.is_file():
        raise ValueError(f"Need a complete checkpoint directory (not params/): {metadata}, {stats}")
    tree = json.loads(metadata.read_text())["tree_metadata"]
    shapes = {}
    for item in tree.values():
        keys = [entry["key"] for entry in item["key_metadata"]]
        if keys[0] != "params":
            raise ValueError("Unexpected checkpoint parameter root")
        shapes["/".join(keys[1:])] = tuple(item["value_metadata"]["write_shape"])
    # Match OpenPI restore_params: training checkpoints wrap every NNX leaf in
    # /value, while exported base checkpoints may already use pure parameter keys.
    # Normalize only metadata for the CPU shape check; never rewrite the weights.
    if shapes and all(key.endswith("/value") for key in shapes):
        shapes = {key.removesuffix("/value"): shape for key, shape in shapes.items()}
    has_lora = any("lora" in key for key in shapes)
    if has_lora != (profile == "hbot-lora"):
        actual = "LoRA parameters" if has_lora else "no LoRA parameters"
        hint = "hbot-lora" if has_lora else "hbot-full"
        raise ValueError(f"Checkpoint contains {actual}, incompatible with --profile {profile}. "
                         f"For a matching H-Bot finetuned checkpoint, use --profile {hint}; "
                         "base-experimental is only for the DROID base smoke test.")
    raw_stats = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            if len(raw_stats[name][key]) != 8:
                raise ValueError("This profile requires eight-dimensional state/action normalization")
    return shapes, stats


def check_resources(gpu, host, port, allow_busy):
    if not gpu.isdigit():
        raise ValueError("--gpu must be one physical NVIDIA GPU index")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).strip()
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
        text=True, timeout=10).splitlines()
    occupants = [row for row in rows if row.split(",")[0].strip() == gpu_uuid]
    if occupants and not allow_busy:
        raise ValueError(f"GPU {gpu} is already occupied: {occupants}. No process was stopped. "
                         "Free it yourself or explicitly use --allow-busy-gpu after checking memory.")
    with socket.socket() as probe:
        probe.bind((host, port))


def cpu_check(cfg, adapter, profile, settings, checkpoint_shapes, stats):
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    import numpy as np
    from openpi import transforms as T
    from openpi.shared import normalize
    from pi05_server_config import example_observation

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Preflight must not initialize a GPU")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = normalize.load(stats.parent)
    obs = example_observation()
    raw = adapter.build_model_input(obs)
    inputs = T.compose([*data.data_transforms.inputs, T.Normalize(norm, use_quantiles=data.use_quantile_norm),
                        *data.model_transforms.inputs])(raw)
    assert inputs["state"].shape == (32,)
    assert inputs["tokenized_prompt"].shape == (200,)
    assert all(image.shape == (224, 224, 3) for image in inputs["image"].values())
    assert sum(bool(v) for v in inputs["image_mask"].values()) == (2 if profile == "base-experimental" else 3)
    # Exercise the exact unnormalize/output path with synthetic normalized output.
    outputs = T.compose([*data.model_transforms.outputs,
                         T.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
                         *data.data_transforms.outputs])(
                             {"state": inputs["state"].copy(),
                              "actions": np.zeros((cfg.model.action_horizon, 32), np.float32)})
    actions = adapter.build_action_sequence(outputs["actions"], obs, steps=settings["action_steps"])
    assert len(actions) == settings["action_steps"]
    assert all(action["values"].shape == (1, 9) and np.isfinite(action["values"]).all() for action in actions)
    # Abstract tensors only: verify checkpoint compatibility without allocating weights.
    abstract = jax.eval_shape(lambda key: nnx.state(cfg.model.create(key)), jax.random.key(0))
    expected = flax.traverse_util.flatten_dict(abstract.to_pure_dict(), sep="/")
    for key, value in expected.items():
        if tuple(value.shape) != checkpoint_shapes.get(key):
            raise ValueError(f"Parameter mismatch: {key}: model={value.shape}, checkpoint={checkpoint_shapes.get(key)}")
    print(json.dumps({"status": "CPU_PREFLIGHT_PASSED", "profile": profile,
                      "parameter_shapes_checked": len(expected), "camera_count": len(settings["camera_paths"]),
                      "action_steps": len(actions), "normalization": str(stats),
                      "gpu_inference_tested": False, "weights_loaded": False,
                      "official_baseline": False}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="base-experimental")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-yaml", type=Path, default=ROOT / "config/pi05-server.yaml")
    parser.add_argument("--gpu", default="1", help="Physical GPU index; default 1, same GPU as HoloBrain")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1", help="Local simulation only")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--check", action="store_true", help="CPU-only preflight; no weights or network listener")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Explicitly allow sharing an occupied GPU")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.profile in ("hbot-lora", "hbot-full") and args.checkpoint is None:
        parser.error(f"{args.profile} requires --checkpoint pointing to a directory containing params/ and assets/")
    checkpoint = (args.checkpoint or ROOT / "models/pi05_base").expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error("Checkpoint not found: " + str(checkpoint))
    os.environ["HBOT_PI05_CHECKPOINT"] = str(checkpoint)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.check else args.gpu
    os.environ["JAX_PLATFORMS"] = "cpu" if args.check else "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not args.check:
        check_resources(args.gpu, args.host, args.port, args.allow_busy_gpu)
    # No GPU/model imports above this point. Arrow order is important on this host.
    import pyarrow.parquet  # noqa: F401
    import yaml
    from pi05_server_config import get_config, make_adapter

    tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", ROOT / "models")) / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise ValueError(f"Missing local tokenizer; refusing an implicit download: {tokenizer}")
    settings = yaml.safe_load(args.model_yaml.read_text())[args.profile]
    cfg = get_config(settings["config_name"])
    shapes, stats = checkpoint_metadata(checkpoint, cfg.data.assets.asset_id, args.profile)
    adapter = make_adapter(args.profile, settings)
    if args.check:
        cpu_check(cfg, adapter, args.profile, settings, shapes, stats)
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.profile == "base-experimental":
        logging.warning("EXPERIMENTAL BASE SMOKE TEST: approximate scaled velocity integration; "
                        "not a faithful DROID controller, not an official H-Bot baseline.")
    from openpi.policies.policy_config import create_trained_policy
    from robo_orchard_server.policy.pi05.policy import Pi05Policy
    from robo_orchard_server.server import PolicyWebsocketServer
    print(f"Loading {checkpoint}; profile={args.profile}; physical GPU={args.gpu}; port={args.port}", flush=True)
    pipeline = create_trained_policy(cfg, checkpoint)
    shared = Pi05Policy(pipeline, adapter, steps=settings["action_steps"])
    server = PolicyWebsocketServer(policy_factory=shared.new_session, host=args.host,
                                   port=args.port, logging_tag="pi05-" + args.profile)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        print("PI05 SERVER ERROR: " + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)

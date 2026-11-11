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

ROOT = Path(os.environ.get("HBOT_HOME", str(Path(__file__).resolve().parent / "runtime")))
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
                         "Select an idle GPU or release your own workload first.")
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

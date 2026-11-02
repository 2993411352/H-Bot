"""H-Bot data/config adapter for the pinned official OpenPI JAX trainer.

Training source: six local LeRobot v2.1 datasets, all 600 demonstrations.
Sampling: uniform over frames (not task-balanced); no held-out split yet.
Raw labels are absolute 7-joint targets + absolute gripper. Train on joint
deltas relative to the observation at the start of each 15-action chunk.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
# On this Ubuntu 20.04 environment, loading Arrow after OpenPI's native-library
# imports crashes in the Arrow import. Preload it before JAX/transformers.
# Keep this order in both the parent and spawned data-loader workers.
import pyarrow.parquet as pq  # noqa: F401
import flax.nnx as nnx
from openpi import transforms as T
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training import config as C
from openpi.training import optimizer, weight_loaders

ROOT = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot")).resolve()
DATA_ROOT = ROOT / "datasets/InstructMove_hbot/lerobot_unpacked"
CONFIG_NAME = "pi05_hbot_lora"
REPO_ID = "hbot/instructmove_6tasks"
ASSET_ID = "hbot_panda_joint_delta_h15"
ACTION_HORIZON = 15
FREEZE_STRATEGY = "lora_plus_projections_frozen_vision_v1"
TASKS = ("joint_direction", "open_drawer", "pick_color", "place_a2b", "spatial_pick", "stack_cubes")
DATASETS = tuple(f"panda_droid-{name}-20261013" for name in TASKS)
CAMERA_MAP = {
    "base_0_rgb": "observation.images.ext1_camera",
    "left_wrist_0_rgb": "observation.images.wrist_camera",
    # This model slot carries the second EXTERNAL camera, not a second wrist.
    "right_wrist_0_rgb": "observation.images.ext2_camera",
}
DELTA_MASK = (True,) * 7 + (False,)
NORM_DIR = ROOT / "config/openpi_assets" / CONFIG_NAME / ASSET_ID


@dataclasses.dataclass(frozen=True)
class HBotInputs(T.DataTransformFn):
    def __call__(self, data):
        images = {}
        for slot, key in CAMERA_MAP.items():
            image = np.asarray(data[key])
            if image.ndim != 3:
                raise ValueError(f"{key}: expected one RGB frame, got {image.shape}")
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.shape[-1] != 3:
                raise ValueError(f"{key}: not RGB")
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            images[slot] = image
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 8D joint+gripper state")
        out = {"state": state.copy(), "image": images,
               "image_mask": {key: np.True_ for key in images}, "prompt": data["prompt"]}
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 8 or not np.isfinite(actions).all():
                raise ValueError("Expected finite [horizon, 8] absolute action targets")
            out["actions"] = actions.copy()
        return out


@dataclasses.dataclass(frozen=True)
class HBotOutputs(T.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclasses.dataclass(frozen=True)
class HBotDataConfig(C.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        data_transforms = T.Group(inputs=[HBotInputs()], outputs=[HBotOutputs()]).push(
            inputs=[T.DeltaActions(DELTA_MASK)], outputs=[T.AbsoluteActions(DELTA_MASK)])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=C.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def build_config(exp_name="unstarted", batch_size=1, num_train_steps=20000,
                 num_workers=2, resume=False, fsdp_devices=1, log_interval=10,
                 lr_decay_steps=20000, save_interval=1000, keep_period=5000):
    if lr_decay_steps <= 1000 or save_interval < 1 or keep_period < 1 or keep_period % save_interval:
        raise ValueError("Decay must exceed warmup; keep_period must be a multiple of save_interval")
    model = Pi0Config(pi05=True, action_horizon=ACTION_HORIZON,
                      paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    return C.TrainConfig(
        name=CONFIG_NAME, project_name="hbot-pi05", exp_name=exp_name, model=model,
        # Keep pretrained vision features fixed as well as the base LLM weights.
        # OpenPI casts all frozen parameters to bf16 and excludes their optimizer
        # state/gradients. LoRA and small action/time projections stay trainable.
        freeze_filter=nnx.Any(model.get_freeze_filter(), nnx_utils.PathRegex(r"PaliGemma/img/.*")),
        ema_decay=None,
        data=HBotDataConfig(repo_id=REPO_ID, assets=C.AssetsConfig(asset_id=ASSET_ID)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT / "models/pi05_base/params")),
        assets_base_dir=str(ROOT / "config/openpi_assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        batch_size=batch_size, num_workers=num_workers, num_train_steps=num_train_steps,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5,
                                                 decay_steps=lr_decay_steps, decay_lr=2.5e-6),
        log_interval=log_interval, save_interval=save_interval, keep_period=keep_period,
        wandb_enabled=False, resume=resume, overwrite=False, fsdp_devices=fsdp_devices,
        policy_metadata={"action_space": "absolute_joint_position", "action_dim": 8,
                         "camera_map": CAMERA_MAP, "training_joint_deltas": True,
                         "freeze_strategy": FREEZE_STRATEGY},
    )


def create_hbot_dataset(action_horizon=ACTION_HORIZON):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import ConcatDataset
    from openpi.training.data_loader import TransformedDataset
    parts = []
    for name in DATASETS:
        root = DATA_ROOT / name
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 30 or info["features"]["action"]["shape"] != [8]:
            raise ValueError(f"Unexpected dataset schema: {name}")
        dataset = LeRobotDataset(
            name, root=root, video_backend="pyav", download_videos=False,
            delta_timestamps={"action": [t / info["fps"] for t in range(action_horizon)]},
        )
        # Resolve task IDs separately: task_index=0 means different things in each ZIP.
        parts.append(TransformedDataset(dataset, [T.PromptFromLeRobotTask(dataset.meta.tasks)]))
    return ConcatDataset(parts)


def install_data_loader_hook():
    """Adapt only our repo ID in this process; leave all other OpenPI configs alone."""
    from openpi.training import data_loader
    if getattr(data_loader.create_torch_dataset, "_hbot_adapter", False):
        return
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if data_config.repo_id == REPO_ID:
            return create_hbot_dataset(action_horizon)
        return original(data_config, action_horizon, model_config)

    create._hbot_adapter = True
    data_loader.create_torch_dataset = create


def check_prepared():
    import hashlib
    report = json.loads((NORM_DIR / "preparation.json").read_text())
    if report["action_horizon"] != ACTION_HORIZON or report["datasets"] != list(DATASETS):
        raise ValueError("Normalization metadata does not match training configuration")
    if not (NORM_DIR / "norm_stats.json").is_file():
        raise FileNotFoundError("Run prepare-pi05-hbot.py first")
    if hashlib.sha256((NORM_DIR / "norm_stats.json").read_bytes()).hexdigest() != report["norm_stats_sha256"]:
        raise ValueError("Normalization changed after preparation; revalidate before training")
    return report


def validate_cpu(config, report):
    """Real video/data checks and abstract parameter-shape checks; no weight allocation."""
    import hashlib
    import math
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    from openpi.training import data_loader

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Preflight must use CPU only")
    install_data_loader_hook()
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_hbot_dataset()
    assert len(dataset) == report["frames"] == 156867
    transform = T.compose([*data_config.data_transforms.inputs,
                           T.Normalize(data_config.norm_stats, use_quantiles=True),
                           *data_config.model_transforms.inputs])
    inverse = T.compose([T.Unnormalize(data_config.norm_stats, use_quantiles=True),
                         *data_config.data_transforms.outputs])
    checks = []
    for name, part in zip(DATASETS, dataset.datasets, strict=True):
        for index in (0, len(part) - 1):
            raw = part[index]
            expected = np.asarray(raw["action"]).copy()
            out = transform(raw)
            assert out["state"].shape == (32,)
            assert out["actions"].shape == (ACTION_HORIZON, 32)
            assert out["tokenized_prompt"].shape == (200,)
            assert all(image.shape == (224, 224, 3) for image in out["image"].values())
            assert all(out["image_mask"].values())
            restored = inverse({"actions": out["actions"].copy(), "state": out["state"].copy()})
            np.testing.assert_allclose(restored["actions"], expected, rtol=1e-4, atol=1e-5)
            if index == len(part) - 1:
                np.testing.assert_allclose(expected, np.repeat(expected[:1], ACTION_HORIZON, axis=0))
            checks.append({"dataset": name, "index": index, "prompt": raw["prompt"]})
        print("DATA/CAMERAS/ROUNDTRIP OK", name, flush=True)
    # Exercise the same spawned-worker data pipeline as training, with CPU arrays.
    transformed = data_loader.transform_dataset(dataset, data_config)
    loader = data_loader.TorchDataLoader(transformed, local_batch_size=config.batch_size,
                                        num_workers=config.num_workers, num_batches=1, framework="pytorch")
    for batch in loader:
        assert tuple(batch["actions"].shape) == (config.batch_size, ACTION_HORIZON, 32)
    del loader
    print("SPAWNED DATA LOADER OK", flush=True)
    shape = jax.eval_shape(lambda key: nnx.state(config.model.create(key)), jax.random.key(42))
    flat = flax.traverse_util.flatten_dict(shape.to_pure_dict(), sep="/")
    trainable = flax.traverse_util.flatten_dict(shape.filter(config.trainable_filter).to_pure_dict(), sep="/")
    metadata = json.loads((ROOT / "models/pi05_base/params/_METADATA").read_text())["tree_metadata"]
    checkpoint_shapes = {}
    for item in metadata.values():
        keys = [part["key"] for part in item["key_metadata"]]
        assert keys[0] == "params"
        checkpoint_shapes["/".join(keys[1:])] = tuple(item["value_metadata"]["write_shape"])
    for key, value in flat.items():
        if "lora" not in key:
            assert tuple(value.shape) == checkpoint_shapes[key], (key, value.shape, checkpoint_shapes.get(key))
        if "llm" in key and "lora" not in key:
            assert key not in trainable, "Non-LoRA LLM parameter is unexpectedly trainable: " + key
        if key.startswith("PaliGemma/img/"):
            assert key not in trainable, "Vision parameter is unexpectedly trainable: " + key
    assert any("lora" in key for key in trainable)
    assert sum(math.prod(v.shape) for v in trainable.values()) == 52153376
    assert set(k.split("/")[0] for k in trainable if "lora" not in k) == {
        "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
    summary = {"status": "CPU_PREFLIGHT_PASSED_NO_GPU_TRAINING_IN_THIS_CHECK",
               "episodes": report["episodes"], "frames": len(dataset), "data_samples_checked": checks,
               "action_horizon": ACTION_HORIZON, "action_dim_padded": 32, "batch_size": config.batch_size,
               "trainable_parameters": sum(math.prod(v.shape) for v in trainable.values()),
               "lora_parameters": sum(math.prod(v.shape) for k, v in trainable.items() if "lora" in k),
               "total_parameters": sum(math.prod(v.shape) for v in flat.values()),
               "checkpoint_shape_checks": sum("lora" not in key for key in flat),
               "freeze_strategy": FREEZE_STRATEGY,
               "note": "Base LLM and entire vision encoder frozen; LoRA plus action/time projections trained.",
               "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = ROOT / "logs/pi05_hbot_preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "data_samples_checked"}, indent=2), flush=True)

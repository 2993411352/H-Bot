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

ROOT = Path(os.environ.get("HBOT_HOME", str(Path(__file__).resolve().parent / "runtime"))).resolve()
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

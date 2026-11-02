"""Portable full-parameter configuration; same validated H-Bot data semantics.

All 600 demonstrations, uniform-frame sampling; no held-out validation split.
No LoRA modules, no frozen vision/LLM/expert parameters, and no EMA copy.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # noqa: F401 -- preload before other native libraries
import flax.nnx as nnx
from openpi import transforms as T
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as C
from openpi.training import optimizer, weight_loaders

PACKAGE = Path(os.environ.get("HBOT_PACKAGE_ROOT", Path(__file__).resolve().parents[2]))
INPUT = Path(os.environ.get("HBOT_INPUT_ROOT", "/input"))
OUTPUT = Path(os.environ.get("HBOT_OUTPUT_ROOT", "/output"))
DATA_ROOT = INPUT / "datasets/InstructMove_hbot/lerobot_unpacked"
BASE_PARAMS = INPUT / "models/pi05_base/params"
CONFIG_NAME = "pi05_hbot_full"
REPO_ID = "hbot/instructmove_6tasks"
ASSET_ID = "hbot_panda_joint_delta_h15"
ACTION_HORIZON = 15
TASKS = ("joint_direction", "open_drawer", "pick_color", "place_a2b", "spatial_pick", "stack_cubes")
DATASETS = tuple(f"panda_droid-{name}-20261013" for name in TASKS)
CAMERA_MAP = {
    "base_0_rgb": "observation.images.ext1_camera",
    "left_wrist_0_rgb": "observation.images.wrist_camera",
    "right_wrist_0_rgb": "observation.images.ext2_camera",
}
DELTA_MASK = (True,) * 7 + (False,)
NORM_DIR = PACKAGE / "assets" / ASSET_ID
NORM_SHA256 = "48fc140c4306c12e57f991a1b75df7c1ae3071a78bcda69e983950c2ed389c8d"


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
            if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
                raise ValueError("Expected finite [15, 8] absolute action targets")
            out["actions"] = actions.copy()
        return out


@dataclasses.dataclass(frozen=True)
class HBotOutputs(T.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclasses.dataclass(frozen=True)
class HBotDataConfig(C.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        transforms = T.Group(inputs=[HBotInputs()], outputs=[HBotOutputs()]).push(
            inputs=[T.DeltaActions(DELTA_MASK)], outputs=[T.AbsoluteActions(DELTA_MASK)])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=transforms, model_transforms=C.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def build_config(args):
    model = Pi0Config(pi05=True, action_horizon=ACTION_HORIZON,
                      paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    return C.TrainConfig(
        name=CONFIG_NAME, project_name="hbot-pi05-full", exp_name=args.exp_name,
        model=model, freeze_filter=nnx.Nothing, ema_decay=None,
        data=HBotDataConfig(repo_id=REPO_ID, assets=C.AssetsConfig(
            assets_dir=str(PACKAGE / "assets"), asset_id=ASSET_ID)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        assets_base_dir=str(PACKAGE / "assets"), checkpoint_base_dir=str(OUTPUT / "checkpoints"),
        batch_size=args.batch_size, num_workers=args.num_workers, num_train_steps=args.num_train_steps,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr, decay_steps=args.lr_decay_steps, decay_lr=args.end_lr),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        log_interval=args.log_interval, save_interval=args.save_interval, keep_period=args.keep_period,
        wandb_enabled=False, resume=args.resume, overwrite=False, fsdp_devices=1, seed=args.seed,
        policy_metadata={"action_space": "absolute_joint_position", "action_dim": 8,
            "camera_map": CAMERA_MAP, "training_joint_deltas": True, "freeze_strategy": "full_parameters_no_lora"},
    )


def create_hbot_dataset(action_horizon=ACTION_HORIZON):
    if action_horizon != ACTION_HORIZON:
        raise ValueError("Changing horizon requires recomputing normalization statistics")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import ConcatDataset
    from openpi.training.data_loader import TransformedDataset
    parts = []
    for name in DATASETS:
        root = DATA_ROOT / name
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 30 or info["features"]["action"]["shape"] != [8]:
            raise ValueError(f"Unexpected dataset schema: {name}")
        dataset = LeRobotDataset(name, root=root, video_backend="pyav", download_videos=False,
            delta_timestamps={"action": [t / info["fps"] for t in range(action_horizon)]})
        parts.append(TransformedDataset(dataset, [T.PromptFromLeRobotTask(dataset.meta.tasks)]))
    return ConcatDataset(parts)


def install_data_loader_hook():
    from openpi.training import data_loader
    if getattr(data_loader.create_torch_dataset, "_hbot_full_adapter", False):
        return
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if data_config.repo_id == REPO_ID:
            return create_hbot_dataset(action_horizon)
        return original(data_config, action_horizon, model_config)

    create._hbot_full_adapter = True
    data_loader.create_torch_dataset = create


def check_norm():
    report = json.loads((NORM_DIR / "preparation.json").read_text())
    if report["action_horizon"] != ACTION_HORIZON or report["datasets"] != list(DATASETS):
        raise ValueError("Normalization metadata mismatch")
    digest = hashlib.sha256((NORM_DIR / "norm_stats.json").read_bytes()).hexdigest()
    if digest != NORM_SHA256 or digest != report["norm_stats_sha256"]:
        raise ValueError("Normalization checksum mismatch")
    return report

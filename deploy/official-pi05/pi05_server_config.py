"""Local inference adapters. Does not modify OpenPI or RoboOrchard source.

base smoke: DROID velocity output -> APPROXIMATE, bounded joint targets.
H-Bot LoRA/full: reuse the actual training inputs and absolute-action transform.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robo_orchard_server.policy.pi05.adapter import Pi05Adapter

ROOT = Path(os.environ.get("HBOT_HOME", str(Path(__file__).resolve().parent / "runtime")))
ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


@dataclasses.dataclass(frozen=True)
class SplitDroidState:
    """The canonical adapter emits one 8D vector; OpenPI DROID expects two."""
    def __call__(self, data):
        state = np.asarray(data["hbot_state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite [7 arm joints, 1 normalized gripper]")
        return {**data, "observation/joint_position": state[:7],
                "observation/gripper_position": state[7:]}


def get_config(name):
    # This host needs Arrow loaded before JAX/transformers native libraries.
    import pyarrow.parquet  # noqa: F401
    from openpi import transforms as T
    from openpi.policies import droid_policy
    from openpi.training import config as C

    checkpoint = Path(os.environ["HBOT_PI05_CHECKPOINT"])
    if name == "pi05_hbot_base_smoke":
        source = C.get_config("pi05_droid")
        data = dataclasses.replace(
            source.data,
            assets=C.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id="droid"),
            data_transforms=lambda model: T.Group(
                inputs=[SplitDroidState(), droid_policy.DroidInputs(model_type=model.model_type)],
                outputs=[droid_policy.DroidOutputs()]),
        )
        return dataclasses.replace(source, name=name, data=data,
                                   assets_base_dir=str(ROOT / "config/openpi_assets"))
    if name in ("pi05_hbot_lora", "pi05_hbot_full"):
        from pi05_hbot import build_config
        cfg = build_config(exp_name="inference")
        if cfg.name != "pi05_hbot_lora" or cfg.model.action_horizon != 15:
            raise ValueError("H-Bot training config changed; revalidate serving")
        if name == "pi05_hbot_full":
            # Full finetuning changes the parameter tree, not camera/action semantics.
            # Keep pi05, horizon=15, state tokenization and all H-Bot transforms.
            model = dataclasses.replace(cfg.model, paligemma_variant="gemma_2b",
                                        action_expert_variant="gemma_300m")
            cfg = dataclasses.replace(
                cfg, name=name, model=model, freeze_filter=model.get_freeze_filter(),
                policy_metadata={**cfg.policy_metadata, "freeze_strategy": "full_finetune"})
        data = dataclasses.replace(cfg.data, assets=C.AssetsConfig(
            assets_dir=str(checkpoint / "assets"), asset_id=cfg.data.assets.asset_id))
        return dataclasses.replace(cfg, data=data)
    raise ValueError("Unsupported inference config: " + name)


class PandaAdapter(Pi05Adapter):
    def layout(self, obs):
        specs = super().layout(obs)
        if (obs["action_layout"]["embodiment_type"] not in ("panda_droid", "franka_panda")
                or len(specs) != 1 or tuple(specs[0]["arm_joint_names"]) != ARM_NAMES
                or specs[0]["gripper_policy_representation"] != "first_joint"):
            raise ValueError("This configuration requires a single Panda with the expected joint order")
        return specs


class ExperimentalVelocityAdapter(PandaAdapter):
    """Approximation for simulation only; not a calibrated DROID controller.

Repeat each 15 Hz command twice at 30 Hz; integrate from observed joints at
each inference. Clip command, chunk displacement, gripper and URDF limits.
No persistent target state is shared between policy sessions or resets.
"""
    def __init__(self, cfg):
        super().__init__(cfg["camera_paths"], cfg["state_path"], cfg["prompt_path"])
        self.cfg = cfg
        self.sim_hz = float(cfg["sim_hz"])
        model_hz = float(cfg["model_hz"])
        numeric = [self.sim_hz, model_hz, cfg["velocity_gain_rad_s"], cfg["command_clip"],
                   cfg["max_chunk_displacement_rad"], cfg["joint_limit_margin_rad"]]
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError("Frequencies and experimental limits must be finite and positive")
        ratio = self.sim_hz / model_hz
        if not ratio.is_integer() or ratio < 1:
            raise ValueError("Simulation rate must be a positive integer multiple of model rate")
        self.repeat = int(ratio)
        joints = {joint.attrib["name"]: joint for joint in ET.parse(ROOT / cfg["urdf"]).getroot().findall("joint")}
        self.lower = np.array([float(joints[n].find("limit").attrib["lower"]) for n in ARM_NAMES])
        self.upper = np.array([float(joints[n].find("limit").attrib["upper"]) for n in ARM_NAMES])
        self.lower += cfg["joint_limit_margin_rad"]
        self.upper -= cfg["joint_limit_margin_rad"]
        if not np.all(self.lower < self.upper):
            raise ValueError("Invalid URDF joint limits/margin")

    def build_action_sequence(self, actions, obs, steps=None):
        self.layout(obs)
        model_input = self.build_model_input(obs)
        state = np.asarray(model_input[self.state_path[0]])
        if len(self.state_path) != 1 or state.shape != (8,):
            raise ValueError("Experimental adapter requires a flat 8D state mapping")
        if np.any(state[:7] < self.lower) or np.any(state[:7] > self.upper):
            raise ValueError("Observed joints are outside the guarded URDF bounds")
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 8 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected finite [horizon, 8] DROID commands")
        values = np.repeat(values, self.repeat, axis=0)
        if steps is not None:
            if not 1 <= steps <= len(values):
                raise ValueError("Execution steps exceed the resampled chunk")
            values = values[:steps]
        velocity = np.clip(values[:, :7], -self.cfg["command_clip"], self.cfg["command_clip"])
        velocity *= self.cfg["velocity_gain_rad_s"]
        displacement = np.cumsum(velocity / self.sim_hz, axis=0)
        bound = self.cfg["max_chunk_displacement_rad"]
        targets = np.clip(state[:7] + np.clip(displacement, -bound, bound), self.lower, self.upper)
        # Match the DROID example's binary gripper convention (0=open, 1=closed).
        gripper = (np.clip(values[:, 7:], 0, 1) > 0.5).astype(np.float32)
        absolute = np.concatenate([targets, gripper], axis=1)
        return super().build_action_sequence(absolute, obs)


def make_adapter(profile, cfg):
    if profile == "base-experimental":
        return ExperimentalVelocityAdapter(cfg)
    return PandaAdapter(cfg["camera_paths"], cfg["state_path"], cfg["prompt_path"])


def example_observation():
    """Synthetic canonical protocol observation; no simulator or GPU."""
    spec = {"arm_joint_names": list(ARM_NAMES), "gripper_joint_names": ["finger_a", "finger_b"],
            "gripper_policy_representation": "first_joint", "gripper_decode_coupling": "symmetric",
            "gripper_policy_scale": 1.25}
    return {"instruction": "pick up the red cube",
            "cameras": {c: {"rgb": np.full((1, 24, 32, 3), i * 40, np.uint8)}
                        for i, c in enumerate(("ext1_camera", "wrist_camera", "ext2_camera"), 1)},
            "manipulators": {"single_arm": {"joint_position": np.array([[0, -.6, 0, -2, 0, 1.8, 0]], np.float32),
                                             "gripper_position": np.array([[.4, .4]], np.float32)}},
            "action_layout": {"embodiment_type": "panda_droid", "schema_version": "1",
                              "manipulator_order": ["single_arm"], "manipulators": {"single_arm": spec}}}

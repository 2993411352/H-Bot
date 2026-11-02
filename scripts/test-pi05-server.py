"""CPU-only tests for experimental simulation actions and launch safeguards."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import yaml
from pi05_server_config import (ExperimentalVelocityAdapter, PandaAdapter, SplitDroidState,
                               example_observation, get_config, make_adapter)

ROOT = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot"))
spec = importlib.util.spec_from_file_location("pi05_launcher", Path(__file__).with_name("pi05-server.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        config_file = Path(os.environ.get("HBOT_PI05_TEST_CONFIG", ROOT / "config/pi05-server.yaml"))
        self.settings = yaml.safe_load(config_file.read_text())
        self.cfg = self.settings["base-experimental"]
        self.adapter = ExperimentalVelocityAdapter(self.cfg)
        self.obs = example_observation()
        self.state = self.obs["manipulators"]["single_arm"]["joint_position"][0]

    def test_camera_and_state_mapping(self):
        data = self.adapter.build_model_input(self.obs)
        self.assertEqual(data["hbot_state"].shape, (8,))
        self.assertAlmostEqual(data["hbot_state"][-1], .5)
        self.assertEqual(len(self.cfg["camera_paths"]), 2)
        split = SplitDroidState()(data)
        np.testing.assert_array_equal(split["observation/joint_position"], self.state)
        np.testing.assert_array_equal(split["observation/gripper_position"], [.5])

    def test_velocity_integration_and_resampling(self):
        values = np.full((15, 8), .5, np.float32)
        actions = self.adapter.build_action_sequence(values, self.obs, steps=15)
        for index, action in enumerate(actions):
            np.testing.assert_allclose(action["values"][0, :7], self.state + (index + 1) * .5 * .25 / 30,
                                       atol=2e-7)
        self.assertEqual(len(actions), 15)

    def test_resampling_repeats_each_command(self):
        values = np.zeros((2, 8), np.float32)
        values[0, :7] = .5
        values[1, :7] = -.5
        actions = self.adapter.build_action_sequence(values, self.obs)
        offsets = [.5 * .25 / 30, .5 * .25 * 2 / 30, .5 * .25 / 30, 0]
        for action, offset in zip(actions, offsets):
            np.testing.assert_allclose(action["values"][0, :7], self.state + offset, atol=2e-7)

    def test_zero_velocity_and_binary_gripper(self):
        values = np.zeros((15, 8), np.float32)
        values[:, 7] = .8
        actions = self.adapter.build_action_sequence(values, self.obs, 15)
        for action in actions:
            np.testing.assert_array_equal(action["values"][0, :7], self.state)
            np.testing.assert_allclose(action["values"][0, 7:], [.8, .8])

    def test_caps_joint_limits_and_chunk_displacement(self):
        values = np.full((100, 8), 1000, np.float32)
        actions = self.adapter.build_action_sequence(values, self.obs)
        targets = np.array([a["values"][0, :7] for a in actions])
        self.assertTrue(np.all(targets <= self.adapter.upper + 1e-6))
        self.assertTrue(np.all(targets >= self.adapter.lower - 1e-6))
        self.assertLessEqual(np.max(np.abs(targets - self.state)), .150001)
        np.testing.assert_allclose(targets[0] - self.state, np.full(7, .25 / 30), atol=2e-7)

    def test_urdf_limit_saturates(self):
        self.obs["manipulators"]["single_arm"]["joint_position"][0, 0] = self.adapter.upper[0] - .0001
        actions = self.adapter.build_action_sequence(np.ones((15, 8)), self.obs, 15)
        self.assertAlmostEqual(float(actions[-1]["values"][0, 0]), float(self.adapter.upper[0]), places=5)

    def test_invalid_inputs_rejected(self):
        for bad in (np.full((15, 8), np.nan), np.zeros((15, 7)), np.zeros((0, 8))):
            with self.assertRaises(ValueError):
                self.adapter.build_action_sequence(bad, self.obs)
        self.obs["action_layout"]["manipulators"]["single_arm"]["arm_joint_names"].reverse()
        with self.assertRaises(ValueError):
            self.adapter.build_action_sequence(np.zeros((15, 8)), self.obs)

    def test_adapter_has_no_cross_session_target_accumulation(self):
        values = np.ones((15, 8))
        first = self.adapter.build_action_sequence(values, self.obs, 15)
        second = self.adapter.build_action_sequence(values, copy.deepcopy(self.obs), 15)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a["values"], b["values"])

    def test_lora_adapter_does_not_integrate_absolute_targets(self):
        cfg = self.settings["hbot-lora"]
        adapter = PandaAdapter(cfg["camera_paths"], cfg["state_path"], cfg["prompt_path"])
        target = np.array([.1, -.7, .1, -1.9, .2, 1.7, .1, .5])
        actions = adapter.build_action_sequence(np.repeat(target[None], 15, axis=0), self.obs, 15)
        np.testing.assert_allclose(actions[-1]["values"][0, :7], target[:7])
        self.assertEqual(len(cfg["camera_paths"]), 3)

    def test_full_and_lora_use_identical_hbot_mapping_and_actions(self):
        lora = self.settings["hbot-lora"]
        full = self.settings["hbot-full"]
        for key in ("camera_paths", "state_path", "prompt_path", "action_steps"):
            self.assertEqual(full[key], lora[key])
        adapter = make_adapter("hbot-full", full)
        self.assertIs(type(adapter), PandaAdapter)
        data = adapter.build_model_input(self.obs)
        self.assertEqual(data["observation.state"].shape, (8,))
        for camera in ("ext1_camera", "wrist_camera", "ext2_camera"):
            self.assertIn("observation.images." + camera, data)
        target = np.array([.1, -.7, .1, -1.9, .2, 1.7, .1, .5])
        actions = adapter.build_action_sequence(np.repeat(target[None], 15, axis=0), self.obs, 15)
        self.assertEqual(len(actions), 15)
        for action in actions:
            np.testing.assert_allclose(action["values"][0, :7], target[:7])

    def test_full_config_uses_non_lora_model_and_checkpoint_assets(self):
        checkpoint = "/checkpoint-fixture"
        with patch.dict(os.environ, HBOT_PI05_CHECKPOINT=checkpoint):
            full = get_config("pi05_hbot_full")
            lora = get_config("pi05_hbot_lora")
        self.assertEqual(full.name, "pi05_hbot_full")
        self.assertEqual(full.model.paligemma_variant, "gemma_2b")
        self.assertEqual(full.model.action_expert_variant, "gemma_300m")
        self.assertEqual(lora.model.paligemma_variant, "gemma_2b_lora")
        self.assertEqual(lora.model.action_expert_variant, "gemma_300m_lora")
        for key in ("pi05", "action_horizon", "action_dim", "max_token_len", "discrete_state_input"):
            self.assertEqual(getattr(full.model, key), getattr(lora.model, key))
        self.assertEqual(full.model.action_horizon, 15)
        self.assertEqual(full.data, lora.data)
        self.assertEqual(full.data.assets.assets_dir, checkpoint + "/assets")
        self.assertEqual(full.data.assets.asset_id, "hbot_panda_joint_delta_h15")

    def test_unsupported_frequency_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentalVelocityAdapter(dict(self.cfg, sim_hz=20))

    def test_busy_gpu_refuses_before_any_model_load(self):
        with patch.object(launcher.subprocess, "check_output", side_effect=["GPU-test\n", "GPU-test, 123, 2854 MiB\n"]):
            with self.assertRaisesRegex(ValueError, "already occupied"):
                launcher.check_resources("1", "127.0.0.1", 8766, False)

    def test_occupied_port_refused(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            with patch.object(launcher.subprocess, "check_output", side_effect=["GPU-test\n", ""]):
                with self.assertRaises(OSError):
                    launcher.check_resources("1", "127.0.0.1", occupied.getsockname()[1], False)


class CheckpointProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hbot-serving-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "params").mkdir()
        stats = self.root / "assets/hbot_panda_joint_delta_h15/norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text(json.dumps({"norm_stats": {
            name: {key: [0] * 8 for key in ("mean", "std", "q01", "q99")}
            for name in ("state", "actions")}}))

    def write_metadata(self, lora=False, wrapped=True):
        paths = [("params", "action_out_proj", "kernel", "value")]
        if lora:
            paths.append(("params", "PaliGemma", "llm", "lora_a", "value"))
        if not wrapped:
            paths = [path[:-1] for path in paths]
        tree = {str(i): {"key_metadata": [{"key": key} for key in path],
                         "value_metadata": {"write_shape": [2, 2]}}
                for i, path in enumerate(paths)}
        (self.root / "params/_METADATA").write_text(json.dumps({"tree_metadata": tree}))

    def test_full_accepted_lora_rejected_with_actionable_hint(self):
        self.write_metadata()
        shapes, _ = launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-full")
        self.assertEqual(shapes, {"action_out_proj/kernel": (2, 2)})
        with self.assertRaisesRegex(ValueError, "use --profile hbot-full"):
            launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-lora")

    def test_pure_keys_and_nnx_wrapped_keys_have_same_shapes(self):
        self.write_metadata(lora=True)
        wrapped, _ = launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-lora")
        self.write_metadata(lora=True, wrapped=False)
        pure, _ = launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-lora")
        self.assertEqual(pure, wrapped)

    def test_lora_accepted_full_rejected(self):
        self.write_metadata(lora=True)
        launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-lora")
        with self.assertRaisesRegex(ValueError, "use --profile hbot-lora"):
            launcher.checkpoint_metadata(self.root, "hbot_panda_joint_delta_h15", "hbot-full")

    def test_full_requires_explicit_checkpoint_before_resource_checks(self):
        with patch.object(sys, "argv", ["pi05-server.py", "--profile", "hbot-full"]):
            with patch.object(launcher, "check_resources") as resources:
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 2)
                resources.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

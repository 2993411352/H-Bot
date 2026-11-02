"""CPU-only package contracts, also run inside the Docker build."""
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import hbot_full as h
from train_full import parse_args, parameter_audit, committed_steps, require_smoke


class ContractTests(unittest.TestCase):
    def test_full_sft_has_no_frozen_or_lora_parameters(self):
        config = h.build_config(parse_args([]))
        result = parameter_audit(config)
        self.assertEqual(result["trainable_parameters"], result["total_parameters"])
        self.assertGreater(result["total_parameters"], 3_000_000_000)
        self.assertEqual((result["lora_parameters"], result["frozen_parameters"]), (0, 0))
        self.assertFalse(result["ema_enabled"])

    def test_schedule_and_safety(self):
        config = h.build_config(parse_args([]))
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.fsdp_devices, 1)
        self.assertFalse(config.overwrite)
        self.assertEqual(config.num_train_steps, 100000)
        self.assertEqual(dataclasses.asdict(config.lr_schedule)["decay_steps"], 100000)
        self.assertEqual(config.keep_period, 100000)
        self.assertEqual(config.model.action_horizon, 15)
        self.assertEqual(config.model.paligemma_variant, "gemma_2b")
        self.assertEqual(config.model.action_expert_variant, "gemma_300m")

    def test_norm_provenance(self):
        report = h.check_norm()
        self.assertEqual((report["episodes"], report["frames"]), (600, 156867))
        self.assertEqual(report["norm_stats_sha256"], h.NORM_SHA256)

    def test_invalid_arguments(self):
        for argv in (["--batch-size", "0"], ["--exp-name", "../bad"],
                     ["--peak-lr", "nan"], ["--lr-decay-steps", "999"],
                     ["--keep-period", "1001"], ["--num-workers", "-1"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_delta_semantics_and_no_input_mutation(self):
        import numpy as np
        from openpi import transforms as T
        state = np.arange(8, dtype=np.float32) * .1
        actions = np.ones((15, 8), dtype=np.float32)
        raw = {"observation.state": state.copy(), "action": actions.copy(), "prompt": "pick red"}
        for key in h.CAMERA_MAP.values():
            raw[key] = np.zeros((3, 224, 224), dtype=np.uint8)
        out = T.DeltaActions(h.DELTA_MASK)(h.HBotInputs()(raw))
        np.testing.assert_allclose(out["actions"][:, :7], actions[:, :7] - state[None, :7])
        np.testing.assert_equal(out["actions"][:, 7], actions[:, 7])
        np.testing.assert_equal(raw["action"], actions)
        restored = T.AbsoluteActions(h.DELTA_MASK)(out)
        np.testing.assert_allclose(restored["actions"], actions)

    def test_tokenizer_is_bundled(self):
        path = h.PACKAGE / "resources/big_vision/paligemma_tokenizer.model"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6")

    def test_resume_requires_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "1000").mkdir()
            (root / "1000/_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": null}')
            (root / "2000.orbax-checkpoint-tmp").mkdir()
            self.assertEqual(committed_steps(root), [])
            (root / "1000/_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1234}')
            self.assertEqual(committed_steps(root), [1000])

    def test_smoke_gate_requires_completed_matching_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch_16.json"
            profile = {"batch_size": 16, "gpu_kind": "NVIDIA A800 80GB", "trainer_sha256": "new-code"}
            with self.assertRaises(RuntimeError):
                require_smoke(path, profile)
            path.write_text(json.dumps({"status": "SMOKE_RUNNING", "profile": profile}))
            with self.assertRaises(RuntimeError):
                require_smoke(path, profile)
            path.write_text(json.dumps({"status": "SMOKE_PASSED", "profile": profile}))
            require_smoke(path, profile)
            for mismatch in (dict(profile, batch_size=32), dict(profile, trainer_sha256="old-code"),
                             dict(profile, gpu_kind="Different GPU")):
                with self.assertRaises(RuntimeError):
                    require_smoke(path, mismatch)

    def test_smoke_is_bounded_and_not_resume(self):
        with self.assertRaises(SystemExit):
            parse_args(["--smoke"])
        with self.assertRaises(SystemExit):
            parse_args(["--smoke", "--num-train-steps", "20", "--resume"])
        args = parse_args(["--smoke", "--num-train-steps", "20"])
        self.assertEqual(args.num_train_steps, 20)
        self.assertEqual(args.lr_decay_steps, 100000)

    def test_manifest_excludes_sim_assets_and_checkpoints(self):
        manifest = json.loads((h.PACKAGE / "resources/input-manifest.json").read_text())
        self.assertTrue(manifest["files"])
        for entry in manifest["files"]:
            self.assertTrue(entry["path"].startswith(("models/pi05_base/", "datasets/InstructMove_hbot/lerobot_unpacked/")))
            self.assertNotIn("..", Path(entry["path"]).parts)
            # Official Orbax commit_success.txt is intentionally zero bytes.
            self.assertGreaterEqual(entry["size"], 0)


if __name__ == "__main__":
    unittest.main()

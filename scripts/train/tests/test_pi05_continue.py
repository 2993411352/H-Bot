"""CPU-only continuation tests. No real GPU workload, sudo or systemd changes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "supervise-pi05-continue.py"


class ContinueTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("continue_test", SOURCE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.m.guard.MIN_FREE_BYTES = 0
        self.tmp = tempfile.TemporaryDirectory(prefix="hbot-continue-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("models/pi05_base/params", "datasets/InstructMove_hbot/lerobot_unpacked"):
            (self.root / name).mkdir(parents=True)
        files = set(self.m.EXTRA_FILES) | {
            "scripts/run-pi05-lora.sh", "scripts/train/train-pi05-hbot.py",
            "scripts/train/pi05_hbot.py", "openpi/.venv/bin/python",
            "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/norm_stats.json",
            "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/preparation.json"}
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (self.root / "openpi/.venv/bin/python").chmod(0o755)
        config, self.source = self.m.source_paths(self.root)
        self.make_checkpoint(self.source)
        config.parent.mkdir(parents=True)
        original = dict(self.m.original_manifest(self.root), profile=self.m.SOURCE_PROFILE)
        config.write_text(json.dumps(original))
        self.sample = dict(uuid=self.m.guard.GPU_UUID, memory_mib=18,
                           utilization=0, temperature_c=50, power_w=350)

    def make_checkpoint(self, path):
        for name in ("params/_METADATA", "train_state/_METADATA", self.m.NORM,
                     "params/data/0", "train_state/data/0"):
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("{}")
        (path / "_CHECKPOINT_METADATA").write_text(json.dumps(dict(commit_timestamp_nsecs=1)))

    def preflight(self):
        with mock.patch.object(self.m.guard, "query_gpu", return_value=self.sample):
            return self.m.preflight(self.root, "resume")

    def test_prepare_copy_resume_args_and_no_gpu(self):
        with mock.patch.object(self.m.guard, "query_gpu", side_effect=AssertionError("GPU touched")):
            self.m.prepare(self.root)
        target = self.m.guard.paths(self.root)[1] / "39999"
        self.assertEqual(self.m.inventory(target), self.m.inventory(self.source))
        self.assertNotEqual((target / "train_state/data/0").stat().st_ino,
                            (self.source / "train_state/data/0").stat().st_ino)
        self.preflight()
        args = self.m.guard.command(self.root, "resume")
        self.assertEqual(args[-1], "--resume")
        for key, value in (("--num-train-steps", "60000"), ("--lr-decay-steps", "40000"),
                           ("--batch-size", "4"), ("--save-interval", "500")):
            self.assertEqual(args[args.index(key) + 1], value)

    def test_duplicate_prepare_does_not_overwrite(self):
        self.m.prepare(self.root)
        target = self.m.guard.paths(self.root)[1] / "39999/params/data/0"
        target.write_text("preserve")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.m.prepare(self.root)
        self.assertEqual(target.read_text(), "preserve")
        self.assertEqual((self.source / "params/data/0").read_text(), "{}")

    def test_incompatible_source_rejected(self):
        (self.root / "scripts/train/pi05_hbot.py").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Original training manifest"):
            self.m.prepare(self.root)
        self.assertFalse(self.m.guard.paths(self.root)[1].exists())

    def test_missing_optimizer_rejected(self):
        (self.source / "train_state/_METADATA").unlink()
        with self.assertRaisesRegex(RuntimeError, "Incomplete full-state"):
            self.m.prepare(self.root)

    def test_failed_copy_never_publishes_manifest(self):
        with mock.patch.object(self.m.subprocess, "run", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                self.m.prepare(self.root)
        self.assertFalse((self.m.guard.paths(self.root)[0] / "config.json").exists())
        with self.assertRaisesRegex(RuntimeError, "No eligible"):
            self.preflight()

    def test_no_fresh_start_or_incomplete_resume(self):
        with self.assertRaisesRegex(RuntimeError, "never fresh"):
            self.m.preflight(self.root, "start")
        self.m.prepare(self.root)
        target = self.m.guard.paths(self.root)[1] / "39999/train_state/_METADATA"
        target.unlink()
        with self.assertRaisesRegex(RuntimeError, "Incomplete full-state"):
            self.preflight()

    def test_completed_run_not_restarted(self):
        self.m.prepare(self.root)
        self.make_checkpoint(self.m.guard.paths(self.root)[1] / "59999")
        with self.assertRaisesRegex(RuntimeError, "No eligible"):
            self.preflight()

    def test_changed_continuation_sources_rejected(self):
        self.m.prepare(self.root)
        (self.root / "scripts/train/supervise-pi05-continue.py").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "manifest missing or mismatched"):
            self.preflight()

    def test_busy_gpu_rejected(self):
        self.m.prepare(self.root)
        self.sample["memory_mib"] = 3000
        with self.assertRaisesRegex(RuntimeError, "GPU busy"):
            self.preflight()


if __name__ == "__main__":
    unittest.main(verbosity=2)

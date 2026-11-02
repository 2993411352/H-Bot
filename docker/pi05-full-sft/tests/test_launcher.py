"""Exercise shell safety contracts with fake Docker/nvidia-smi, no GPU/container."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hbot-full-test-")
        self.root = Path(self.temp.name)
        binary = self.root / "bin"
        binary.mkdir()
        self.script = Path(__file__).resolve().parents[1] / "scripts/run-sft.sh"
        self.input = self.root / "inputs"
        (self.input / "models/pi05_base/params").mkdir(parents=True)
        (self.input / "datasets/InstructMove_hbot/lerobot_unpacked").mkdir(parents=True)
        self.capture = self.root / "docker.jsonl"
        docker = binary / "docker"
        docker.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
            "with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1:3]==['container','inspect']: sys.exit(int(os.environ.get('EXISTS','1')))\n"
            "if sys.argv[1]=='inspect': print(os.environ.get('OWNER','pi05-full-sft-v2'))\n")
        docker.chmod(0o755)
        smi = binary / "nvidia-smi"
        smi.write_text('#!/usr/bin/env bash\nprintf "%s, %s\\n" "${MEMORY:-81920}" "${USED:-0}"\n')
        smi.chmod(0o755)
        self.env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                        HBOT_INPUT_ROOT=str(self.input), HBOT_OUTPUT_ROOT=str(self.root / "output"),
                        CAPTURE=str(self.capture), HBOT_EXP_NAME="test", HBOT_GPU="0", HBOT_BATCH_SIZE="16",
                        HBOT_STEPS="100000", HBOT_WORKERS="2")

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, mode, **extra):
        return subprocess.run(["bash", str(self.script), mode], env=dict(self.env, **extra),
                              capture_output=True, text=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def test_check_never_exposes_gpu(self):
        result = self.run_script("check", USED="40000", MEMORY="24576")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = self.calls()[-1]
        self.assertNotIn("--gpus", command)
        self.assertIn("JAX_PLATFORMS=cpu", command)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("--check", command)
        self.assertTrue(any("dst=/input,readonly" in arg for arg in command))

    def test_busy_or_small_gpu_never_runs_container(self):
        for extra in ({"USED": "2000"}, {"MEMORY": "24576"}):
            result = self.run_script("train", **extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.calls())

    def test_training_detached_offline_single_gpu_and_not_privileged(self):
        result = self.run_script("train")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = self.calls()[-1]
        self.assertIn("--detach", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--gpus") + 1], "device=0")
        self.assertNotIn("--privileged", command)
        self.assertEqual(command[command.index("--batch-size") + 1], "16")
        self.assertEqual(command[command.index("--num-train-steps") + 1], "100000")
        self.assertEqual(command[command.index("--lr-decay-steps") + 1], "100000")

    def test_smoke_has_twenty_updates_but_long_run_schedule(self):
        result = self.run_script("smoke", HBOT_BATCH_SIZE="32")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = self.calls()[-1]
        self.assertIn("--smoke", command)
        self.assertEqual(command[command.index("--batch-size") + 1], "32")
        self.assertEqual(command[command.index("--num-train-steps") + 1], "20")
        self.assertEqual(command[command.index("--lr-decay-steps") + 1], "100000")

    def test_existing_container_not_removed_or_overwritten(self):
        result = self.run_script("train", EXISTS="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [["container", "inspect", "hbot-pi05-full-test"]])

    def test_resume_is_explicit(self):
        result = self.run_script("resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", self.calls()[-1])

    def test_stop_rejects_wrong_owner(self):
        result = self.run_script("stop", OWNER="other-package")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == "stop" for call in self.calls()))

    def test_invalid_name_or_gpu_rejected(self):
        for extra in ({"HBOT_EXP_NAME": "../bad"}, {"HBOT_GPU": "0,1"}):
            result = self.run_script("train", **extra)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls())

    def test_nested_output_rejected(self):
        result = self.run_script("check", HBOT_OUTPUT_ROOT=str(self.input / "output"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls())


if __name__ == "__main__":
    unittest.main()

"""CPU-only shell-launch safety tests using fake nvidia-smi and fake training."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hbot-pi05-launch-test-")
        self.root = Path(self.temp.name)
        for folder in ("config", "scripts/train", "openpi/.venv/bin", "bin"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "config/hbot-env.sh").write_text("export HBOT_HOME=" + str(self.root) + "\n")
        source_root = Path(__file__).resolve().parents[2]
        source = (source_root / "run-pi05-lora.sh").read_text()
        self.script = self.root / "scripts/run-pi05-lora.sh"
        self.script.write_text(source.replace("/study/wangmengfan/hbot", str(self.root)))
        prep_source = (source_root / "prepare-pi05-data.sh").read_text()
        self.prep_script = self.root / "scripts/prepare-pi05-data.sh"
        self.prep_script.write_text(prep_source.replace("/study/wangmengfan/hbot", str(self.root)))
        fake = self.root / "openpi/.venv/bin/python"
        fake.write_text("#!/usr/bin/env python3\nimport json,os,pathlib,sys\n"
                        "pathlib.Path(os.environ['HBOT_HOME'],'invoked.json').write_text(json.dumps("
                        "{'argv':sys.argv[1:], 'gpu':os.environ['CUDA_VISIBLE_DEVICES'],"
                        "'platform':os.environ['JAX_PLATFORMS'],'offline':os.environ['HF_HUB_OFFLINE']}))\n")
        fake.chmod(0o755)
        smi = self.root / "bin/nvidia-smi"
        smi.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "${MOCK_USED:-18}"\n')
        smi.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"])
        self.env.pop("HBOT_TRAIN_GPU", None)

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, *args, **env):
        return subprocess.run(["bash", str(self.script), *args], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=10)

    def test_busy_gpu_refuses_without_starting(self):
        result = self.run_script("--exp-name", "test", MOCK_USED="2934")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Refusing", result.stderr)
        self.assertFalse((self.root / "invoked.json").exists())

    def test_free_gpu_uses_gpu_zero_and_offline_env(self):
        result = self.run_script("--exp-name", "test", "--num-train-steps", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((self.root / "invoked.json").read_text())
        self.assertEqual((data["gpu"], data["platform"], data["offline"]), ("0", "cuda", "1"))
        self.assertEqual(data["argv"][0], str(self.root / "scripts/train/train-pi05-hbot.py"))
        self.assertEqual(data["argv"][-4:], ["--exp-name", "test", "--num-train-steps", "5"])

    def test_check_always_uses_cpu_even_if_gpu_busy(self):
        result = self.run_script("--check", MOCK_USED="2934")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((self.root / "invoked.json").read_text())
        self.assertEqual((data["gpu"], data["platform"]), ("", "cpu"))

    def test_invalid_gpu_refuses(self):
        result = self.run_script("--exp-name", "test", HBOT_TRAIN_GPU="0,1")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "invoked.json").exists())

    def test_preparation_launcher_points_to_train_subdir_and_cpu(self):
        result = subprocess.run(["bash", str(self.prep_script), "--help"], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((self.root / "invoked.json").read_text())
        self.assertEqual(data["argv"], [str(self.root / "scripts/train/prepare-pi05-hbot.py"), "--help"])
        self.assertEqual((data["gpu"], data["platform"], data["offline"]), ("", "cpu", "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU-only fixtures: no real CUDA, root privileges or systemd changes."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "supervise-pi05-longtrain.py"


class LongTrainTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("longtrain_test_module", SOURCE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.m.SAMPLE_SECONDS = 0.02
        self.m.STOP_GRACE_SECONDS = 0.3
        self.m.MIN_FREE_BYTES = 0
        self.tmp = tempfile.TemporaryDirectory(prefix="hbot-longtrain-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ("scripts/train", "openpi/.venv/bin", "models/pi05_base/params",
                          "datasets/InstructMove_hbot/lerobot_unpacked",
                          "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15"):
            (self.root / directory).mkdir(parents=True)
        for file in ("scripts/train/train-pi05-hbot.py", "scripts/train/pi05_hbot.py",
                     "config/hbot-env.sh", "openpi/.venv/bin/python",
                     "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/norm_stats.json",
                     "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/preparation.json"):
            (self.root / file).touch()
        (self.root / "openpi/.venv/bin/python").chmod(0o755)
        self.launcher("import time; time.sleep(5)")

    def launcher(self, code):
        (self.root / "scripts/run-pi05-lora.sh").write_text(
            "#!/usr/bin/bash\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(code) + ' "$@"\n')

    def sample(self, **values):
        result = dict(time=self.m.now(), uuid=self.m.GPU_UUID, memory_mib=18,
                      utilization=0, temperature_c=40, power_w=280)
        result.update(values)
        return result

    def report(self):
        logdir, _ = self.m.paths(self.root)
        return json.loads(next(logdir.glob("report_*.json")).read_text())

    def test_normal_completion_and_arguments(self):
        _, checkpoint = self.m.paths(self.root)
        self.launcher("import pathlib,json,sys,time; "
                      "time.sleep(0.15); print('Step 39990: loss=0.03'); "
                      f"p=pathlib.Path({str(checkpoint / '39999')!r}); p.mkdir(parents=True); "
                      "(p/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':1})); "
                      f"pathlib.Path({str(self.root / 'args.json')!r}).write_text(json.dumps(sys.argv[1:]))")
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample()):
            self.assertEqual(self.m.run(self.root, "start"), 0)
        report = self.report()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["latest_committed_checkpoint"], 39999)
        self.assertEqual(report["last_logged_step_lower_bound"], 39990)
        args = json.loads((self.root / "args.json").read_text())
        for option, value in (("--batch-size", "4"), ("--num-train-steps", "40000"),
                              ("--lr-decay-steps", "40000"), ("--save-interval", "500"),
                              ("--keep-period", "5000")):
            self.assertEqual(args[args.index(option)+1], value)

    def assert_stop(self, failing_sample):
        unrelated = subprocess.Popen(["sleep", "10"], start_new_session=True)
        try:
            with mock.patch.object(self.m, "query_gpu", side_effect=[self.sample(), failing_sample]):
                self.assertEqual(self.m.run(self.root, "start"), 1)
            report = self.report()
            self.assertEqual(report["status"], "stopped")
            self.assertIsNone(unrelated.poll(), "The unrelated fixture must remain running")
            with self.assertRaises(ProcessLookupError):
                os.killpg(report["child_pid"], 0)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)
        return report

    def test_thermal_stop_only_own_group(self):
        self.assertIn("Temperature cutoff", self.assert_stop(self.sample(temperature_c=90))["stop_reason"])

    def test_changed_power_stops(self):
        self.assertIn("power mismatch", self.assert_stop(self.sample(power_w=350))["stop_reason"])

    def test_telemetry_failure_stops(self):
        self.assertIn("unavailable", self.assert_stop(RuntimeError("telemetry unavailable"))["stop_reason"])

    def test_busy_gpu_preflight_is_read_only(self):
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample(memory_mib=2000, power_w=350)):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                self.m.preflight(self.root, "start")
        self.assertFalse(self.m.paths(self.root)[0].exists())

    def test_valid_preflight_and_duplicate_start_guard(self):
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample(power_w=350)):
            self.m.preflight(self.root, "start")
            self.m.paths(self.root)[1].mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.m.preflight(self.root, "start")

    def test_resume_requires_identical_manifest(self):
        logdir, checkpoint = self.m.paths(self.root)
        logdir.mkdir(parents=True)
        checkpoint.mkdir(parents=True)
        (logdir / "config.json").write_text(json.dumps(self.m.manifest(self.root)))
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample(power_w=350)):
            self.m.preflight(self.root, "resume")
            self.assertEqual(self.m.command(self.root, "resume")[-1], "--resume")
            (self.root / "scripts/train/pi05_hbot.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "mismatched"):
                self.m.preflight(self.root, "resume")

    def test_exit_zero_without_final_checkpoint_is_not_success(self):
        self.launcher("pass")
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample()):
            self.assertEqual(self.m.run(self.root, "start"), 1)
        self.assertEqual(self.report()["status"], "failed")

    def test_sigterm_is_handled_and_child_stopped(self):
        self.launcher("import os,signal,time; time.sleep(0.1); os.kill(os.getppid(), signal.SIGTERM); time.sleep(5)")
        with mock.patch.object(self.m, "query_gpu", return_value=self.sample()):
            self.assertEqual(self.m.run(self.root, "start"), 1)
        self.assertIn("SIGTERM", self.report()["stop_reason"])


class LongPowerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hbot-long-power-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "power").write_text("350.00")
        self.env = dict(os.environ, MOCK_ROOT=str(self.root), INVOCATION_ID="a"*32)
        fake = self.root / "nvidia-smi"
        fake.write_text(f"#!{sys.executable}\n" + '''import os,pathlib,sys
r=pathlib.Path(os.environ['MOCK_ROOT']); a=sys.argv[1:]
if '-pl' in a:
    with (r/'changes').open('a') as f: f.write(a[a.index('-pl')+1]+'\\n')
    if os.environ.get('MOCK_SET_FAIL')=='1': sys.exit(1)
    (r/'power').write_text(f"{float(a[a.index('-pl')+1]):.2f}")
elif '--query-gpu=uuid' in a: print('GPU-d32118cd-3b44-94bf-d286-79df4d357528')
elif '--query-gpu=memory.used' in a: print(os.environ.get('MOCK_USED','18'))
elif '--query-gpu=power.limit' in a: print((r/'power').read_text())
''')
        fake.chmod(0o755)
        source = (SOURCE.parent / "pi05-longtrain-power.sh").read_text()
        source = source.replace('[[ $EUID == 0 ]]', '[[ 1 == 1 ]]')
        source = source.replace('/usr/bin/nvidia-smi', str(fake))
        source = source.replace('/run/hbot-pi05-longtrain', str(self.root))
        self.helper = self.root / "power.sh"
        self.helper.write_text(source)

    def invoke(self, mode, **env):
        return subprocess.run(["bash", str(self.helper), mode], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=10)

    def test_start_and_restore(self):
        result = self.invoke("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "power").read_text(), "280.00")
        result = self.invoke("restore")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "power").read_text(), "350.00")
        self.assertFalse((self.root / "gpu0-power.state").exists())

    def test_failed_set_can_restore(self):
        self.assertNotEqual(self.invoke("start", MOCK_SET_FAIL="1").returncode, 0)
        self.assertTrue((self.root / "gpu0-power.state").exists())
        self.assertEqual(self.invoke("restore").returncode, 0)
        self.assertEqual((self.root / "power").read_text(), "350.00")

    def test_without_marker_does_not_touch_power(self):
        self.assertEqual(self.invoke("restore").returncode, 0)
        self.assertFalse((self.root / "changes").exists())

    def test_wrong_invocation_does_not_restore(self):
        self.assertEqual(self.invoke("start").returncode, 0)
        self.assertNotEqual(self.invoke("restore", INVOCATION_ID="b"*32).returncode, 0)
        self.assertEqual((self.root / "power").read_text(), "280.00")
        self.assertTrue((self.root / "gpu0-power.state").exists())

    def test_busy_gpu_no_change(self):
        self.assertNotEqual(self.invoke("start", MOCK_USED="2000").returncode, 0)
        self.assertFalse((self.root / "changes").exists())
        self.assertFalse((self.root / "gpu0-power.state").exists())

    def test_failed_restore_retains_marker(self):
        self.assertEqual(self.invoke("start").returncode, 0)
        self.assertNotEqual(self.invoke("restore", MOCK_SET_FAIL="1").returncode, 0)
        self.assertTrue((self.root / "gpu0-power.state").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

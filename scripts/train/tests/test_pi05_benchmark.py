"""CPU-only integration tests: fake GPU monitor and tiny child process, no model/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BenchmarkSafetyTest(unittest.TestCase):
    def run_fake(self, temperature):
        with tempfile.TemporaryDirectory(prefix='hbot-benchmark-test-') as folder:
            root = Path(folder)
            (root / 'scripts').mkdir()
            (root / 'logs').mkdir()
            (root / 'bin').mkdir()
            trainer = root / 'fake_train.py'
            trainer.write_text('''import json, os, pathlib, sys, time
args = sys.argv
experiment = args[args.index('--exp-name') + 1]
for step in range(3):
    time.sleep(0.3)
    print(f'Step {step}: grad_norm=0.2, loss=0.1, param_norm=1.0', flush=True)
checkpoint = pathlib.Path(os.environ['HBOT_HOME']) / 'checkpoints/pi05_hbot_lora' / experiment / '2'
checkpoint.mkdir(parents=True)
(checkpoint / '_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs': 1}))
''')
            (root / 'scripts/run-pi05-lora.sh').write_text(
                f'#!/bin/sh\nexec "{sys.executable}" -u "{trainer}" "$@"\n')
            smi = root / 'bin/nvidia-smi'
            smi.write_text(f'#!{sys.executable}\nimport os,time\n'
                           'while True:\n'
                           '    print("2026/10/27 00:00:00, 0, 200, 10, " + os.environ["MOCK_TEMP"], flush=True)\n'
                           '    time.sleep(0.1)\n')
            smi.chmod(0o755)
            runner = Path(__file__).resolve().parents[1] / 'benchmark-pi05-lora.py'
            env = dict(os.environ, HBOT_HOME=str(root), MOCK_TEMP=str(temperature),
                       PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
            result = subprocess.run([sys.executable, str(runner), '--batch-size', '2', '--steps', '3',
                                     '--warmup-steps', '1', '--exp-name', 'test'],
                                    env=env, capture_output=True, text=True, timeout=30)
            report = json.loads((root / 'logs/pi05_bench_test/report.json').read_text())
            return result, report

    def test_normal_run_reports_three_steps(self):
        result, report = self.run_fake(40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual(report['steps_completed'], 3)
        self.assertEqual(report['steady_step_intervals'], 2)
        self.assertFalse(report['thermal_cutoff_triggered'])

    def test_temperature_cutoff_stops_own_run(self):
        result, report = self.run_fake(90)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report['status'], 'FAILED')
        self.assertTrue(report['thermal_cutoff_triggered'])
        self.assertFalse(report['checkpoint_committed'])
        self.assertIn('stopping only this benchmark', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)

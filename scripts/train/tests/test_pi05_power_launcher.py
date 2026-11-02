"""No sudo/GPU/systemd use: mock privileged commands to test power restoration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PowerLauncherTest(unittest.TestCase):
    def run_fake(self, **options):
        with tempfile.TemporaryDirectory(prefix='hbot-power-test-') as folder:
            root = Path(folder)
            for name in ('scripts/train', 'openpi/.venv/bin', 'lock', 'bin'):
                (root / name).mkdir(parents=True, exist_ok=True)
            (root / 'scripts/train/benchmark-pi05-lora.py').touch()
            python = root / 'openpi/.venv/bin/python'
            python.touch()
            python.chmod(0o755)
            (root / 'power').write_text(options.pop('original_power', '350.00'))
            source = (Path(__file__).resolve().parents[2] / 'run-pi05-280w-test.sh').read_text()
            # Only this temporary test copy bypasses the real helper's root guard.
            source = source.replace('[[ $EUID == 0 ]]', '[[ 1 == 1 ]]')
            source = source.replace('/study/wangmengfan/hbot', str(root))
            source = source.replace('/run/lock/', str(root / 'lock') + '/')
            for command in ('nvidia-smi', 'systemd-run', 'systemctl', 'runuser', 'timeout'):
                fake = root / 'bin' / command
                fake.write_text(f'#!{sys.executable}\n' + '''import json,os,pathlib,sys
r=pathlib.Path(os.environ['MOCK_ROOT'])
name=pathlib.Path(sys.argv[0]).name
a=sys.argv[1:]
with (r/'calls').open('a') as f: f.write(json.dumps([name,a])+'\\n')
if name=='nvidia-smi':
    if '-pl' in a: (r/'power').write_text(f"{float(a[a.index('-pl')+1]):.2f}")
    elif '--query-gpu=uuid' in a: print('GPU-d32118cd-3b44-94bf-d286-79df4d357528')
    elif '--query-gpu=memory.used' in a: print(os.environ.get('MOCK_USED','18'))
    elif '--query-gpu=power.limit' in a: print((r/'power').read_text())
elif name=='systemd-run': sys.exit(int(os.environ.get('MOCK_TIMER_FAIL','0')))
elif name=='runuser': sys.exit(int(os.environ.get('MOCK_TRAIN_STATUS','0')))
''')
                fake.chmod(0o755)
                original = '/usr/sbin/runuser' if command == 'runuser' else '/usr/bin/' + command
                source = source.replace(original, str(fake))
            helper = root / 'helper.sh'
            helper.write_text(source)
            result = subprocess.run(['bash', str(helper)],
                env=dict(os.environ, MOCK_ROOT=str(root), **options),
                capture_output=True, text=True, timeout=10)
            calls = [json.loads(line) for line in (root/'calls').read_text().splitlines()]
            return result, (root/'power').read_text(), calls

    def assert_restored(self, training_status):
        result, power, calls = self.run_fake(MOCK_TRAIN_STATUS=str(training_status))
        self.assertEqual(result.returncode, training_status, result.stdout + result.stderr)
        self.assertEqual(power, '350.00')
        changes = [args[args.index('-pl')+1] for name,args in calls if name=='nvidia-smi' and '-pl' in args]
        self.assertEqual(changes, ['280', '350.00'])
        self.assertLess(next(i for i,(name,_) in enumerate(calls) if name=='systemd-run'),
                        next(i for i,(name,args) in enumerate(calls) if name=='nvidia-smi' and '-pl' in args))
        self.assertTrue(any(name=='systemctl' and args[0]=='stop' for name,args in calls))
        self.assertIn('RESTORED_POWER_LIMIT_W=350.00', result.stdout)

    def test_success_restores_original_power(self):
        self.assert_restored(0)

    def test_training_failure_restores_original_power(self):
        self.assert_restored(7)

    def test_failed_fallback_setup_never_changes_power(self):
        result, power, calls = self.run_fake(MOCK_TIMER_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(power, '350.00')
        self.assertFalse(any(name=='nvidia-smi' and '-pl' in args for name,args in calls))

    def test_busy_gpu_is_untouched(self):
        result, power, calls = self.run_fake(MOCK_USED='2000')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(name in ('systemd-run','runuser') for name,_ in calls))
        self.assertEqual(power, '350.00')

    def test_unexpected_original_power_is_untouched(self):
        result, power, calls = self.run_fake(original_power='250.00')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(name in ('systemd-run','runuser') for name,_ in calls))
        self.assertEqual(power, '250.00')


if __name__ == '__main__':
    unittest.main(verbosity=2)

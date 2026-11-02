"""Mocked CLI selection/stop: never call the machine's actual service manager."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SystemctlSelectionTest(unittest.TestCase):
    def exercise(self, primary_matches=False, no_match=False, stop_failure=False):
        with tempfile.TemporaryDirectory(prefix="hbot-systemctl-test-") as tmp:
            root = Path(tmp)
            primary, original = root / "systemctl", root / "mctes"
            source = f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
r=pathlib.Path(os.environ['MOCK_ROOT'])
with (r/'calls.jsonl').open('a') as f:
    f.write(json.dumps([pathlib.Path(sys.argv[0]).name,sys.argv[1:]])+'\\n')
if sys.argv[1:2]==['stop']: sys.exit(int(os.environ.get('MOCK_STOP_RC','0')))
'''
            primary.write_text(source if primary_matches else source + "# replacement\n")
            original.write_text(source)
            primary.chmod(0o755)
            original.chmod(0o755)
            expected = "0" * 32 if no_match else hashlib.md5(source.encode()).hexdigest()
            reference = root / "systemd.md5sums"
            reference.write_text(expected + "  bin/systemctl\n")
            smi = root / "nvidia-smi"
            smi.write_text("#!/usr/bin/bash\necho '350.00 W, 18 MiB'\n")
            smi.chmod(0o755)
            script = (Path(__file__).resolve().parents[2] / "run-pi05-longtrain.sh").read_text()
            script = script.replace('[[ $EUID == 0 ]]', '[[ 1 == 1 ]]')
            script = script.replace('[[ $(stat -Lc \'%u\' "$candidate") == 0 ]]', '[[ 1 == 1 ]]')
            script = script.replace('/var/lib/dpkg/info/systemd.md5sums', str(reference))
            script = script.replace('/usr/bin/systemctl', str(primary)).replace('/usr/bin/mctes', str(original))
            script = script.replace('/usr/bin/nvidia-smi', str(smi))
            launcher = root / "launcher.sh"
            launcher.write_text(script)
            result = subprocess.run(["bash", str(launcher), "stop"], capture_output=True, text=True,
                                    env=dict(os.environ, MOCK_ROOT=str(root), MOCK_STOP_RC="7" if stop_failure else "0"), timeout=10)
            calls_file = root / "calls.jsonl"
            calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
            return result, calls

    def test_prefers_standard_binary_when_package_matches(self):
        result, calls = self.exercise(primary_matches=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [["systemctl", ["stop", "hbot-pi05-lora-40k.service"]]])

    def test_selects_verified_original_instead_of_replacement(self):
        result, calls = self.exercise()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [["mctes", ["stop", "hbot-pi05-lora-40k.service"]]])

    def test_no_matching_binary_refuses_all_service_actions(self):
        result, calls = self.exercise(no_match=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_stop_failure_is_not_hidden(self):
        result, calls = self.exercise(stop_failure=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls[0][0], "mctes")


if __name__ == "__main__":
    unittest.main(verbosity=2)

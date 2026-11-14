"""Exercise WSL dispatch without installing packages or loading model weights."""
import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ENTRY = Path(__file__).resolve().parents[1] / "windows-wsl.sh"


@unittest.skipUnless(sys.platform == "linux", "Bash dispatcher tests run on Linux/WSL")
class WindowsEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "install with spaces"
        self.root.mkdir()
        (self.root / ".hbot-windows-install").touch()
        self.package = self.root / "package"
        self.package.mkdir()
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.write_command("uname", "case $1 in -m) echo x86_64;; -r) echo 6.6-microsoft-standard-WSL2;; esac")
        self.write_command("tmux", "exit 1")
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")

    def write_command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o700)

    def call(self, action, checkpoint="", root=None):
        values = [str(root or self.root), checkpoint, "1", "direct", ""]
        encoded = ["b64:" + base64.b64encode(v.encode()).decode() for v in values]
        return subprocess.run(["bash", str(ENTRY), action, *encoded],
                              env=self.env, capture_output=True, text=True)

    def test_configuration_preserves_literal_path(self):
        checkpoint = "/home/alice/weights with 'quotes'/$literal;echo unsafe"
        result = self.call("Configure", checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$HBOT_PI05_CHECKPOINT"',
                                 "test", str(self.package / "local.env")], capture_output=True, text=True)
        self.assertEqual(result.stdout, checkpoint)

    def test_active_session_blocks_configuration(self):
        self.write_command("tmux", "exit 0")
        result = self.call("Configure", "/weights")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.package / "local.env").exists())

    def test_setup_refuses_unowned_directory(self):
        (self.root / ".hbot-windows-install").unlink()
        result = self.call("Setup")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not managed", result.stderr)

    def test_windows_mount_not_used_for_runtime(self):
        result = self.call("Configure", root=Path("/mnt/c/hbot"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Linux filesystem", result.stderr)

    def test_check_dispatches_original_entry(self):
        (self.package / "test.sh").write_text('printf "%s" "$1"\n')
        result = self.call("Check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "--check")

    def test_cannot_clear_live_session(self):
        self.write_command("tmux", 'case "$*" in *list-panes*) echo 0;; *kill-session*) exit 99;; esac')
        result = self.call("ClearSession")
        self.assertEqual(result.returncode, 2)
        self.assertIn("still running", result.stderr)

    @unittest.skipUnless(shutil.which("tmux"), "tmux not installed")
    def test_real_session_keeps_exit_status_without_relaunch(self):
        (self.bin / "tmux").unlink()
        (self.package / "run-official.sh").write_text('echo run >> launches\nexit 7\n')
        try:
            # Attaching has no TTY in this test; the detached service still runs.
            self.call("Start")
            for _ in range(30):
                status = self.call("Status")
                if "dead=1 exit=7" in status.stdout:
                    break
                time.sleep(0.1)
            self.assertIn("dead=1 exit=7", status.stdout)
            self.call("Start")
            self.assertEqual((self.package / "launches").read_text(), "run\n")
            result = self.call("ClearSession")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No evaluation session", self.call("Status").stdout)
        finally:
            self.call("ClearSession")


if __name__ == "__main__":
    unittest.main()

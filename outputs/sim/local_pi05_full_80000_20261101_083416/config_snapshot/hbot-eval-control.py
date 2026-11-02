#!/usr/bin/env python3
"""CPU-only supervisor for one H-Bot evaluation; run inside hbot-sim.

No model/simulator imports. Cancellation only targets verified eval_policy.py
processes with the exact output directory. Never stops the container or server.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "robo_orchard_sim"
EVAL_REL = "examples/manipulation-app/scripts/eval_policy.py"
OUTPUTS = ROOT / "outputs/sim"
STATE = ROOT / "tmp/local-eval-control"
RUN_PATTERN = re.compile(r"local_[A-Za-z0-9_-]+\Z")


def validate_run(name):
    if not RUN_PATTERN.fullmatch(name):
        raise ValueError("Invalid run name; expected local_NAME without slashes")
    return name


def argument(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def process_info(pid):
    """Read identity and exact argv; reject other users, zombies and other code."""
    try:
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != os.getuid():
            return None
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        argv = [os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
            return None
        if argv[1] not in (EVAL_REL, str(REPO / EVAL_REL)):
            return None
        output = argument(argv, "--output-dir")
        if not output or Path(output).parent != OUTPUTS:
            return None
        if not RUN_PATTERN.fullmatch(Path(output).name):
            return None
        return {
            "pid": int(pid), "start_ticks": fields[19], "output": output,
            "role": "worker" if "--_shard-id" in argv else "dispatcher",
        }
    except (OSError, ValueError, IndexError):
        return None


def processes(run_name=None):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = process_info(path.name)
            if item and (run_name is None or Path(item["output"]).name == run_name):
                result.append(item)
    return sorted(result, key=lambda item: item["pid"])


def send_signal(item, sig, group=False):
    # Verify start time and argv again immediately before signalling (PID reuse).
    if process_info(item["pid"]) != item:
        return
    try:
        if group and os.getpgid(item["pid"]) == item["pid"]:
            os.killpg(item["pid"], sig)
        else:
            os.kill(item["pid"], sig)
    except ProcessLookupError:
        pass


@contextlib.contextmanager
def lock(name, timeout=0):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / name).open("a+") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Evaluation lock is busy: " + name)
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def cancel_path(run_name):
    return STATE / (run_name + ".cancel")


def wait_stopped(run_name, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not processes(run_name):
            return True
        time.sleep(0.2)
    return not processes(run_name)


def stop(run_name):
    validate_run(run_name)
    STATE.mkdir(parents=True, exist_ok=True)
    cancel_path(run_name).touch()
    # Serialize host cleanup, external --stop, and supervisor cleanup.
    with lock(run_name + ".stop.lock", timeout=30):
        # Synchronize with check/spawn so an early Ctrl+C cannot miss the child.
        with lock(run_name + ".launch.lock", timeout=5):
            initial = processes(run_name)
            for item in initial:
                if item["role"] == "dispatcher":
                    send_signal(item, signal.SIGTERM)
            if not any(item["role"] == "dispatcher" for item in initial):
                for item in initial:
                    send_signal(item, signal.SIGTERM, group=True)
        if not wait_stopped(run_name, 12):
            for item in processes(run_name):
                send_signal(item, signal.SIGTERM, group=item["role"] == "worker")
            if not wait_stopped(run_name, 6):
                for item in processes(run_name):
                    send_signal(item, signal.SIGKILL, group=True)
                if not wait_stopped(run_name, 3):
                    raise RuntimeError("Some evaluation processes remain: " + run_name)
    print("STOPPED " + run_name + " (model, container and files preserved)", flush=True)


def run(run_name, config):
    validate_run(run_name)
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.yaml", config):
        raise ValueError("Invalid config filename")
    if not (ROOT / "config" / config).is_file():
        raise ValueError("Evaluation config not found: " + config)
    num_record = int(os.environ.get("HBOT_EVAL_NUM_RECORD", "2"))
    if not 0 <= num_record <= 10:
        raise ValueError("HBOT_EVAL_NUM_RECORD must be between 0 and 10")
    export_video = os.environ.get("HBOT_EVAL_EXPORT_VIDEO", "1")
    if export_video not in ("0", "1"):
        raise ValueError("HBOT_EVAL_EXPORT_VIDEO must be 0 or 1")
    requested_signal = []

    def interrupted(signum, frame):
        requested_signal.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    # Held in the container, not just by the docker CLI. Legacy runs are scanned.
    with lock("gpu-0.lock"):
        active = processes()
        if active:
            raise RuntimeError("Another evaluation is active; no new run started:\n" +
                               json.dumps(active, indent=2))
        child = None
        try:
            with lock(run_name + ".launch.lock", timeout=5):
                if requested_signal or cancel_path(run_name).exists():
                    return 130
                output = OUTPUTS / run_name
                if output.exists():
                    raise ValueError("Refusing to reuse an output directory: " + str(output))
                command = [
                    sys.executable, EVAL_REL,
                    "--eval-config", str(ROOT / "config" / config),
                    "--output-dir", str(output), "--gpus", "0",
                ]
                if num_record:
                    command.extend([
                        "--enable-recording", "--num-record", str(num_record),
                    ])
                    if export_video == "1":
                        command.append("--export-video")
                child = subprocess.Popen(command, cwd=REPO, start_new_session=True)
                print("DISPATCHER " + json.dumps({"run": run_name, "pid": child.pid}), flush=True)
            while child.poll() is None:
                if requested_signal or cancel_path(run_name).exists():
                    stop(run_name)
                    child.wait(timeout=5)
                    return 128 + (requested_signal[0] if requested_signal else signal.SIGTERM)
                time.sleep(0.2)
            return child.returncode if child.returncode >= 0 else 128 - child.returncode
        finally:
            if child is not None:
                # Also cover exceptions or an abnormal dispatcher exit with workers left.
                if processes(run_name):
                    stop(run_name)
                child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    stop_parser = commands.add_parser("stop")
    stop_parser.add_argument("run_name")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("run_name")
    run_parser.add_argument("config")
    args = parser.parse_args()
    try:
        if args.command == "status":
            print(json.dumps(processes(), indent=2))
            return 0
        if args.command == "stop":
            stop(args.run_name)
            return 0
        return run(args.run_name, args.config)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print("EVAL CONTROL ERROR: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fixed 40k-step job: ordinary OpenPI launcher with a fail-closed GPU guard.

This process never changes power or stops an unrelated process. The systemd
unit runs it as wangmengfan and restores power separately through ExecStopPost.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


PROFILE = dict(exp_name="hbot_pi05_lora_b4_40k_v1", batch_size=4,
               num_train_steps=40000, num_workers=2, log_interval=10,
               lr_decay_steps=40000, save_interval=500, keep_period=5000)
GPU_UUID = "GPU-d32118cd-3b44-94bf-d286-79df4d357528"
POWER_W = 280
CUTOFF_C = 90
SAMPLE_SECONDS = 1.0
STOP_GRACE_SECONDS = 20
MIN_FREE_BYTES = 100 * 1024**3
NVIDIA_SMI = "/usr/bin/nvidia-smi"


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(root):
    exp = PROFILE["exp_name"]
    return (root / "logs/pi05_longtrain" / exp,
            root / "checkpoints/pi05_hbot_lora" / exp)


def manifest(root):
    files = ("scripts/run-pi05-lora.sh", "scripts/train/train-pi05-hbot.py",
             "scripts/train/pi05_hbot.py",
             "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/norm_stats.json")
    return dict(schema=1, profile=PROFILE, gpu_uuid=GPU_UUID, power_w=POWER_W,
                cutoff_c=CUTOFF_C,
                sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in files})


def query_gpu():
    result = subprocess.run(
        [NVIDIA_SMI, "-i", "0", "--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.limit",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    fields = next(csv.reader(result.stdout.splitlines()))
    if len(fields) != 5:
        raise RuntimeError("Invalid GPU telemetry: expected exactly five fields")
    uuid, used, utilization, temperature, power = [v.strip() for v in fields]
    sample = dict(time=now(), uuid=uuid, memory_mib=float(used),
                  utilization=float(utilization), temperature_c=float(temperature), power_w=float(power))
    if not all(math.isfinite(sample[k]) for k in ("memory_mib", "utilization", "temperature_c", "power_w")):
        raise RuntimeError("Invalid/non-finite GPU telemetry")
    if uuid != GPU_UUID:
        raise RuntimeError("GPU 0 identity changed; refusing to use it")
    if not (0 <= sample["temperature_c"] <= 120 and sample["memory_mib"] >= 0
            and 0 <= sample["utilization"] <= 100):
        raise RuntimeError("Out-of-range GPU telemetry")
    return sample


def check_sample(sample, expected_power, idle=False):
    if sample["uuid"] != GPU_UUID or abs(sample["power_w"] - expected_power) > 0.1:
        raise RuntimeError(f"GPU identity/power mismatch: expected {GPU_UUID}, {expected_power} W")
    if sample["temperature_c"] >= CUTOFF_C:
        raise RuntimeError(f"Temperature cutoff: {sample['temperature_c']} C >= {CUTOFF_C} C")
    if idle and sample["memory_mib"] > 256:
        raise RuntimeError(f"GPU busy ({sample['memory_mib']} MiB); existing jobs are untouched")


def preflight(root, mode, expected_power=350):
    required = ("scripts/run-pi05-lora.sh", "scripts/train/train-pi05-hbot.py",
                "scripts/train/pi05_hbot.py", "config/hbot-env.sh", "openpi/.venv/bin/python",
                "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/preparation.json",
                "models/pi05_base/params", "datasets/InstructMove_hbot/lerobot_unpacked")
    for name in required:
        if not (root / name).exists():
            raise RuntimeError(f"Missing prerequisite: {root / name}")
    if not os.access(root / "openpi/.venv/bin/python", os.X_OK):
        raise RuntimeError("Virtual-environment Python is not executable")
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        raise RuntimeError("Less than 100 GiB free; refusing long training")
    expected = manifest(root)
    logdir, checkpoint = paths(root)
    if mode == "start":
        if logdir.exists() or checkpoint.exists():
            raise RuntimeError("Experiment already exists. Nothing overwritten; use resume only for the same run.")
    else:
        saved = logdir / "config.json"
        if not saved.is_file() or json.loads(saved.read_text()) != expected:
            raise RuntimeError("Resume manifest missing or mismatched; refusing incompatible resume")
        if not checkpoint.is_dir():
            raise RuntimeError("No checkpoint directory exists for resume; inspect the previous failure first")
    check_sample(query_gpu(), expected_power, idle=True)
    return expected


def command(root, mode):
    args = ["/usr/bin/bash", str(root / "scripts/run-pi05-lora.sh")]
    for key, value in PROFILE.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    if mode == "resume":
        args.append("--resume")
    return args


def stop_owned_group(child):
    """Only the session/process group created by our own Popen is targeted."""
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # Include any data-loader/tee descendants left after the direct child exits.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def latest_committed(checkpoint):
    found = []
    if checkpoint.is_dir():
        for entry in checkpoint.iterdir():
            if entry.name.isdigit():
                metadata = entry / "_CHECKPOINT_METADATA"
                try:
                    if json.loads(metadata.read_text()).get("commit_timestamp_nsecs"):
                        found.append(int(entry.name))
                except (OSError, ValueError):
                    continue
    return max(found) if found else None


def logged_step(logfile):
    # Read a bounded tail; progress is a lower bound because log_interval is 10.
    with logfile.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 256 * 1024))
        text = handle.read().decode(errors="replace")
    steps = re.findall(r"\bStep\s+(\d+)\s*:", text)
    return max(map(int, steps)) if steps else None


def run(root, mode):
    expected = preflight(root, mode, expected_power=POWER_W)
    logdir, checkpoint = paths(root)
    if mode == "start":
        logdir.mkdir(parents=True, exist_ok=False)
        with (logdir / "config.json").open("x") as handle:
            json.dump(expected, handle, indent=2)
    # Do not precreate the checkpoint directory: OpenPI owns its lifecycle.
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report = dict(attempt=attempt, start=now(), mode=mode, command=command(root, mode),
                  profile=PROFILE, status="starting", stop_reason=None)
    requested = []

    def request_stop(signum, _frame):
        requested.append(f"Received signal {signal.Signals(signum).name}")

    previous = {sig: signal.signal(sig, request_stop)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    child = None
    trainlog = logdir / "train.log"
    rc = 1
    try:
        with trainlog.open("a", buffering=1) as output, (logdir / "gpu.csv").open("a", buffering=1) as gpu_log:
            output.write(f"\n[{now()}] ATTEMPT {attempt} mode={mode} profile={json.dumps(PROFILE)}\n")
            writer = csv.DictWriter(gpu_log, fieldnames=("time", "uuid", "memory_mib", "utilization", "temperature_c", "power_w"))
            if gpu_log.tell() == 0:
                writer.writeheader()
            if requested:
                raise RuntimeError(requested[0])
            child = subprocess.Popen(command(root, mode), cwd=root, start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, HBOT_HOME=str(root), HBOT_TRAIN_GPU="0", PYTHONUNBUFFERED="1"))
            report.update(status="running", child_pid=child.pid)
            print(f"Started owned training PID {child.pid}; log={trainlog}", flush=True)
            peak_temp = 0
            peak_mem = 0
            while child.poll() is None:
                if requested:
                    raise RuntimeError(requested[0])
                sample = query_gpu()  # bounded timeout; failure stops, never runs blind
                writer.writerow(sample)
                peak_temp = max(peak_temp, sample["temperature_c"])
                peak_mem = max(peak_mem, sample["memory_mib"])
                report.update(peak_temperature_c=peak_temp, peak_memory_mib=peak_mem)
                check_sample(sample, POWER_W)
                time.sleep(SAMPLE_SECONDS)
            rc = child.returncode
            committed = latest_committed(checkpoint)
            if rc == 0 and committed == PROFILE["num_train_steps"] - 1:
                report["status"] = "completed"
            else:
                report.update(status="failed", stop_reason=f"Child exit={rc}; latest committed checkpoint={committed}")
                rc = rc or 1
    except Exception as exc:
        report.update(status="stopped", stop_reason=f"{type(exc).__name__}: {exc}")
        print(report["stop_reason"], flush=True)
        rc = 1
    finally:
        try:
            if child is not None:
                stop_owned_group(child)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            report.update(end=now(), returncode=rc, latest_committed_checkpoint=latest_committed(checkpoint),
                          last_logged_step_lower_bound=logged_step(trainlog) if trainlog.exists() else None)
            with (logdir / f"report_{attempt}.json").open("x") as handle:
                json.dump(report, handle, indent=2)
            with trainlog.open("a") as handle:
                handle.write(f"\n[{now()}] SUPERVISOR_RESULT {json.dumps(report)}\n")
            print(json.dumps(report), flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=("start", "resume"), default="start")
    args = parser.parse_args()
    root = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot")).resolve()
    if args.preflight:
        preflight(root, args.mode)
        print("PREFLIGHT_OK: read-only; no GPU workload or power changes")
        return 0
    return run(root, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

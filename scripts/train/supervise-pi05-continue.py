#!/usr/bin/env python3
"""Prepare an independent full-state 40k -> 60k continuation; never start fresh.

Reuses the tested GPU0 power/temperature guard. Preparation only copies files;
training requires the separate sudo systemd launcher to apply the 280W cap.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


spec = importlib.util.spec_from_file_location(
    "pi05_guard", Path(__file__).with_name("supervise-pi05-longtrain.py"))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
SOURCE_PROFILE = dict(guard.PROFILE)
SOURCE_STEP = 39999
TARGET_STEPS = 60000
guard.PROFILE = dict(SOURCE_PROFILE, exp_name="hbot_pi05_lora_b4_60k_from40k_v1",
                     num_train_steps=TARGET_STEPS)
original_manifest = guard.manifest
original_preflight = guard.preflight
EXTRA_FILES = (
    "scripts/train/supervise-pi05-longtrain.py",
    "scripts/train/supervise-pi05-continue.py",
    "scripts/train/pi05-continue-60k-power.sh",
    "scripts/train/hbot-pi05-lora-60k.service",
    "scripts/run-pi05-continue-60k.sh",
    "config/hbot-env.sh",
)
NORM = "assets/hbot_panda_joint_delta_h15/norm_stats.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(root):
    exp = SOURCE_PROFILE["exp_name"]
    return (root / "logs/pi05_longtrain" / exp / "config.json",
            root / "checkpoints/pi05_hbot_lora" / exp / str(SOURCE_STEP))


def check_checkpoint(path):
    for name in ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA", NORM):
        if not (path / name).is_file():
            raise RuntimeError(f"Incomplete full-state checkpoint: {path / name}")
    if not json.loads((path / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs"):
        raise RuntimeError(f"Checkpoint not committed: {path}")


def continuation_manifest(root):
    result = original_manifest(root)
    saved_config, source = source_paths(root)
    check_checkpoint(source)
    # Require the same training/data-transform sources as the original 40k run.
    expected_source = dict(result, profile=SOURCE_PROFILE)
    if json.loads(saved_config.read_text()) != expected_source:
        raise RuntimeError("Original training manifest changed; refusing incompatible continuation")
    norm_config = root / "config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/norm_stats.json"
    if digest(source / NORM) != digest(norm_config):
        raise RuntimeError("Checkpoint normalization differs from current training assets")
    result["sha256"].update({name: digest(root / name) for name in EXTRA_FILES})
    result["continuation"] = dict(
        source_checkpoint=str(source), source_step=SOURCE_STEP,
        expected_restored_train_step=40000, target_train_step=TARGET_STEPS,
        source_manifest_sha256=digest(saved_config),
        source_checkpoint_metadata_sha256=digest(source / "_CHECKPOINT_METADATA"),
        restore="params + train_state (optimizer and global step) + assets",
        learning_rate="Original decay_steps=40000 retained; floor 2.5e-6",
        data_iterator="OpenPI recreates its shuffled data iterator on resume",
    )
    return result


def inventory(path):
    result = {}
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(f"Unexpected symlink in checkpoint: {entry}")
        if entry.is_file():
            result[str(entry.relative_to(path))] = entry.stat().st_size
    return result


def prepare(root):
    expected = continuation_manifest(root)
    logdir, checkpoint = guard.paths(root)
    if logdir.exists() or checkpoint.exists():
        raise RuntimeError("Continuation already exists or preparation was interrupted; nothing overwritten")
    if guard.shutil.disk_usage(root).free < guard.MIN_FREE_BYTES:
        raise RuntimeError("Less than 100 GiB free; cannot prepare continuation")
    _, source = source_paths(root)
    original_files = inventory(source)
    # A separate copy/reflink, never hardlinks. Original checkpoints remain intact
    # when OpenPI later prunes checkpoints inside the new experiment directory.
    checkpoint.mkdir(parents=True, exist_ok=False)
    target = checkpoint / str(SOURCE_STEP)
    subprocess.run(["/usr/bin/cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
    check_checkpoint(target)
    if inventory(target) != original_files or inventory(source) != original_files:
        raise RuntimeError("Checkpoint copy inventory mismatch; preparation not committed")
    for name in ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA", NORM):
        if digest(source / name) != digest(target / name):
            raise RuntimeError(f"Checkpoint copy metadata mismatch: {name}")
    logdir.mkdir(parents=True, exist_ok=False)
    with (logdir / "config.json").open("x") as handle:
        json.dump(expected, handle, indent=2)
    print(json.dumps(dict(status="PREPARED_NOT_STARTED", checkpoint=str(target),
                          copied_files=len(original_files), copied_bytes=sum(original_files.values()),
                          source_preserved=True, profile=guard.PROFILE)), flush=True)


def preflight(root, mode, expected_power=350):
    if mode != "resume":
        raise RuntimeError("Continuation only supports full-state resume, never fresh training")
    _, checkpoint = guard.paths(root)
    latest = guard.latest_committed(checkpoint)
    if latest is None or latest < SOURCE_STEP or latest >= TARGET_STEPS - 1:
        raise RuntimeError(f"No eligible checkpoint to resume (latest={latest}); not starting")
    check_checkpoint(checkpoint / str(latest))
    return original_preflight(root, "resume", expected_power=expected_power)


guard.manifest = continuation_manifest
guard.preflight = preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=("resume",), default="resume")
    args = parser.parse_args()
    root = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot")).resolve()
    if args.prepare:
        prepare(root)
        return 0
    if args.preflight:
        preflight(root, args.mode)
        print("PREFLIGHT_OK: full-state continuation ready; no GPU workload or power changes")
        return 0
    return guard.run(root, "resume")


if __name__ == "__main__":
    raise SystemExit(main())

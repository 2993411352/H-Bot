#!/usr/bin/env python3
"""Validate/extract the six ZIPs and compute exact joint-delta statistics on CPU.

Raw archives are retained. No videos are decoded for statistics, no GPU is used.
Re-runs verify existing files rather than overwriting them.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import base64
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

ROOT = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot")).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_hbot import ROOT, DATA_ROOT, DATASETS, ACTION_HORIZON, NORM_DIR
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_one(archive, expected):
    if archive.stat().st_size != expected["size"] or digest_file(archive) != expected["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != archive.stem or ".." in parts or PurePosixPath(member.filename).is_absolute():
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symlinks are not permitted")
            target = DATA_ROOT.joinpath(*parts)
            if not target.resolve().is_relative_to(DATA_ROOT.resolve()):
                raise ValueError("Extracted path escapes dataset directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            # zipfile verifies member CRC while reading to EOF.
            content = z.read(member)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"Existing dataset file differs; not overwriting {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
    print("VERIFIED/EXTRACTED", archive.name, flush=True)


def calculate_statistics():
    states, deltas, rows = [], [], []
    for name in DATASETS:
        root = DATA_ROOT / name
        info = json.loads((root / "meta/info.json").read_text())
        files = sorted((root / "data").rglob("*.parquet"))
        if len(files) != 100 or info["total_episodes"] != 100 or info["fps"] != 30:
            raise ValueError(f"Unexpected episode count or fps: {name}")
        frames = 0
        for path in files:
            table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if state.shape != action.shape or state.shape[1:] != (8,):
                raise ValueError(f"Invalid state/action shapes: {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Non-finite labels: {path}")
            if len(set(table["episode_index"].to_pylist())) != 1:
                raise ValueError("Parquet spans multiple episodes")
            if table["frame_index"].to_pylist() != list(range(len(state))):
                raise ValueError("Unexpected frame order")
            # Same end-of-episode repeat-padding as LeRobot's action time queries.
            ids = np.minimum(np.arange(len(action))[:, None] + np.arange(ACTION_HORIZON), len(action) - 1)
            chunk = action[ids].copy()
            chunk[..., :7] -= state[:, None, :7]
            states.append(state)
            deltas.append(chunk.reshape(-1, 8))
            frames += len(state)
        if frames != info["total_frames"]:
            raise ValueError(f"Frame count mismatch: {name}")
        rows.append({"name": name, "episodes": len(files), "frames": frames})
        print("COUNTED", rows[-1], flush=True)
    stats = {}
    for key, arrays in (("state", states), ("actions", deltas)):
        values = np.concatenate(arrays)
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
        stats[key] = normalize.NormStats(mean=values.mean(axis=0, dtype=np.float64),
                                         std=values.std(axis=0, dtype=np.float64), q01=q01, q99=q99)
    return stats, rows


def main():
    manifest = json.loads((ROOT / "config/pi05-lerobot-download-manifest.json").read_text())
    jobs = {job["name"]: job for job in manifest["jobs"]}
    expected_files = {entry["path"]: entry for entry in jobs["lerobot"]["files"]}
    for name in DATASETS:
        archive = ROOT / "datasets/InstructMove_hbot/lerobot" / (name + ".zip")
        extract_one(archive, expected_files[archive.name])
    # Verify all checkpoint objects against their pinned download-manifest sizes.
    # CRC32C validation, when the existing dependency is available, also detects corruption.
    import google_crc32c
    count = 0
    for entry in jobs["pi05"]["files"]:
        path = ROOT / "models/pi05_base" / entry["path"]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        crc = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                crc.update(block)
        if base64.b64encode(crc.digest()).decode() != entry["crc32c"]:
            raise ValueError(f"Checkpoint CRC32C mismatch: {path}")
        count += 1
    print("CHECKPOINT VERIFIED", count, "files", flush=True)
    stats, rows = calculate_statistics()
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    payload = normalize.serialize_json(stats)
    target = NORM_DIR / "norm_stats.json"
    if target.exists() and json.loads(target.read_text()) != json.loads(payload):
        raise ValueError("Existing normalization differs; not overwriting")
    target.write_text(payload)
    report = {"datasets": list(DATASETS), "action_horizon": ACTION_HORIZON,
              "episodes": sum(x["episodes"] for x in rows), "frames": sum(x["frames"] for x in rows),
              "tasks": rows, "action_representation": "7 joint deltas relative to current state; absolute gripper",
              "sampling": "uniform_frames_all_600_episodes_no_holdout", "checkpoint_files_verified": count,
              "source": jobs["lerobot"]["source"], "norm_stats_sha256": digest_file(target)}
    (NORM_DIR / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()

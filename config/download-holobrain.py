#!/usr/bin/env python3
"""Download pinned official model exports; resume and verify every file."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi, snapshot_download

ROOT = Path("/study/wangmengfan/hbot")
MODEL = ROOT / "models/HoloBrain_v0.5_sim"
EXPORT = MODEL / "robo_orchard_sim_contest_six_tasks"
JOBS = [
    ("HorizonRobotics/HoloBrain_v0.5_sim", "e543e04c40802c21b1496284a7b424b08ac6aafc", MODEL),
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", EXPORT / "ckpt/Qwen3.5-2B"),
]


def verify(info, destination):
    for item in info.siblings:
        path = destination / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Missing/wrong-size file: {path}")
        sha = item.lfs.sha256 if item.lfs else None
        digest = hashlib.sha256() if sha else hashlib.sha1()
        if not sha:
            digest.update(f"blob {item.size}\0".encode())
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        expected = sha or item.blob_id
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch; retained for inspection: {path}")


for repo, revision, destination in JOBS:
    for attempt in range(1, 25):
        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            print(f"START {repo}@{revision} files={len(info.siblings)} bytes={sum(x.size for x in info.siblings)}", flush=True)
            snapshot_download(repo, revision=revision, local_dir=str(destination), max_workers=1)
            verify(info, destination)
            print(f"VERIFIED {repo}@{revision}", flush=True)
            break
        except Exception as exc:
            print(f"RETRY {repo} attempt={attempt} {type(exc).__name__}: {exc}", flush=True)
            if attempt == 24:
                raise
            time.sleep(60)

manifest = {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "models": [{"repo": r, "revision": v, "path": str(p)} for r, v, p in JOBS]}
(ROOT / "logs/holobrain-models-verified.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("ALL_MODELS_VERIFIED", flush=True)

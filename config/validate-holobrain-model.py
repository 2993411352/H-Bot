#!/usr/bin/env python3
"""Load verified weights through the official policy, offline and on CPU only."""
import json
import os
from pathlib import Path
import resource
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from robo_orchard_server.policy.holobrain.main import load_policy_config
from robo_orchard_server.policy.holobrain.policy import HolobrainPolicy

root = Path("/study/wangmengfan/hbot")
manifest = root / "logs/holobrain-models-verified.json"
if not manifest.is_file():
    raise SystemExit("Model files have not passed download checksum validation")
runtime = root / "models/HoloBrain_v0.5_sim_sdpa"
assert json.loads((runtime / "model.config.json").read_text())["attn_implementation"] == "sdpa"
start = time.monotonic()
print("OFFLINE_CPU_POLICY_LOAD_STARTED", flush=True)
cfg = load_policy_config(model_dir=str(runtime), device="cpu")
policy = HolobrainPolicy(cfg)
model = policy._pipeline.model
assert not model.training
assert all(parameter.device.type == "cpu" for parameter in model.parameters())
session = policy.new_session()
assert session._pipeline is policy._pipeline
session.reset()
session.close()
print("MODEL_CLASS", type(model).__name__)
print("PROCESSOR_CLASS", type(policy._pipeline.processor).__name__)
print("PARAMETERS", sum(parameter.numel() for parameter in model.parameters()))
print("PEAK_HOST_MEMORY_MIB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print("ELAPSED_SECONDS", round(time.monotonic() - start, 2))
print("OFFLINE_CPU_POLICY_LOAD_PASSED", flush=True)

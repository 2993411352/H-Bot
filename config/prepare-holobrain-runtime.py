#!/usr/bin/env python3
"""Make an SDPA runtime view without modifying verified upstream weights."""
import json
from pathlib import Path

root = Path("/study/wangmengfan/hbot")
verified = root / "logs/holobrain-models-verified.json"
if not verified.is_file():
    raise SystemExit("Weights are not verified yet; see logs/holobrain-download.log")
source = root / "models/HoloBrain_v0.5_sim/robo_orchard_sim_contest_six_tasks"
target = root / "models/HoloBrain_v0.5_sim_sdpa"
config = json.loads((source / "model.config.json").read_text())
config["attn_implementation"] = "sdpa"
target.mkdir(exist_ok=True)
for item in source.iterdir():
    if item.name in {"model.config.json", ".cache"}:
        continue
    link = target / item.name
    if link.is_symlink() and link.resolve() == item.resolve():
        continue
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Refusing to replace existing runtime path: {link}")
    link.symlink_to(item, target_is_directory=item.is_dir())
config_path = target / "model.config.json"
if config_path.exists() and json.loads(config_path.read_text()) != config:
    raise SystemExit(f"Refusing to overwrite differing configuration: {config_path}")
if not config_path.exists():
    config_path.write_text(json.dumps(config, indent=2) + "\n")
print(f"SDPA_RUNTIME_READY {target}")

#!/usr/bin/env python3
"""CPU preflight or explicit training using the official OpenPI training loop."""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU-only real-data/model-shape preflight; does NOT train")
    parser.add_argument("--exp-name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-train-steps", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--lr-decay-steps", type=int, default=20000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif not args.exp_name or not re.fullmatch(r"[A-Za-z0-9_-]+", args.exp_name):
        parser.error("Training requires a safe --exp-name (letters/digits/_/-)")
    if min(args.batch_size, args.num_train_steps, args.fsdp_devices, args.log_interval,
           args.save_interval, args.keep_period) < 1 or args.num_workers < 0:
        parser.error("Invalid batch size, step count, workers or FSDP device count")
    if args.lr_decay_steps <= 1000 or args.keep_period % args.save_interval:
        parser.error("lr-decay-steps must exceed warmup=1000; keep-period must be a multiple of save-interval")
    logging.basicConfig(level=logging.INFO)
    root = Path(os.environ.get("HBOT_HOME", "/study/wangmengfan/hbot")).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pi05_hbot import build_config, check_prepared, install_data_loader_hook
    report = check_prepared()
    config = build_config(exp_name=args.exp_name or "preflight", batch_size=args.batch_size,
                          num_train_steps=args.num_train_steps, num_workers=args.num_workers,
                          resume=args.resume, fsdp_devices=args.fsdp_devices, log_interval=args.log_interval,
                          lr_decay_steps=args.lr_decay_steps, save_interval=args.save_interval,
                          keep_period=args.keep_period)
    install_data_loader_hook()
    if args.check:
        from pi05_hbot import validate_cpu
        validate_cpu(config, report)
        return
    import jax
    if not jax.devices() or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("GPU backend unavailable; refusing accidental CPU training")
    if config.batch_size % jax.device_count() or jax.device_count() % config.fsdp_devices:
        raise ValueError("batch_size and fsdp_devices must match visible GPU count")
    sys.path.insert(0, str(root / "openpi/scripts"))
    import train
    print("FREEZE STRATEGY", config.policy_metadata["freeze_strategy"], flush=True)
    print("SCHEDULE", dataclasses.asdict(config.lr_schedule), "save_interval=", config.save_interval,
          "keep_period=", config.keep_period, flush=True)
    print("TRAINING", config.name, "devices=", jax.devices(), "checkpoint=", config.checkpoint_dir, flush=True)
    train.main(config)


if __name__ == "__main__":
    main()

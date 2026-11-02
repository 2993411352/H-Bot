#!/usr/bin/env python3
"""CPU preflight or explicit full-parameter SFT via the pinned official trainer."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--environment-check", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="20-step capacity check; authorizes matching long-run batch only on success")
    parser.add_argument("--exp-name", default="pi05_full_b16_100k_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=100000)
    parser.add_argument("--lr-decay-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--peak-lr", type=float, default=2.5e-5)
    parser.add_argument("--end-lr", type=float, default=2.5e-6)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.exp_name):
        parser.error("Unsafe experiment name")
    if min(args.batch_size, args.num_train_steps, args.lr_decay_steps,
           args.save_interval, args.keep_period, args.log_interval) < 1 or args.num_workers < 0:
        parser.error("Counts must be positive; workers can be zero")
    if not 0 <= args.warmup_steps < args.lr_decay_steps or args.keep_period % args.save_interval:
        parser.error("warmup must be less than decay; keep-period must be a multiple of save-interval")
    if not 0 < args.end_lr <= args.peak_lr or not math.isfinite(args.peak_lr):
        parser.error("Require finite 0 < end-lr <= peak-lr")
    if args.smoke and (args.num_train_steps != 20 or args.resume):
        parser.error("--smoke requires exactly 20 fresh steps, never --resume")
    return args


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def committed_steps(checkpoint_dir):
    steps = []
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and child.name.isdecimal():
            marker = child / "_CHECKPOINT_METADATA"
            if marker.is_file() and json.loads(marker.read_text()).get("commit_timestamp_nsecs"):
                steps.append(int(child.name))
    return sorted(steps)


def smoke_profile(args, h, validated, gpu_kind):
    """Tie capacity evidence to batch, device type, code, inputs and schedule."""
    return {"batch_size": args.batch_size, "gpu_kind": gpu_kind,
            "jax": importlib.metadata.version("jax"),
            "trainer_sha256": sha256(Path(__file__)), "adapter_sha256": sha256(Path(h.__file__)),
            "manifest_sha256": validated["manifest_sha256"],
            "input_fingerprint": validated["input_fingerprint"], "norm_sha256": h.NORM_SHA256,
            "lr_decay_steps": args.lr_decay_steps, "warmup_steps": args.warmup_steps,
            "peak_lr": args.peak_lr, "end_lr": args.end_lr, "num_workers": args.num_workers}


def require_smoke(path, expected):
    if not path.is_file():
        raise RuntimeError("No successful capacity check for this batch. Run run-sft.sh smoke first.")
    record = json.loads(path.read_text())
    if record.get("status") != "SMOKE_PASSED" or record.get("profile") != expected:
        raise RuntimeError("Capacity check is stale or mismatched. Rerun smoke with the same batch, GPU type and schedule.")


def resource_files(h):
    """Explicit manifest, not a recursive scan of unrelated user directories."""
    manifest = json.loads((h.PACKAGE / "resources/input-manifest.json").read_text())
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe resource manifest path")
        path = h.INPUT / relative
        if not path.resolve().is_relative_to(h.INPUT.resolve()):
            raise ValueError("Input symlink escapes mounted input tree")
        yield path, item


def input_fingerprint(h, *, verify_content=False):
    import google_crc32c
    fingerprint = hashlib.sha256()
    count = 0
    for path, entry in resource_files(h):
        stat = path.stat()
        if stat.st_size != entry["size"]:
            raise ValueError(f"Input size mismatch: {path}")
        if verify_content:
            checksum = hashlib.sha256() if "sha256" in entry else google_crc32c.Checksum()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            actual = checksum.hexdigest() if "sha256" in entry else base64.b64encode(checksum.digest()).decode()
            expected = entry.get("sha256", entry.get("crc32c"))
            if actual != expected:
                raise ValueError(f"Input checksum mismatch: {path}")
        fingerprint.update(f"{entry['path']}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        count += 1
    return {"input_fingerprint": fingerprint.hexdigest(), "files": count,
            "manifest_sha256": sha256(h.PACKAGE / "resources/input-manifest.json")}


def parameter_audit(config, *, metadata_path=None):
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    shape = jax.eval_shape(lambda key: nnx.state(config.model.create(key)), jax.random.key(42))
    flat = flax.traverse_util.flatten_dict(shape.to_pure_dict(), sep="/")
    trained = flax.traverse_util.flatten_dict(shape.filter(config.trainable_filter).to_pure_dict(), sep="/")
    if set(flat) != set(trained) or any("lora" in key for key in flat):
        raise ValueError("Not full SFT: found frozen parameters or LoRA tensors")
    if not any(key.startswith("PaliGemma/img/") for key in trained):
        raise ValueError("Vision encoder not trainable")
    if not any("llm" in key for key in trained) or not any("llm" in key and "_1" in key for key in trained):
        raise ValueError("Missing trainable language/action expert tensors")
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text())["tree_metadata"]
        checkpoint = {}
        for item in metadata.values():
            keys = [part["key"] for part in item["key_metadata"]]
            if keys[0] != "params":
                raise ValueError("Unexpected base checkpoint root")
            checkpoint["/".join(keys[1:])] = tuple(item["value_metadata"]["write_shape"])
        if set(flat) != set(checkpoint):
            raise ValueError("Base checkpoint parameter keys do not match full model")
        for key, value in flat.items():
            if tuple(value.shape) != checkpoint[key]:
                raise ValueError(f"Base checkpoint shape mismatch: {key}")
    count = sum(math.prod(value.shape) for value in flat.values())
    return {"total_parameters": count, "trainable_parameters": count,
            "frozen_parameters": 0, "lora_parameters": 0, "parameter_tensors": len(flat),
            "ema_enabled": config.ema_decay is not None}


def environment_check(config):
    import jax
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Environment check must use CPU")
    versions = {name: importlib.metadata.version(name) for name in (
        "jax", "jaxlib", "flax", "torch", "pyarrow", "av", "polars", "orbax-checkpoint", "lerobot")}
    return {"status": "ENVIRONMENT_AND_FULL_PARAMETER_SHAPES_OK", "versions": versions,
            **parameter_audit(config)}


def validate_data(config, h):
    import numpy as np
    from openpi import transforms as T
    from openpi.training import data_loader
    report = h.check_norm()
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.use_quantile_norm:
        raise ValueError("Expected quantile normalization")
    dataset = h.create_hbot_dataset()
    if len(dataset) != report["frames"] or len(dataset) != 156867:
        raise ValueError("Dataset length mismatch")
    transform = T.compose([*data_config.data_transforms.inputs,
        T.Normalize(data_config.norm_stats, use_quantiles=True), *data_config.model_transforms.inputs])
    inverse = T.compose([T.Unnormalize(data_config.norm_stats, use_quantiles=True),
                         *data_config.data_transforms.outputs])
    samples = []
    for name, part in zip(h.DATASETS, dataset.datasets, strict=True):
        for index in (0, len(part) - 1):
            raw = part[index]
            expected = np.asarray(raw["action"]).copy()
            out = transform(raw)
            assert out["state"].shape == (32,)
            assert out["actions"].shape == (15, 32)
            assert out["tokenized_prompt"].shape == (200,)
            assert all(image.shape == (224, 224, 3) for image in out["image"].values())
            assert all(out["image_mask"].values())
            restored = inverse({"actions": out["actions"].copy(), "state": out["state"].copy()})
            np.testing.assert_allclose(restored["actions"], expected, rtol=1e-4, atol=1e-5)
            if index == len(part) - 1:
                np.testing.assert_allclose(expected, np.repeat(expected[:1], 15, axis=0))
            samples.append({"dataset": name, "index": index, "prompt": raw["prompt"]})
        print("DATA/CAMERAS/ROUNDTRIP OK", name, flush=True)
    transformed = data_loader.transform_dataset(dataset, data_config)
    loader = data_loader.TorchDataLoader(transformed, local_batch_size=config.batch_size,
        num_workers=config.num_workers, num_batches=1, framework="pytorch")
    for batch in loader:
        assert tuple(batch["actions"].shape) == (config.batch_size, 15, 32)
    return {"frames": len(dataset), "episodes": report["episodes"], "samples": samples,
            "norm_sha256": h.NORM_SHA256, "validation_split": "none"}


def main():
    args = parse_args()
    if args.check or args.environment_check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    logging.basicConfig(level=logging.INFO)
    # Import order is deliberate: hbot_full preloads Arrow before JAX/Flax.
    import hbot_full as h
    config = h.build_config(args)
    h.check_norm()
    if args.environment_check:
        print(json.dumps(environment_check(config), indent=2), flush=True)
        return
    h.OUTPUT.mkdir(parents=True, exist_ok=True)
    validation_path = h.OUTPUT / "validation.json"
    h.install_data_loader_hook()
    if args.check:
        report = environment_check(config)
        report.update(input_fingerprint(h, verify_content=True))
        report.update(parameter_audit(config, metadata_path=h.BASE_PARAMS / "_METADATA"))
        report.update(validate_data(config, h))
        report.update(status="CPU_PREFLIGHT_PASSED", checked_at=time.time(),
                      trainer_sha256=sha256(Path(__file__)), adapter_sha256=sha256(Path(h.__file__)))
        validation_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    if not validation_path.is_file():
        raise RuntimeError("Run run-sft.sh check before training")
    validated = json.loads(validation_path.read_text())
    for key, value in input_fingerprint(h).items():
        if validated.get(key) != value:
            raise RuntimeError("Inputs changed since CPU preflight; rerun check")
    if (validated.get("status") != "CPU_PREFLIGHT_PASSED"
            or validated.get("adapter_sha256") != sha256(Path(h.__file__))
            or validated.get("trainer_sha256") != sha256(Path(__file__))):
        raise RuntimeError("Missing/stale preflight; rerun check")
    if shutil.disk_usage(h.OUTPUT).free < 100 * 1024**3:
        raise RuntimeError("Require at least 100 GiB free output space for full optimizer checkpoints")
    import jax
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("This package is configured for exactly one GPU")
    memory = jax.devices()[0].memory_stats() or {}
    if memory.get("bytes_limit", 0) < 70 * 1024**3:
        raise RuntimeError("JAX allocator exposes less than 70 GiB; full SFT needs an idle 80GB-class GPU")
    profile = smoke_profile(args, h, validated, jax.devices()[0].device_kind)
    smoke_path = h.OUTPUT / "capacity_checks" / f"batch_{args.batch_size}.json"
    if not args.smoke:
        require_smoke(smoke_path, profile)
    # Re-audit the actual config; never trust an experiment's name to imply full SFT.
    audit = parameter_audit(config, metadata_path=h.BASE_PARAMS / "_METADATA")
    if audit["ema_enabled"]:
        raise RuntimeError("EMA must be disabled for this memory-conservative profile")
    metadata = dataclasses.asdict(config.lr_schedule)
    signature = {key: value for key, value in vars(args).items()
                 if key not in ("check", "environment_check", "resume")}
    signature.update(adapter_sha256=sha256(Path(h.__file__)), norm_sha256=h.NORM_SHA256,
                     manifest_sha256=validated["manifest_sha256"])
    run_dir = h.OUTPUT / "logs" / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    signature_file = run_dir / "run-config.json"
    if args.resume:
        if not signature_file.is_file() or json.loads(signature_file.read_text()) != signature:
            raise RuntimeError("Resume requires identical model/data/batch/LR/steps settings and an existing run config")
        if not config.checkpoint_dir.is_dir():
            raise RuntimeError("Cannot resume without checkpoints")
        if not committed_steps(config.checkpoint_dir):
            raise RuntimeError("No committed checkpoint; use a new experiment name for a fresh start")
    elif config.checkpoint_dir.exists() or signature_file.exists():
        raise RuntimeError("Experiment already exists; choose a new name or explicit resume (no overwrites)")
    else:
        signature_file.write_text(json.dumps(signature, indent=2) + "\n")
    if args.smoke:
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps({"status": "SMOKE_RUNNING", "profile": profile,
            "experiment": args.exp_name}, indent=2) + "\n")
    print(json.dumps({"training": config.name, "devices": [str(x) for x in jax.devices()],
        "schedule": metadata, "checkpoint_dir": str(config.checkpoint_dir), **audit}, indent=2), flush=True)
    # Official unmodified train.py uses ~/.cache/jax; entrypoint HOME is a persistent mount.
    import train
    train.main(config)
    final = config.checkpoint_dir / str(config.num_train_steps - 1)
    committed = json.loads((final / "_CHECKPOINT_METADATA").read_text())
    if not committed.get("commit_timestamp_nsecs"):
        raise RuntimeError("Final checkpoint was not committed")
    if sha256(final / "assets" / h.ASSET_ID / "norm_stats.json") != h.NORM_SHA256:
        raise RuntimeError("Saved normalization differs from training input")
    completion = {"status": "TRAINING_COMPLETED", "steps": args.num_train_steps,
                  "checkpoint": str(final), "completed_at": time.time(), **audit}
    (run_dir / "completed.json").write_text(json.dumps(completion, indent=2) + "\n")
    if args.smoke:
        smoke_path.write_text(json.dumps({"status": "SMOKE_PASSED", "profile": profile,
            "experiment": args.exp_name, "steps": args.num_train_steps,
            "checkpoint": str(final), "completed_at": time.time()}, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Independent H-Bot v3 adapter. Never writes the source data or frozen bundle."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader

CAMERAS = ("observation.images.ext1_camera", "observation.images.wrist_camera",
           "observation.images.ext2_camera")
SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
HORIZON = 15
ASSET = "hbot_panda_joint_delta_h15"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_plan(root, seed=20261106):
    root = Path(root)
    bundle = root / "datasets/processed/hbot_sampling_v3_quality_20261107"
    script = root / "scripts/hbot_training_reviewed_v3/hbot_sampling.py"
    expected = read_json(bundle / "implementation.json")["hbot_sampling.py"]
    require(sha(script) == expected, "Reviewed sampler implementation changed")
    spec = importlib.util.spec_from_file_location("reviewed_hbot_sampling_v3", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.SamplingPlan(bundle, profile="stage", seed=seed)
    require(plan.config["action_horizon"] == HORIZON, "Frozen horizon mismatch")
    return plan, bundle


def contract(root, plan, bundle, microbatch, accumulation):
    return dict(schema=1, bundle_checksums_sha256=sha(bundle / "checksums.json"),
                sampler_sha256=sha(Path(root) / "scripts/hbot_training_reviewed_v3/hbot_sampling.py"),
                normalization_sha256=sha(bundle / ASSET / "norm_stats.json"),
                seed=plan.seed, profile=plan.profile, camera_order=list(CAMERAS),
                slots=list(SLOTS), state_dim=8, action_dim=8, padded_action_dim=32,
                horizon=HORIZON, joint_delta_dims=list(range(7)), gripper="absolute",
                normalization="q01/q99, denominator +1e-6, no clipping",
                state_tokens="8 normalized values, 256 bins, before padding",
                image_resize="PIL bilinear aspect-preserving zero pad 224",
                task_labels_in_prompt=False, loss_dims=8,
                microbatch=microbatch, accumulation=accumulation,
                effective_batch=microbatch * accumulation,
                sampler_address="optimizer_step -> global_batch(effective_batch), then split",
                parameter_storage="float32", optimizer_moments="float32", compute="bf16 autocast",
                augmentation="none; source RGB only", world_size=1)


class UpdateBatchSampler:
    """Address by committed optimizer step, invariant to prefetch or accumulation."""
    def __init__(self, plan, microbatch, accumulation, start, end):
        require(microbatch > 0 and accumulation > 0 and 0 <= start < end,
                "Invalid batch/step bounds")
        self.plan, self.microbatch, self.accumulation = plan, microbatch, accumulation
        self.start, self.end = start, end

    def __len__(self):
        return (self.end - self.start) * self.accumulation

    def __iter__(self):
        for step in range(self.start, self.end):
            full = self.plan.global_batch(step, self.microbatch * self.accumulation)
            for part in np.split(full, self.accumulation):
                yield part.tolist()


class HBotTransform:
    def __init__(self, root, bundle):
        self.root, self.bundle = str(root), str(bundle)
        self.stats = read_json(Path(bundle) / ASSET / "norm_stats.json")["norm_stats"]
        self._tokenizer = None

    def norm(self, x, key):
        q0, q1 = [np.asarray(self.stats[key][q], dtype=np.float64) for q in ("q01", "q99")]
        return (np.asarray(x) - q0) / (q1 - q0 + 1e-6) * 2 - 1

    def decode_actions(self, normalized, state):
        q0, q1 = [np.asarray(self.stats["actions"][q], dtype=np.float64) for q in ("q01", "q99")]
        out = (np.asarray(normalized)[..., :8] + 1) / 2 * (q1 - q0 + 1e-6) + q0
        out[..., :7] += np.asarray(state)[..., None, :7]
        return out

    def __call__(self, raw):
        state = np.asarray(raw["observation.state"], dtype=np.float32)
        actions = np.asarray(raw["action"], dtype=np.float32).copy()
        require(state.shape == (8,) and actions.shape == (HORIZON, 8), "Wrong robot dimensions")
        require(np.isfinite(state).all() and np.isfinite(actions).all(), "Non-finite robot values")
        actions[:, :7] -= state[None, :7]
        normalized = self.norm(actions, "actions")
        padded = np.pad(normalized, ((0, 0), (0, 24))).astype(np.float32)
        discretized = np.digitize(self.norm(state, "state"), np.linspace(-1, 1, 257)[:-1]) - 1
        task = raw["task"].strip().replace("_", " ").replace("\n", " ")
        require(bool(task), "Missing instruction; fallback prompts are prohibited")
        prompt = f"Task: {task}, State: {' '.join(map(str, discretized))};\nAction: "
        if self._tokenizer is None:
            import sentencepiece as spm
            self._tokenizer = spm.SentencePieceProcessor(
                model_file=str(Path(self.root) / "models/paligemma-3b-pt-224/tokenizer.model"))
        tokens = self._tokenizer.encode(prompt, add_bos=True)
        require(len(tokens) <= 200, "Prompt would be truncated")
        n = len(tokens)
        images = []
        for camera in CAMERAS:
            image = np.asarray(raw[camera])
            require(image.ndim == 3 and image.shape[0] == 3, "Expected CHW RGB")
            image = np.clip(image.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            im = Image.fromarray(image)
            ratio = max(im.width / 224, im.height / 224)
            size = (int(im.width / ratio), int(im.height / ratio))
            resized = im.resize(size, resample=Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (224, 224), 0)
            canvas.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
            images.append(torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.)
        return dict(images_list=images, img_masks=[True]*3,
                    input_ids=torch.tensor(tokens + [0]*(200-n), dtype=torch.long),
                    attention_mask=torch.tensor([True]*n + [False]*(200-n)),
                    action=torch.from_numpy(padded))


class ReviewedDataset(Dataset):
    def __init__(self, root, plan, bundle):
        from loongforge.embodied.data.datasets.lerobot_dataset import LeRobotV2Dataset
        source = Path(root) / "datasets/InstructMove_hbot/lerobot_unpacked"
        self.eligible = plan.arrays["eligible"].copy()
        self.transform = HBotTransform(root, bundle)
        parts = []
        for name in plan.config["dataset_order"]:
            ds = LeRobotV2Dataset(source / name, action_horizon=HORIZON, video_backend="pyav")
            require(ds.fps == 30, "FPS changed")
            require(ds.info["features"]["action"]["shape"] == [8], "Action schema changed")
            require(ds.info["features"]["observation.state"]["shape"] == [8], "State schema changed")
            require(set(ds._video_keys) == set(CAMERAS), "Camera schema changed")
            expected = [e for e in plan.episodes if e["dataset"] == name]
            require([(e["episode_index"], e["length"]) for e in ds._episodes] ==
                    [(e["episode_index"], e["length"]) for e in expected], "Episode ordering changed")
            parts.append(ds)
        plan.validate_dataset_order(plan.config["dataset_order"], [len(ds) for ds in parts])
        self.raw = ConcatDataset(parts)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        require(bool(self.eligible[index]), f"Excluded anchor requested: {index}")
        return self.transform(self.raw[index])


def collate(examples):
    from loongforge.embodied.data.datasets.pi05.transforms.pi05_collator import Pi05Preprocessor
    return Pi05Preprocessor(num_images=3)(examples)


def worker_init(_):
    torch.set_num_threads(1)


def loader(dataset, plan, *, microbatch, accumulation, start, end, workers=2):
    sampler = UpdateBatchSampler(plan, microbatch, accumulation, start, end)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                      num_workers=workers, pin_memory=True, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(plan.seed + 17),
                      **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}))

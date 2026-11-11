"""H-Bot reviewed-v3 inference contract. No training imports or default processors."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import sentencepiece as spm
from robo_orchard_server.policy.pi05.adapter import Pi05Adapter
from robo_orchard_server.server.policy import BasePolicy

CAMERAS = ('ext1_camera', 'wrist_camera', 'ext2_camera')
ARM_NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))
GRIPPER_NAMES = ('finger_joint',)
GRIPPER_SCALE = 4.0 / math.pi
BASE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class PandaAdapter(Pi05Adapter):
    def __init__(self):
        super().__init__({c: ['observation.images.'+c] for c in CAMERAS},
                         ['observation.state'], ['prompt'])

    def layout(self, obs):
        specs = super().layout(obs)
        require(obs['action_layout']['embodiment_type'] == 'panda_droid' and len(specs) == 1,
                'Only the training panda_droid single-arm embodiment is supported')
        s = specs[0]
        require(tuple(s['arm_joint_names']) == ARM_NAMES and
                tuple(s['gripper_joint_names']) == GRIPPER_NAMES and
                s['gripper_policy_representation'] == 'first_joint' and
                s['gripper_decode_coupling'] == 'symmetric' and
                math.isclose(s['gripper_policy_scale'], GRIPPER_SCALE, rel_tol=1e-7),
                'Panda-DROID requires joints 1..7 and finger_joint, first_joint/symmetric, scale 4/pi: '+str(s))
        return specs


class Codec:
    def __init__(self, stats_file, contract_file, tokenizer_file):
        self.contract = read(contract_file)
        expected = read(BASE/'fixtures/hbot_contract.json')
        keys = ('camera_order', 'slots', 'state_dim', 'action_dim', 'padded_action_dim', 'horizon',
                'joint_delta_dims', 'gripper', 'normalization', 'state_tokens', 'image_resize',
                'task_labels_in_prompt', 'parameter_storage', 'compute')
        require(all(self.contract.get(k) == expected[k] for k in keys), 'Checkpoint data contract mismatch')
        require(sha(stats_file) == self.contract['normalization_sha256'], 'Normalization hash mismatch')
        self.stats = read(stats_file)['norm_stats']
        for key in ('state', 'actions'):
            q0, q1 = (np.asarray(self.stats[key][q], dtype=np.float64) for q in ('q01', 'q99'))
            require(q0.shape == q1.shape == (8,) and np.isfinite([q0, q1]).all() and
                    np.all(q1 >= q0), 'Invalid eight-dimensional quantiles')
        provenance = read(BASE/'provenance.json')
        require(sha(tokenizer_file) == provenance['fixtures/tokenizer.model']['sha256'],
                'Tokenizer differs from training')
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_file))

    def norm(self, x, key):
        q0, q1 = (np.asarray(self.stats[key][q], dtype=np.float64) for q in ('q01', 'q99'))
        return (np.asarray(x)-q0)/(q1-q0+1e-6)*2-1

    def encode(self, raw):
        state = np.asarray(raw['observation.state'], dtype=np.float32)
        require(state.shape == (8,) and np.isfinite(state).all(), 'Expected finite state[8]')
        task = raw['prompt']
        require(isinstance(task, str), 'Instruction must be text')
        task = task.strip().replace('_', ' ').replace('\n', ' ')
        require(bool(task), 'Missing instruction; no fallback prompt')
        bins = np.digitize(self.norm(state, 'state'), np.linspace(-1, 1, 257)[:-1])-1
        prompt = f"Task: {task}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        require(len(tokens) <= 200, 'Prompt exceeds 200 tokens; refusing silent truncation')
        n = len(tokens)
        images = []
        for camera in CAMERAS:
            image = np.asarray(raw['observation.images.'+camera])
            require(image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8 and
                    min(image.shape[:2]) > 0, 'Expected uint8 HWC RGB')
            # Match dataset float32 RGB -> uint8 truncation used by HBotTransform.
            image = np.clip((image.astype(np.float32)/255)*255, 0, 255).astype(np.uint8)
            im = Image.fromarray(image)
            ratio = max(im.width/224, im.height/224)
            size = (int(im.width/ratio), int(im.height/ratio))
            require(min(size) > 0, 'Degenerate image aspect ratio')
            canvas = Image.new('RGB', (224, 224), 0)
            canvas.paste(im.resize(size, resample=Image.Resampling.BILINEAR),
                         ((224-size[0])//2, (224-size[1])//2))
            images.append(np.array(canvas).transpose(2, 0, 1).astype(np.float32)/255)
        return dict(state=state.copy(), images=images,
                    input_ids=np.asarray(tokens+[0]*(200-n), dtype=np.int64),
                    attention_mask=np.asarray([True]*n+[False]*(200-n)))

    def decode(self, normalized, state):
        values = np.asarray(normalized)
        require(values.shape == (15, 32) and np.isfinite(values).all(), 'Invalid normalized chunk')
        q0, q1 = (np.asarray(self.stats['actions'][q], dtype=np.float64) for q in ('q01', 'q99'))
        out = (values[:, :8]+1)/2*(q1-q0+1e-6)+q0
        # Every row is relative to the same observed qpos, NOT an accumulated velocity.
        out[:, :7] += np.asarray(state)[None, :7]
        require(np.isfinite(out).all(), 'Decoded non-finite action')
        return out.astype(np.float32)


class SessionPolicy(BasePolicy):
    """Per-Client cache and CPU RNG; model weights/lock alone are shared."""
    def __init__(self, engine, codec, *, action_steps=15, seed=20261106):
        require(type(action_steps) is int and 1 <= action_steps <= 15, 'Action steps must be 1..15')
        import torch
        self.engine, self.codec = engine, codec
        self.adapter = PandaAdapter()
        self.action_steps, self.seed = action_steps, seed
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self._cache, self._layout = [], None
        self._lock = threading.Lock()

    def new_session(self):
        return SessionPolicy(self.engine, self.codec, action_steps=self.action_steps, seed=self.seed)

    def reset(self):
        with self._lock:
            self._cache.clear()  # RNG progression retained; never shared across clients.

    def close(self):
        self.reset()

    def _check_layout(self, obs):
        self.adapter.layout(obs)
        if self._layout is None:
            self._layout = copy.deepcopy(obs['action_layout'])
            print('CLIENT_LAYOUT_VALIDATED '+json.dumps(self._layout), flush=True)
        require(self._layout == obs['action_layout'], 'Client action layout changed')

    def _sequence(self, obs):
        self._check_layout(obs)
        encoded = self.codec.encode(self.adapter.build_model_input(obs))
        normalized = self.engine.predict(encoded, self.rng)
        actions = self.codec.decode(normalized, encoded['state'])
        self._cache.clear()
        return self.adapter.build_action_sequence(actions, obs, self.action_steps)

    def act_sequence(self, obs):
        with self._lock:
            return self._sequence(obs)

    def act(self, obs):
        with self._lock:
            self._check_layout(obs)
            if not self._cache:
                self._cache = self._sequence(obs)
            return self._cache.pop(0)


def example_observation():
    spec = dict(arm_joint_names=list(ARM_NAMES), gripper_joint_names=list(GRIPPER_NAMES),
                gripper_policy_representation='first_joint', gripper_decode_coupling='symmetric',
                gripper_policy_scale=GRIPPER_SCALE)
    return dict(instruction='pick up the red cube',
        cameras={c: {'rgb': np.full((1, 48, 64, 3), 30+i*70, np.uint8)} for i, c in enumerate(CAMERAS)},
        manipulators={'single_arm': {'joint_position': np.array([[0, -.6, 0, -2, 0, 1.8, 0]], np.float32),
                                    'gripper_position': np.array([[math.pi/8]], np.float32)}},
        action_layout=dict(embodiment_type='panda_droid', schema_version='1',
                           manipulator_order=['single_arm'], manipulators={'single_arm': spec}))

"""Strict loader for this formal run. Never use load_pretrained key remapping."""
import json
from pathlib import Path
import struct
import threading
import time
from importlib.metadata import version

from hbot_inference import BASE, require, read, sha

def verify_runtime():
    expected = {'torch': '2.10.0', 'transformers': '5.3.0', 'numpy': '2.2.6',
                'Pillow': '13.0.0', 'sentencepiece': '0.2.1', 'safetensors': '0.8.0'}
    for package, ver in expected.items():
        require(version(package).split('+')[0] == ver, 'Runtime version mismatch: '+package)
    for rel, info in read(BASE/'provenance.json').items():
        if rel.startswith('vendor/'):
            require(sha(BASE/rel) == info['sha256'], 'Pinned source changed: '+rel)

def check_model_config(config):
    require(config == read(BASE/'fixtures/model_config.json'), 'Unsupported model configuration')

def core_model(meta=False):
    import torch
    from loongforge.embodied.model.pi05.modeling_pi05 import PI05Pytorch, _PI05InternalConfig
    from transformers.initialization import no_init_weights
    cfg = _PI05InternalConfig(chunk_size=15, n_action_steps=15, dtype='float32',
                              gradient_checkpointing=False, compile_model=False)
    # CPU construction keeps generated nonpersistent RoPE/position buffers valid.
    # no_init_weights avoids wasting time initializing parameters that will be loaded.
    if meta:
        with torch.device('meta'):
            return PI05Pytorch(cfg)
    with no_init_weights():
        return PI05Pytorch(cfg)

def strip_wrapper(state):
    require(all(k.startswith('model.') for k in state), 'Expected canonical model.* keys only')
    return {k[len('model.'):]: v for k, v in state.items()}

def check_shapes(header):
    model = core_model(meta=True)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    header = {k: v for k, v in header.items() if k != '__metadata__'}
    actual = strip_wrapper(header)
    require(set(actual) == set(expected), 'Checkpoint parameter key set differs from model')
    for name, shape in expected.items():
        require(tuple(actual[name]['shape']) == shape and actual[name]['dtype'] == 'F32',
                'Parameter shape or dtype mismatch: '+name)
    return dict(parameter_tensors=len(expected), parameter_elements=sum(v.numel() for v in model.parameters()))

def check_checkpoint(folder, *, hash_weights=True):
    folder = Path(folder).resolve(strict=True)
    require(not folder.name.startswith('.'), 'Pending checkpoint cannot be served')
    marker = read(folder/'verified.json')
    require(marker['status'] == 'VERIFIED' and marker['kind'] in ('inference_weights', 'full_state'),
            'Checkpoint must be verified by the formal trainer')
    require(marker['run_id'] == '8fc94f26b3da464c83d673010894b764' and marker['step'] > 0,
            'Wrong run or a test checkpoint; never serve acceptance-test weights')
    needed = ('model.safetensors', 'model_config.json', 'processor_config.json',
              'hbot_contract.json', 'norm_stats.json', 'resume_meta.json')
    for name in needed:
        p = folder/name
        require(p.is_file() and not p.is_symlink(), 'Missing or symbolic checkpoint file: '+name)
        expected = marker['files'][name]
        require(p.stat().st_size == expected['size'], 'Checkpoint size mismatch: '+name)
        if name != 'model.safetensors' or hash_weights:
            require(sha(p) == expected['sha256'], 'Checkpoint hash mismatch: '+name)
    require(read(folder/'resume_meta.json')['completed_steps'] == marker['step'], 'Step metadata mismatch')
    contract = read(folder/'hbot_contract.json')
    require(contract == read(BASE/'fixtures/hbot_contract.json'), 'Formal run contract changed')
    check_model_config(read(folder/'model_config.json'))
    require(read(folder/'processor_config.json') == read(BASE/'fixtures/processor_config.json'),
            'Processor metadata changed')
    with (folder/'model.safetensors').open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        require(size < 16*1024*1024, 'Invalid safetensors header length')
        header = json.loads(f.read(size))
    return {**check_shapes(header), 'step': marker['step'], 'weight_sha256': marker['files']['model.safetensors']['sha256']}

def load_exact_core(model, state):
    # Do NOT copy lm_head into embed_tokens: both have distinct saved values after SFT.
    result = model.load_state_dict(strip_wrapper(state), strict=True, assign=True)
    require(not result.missing_keys and not result.unexpected_keys, 'Strict loading failed')
    return model

class TorchEngine:
    def __init__(self, checkpoint, denoise_steps=10):
        import torch
        from safetensors.torch import load_file
        require(1 <= denoise_steps <= 100, 'Invalid denoising step count')
        self.denoise_steps = denoise_steps
        self.lock = threading.Lock()
        model = core_model()
        tensors = load_file(str(Path(checkpoint)/'model.safetensors'), device='cpu')
        self.model = load_exact_core(model, tensors).eval().requires_grad_(False)
        del tensors
        require(not any(t.is_meta for t in self.model.state_dict().values()), 'Meta weights remain')
        require(not any(b.is_meta for b in self.model.buffers()), 'Uninitialized generated buffers')
        # Keep the saved FP32 parameters and BF16 autocast convention used in training.
        self.model.to(device='cuda:0')
        self.last_latency_s = None

    def predict(self, encoded, generator):
        import torch
        with self.lock, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            start = time.perf_counter()
            images = [torch.from_numpy(x).unsqueeze(0).to('cuda:0')*2-1 for x in encoded['images']]
            masks = [torch.ones(1, dtype=torch.bool, device='cuda:0') for _ in images]
            tokens = torch.from_numpy(encoded['input_ids']).unsqueeze(0).to('cuda:0')
            attention = torch.from_numpy(encoded['attention_mask']).unsqueeze(0).to('cuda:0')
            noise = torch.randn((1, 15, 32), generator=generator, dtype=torch.float32).to('cuda:0')
            result = self.model.sample_actions(images, masks, tokens, attention, noise=noise,
                                               num_steps=self.denoise_steps)
            result = result[0].float().cpu().numpy()
            self.last_latency_s = time.perf_counter()-start
            return result

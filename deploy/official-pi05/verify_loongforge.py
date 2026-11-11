"""LoongForge import, frozen-source and meta-shape checks; never loads real weights."""
import json
import os
from pathlib import Path

from backends import enable_loongforge
from official_server import verify_frozen

def main():
    root = enable_loongforge()
    from model_runtime import verify_runtime, check_shapes
    from hbot_inference import Codec, read
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Installer checks must hide GPUs')
    verify_frozen()
    verify_runtime()
    tokenizer = Path(os.environ['OPENPI_DATA_HOME']) / 'big_vision/paligemma_tokenizer.model'
    Codec(root/'fixtures/norm_stats.json', root/'fixtures/hbot_contract.json', tokenizer)
    shapes = check_shapes(read(root/'fixtures/checkpoint_header.json'))
    report = dict(status='LOONGFORGE_INSTALL_CPU_PASSED', **shapes, torch=torch.__version__,
                  cuda_build=torch.version.cuda, weights_loaded=False, gpu_inference_tested=False)
    (Path(os.environ['HBOT_RUNTIME'])/'installed-loongforge.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

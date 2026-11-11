"""Check pinned imports and protocol provenance without initializing a GPU."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

EXPECTED = {
    'jax': '0.5.3', 'jaxlib': '0.5.3', 'flax': '0.10.2',
    'orbax-checkpoint': '0.11.13', 'tensorstore': '0.1.74', 'ml-dtypes': '0.4.1',
    'torch': '2.7.1', 'numpy': '1.26.4', 'websockets': '15.0.1',
    'pyarrow': '20.0.0', 'av': '17.0.0', 'pyyaml': '6.0.2',
    'transformers': '4.53.2', 'sentencepiece': '0.2.0',
}

def main():
    versions = {name: metadata.version(name) for name in EXPECTED}
    mismatches = {name: version for name, version in versions.items()
                  if version.split('+')[0] != EXPECTED[name]}
    if mismatches:
        raise RuntimeError(f'Runtime differs from tested pins: {mismatches}')
    import pyarrow.parquet  # Keep native import ordering.
    from official_server import verify_frozen
    from pi05_server_config import example_observation, make_adapter
    import yaml
    import jax
    from openpi.models.tokenizer import PaligemmaTokenizer
    verify_frozen()
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('Installation checks must be CPU-only')
    package = Path(__file__).resolve().parent
    settings = yaml.safe_load((package / 'model.yaml').read_text())['hbot-full']
    raw = make_adapter('hbot-full', settings).build_model_input(example_observation())
    assert raw['observation.state'].shape == (8,)
    assert PaligemmaTokenizer(200).tokenize('Pick up the red cube.')[0].shape == (200,)
    tokenizer = Path(os.environ['OPENPI_DATA_HOME']) / 'big_vision/paligemma_tokenizer.model'
    report = {'status': 'INSTALL_CPU_IMPORTS_PASSED', 'versions': versions,
              'tokenizer_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
              'weights_loaded': False, 'gpu_inference_tested': False}
    (Path(os.environ['HBOT_RUNTIME']) / 'installed.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

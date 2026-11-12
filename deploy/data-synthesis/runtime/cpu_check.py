"""Container build-time imports; no GPU allocation or simulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import importlib
from importlib.metadata import version
import json

for module in ('torch', 'warp', 'numpy', 'scipy', 'pyarrow.parquet', 'av',
               'robo_orchard_core', 'robo_orchard_schemas', 'pyvirtualdisplay',
               'mcap.reader', 'mcap_protobuf.decoder',
               'lerobot.datasets.lerobot_dataset',
               'tools.mcap_to_lerobot.convert'):
    importlib.import_module(module)
assert version('torch') == '2.7.0+cu128', 'Unexpected torch replacement'
assert version('lerobot') == '0.3.3', 'Wrong converter reader'
print(json.dumps({'cpu_imports': 'passed', 'gpu_tested': False,
                  'torch': version('torch'), 'lerobot': version('lerobot')}))

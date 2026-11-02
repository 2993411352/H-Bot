#!/usr/bin/env bash
# CPU-only environment validation: no model download, training, or simulator.
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
hbot_proxy_off
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
venv="$HBOT_HOME/robo_orchard_lab/.venv"
uv pip check --python "$venv/bin/python"
"$venv/bin/python" - <<'PY'
import sys
from importlib.metadata import version
import numpy as np
import pyarrow as pa
import datasets
import pynvml
import torch
import torchvision
import pytorch3d._C
import robo_orchard_lab
from robo_orchard_lab.models.holobrain_0_5 import HoloBrainInferencePipeline
from robo_orchard_server.policy.holobrain.main import load_policy_config
from pytorch3d.transforms import quaternion_to_matrix
assert sys.prefix == '/study/wangmengfan/hbot/robo_orchard_lab/.venv'
assert 'openpi/.venv' not in '\n'.join(sys.path)
assert torch.version.cuda == '12.8', torch.version.cuda
assert pa.array(np.array([1, 2, 3])).to_pylist() == [1, 2, 3]
assert datasets.Dataset.from_dict({'x': [1, 2]})[1]['x'] == 2
assert torch.allclose(quaternion_to_matrix(torch.tensor([[1., 0., 0., 0.]])), torch.eye(3)[None])
assert torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5).tolist() == [0]
for name in ['torch', 'torchvision', 'transformers', 'numpy', 'pyarrow', 'datasets', 'nvidia-ml-py', 'pytorch3d', 'robo-orchard-core', 'robo-orchard-lab', 'robo-orchard-server']:
    print(name, version(name))
print('LOCAL_SOURCE', robo_orchard_lab.__file__)
print('POLICY_CONFIG', load_policy_config())
print('IMPORT_SMOKE_PASSED')
PY
"$venv/bin/python" -m robo_orchard_server.policy.holobrain.main --help
printf 'ENV_VALIDATION_PASSED %s\n' "$(date -Is)"

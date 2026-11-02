#!/usr/bin/env bash
set -euo pipefail
root=/study/wangmengfan/hbot
source "$root/config/hbot-env.sh"
export PYTHONPATH="$root/config:$root/scripts/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Allocate only what inference needs, instead of JAX's default preallocation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$root/logs"
exec > >(tee -a "$root/logs/pi05-server_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
cd "$root/openpi"
# No uv sync / dependency changes at launch, and no use of the HoloBrain venv.
exec "$root/openpi/.venv/bin/python" "$root/scripts/pi05-server.py" "$@"

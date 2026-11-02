#!/usr/bin/env bash
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
root=/study/wangmengfan/hbot
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME="$root/datasets/InstructMove_hbot/lerobot_unpacked"
export JAX_COMPILATION_CACHE_DIR="$root/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="$root/scripts/train${PYTHONPATH:+:$PYTHONPATH}"
cd "$root/openpi"
if [[ " $* " == *" --check "* || " $* " == *" --help "* || " $* " == *" -h "* ]]; then
  export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
else
  # Select GPU 0 by default; never stop another job to make room.
  gpu=${HBOT_TRAIN_GPU:-0}
  [[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'HBOT_TRAIN_GPU must be one GPU index.\n' >&2; exit 2; }
  mkdir -p "$root/tmp" "$root/logs"
  exec 9>"$root/tmp/pi05-train-gpu-$gpu.lock"
  flock -n 9 || { printf 'Another pi05 launch already holds this GPU lock.\n' >&2; exit 1; }
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || exit 1
  if (( used > 256 )); then
    printf 'GPU %s already uses %s MiB. Refusing to start; existing jobs are untouched.\n' "$gpu" "$used" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda
  log="$root/logs/pi05_lora_$(date +%Y%m%d_%H%M%S)_$$.log"
  exec > >(tee -a "$log") 2>&1
  printf 'Log: %s\n' "$log"
fi
# Direct .venv invocation avoids uv resolving/updating dependencies at launch.
exec "$root/openpi/.venv/bin/python" "$root/scripts/train/train-pi05-hbot.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
root=/study/wangmengfan/hbot
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$root/scripts/train${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/tmp"
exec 9>"$root/tmp/pi05-data-preparation.lock"
flock -n 9 || { printf 'Data preparation is already running.\n' >&2; exit 1; }
cd "$root/openpi"
exec "$root/openpi/.venv/bin/python" "$root/scripts/train/prepare-pi05-hbot.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
export HOME=/output/cache/home
export XDG_CACHE_HOME=/output/cache/xdg
export HF_HOME=/output/cache/huggingface
export OPENPI_DATA_HOME=/output/cache/openpi
export TMPDIR=/output/tmp
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DIR=/output/logs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
mkdir -p "$HOME/.cache/jax" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$WANDB_DIR" \
  "$OPENPI_DATA_HOME/big_vision" /output/checkpoints
tokenizer="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [[ ! -e "$tokenizer" ]]; then
  cp /opt/hbot-sft/resources/big_vision/paligemma_tokenizer.model "$tokenizer"
fi
[[ "$(sha256sum "$tokenizer" | cut -d ' ' -f 1)" == \
  8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 ]] || {
  echo 'Tokenizer checksum mismatch; refusing to overwrite it' >&2; exit 2;
}
if [[ "${1:-}" == bash ]]; then exec "$@"; fi
# Exclusive lock protects this output tree; it does not manage unrelated jobs.
exec flock --no-fork --nonblock /output/.training.lock \
  python /opt/hbot-sft/scripts/train/train_full.py "$@"

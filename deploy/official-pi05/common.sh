#!/usr/bin/env bash
# Source this file from the launchers; local.env is trusted local shell code.
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$package/local.env" ]]; then
  source "$package/local.env"
fi
export HBOT_RUNTIME="${HBOT_RUNTIME:-$package/runtime}"
[[ "$HBOT_RUNTIME" = /* ]] || { echo 'HBOT_RUNTIME must be an absolute path' >&2; return 2; }
export HBOT_HOME="$HBOT_RUNTIME"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HBOT_RUNTIME/cache}"
export HBOT_BACKEND="${HBOT_BACKEND:-jax}"
# A CLI backend flag must select the matching Python before importing either ML stack.
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  case "${arguments[i]}" in
    --backend)
      [[ $((i+1)) -lt ${#arguments[@]} ]] || { echo '--backend needs a value' >&2; return 2; }
      HBOT_BACKEND="${arguments[i+1]}" ;;
    --backend=*) HBOT_BACKEND="${arguments[i]#--backend=}" ;;
  esac
done
case "$HBOT_BACKEND" in
  jax) runtime_python="$HBOT_RUNTIME/openpi/.venv/bin/python" ;;
  loongforge) runtime_python="$HBOT_RUNTIME/loongforge/.venv/bin/python" ;;
  *) echo 'HBOT_BACKEND must be jax or loongforge' >&2; return 2 ;;
esac
export HBOT_PYTHON="${HBOT_PYTHON:-$runtime_python}"
export HBOT_GPU="${HBOT_GPU:-0}"
export HBOT_OFFICIAL_NETWORK="${HBOT_OFFICIAL_NETWORK:-direct}"
export PYTHONPATH="$package:$package/vendor"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

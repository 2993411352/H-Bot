#!/usr/bin/env bash
# Opt-in performance runner. Does not stop the model or alter host power settings.
set -euo pipefail
root=/study/wangmengfan/hbot
container=hbot-sim
inside=/home/users/wangmengfan
config=${1:-hbot-local-pi05.yaml}
workers=${2:-1}
threads=${3:-8}
record=${4:-2}
label=${5:-speed}
affinity=${HBOT_SIM_CPU_AFFINITY:-}
cpu_command=()
if [[ -n "$affinity" ]]; then
  [[ "$affinity" =~ ^[0-9,-]+$ ]] || { printf 'Invalid CPU affinity\n' >&2; exit 2; }
  cpu_command=(taskset -c "$affinity")
fi
if (( $# > 5 )) || [[ ! "$config" =~ ^[a-zA-Z0-9_-]+\.yaml$ ]] ||
   [[ ! "$workers" =~ ^[12]$ ]] || [[ ! "$threads" =~ ^[0-9]+$ ]] ||
   (( threads < 1 || threads > 32 )) || [[ ! "$record" =~ ^[0-9]+$ ]] ||
   (( record > 10 )) || [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'Usage: bash %s [config.yaml] [workers:1|2] [threads:1..32] [record:0..10] [label]\n' "$0" >&2
  exit 2
fi
test -f "$root/config/$config"
test "$(docker inspect -f '{{.State.Running}}' "$container")" = true
# Refuse to compete with an existing GPU0 workload. GPU1 model is expected.
gpu0_pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$gpu0_pids" ]]; then
  printf 'GPU0 is already in use; no new evaluation started. PIDs: %s\n' "$gpu0_pids" >&2
  exit 1
fi
gpu0_free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ ! "$gpu0_free" =~ ^[0-9]+$ ]] || (( gpu0_free < workers * 8192 + 1024 )); then
  printf 'Insufficient free GPU0 memory for %s simulation workers.\n' "$workers" >&2
  exit 1
fi
run="local_${label}_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
mkdir -p "$root/logs"
exec > >(tee -a "$root/logs/$run.log") 2>&1
printf 'Run: %s\nOutput: %s/outputs/sim/%s\n' "$run" "$root" "$run"
printf 'GPU0 simulation workers=%s; per-process CPU threads=%s; recordings/task=%s\n' "$workers" "$threads" "$record"
printf 'CPU affinity=%s; safety stop=90C; GPU power limits unchanged\n' "${affinity:-unchanged}"
cli_pid=
completed=0
cleanup() {
  local rc=$?
  trap - EXIT
  trap '' INT TERM HUP
  if (( ! completed )) && [[ -n "$cli_pid" ]]; then
    docker exec "$container" python3 "$inside/scripts/hbot-eval-control.py" stop "$run" || true
    wait "$cli_pid" || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
docker exec \
  -e HBOT_EVAL_PROFILE=1 -e HBOT_EVAL_NUM_RECORD="$record" \
  -e HBOT_EVAL_DEEP_PROFILE=1 \
  -e HBOT_SIM_WORKERS_PER_GPU="$workers" -e HBOT_SIM_CPU_THREADS="$threads" \
  -e OMP_NUM_THREADS="$threads" -e OPENBLAS_NUM_THREADS="$threads" \
  -e MKL_NUM_THREADS="$threads" -e PXR_WORK_THREAD_LIMIT="$threads" \
  -w "$inside/robo_orchard_sim" "$container" \
  "${cpu_command[@]}" python3 "$inside/scripts/hbot-eval-control.py" run "$run" "$config" &
cli_pid=$!
while kill -0 "$cli_pid" 2>/dev/null; do
  sleep 5
  if ! kill -0 "$cli_pid" 2>/dev/null; then break; fi
  # Monitor both physical GPUs, but stop only this run on a safety trip.
  readings=$(nvidia-smi --query-gpu=index,temperature.gpu,memory.free --format=csv,noheader,nounits)
  printf '[resource] %s %s\n' "$(date -Is)" "${readings//$'\n'/ | }"
  critical=0
  while IFS=, read -r gpu temperature free_mib; do
    temperature=${temperature// /}
    free_mib=${free_mib// /}
    [[ "$temperature" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] || exit 1
    (( temperature < 90 )) || critical=1
    # Avoid a near-OOM condition on the simulation device.
    if [[ "${gpu// /}" == 0 ]] && (( free_mib < 768 )); then critical=1; fi
  done <<< "$readings"
  if (( critical )); then
    printf 'SAFETY_STOP: GPU temperature or memory limit; model service is preserved.\n' >&2
    exit 1
  fi
done
status=0
wait "$cli_pid" || status=$?
if (( status != 0 )); then exit "$status"; fi
completed=1
printf 'Completed: %s\n' "$run"

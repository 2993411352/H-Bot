#!/usr/bin/env bash
set -euo pipefail
mode=${1:---help}
if (( $# )); then shift; fi
if [[ "$mode" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/run-sft.sh check|smoke|train|resume|shell|logs|stop [trainer args]
Required: HBOT_INPUT_ROOT (existing model/data root), HBOT_OUTPUT_ROOT (new output root).
Optional: HBOT_GPU=0, HBOT_BATCH_SIZE=16, HBOT_STEPS=100000, HBOT_WORKERS=2,
          HBOT_EXP_NAME=pi05_full_b<BATCH>_<STEPS>s_v2,
          HBOT_IMAGE=hbot/pi05-full-sft:215abfb-a800-v2.
check: CPU only, model/data checksums, real video pipeline and full-parameter shapes.
smoke: foreground 20-step GPU run and checkpoint check; records evidence for this batch.
train/resume: requires matching smoke success; detached, independent of SSH connection.
logs/stop: operate only on the explicitly named package-owned experiment container.
HELP
  exit 0
fi
[[ "$mode" =~ ^(check|smoke|train|resume|shell|logs|stop)$ ]] || exit 2
image=${HBOT_IMAGE:-hbot/pi05-full-sft:215abfb-a800-v2}
batch=${HBOT_BATCH_SIZE:-16}
steps=${HBOT_STEPS:-100000}
workers=${HBOT_WORKERS:-2}
[[ "$batch" =~ ^[1-9][0-9]*$ && "$steps" =~ ^[1-9][0-9]*$ && "$workers" =~ ^[0-9]+$ ]] || exit 2
experiment=${HBOT_EXP_NAME:-pi05_full_b${batch}_${steps}s_v2}
[[ "$experiment" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe experiment name' >&2; exit 2; }
name="hbot-pi05-full-${experiment}"
if [[ "$mode" == logs || "$mode" == stop ]]; then
  label=$(docker inspect --format '{{index .Config.Labels "hbot.package"}}' "$name")
  [[ "$label" == pi05-full-sft-v2 ]] || { echo 'Not a v2 package-owned container' >&2; exit 2; }
  if [[ "$mode" == logs ]]; then exec docker logs --follow --tail 80 "$name"; fi
  exec docker stop --time 120 "$name"
fi
input=${HBOT_INPUT_ROOT:?Set HBOT_INPUT_ROOT}
output=${HBOT_OUTPUT_ROOT:?Set HBOT_OUTPUT_ROOT}
[[ "$input" == /* && "$output" == /* && "$output" != / ]] || exit 2
[[ -d "$input/models/pi05_base/params" && -d "$input/datasets/InstructMove_hbot/lerobot_unpacked" ]] || {
  echo 'Missing base params or unpacked LeRobot data; use copy-inputs.sh first' >&2; exit 2;
}
mkdir -p "$output"
input=$(realpath -- "$input")
output=$(realpath -- "$output")
[[ "$input" != "$output" && "$output" != "$input/"* && "$input" != "$output/"* ]] || {
  echo 'Keep input and output directories separate (not nested)' >&2; exit 2;
}
# Docker --mount uses commas as delimiters.
[[ "$input$output" != *,* ]] || { echo 'Mount paths cannot contain commas' >&2; exit 2; }
common=(--init --network none --shm-size 8g --user "$(id -u):$(id -g)"
  --security-opt no-new-privileges --cap-drop ALL
  --mount "type=bind,src=$input,dst=/input,readonly"
  --mount "type=bind,src=$output,dst=/output"
  --label hbot.package=pi05-full-sft-v2)
if [[ "$mode" == check ]]; then
  exec docker run --rm "${common[@]}" -e NVIDIA_VISIBLE_DEVICES=void \
    -e JAX_PLATFORMS=cpu -e CUDA_VISIBLE_DEVICES= \
    "$image" --check --batch-size "$batch" --num-workers "$workers" "$@"
fi
gpu=${HBOT_GPU:-0}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'HBOT_GPU must be ONE integer GPU index' >&2; exit 2; }
# Do not modify power or kill processes to make room. Refuse a busy/undersized GPU.
read -r total used < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.used \
  --format=csv,noheader,nounits | tr ',' ' ')
[[ "$total" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || exit 2
(( total >= 79000 && used < 1024 )) || {
  echo "Refusing: require an idle 80GB-class GPU; found total=${total}MiB used=${used}MiB" >&2; exit 2;
}
common+=(--gpus "device=$gpu" -e JAX_PLATFORMS=cuda -e CUDA_VISIBLE_DEVICES=0)
if [[ "$mode" == shell ]]; then
  exec docker run --rm -it "${common[@]}" "$image" bash
fi
if [[ "$mode" == smoke ]]; then
  smoke="smoke_$(date +%Y%m%d_%H%M%S)_$$"
  exec docker run --rm "${common[@]}" "$image" --smoke --exp-name "$smoke" \
    --batch-size "$batch" --num-workers "$workers" --num-train-steps 20 \
    --log-interval 1 --save-interval 20 --keep-period 100000 --lr-decay-steps "$steps" "$@"
fi
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists. Inspect its logs/state; remove ONLY this stopped container manually before resuming." >&2
  exit 2
fi
resume=()
[[ "$mode" != resume ]] || resume=(--resume)
docker run --detach --name "$name" "${common[@]}" "$image" \
  --exp-name "$experiment" --batch-size "$batch" --num-workers "$workers" \
  --num-train-steps "$steps" --lr-decay-steps "$steps" "${resume[@]}" "$@"
echo "Start requested (not proof of successful training): docker logs -f $name"
echo "Stop this job only: docker stop --time 120 $name"

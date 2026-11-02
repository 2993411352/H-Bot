#!/usr/bin/env bash
# One explicitly authorized test. Run with sudo; model training drops privileges.
set -euo pipefail
export LC_ALL=C
root=/study/wangmengfan/hbot
gpu_uuid=GPU-d32118cd-3b44-94bf-d286-79df4d357528
experiment=hbot_lora_b4_280w_v1
if [[ ${1:-} == --help ]]; then
  printf 'Usage: sudo bash %s\n' "$root/scripts/run-pi05-280w-test.sh"
  printf 'GPU 0: 350W -> 280W; batch 4; 100 steps; 90C cutoff; restore on exit.\n'
  printf 'Training runs as wangmengfan, not root. An independent timer restores power after 10 minutes as a fallback.\n'
  exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments.\n' >&2; exit 2; }
[[ $EUID == 0 ]] || { printf 'Requires sudo; no GPU setting was changed.\n' >&2; exit 1; }
for command in /usr/bin/nvidia-smi /usr/bin/systemd-run /usr/bin/systemctl /usr/sbin/runuser /usr/bin/timeout; do
  [[ -x "$command" ]] || { printf 'Missing %s; no GPU setting was changed.\n' "$command" >&2; exit 1; }
done
[[ -f "$root/scripts/train/benchmark-pi05-lora.py" && -x "$root/openpi/.venv/bin/python" ]]
[[ ! -e "$root/checkpoints/pi05_hbot_lora/$experiment" && ! -e "$root/logs/pi05_bench_$experiment" ]] || {
  printf 'Experiment already exists; refusing to overwrite or repeat.\n' >&2; exit 1;
}
# Keep two copies of this privileged helper from changing the same GPU together.
exec 8>"/run/lock/hbot-pi05-power-$gpu_uuid.lock"
flock -n 8 || { printf 'Another power test holds the lock.\n' >&2; exit 1; }
actual_uuid=$(/usr/bin/nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits)
[[ "$actual_uuid" == "$gpu_uuid" ]] || { printf 'GPU index/UUID mismatch; aborting.\n' >&2; exit 1; }
used=$(/usr/bin/nvidia-smi -i "$gpu_uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" =~ ^[0-9]+$ ]] && (( used <= 256 )) || { printf 'GPU is not idle; existing jobs are untouched.\n' >&2; exit 1; }
original_power=$(/usr/bin/nvidia-smi -i "$gpu_uuid" --query-gpu=power.limit --format=csv,noheader,nounits)
[[ "$original_power" =~ ^350(\.0+)?$ ]] || {
  printf 'Expected approved starting limit 350W, found %sW; aborting.\n' "$original_power" >&2; exit 1;
}
restore_unit="hbot-gpu0-power-restore-$(date +%Y%m%d%H%M%S)-$$"
power_change_attempted=0
timer_created=0
restore_power() {
  local status=$?
  trap - EXIT
  trap '' HUP INT TERM
  if (( power_change_attempted )); then
    printf 'Restoring GPU 0 power limit to %sW...\n' "$original_power"
    if /usr/bin/nvidia-smi -i "$gpu_uuid" -pl "$original_power" &&
       [[ $(/usr/bin/nvidia-smi -i "$gpu_uuid" --query-gpu=power.limit --format=csv,noheader,nounits) == "$original_power" ]]; then
      printf 'RESTORED_POWER_LIMIT_W=%s\n' "$original_power"
    else
      printf 'RESTORE FAILED: fallback timer retained; verify GPU power manually.\n' >&2
      exit 1
    fi
  fi
  if (( timer_created )); then
    /usr/bin/systemctl stop "$restore_unit.timer" || printf 'Warning: fallback timer could not be cancelled.\n' >&2
  fi
  exit "$status"
}
trap restore_power EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# Register fallback BEFORE changing power; if setup fails, do not change the GPU.
/usr/bin/systemd-run --quiet --collect --unit="$restore_unit" --on-active=10m \
  /usr/bin/nvidia-smi -i "$gpu_uuid" -pl "$original_power"
timer_created=1
printf 'ORIGINAL_POWER_LIMIT_W=%s GPU_UUID=%s FALLBACK_TIMER=%s.timer\n' "$original_power" "$gpu_uuid" "$restore_unit"
power_change_attempted=1
/usr/bin/nvidia-smi -i "$gpu_uuid" -pl 280
actual_power=$(/usr/bin/nvidia-smi -i "$gpu_uuid" --query-gpu=power.limit --format=csv,noheader,nounits)
[[ "$actual_power" =~ ^280(\.0+)?$ ]] || { printf '280W setting was not confirmed.\n' >&2; exit 1; }
printf 'TEST_POWER_LIMIT_W=%s; running training as wangmengfan.\n' "$actual_power"
# Timeout is shorter than the fallback timer. The benchmark handles TERM/INT and
# cleans up only its own training children. Neither shell nor training uses root
# to open dataset/checkpoint files.
set +e
/usr/sbin/runuser -u wangmengfan -- /usr/bin/env HBOT_HOME="$root" /usr/bin/timeout --foreground --signal=TERM --kill-after=30s 8m \
  "$root/openpi/.venv/bin/python" -u "$root/scripts/train/benchmark-pi05-lora.py" \
  --batch-size 4 --steps 100 --warmup-steps 10 --exp-name "$experiment" --gpu 0 --max-temperature-c 90
status=$?
set -e
printf 'BENCHMARK_EXIT_STATUS=%s\n' "$status"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
root=/study/wangmengfan/hbot
container=hbot-sim
container_root=/home/users/wangmengfan
controller="$container_root/scripts/hbot-eval-control.py"

case "${1:-}" in
  --status)
    exec docker exec "$container" python3 "$controller" status
    ;;
  --stop)
    if [[ $# != 2 || ! "$2" =~ ^local_[a-zA-Z0-9_-]+$ ]]; then
      printf 'Usage: bash %s --stop local_RUN_NAME\n' "$0" >&2
      exit 2
    fi
    exec docker exec "$container" python3 "$controller" stop "$2"
    ;;
esac
if (( $# > 2 )); then
  printf 'Usage: bash %s [label] [config.yaml] | --status | --stop RUN_NAME\n' "$0" >&2
  exit 2
fi
label=${1:-dummy}
config=${2:-hbot-local.yaml}
if [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'Invalid output label\n' >&2
  exit 2
fi
if [[ ! "$config" =~ ^[a-zA-Z0-9_-]+\.yaml$ ]]; then
  printf 'Invalid config filename\n' >&2
  exit 2
fi
if [[ ! -f "$root/config/$config" ]]; then
  printf 'Missing evaluation config: %s/config/%s\n' "$root" "$config" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
  printf 'Container %s is not running.\n' "$container" >&2
  exit 1
fi
docker exec "$container" test -r "$controller"
mkdir -p "$root/logs"
# Include a PID and random suffix to prevent output-directory collisions.
run_name="local_${label}_$(date +%Y%m%d_%H%M%S)_${BASHPID}_${RANDOM}"
exec > >(tee -a "$root/logs/$run_name.log") 2>&1
docker_pid=
launch_requested=0
completed=0

cleanup() {
  local status=$?
  trap - EXIT
  trap '' INT TERM HUP
  set +e
  if (( launch_requested && ! completed )); then
    printf '\n[%s] Cleaning up this evaluation only: %s\n' "$(date -Is)" "$run_name"
    # A cancellation marker also covers Ctrl+C during container-process startup.
    if ! timeout 45s docker exec "$container" python3 "$controller" stop "$run_name"; then
      printf 'Cleanup could not be confirmed. Retry: bash %s --stop %s\n' "$0" "$run_name" >&2
      (( status == 0 )) && status=1
    fi
    if [[ -n "$docker_pid" ]] && kill -0 "$docker_pid" 2>/dev/null; then
      # This is only our local docker CLI child, never the model or container.
      kill -TERM "$docker_pid" 2>/dev/null
    fi
    [[ -z "$docker_pid" ]] || wait "$docker_pid" 2>/dev/null
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '[%s] Starting local evaluation: %s\n' "$(date -Is)" "$run_name"
printf 'Host output: %s/outputs/sim/%s\n' "$root" "$run_name"
printf 'Recording: first 2 episodes per task; evaluation count comes from YAML.\n'
launch_requested=1
# Background + wait lets Bash handle Ctrl+C immediately and forward cancellation.
docker exec -w "$container_root/robo_orchard_sim" "$container" \
  python3 "$controller" run "$run_name" "$config" &
docker_pid=$!
status=0
wait "$docker_pid" || status=$?
if (( status != 0 )); then
  exit "$status"
fi
completed=1
printf '[%s] Local evaluation command completed: %s\n' "$(date -Is)" "$run_name"

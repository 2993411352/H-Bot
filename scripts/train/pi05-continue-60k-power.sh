#!/usr/bin/env bash
# Installed root-owned in /run by the sudo launcher; called only by systemd.
set -euo pipefail
export LC_ALL=C
[[ $EUID == 0 ]] || { echo 'Power helper requires root.' >&2; exit 1; }
[[ ${INVOCATION_ID:-} =~ ^[a-fA-F0-9]{32}$ ]] || { echo 'Missing systemd invocation ID.' >&2; exit 1; }
gpu=GPU-d32118cd-3b44-94bf-d286-79df4d357528
state=/run/hbot-pi05-continue-60k/gpu0-power.state
case ${1:-} in
  start)
    [[ ! -e "$state" ]] || { echo 'Unrestored power state exists; aborting.' >&2; exit 1; }
    [[ $(/usr/bin/nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits) == "$gpu" ]]
    used=$(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 256 )) || { echo 'GPU busy; refusing power change.' >&2; exit 1; }
    original=$(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader,nounits)
    [[ "$original" =~ ^350(\.0+)?$ ]] || { echo "Unexpected starting limit: $original W" >&2; exit 1; }
    # Record before the write, so ExecStopPost can recover even a partial failure.
    (set -o noclobber; umask 077; printf '%s %s\n' "$INVOCATION_ID" "$original" > "$state")
    /usr/bin/nvidia-smi -i "$gpu" -pl 280
    actual=$(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader,nounits)
    [[ "$actual" =~ ^280(\.0+)?$ ]]
    echo "POWER_SET_W=$actual ORIGINAL_POWER_W=$original INVOCATION=$INVOCATION_ID"
    ;;
  restore)
    [[ -e "$state" ]] || { echo 'No power change recorded for this run.'; exit 0; }
    read -r owner original < "$state"
    [[ "$owner" == "$INVOCATION_ID" ]] || { echo 'State belongs to another invocation; leaving it untouched.' >&2; exit 1; }
    [[ "$original" =~ ^350(\.0+)?$ ]] || { echo 'Invalid saved power limit.' >&2; exit 1; }
    /usr/bin/nvidia-smi -i "$gpu" -pl "$original"
    [[ $(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader,nounits) == "$original" ]]
    rm -- "$state"
    echo "RESTORED_POWER_LIMIT_W=$original INVOCATION=$INVOCATION_ID"
    ;;
  *) echo 'Expected start or restore.' >&2; exit 2 ;;
esac

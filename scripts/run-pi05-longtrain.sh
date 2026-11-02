#!/usr/bin/env bash
# Install an ephemeral, root-owned unit. Only power setup/restoration runs as root.
set -euo pipefail
export LC_ALL=C
root=/study/wangmengfan/hbot
unit=hbot-pi05-lora-40k.service
runtime=/run/hbot-pi05-longtrain
mode=${1:-start}
if [[ "$mode" == --help ]]; then
  echo "Usage: sudo bash $root/scripts/run-pi05-longtrain.sh [start|resume|stop]"
  echo "Status: systemctl status $unit --no-pager"
  echo 'GPU0 280W, batch4, 40000 steps, decay40000, save500, keep5000, cutoff90C.'
  echo 'Runs in background; SSH disconnect does not stop it. No auto-restart.'
  exit 0
fi
[[ $# -le 1 && "$mode" =~ ^(start|resume|stop)$ ]] || { echo 'Invalid arguments.' >&2; exit 2; }
[[ $EUID == 0 ]] || { echo 'Requires sudo; no settings were changed.' >&2; exit 1; }
# This host's /usr/bin/systemctl is a replacement that fails daemon-reload.
# Its original packaged binary is retained as mctes. Never replace system files;
# select only a root-owned executable matching the installed package manifest.
expected=$(awk '$2 == "bin/systemctl" || $2 == "usr/bin/systemctl" {print $1}' /var/lib/dpkg/info/systemd.md5sums)
[[ "$expected" =~ ^[a-f0-9]{32}$ ]] || { echo 'Cannot resolve package systemctl checksum.' >&2; exit 1; }
hbot_systemctl=
for candidate in /usr/bin/systemctl /usr/bin/mctes; do
  [[ -f "$candidate" && -x "$candidate" ]] || continue
  [[ $(stat -Lc '%u' "$candidate") == 0 ]] || continue
  digest=$(md5sum "$candidate")
  if [[ ${digest%% *} == "$expected" ]]; then
    hbot_systemctl=$candidate
    break
  fi
done
[[ -n "$hbot_systemctl" ]] || { echo 'No package-matched systemctl found; no service changes made.' >&2; exit 1; }
printf 'Using package-matched systemctl: %s\n' "$hbot_systemctl"
if [[ "$mode" == stop ]]; then
  "$hbot_systemctl" stop "$unit"
  /usr/bin/nvidia-smi -i 0 --query-gpu=power.limit,memory.used --format=csv,noheader
  exit 0
fi
state=$("$hbot_systemctl" show "$unit" --property=ActiveState --value)
[[ "$state" != active && "$state" != activating && "$state" != deactivating ]] || { echo 'Service already running/stopping; no changes made.' >&2; exit 1; }
[[ ! -e /etc/systemd/system/$unit && ! -e /lib/systemd/system/$unit ]] || { echo 'Persistent unit with this name exists; refusing to shadow it.' >&2; exit 1; }
[[ ! -e "$runtime/gpu0-power.state" ]] || { echo 'Pending power restoration; inspect state before restart.' >&2; exit 1; }
/usr/sbin/runuser -u wangmengfan -- /usr/bin/env HBOT_HOME="$root" \
  "$root/openpi/.venv/bin/python" "$root/scripts/train/supervise-pi05-longtrain.py" --preflight --mode="$mode"
install -d -o root -g root -m 0700 "$runtime"
install -o root -g root -m 0700 "$root/scripts/train/pi05-longtrain-power.sh" "$runtime/power.sh"
printf 'HBOT_LONGTRAIN_MODE=%s\n' "$mode" > "$runtime/job.env"
chmod 0600 "$runtime/job.env"
install -o root -g root -m 0644 "$root/scripts/train/hbot-pi05-lora-40k.service" "/run/systemd/system/$unit"
/usr/bin/systemd-analyze --generators=no --man=no verify "/run/systemd/system/$unit"
if ! "$hbot_systemctl" daemon-reload; then
  echo 'daemon-reload failed; training was not started.' >&2
  exit 1
fi
if ! "$hbot_systemctl" start "$unit"; then
  "$hbot_systemctl" status "$unit" --no-pager -l || true
  echo 'Service start failed; inspect status above and the service journal.' >&2
  exit 1
fi
if ! "$hbot_systemctl" is-active --quiet "$unit"; then
  "$hbot_systemctl" status "$unit" --no-pager -l || true
  echo 'Service did not remain active; do not assume training is running.' >&2
  exit 1
fi
echo "START_REQUEST_ACCEPTED: $unit; verify status and logs."
echo "Log: $root/logs/pi05_longtrain/hbot_pi05_lora_b4_40k_v1/train.log"
echo "Stop: sudo bash $root/scripts/run-pi05-longtrain.sh stop"
echo 'Power is restored by ExecStopPost after exit, thermal stop, failure or service stop.'

#!/usr/bin/env bash
# Administrator entry point. Never runs the user-owned proxy or probe as root.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID -eq 0 ]] || { printf 'Run with sudo.\n' >&2; exit 1; }
source_dir=/study/wangmengfan/hbot/config
install_dir=/etc/hbot-mihomo-bypass
service=/etc/systemd/system/hbot-mihomo-bypass.service
timer=/etc/systemd/system/hbot-mihomo-bypass.timer
ctl=/usr/bin/mctes

# This host's systemctl is a wrapper; mctes was verified as the packaged original.
[[ $(/usr/bin/md5sum "$ctl" | /usr/bin/awk '{print $1}') == 98e9e7c9213bbd9987b015b512382076 ]] || {
  printf 'Native systemctl identity changed; stop and re-inspect before installation.\n' >&2
  exit 1
}
if [[ ${1:-} == --remove ]]; then
  [[ -f "$install_dir/ensure.sh" && ! -L "$install_dir/ensure.sh" ]]
  "$ctl" disable --now hbot-mihomo-bypass.timer
  "$ctl" stop hbot-mihomo-bypass.service
  /bin/bash "$install_dir/ensure.sh" remove
  printf 'Maintenance stopped and the exact bypass removed. Root-owned files retained for recovery.\n'
  exit 0
fi
[[ $# -eq 0 ]] || exit 2
[[ $(/usr/bin/id -u wangmengfan) == 1013 ]]
for item in "$install_dir" "$service" "$timer"; do
  if [[ -e "$item" || -L "$item" ]]; then
    printf 'Install target already exists: %s; refusing to overwrite.\n' "$item" >&2
    exit 1
  fi
done
/bin/bash -n "$source_dir/ensure-hbot-mihomo-bypass.sh"
/usr/sbin/runuser -u wangmengfan -- /usr/bin/timeout -k 5s 15s \
  /usr/bin/python3 "$source_dir/probe-single-node.py" --check-config

# Install immutable-to-the-user copies. The root service never executes files in /study or home.
/usr/bin/install -d -o root -g root -m 0755 "$install_dir"
/usr/bin/install -o root -g root -m 0700 "$source_dir/ensure-hbot-mihomo-bypass.sh" "$install_dir/ensure.sh"
/usr/bin/install -o root -g root -m 0644 "$source_dir/hbot-mihomo-bypass.service" "$service"
/usr/bin/install -o root -g root -m 0644 "$source_dir/hbot-mihomo-bypass.timer" "$timer"
committed=0
rollback_on_failure() {
  if [[ "$committed" != 1 ]]; then
    "$ctl" disable --now hbot-mihomo-bypass.timer || true
    "$ctl" stop hbot-mihomo-bypass.service || true
    /bin/bash "$install_dir/ensure.sh" remove || {
      printf 'ROLLBACK_FAILED: report the complete output; do not delete other rules.\n' >&2
      return 1
    }
    printf 'Installation did not commit. Files retained; asset download settings unchanged.\n' >&2
  fi
}
trap rollback_on_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
"$ctl" daemon-reload
"$ctl" start hbot-mihomo-bypass.service
/bin/bash "$install_dir/ensure.sh" status
printf '\nValidate real HTTPS and an asset range through an isolated, unprivileged core:\n'
/usr/sbin/runuser -u wangmengfan -- /usr/bin/timeout -k 5s 55s \
  /usr/bin/python3 "$source_dir/probe-single-node.py"
"$ctl" enable --now hbot-mihomo-bypass.timer
"$ctl" is-active --quiet hbot-mihomo-bypass.timer
"$ctl" is-enabled --quiet hbot-mihomo-bypass.timer
committed=1
trap - EXIT INT TERM HUP
printf '\nBYPASS_INSTALLED_AND_VERIFIED\n'
printf 'Only UID 1013 -> 54.176.102.50:443 is exempted. A root-owned timer checks once per minute.\n'
printf 'No existing service restarted. Main Mihomo and asset download configurations are unchanged.\n'
printf 'Next: validate the main proxy, then switch the managed asset downloader.\n'

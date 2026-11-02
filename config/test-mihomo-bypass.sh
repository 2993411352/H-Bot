#!/usr/bin/env bash
# Run manually with sudo. The actual proxy probes run unprivileged as UID 1013.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
if [[ $EUID -ne 0 ]]; then
  printf 'Run this bounded diagnostic with sudo.\n' >&2
  exit 1
fi
[[ $(/usr/bin/id -u wangmengfan) == 1013 ]]
ipt=/usr/sbin/iptables-legacy
probe=/study/wangmengfan/hbot/config/probe-single-node.py
rule=(-p tcp -d 54.176.102.50/32 --dport 443 -m owner --uid-owner 1013
      -m comment --comment hbot-mihomo-single-node-test -j RETURN)
"$ipt" -w 5 -t nat -C OUTPUT -p tcp -j TP_OUT
"$ipt" -w 5 -t nat -C TP_OUT -j TP_RULE
"$ipt" -w 5 -t nat -C TP_RULE -p tcp -j REDIRECT --to-ports 52345
if "$ipt" -w 5 -t nat -C TP_OUT "${rule[@]}" 2>/dev/null; then
  printf 'An identical diagnostic exception already exists; refusing to modify it.\n' >&2
  exit 1
fi
/usr/sbin/runuser -u wangmengfan -- /usr/bin/timeout -k 5s 15s /usr/bin/python3 "$probe" --check-config
probe_once() {
  /usr/sbin/runuser -u wangmengfan -- /usr/bin/timeout -k 5s 55s /usr/bin/python3 "$probe"
}
printf '\nBASELINE: existing transparent-proxy path\n'
baseline_status=0
probe_once || baseline_status=$?
printf 'BASELINE_EXIT=%s\n' "$baseline_status"
if [[ $baseline_status == 0 ]]; then
  printf 'Baseline already works; no firewall change needed or performed.\n'
  exit 0
fi
if [[ $baseline_status != 1 ]]; then
  printf 'Probe infrastructure error, not a connectivity result; no firewall change performed.\n' >&2
  exit "$baseline_status"
fi

cleanup() {
  if "$ipt" -w 5 -t nat -C TP_OUT "${rule[@]}" 2>/dev/null; then
    "$ipt" -w 5 -t nat -D TP_OUT "${rule[@]}" || return 1
  fi
  # Verify via a successful chain listing, rather than treating a query error as absence.
  local remaining
  remaining=$("$ipt" -w 5 -t nat -S TP_OUT) || return 1
  if [[ "$remaining" == *hbot-mihomo-single-node-test* ]]; then
    printf 'CLEANUP FAILED: diagnostic exception remains; report this output.\n' >&2
    return 1
  fi
  printf 'CLEANUP_OK: temporary exception absent; original rules otherwise unchanged.\n'
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
"$ipt" -w 5 -t nat -I TP_OUT 1 "${rule[@]}"
printf '\nBYPASS: only UID 1013 -> 54.176.102.50:443\n'
bypass_status=0
probe_once || bypass_status=$?
printf 'BYPASS_EXIT=%s\n' "$bypass_status"
"$ipt" -w 5 -t nat -L TP_OUT -n -v -x --line-numbers
cleanup
trap - EXIT INT TERM HUP
printf '\nAFTER_ROLLBACK: original transparent-proxy path\n'
rollback_status=0
probe_once || rollback_status=$?
printf 'RESULT: baseline=%s bypass=%s after_rollback=%s (0=passed, 1=network-test-failed)\n' \
  "$baseline_status" "$bypass_status" "$rollback_status"
printf 'No services restarted; main Mihomo selection and asset downloads unchanged.\n'

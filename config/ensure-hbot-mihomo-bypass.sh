#!/usr/bin/env bash
# Root-owned installed copy. Only the approved UID + IPv4 + TCP port is exempted.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID -eq 0 && $(/usr/bin/id -u wangmengfan) == 1013 ]]
ipt=/usr/sbin/iptables-legacy
rule=(-p tcp -d 54.176.102.50/32 --dport 443 -m owner --uid-owner 1013
      -m comment --comment hbot-mihomo-us01-bypass -j RETURN)
action=${1:-ensure}

case "$action" in
  remove)
    if "$ipt" -w 5 -t nat -C TP_OUT "${rule[@]}" 2>/dev/null; then
      "$ipt" -w 5 -t nat -D TP_OUT "${rule[@]}"
      printf 'Removed only the H-Bot US01 bypass rule.\n'
    fi
    # Reading the whole table must succeed before reporting successful absence.
    remaining=$("$ipt" -w 5 -t nat -S)
    if [[ "$remaining" == *hbot-mihomo-us01-bypass* ]]; then
      printf 'An H-Bot bypass entry remains; inspect manually.\n' >&2
      exit 1
    fi
    printf 'BYPASS_REMOVED\n'
    exit 0
    ;;
  ensure|status) ;;
  *) printf 'Expected ensure, status, or remove.\n' >&2; exit 2 ;;
esac

# No new chains or global jumps are created. If the old proxy is stopped, wait.
if ! "$ipt" -w 5 -t nat -C OUTPUT -p tcp -j TP_OUT 2>/dev/null; then
  printf 'Old transparent-proxy OUTPUT hook absent; no rule change.\n'
  [[ "$action" == ensure ]] && exit 0 || exit 1
fi
"$ipt" -w 5 -t nat -C TP_OUT -j TP_RULE
"$ipt" -w 5 -t nat -C TP_RULE -p tcp -j REDIRECT --to-ports 52345
if ! "$ipt" -w 5 -t nat -C TP_OUT "${rule[@]}" 2>/dev/null; then
  [[ "$action" == ensure ]] || exit 1
  "$ipt" -w 5 -t nat -I TP_OUT 1 "${rule[@]}"
  printf 'Installed H-Bot bypass: UID 1013 -> 54.176.102.50:443 only.\n'
fi
first_rule=$("$ipt" -w 5 -t nat -S TP_OUT | /usr/bin/awk '$1 == "-A" {print; exit}')
if [[ "$first_rule" != *hbot-mihomo-us01-bypass* ]]; then
  printf 'H-Bot rule exists but is no longer first; refusing to reorder other rules.\n' >&2
  exit 1
fi
if [[ "$action" == status ]]; then
  printf 'BYPASS_READY: UID=1013 destination=54.176.102.50/32 TCP=443\n'
  "$ipt" -w 5 -t nat -L TP_OUT -n -v -x --line-numbers
fi

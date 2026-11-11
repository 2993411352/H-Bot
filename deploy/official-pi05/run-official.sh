#!/usr/bin/env bash
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ -x "$HBOT_PYTHON" ]] || { echo 'Run setup.sh first (runtime Python is missing)' >&2; exit 2; }
case "$HBOT_OFFICIAL_NETWORK" in
  direct) unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy ;;
  inherit) ;;
  *) echo 'HBOT_OFFICIAL_NETWORK must be direct or inherit' >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$package"
exec "$HBOT_PYTHON" -u "$package/official_server.py" "$@"

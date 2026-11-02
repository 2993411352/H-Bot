#!/usr/bin/env bash
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
export HTTP_PROXY=http://127.0.0.1:27892
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export OMP_NUM_THREADS=2
exec nice -n 10 "$HBOT_HOME/openpi/.venv/bin/python" -u \
  "$HBOT_HOME/config/download-holobrain.py" \
  >> "$HBOT_HOME/logs/holobrain-download.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
hbot_proxy_on
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec nice -n 10 "$HBOT_HOME/openpi/.venv/bin/python" -u \
  "$HBOT_HOME/config/download-pi05-lerobot.py" --rate-mib 8 \
  >> "$HBOT_HOME/logs/pi05-lerobot-download.log" 2>&1

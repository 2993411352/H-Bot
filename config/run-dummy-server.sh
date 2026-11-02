#!/usr/bin/env bash
set -euo pipefail
root=/study/wangmengfan/hbot
exec > >(tee -a "$root/logs/local-dummy-server.log") 2>&1
printf '[%s] Starting local dummy server on 127.0.0.1:8765; action_steps=15\n' "$(date -Is)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
cd "$root/robo_orchard_sim"
exec "$root/robo_orchard_lab/.venv/bin/python" \
  -m robo_orchard_server.policy.dummy.main \
  --host 127.0.0.1 --port 8765 --action-steps 15

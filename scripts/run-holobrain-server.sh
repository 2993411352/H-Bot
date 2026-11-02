#!/usr/bin/env bash
set -euo pipefail
root=/study/wangmengfan/hbot
if ss -H -lnt 'sport = :8765' | grep -q .; then
  printf 'Port 8765 is in use. Stop the existing local policy server first.\n' >&2
  exit 1
fi
exec > >(tee -a "$root/logs/local-holobrain-server.log") 2>&1
printf '[%s] Starting HoloBrain on physical GPU 1, listening on 127.0.0.1:8765\n' "$(date -Is)"
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec bash "$root/config/holobrain-run.sh" \
  -m robo_orchard_server.policy.holobrain.main \
  --model-dir "$root/models/HoloBrain_v0.5_sim_sdpa" \
  --device cuda:0 --host 127.0.0.1 --port 8765

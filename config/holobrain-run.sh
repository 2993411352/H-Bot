#!/usr/bin/env bash
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
export HOLOBRAIN_CKPT_BASE="$HBOT_HOME/models/HoloBrain_v0.5_sim/robo_orchard_sim_contest_six_tasks/ckpt"
export HOLOBRAIN_URDF_BASE="$HBOT_HOME/models/HoloBrain_v0.5_sim/robo_orchard_sim_contest_six_tasks/urdf"
export PYTHONPATH="$HBOT_HOME/robo_orchard_lab${PYTHONPATH:+:$PYTHONPATH}"
export ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR="$HBOT_HOME/models/HoloBrain_v0.5_sim_sdpa"
if [[ " $* " == *" robo_orchard_server.policy.holobrain.main "* && " $* " != *" --help "* ]]; then
  "$HBOT_HOME/robo_orchard_lab/.venv/bin/python" "$HBOT_HOME/config/prepare-holobrain-runtime.py"
fi
cd "$HBOT_HOME/robo_orchard_lab"
exec "$HBOT_HOME/robo_orchard_lab/.venv/bin/python" "$@"

#!/usr/bin/env bash
set -euo pipefail

source /study/wangmengfan/hbot/config/hbot-env.sh
cd /study/wangmengfan/hbot/openpi
exec uv run "$@"


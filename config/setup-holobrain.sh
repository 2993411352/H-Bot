#!/usr/bin/env bash
# Project-local installation only; never alters OpenPI, Docker daemon, or assets.
set -euo pipefail
source /study/wangmengfan/hbot/config/hbot-env.sh
exec >> "$HBOT_HOME/logs/holobrain-setup.log" 2>&1
trap 'printf "SETUP_FAILED line=%s exit=%s time=%s\n" "$LINENO" "$?" "$(date -Is)"' ERR
export HTTP_PROXY=http://127.0.0.1:27892
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2
export UV_HTTP_TIMEOUT=7200 UV_HTTP_RETRIES=5 UV_LINK_MODE=copy
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=2
export DS_BUILD_OPS=0 CUDA_HOME="$HBOT_HOME/tools/cuda-12.8"
export TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1
export PATH="$CUDA_HOME/bin:$PATH"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1
project="$HBOT_HOME/robo_orchard_lab"
venv="$project/.venv"
printf 'SETUP_STARTED %s\n' "$(date -Is)"
test "$(git -C "$project" rev-parse HEAD)" = 48ac43987df9ac1ba40fef72dd7a2b4679c592b4
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
if [ ! -e "$venv/pyvenv.cfg" ]; then
  uv venv --python "$HBOT_HOME/openpi/.venv/bin/python" "$venv"
fi
"$venv/bin/python" -c 'import sys; print("ISOLATED_PYTHON", sys.executable)'
printf 'STAGE core-dependencies %s\n' "$(date -Is)"
uv pip install --python "$venv/bin/python" \
  -c "$HBOT_HOME/config/holobrain-constraints.txt" \
  torch==2.8.0 torchvision==0.23.0 'setuptools>=77.0.3,<80' wheel ninja packaging psutil pip nvidia-ml-py pyarrow
printf 'STAGE pytorch3d %s\n' "$(date -Is)"
uv pip install --python "$venv/bin/python" --no-build-isolation \
  -c "$HBOT_HOME/config/holobrain-constraints.txt" \
  'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
printf 'STAGE holobrain-and-server %s\n' "$(date -Is)"
cd "$project"
make version
uv pip install --python "$venv/bin/python" --no-build-isolation \
  --config-settings editable_mode=compat \
  -c "$HBOT_HOME/config/holobrain-constraints.txt" \
  --overrides "$HBOT_HOME/config/holobrain-overrides.txt" \
  -e '.[holobrain_0_5]' -e "$HBOT_HOME/robo_orchard_sim/robo_orchard_server"
printf 'STAGE validation %s\n' "$(date -Is)"
bash "$HBOT_HOME/config/validate-holobrain.sh"
uv pip freeze --python "$venv/bin/python" > "$HBOT_HOME/config/holobrain-installed.txt"
printf 'SETUP_COMPLETE %s\n' "$(date -Is)"

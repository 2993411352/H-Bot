#!/usr/bin/env bash

export HBOT_HOME=/study/wangmengfan/hbot
export PATH="$HBOT_HOME/bin:$PATH"

export UV_CACHE_DIR="$HBOT_HOME/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HBOT_HOME/python"
export HF_HOME="$HBOT_HOME/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HBOT_HOME/cache/xdg"
export OPENPI_DATA_HOME="$HBOT_HOME/models"
export WANDB_DIR="$HBOT_HOME/logs/wandb"
export TMPDIR="$HBOT_HOME/tmp"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

hbot_proxy_on() {
  # H-Bot default: server-local Mihomo. Credentials stay in its private secret file.
  if [[ ! -r /home/wangmengfan/mihomo/proxy.secret ]]; then
    printf 'Mihomo proxy credentials are not readable; proxy not enabled.\n' >&2
    return 1
  fi
  unset ALL_PROXY all_proxy
  source /home/wangmengfan/mihomo/proxy-env.sh
}

# Explicit fallback only; does not change the default or existing processes.
hbot_ssh_proxy_on() {
  export HTTP_PROXY=http://127.0.0.1:27891
  export HTTPS_PROXY=http://127.0.0.1:27891
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export NO_PROXY=localhost,127.0.0.1,::1
  export no_proxy="$NO_PROXY"
}

hbot_proxy_off() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY WS_PROXY WSS_PROXY
  unset http_proxy https_proxy all_proxy ws_proxy wss_proxy
  unset NO_PROXY no_proxy
}


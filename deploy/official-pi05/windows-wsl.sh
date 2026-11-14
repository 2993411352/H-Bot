#!/usr/bin/env bash
set -euo pipefail
umask 077
action=${1:?Missing action}
shift
[[ $# == 5 ]] || { echo 'Expected five encoded arguments' >&2; exit 2; }
values=()
for value in "$@"; do
  [[ "$value" == b64:* ]] || exit 2
  values+=("$(printf '%s' "${value#b64:}" | base64 -d)")
done
root=${values[0]:-$HOME/hbot-official}
checkpoint=${values[1]}
gpu=${values[2]}
network=${values[3]}
proxy=${values[4]}
source_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "$(uname -m)" == x86_64 && "$(uname -r)" == *[Mm]icrosoft*WSL2* ]] || {
  echo 'Use an x86_64 WSL2 distribution. Check wsl --list --verbose.' >&2; exit 2;
}
[[ "$root" == /* && "$root" != / && "$root" != /mnt/* ]] || {
  echo 'InstallDirectory must be inside the Linux filesystem, for example /home/alice/hbot-official.' >&2; exit 2;
}
[[ "$gpu" =~ ^[0-9]+$ && "$network" =~ ^(direct|inherit)$ ]] || exit 2
root=$(realpath -m -- "$root")
package="$root/package"
socket="hbot-official-$(printf '%s' "$root" | sha256sum | cut -c1-16)"
session=hbot-official
has_session() { tmux -L "$socket" has-session -t "=$session" 2>/dev/null; }

if [[ "$action" == Doctor ]]; then
  cat /etc/os-release
  uname -r
  free -h
  df -h "$HOME"
  export PATH="/usr/lib/wsl/lib:$PATH"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  exit
fi

if [[ "$action" == Setup ]]; then
  [[ ! -e "$root" || -f "$root/.hbot-windows-install" ]] || {
    echo "Choose a new installation directory; $root is not managed by this installer." >&2; exit 2;
  }
  mkdir -p "$root"
  touch "$root/.hbot-windows-install"
else
  [[ -f "$root/.hbot-windows-install" ]] || { echo 'Run Setup first' >&2; exit 2; }
fi
exec 9>"$root/windows.lock"
flock -n 9 || { echo 'Another Windows entry command is running' >&2; exit 2; }

if [[ "$action" == Setup || "$action" == Configure ]]; then
  if command -v tmux >/dev/null && has_session; then
    echo 'An evaluation session exists. Finish it and clear its exited pane before changing the installation.' >&2
    exit 2
  fi
  if [[ "$action" == Setup ]]; then
    . /etc/os-release
    [[ "$ID" == ubuntu ]] || { echo 'This installer requires Ubuntu' >&2; exit 2; }
    sudo apt-get update
    sudo apt-get install -y git curl ca-certificates python3 python3-venv openssl util-linux libgl1 libglib2.0-0 tmux
    mkdir -p "$package"
    if [[ "$source_package" != "$package" ]]; then
      tar -C "$source_package" --exclude=./runtime --exclude=./local.env --exclude=./__pycache__ \
        --exclude='*.pyc' --exclude='*.pem' --exclude='*.key' --exclude='session*.txt' \
        --exclude='*.safetensors' --exclude='*.pt' -cf - . | tar -C "$package" -xf -
    fi
  fi
  # A repeat Setup reuses the existing configuration unless explicitly configured.
  if [[ "$action" == Configure || ! -f "$package/local.env" || -n "$checkpoint" ]]; then
    [[ -z "$checkpoint" || "$checkpoint" == /* ]] || { echo 'Use an absolute checkpoint path' >&2; exit 2; }
    {
      printf 'export HBOT_BACKEND=loongforge\n'
      printf 'export HBOT_RUNTIME=%q\n' "$root/runtime"
      printf 'export HBOT_GPU=%q\n' "$gpu"
      printf 'export HBOT_OFFICIAL_NETWORK=%q\n' "$network"
      printf 'export HBOT_PI05_CHECKPOINT=%q\n' "$checkpoint"
      printf 'export PATH=/usr/lib/wsl/lib:$PATH\n'
      if [[ -n "$proxy" ]]; then
        printf 'export HTTPS_PROXY=%q\nexport HTTP_PROXY=%q\n' "$proxy" "$proxy"
      fi
    } > "$package/local.env.tmp"
    mv -- "$package/local.env.tmp" "$package/local.env"
  fi
  if [[ "$action" == Setup ]]; then bash "$package/setup.sh"; fi
  echo "Configuration saved in $package/local.env"
  echo 'Next: Check, SelfTest, then Start. Supply the platform WSS URL only in the hidden prompt.'
  exit
fi

case "$action" in
  Check) exec bash "$package/test.sh" --check ;;
  SelfTest) exec bash "$package/test.sh" --self-test ;;
  Status)
    if has_session; then
      tmux -L "$socket" list-panes -t "=$session" -F 'session=#{session_name} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid}'
    else echo 'No evaluation session'; fi
    ;;
  ClearSession)
    if has_session; then
      [[ "$(tmux -L "$socket" list-panes -t "=$session" -F '#{pane_dead}')" == 1 ]] || {
        echo 'The evaluation process is still running. Attach to inspect it.' >&2; exit 2;
      }
      tmux -L "$socket" kill-session -t "=$session"
    fi
    ;;
  Start|Attach)
    if [[ "$action" == Start ]] && ! has_session; then
      # Keep the exited pane visible. A failed WSS session never reconnects itself.
      printf 'set-window-option -g remain-on-exit on\n' > "$root/tmux.conf"
      printf -v launch 'exec bash %q' "$package/run-official.sh"
      tmux -L "$socket" -f "$root/tmux.conf" new-session -d -s "$session" -c "$package" "$launch" 9>&-
    fi
    has_session || { echo 'No session exists. Run Start first.' >&2; exit 2; }
    flock -u 9
    exec 9>&-
    exec tmux -L "$socket" attach-session -t "=$session"
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac

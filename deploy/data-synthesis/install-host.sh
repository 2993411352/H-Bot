#!/usr/bin/env bash
# Run on the collector's OWN clean Ubuntu host. No driver installation.
set -euo pipefail
if [[ "${1:-}" != --apply ]]; then
  cat <<'EOF'
Installs Docker CE + NVIDIA Container Toolkit on Ubuntu 22.04/24.04 x86_64.
Requires an already-working NVIDIA driver. Does not remove conflicting packages.
Refuses to change Docker while containers are running. No data/image cleanup.
Apply on your own machine: sudo bash install-host.sh --apply
EOF
  exit 0
fi
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) && "$(uname -m)" == x86_64 ]] || { echo 'Supported: native Ubuntu 22.04/24.04 x86_64'; exit 1; }
[[ ! -e /.dockerenv ]] || { echo 'Install on host, not inside a rented container'; exit 1; }
nvidia-smi >/dev/null
if command -v docker >/dev/null; then
  docker info >/dev/null || { echo 'Existing Docker is unhealthy; repair it separately'; exit 1; }
  if [[ -n "$(docker ps -q)" ]]; then
    echo 'Running containers found; no installation/restart was performed.'; exit 1
  fi
else
  for package in docker.io docker-compose docker-compose-v2 podman-docker containerd runc; do
    if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
      echo "Conflicting package $package: review manually; nothing removed"; exit 1
    fi
  done
  apt-get update
  apt-get install -y ca-certificates curl gnupg python3
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/hbot-docker.asc
  chmod a+r /etc/apt/keyrings/hbot-docker.asc
  cat > /etc/apt/sources.list.d/hbot-docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $VERSION_CODENAME
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/hbot-docker.asc
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if ! command -v nvidia-ctk >/dev/null; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg python3
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    gpg --dearmor --yes -o /etc/apt/keyrings/hbot-nvidia.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/hbot-nvidia.gpg] https://#g' > /etc/apt/sources.list.d/hbot-nvidia.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  if [[ -f /etc/docker/daemon.json ]]; then
    cp -p /etc/docker/daemon.json "/etc/docker/daemon.json.hbot-backup-$(date +%Y%m%d%H%M%S)"
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
docker info --format 'Docker ready: {{.ServerVersion}}'
echo 'No users were added to the privileged docker group automatically.'
echo 'If appropriate, an administrator can run: sudo usermod -aG docker YOUR_USER'
echo 'Then log out and log back in. Continue with python3 manage.py doctor.'

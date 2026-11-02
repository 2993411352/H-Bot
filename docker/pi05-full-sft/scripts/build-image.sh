#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${HBOT_IMAGE:-hbot/pi05-full-sft:215abfb-a800-v2}
cache_seed=${HBOT_UV_CACHE_SEED:-$package/resources/uv-cache-seed}
[[ -d "$cache_seed" ]] || { echo 'UV cache seed directory does not exist' >&2; exit 2; }
cd "$package"
sha256sum -c SHA256SUMS
# Standard proxy build arguments are intentionally not persisted in the image.
exec docker build --platform linux/amd64 --progress=plain \
  --build-context "hbot_uv_seed=$cache_seed" \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --tag "$image" "$@" .

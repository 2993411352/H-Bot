#!/usr/bin/env bash
# Installs only in HBOT_RUNTIME; does not install drivers, edit services or start a GPU workload.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'Supported installer: Linux x86_64. Use Linux on the GPU server.' >&2; exit 2;
}
for command in git python3 curl openssl flock sha256sum; do
  command -v "$command" >/dev/null || { echo "Missing system command: $command" >&2; exit 2; }
done
mkdir -p "$HBOT_RUNTIME"
exec 9>"$HBOT_RUNTIME/setup.lock"
flock -n 9 || { echo 'Another setup is running' >&2; exit 2; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HBOT_RUNTIME/uv-cache}"
export UV_PYTHON_INSTALL_DIR="$HBOT_RUNTIME/python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export GIT_LFS_SKIP_SMUDGE=1
if [[ "$HBOT_BACKEND" == jax ]]; then
revision=215abfb217dbac7d5f1273282331b9b1866c0479
source_url="${HBOT_OPENPI_SOURCE:-https://github.com/Physical-Intelligence/openpi.git}"
upstream="$HBOT_RUNTIME/openpi"
if [[ ! -e "$upstream" ]]; then
  git clone --no-checkout "$source_url" "$upstream"
  git -C "$upstream" checkout --detach "$revision"
fi
[[ "$(git -C "$upstream" rev-parse HEAD)" == "$revision" ]] || {
  echo 'Existing OpenPI checkout has a different revision; choose a new HBOT_RUNTIME.' >&2; exit 2;
}
[[ -z "$(git -C "$upstream" status --porcelain --untracked-files=no)" ]] || {
  echo 'Existing OpenPI checkout has changes; refusing to overwrite it.' >&2; exit 2;
}
fi
if [[ ! -x "$HBOT_RUNTIME/bootstrap/bin/python" ]]; then
  python3 -m venv "$HBOT_RUNTIME/bootstrap"
fi
"$HBOT_RUNTIME/bootstrap/bin/python" -m pip install --disable-pip-version-check 'uv==0.9.5'
uv="$HBOT_RUNTIME/bootstrap/bin/uv"
if [[ "$HBOT_BACKEND" == jax ]]; then
  export UV_PROJECT_ENVIRONMENT="$upstream/.venv"
  "$uv" sync --project "$upstream" --python 3.11 --frozen --no-dev
  installed_python="$upstream/.venv/bin/python"
else
  lf_env="$HBOT_RUNTIME/loongforge/.venv"
  if [[ ! -x "$lf_env/bin/python" ]]; then
    "$uv" venv --python 3.12 "$lf_env"
  fi
  "$uv" pip sync --python "$lf_env/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    "$package/loongforge_backend/requirements.lock"
  installed_python="$lf_env/bin/python"
fi
tokenizer="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
expected=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
mkdir -p "$(dirname "$tokenizer")"
if [[ ! -f "$tokenizer" ]]; then
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 300 \
    https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o "$tokenizer.partial"
  printf '%s  %s\n' "$expected" "$tokenizer.partial" | sha256sum --check --status
  mv -- "$tokenizer.partial" "$tokenizer"
fi
printf '%s  %s\n' "$expected" "$tokenizer" | sha256sum --check --status
if [[ "$HBOT_BACKEND" == jax ]]; then
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$installed_python" "$package/verify_install.py"
else
  CUDA_VISIBLE_DEVICES='' "$installed_python" "$package/verify_loongforge.py"
  CUDA_VISIBLE_DEVICES='' "$installed_python" "$package/loongforge_backend/test_adapter.py"
fi
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$installed_python" -m unittest discover -s "$package/tests" -v
echo 'SETUP_OK: run test.sh --check, then test.sh --self-test on an idle GPU.'

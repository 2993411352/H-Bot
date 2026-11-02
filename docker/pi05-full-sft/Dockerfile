FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM ubuntu:22.04

# CUDA user-space libraries come from the pinned JAX/PyTorch wheels. The host
# supplies its NVIDIA driver via NVIDIA Container Toolkit; no host CUDA needed.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY --from=uv /uv /uvx /usr/local/bin/
ARG UBUNTU_MIRROR=
ARG APT_DIRECT=0
RUN --mount=type=cache,id=hbot-pi05-full-apt,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hbot-pi05-full-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    set -eu; \
    if [ -n "$UBUNTU_MIRROR" ]; then \
      sed -i -e "s|http://archive.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" \
             -e "s|http://security.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
    fi; \
    if [ "$APT_DIRECT" = 1 ]; then \
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; \
    fi; \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    apt_options='-o Acquire::Retries=4 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30'; \
    apt-get $apt_options update && apt-get $apt_options install -y --no-install-recommends \
      ca-certificates git git-lfs ffmpeg libgl1 libglib2.0-0 \
      build-essential pkg-config curl util-linux

# Official source at 215abfb217dbac7d5f1273282331b9b1866c0479, without .venv,
# local training patches, datasets, weights, or unrelated submodule checkouts.
ADD vendor/openpi-215abfb.tar.gz /opt/
WORKDIR /opt/openpi
RUN --mount=type=cache,id=hbot-pi05-full-uv,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=hbot_uv_seed,target=/tmp/uv-seed \
    set -eu; if [ -d /tmp/uv-seed/archive-v0 ]; then \
      cp -an /tmp/uv-seed/. /root/.cache/uv/; \
    fi; \
    test "$(sha256sum uv.lock | cut -d ' ' -f 1)" = \
      '793488b5a55bb87200db90a61fd0af51922b686d94e1da4f4c587ab119b37d74' \
    && uv python install 3.11.17 \
    && uv sync --frozen --no-dev --python 3.11.17
ENV PATH="/opt/openpi/.venv/bin:${PATH}" \
    PYTHONPATH=/opt/hbot-sft/scripts/train:/opt/openpi/scripts \
    HBOT_PACKAGE_ROOT=/opt/hbot-sft \
    HBOT_INPUT_ROOT=/input \
    HBOT_OUTPUT_ROOT=/output \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY scripts /opt/hbot-sft/scripts
COPY assets /opt/hbot-sft/assets
COPY resources /opt/hbot-sft/resources
COPY tests /opt/hbot-sft/tests
COPY PROVENANCE.json /opt/hbot-sft/PROVENANCE.json
RUN chmod -R a+rX /opt/hbot-sft \
    && export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu; \
      python /opt/hbot-sft/scripts/train/train_full.py --environment-check \
    && python -m unittest discover -s /opt/hbot-sft/tests -v \
    && runuser -u nobody -- python /opt/hbot-sft/scripts/train/train_full.py --environment-check
WORKDIR /opt/hbot-sft
ENTRYPOINT ["bash", "/opt/hbot-sft/scripts/train/entrypoint.sh"]
CMD ["--help"]

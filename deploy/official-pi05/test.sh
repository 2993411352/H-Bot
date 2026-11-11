#!/usr/bin/env bash
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
case "${1:---check}" in
  --protocol)
    shift
    export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
    exec "$HBOT_PYTHON" -m unittest discover -s "$package/tests" -v
    ;;
  --check|--self-test)
    if [[ $# == 0 ]]; then set -- --check; fi
    exec bash "$package/run-official.sh" "$@"
    ;;
  *) echo 'Usage: bash test.sh --protocol | --check | --self-test [--gpu N] [--fixtures DIR]' >&2; exit 2 ;;
esac

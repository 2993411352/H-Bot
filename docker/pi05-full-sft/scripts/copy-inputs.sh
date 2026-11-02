#!/usr/bin/env bash
# Run ON THE TARGET machine. No simulation assets, LoRA checkpoints or venv.
set -euo pipefail
source_host=${HBOT_SOURCE_HOST:-wangmengfan@172.25.10.54}
source_root=${HBOT_SOURCE_ROOT:-/study/wangmengfan/hbot}
destination=${HBOT_INPUT_ROOT:?Set HBOT_INPUT_ROOT to an absolute target directory}
[[ "$destination" == /* && "$destination" != / ]] || { echo 'Unsafe destination' >&2; exit 2; }
[[ "$source_host" != -* && "$source_root" == /* && "$source_root" != *:* ]] || exit 2
mkdir -p "$destination/models/pi05_base" "$destination/datasets/InstructMove_hbot/lerobot_unpacked"
rsync -a --partial --info=progress2 --protect-args \
  "$source_host:$source_root/models/pi05_base/" "$destination/models/pi05_base/"
rsync -a --partial --info=progress2 --protect-args \
  "$source_host:$source_root/datasets/InstructMove_hbot/lerobot_unpacked/" \
  "$destination/datasets/InstructMove_hbot/lerobot_unpacked/"
echo 'Transfer finished. Run run-sft.sh check to verify checksums, videos and model shapes.'

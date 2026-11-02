#!/usr/bin/env bash
set -uo pipefail

source /study/wangmengfan/hbot/config/hbot-env.sh
hbot_proxy_on || exit 1

export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
# Re-enabled at the user's request after repeated proxy and CDN checks passed.
# Lowered after confirmed anonymous-IP HTTP 429; preserve Xet and existing cache.
export HF_HUB_DISABLE_XET=0
export HF_HUB_DISABLE_TELEMETRY=1
# Supported by the installed hf_xet 1.1.2; keep per-file range concurrency modest.
export HF_XET_NUM_CONCURRENT_RANGE_GETS=1
export HF_XET_HIGH_PERFORMANCE=0

repo=HorizonRobotics/robo_orchard_sim_assets
revision=instructmove_v2
target=/study/wangmengfan/hbot/datasets/robo_orchard_sim_assets
log=/study/wangmengfan/hbot/logs/sim-assets-download.log
python=/study/wangmengfan/hbot/openpi/.venv/bin/python
cli=/study/wangmengfan/hbot/openpi/.venv/bin/huggingface-cli

mkdir -p "$target" "$(dirname "$log")"
exec 9> /study/wangmengfan/hbot/tmp/sim-assets-download.lock
if ! flock -n 9; then
  printf 'Another managed asset downloader is running; refusing a duplicate.\n' >&2
  exit 1
fi
exec > >(tee -a "$log") 2>&1
printf '[%s] Asset proxy: server-local Mihomo 127.0.0.1:17890 (authenticated); existing cache retained\n' "$(date -Is)"

download_pid=
stop_download() {
  trap - INT TERM
  if [[ -n "$download_pid" ]] && kill -0 "$download_pid" 2>/dev/null; then
    kill -TERM "$download_pid"
    wait "$download_pid" || true
  fi
  printf '[%s] Downloader stopped; all files and partial downloads retained\n' "$(date -Is)"
  exit 0
}
trap stop_download INT TERM

cooldown() {
  sleep "$1" &
  download_pid=$!
  wait "$download_pid"
  download_pid=
}

# A one-time restart deadline can be supplied without delaying every future run.
not_before=${HBOT_ASSET_NOT_BEFORE:-0}
now_epoch=$(date +%s)
if [[ "$not_before" =~ ^[0-9]+$ ]] && (( not_before > now_epoch )); then
  printf '[%s] Initial rate-limit cooldown until %s; existing files retained\n' \
    "$(date -Is)" "$(date -d "@$not_before" -Is)"
  cooldown "$((not_before - now_epoch))"
fi

verify_snapshot() {
  "$python" - "$repo" "$revision" "$target" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, '/study/wangmengfan/hbot/config')
from hbot_hf_rate_limit import configure
configure()
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

repo, revision, target_text = sys.argv[1:]
target = Path(target_text)
count = 0
total_bytes = 0
bad_count = 0
examples = []
# The remote listing must finish successfully: a cached CLI return is not proof
# of completion. Verify current-revision file presence and size, not only du.
for item in HfApi().list_repo_tree(
    repo_id=repo, repo_type="dataset", revision=revision, recursive=True
):
    if not isinstance(item, RepoFile):
        continue
    count += 1
    total_bytes += item.size
    path = target / item.path
    if not path.is_file() or path.stat().st_size != item.size:
        bad_count += 1
        if len(examples) < 20:
            examples.append(item.path)
incomplete = sum(1 for _ in (target / ".cache").rglob("*.incomplete"))
print(f"Verification: remote_files={count} bytes={total_bytes} "
      f"missing_or_wrong_size={bad_count} incomplete_cache_files={incomplete}")
for name in examples:
    print(f"  Missing/wrong size: {name}")
if incomplete and not bad_count:
    print("Unused partial cache files retained; all current-revision files are present at expected sizes.")
raise SystemExit(0 if count and not bad_count else 1)
PY
}

while true; do
  # Test a real HTTPS response, not just whether the tunnel port is listening.
  # curl reads HTTPS_PROXY from the environment: do not expose credentials in argv.
  proxy_http=$(curl --noproxy '' -sS -o /dev/null \
    -w '%{http_code}' --connect-timeout 8 --max-time 25 https://huggingface.co/)
  proxy_status=$?
  if [[ $proxy_status -ne 0 || ! "$proxy_http" =~ ^[23][0-9][0-9]$ ]]; then
    if [[ "$proxy_http" == 429 ]]; then
      printf '[%s] Hugging Face HTTP 429; cooling down for 600 seconds\n' "$(date -Is)"
      cooldown 600
    else
      printf '[%s] Proxy unhealthy: curl=%s HTTP=%s; retrying in 60 seconds\n' \
        "$(date -Is)" "$proxy_status" "$proxy_http"
      cooldown 60
    fi
    continue
  fi
  printf '\n[%s] Starting/resuming RoboOrchardSim asset download (Xet enabled; workers=1; Hub interval=5s; 429 cooldown>=600s)\n' "$(date -Is)"
  "$python" "$HBOT_HOME/config/hbot_hf_rate_limit.py" "$cli" download \
    "$repo" \
    --repo-type dataset \
    --revision "$revision" \
    --local-dir "$target" \
    --max-workers 1 &
  download_pid=$!
  wait "$download_pid"
  download_status=$?
  download_pid=

  if [[ $download_status -eq 0 ]]; then
    printf '[%s] Download command exited successfully; verifying snapshot\n' "$(date -Is)"
    verify_snapshot
    verify_status=$?
    if [[ $verify_status -eq 0 ]]; then
      printf '[%s] Asset download and verification completed successfully\n' "$(date -Is)"
      exit 0
    fi
    printf '[%s] Snapshot verification failed; retrying in 600 seconds\n' "$(date -Is)"
  else
    printf '[%s] Download exited with status %s; retrying in 600 seconds\n' \
      "$(date -Is)" "$download_status"
  fi
  cooldown 600
done

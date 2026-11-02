#!/usr/bin/env bash
set -euo pipefail
root=/study/wangmengfan/hbot
if [[ ! -f "$root/cache/sim-asset-index/asset_index.v7.parquet" ]]; then
  printf 'Runtime asset index missing; prepare it before starting local evaluation.\n' >&2
  exit 1
fi
if docker container inspect hbot-sim >/dev/null 2>&1; then
  image=$(docker inspect -f '{{.Config.Image}}' hbot-sim)
  project=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' hbot-sim)
  if [[ "$image" != hbot/robo_orchard_sim:track1-f2e041e || "$project" != hbot ]]; then
    printf 'Container name is occupied by another configuration; inspect it first.\n' >&2
    exit 1
  fi
  if [[ $(docker inspect -f '{{.State.Running}}' hbot-sim) != true ]]; then
    docker start hbot-sim
  fi
else
  export HBOT_UID=$(id -u) HBOT_GID=$(id -g) HBOT_SIM_GPU=0
  docker compose -p hbot \
    -f "$root/config/docker-compose.sim.yml" \
    -f "$root/config/docker-compose.local.yml" \
    run -d --no-deps --name hbot-sim --entrypoint /bin/bash \
    robo-orchard-sim -lc 'exec sleep infinity'
fi
docker inspect hbot-sim --format 'name={{.Name}} status={{.State.Status}} image={{.Config.Image}}'

# H-Bot server environment

Root: `/study/wangmengfan/hbot`

## Local self-test workflow (2026-10-27)

The persistent container is `hbot-sim`, using physical GPU 0. The policy server
runs in the host's independent RoboOrchardLab environment. Host networking lets
the container connect to `127.0.0.1:8765`; no public listener, port forwarding,
platform evaluation request, or supervisor URI is used.

```bash
bash /study/wangmengfan/hbot/config/start-sim-container.sh
# Terminal 1: leave the policy process running.
bash /study/wangmengfan/hbot/config/run-dummy-server.sh
# Terminal 2: six official tasks, one episode each, MCAP plus MP4.
bash /study/wangmengfan/hbot/config/run-local-eval.sh dummy
```

Do not launch duplicate servers if an existing tmux server already occupies
8765. Stop the dummy server after its Client has finished. Then use:

```bash
# Terminal 1: physical GPU 1; only local verified model weights are used.
bash /study/wangmengfan/hbot/config/run-holobrain-server.sh
# Terminal 2: one pick_color episode first.
bash /study/wangmengfan/hbot/config/run-local-eval.sh holobrain_smoke hbot-holobrain-smoke.yaml
# Later, full six-task baseline:
bash /study/wangmengfan/hbot/config/run-local-eval.sh holobrain
```

`config/hbot-local.yaml` preserves the official tasks, seed, splits and step
limits; only the Server config path and episode count differ. The official
repository stays unchanged. Every invocation creates a timestamped directory
under `outputs/sim/` and a corresponding dispatcher log under `logs/`.
`config/hbot-local-recheck.yaml` omits only the already-tested stack_cubes task
for the one-time five-task repair rerun.

The published v7 asset index contains relative paths, while this source revision
uses them directly. `prepare-asset-runtime-index.py` creates a derived index in
`cache/sim-asset-index/`, converting only five filesystem path columns to
container-absolute paths. All 3,244 asset rows, UUIDs, semantic fields and schema
metadata are preserved; every USD and URDF is checked for existence. The source
index and downloaded asset snapshot are not edited. `docker-compose.local.yml`
mounts only the derived index directory over the container's default index path.
Always include this override when recreating the container. The pre-fix container
was stopped and retained as `hbot-sim-pre-index-20261027` for recovery.

Worker exit code zero is not sufficient: scene construction errors can become
zero-step failed episodes. `verify-local-eval.py` checks summary failure counts,
actual shard action steps, stop reasons, MCAPs and exported preview videos.
Dummy task scores of zero are expected and do not indicate a communication error.
For multiple repair runs, pass directories from oldest to newest; newer results
replace older results for the same task. Use `--tasks pick_color` for the model
smoke test. Detailed results from initial and repaired runs are kept separately.

```bash
/study/wangmengfan/hbot/robo_orchard_lab/.venv/bin/python \
  /study/wangmengfan/hbot/config/verify-local-eval.py /absolute/path/to/run
# Enter the simulator shell without starting another simulator process:
docker exec -it hbot-sim bash
# Stop only this project's container when its Client is finished:
docker stop hbot-sim
```

The full asset snapshot finished on 2026-10-27 at 19:51 CST: 78,493 files,
247,507,438,637 bytes, no missing/wrong-size catalog files. Supplemented config
files also passed Git blob hashes. `logs/sim-assets-verification.json` records
the pinned revision and exact verification scope. Older download notes below
are historical, not the current readiness state.

### Verified local results, 2026-10-27

- Initial dummy run: `outputs/sim/local_dummy_20261027_195904`. stack_cubes
  completed 900 steps; five other tasks initially failed on portable index paths.
  Failed results are retained rather than overwritten.
- Repaired dummy run: `outputs/sim/local_dummy_recheck_20261027_201413`.
  The other five tasks completed 200/200/300/300/300 steps. Across the two runs,
  all six tasks have a valid full episode, MCAP and MP4. Combined verification:
  `logs/local-dummy-validation.json`, `passed=true`.
- Real-model smoke: `outputs/sim/local_holobrain_smoke_20261027_203310`.
  pick_color completed 200 steps with no failed episodes, score 0.5, but task
  success=false (reached and contacted the carrot; did not grasp/lift it).
  This confirms the runtime loop, not overall baseline performance.
  `logs/local-holobrain-validation.json` records `passed=true` for runtime checks.
- Both previews are three-camera 3840x720, 30 fps mosaics. The HoloBrain video
  is 6.667 seconds. Preview frames from both dummy and HoloBrain videos were
  visually inspected: all three cameras render the robot and scene rather
  than blank frames.
- Two-second GPU samples during this one HoloBrain episode observed total
  GPU-memory maxima of 7545 MiB on simulator GPU 0 and 2907 MiB on model GPU 1.
  These are sampled device totals, not guaranteed peaks or minimum hardware
  requirements. Log: `logs/holobrain-smoke-gpu-20261027-2033.log`.
- The test dummy and HoloBrain servers were stopped after their clients exited.
  `hbot-sim` remains running but idle; port 8765 is free. At shutdown the GPUs
  returned to 17/77 MiB usage. No pre-existing training process was stopped.

## Training environment

```bash
source /study/wangmengfan/hbot/config/hbot-env.sh
cd /study/wangmengfan/hbot/openpi
uv run python -c 'import jax, torch; print(jax.devices()); print(torch.cuda.is_available())'
```

`hbot_proxy_on` enables the existing local proxy for downloads. It is not enabled automatically.

The lightweight `robo_orchard_server` package is installed in the same
environment so the model process can expose a policy endpoint without loading
the Isaac simulator stack.

## Simulation environment

The simulator is deliberately not started automatically, so it cannot take GPU memory from existing jobs.

```bash
cd /study/wangmengfan/hbot/config
HBOT_UID=$(id -u) HBOT_GID=$(id -g) HBOT_SIM_GPU=0 \
  docker compose -f docker-compose.sim.yml run --rm robo-orchard-sim
```

Select a currently idle GPU before starting the simulator.
Simulation assets belong under
`/study/wangmengfan/hbot/datasets/robo_orchard_sim_assets`.

The local runtime image is `hbot/robo_orchard_sim:track1-f2e041e`. It extends
the official image with RoboOrchardSim 1.0.0, RoboOrchard Core 0.4.0, and the
official runtime constraints.

### Headless Vulkan

On this host NVIDIA manifests are in `/usr/share/vulkan`, not `/etc/vulkan`.
The compose file mounts those exact files read-only and selects the
project-local `config/nvidia-egl-icd.json` for NVIDIA EGL. No desktop, DISPLAY,
X11 socket, driver replacement, or system-wide Vulkan change is needed.
Camera rendering is still required: it produces the images consumed by the
policy even when there is no on-screen window.

The compose configuration and `vulkaninfo --summary` passed on 2026-10-26,
identifying an RTX 3090 and driver 570.133.20. This verifies device discovery,
not an Isaac scene/camera or policy rollout. The previous compose file is
preserved as `docker-compose.sim.yml.bak-20261026-vulkan`.

```bash
docker compose -f /study/wangmengfan/hbot/config/docker-compose.sim.yml \
  run --rm --no-deps -T --entrypoint /bin/bash robo-orchard-sim \
  -lc 'vulkaninfo --summary'
```

## HoloBrain baseline: separate environment

Official source: `/study/wangmengfan/hbot/robo_orchard_lab`, pinned to
`48ac43987df9ac1ba40fef72dd7a2b4679c592b4`. Its `.venv` is separate from
OpenPI and does not inherit system site-packages. Installation targets
PyTorch 2.8.0 / CUDA 12.8 / Transformers 5.10.2 and the official PyTorch3D
v0.7.9 source. FlashAttention is not required for the prepared SDPA path.

The local CUDA 12.8 compilation tools were copied from the already present
simulator image, without a new image download. Project-only GCC 10 wrappers
exclude this host's conflicting `/usr/local/include/assert.h`; no system
compiler or headers were changed. A compile-only CUDA test passed without
launching GPU work.

Two pinned downloads are required:

- `HorizonRobotics/HoloBrain_v0.5_sim` at
  `e543e04c40802c21b1496284a7b424b08ac6aafc`, under `models/HoloBrain_v0.5_sim`.
- `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`, under the
  export's `robo_orchard_sim_contest_six_tasks/ckpt/Qwen3.5-2B` directory.

Downloads resume from existing Hugging Face partial files. Every upstream
file is checked by size and SHA-256 (LFS) or Git blob SHA-1. Completion is
recorded in `logs/holobrain-models-verified.json` only after both repos pass.
The SDPA runtime view uses symlinks to these weights and its own configuration;
the verified upstream files are not modified or duplicated.

### Background jobs and completion criteria

The jobs were launched on 2026-10-26; launching is not proof of completion.

| tmux session | Purpose | Success evidence |
| --- | --- | --- |
| `hbot-download-proxy-v2` | Loopback-only 27892 to existing proxy 27891 | `READY` in proxy log |
| `hbot-models` | Baseline + Qwen downloads and verification | `ALL_MODELS_VERIFIED` |
| `hbot-env` (initial setup, exited) | Isolated installation | Latest repair: `ENV_VALIDATION_PASSED` in `holobrain-validation.log` |

With user approval on 2026-10-27, model and dependency downloads share an aggregate **2 MiB/s** cap on
port 27892. Existing `hbot-assets` continues to use port 27891 without changes.
This limits contention but cannot promise zero impact on shared bandwidth.
The original Windows reverse SSH tunnel and local proxy must remain running.
tmux survives disconnecting a shell, but cannot keep a broken proxy working.

```bash
tmux list-sessions
tail -n 30 /study/wangmengfan/hbot/logs/holobrain-download.log
tail -n 30 /study/wangmengfan/hbot/logs/holobrain-setup.log
tail -n 10 /study/wangmengfan/hbot/logs/holobrain-proxy.log
```

If a job exits with an error, inspect its log and restore proxy connectivity
first. Do not launch duplicate sessions/downloaders. Only when the respective
session is absent, resume it with:

```bash
tmux new-session -d -s hbot-models \
  'bash /study/wangmengfan/hbot/config/download-holobrain.sh'
tmux new-session -d -s hbot-env \
  'nice -n 10 bash /study/wangmengfan/hbot/config/setup-holobrain.sh'
```

The environment is repaired using `pyarrow==21.0.0` with `numpy==1.26.4`, plus
`nvidia-ml-py==12.570.172`. These versions are recorded in
`config/holobrain-constraints.txt`; setup also installs them explicitly.
CPU dependency/import checks and official CLI help passed on 2026-10-27.
`config/holobrain-installed.txt` records the repaired installed versions.
The initial setup log retains its old failure; use the newer validation log.

Repeat the CPU-only environment checks with:

```bash
bash /study/wangmengfan/hbot/config/validate-holobrain.sh
```

After model download/checksum completion, validate full offline loading through
the official policy (CPU only; no action inference, simulation or service):

```bash
/study/wangmengfan/hbot/robo_orchard_lab/.venv/bin/python \
  /study/wangmengfan/hbot/config/prepare-holobrain-runtime.py
bash /study/wangmengfan/hbot/config/holobrain-run.sh \
  /study/wangmengfan/hbot/config/validate-holobrain-model.py
```

The success marker is `OFFLINE_CPU_POLICY_LOAD_PASSED`. CPU loading does not
measure GPU inference memory or prove the simulation evaluation loop.

On 2026-10-27 at 14:52 CST, both model repositories passed every file's size
and hash verification (23 baseline files; 13 Qwen files, 4,571,274,023 bytes).
The offline CPU policy load passed at 14:53 CST: the official model, processor
and shared-policy session initialized successfully. Loading took 12.9 seconds
and peak host memory was about 5.60 GiB (not GPU memory). The log is
`logs/holobrain-model-validation.log`. A separate tiny CUDA tensor test passed
on RTX 3090 and exited. No actual model action inference or simulation ran.

The optional flash-linear-attention / causal-conv1d fast path is not installed;
Transformers reports its PyTorch fallback. The runtime uses SDPA instead of
FlashAttention 2. Performance tuning and real GPU policy inference remain
separate validation steps.

### Next: model service and simulation validation

After model checksum and environment validation succeed, and after choosing an idle GPU, this starts
the actual model server (not started automatically by setup):

```bash
CUDA_VISIBLE_DEVICES=1 /study/wangmengfan/hbot/config/holobrain-run.sh \
  -m robo_orchard_server.policy.holobrain.main \
  --host 127.0.0.1 --port 8765 --device cuda:0
```

The wrapper prepares `models/HoloBrain_v0.5_sim_sdpa` only after model checksum
verification. Physical GPU 1 becomes logical `cuda:0` inside this process.
Keep the simulator on the other GPU; its host-networked container can connect
to `ws://127.0.0.1:8765`. Complete scene/camera rendering and one baseline
rollout still need to be tested after the required assets are available.

## Simulation assets

The `instructmove_v2` asset branch contains 78,493 files (230.51 GiB). Its
resumable download runs in the `hbot-assets` tmux session.

On 2026-10-27, repeated TLS failures on the Xet CAS/range-transfer path were
observed while ordinary HTTPS range requests succeeded. Xet was temporarily
disabled; later checks also reproduced failures on ordinary HTTPS and the
local proxy, so the fault was not isolated to Xet. After the network recovered,
three consecutive Hugging Face requests and a CDN range request passed. At the
user's request, Xet was re-enabled with `HF_HUB_DISABLE_XET=0` at 14:26 CST.
The downloader keeps two workers, checks the proxy using an actual
HTTPS response before each attempt, and uses `flock` to prevent duplicates.
Finished files are reused. Completion requires a successful remote listing
and expected file-size checks, not just the CLI exit status or rounded `du -h`.
The original script is backed up as `download-sim-assets.sh.bak-20261027-xet`.
The intermediate HTTP version is `download-sim-assets.sh.bak-20261027-http`.
Retries still restart the full snapshot command; per-file retry with a cached
manifest has not been implemented.
One unfinished sparse Xet fragment for `2924.png` was preserved under
`tmp/assets-xet-partials-20261027-1235`; that file is fetched afresh because
its sparse-file length is not a safe sequential HTTP resume offset.

### Readiness audit, updated 2026-10-27 14:53 CST

- Both RTX 3090 cards were idle; Vulkan device discovery had already passed.
- PyTorch 2.8.0+cu128 and the PyTorch3D compiled extension passed CPU-only imports.
- The HoloBrain baseline and Qwen repositories are fully downloaded and hash
  verified. The resumed HTTP download reused its 2.90 GB partial file.
- The isolated environment passed all 164-package dependency checks, imports,
  CPU operations, official CLI help, robot processor construction and full
  offline policy loading. PyArrow/NumPy compatibility and missing
  `nvidia-ml-py` were repaired. A tiny CUDA tensor operation also passed.
- The full asset snapshot is not yet verified. Robot and task assets must be
  checked explicitly before trying a task; aggregate download size is not enough.
  The default Panda Droid/room-table task currently lacks
  `ROBOTS/FRANKA/franka_panda_robotiq_flange.usd`,
  `ROBOTS/FRANKA/franka_droid.urdf`, and the table's `table_001.usd` and
  `table_interaction.json` under `PUBLIC_OBJECTS/evaluation_assets/tables/table_001`.

No baseline action inference or Isaac scene was launched. Next validation should proceed from a headless
camera smoke test, through a single task/dummy-policy run, to one baseline episode
after the required task assets are ready.

```bash
# Watch the live downloader.
tmux attach -t hbot-assets

# Detach without stopping it: press Ctrl-b, then d.

# Check progress without attaching.
du -sh /study/wangmengfan/hbot/datasets/robo_orchard_sim_assets
tail -n 50 /study/wangmengfan/hbot/logs/sim-assets-download.log

# Stop the downloader if needed.
tmux kill-session -t hbot-assets

# Resume later; completed files are reused.
tmux new-session -d -s hbot-assets \
  /study/wangmengfan/hbot/config/download-sim-assets.sh
```

## Independent Mihomo proxy (2026-10-27)

Mihomo v1.19.33 and MetaCubeXD v1.273.1 are installed separately under
`/home/wangmengfan/mihomo`. Both release archives were checked against the
SHA256 digests published on the official GitHub release pages.

- Mixed HTTP/SOCKS proxy: `127.0.0.1:17890`, with authentication.
- Controller and locally hosted dashboard: `127.0.0.1:19090`, with API secret.
- TUN, DNS interception and LAN access are disabled. No root privileges needed.
- Service: `systemctl --user status mihomo.service`.
- Log: `/home/wangmengfan/mihomo/logs/mihomo.log`.
- This install does not change H-Bot download proxy settings or stop any jobs.

Installation starts in **REJECT / no-subscription mode**, not as a usable VPN.
Run the following in Windows PowerShell and paste the subscription at the hidden
prompt. Port 27891 is used only to fetch the subscription through the existing
tunnel; omit `--bootstrap-proxy ...` to fetch directly from the server.

```powershell
ssh -t wangmengfan@172.25.10.54 "python3 /home/wangmengfan/mihomo/manage.py import --bootstrap-proxy http://127.0.0.1:27891"
```

The importer accepts node-list YAML / URI / base64 formats supported by Mihomo.
It ignores subscription top-level listener, DNS and routing settings, uses a
file provider, and restores the prior configuration if no nodes can be loaded.
Unsupported sing-box JSON must be re-exported in a supported format by the
provider. No public third-party conversion service is used. URL, nodes, secrets
and backups stay under the private installation directory. Subscription updates
are manual (rerun import); HBot stays REJECT until a node is explicitly selected.

To open the dashboard, keep this command running on Windows:

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:19090:127.0.0.1:19090 wangmengfan@172.25.10.54
```

Visit `http://127.0.0.1:19090/ui/`; use controller
`http://127.0.0.1:19090`. Read the secret in your own SSH terminal with
`cat /home/wangmengfan/mihomo/controller.secret` and do not paste it into chat.
Select a node in group `HBot`, then verify real HTTPS access before switching
any downloader. The terminal-only opt-in environment helper is
`source /home/wangmengfan/mihomo/proxy-env.sh`; it does not edit shell profiles.

Status without revealing credentials:

```bash
python3 /home/wangmengfan/mihomo/manage.py status
```

The service is enabled for the user manager, but **Linger was still disabled at
installation**. To guarantee it starts at boot and remains after the last login
ends, an administrator must run `sudo loginctl enable-linger wangmengfan`.
Until that is done and node connectivity is tested, keep the old reverse tunnel.

### Verified installation state, 2026-10-27 13:38 CST

The user imported 32 VLESS entries (28 actual nodes plus 4 account-information
entries). All 28 actual nodes failed direct Hugging Face health checks. Sampled
nodes resolved and accepted TCP connections, but Reality handshakes returned EOF.
The sampled US node's system DNS address matched a public HTTPS DNS response;
its fingerprint, SNI and Reality parameters were present.

An isolated comparison using the **same node and configuration**, but reaching
that node through existing proxy 27891, succeeded (Hugging Face delay 1341 ms).
This isolates a reachability difference in the server's direct outbound path;
it does not establish whether the precise cause is filtering, provider policy
or another network-path issue. The temporary test core was stopped, and its
temporary files removed. No chained proxy was left configured.

The main service is running, but `HBot` is intentionally **REJECT**. Installation
and subscription parsing are complete; independent outbound access is **not**.
Existing asset downloads still use their old proxy. Do not switch them yet.

The dashboard returns HTTP 200, unauthenticated API requests return 401 and
unauthenticated proxy requests return 407. The install root is mode 0700 and
secret/config files are 0600. Offline importer safety/rollback tests passed.

The explicit test command below checks the new proxy only. It selects a node
only if repeated real HTTPS requests succeed, restoring the previous selection
on failure. It never modifies the asset downloader:

```bash
python3 /home/wangmengfan/mihomo/check_nodes.py --activate
```

### Repair and download migration, 2026-10-27 14:40 CST

The transparent-proxy conflict was confirmed with an A/B/A test: the original
path failed, a narrowly scoped exception succeeded, and restoring the original
path failed again. The successful test included HTTPS 200 responses and a real
asset HTTP 206 response carrying 65,536 bytes. The main Mihomo service also had
its only `MATCH -> HBot` rule disabled at runtime; that rule was re-enabled using
the documented `/rules/disable` API. Keep this rule enabled and the selected
node **United States 01**. Other nodes have not been exempted from the old proxy.

The administrator installed the persistent exception for **UID 1013 ->
54.176.102.50/32, TCP 443 only**, at the start of the existing `nat/TP_OUT` chain.
Root-owned files under `/etc/hbot-mihomo-bypass/` and the system units
`hbot-mihomo-bypass.service` / `.timer` maintain only this exception, checking
once per minute. They do not flush rules or restart firewalld, Docker, or v2rayA.
If the node's IP changes, revalidate it before changing the exception; do not
broaden it to all destinations. The user service is enabled and **Linger=yes**,
so the proxy does not require an open SSH session to remain running.

`hbot_proxy_on` now explicitly loads the private authenticated Mihomo helper
for `127.0.0.1:17890`. Credentials are not embedded in this repository or passed
in curl command arguments. `hbot_ssh_proxy_on` preserves the old reverse tunnel
at `127.0.0.1:27891` as an explicit fallback. Already-running processes retain
their existing environment; only the managed asset downloader was restarted.

The restart at 14:39:37 retained all completed assets and partial files, along
with the user's latest settings: **Xet enabled, two file workers**. Subsequent
download logs showed new completed files; live Mihomo connections to Hugging
Face, `cas-server.xethub.hf.co`, and its CDN used `United States 01 -> HBot`.
These are progress checks, not a claim that the complete snapshot is verified.

Read-only checks (no credentials printed):

```bash
source /study/wangmengfan/hbot/config/hbot-env.sh
hbot_proxy_on
python3 /study/wangmengfan/hbot/config/check-hbot-proxy.py
python3 /study/wangmengfan/hbot/config/inspect-assets-status.py
/usr/bin/mctes is-active hbot-mihomo-bypass.timer
/usr/bin/mctes --user is-active mihomo.service
```

The pre-switch environment and downloader scripts are backed up in:
`/study/wangmengfan/hbot/tmp/mihomo-switch-backup.70QFKE/`.
For download rollback, first validate the old tunnel, gracefully stop only the
identified asset wrapper, restore those two scripts, and restart the downloader
once with the same dataset path. Never remove its cache or partial downloads.
The retained old SSH tunnel is not stopped automatically.

To withdraw the root exception and disable its maintenance timer, after no job
depends on the new proxy, the administrator can run:

```bash
sudo bash /study/wangmengfan/hbot/config/install-hbot-mihomo-bypass.sh --remove
```

This removes only the tagged exception and leaves the root-owned files for
recovery. It does not clear other firewall rules or change download settings.


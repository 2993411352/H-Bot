H-Bot pi0.5 LoRA training configuration (2026-10-27)

Status: data/checkpoint integrity + CPU preflight passed during setup.
The user's hbot_lora_smoke_v1 GPU run loaded weights and initialized train state,
but the first training step failed with CUDA OOM (extra allocation ~5.37 GiB).
No training step completed in that run. The revised configuration now freezes
the entire visual encoder in addition to the pretrained base LLM weights.
LoRA and the small action/time projection layers remain trainable.
The revised hbot_lora_smoke_v2 GPU test completed all 5 steps on GPU 0 with exit 0.
Use a NEW experiment name for further training; both test directories are retained.

Verified GPU smoke result (2026-10-27 22:45:59--22:48:17 CST)
  Batch 1; three RGB views; horizon 15; exactly the configuration below.
  GPU 0 peak sampled memory: 8518 MiB (8.32 GiB), at 500 ms sampling intervals.
  This is a sampled device-memory peak, not a guarantee for larger batches/runs.
  Five losses: 0.1297, 0.0364, 0.0516, 0.0659, 0.0546; gradients finite.
  Wall time about 2 min 18 sec includes initialization/JIT/checkpoint writing;
  a 5-step run is NOT a reliable sustained-training throughput measurement.
  Checkpoint: checkpoints/pi05_hbot_lora/hbot_lora_smoke_v2/4/ (~5.4 GiB on disk).
  Directory 4 is the zero-based loop index; restored train_state.step is 5.
  Full parameters AND optimizer state read back on CPU; all values finite.
  Checkpoint normalization file matches the prepared norm statistics byte-for-byte.
  GPU 0 released to 18 MiB after exit; no long training run was started.
  Details: logs/pi05_lora_smoke_v2_validation.json.
  This verifies the training/save/read path only, not policy quality or convergence.

Batch-size short benchmarks (2026-10-27, GPU 0, each 30 steps)
  Same model/data/freezing; batch sizes are explicit CLI overrides, not new defaults.
  Speed excludes steps 0--9 and checkpoint saving; includes data loading/logging.
  Batch   Mean seconds/step   Samples/second   Sampled peak GiB   Peak GPU C
    2          0.575               3.476             16.342           77
    4          1.068               3.745             16.348           87
    8          2.034               3.933             16.363           89
  All three passed 30 updates with finite logged loss/gradients and saved step 29.
  CPU read-back confirmed train_state.step=30 and finite optimizer arrays in all
  three; parameter metadata count and byte-identical saved norm stats also passed.
  These 30-step peaks are not directly comparable to the earlier 5-step batch-1
  test; do not extrapolate larger-batch capacity from nearly equal allocated VRAM.
  Batch 8 is only ~5% faster in samples/second than batch 4 and reached 89 C.
  GPU 0 reports target 83 C, max operating 93 C, slowdown 95 C, shutdown 98 C.
  Current cooling/stability deserves attention before unattended long training;
  post-run thermal flags do not prove whether throttling occurred during training.
  These short tests did not establish long-run thermal stability. The subsequent
  batch-4 100-step attempt hit the temperature cutoff; see the result below.
  Batch 2 has not been tested to thermal equilibrium. No long training was started.
  Reports: logs/pi05_bench_hbot_lora_bench_b{2,4,8}_20261027/report.json.
  Checkpoints: checkpoints/pi05_hbot_lora/hbot_lora_bench_b{2,4,8}_20261027/29/.
  Keep these benchmark experiments separate from actual training.

100-step stability attempt: NOT PASSED (2026-10-27 23:14:49--23:17:28 CST)
  Experiment hbot_lora_b4_stability_v1; GPU 0; batch 4; cutoff 90 C.
  Logged completed updates: 96/100 (steps 0--95). Temperature reached 90 C;
  the benchmark guard interrupted ONLY its own training subprocess group.
  No OOM; all logged loss/gradient/parameter norms remained finite.
  Sampled peak memory 16714 MiB = 16.32 GiB; average 1.078 s/step, 3.709 samples/s
  after excluding the first 10 steps. Early steps 10--29 averaged 1.029 s;
  later steps 60--95 averaged 1.120 s. A thermal-throttling cause is not proven.
  There is NO committed checkpoint from this attempt (the final save was not
  reached). Earlier successful smoke/benchmark checkpoints are unchanged.
  Training process group verified gone, GPU 0 returned to 18 MiB; no power limit,
  fan settings or temperature cutoff were changed. GPU 0 power limit is 350 W.
  Report: logs/pi05_bench_hbot_lora_b4_stability_v1/report.json.
  Full log and 500 ms GPU samples are train.log and gpu.csv in the same directory.
  Resolve the sustained-temperature issue before an unattended long training run.
  Lowering power/fan changes require separate authorization; do not raise the guard
  just to make the test pass. Retry with a NEW experiment name after mitigation.

280 W batch-4 stability test: PASSED (2026-10-28)
  hbot_lora_b4_280w_v1 completed 100 steps; checkpoint 99 restored step=100 on CPU.
  Mean 1.139 s/step, 3.512 samples/s; peak sampled memory 16718 MiB, temperature 88 C.
  Power was restored from 280 W to the original 350 W after the test.
  A 100-step pass is not a guarantee of 13+ hours of thermal stability.

Approved 40k-step long job (prepared; requires the user's sudo start)
  GPU 0 only; 280 W; batch 4; 40000 updates; 2 data workers; log every 10 steps.
  Warmup 1000; peak LR 2.5e-5; cosine decay to 2.5e-6 at step 40000.
  Checkpoint interval 500; retain milestones every 5000 and the latest checkpoint.
  Final directory is 39999 (zero-based index), representing 40000 updates.
  Full checkpoints are about 5.4 GiB each; budget ~50 GiB for retained checkpoints
  and additional temporary space. Preflight requires at least 100 GiB free.
  Estimated 13--14 hours including checkpoint overhead, based on the 100-step
  measurement; actual sustained speed/temperature may differ. 40000 x batch4 is
  about 1.02 frame-epochs, a first training budget rather than proven convergence.

  # Run in the zhangxuankai server terminal (sudo required):
  sudo bash /study/wangmengfan/hbot/scripts/run-pi05-longtrain.sh

  # Background status and user-readable training log:
  systemctl status hbot-pi05-lora-40k.service --no-pager -l
  tail -f /study/wangmengfan/hbot/logs/pi05_longtrain/hbot_pi05_lora_b4_40k_v1/train.log

  # Stop this service only, including its workers; restore the original power:
  sudo bash /study/wangmengfan/hbot/scripts/run-pi05-longtrain.sh stop

  # After diagnosing a stop and cooling the GPU, explicitly resume the same job:
  sudo bash /study/wangmengfan/hbot/scripts/run-pi05-longtrain.sh resume

  Training and all model code run as wangmengfan, NOT as root. Only fixed GPU0
  power setup/restoration runs as root. The launcher installs a runtime-only
  systemd unit under /run, not an enabled boot service. SSH disconnection or
  Ctrl+C in tail does not stop training. There is NO automatic restart.
  Host compatibility: /usr/bin/systemctl is a replacement that returned 1 at
  daemon-reload, before training was started. The launcher now selects the
  root-owned binary matching the installed systemd package checksum, preferring
  /usr/bin/systemctl and otherwise the preserved original /usr/bin/mctes.
  It does not replace any system executable. If neither matches, it refuses
  service actions. It also checks that the service is active after start.
  The unit has a 20-hour per-attempt maximum runtime as a secondary safeguard.
  Stop at >=90 C, missing/invalid GPU telemetry or changed GPU/power identity.
  The monitor samples approximately once per second (query timeout 5 seconds);
  it is not a hard real-time guarantee against temperature overshoot.
  On stop, send SIGINT only to the owned child process group; force cleanup if
  needed. systemd also cleans its cgroup. ExecStopPost restores the saved 350 W
  only if this invocation recorded its power change. A failed restoration leaves
  a root-owned marker and blocks restart; inspect the service journal in that case.
  No emergency/current-step checkpoint is guaranteed. Resume uses the latest
  completed periodic checkpoint; it may lose the work after that checkpoint.
  A stop before the first checkpoint means restarting from the base weights.
  Checkpoint/config/model-adapter hashes are guarded against incompatible resume.
  Long-job logs: logs/pi05_longtrain/hbot_pi05_lora_b4_40k_v1/{train.log,gpu.csv,
  config.json,report_<attempt>.json}; checkpoints are under checkpoints/pi05_hbot_lora/.
  Do not start evaluation/another workload on GPU0 during this job.
  run-pi05-lora.sh ALONE still does not set power or monitor temperature.

Files
  scripts/run-pi05-lora.sh       environment, GPU-idle guard, logs, training entry
  scripts/run-pi05-longtrain.sh  sudo background start/resume/stop, fixed 40k job
  scripts/train/supervise-pi05-longtrain.py  temperature/telemetry monitor and logs
  scripts/train/pi05-longtrain-power.sh     invocation-scoped power setup/restore
  scripts/train/hbot-pi05-lora-40k.service   user training + root power hooks
  scripts/train/tests/test_pi05_longtrain.py CPU-only long-job/power safety tests
  scripts/prepare-pi05-data.sh  CPU-only data-preparation launcher
  scripts/train/train-pi05-hbot.py    CPU --check or official OpenPI train.main(config)
  scripts/train/prepare-pi05-hbot.py  ZIP SHA256/CRC + checkpoint CRC32C + norm statistics
  scripts/train/pi05_hbot.py          task data adapter, transforms, TrainConfig
  scripts/train/tests/test_pi05_launcher.py  mocked launcher safety tests
  scripts/train/benchmark-pi05-lora.py      bounded real-launcher speed/memory tests
  scripts/train/tests/test_pi05_benchmark.py CPU-only monitor/temperature tests
  scripts/train/README.txt            this guide
  config/openpi_assets/pi05_hbot_lora/hbot_panda_joint_delta_h15/
                              norm_stats.json and preparation.json
  logs/pi05_hbot_preflight.json real-data and abstract model-shape test report

Data and supervision
  Six local LeRobot v2.1 datasets; 600 episodes; 156867 frames at 30 Hz.
  Originals retained in datasets/InstructMove_hbot/lerobot/*.zip.
  Extracted data in datasets/InstructMove_hbot/lerobot_unpacked/.
  All six datasets are concatenated without copying/reindexing Parquet/video files.
  Prompts resolved independently per dataset (task IDs overlap between tasks).
  Sampling is uniform over frames, NOT equal per task. Stack-cubes supplies about
  46.6% of frames. All 600 demonstrations are used; no held-out split is configured.
  Training loss alone is NOT evidence of improved competition performance.

Model inputs/outputs
  ext1 -> base_0_rgb; wrist -> left_wrist_0_rgb; ext2 -> right_wrist_0_rgb.
  Third slot is the second external view, despite its built-in OpenPI slot name.
  All three RGB cameras enabled; resized with official transforms to 224 x 224.
  Raw state/action: 7 Panda joints + normalized absolute gripper.
  Joint targets are converted to deltas relative to CURRENT observation state,
  not differences between adjacent future actions. Gripper remains absolute.
  Action horizon = 15 (0.5 seconds). No action chunks cross episode boundaries.
  State/action padded to 32 for pi0.5; action outputs cropped back to 8.
  Inference must load the SAME norm stats and apply the inverse joint transform.
  This configuration is NOT yet a robo_orchard_server policy implementation.

Default starter configuration (not a tuned final training schedule)
  JAX/Flax pi0.5; base weights models/pi05_base/params, all objects CRC32C verified.
  gemma_2b_lora rank16 + gemma_300m_lora rank32; bfloat16; EMA disabled.
  Official get_freeze_filter() PLUS frozen PaliGemma/img/*:
  49,987,584 LoRA + 2,165,792 action/time projection parameters are trainable.
  Total: 52,153,376 trainable (formerly 466,957,072); vision is fully frozen.
  This changes the fine-tuning strategy, not the dataset/camera/action semantics.
  Batch 1, single GPU 0, workers 2, AdamW, peak LR 2.5e-5, warmup 1000,
  default 20000 steps, checkpoint every 1000, keep milestones every 5000.
  20000 steps x batch1 covers only about 0.13 frame-epochs; it is a starter budget,
  not a claim of convergence. Tune batch/steps after memory and throughput tests.
  Launcher never stops other processes; it refuses a GPU using >256 MiB.
  GPU guard is a point-in-time check; do not start evaluation on the training GPU.
  No new network downloads during training; W&B cloud logging disabled.

Commands (run on the host, not in the simulation Docker container)
  cd /study/wangmengfan/hbot/scripts
  bash run-pi05-lora.sh --check

  # Data preparation entry (already completed; no need to rerun for this move):
  bash prepare-pi05-data.sh --help

  # Optional repeat: choose a NEW unused name (v2 has already passed):
  bash run-pi05-lora.sh --exp-name hbot_lora_smoke_v3 --num-train-steps 5 --log-interval 1

  # Pilot training AFTER the smoke test passes:
  # Legacy starter example; choose batch/step budget and verify cooling first.
  bash run-pi05-lora.sh --exp-name hbot_lora_pilot_v1 --num-train-steps 20000

  # Resume the same experiment; existing checkpoints are never overwritten:
  bash run-pi05-lora.sh --exp-name hbot_lora_pilot_v1 --num-train-steps 20000 --resume
  # Resume only checkpoints from the SAME freeze strategy; optimizer state differs.

  # Select a different idle single GPU explicitly:
  HBOT_TRAIN_GPU=1 bash run-pi05-lora.sh --exp-name another_run

  # AFTER addressing temperature, optional repeat with a NEW name:
  ../openpi/.venv/bin/python train/benchmark-pi05-lora.py --batch-size 4 --steps 100 --warmup-steps 10 --exp-name hbot_lora_b4_stability_v2 --gpu 0
  # Benchmark runner stops ONLY its own training subprocess group at >=90 C.
  # This protection is in the benchmark runner, NOT the ordinary training launcher.
  # It is a conservative test cutoff, not a manufacturer temperature guarantee.

Outputs
  logs/pi05_lora_<timestamp>_<pid>.log
  checkpoints/pi05_hbot_lora/<exp-name>/<step>/
  Checkpoints contain more than just LoRA tensors; budget disk space accordingly.

Compatibility changes
  H-Bot adapter preloads pyarrow before the OpenPI/native-library imports to avoid
  a reproducible import-order segfault on this Ubuntu 20.04 environment.
  Official openpi/scripts/train.py changes only the JAX cache directory setting
  to honor JAX_COMPILATION_CACHE_DIR; the training algorithm is unchanged.
  Launcher sets that variable to hbot/cache/jax. Original source is backed up.

Next validation
  1. Start the approved guarded 280 W long job using the sudo launcher above.
     Confirm progress, power limit and temperature in the first 10--15 minutes;
     the short stability pass does not establish unattended long-run cooling.
  2. Implement pi0.5 adapter for robo_orchard_server (RGB order, gripper scale,
     state/action layout, action chunk/reset semantics) and compare with HoloBrain.

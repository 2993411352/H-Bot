H-Bot pi05 local service
=======================

Scripts live in /study/wangmengfan/hbot/scripts; configs live in config/.
Official OpenPI and RoboOrchard source files are not modified.
Uses openpi/.venv directly, without uv sync or changes to the HoloBrain venv.

1. CPU-only preflight (no GPU, weights, listener or simulator):
   bash /study/wangmengfan/hbot/scripts/run-pi05-server.sh --check

2. Start experimental pi05_base server after GPU 1 is free:
   bash /study/wangmengfan/hbot/scripts/run-pi05-server.sh

   Default: physical GPU 1, 127.0.0.1:8766.
   HoloBrain also uses GPU 1; this launcher refuses an occupied GPU by default.
   It never stops any existing process. --allow-busy-gpu is an explicit override,
   not a guarantee of sufficient memory. Do not override just to bypass the check.
   --gpu INDEX selects another GPU, but the simulator normally needs GPU 0.
   First real inference includes JAX compilation; CPU preflight does NOT verify
   GPU memory requirements, successful weight restore, or task performance.

3. In another terminal, after the server reports that it is listening:
   cd /study/wangmengfan/hbot/scripts
   bash run-local-eval.sh pi05_base_experimental hbot-local-pi05.yaml

   This independent YAML runs the same six tasks with one episode per task.
   run-local-eval.sh still records at most the first two episodes per task.
   The second argument is IMPORTANT: it selects port 8766, not HoloBrain 8765.
   Changing the first argument only changes the result label, not the model.
   If changing --port, also change config/local_server_pi05.yaml to match.
   Do not run another evaluation concurrently; the evaluation wrapper rejects it.
   Ctrl+C in the server terminal stops this server, not Docker or other servers.

EXPERIMENTAL BASE PROFILE (not an official baseline)
---------------------------------------------------
Official DROID commands are joint velocities + absolute normalized gripper.
The canonical simulator server expects joint POSITION targets. This local
adapter is an explicitly experimental approximation, not calibrated DROID control:
  v = clip(model_command, -1, 1) * 0.25 rad/s
  q[k] = q_observed + sum(v[0:k]) / 30
Each 15 Hz command is repeated twice for the current 30 Hz simulation.
Execute the first 15 simulation steps per chunk, then use a new observation.
Limit chunk displacement to 0.15 rad and clamp to local Panda URDF joint limits
with a 0.01 rad margin. Reject invalid layouts, NaN actions and out-of-bounds states.
Binary gripper: model output >0.5 means closed; convert via the canonical codec.
These guards are NOT collision avoidance or a real-robot safety guarantee.
The fixed 30 Hz setting must be rechecked if simulator control timing changes.

Base mode uses ext1_camera + wrist_camera, matching OpenPI's DROID input adapter.
ext2_camera is NOT inserted into the masked second-wrist slot of the base model.
Stats come from pi05_base/assets/droid, not the H-Bot training stats.
For this reason and the approximate controller, results are communication/inference
smoke-test results ONLY, not comparable official baseline scores.

References (also checked in the local pinned OpenPI source):
https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md
https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md

MATCHING H-BOT LoRA CHECKPOINTS
-----------------------------
After training produces a saved step, use its directory containing params/ and
assets/, NOT its params/ subdirectory and not pi05_base:
   bash run-pi05-server.sh --profile hbot-lora --checkpoint /absolute/path/to/STEP --check
   bash run-pi05-server.sh --profile hbot-lora --checkpoint /absolute/path/to/STEP

This profile reuses scripts/train/pi05_hbot.py (pi05_hbot_lora, horizon 15), all three
training cameras and the checkpoint's H-Bot normalization. OpenPI restores joint
deltas to absolute targets; NO velocity integration is applied. It expects the
complete OpenPI checkpoint with LoRA tensors, not adapter-only files or a merged
checkpoint made for a different model configuration. Revalidate if training changes.
No trained H-Bot checkpoint is created by this setup, and no training is started.

MATCHING H-BOT FULL-FINETUNED CHECKPOINTS
--------------------------------------
Use hbot-full for a non-LoRA parameter tree trained with the SAME H-Bot setup:
pi05, horizon 15, all three cameras, 7 joint deltas + absolute gripper, quantile
normalization, and the checkpoint's hbot_panda_joint_delta_h15 assets.
Only the model variants change to gemma_2b and gemma_300m. Camera mapping and
delta-to-absolute output conversion are identical to hbot-lora, not DROID base.

The current full 30k export (params/ + assets/; no numeric step directory needed):
   bash /study/wangmengfan/hbot/scripts/run-pi05-server.sh \
     --profile hbot-full \
     --checkpoint /study/wangmengfan/hbot/checkpoints/pi05_hbot_full/hbot_pi05_30k \
     --gpu 1 --port 8766 --check

After the CPU preflight passes, omit --check to load weights and start the server.
The CPU check validates transforms and parameter shapes; it does not load weights
or verify GPU inference. Training optimizer state is not needed for inference.
Do not use hbot-full for arbitrary full models with different training semantics.

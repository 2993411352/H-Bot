# H-Bot π₀.₅ 全参数 SFT 使用教程

适用于单卡 A800 80GB。以下命令均在目标 Linux 服务器执行。

## 1. 导入镜像

宿主机需安装 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit，当前用户能够运行 Docker。训练时需要一张空闲的完整 80GB GPU；建议主机内存至少 128GB，输出磁盘预留 200GiB 以上（启动最低检查 100GiB）。

将镜像及同名校验文件放在同一目录，执行：

```bash
sha256sum -c hbot-pi05-full-sft-215abfb-a800-v2.image.tar.gz.sha256
docker load -i hbot-pi05-full-sft-215abfb-a800-v2.image.tar.gz
```

镜像已包含 Python、训练依赖、源码、数据适配、归一化统计和分词器，无需在宿主机配置 Python 环境。数据集与基础权重需单独准备。

## 2. 提取启动脚本

在一个新的工作目录执行，避免覆盖自己的同名脚本：

```bash
mkdir -p hbot-full-sft-launcher/scripts
cd hbot-full-sft-launcher
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint cat \
  hbot/pi05-full-sft:215abfb-a800-v2 \
  /opt/hbot-sft/scripts/run-sft.sh > scripts/run-sft.sh
```

后续命令均在此目录执行。

## 3. 下载数据集与 π₀.₅ 权重

先设置路径；输入与输出使用两个互不包含的绝对路径：

```bash
export HBOT_INPUT_ROOT=/data/hbot-input
export HBOT_OUTPUT_ROOT=/data/hbot-full-sft-output
mkdir -p "$HBOT_INPUT_ROOT/models/pi05_base"
mkdir -p "$HBOT_INPUT_ROOT/datasets/InstructMove_hbot/lerobot_unpacked"
mkdir -p "$HBOT_OUTPUT_ROOT"
```

### LeRobot v2.1 数据集

官方下载页面：

https://huggingface.co/datasets/HorizonRobotics/InstructMove-Dataset/tree/h-bot/horizon_isaac/lerobot

本镜像的输入校验清单对应以下固定版本，建议从此处下载六个任务的压缩包，避免分支后续更新导致校验不一致：

https://huggingface.co/datasets/HorizonRobotics/InstructMove-Dataset/tree/44e8964b83fbcf7575f66dddbfc36c3a8d06ec06/horizon_isaac/lerobot

解压后按下方结构放置，每个任务目录内直接包含 `meta/`、`data/` 和 `videos/`，不要重复嵌套一层目录。只需 LeRobot 数据，不需要 LMDB 或仿真资产。

### π₀.₅ Base 权重

官方权重说明：

https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479#base-models

官方下载地址（Google Cloud Storage）：

```text
gs://openpi-assets/checkpoints/pi05_base
```

需要下载整个 `pi05_base/`，包含 `params/`、`assets/` 和所有元数据。本镜像使用 JAX/Orbax 权重，不要替换成 PyTorch 转换版、`pi05_droid` 或 LoRA checkpoint。

如果下载机器已安装 `gsutil`，可执行（不删除本地已有文件）：

```bash
gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_base \
  "$HBOT_INPUT_ROOT/models/pi05_base"
```

也可以把旧服务器上已经下载完成的完整权重和解压数据复制过来，无需重复下载。最终目录应为：

```text
/data/hbot-input/
├── models/pi05_base/
│   ├── params/
│   └── assets/
└── datasets/InstructMove_hbot/lerobot_unpacked/
    ├── panda_droid-joint_direction-20261013/
    ├── panda_droid-open_drawer-20261013/
    ├── panda_droid-pick_color-20261013/
    ├── panda_droid-place_a2b-20261013/
    ├── panda_droid-spatial_pick-20261013/
    └── panda_droid-stack_cubes-20261013/
```

## 4. 校验输入并验证 batch size

在同一个终端设置训练参数；新开终端后也需要重新设置这些变量及第 3 节的路径变量：

```bash
export HBOT_GPU=0
export HBOT_BATCH_SIZE=16
export HBOT_STEPS=100000
export HBOT_EXP_NAME=pi05_full_b16_100k_v2

bash scripts/run-sft.sh check
bash scripts/run-sft.sh smoke
```

`check` 使用 CPU 校验权重、数据文件和数据加载；`smoke` 在 GPU 上真实训练 20 步并保存 checkpoint。首次编译需等待，全部完成才算通过。

batch=16 是待目标 A800 验证的起始值，不是显存保证。若显存不足，改为 8 或 4 后重新 smoke；若有充足余量，可改为 32、64 后重新 smoke。正式训练必须使用已通过 smoke 的相同 batch 和训练计划。每次 smoke 的完整 checkpoint 可能约 38GiB，测试多个 batch 前检查磁盘空间。

## 5. 启动 10 万步训练

保持上一步已验证的参数，执行：

```bash
bash scripts/run-sft.sh train
bash scripts/run-sft.sh logs
```

默认全参数训练 100000 个优化器更新步，学习率衰减同步为 100000 步，无梯度累积；默认每 1000 步保存 checkpoint。若更换 batch，请使用新的实验名，例如 `pi05_full_b32_100k_v2`。

训练在后台容器中运行。SSH 断开或对 `logs` 按 Ctrl+C 不会停止训练。checkpoint 等产物保存在 `$HBOT_OUTPUT_ROOT`，输入只读挂载；训练容器默认断网。

另开终端查看 GPU：

```bash
watch -n 2 nvidia-smi
```

## 6. 停止与断点恢复

保持原来的路径、实验名、batch 和步数变量。停止当前实验：

```bash
bash scripts/run-sft.sh stop
```

恢复前，先保存日志并确认对应容器已停止，再移除这一只旧容器：

```bash
docker logs "hbot-pi05-full-$HBOT_EXP_NAME" > "$HBOT_EXP_NAME.log" 2>&1
docker inspect --format '{{.State.Status}}' "hbot-pi05-full-$HBOT_EXP_NAME"
# 确认上一步为 exited 后执行；不要添加 -f。
docker rm "hbot-pi05-full-$HBOT_EXP_NAME"
bash scripts/run-sft.sh resume
```

移除容器不会删除挂载在输出目录中的 checkpoint。恢复从最后一个完整保存点继续，参数需与原训练一致；若尚无完整 checkpoint，使用新实验名重新 `train`。

## 7. 进入容器环境

未运行训练且 GPU 空闲时，可进入交互环境检查：

```bash
bash scripts/run-sft.sh shell
```

退出交互容器使用 `exit`。正式训练使用上面的 `train` 命令。

# 验收记录（2026-10-28）

## 已通过

- v2 已通过19项 CPU 单元/启动器测试，包括100000步/LR衰减同步、batch16默认值、短跑必须20步、同 batch 成功记录的检查，以及原有全参数/数据/启动器保护。
- 所有 shell 脚本 `bash -n` 语法检查。
- 2483 个输入文件共 15,453,019,232 bytes，逐个大小 + SHA256/CRC32C 校验通过（官方 `commit_success.txt` 合法地为空）。
- 600 episodes / 156867 帧数量核对通过。
- 六任务各首/尾真实样本，共 12 个样本的三相机解码、各自任务语言、归一化与反归一化、末帧15步重复填充检查通过。
- 与实际训练相同的 spawned worker 数据读取通过。
- 用全新空 Hugging Face 缓存、开启 offline 环境变量，再次完成本地数据读取/三相机/动作往返测试；分词器仅从包内文件提供。不需要复制旧 `.venv` 或 Hugging Face 缓存。
- 全参数抽象模型校验：51 个参数张量，与 base checkpoint 结构和形状逐项匹配；**3,353,433,872 总参数 = 可训练参数**，冻结/LoRA 参数为 0。

- Docker 镜像已实际构建成功：`hbot/pi05-full-sft:215abfb-a800-v2`，Linux/amd64，Image ID 为 `sha256:6dea27ab483bdab2a3073337cdaa7afc1af320da62e1b3dfbc5380ade0d521c1`，未压缩大小 8,882,249,908 bytes。
- 构建内19项测试通过；额外以 `nobody` 用户通过环境/模型结构检查。修复了源码复制权限过严导致普通用户无法读取入口文件的问题。
- **实际容器 CPU 验收通过**：使用非 root UID/GID、`--network none`、`NVIDIA_VISIBLE_DEVICES=void`、CPU JAX、只读权重/数据挂载及全新输出缓存；2483文件校验、600 episodes / 156867帧、12个真实样本的三相机/语言/动作往返、batch16 + 2个 spawned workers 均通过。
- 镜像布局检查通过：无原服务器 `/study/wangmengfan/hbot` 目录依赖、无失效 venv 符号链接、无整份 uv 下载缓存，环境变量未持久化代理或凭据，本次回环代理地址也未进入 image history。
- 已导出 `hbot-pi05-full-sft-215abfb-a800-v2.image.tar.gz`，4,361,688,093 bytes，SHA256 `1b11caf8ff1388c36cde3b7c8431af04920fc7f821e65179b81ebb280e00e992`；gzip 完整性测试通过，实际 `docker load` 成功，导入后的 Image ID 与原镜像一致。

`validation/cpu-preflight.json` 是此前宿主环境报告；**`validation/container-cpu-preflight.json` 是本次真实 Docker 容器报告**，仍不是 A800 GPU 训练报告。完整构建日志与 Python/系统依赖清单也在 `validation/` 中。目标机必须重新执行 check，报告会按其真实输入重新生成到输出目录。

## 构建说明

早期基础镜像网络失败以及 apt 的502已绕过/修复。此次采用构建专用临时代理、保留签名校验的 Ubuntu 镜像源，以及只读复用的 uv 缓存完成构建；未修改 Docker daemon、宿主软件源或原有代理服务。导出文件的哈希及回读结果见交付目录 `image-manifest.json`。

## 尚未完成 / 不作保证

- 没有对旧服务器的 3090 启动新 GPU 训练，没有修改/停止既有训练或仿真进程，也没有调整功耗。
- 没有目标 A800 的连接权限，因此 **未实测 A800 峰值显存、步时、20步训练、checkpoint 保存/恢复**。代码提供对应的 smoke 路径和恢复保护；需在目标机执行，不应把 CPU shape 测试当成完整训练成功。
- 没有训练/验证集划分或模型效果验收；当前是全600条训练数据，不提供泛化或比赛得分保证。

## 版本

- OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`，源码快照未修改。
- Python 3.11.17 / JAX & jaxlib 0.5.3 / Flax 0.10.2 / PyTorch 2.7.1。
- PyArrow 20.0.0 / PyAV 17.0.0 / Polars 1.30.0 / Orbax 0.11.13。
- Dockerfile 基于官方原始 `uv.lock`，安装 `--frozen --no-dev`，不包含 RLDS 可选依赖组。

Torchvision 视频 API 退役提示与 JAX/Flax deprecation 警告在固定版本下不影响本次 CPU 验收；不应为了消除提示擅自升级单个依赖。

## v2 更新范围

- 默认100000个优化器更新步，LR decay=100000，候选 batch16，无梯度累积。
- 镜像标签改为 `215abfb-a800-v2`，不混用旧v1镜像和启动器。
- 正式训练必须有同一 batch、GPU类型、代码、输入、worker及LR计划的20步短跑成功记录。
- batch16/32/64 尚未在 A800 上实测；新增检查不会把 CPU 测试当成显存容量证明。
- v2 使用实际 batch16、2个 spawned workers 重新通过完整 CPU preflight：2483文件校验、六任务视频/语言/动作往返和51张量权重形状均通过；详细报告已更新。

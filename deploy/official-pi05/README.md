# 在新机器上运行 π₀.₅ 正式测评服务

这是本项目正式服务的可迁移版本，支持旧 B16 · 100k JAX 权重与本轮 LoongForge/PyTorch 权重。保留已使用的 `run-official.sh`、`--check` 和 `--self-test` 流程，去除原机器目录、账户和桌面代理依赖。

选择与权重匹配的后端，两种环境独立安装：

| `HBOT_BACKEND` | 权重格式 | 运行环境 |
|---|---|---|
| `loongforge`（配置模板默认） | 本轮 reviewed-v3 的 `steps_N/model.safetensors` 和配套元数据 | Python 3.12、Torch 2.10.0+cu128、Transformers 5.3.0 |
| `jax` | B16 · 100k 的 `99999/params/` 与 `assets/` | Python 3.11、JAX 0.5.3、Flax 0.10.2 |

两种后端均固定三路 RGB、15 步动作、10 次去噪、PandaDroid 关节位置接口；不启用事件触发规则。不做 JAX↔Torch 权重转换。LoRA、DROID base 或其他未核验的训练契约不能直接替换进来。

正式测评的仿真 Client 运行在平台。你这里只运行模型服务，主动连接平台 WSS，无需在本机安装 Isaac Sim、Docker 或数百 GB 的仿真资产，也无需开放公网监听端口。本地 Isaac 仿真评测是另一条流程，见文末。

## 1. 机器与依赖

- Linux x86_64；本项目原工作环境为 Ubuntu 20.04、RTX 3090 24GB、NVIDIA 驱动 570.133.20。新安装脚本分别创建独立 Python 3.11（JAX）或 3.12（LoongForge）环境。
- 选择一张空闲 NVIDIA GPU。24GB 卡是已有工作配置；其他架构/显存容量须以本机 `--self-test` 为准。本入口默认拒绝占用已有计算任务的卡，不结束其他进程。
- 建议至少 32GB 主机 RAM，并为环境、下载缓存和 checkpoint 预留约 60GB 磁盘；实际用量取决于权重体积。
- 能访问 GitHub、PyPI、Python 下载源、Google Storage，以及比赛平台的 HTTPS/WSS。安装可走代理；正式连接的网络模式另行配置。
- NVIDIA 驱动由机器管理员预先安装，先确认 `nvidia-smi` 正常。脚本不会改驱动、功率、系统服务或防火墙。

Ubuntu 缺依赖时由管理员执行（已有的不必重复安装）：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates python3 python3-venv \
  openssl util-linux libgl1 libglib2.0-0 tmux
```

## 2. 下载代码并安装环境

```bash
git clone https://github.com/2993411352/H-Bot.git
cd H-Bot/deploy/official-pi05
cp local.env.example local.env
```

修改 `local.env`，至少设置后端、权重目录与 GPU 编号。使用 LoongForge 权重时保留模板中的 `HBOT_BACKEND=loongforge`。若要换安装盘，在运行 setup **前** 设置 `HBOT_RUNTIME` 的绝对路径。默认安装在本目录的 `runtime/`，不修改系统 Python、已有训练环境或 conda base。

```bash
bash setup.sh
```

安装脚本按后端执行：JAX 克隆固定 OpenPI commit 并使用上游 `uv.lock`；LoongForge 使用随包保留的最小推理源码和 `requirements.lock`（包括 CUDA 12.8 Torch wheel）。然后下载并校验 tokenizer，运行 CPU 导入、参数结构和协议测试。不会在推理机器安装训练系统。成功末尾是 `SETUP_OK`。重复运行可复用缓存；若既有 OpenPI checkout 被修改，脚本会停止，不覆盖它。

JAX 关键版本：JAX/jaxlib `0.5.3`、Flax `0.10.2`、Orbax `0.11.13`、Torch `2.7.1`、NumPy `1.26.4`、websockets `15.0.1`。JAX 导入验收写入 `runtime/installed.json`，LoongForge 写入 `runtime/installed-loongforge.json`。不要把两个后端装进同一个 venv。

若安装网络需要代理，可先在当前终端设置 `HTTPS_PROXY` / `HTTP_PROXY`；脚本继承安装时的网络环境。不要用跳过 TLS 验证来处理网络错误。

## 3. 准备完整的推理 checkpoint

### LoongForge / PyTorch（当前新机器使用这一项）

下载整个推理快照目录，至少保留以下 7 个文件：

```text
steps_30000/
├── model.safetensors
├── model_config.json
├── processor_config.json
├── hbot_contract.json
├── norm_stats.json
├── resume_meta.json
└── verified.json
```

不需要 `training_state.pt`。`verified.json` 应是训练器生成的原文件，不可手改它来跳过检查。下载中的临时目录、文件长度/哈希不符、验收短测 checkpoint 都会被拒绝。检查会读取并哈希整个模型文件，可能需要几十秒，期间不占用 GPU。

`local.env` 设置：

```bash
export HBOT_BACKEND=loongforge
export HBOT_PI05_CHECKPOINT="/home/user/checkpoints/steps_30000"
export HBOT_GPU=0
export HBOT_OFFICIAL_NETWORK=direct
```

该后端保留本轮 H20 reviewed-v3 训练约定：三相机顺序为外部1→腕部→外部2，PIL 等比例双线性缩放与补零到 224，原 SentencePiece tokenizer，8维状态分词，q01/q99 归一化（分母加 1e-6），7关节增量相对**同一观测状态**还原，夹爪为绝对值，FP32 参数和 BF16 autocast。严格加载全部 813 项参数，不覆盖 embedding 与独立 lm_head，不启用量化或推理 compile。

当前锁定的是已验证训练 run `8fc94f26b3da464c83d673010894b764` 的正步数快照，例如 5k/10k/15k/25k/30k。新训练 run 即使也叫 π₀.₅，仍须核对契约、统计量、模型配置和训练器元数据；本包会拒绝未经核验的替换，不会静默套用旧统计量。

LoongForge 使用实际 PandaDroid 的 `panda_joint1..7 + finger_joint`、夹爪比例 `4/pi`。不要改成通用 Franka 的两手指协议或硬编码九维输出；适配器会检查输入 `action_layout` 是否匹配。

### JAX / OpenPI（旧 B16 · 100k）

权重不放进本 Git 仓库，需要从你自己的训练机或模型仓库复制。目录应类似：

```text
99999/
├── assets/
│   └── hbot_panda_joint_delta_h15/
│       └── norm_stats.json
└── params/
    ├── _METADATA
    ├── _CHECKPOINT_METADATA       # 具体文件取决于 Orbax 版本
    ├── manifest.ocdbt
    ├── d/...
    └── ocdbt.process_0/...        # 如原目录存在，必须一起复制
```

**复制整个 `params/` 和整个 `assets/`，保留相对结构。** 不要只复制 `_METADATA` 或一个分片，也不要把 `--checkpoint` 指到 `params/`。纯推理不需要优化器 `train_state`。使用这份权重自己的 `norm_stats.json`，不能拿另一轮训练的统计量替换。

JAX 的 `local.env`：

```bash
export HBOT_BACKEND=jax
export HBOT_PI05_CHECKPOINT="/home/user/checkpoints/99999"
export HBOT_GPU=0
export HBOT_OFFICIAL_NETWORK=direct
```

`--backend jax|loongforge`、`--checkpoint /absolute/path/to/checkpoint`、`--gpu 0` 可以覆盖配置。安装也可用 `bash setup.sh --backend loongforge` 或 `--backend jax` 分别安装，常规使用只安装需要的一项。GPU 是本机器 `nvidia-smi` 中的物理编号，程序内部只暴露选中的一张卡。

## 4. 申请正式链接前先完成两级检查

```bash
# CPU：路径、归一化维度、三相机变换、动作逆变换、参数形状
bash test.sh --check

# GPU：真正加载权重，六任务预热，执行原有本地 TLS/WSS 测试
bash test.sh --self-test --gpu 0
```

`--check` 成功输出 JAX 的 `CPU_PREFLIGHT_PASSED` 或 LoongForge 的 `CPU_CHECKPOINT_PASSED`，**不会恢复完整权重或验证 GPU 推理**。`--self-test` 成功输出 `REAL_WEIGHT_LOCAL_WSS_PASSED`，才证明这台机器能加载权重并通过本地协议测试。两者都不会使用正式评测链接或消耗平台次数。

`--self-test` 沿用原来的 `smoke_wss.py` 检查：TLS 证书验证、`sessions-v1` 握手、三相机输入、与输入关节布局一致的 float32 动作、Client 状态/RNG 隔离、reset 清理未执行动作、close/recreate，以及 4/8 个逻辑 Client 的请求突发。请求仍串行推理；突发测试不等于启动 8 个模型。

为让新机器不依赖训练集，默认用六种指令和合成 RGB 作为协议样本，报告明确标记 `fixture_source=synthetic_rgb`。**这不是任务得分测试。** 如果已下载六任务 LeRobot 数据，可完全复用原来的真实观测样本：

```bash
bash test.sh --self-test --gpu 0 --fixtures /absolute/path/to/lerobot_unpacked
```

该目录应含 `panda_droid-{pick_color,spatial_pick,place_a2b,stack_cubes,open_drawer,joint_direction}-20261013/` 六个子目录，内含首个 episode 的 parquet、三路 MP4 与 `meta/tasks.jsonl`。不提供该参数就不会读取任何数据集。

只有检查通过且网络稳定后，再去平台申请链接，以免消耗链接有效时间。

## 5. 启动正式测评

建议在 **tmux** 中运行，避免关闭 SSH 终端影响进程：

```bash
tmux new -s hbot-official
cd /absolute/path/to/H-Bot/deploy/official-pi05
bash run-official.sh --gpu 0
```

看到六个 `warmup` 以及 `MODEL_READY` 后，在隐藏输入提示处粘贴完整 WSS 链接并回车。链接不回显、不写入配置文件。看到 `WSS_NEGOTIATION_PASSED` 以及持续追加的请求日志，才说明已连接并开始接收 Client 请求；`MODEL_READY` 本身只是权重加载完成。

按 **Ctrl+B，再按 D** 离开 tmux。重新查看：

```bash
tmux attach -t hbot-official
```

不要在正式运行终端按 Ctrl+C，不要重复连接同一个 Session。入口只连接一次，没有自动重连/续评；连接断开后先查看平台状态，再按平台规则处理。关闭 SSH 不影响 tmux 内进程，但如果 WSS 依赖电脑上的代理/SSH 隧道，关闭电脑或断开该隧道仍会切断测评网络。

必须使用 URL 文件自动化时，可把链接存入权限 `600` 的本地文件，再传 `--supervisor-uri-file /path/to/private-session.txt`。优先用交互隐藏输入；不要把真实链接放在命令行参数、Git、截图或公开日志中。

## 6. 直连、代理与日志

`HBOT_OFFICIAL_NETWORK=direct` 是默认值，启动时清除继承的 HTTP/HTTPS/WS 代理。若服务器确实需要代理，改成 `inherit` 并在 `local.env` 设置 `HTTPS_PROXY`。代理应支持 HTTPS CONNECT/WSS。不会自动创建桌面代理转发，也不会隐式修改网络配置。自测访问回环地址时会绕过代理，并在测试结束后恢复进程内设置。

每次运行创建独立 `runtime/logs/<backend>_official_<时间>_<PID>/`，也可传 `--output-dir /new/path`（目录必须尚不存在）。

| 文件 | 意义 |
|---|---|
| `ready.json` | 权重、GPU、预热、归一化 SHA256、推理配置 |
| `connection_opened.json` | WSS 协议协商完成 |
| `requests.jsonl` | 每个逻辑 Client 的 reset、指令、请求次数、服务耗时和错误类型 |
| `server.log` | 服务日志 |
| `wss_smoke.json` | 本地自测结果，包含合成/真实样本来源 |
| `transport_closed.json` / `connection_error.json` | 连接关闭或连接异常 |

请求次数、进程退出码 0、WSS 正常关闭，都**不能当作任务成功率或全量完成凭据**。正式分数、成功率和最终状态以平台报告为准。应用日志的单次服务耗时不包含全部平台网络/排队时间。

## 7. 常见故障

| 现象 | 处理 |
|---|---|
| `source: not found` | 使用 `bash setup.sh` / `bash run-official.sh`，不要用 `sh` |
| 参数结构不匹配 | 确认选择了权重对应的后端，采用匹配的训练契约；JAX/LoongForge 不能混用环境或目录 |
| 缺 tokenizer | 重跑 setup，或检查 `OPENPI_DATA_HOME`，不要等正式连接后隐式下载 |
| `GPU ... already occupied` | 选空闲卡或自行结束自己的其他任务；脚本不清理进程 |
| GPU/CUDA/显存错误 | 先解决本机 `--self-test`；不通过就不要申请正式链接 |
| 版本或冻结文件校验失败 | 使用未修改的部署包及其独立环境；不要在运行时随意升级依赖 |
| 安装下载报 SSL EOF / timeout | 检查当前代理、DNS、CA 与包源；保留 TLS 验证，恢复网络后重跑 |
| 平台没有分数但连接结束 | 查看平台执行状态和服务日志；传输关闭不代表六任务完成 |

## 8. 与本地仿真测试的关系

本包的 `test.sh --self-test` 是模型/接口验收，不启动 Isaac，也不产生仿真成绩。若需要本地物理仿真，继续使用仓库现有 `scripts/run-local-eval.sh`、`scripts/run-pi05-server.sh` 与 `config/docker-compose.sim.yml`，并按 [`config/README.md`](../../config/README.md) 准备 Isaac 容器、资产和主机挂载；这些历史脚本仍包含原服务器路径，需要按机器调整。不要把正式 WSS 链接用于本地烟雾测试。

来源、第三方说明和本次验证边界分别见 [THIRD_PARTY.md](THIRD_PARTY.md)、[VALIDATION.md](VALIDATION.md)。

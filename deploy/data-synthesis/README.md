# H-Bot 合成数据采集

在自己的 Linux 机器上安装环境并采集数据，支持 RTX A5000、RTX 5070 Ti 等 NVIDIA RTX 显卡。脚本自动检测 GPU，并构建对应的 Docker 环境，无需提前安装 RoboOrchardSim。

**每人每个任务采集 400 条合格数据，共 1600 条。**

| 任务 | 命令中的任务名 | 每人合格数量 |
|---|---|---:|
| 颜色抓取 | `pick_attribute` | 400 |
| 空间抓取 | `spatial_directions` | 400 |
| 物体放置 | `place_a2b` | 400 |
| 堆叠方块 | `stack_cubes` | 400 |

当前脚本不采集 Open_Drawer 和 Joint_Direction。合成数据无需模型权重或平台测评链接；正式模型测评使用 [official-pi05 部署包](../official-pi05/README.md)。

## 安装环境

准备原生 Ubuntu 22.04/24.04 x86_64、可用的 NVIDIA 驱动，以及足够保存资产和采集数据的磁盘空间。建议 64 GB 以上内存、至少 1 TB 空余磁盘。Windows 用户通过 SSH 连接 Linux 机器操作。

```bash
# 缺少以下工具时先安装
sudo apt-get update && sudo apt-get install -y git python3 tmux
nvidia-smi
git clone https://github.com/2993411352/H-Bot.git
cd H-Bot/deploy/data-synthesis

# 已安装 Docker 和 NVIDIA Container Toolkit 时可跳过
sudo bash install-host.sh --apply
```

如果当前用户没有 Docker 权限，由管理员添加权限后重新登录：

```bash
sudo usermod -aG docker "$USER"
```

复制配置文件并编辑本机路径：

```bash
cp local.example.json local.json
```

| 配置项 | 含义 |
|---|---|
| `gpu` | 使用的 GPU 编号，例如 `0` |
| `assets` | 仿真资产根目录，包含 `OBJECTS_v2`、`ROBOTS`、`NVIDIA` |
| `work` | 采集结果保存目录 |
| `reference` | 团队共享去重索引目录 |
| `image_prefix` | Docker 镜像名前缀，一般无需修改 |
| `accept_eula` | 阅读并接受 [NVIDIA 软件许可](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)后设为 `true` |

```bash
python3 manage.py doctor
python3 manage.py build
```

脚本按 GPU 架构编译 cuRobo。A5000 使用标准模式，5070 Ti 自动使用较低显存的规划模式。每张 GPU 运行一个采集 Client，不需要桌面环境。

## 准备资产

没有资产时执行：

```bash
python3 manage.py download-assets
python3 manage.py assets-check
```

使用官方 `HorizonRobotics/robo_orchard_sim_assets` 的 `instructmove_v2` 版本，支持断点续传。已有完整资产时，将 `assets` 配置为现有目录，然后执行 `assets-check` 即可。

## 准备共享去重索引

从项目协调者获取 `team-reference` 目录，其中包含：

```text
team-reference/
├── registry.sqlite3
└── reference.json
```

将 `local.json` 中的 `reference` 指向此目录。采集时会检查与已有数据及本机已采集数据是否重复，汇总时还会检查参与者之间的重复数据。

协调者可从项目现有去重库导出共享索引：

```bash
python3 snapshot-reference.py \
  --registry /absolute/path/to/dedup/registry.sqlite3 \
  --output /absolute/path/to/team-reference
```

索引通过团队文件传输共享，不提交到公开 GitHub。

## 开始采集

每位参与者使用不同的 `--worker` 编号，例如 `101`、`102`。首次运行直接开始正式采集，每个任务默认 400 条。

```bash
tmux new -s hbot-collect
python3 manage.py collect --worker 101 --quota 400 --max-attempts 2000
```

按 `Ctrl+B` 后按 `D` 离开 tmux，断开 SSH 后任务仍会运行。再次进入：

```bash
tmux attach -t hbot-collect
```

四个任务依次执行，失败或重复的候选不计入 400 条配额。结果保存在 `work/worker_0101/`：

```text
worker_0101/
├── plan.json             # 本次运行配置
├── status.json           # 进度和状态
├── manifest.json         # 合格数据清单
├── config_snapshot/      # 任务配置
├── batches/              # 各候选的日志、MCAP 和质量检查
└── dedup/                # 去重记录
```

## 查看进度与暂停

```bash
python3 manage.py status --worker 101
python3 manage.py pause --worker 101
```

暂停会等待当前候选结束。确认状态为 `paused` 后，如需继续，删除对应暂停标记，再使用原参数续采：

```bash
rm /data/hbot/collection/worker_0101/PAUSE
python3 manage.py collect --worker 101 --quota 400 --max-attempts 2000 --resume
```

将示例路径替换为自己的 `work` 路径。只有同一运行续采时复用 worker 编号；新增采集使用新编号，避免重复种子。

## 汇总数据与导出视频

将各参与者完整的 `worker_XXXX` 目录复制到协调者的 `work` 目录，保留原目录名。先进行跨机器去重：

```bash
python3 manage.py merge-check --workers 101 102 --name team_v1
```

结果在 `work/merged/team_v1/`，包括 `accepted.json`、`rejected.json` 和汇总报告。源文件不会被删除。

然后分别导出 LeRobot 数据和三相机 MP4：

```bash
python3 manage.py export --worker 101 --name export_v1
python3 manage.py export --worker 102 --name export_v1
```

导出路径为 `worker_XXXX/exports/export_v1/lerobot_v21/`，视频在其中的 `videos/` 目录。训练时使用汇总通过且人工审核通过的数据，并配合导出的 `training_anchor_masks.json` 过滤无效窗口。

## 可视化回放与标注

导出完成后，启动之前使用的三相机标注工作台：

```bash
python3 manage.py viewer --workers 101 --export-name export_v1 --name review_v1
# 同时查看多位参与者的数据
# python3 manage.py viewer --workers 101 102 --export-name export_v1 --name team_review_v1
```

首次运行自动安装独立的 CPU 网页环境，不占用采集 GPU。看到 `VIEWER_READY` 后，在自己的电脑终端或 PowerShell 中建立转发：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:18765:127.0.0.1:18765 用户名@采集机器IP
```

SSH 端口不是 22 时，在命令中加入 `-p 端口号`。保持转发终端打开，然后用浏览器访问 **http://127.0.0.1:18765**；在采集机器本机使用浏览器时无需转发。

网页可同步播放两路外部相机和腕部相机，逐帧定位，查看语言指令、夹爪曲线和种子，标注阶段、质量、堆叠循环及关键区间。左侧按实际轨迹数量统计审核进度，`W101` 等标记区分不同采集者。

1. 选择任务与轨迹，播放视频并逐段确定边界。
2. 用 `I`、`O` 标记起止帧，选择阶段与质量，添加区间；区间包含起止帧。
3. 可先保存草稿；补齐全轨迹并填写标注者后，点击“确认审核完成”。
4. 点击“导出标签”下载 JSON。需排除的片段标为“不用于正向模仿”，后续训练时据此筛选。

标签和历史版本保存在 `work/reviews/review_v1/`，原视频与 Parquet 只读。再次运行相同命令可继续审核；更换导出批次或参与者列表时使用新的 `--name`。备份时先停止网页，再复制整个审核目录。网页不会自动修改训练配置。

网页也可在独立 tmux 会话中运行。端口被占用时加 `--port 18766`，并将转发命令两侧端口相应替换。更新网页代码后运行原命令加 `--rebuild` 即可。

版本、质量检查及实现说明见 [技术说明](TECHNICAL_NOTES.md)。

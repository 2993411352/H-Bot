# Windows 正式测评部署

在 Windows PowerShell 中执行入口，模型在 WSL2 Ubuntu 内运行。默认使用 LoongForge / PyTorch 权重，复用本目录的安装脚本、PandaDroid 适配和正式 WSS 服务。平台负责仿真。

## 准备机器

- Windows 11 或支持 WSL2 的 Windows 10，x86_64 CPU、NVIDIA GPU。
- 安装支持 WSL CUDA 的 Windows NVIDIA 驱动。WSL 使用 Windows 提供的 GPU 驱动接口。
- 原 LoongForge 推理配置在 RTX 3090 上约使用 14.8 GiB 显存，建议使用 24GB 卡，并以这台机器的 SelfTest 结果确认可用性。
- 建议 32GB 以上内存，并在 WSL 所在磁盘预留至少 60GB 安装和权重空间。

WSL 的安装和 GPU 支持可参阅 [Microsoft WSL 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)及 [CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)。

## 一键配置

将仓库下载或克隆到 Windows，在仓库根目录打开 PowerShell。以下命令兼容 Windows PowerShell 5.1 和 PowerShell 7。

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Setup
```

默认使用 `Ubuntu-22.04`。缺少该发行版时，脚本调用 WSL 安装。若 Windows 提示需要提权，在管理员 PowerShell 中运行 `wsl --install -d Ubuntu-22.04`，按提示重启、创建 Ubuntu 用户，再执行同一条 Setup 命令。已有其他 Ubuntu 发行版可在每次命令中追加 `-Distribution Ubuntu`。

Setup 会完成以下操作。

1. 检查 WSL2 和 x86_64 架构。
2. 在 Ubuntu 中通过 sudo 安装基础依赖，密码输入 Ubuntu 用户的密码。
3. 将部署代码复制到 Linux 用户目录 `~/hbot-official/package`。
4. 在 `~/hbot-official/runtime` 创建独立 Python 3.12 和固定版本的 PyTorch 推理环境。
5. 下载并校验 tokenizer，执行现有 CPU 导入和协议测试。

安装结束应出现 `SETUP_OK`。网络中断后可重新执行 Setup 复用已下载缓存。需要选择 Linux 安装位置时，在每次命令中带上相同的 `-InstallDirectory /home/alice/hbot-official`。安装目录放在 WSL 的 Linux 文件系统中。

## 放置权重并配置

准备完整的 LoongForge 推理快照，例如 `steps_30000`，包含以下文件。

```text
model.safetensors
model_config.json
processor_config.json
hbot_contract.json
norm_stats.json
resume_meta.json
verified.json
```

训练状态 `training_state.pt` 可保留在训练机。权重兼容范围和元数据校验规则见 [LoongForge 权重说明](README.md#loongforge--pytorch当前新机器使用这一项)。

可以通过资源管理器的 `\\wsl.localhost\Ubuntu-22.04\home\<Ubuntu用户名>\` 将权重复制到 Linux 用户目录，也可以在 Ubuntu 中直接下载自己的私有 Hugging Face 仓库。访问令牌在本机登录时输入。

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Configure -Checkpoint /home/alice/checkpoints/steps_30000 -Gpu 0
```

支持直接指定 Windows 权重目录，例如 `-Checkpoint 'D:\H-Bot models\steps_30000'`，脚本自动转换为 WSL 路径。大文件加载建议使用 Linux 文件系统中的副本。

也可在首次 Setup 时同时提供 `-Checkpoint`。已配置后重复 Setup 默认复用配置。Configure 将按提供的参数保存整份配置，GPU 默认 0、正式网络默认 direct。配置保存在 Linux 安装目录的 `package/local.env`。

## 自检与正式启动

```powershell
# 查看 WSL、内存、磁盘和 GPU
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Doctor

# 检查权重文件、哈希与训练契约
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Check

# 实际加载权重，执行六任务预热及本地 TLS/WSS 多客户端自测
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 SelfTest

# 自测通过后申请正式链接，再启动服务
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Start
```

Start 创建独立 tmux 会话并进入该会话。看到 `MODEL_READY` 后，在隐藏输入提示处粘贴完整 WSS 链接。连接成功后观察 `WSS_NEGOTIATION_PASSED` 和推理日志。

再次执行 Start 会进入已有会话。按 `Ctrl+B`，松开后按 `D` 可以离开 tmux，让服务继续运行。Windows 需要保持开机、网络在线并关闭自动休眠。Windows 重启、WSL 关闭或网络断开会影响测评连接。

```powershell
# 重新进入终端
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Attach

# 查看进程是否退出和退出码
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Status

# 查看结束原因后，清除已经退出的会话，为下一次正式申请准备入口
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 ClearSession
```

退出的会话保留终端内容，ClearSession 仅清除已退出的会话。需要主动停止当前测评时，进入 Attach 后按 `Ctrl+C`。正式连接沿用一次连接策略，连接中断后需按赛事规则处理后续申请。日志保存在服务启动时打印的目录中。

## 网络与代理

默认正式服务使用 direct。若安装需要代理，可以在 Setup 时提供 WSL 可访问的 HTTP 代理地址。

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\official-pi05\windows.ps1 Setup -ProxyUrl http://172.30.96.1:7890
```

地址需要替换为实际值。该参数配置 Python 依赖和 tokenizer 下载代理，Ubuntu apt 使用其自身网络配置。已有安装需要更新代理时，用 Configure 并同时提供权重路径。如果正式服务也需要该代理，设置 `-Network inherit`。

WSL 默认 NAT 模式下，Windows 的 `127.0.0.1` 代理通常需要改用 WSL 所见的 Windows 主机 IP，同时让代理允许来自 WSL 的连接。可在 Ubuntu 中通过 `ip route show default` 查看网关。支持 mirrored networking 的系统可按 Microsoft 文档配置 localhost 互通。网络配置完成后重新运行 SelfTest，并确认实际比赛域名可访问。详见 [WSL 网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking)。

整个流程在执行脚本的 Windows 电脑及其 WSL 内安装。自检通过后再申请有次数限制的正式测评链接。

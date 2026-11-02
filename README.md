# H-Bot Competition Workspace

本仓库用于管理 H-Bot 机器人比赛的训练、评测、配置与成绩记录。服务器上的模型、数据集、检查点、视频和运行日志属于本地运行资产，不进入 Git。

## 当前最佳本地结果

| 模型 | 回合数 | 平均分 | 成功率 |
| --- | ---: | ---: | ---: |
| Full SFT 99999 steps | 300 | **0.5704** | **39.00%** |

六任务各 50 回合。详细分项与不同训练阶段对比见 [`results/2026-11-02-full-sft-99999.md`](results/2026-11-02-full-sft-99999.md)。

## 仓库内容

- `scripts/`：训练、服务、评测、汇总及其测试。
- `config/`：可复现配置、环境准备与校验脚本。
- `patches/`：对 OpenPI、RoboOrchardSim 上游仓库的本地改动。
- `results/`：经整理的关键成绩报告。
- `outputs/sim/`：结构化评测结果、配置快照与运行日志；MP4 和 MCAP 录制文件不进入 Git。
- `docs/`：上游版本、资产边界和协作约定。

服务器仍使用以下本地目录，但它们均被 `.gitignore` 排除：`datasets/`、`models/`、`checkpoints/`、`logs/`、缓存、虚拟环境和上游源码检出。`outputs/sim/` 仅跟踪非媒体结果；其中的 MP4、MCAP 始终排除。

## 常用入口

```bash
# 训练配置与安全说明
less scripts/train/README.txt

# 运行本地评测
bash scripts/run-local-eval.sh --help

# 校验与生成评测报告
python scripts/report-pi05-eval.py --help

# CPU 单元测试
python -m unittest discover -s scripts -p 'test*.py'
python -m unittest discover -s scripts/train/tests -p 'test*.py'
```

运行前请阅读 `config/README.md`。脚本默认面向服务器路径 `/study/wangmengfan/hbot`，不要在未检查 GPU、温度、端口和现有任务的情况下启动训练或评测。

## GitHub 使用方式

1. 在 GitHub 创建空仓库，不要自动生成 README 或许可证。
2. 在服务器中为本仓库添加远端：`git remote add origin <仓库地址>`。
3. 检查 `git status` 和 `git diff --cached` 后再提交、推送。

任何数据、模型、检查点或原始视频都不应使用普通 Git 推送。若比赛需要共享大文件，应单独使用对象存储或明确评估 Git LFS 配额。

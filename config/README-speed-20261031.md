# H-Bot 仿真速度试验（2026-10-31）

## 范围与结论

在现有 GPU 1 的 π₀.₅ 全量 SFT 模型服务之外，仅使用 GPU 0 做独立仿真测速。
没有停止模型服务，没有改变显卡功率、CPU governor、物理频率、相机频率、
模型采样步数、动作序列长度或正式任务的步数预算。

当前证据表明：主要瓶颈在仿真侧，不是模型请求。单纯改成 8 个 CPU 线程
没有可靠收益；同卡两个仿真进程的总吞吐有提升，但不能按进程数量推算翻倍。
本次小样本不足以承诺六任务 300 回合在 5 小时内结束。

运行中 GPU 0 在 84℃出现过驱动明确报告的 `SW Thermal Slowdown: Active`。
驱动报告目标温度为 83℃，硬件 Slowdown Temp 为 95℃，二者不是同一阈值。
按用户要求保持 350 W 功率上限。早期测试用过保守的 85℃连续触发保护；
最终脚本采用 90℃保护线，只停止自己启动的本轮评测，不停止模型和容器。

## 新增入口

服务器文件：

- `scripts/run-local-eval-speed.sh`：可选的双进程、线程数、CPU 亲和性、计时和温度保护入口。
- `config/hbot-local-pi05-parallel.yaml`：独立全量并行配置，仍为六任务各 50 回合。
- `config/hbot-perf-baseline.yaml`：一个叠方块回合，900 步上限，仅用于测速。
- `config/hbot-perf-pair.yaml`：两个叠方块回合，各 900 步上限，分成两个独立 Client。
- `robo_orchard_sim/robo_orchard_sim/pipeline/evaluator/performance.py`：默认关闭的分项计时器。

原 `scripts/run-local-eval.sh` 和 `config/hbot-local-pi05.yaml` 未被覆盖。
原启动方式依旧默认串行，不自动开启线程调整或性能分析。

脚本位置的共同前缀为 `/study/wangmengfan/hbot/`。

## 如何使用

确保模型服务已经在 GPU 1、端口 8766 启动；本脚本不启动或重启模型。
先做两个回合的小测：

```bash
HBOT_SIM_CPU_AFFINITY=0-23,48-71 \
bash /study/wangmengfan/hbot/scripts/run-local-eval-speed.sh \
  hbot-perf-pair.yaml 2 8 0 speed_pair
```

五个参数依次为：配置文件名、每卡进程数、每进程 CPU 线程数、每任务录制数量、日志标签。
CPU 亲和性列表来自本机 `nvidia-smi topo -m` 的 GPU 0 拓扑，不应照搬到其他服务器。
它只约束新仿真进程，不改变现有模型的 CPU 亲和性。

确认资源与稳定性后，可手动启动全量并行评测：

```bash
HBOT_SIM_CPU_AFFINITY=0-23,48-71 \
bash /study/wangmengfan/hbot/scripts/run-local-eval-speed.sh \
  hbot-local-pi05-parallel.yaml 2 8 2 pi05_parallel50
```

这条命令没有在本次优化中自动执行。`2` 个进程不是每任务翻倍到 100 回合。
正式配置仍只录制每任务前两个回合，不是每个分片各录制两个。
脚本会拒绝与 GPU 0 已有计算进程争用资源，并检测显存余量。

长任务的划分：

| 任务 | 回合划分 | 最大动作步数/回合 |
|---|---|---:|
| pick_color | 50 | 200 |
| pick_spatial | 50 | 200 |
| place_a2b | 25 + 25 | 300 |
| stack_cubes | 13 + 13 + 12 + 12 | 900 |
| open_drawer | 50 | 300 |
| joint_direction | 50 | 300 |

已通过程序校验：各任务主种子/场景内回合标识与原配置一致，没有遗漏或重复，
任务其他参数相同。不同 Client 的模型随机状态独立，分片会改变随机采样序列；
若存在无效场景重采样，回退种子分配也可能不同，因此不保证与旧串行轨迹逐步一致。
评估模型差异时应让各个模型使用同一份新配置，并通过足够回合统计比较。

## 结果位置与停止方式

脚本打印本轮运行名，例如 `local_speed_pair_日期_进程号`。

- 主日志：`logs/<运行名>.log`
- 各任务输出：`outputs/sim/<运行名>/`
- 分项性能：各任务 `shards/<分片名>/performance.json`
- 最终成绩：运行目录 `summary.json`，不要把 `performance.json` 当成绩。

保持终端运行；Ctrl+C 会清理本次 dispatcher 和子进程，保留模型、容器和产物。
也可在另一终端使用已有精确停止入口：

```bash
docker exec hbot-sim python3 \
  /home/users/wangmengfan/scripts/hbot-eval-control.py stop '<本轮运行名>'
```

不要把温度保护/人为中止的测速输出当成完整评测成绩，不保证自动断点续评。

## 测量口径

`compare-speed.py` 读取 `[hbot-perf]` 检查点，排除最初 100 个环境步，再计算
各工作进程的环境步吞吐之和。它适用于比较重叠运行期间的短测，不是全量评测
端到端加速比，也不包含所有启动、收尾、视频导出的成本。

计时器记录包含等待的 wall time，不主动同步 CUDA；嵌套计时有重叠，不能直接相加。
单进程与双进程的样本长度、轨迹和温度并不完全相同，结果仅作工程估计。
正式推理 30 秒约束还包括排队和网络，本地热启动延迟不能作为公网评测的保证。

## 修改与回退

既有代码中仅修改了：

1. `scripts/hbot-eval-control.py`：支持可选录制数量；0 时不尝试导出不存在的视频；默认仍为 2。
2. `robo_orchard_sim/examples/manipulation-app/scripts/eval_policy.py`：读取可选并发槽位并记录执行配置。
3. `robo_orchard_sim/robo_orchard_sim/pipeline/evaluator/distributed/scheduling.py`：可选每卡多个槽位；默认 1。
4. `robo_orchard_sim/robo_orchard_sim/pipeline/evaluator/evaluator.py`：可选 per-process 线程参数。
5. `robo_orchard_sim/robo_orchard_sim/pipeline/evaluator/execution.py`：可选性能计时上下文。

修改前的五份文件位于 `tmp/speed-20261031/original/`，未被后续部署覆盖。
若不需要加速入口，继续使用原 `run-local-eval.sh` 即可，不需要删除任何文件。
若要完全回退代码，应在没有评测运行时对照上述路径逐文件恢复备份；先确认没有后续其他修改。

所有性能输出保留用于复核，没有清理旧评测、旧 Docker 镜像或任何训练产物。

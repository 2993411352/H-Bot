# 维护与数据协议

## 固定环境

- 基础镜像：`horizonrobotics/robo_orchard_sim@sha256:69e64ff85395140eb1f5d058c906be6a0d664a1b9771b047db8641651e699710`。
- Ubuntu 22.04、Python 3.11、PyTorch 2.7.0 + CUDA 12.8、Isaac Sim 5.1、Isaac Lab 2.3.2、Warp 1.15.0。
- [RoboOrchardSim](https://github.com/HorizonRobotics/RoboOrchardSim)：`f2e041e5fabe354c1398ea97c934f603bb65d14d`。`patches/collector.patch` 保留现有采集器的六处修复，包括场景初始化拒绝、reset、诊断和当前完成状态记录。附上游 Apache 许可证。
- cuRobo：`2fbffc35225398cf9d5f382804faa9de2608753b`（0.7.6）。五个扩展按本机架构重编译。隔离源码调用 Warp 公开的 `wp.device_from_torch`。
- 额外依赖锁在 `converter-requirements.txt`。隔离 venv 共享基础镜像的 Torch/Isaac，以 `--no-deps` 安装指定组合；上游 schemas 声明的 protobuf 版本范围与基础镜像不一致，保留实际可解码的镜像版本。

系统安装脚本依据 [Docker Ubuntu 文档](https://docs.docker.com/engine/install/ubuntu/)和 [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。[Isaac Sim 5.1 要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)列出 Linux 测试驱动 580.65.06；安装脚本不改驱动。

## GPU 模式

RTX A5000 使用 `sm_86`，RTX 5070 Ti 使用 `sm_120`。16 GB 卡采用 `compact-ik`：将 upstream 固定的 1280 个 IK 候选槽改为 128 槽，并分批遍历输入，避免原方法容量不足时直接截断候选。图像分辨率、物理频率、碰撞判定、IK 精度和种子数量保持原配置。批次边界仍可能改变数值和随机数结果，记录在 `plan.json`。

来源采集器已在 RTX 3090 运行。本移植包的 A5000/5070 Ti 完整 GPU 运行、低显存峰值尚未实机测量；CPU 检查不能证明 RTX 渲染或物理成功。不强制少量试采，按用户要求直接提供正式采集入口。OOM 或进程错误会保留输出并停止，不自动重启、不修改 GPU 功率。

## 数据质量

这四类数据是专家规划器产生的成功示范，不是滑落重抓或完整 DAgger。

采集检查：

- 原始判定器成功，且 `EpisodeSummary.error` 为空；进程退出 0 不单独作为成功证明。
- 额外执行 30 个控制步，后 15 步当前完成条件持续成立。
- 三相机、状态、关节位置动作同频、数量相等、时间偏差小于 2 ms。
- 三路 1280×720 RGB、30 Hz；拒绝过多黑帧，输出连续 15 步有效窗口掩码，不删除/拼接中间帧。
- 目标、容器、方向与语言角色一致；目标物、干扰物均在 seen 资产范围。
- 检查最终抓持、放置或四层堆叠几何，以及稳定性。

任务控制步上限依次为 200、200、300、900；物理频率 600 Hz、控制/录制 30 Hz。记录还可能包含初始化及稳定确认帧，因此 MCAP 总帧数可超过控制循环上限。不同长度的 episode 可训练固定 15 步动作窗口，不应为了等长裁掉关键阶段。

转换使用专门的 Panda Droid 映射，状态/动作均为 8 维，前 7 维关节弧度、第 8 维夹爪绝对量。逐条比较 MCAP 和 Parquet 的状态/动作、时间与三相机 MP4 像素，随后通过 LeRobot 0.3.3 reader 抽查动作窗口。原始末尾完整窗口由数值对齐检查，避免 reader 的浮点末端时间被误判 padding。

格式通过后仍标 `training_ready=false`。训练时同时使用跨机器 accepted 清单、人工复核结果与 `training_anchor_masks.json`，不修改原始数据或标签。

## 多人去重

覆盖采样版本为 `coverage_v5`。当前 seen 索引有 362 个实例，专家抓取候选取 rigid + `has_interactive_poses` + `is_graspable` 的交集；空间参照物可用 rigid + `has_interactive_poses`，不要求被抓起。颜色干扰物必须同类别且与目标颜色集合不相交。配置用目标 UUID 和干扰物 `filter.only_in` 精确指定，保留 seen 约束并在 MCAP 元数据中复核。

覆盖统计与去重分别维护。`coverage.json` 来自原始训练 source_records 与已通过的合成清单；未知字段保留未知。`coverage_status.json` 使用实际 `init_position` 和角色绑定计数，不用请求配置冒充实际覆盖，也不将失败尝试计为成功覆盖。失败次数只参与候选排序。目标的摆放、朝向经过物理稳定可能偏离提议区间，因此记录实际区间。

选择过程计入正在执行的实例预留，worker ID 用于打散并列候选。不同机器的新增数据并不实时共享；仍使用独立种子、共享历史指纹、最后跨机器去重。不能仅凭 worker ID 或物品组合不同宣称轨迹绝不重复。

每人每任务 400 条，默认每任务最多尝试 2000 次。worker ID 全局唯一，1～1999；新运行不得复用历史 ID。种子为 `1000000000 + worker_id*1000000 + task_index*250000 + attempt_index`，与已有评测范围分离。

共享索引来自协调者的原始训练数据与已有合成数据。其覆盖范围由输入索引决定，快照导出不能补上源索引缺失的记录。原始数据若缺少场景元数据，可做轨迹去重，但不能据此声明初始场景近重复检查也完整。

去重标准包括：去连续重复动作后的整条指令序列、15 步运动状态/动作窗口高度重合、相同资产及角色下初始位置差 ≤1 mm 且角度差 ≤1°。同类别或同句指令本身不判重复。

每台机器与共享索引及自身新增记录比较；离线同时采集的新数据必须在集中汇总时再比较。汇总拒收重复项，保留源数据。最终合格总量以汇总与审核后为准。合并时不能把待合并 worker 自身提前写入共享 reference。

## 文件与恢复

种子在进程启动前写盘；运行锁防止重复提交，同一主机使用 GPU 锁。干净暂停可以原参数续采。代码、资产索引、共享指纹库、参数改变则拒绝续采。异常状态由协调者检查后处理，不自动回收种子或覆盖目录。

资产只读挂载，仅索引复制到每个运行的私有目录供 upstream 文件锁使用，禁止自动重建索引。下载命令会固定并记录资产 commit；资源检查只验证必要文件，不等同于所有下载文件的完整性验证。

新增参与者发生 CUDA/Vulkan 故障时，收集显卡型号、驱动、`build.json`、`status.json` 和候选 `stdout.log`。不通过修改成功标准或剔除困难类别来凑配额。

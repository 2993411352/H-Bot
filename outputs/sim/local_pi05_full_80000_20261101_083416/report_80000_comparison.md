# π₀.₅ Full SFT 80k 并行评测对比

完整性：六任务各 50 回合，300 回合逐条复核通过。

本地结果，非平台正式成绩。

## 配置与对照边界

- 权重：`/study/wangmengfan/hbot/checkpoints/pi05_hbot_full/80000`。
- GPU 1：单模型服务，三相机，15 步动作，保持原模型采样设置。
- GPU 0：两个独立仿真进程，每进程 8 线程，CPU 亲和性 0-23,48-71。
- 六任务各 50 回合；叠方块分 4 片，物体放置分 2 片；每任务只录制前 2 回合。
- 主种子与任务预算不变。并行改变 Client 会话/RNG 序列，场景重采样也可能不同；与旧串行结果不是严格逐轨迹对照。
- 未修改显卡功率、物理频率、相机频率或最大步数。

## 总体与分项结果

每项：得分 / 成功率。

| 模型 | 平均 | Place_A2B | Pick_Color | Pick_Spatial | Open_Drawer | Stack_Cubes | Joint_Direction |
|---|---|---|---|---|---|---|---|
| Full SFT 80k (parallel) | 0.5153 / 32.33% | 0.327 / 16.00% | 0.530 / 34.00% | 0.535 / 34.00% | 0.644 / 44.00% | 0.300 / 14.00% | 0.756 / 52.00% |
| Full SFT 60k | 0.5265 / 37.00% | 0.370 / 26.00% | 0.565 / 44.00% | 0.595 / 40.00% | 0.656 / 46.00% | 0.205 / 10.00% | 0.768 / 56.00% |
| Full SFT 30k | 0.4775 / 28.67% | 0.280 / 16.00% | 0.560 / 40.00% | 0.525 / 34.00% | 0.584 / 32.00% | 0.200 / 6.00% | 0.716 / 44.00% |
| LoRA 60k | 0.4124 / 25.00% | 0.277 / 22.00% | 0.560 / 40.00% | 0.395 / 22.00% | 0.368 / 6.00% | 0.195 / 10.00% | 0.680 / 50.00% |
| LoRA 40k | 0.4124 / 23.33% | 0.247 / 14.00% | 0.595 / 44.00% | 0.385 / 20.00% | 0.364 / 4.00% | 0.200 / 8.00% | 0.684 / 50.00% |

仿真调度用时：10.40 小时（不含后续视频导出）。

## 80k 阶段指标

- place_a2b：8/50；reach_pick=62.0%；contact_pick=44.0%；grasp_pick=28.0%；lift_pick=24.0%；reach_place=22.0%；place_within_xy=16.0%。
- pick_color：17/50；reach_pick=76.0%；contact_pick=62.0%；grasp_pick=40.0%；lift_pick=34.0%。
- pick_spatial：17/50；reach_pick=88.0%；contact_pick=54.0%；grasp_pick=38.0%；lift_pick=34.0%。
- open_drawer：22/50；reach=100.0%；contact=64.0%；moved=64.0%；target_joint_state=46.0%；settled_at_target=44.0%。
- stack_cubes：7/50；first_pair_stacked=66.0%；first_three_stacked=26.0%；full_order_stacked=14.0%；settled_stack=14.0%。
- joint_direction：26/50；reach=98.0%；contact=68.0%；moved=88.0%；target_joint_state=52.0%；settled_at_target=52.0%。

## 产物

`run_status.json` 区分仿真完整性与视频导出状态；`resource_samples.jsonl` 记录资源。
`comparison_all_models.json` 保存可复核数值；`config_snapshot/` 保存配置；各任务 `shards/` 保存逐回合结果。

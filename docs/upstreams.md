# Upstream dependencies

本项目不复制三个完整上游仓库。服务器保留本地检出，根仓库只记录来源、固定版本和 H-Bot 改动补丁。

| 目录 | 来源 | 分支 | 本地状态 |
| --- | --- | --- | --- |
| `openpi/` | `https://github.com/Physical-Intelligence/openpi.git` | `main` | `215abfb217dbac7d5f1273282331b9b1866c0479`；有 H-Bot JAX 缓存路径补丁 |
| `robo_orchard_lab/` | `https://github.com/HorizonRobotics/RoboOrchardLab.git` | `master` | `48ac43987df9ac1ba40fef72dd7a2b4679c592b4`；工作区干净 |
| `robo_orchard_sim/` | `https://github.com/HorizonRobotics/RoboOrchardSim.git` | `master` | `f2e041e5fabe354c1398ea97c934f603bb65d14d`；有多 worker 与可选性能分析补丁 |

重建时先检出上游仓库的比赛固定版本，再从仓库根目录应用：

```bash
git -C openpi apply ../patches/openpi-hbot.patch
git -C robo_orchard_sim apply ../patches/robo-orchard-sim-hbot.patch
```

提交前应更新此表中的精确 commit，并运行 `git apply --check` 验证补丁。不要直接把嵌套 `.git` 目录加入根仓库。

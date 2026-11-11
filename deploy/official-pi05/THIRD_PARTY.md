# 来源与改动

- OpenPI：<https://github.com/Physical-Intelligence/openpi>，固定 commit `215abfb217dbac7d5f1273282331b9b1866c0479`。安装时获取上游源码及其 `uv.lock`，Apache-2.0；上游还含 Gemma 使用条款。这里不重新分发模型权重。
- `vendor/robo_orchard_server` 来源于 RoboOrchardSim 的服务组件及本项目已经使用的 Pi05 扩展，基线 commit `f2e041e5fabe354c1398ea97c934f603bb65d14d`：<https://github.com/HorizonRobotics/RoboOrchardSim>。保留源码版权头及 [Apache-2.0 许可证](vendor/LICENSE-RoboOrchard)。
- 相对于原上游服务的 H-Bot 改动包括：Pi05 三相机/PandaDroid 适配、共享权重但独立的每 Client 推理 RNG 和动作缓存、reset/close 生命周期处理、正式反向 WSS 多会话入口。此 vendor 快照是实际使用版本，不等同于直接安装最新版上游。
- `official_server.py`、`smoke_wss.py`、配置与转换来自本项目原 B16 · 100k 正式入口。本次迁移增加路径/权重/GPU 配置、后端选择、独立安装、合成协议样本选项与 URL 异常脱敏；固定动作语义和会话测试保持不变。本地 TLS 测试端关闭单次连接不需要的会话票据，正式客户端 TLS 设置不变。
- 原始协议文件由 `vendor_manifest.json` 校验；迁移后推理文件由 `inference_manifest.json` 校验。更改这些文件应重新验证并有意更新 manifest，不能删掉校验来绕过错误。
- 权重、归一化文件和数据集的使用/共享仍应遵循各自来源条款；本部署包不提供这些文件的再分发授权。

## LoongForge 后端

- LoongForge 源码基线 commit `863019416f141dbc4b801a919e332f953e8edc7f`，来自本轮实际训练环境中的 `loongforge.embodied.model.pi05` 最小源码。保留 [原许可证](loongforge_backend/vendor/LICENSE.LoongForge)。
- LeRobot 源码基线 commit `00b662de02734a6972ec674b8792696ecd1cb28e`，仅包含该模型使用的 `pi_gemma.py` 与导入辅助代码，保留 [原许可证](loongforge_backend/vendor/LICENSE.LeRobot)。
- `loongforge_backend/hbot_inference.py` 和 `model_runtime.py` 复用已有本地仿真推理适配。预处理、严格权重加载和 RNG 行为未改动。本次增加统一正式 WSS 入口及后端环境选择。
- `fixtures/` 只有参数头、数据契约、测试统计量和冻结训练变换，没有模型参数、训练视频或可执行训练任务。真正推理始终读取所选 checkpoint 的统计量并验证哈希；不使用 fixture 统计量替代模型统计量。
- Tokenizer 不随包复制，setup 从公开原始地址下载后校验 SHA256。该 tokenizer 与两个后端原工作环境使用的文件一致。

# deploy-xllm-npu 参考说明

本文件补充 `SKILL.md` 中未展开的细节，包括脚本参数说明、多机配置思路等。

## 1. `scripts/run_xllm_npu.sh` 参数说明

脚本中关键变量：

- `MODEL_PATH`：模型权重所在目录，例如 `/data/models/Qwen3-8B`
- `MASTER_NODE_ADDR`：主节点地址，格式为 `ip:port`，单机一般为 `127.0.0.1:9748`
- `START_PORT`：服务起始端口，多个节点时会在此基础上递增
- `START_DEVICE`：起始逻辑设备号，例如从 0 开始
- `NNODES`：节点数（当前脚本中等价于「进程数/卡数」）
- `XLLM_BIN`：xLLM 二进制路径，例如：
  - 自编译：`/path/to/xllm/build/xllm/core/server/xllm`
  - release 镜像：`/usr/local/bin/xllm`

常见修改场景：

- 单机多卡：增加 `NNODES`，并调整 `ASCEND_RT_VISIBLE_DEVICES`
- 切换模型：只需要改 `MODEL_PATH`
- 更换端口：改 `MASTER_NODE_ADDR` 中的端口和 `START_PORT`

## 2. 单机多卡与多机场景提示（简要）

- **单机多卡**：
  - `NNODES` 设为卡数
  - `ASCEND_RT_VISIBLE_DEVICES` 中包含所有要用的逻辑设备号
  - `MASTER_NODE_ADDR` 保持为本机 IP
- **多机**（仅作提示，细节以官方文档为准）：
  - 各节点共享相同的 `MASTER_NODE_ADDR`
  - 每个节点的 `node_rank` 不同（当前脚本中用 `i` 控制）
  - 确保网络连通性以及 HCCL 配置正确

如需更复杂的部署方式（例如多机多卡、在线/离线服务、PD 拆分等），建议进一步参考：

- `docs/zh/getting_started/online_service.md`
- `docs/zh/getting_started/offline_service.md`
- `docs/zh/getting_started/multi_machine.md`


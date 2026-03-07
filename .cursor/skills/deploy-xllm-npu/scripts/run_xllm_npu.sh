#!/bin/bash
set -e

rm -rf core.*

# Ascend 环境变量，根据实际环境调整路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 可见设备设置：单机单卡默认 0，多卡时按需修改
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

# ======= 需按环境修改的关键变量 =======
MODEL_PATH="/path/to/model/Qwen3-8B"       # 模型路径
MASTER_NODE_ADDR="127.0.0.1:9748"          # Master 节点地址（需全局一致）
START_PORT=18000                           # 服务起始端口
START_DEVICE=0                             # 起始逻辑设备号
LOG_DIR="log"                              # 日志目录
NNODES=1                                   # 节点数（当前脚本启动的进程数）
XLLM_BIN="/path/to/xllm"                   # xLLM 可执行文件路径
# =======================================

mkdir -p "$LOG_DIR"

for (( i=0; i<NNODES; i++ )); do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"

  "$XLLM_BIN" \
    --model "$MODEL_PATH" \
    --devices="npu:$DEVICE" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --max_memory_utilization=0.86 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_shm=true \
    --node_rank="$i" > "$LOG_FILE" 2>&1 &
done

echo "xLLM NPU 服务已启动，日志见目录：$LOG_DIR"


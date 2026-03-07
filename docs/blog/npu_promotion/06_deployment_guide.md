# 从零开始：Qwen3在NPU上的部署指南

> 本文提供Qwen3模型在NPU上的完整部署教程，从环境准备到性能调优，一步到位。

## 一、准备工作

### 1.1 硬件要求

**最低配置**

| 组件 | 要求 |
|------|------|
| NPU型号 | A2 或 A3 |
| NPU数量 | 1张（Qwen3-8B及以下） |
| 内存 | 64GB+ |
| 存储 | 100GB+ SSD |
| CPU | 8核+ |

**推荐配置**

| 模型 | NPU配置 | 内存 | 存储 |
|------|---------|------|------|
| Qwen3-0.6B | 1× A2 | 32GB | 50GB |
| Qwen3-1.7B | 1× A2 | 32GB | 50GB |
| Qwen3-8B | 1× A2/A3 | 64GB | 100GB |
| Qwen3-14B | 2× A2/A3 | 64GB | 150GB |
| Qwen3-32B | 4× A2/A3 | 128GB | 200GB |
| Qwen3-72B | 8× A2/A3 | 256GB | 400GB |

### 1.2 软件环境

**驱动要求**

```bash
# 检查NPU驱动版本
npu-smi info

# 要求输出类似：
# Version: 25.2.0 或更高
```

**Docker环境**

```bash
# 拉取xLLM开发镜像
# A2 x86架构
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86

# A2 ARM架构
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-arm

# A3 ARM架构
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hc-rc2-arm
```

**启动容器**

```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-npu \
--privileged \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v $HOME:$HOME \
-w $HOME \
quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86 \
/bin/bash
```

## 二、模型准备

### 2.1 模型下载

**从HuggingFace下载**

```bash
# 安装huggingface-cli
pip install huggingface_hub

# 下载Qwen3-8B
huggingface-cli download \
    Qwen/Qwen3-8B \
    --local-dir /path/to/models/Qwen3-8B \
    --local-dir-use-symlinks False
```

**从ModelScope下载**

```bash
# 安装modelscope
pip install modelscope

# 下载Qwen3-8B
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='/path/to/models')
```

### 2.2 模型格式检查

xLLM支持safetensors格式的模型权重：

```bash
# 检查模型文件
ls /path/to/models/Qwen3-8B/

# 应包含以下文件：
# config.json
# model.safetensors (或 model-00001-of-000xx.safetensors)
# tokenizer.json
# tokenizer_config.json
# special_tokens_map.json
```

**格式转换（如需要）**

```python
# 将PyTorch格式转换为safetensors
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

model = AutoModelForCausalLM.from_pretrained("/path/to/model")
state_dict = model.state_dict()
save_file(state_dict, "/path/to/model/model.safetensors")
```

### 2.3 模型配置检查

检查`config.json`中的关键配置：

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 22016,
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000
}
```

## 三、部署实战

### 3.1 单卡部署

**基础启动脚本**

```bash
#!/bin/bash
# start_qwen3_8b.sh

set -e

# 环境变量设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

# 模型和服务配置
MODEL_PATH="/path/to/models/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
PORT=18000

# 启动xLLM服务
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0" \
  --port $PORT \
  --master_node_addr=$MASTER_NODE_ADDR \
  --nnodes=1 \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend="hccl" \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_graph=true
```

**验证服务**

```python
# test_service.py
import openai

client = openai.OpenAI(
    base_url="http://localhost:18000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    max_tokens=256
)

print(response.choices[0].message.content)
```

### 3.2 多卡部署

**张量并行启动脚本**

```bash
#!/bin/bash
# start_qwen3_32b_tp4.sh

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_IF_BASE_PORT=43432

MODEL_PATH="/path/to/models/Qwen3-32B"
MASTER_NODE_ADDR="127.0.0.1:9748"
PORT=18000
NNODES=4  # 4张卡

# 启动多卡服务
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0,npu:1,npu:2,npu:3" \
  --port $PORT \
  --master_node_addr=$MASTER_NODE_ADDR \
  --nnodes=$NNODES \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend="hccl" \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_graph=true
```

**流水线并行启动脚本**

```bash
#!/bin/bash
# start_qwen3_72b_pp2_tp4.sh

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_BASE_PORT=43432

MODEL_PATH="/path/to/models/Qwen3-72B"
PORT=18000

# 8卡部署：2个流水线阶段，每阶段4卡张量并行
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7" \
  --port $PORT \
  --pipeline_parallel_size=2 \
  --tensor_parallel_size=4 \
  --max_memory_utilization=0.86 \
  --enable_chunked_prefill=true
```

### 3.3 关键参数解析

**显存管理参数**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `max_memory_utilization` | 显存利用率上限 | 0.85-0.90 |
| `block_size` | KV Cache块大小 | 128 |
| `gpu_memory_utilization` | 同上（旧参数名） | 0.85-0.90 |

**性能优化参数**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `enable_graph` | 启用ACLGraph | true |
| `enable_chunked_prefill` | 启用分块Prefill | true |
| `enable_schedule_overlap` | 启用调度重叠 | true |
| `enable_prefix_cache` | 启用前缀缓存 | 按需 |

**并行配置参数**

| 参数 | 说明 | 使用场景 |
|------|------|----------|
| `tensor_parallel_size` | 张量并行度 | 单机多卡 |
| `pipeline_parallel_size` | 流水线并行度 | 大模型多机 |
| `nnodes` | 节点数（卡数） | 所有场景 |

**服务配置参数**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `port` | 服务端口 | 18000 |
| `master_node_addr` | Master节点地址 | 必填 |
| `max_num_batched_tokens` | 最大批处理token数 | 8192 |
| `max_num_seqs` | 最大并发序列数 | 256 |

## 四、性能调优

### 4.1 吞吐优化

**增大批处理能力**

```bash
# 增大max_num_batched_tokens
--max_num_batched_tokens=16384

# 增大max_num_seqs
--max_num_seqs=512

# 注意：需要足够的显存支持
```

**开启所有优化**

```bash
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0" \
  --enable_graph=true \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --max_num_batched_tokens=16384
```

**调整调度策略**

```bash
# Prefill优先（降低TTFT）
--prefill_priority=true

# Decode优先（降低TPOT）
--decode_priority=true

# 默认：平衡模式
```

### 4.2 延迟优化

**减小batch size**

```bash
# 限制最大batch size
--max_num_seqs=32

# 减小批处理token数
--max_num_batched_tokens=4096
```

**开启前缀缓存**

```bash
# 对于有重复前缀的请求
--enable_prefix_cache=true

# 配合prefix缓存感知调度
--prefix_cache_aware_schedule=true
```

**优化Attention**

```bash
# 使用FlashAttention
--use_flash_attention=true  # 默认开启

# 调整attention实现
--attention_backend=flash  # 或 paged
```

### 4.3 显存优化

**降低显存占用**

```bash
# 降低显存利用率上限
--max_memory_utilization=0.80

# 减小block size
--block_size=64

# 限制KV Cache
--max_model_len=4096  # 限制最大序列长度
```

**启用KV Cache卸载**

```bash
# 启用CPU卸载
--enable_cpu_offload=true

# 启用磁盘卸载（需要配置Mooncake）
--enable_disk_offload=true
```

### 4.4 不同场景配置模板

**场景一：高吞吐离线处理**

```bash
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0,npu:1,npu:2,npu:3" \
  --max_memory_utilization=0.90 \
  --max_num_batched_tokens=16384 \
  --max_num_seqs=512 \
  --enable_graph=true \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true
```

**场景二：低延迟在线服务**

```bash
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0" \
  --max_memory_utilization=0.85 \
  --max_num_batched_tokens=4096 \
  --max_num_seqs=64 \
  --enable_graph=true \
  --enable_prefix_cache=true \
  --prefill_priority=true
```

**场景三：长文本处理**

```bash
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0" \
  --max_model_len=32768 \
  --block_size=128 \
  --max_memory_utilization=0.88 \
  --enable_chunked_prefill=true
```

**场景四：多租户隔离**

```bash
/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0,npu:1" \
  --enable_disagg_pd=true \
  --prefill_instances=1 \
  --decode_instances=1
```

## 五、监控与运维

### 5.1 性能监控

**Prometheus指标**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'xllm'
    static_configs:
      - targets: ['localhost:18000']
```

**关键指标**

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| `xllm_request_latency_seconds` | 请求延迟 | P99 > 5s |
| `xllm_tokens_per_second` | 吞吐量 | < 预期值50% |
| `xllm_memory_used_bytes` | 显存使用 | > 95% |
| `xllm_num_running_requests` | 运行请求数 | > max_num_seqs 80% |

**Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "xLLM Monitoring",
    "panels": [
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(xllm_request_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Throughput",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(xllm_tokens_total[1m])"
          }
        ]
      }
    ]
  }
}
```

### 5.2 日志分析

**日志级别配置**

xLLM使用Google Logging Library (glog)作为日志库，支持以下环境变量：

```bash
# 设置详细日志级别（控制VLOG输出）
export GLOG_v=2  # 输出 VLOG(n) 中 n <= 2 的日志

# 设置最低日志级别
export GLOG_minloglevel=0  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# 日志输出位置
export GLOG_log_dir=/path/to/logs

# 同时输出到stderr
export GLOG_alsologtostderr=1
```

**glog日志级别说明**：

| 级别 | 宏 | 说明 |
|------|-----|------|
| INFO | `LOG(INFO)` | 信息级别 |
| WARNING | `LOG(WARNING)` | 警告级别 |
| ERROR | `LOG(ERROR)` | 错误级别 |
| FATAL | `LOG(FATAL)` | 致命错误（会终止程序） |
| VLOG(n) | `VLOG(n)` | 详细日志，n <= GLOG_v 时输出 |

**关键日志**

```bash
# 查看启动日志
grep "xLLM" /path/to/logs/xllm.log | head -50

# 查看错误日志
grep "ERROR" /path/to/logs/xllm.log

# 查看性能日志
grep "latency\|throughput" /path/to/logs/xllm.log
```

### 5.3 故障排查

**常见问题FAQ**

**问题1：OOM错误**

```
错误信息：OutOfMemoryError: NPU out of memory

解决方案：
1. 降低max_memory_utilization
2. 减小max_num_batched_tokens
3. 减小max_model_len
4. 检查是否有内存泄漏
```

**问题2：服务启动失败**

```
错误信息：Failed to initialize NPU device

排查步骤：
1. 检查NPU驱动：npu-smi info
2. 检查设备权限：ls -la /dev/davinci*
3. 检查Docker配置：--device参数
4. 检查环境变量：ASCEND_RT_VISIBLE_DEVICES
```

**问题3：性能不达预期**

```
排查步骤：
1. 确认优化开关已开启：enable_graph, enable_chunked_prefill
2. 检查batch size是否足够大
3. 分析性能瓶颈：使用npu_timeline.py
4. 对比基准测试数据
```

**问题4：请求超时**

```
排查步骤：
1. 检查服务负载：num_running_requests
2. 检查队列积压：num_pending_requests
3. 调整超时设置
4. 考虑扩容
```

**调试技巧**

```bash
# 启用详细日志（glog的VLOG级别）
export GLOG_v=2  # 输出 VLOG(2) 及以下的详细日志

# 启用内置性能分析
--enable_profile_step_time=true
--enable_profile_token_budget=true
--enable_profile_kv_blocks=true

# 性能分析数据会输出到以下文件：
# - profile_prefill_step_time_xxx.txt
# - profile_decode_step_time_xxx.txt
```

**NPU级别性能分析（MSPTI）**

如需更详细的NPU kernel级别分析，可使用MSPTI工具。MSPTI (NPU Performance Tools Interface) 类似NVIDIA的NVTX，用于采集NPU底层的性能数据。

```bash
# 1. 编译时启用MSPTI（默认关闭）
# USE_MSPTI选项定义在项目根目录 CMakeLists.txt
cmake -DUSE_MSPTI=ON ..
make -j$(nproc)

# 2. 运行程序，MSPTI数据输出到日志
./xllm --model /path/to/model --devices="npu:0" 2>&1 | tee mspti.log

# 3. 转换为Chrome Trace格式
python tools/npu_timeline.py -i mspti.log -o timeline.json

# 4. 在Chrome浏览器中可视化
# 打开 chrome://tracing，加载 timeline.json
```

**MSPTI相关文件**：

| 文件 | 作用 |
|------|------|
| `CMakeLists.txt` | 定义 `USE_MSPTI` 编译选项（默认OFF） |
| `xllm/core/common/mspti_helper.cpp` | MSPTI实现代码 |
| `xllm/core/common/mspti_helper.h` | MSPTI头文件 |
| `tools/npu_timeline.py` | 日志转Chrome Trace格式工具 |

## 六、生产部署建议

### 6.1 高可用配置

**多实例部署**

```yaml
# docker-compose.yml
version: '3'
services:
  xllm-1:
    image: quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
    command: /path/to/xllm --model /models/Qwen3-8B --port 18001
    ports:
      - "18001:18001"
    devices:
      - /dev/davinci0
      
  xllm-2:
    image: quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
    command: /path/to/xllm --model /models/Qwen3-8B --port 18002
    ports:
      - "18002:18002"
    devices:
      - /dev/davinci1
```

**负载均衡**

```nginx
# nginx.conf
upstream xllm_backend {
    least_conn;
    server 127.0.0.1:18001;
    server 127.0.0.1:18002;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://xllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 6.2 容量规划

**计算公式**

```
所需卡数 = 日均请求量 × 平均tokens/请求 / (单卡吞吐 × 目标利用率 × 86400)

示例：
- 日均请求量：1000万
- 平均tokens/请求：500
- 单卡吞吐：1000 tokens/s
- 目标利用率：70%

所需卡数 = 10000000 × 500 / (1000 × 0.7 × 86400) ≈ 82张

建议：预留20%冗余，实际部署100张
```

### 6.3 安全配置

**API Key认证**

```bash
# 设置API Key
--api_key=your_secret_key

# 客户端调用
curl -H "Authorization: Bearer your_secret_key" http://localhost:18000/v1/chat/completions
```

**网络隔离**

```bash
# 限制监听地址
--host=127.0.0.1  # 仅本地访问

# 或通过防火墙
iptables -A INPUT -p tcp --dport 18000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 18000 -j DROP
```

## 七、小结

本文提供了Qwen3模型在NPU上的完整部署指南：

- **环境准备**：硬件要求、软件配置、Docker环境
- **模型准备**：下载、格式检查、配置验证
- **部署实战**：单卡、多卡、关键参数
- **性能调优**：吞吐、延迟、显存优化
- **运维监控**：监控指标、日志分析、故障排查
- **生产部署**：高可用、容量规划、安全配置

**下一篇文章预告**：《DeepSeek-V3在NPU上的高效推理实践》，将深入介绍MoE模型的优化技巧。

---

> **参考资料**：
> - [xLLM快速开始](https://xllm.readthedocs.io/zh-cn/latest/getting_started/quick_start.html)
> - [xLLM启动参数](https://xllm.readthedocs.io/zh-cn/latest/cli_reference.html)
> - [Qwen3模型文档](https://huggingface.co/Qwen)

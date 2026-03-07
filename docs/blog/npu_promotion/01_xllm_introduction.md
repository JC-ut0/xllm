# xLLM：让国产芯片跑出GPU的性能

> 本文是xLLM + NPU技术博客系列的开篇，介绍xLLM框架的核心定位、技术特性和快速上手指南。

## 一、背景与挑战

### 1.1 大模型推理的算力困境

随着大语言模型（LLM）的快速发展，越来越多的企业开始将大模型应用于生产环境。然而，大模型推理面临着严峻的算力挑战：

**GPU供应紧张，成本高昂**
- NVIDIA高端GPU一卡难求，价格持续上涨
- 企业采购周期长，影响业务迭代速度
- TCO（总拥有成本）居高不下

**国产芯片生态不完善**
- 缺乏针对国产芯片优化的推理框架
- 开发者学习成本高，迁移困难
- 性能优化经验匮乏

**推理框架对国产芯片支持不足**
- 主流框架（vLLM、TensorRT-LLM等）主要面向GPU
- 国产芯片适配工作量大
- 性能难以充分发挥

### 1.2 为什么选择NPU？

在国产AI芯片中，NPU（神经网络处理器）具有独特的优势：

**自主可控**
- 完全国产化设计，供应链安全有保障
- 符合信创要求，满足政策合规需求
- 技术自主迭代，不受外部限制

**成本优势明显**
- 硬件采购成本比同级别GPU低30%-50%
- 功耗更低，运营成本更优
- 总体拥有成本（TCO）优势显著

**算力持续提升**
- 新一代NPU算力已接近或达到国际先进水平
- 软件生态日趋完善
- 厂商支持力度大

### 1.3 xLLM的定位

**xLLM** 是京东开源的高效大模型推理框架，专为国产AI芯片优化，具有以下特点：

- **专为国产芯片优化**：深度适配NPU、MLU、ILU、MUSA等国产芯片
- **服务-引擎解耦架构**：灵活调度，高效扩展
- **企业级落地验证**：已在京东核心业务大规模部署

## 二、xLLM核心特性速览

### 2.1 全面的模型支持

xLLM支持主流大模型在国产芯片上的高效部署：

**LLM模型支持**

| 模型 | NPU | MLU | ILU |
|------|:---:|:---:|:---:|
| DeepSeek-V3/R1/V3.1 | ✅ | ✅ | ❌ |
| DeepSeek-V3.2 | ✅ | ✅ | ❌ |
| DeepSeek-R1-Distill-Qwen | ✅ | ❌ | ❌ |
| Qwen2/2.5/QwQ | ✅ | ✅ | ✅ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen3 MoE | ✅ | ✅ | ✅ |
| Kimi-k2 | ✅ | ❌ | ❌ |
| Llama2/3 | ✅ | ❌ | ✅ |
| GLM4.5/4.6/4.7/5 | ✅ | ❌ | ❌ |

**VLM模型支持**

| 模型 | NPU | MLU | ILU |
|------|:---:|:---:|:---:|
| MiniCPM-V | ✅ | ❌ | ❌ |
| MiMo-VL | ✅ | ❌ | ❌ |
| Qwen2.5-VL | ✅ | ✅ | ❌ |
| Qwen3-VL | ✅ | ✅ | ❌ |
| GLM-4.6V | ✅ | ❌ | ❌ |
| VLM-R1 | ✅ | ❌ | ❌ |

**其他模型**

| 类型 | 模型 | NPU支持 |
|------|------|:-------:|
| DiT | Flux | ✅ |
| Rerank | Qwen3-Reranker | ✅ |

### 2.2 关键技术亮点

**多层流水线执行编排**
- 框架调度层异步解耦调度，减少计算空泡
- 模型图层计算通信异步并行
- 算子内核层异构计算单元深度流水

**动态shape的图执行优化**
- 基于参数化与多图缓存方法的动态尺寸适配
- 受管控的显存池，保证地址安全可复用
- 集成适配性能关键的自定义算子（PageAttention、AllReduce）

**高效显存优化**
- 离散物理内存与连续虚拟内存的映射管理
- 按需分配内存空间，减少内存碎片
- 智能调度内存页，增加内存复用

**全局多级KV Cache管理**
- 多级缓存的KV智能卸载与预取
- 以KV Cache为中心的分布式存储架构
- 多节点间KV的智能传输路由

**算法优化**
- 投机推理优化，多核并行提升效率
- MoE专家的动态负载均衡

### 2.3 硬件支持矩阵

| 硬件类型 | 型号 | 驱动要求 |
|----------|------|----------|
| NPU | A2, A3 | HDK Driver 25.2.0+ |
| MLU | MLU590 | - |
| ILU | BI150 | - |
| MUSA | S5000 | - |

## 三、快速上手

### 3.1 环境准备

xLLM提供了开箱即用的Docker镜像，所有镜像存放在Quay.io：

**NPU开发镜像**

```bash
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

### 3.2 一键启动

以Qwen3-8B为例，启动xLLM服务：

```bash
#!/bin/bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
PORT=18000

/path/to/xllm \
  --model $MODEL_PATH \
  --devices="npu:0" \
  --port $PORT \
  --master_node_addr=$MASTER_NODE_ADDR \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend="hccl" \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_graph=true
```

### 3.3 性能初体验

启动服务后，可以通过OpenAI兼容API进行调用：

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:18000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[
        {"role": "user", "content": "你好，请介绍一下xLLM框架"}
    ],
    max_tokens=256
)

print(response.choices[0].message.content)
```

## 四、本文小结

本文作为xLLM + NPU技术博客系列的开篇，介绍了：

- **xLLM是什么**：专为国产芯片优化的高效大模型推理框架
- **能解决什么问题**：降低大模型推理成本，实现国产芯片高效部署
- **核心特性**：全面的模型支持、多层流水线优化、高效显存管理

**后续文章预告**：

1. 架构解析：服务-引擎解耦设计
2. NPU专属优化：ACLGraph深度解析
3. 显存管理：xTensor动态内存池
4. 性能基准：NPU vs GPU全面对比
5. 实战部署：Qwen3在NPU上的最佳实践
6. MoE优化：DeepSeek-V3高效推理
7. 生产落地：京东内部实践案例

## 五、参考资料

- **GitHub仓库**：https://github.com/jd-opensource/xllm
- **技术报告**：https://arxiv.org/abs/2510.14686
- **文档站点**：https://xllm.readthedocs.io/zh-cn/latest/
- **Docker镜像**：https://quay.io/repository/jd_xllm/xllm-ai

---

> **关于作者**：xLLM团队，京东零售技术团队，专注于大模型推理优化与国产芯片适配。欢迎在GitHub上Star、Fork和提Issue！

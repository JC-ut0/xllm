# xLLM架构揭秘：服务-引擎解耦的艺术

> 本文深入解析xLLM的服务-引擎解耦架构设计，揭示其高性能推理背后的技术原理。

## 一、传统推理框架的痛点

### 1.1 单体架构的局限

传统的大模型推理框架通常采用单体架构，将请求调度、模型计算、显存管理等功能耦合在一起。这种设计在早期简单场景下工作良好，但随着业务复杂度的提升，逐渐暴露出诸多问题：

**调度与计算耦合**
- 请求调度逻辑与模型计算逻辑交织
- 难以针对不同场景优化调度策略
- 代码维护成本高

**扩展性差**
- 难以支持多实例部署
- 无法灵活调整计算资源
- 水平扩展困难

**资源利用率低**
- Prefill和Decode阶段资源需求差异大
- 在线和离线请求相互干扰
- 无法充分利用硬件算力

### 1.2 典型问题场景

**场景一：在线/离线请求冲突**

```
在线请求：要求低延迟，batch size小
离线请求：追求高吞吐，batch size大

单体架构下，两者共享同一计算资源，难以兼顾
```

**场景二：Prefill/Decode资源竞争**

```
Prefill阶段：计算密集，需要大量算力
Decode阶段：访存密集，需要高带宽

同一模型实例中，两者资源需求差异大，难以优化
```

**场景三：多租户隔离困难**

```
租户A：高优先级，要求SLA保障
租户B：低优先级，可接受延迟

单体架构难以实现精细化的资源隔离和调度
```

## 二、xLLM架构设计

### 2.1 整体架构图

xLLM采用服务-引擎解耦的分层架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                         Service Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ API Gateway │  │  Scheduler  │  │  Load Balancer          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Rate Limiter│  │  Router     │  │  Metrics & Monitoring   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ SHM / RDMA
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Engine Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  P Instance │  │  D Instance │  │  KV Cache Manager       │  │
│  │  (Prefill)  │  │  (Decode)   │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Memory Pool │  │  Operators  │  │  Communication          │  │
│  │  (xTensor)  │  │  (NPU/GPU)  │  │  (HCCL/NCCL)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Hardware Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    NPU      │  │    MLU      │  │    GPU / ILU / MUSA     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Service层职责

**请求调度与路由**
- 接收来自客户端的推理请求
- 根据请求特征进行智能路由
- 支持多种API协议（OpenAI、Anthropic等）

**弹性资源管理**
- 动态管理计算实例
- 支持在线扩缩容
- 实现资源隔离

**多协议支持**
- OpenAI兼容API
- Anthropic协议
- 自定义协议扩展

### 2.3 Engine层职责

**模型计算执行**
- 高效的模型前向计算
- 支持多种模型架构
- 算子级优化

**显存管理**
- 动态显存分配
- KV Cache管理
- 内存池复用

**算子优化**
- 针对NPU的自定义算子
- FlashAttention适配
- MoE算子优化

### 2.4 通信机制

**SHM共享内存通信**
- 本地实例间高效通信
- 零拷贝数据传输
- 低延迟KV Cache传输

**RDMA高速传输**
- 跨节点通信优化
- 高吞吐数据传输
- 支持分布式部署

## 三、核心调度策略

### 3.1 实例调度

xLLM实现了多种实例调度策略，根据不同场景选择最优方案：

**Round Robin**
- 最简单的负载均衡策略
- 按顺序分配请求到各实例
- 适用于请求均匀的场景

**Prefix Cache-Aware**
- 基于前缀缓存命中率调度
- 将请求路由到缓存命中率最高的实例
- 显著减少重复计算

```python
# 伪代码示例
def prefix_cache_aware_schedule(request, instances):
    best_instance = None
    best_hit_rate = -1
    
    for instance in instances:
        hit_rate = instance.prefix_cache.match_rate(request.prefix)
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_instance = instance
    
    return best_instance
```

**KV Cache-Aware**
- 基于实例显存空闲程度调度
- 优先分配到显存充足的实例
- 避免OOM错误

**自适应PD动态调度**
- 针对PD分离场景
- 动态调整P实例和D实例比例
- 应对流量和长度突变

### 3.2 请求调度

**Continuous Batching**
- 动态批处理
- 请求到达即加入批次
- 最大化GPU利用率

**Chunked Prefill**
- 将长Prefill切分为多个chunk
- 与Decode交替执行
- 减少Prefill对Decode的阻塞

```
时间线示例：
├── Prefill Chunk 1 ├── Decode Batch ├── Prefill Chunk 2 ├── Decode Batch ├── ...
```

**Prefill/Decode优先级**
- Prefill优先：降低TTFT
- Decode优先：降低TPOT
- 可根据业务需求灵活配置

## 四、PD分离架构

### 4.1 为什么需要PD分离？

大模型推理的两个阶段具有截然不同的特点：

| 特性 | Prefill阶段 | Decode阶段 |
|------|-------------|------------|
| 计算特点 | 计算密集 | 访存密集 |
| 并行度 | 高（所有token并行） | 低（逐token生成） |
| 资源需求 | 大量算力 | 高带宽显存 |
| 延迟特点 | 一次性延迟 | 持续延迟 |

传统架构将两者耦合在同一实例中，难以针对各自特点优化。

### 4.2 xLLM的PD分离实现

```
┌─────────────────────────────────────────────────────────────────┐
│                        PD分离架构                                │
│                                                                  │
│  ┌──────────────┐         KV Cache         ┌──────────────┐     │
│  │              │  ───────────────────────▶│              │     │
│  │  P Instance  │      高速传输通道         │  D Instance  │     │
│  │  (Prefill)   │                          │  (Decode)    │     │
│  │              │◀───────────────────────  │              │     │
│  └──────────────┘      请求调度反馈         └──────────────┘     │
│                                                                  │
│  特点：                                                          │
│  - P实例：大batch，高算力利用率                                   │
│  - D实例：小batch，低延迟                                         │
│  - KV Cache：通过共享内存/RDMA高效传输                            │
└─────────────────────────────────────────────────────────────────┘
```

**核心优势**：

1. **独立优化**：P实例和D实例可以独立配置和优化
2. **资源隔离**：避免Prefill阻塞Decode
3. **弹性扩展**：可以根据负载动态调整P/D比例

### 4.3 动态PD比例调整

xLLM实现了自适应的PD动态调度器：

**流量自适应**
- 监控请求队列长度
- 动态调整P/D实例比例
- 应对流量波动

**长度自适应**
- 分析请求输入输出长度分布
- 预测资源需求
- 提前调整实例配置

```python
# 动态调度伪代码
class DynamicPDScheduler:
    def adjust_ratio(self, metrics):
        # 根据P实例队列长度
        p_queue_length = metrics.p_queue_length
        # 根据D实例延迟
        d_latency = metrics.d_latency_p99
        
        if p_queue_length > threshold_high:
            # 增加P实例
            self.scale_up_prefill()
        elif d_latency > latency_threshold:
            # 增加D实例
            self.scale_up_decode()
```

## 五、架构优势总结

xLLM的服务-引擎解耦架构带来了以下优势：

**高扩展性**
- Service层和Engine层独立扩展
- 支持多实例、多节点部署
- 灵活的资源调配

**高资源利用率**
- PD分离优化资源使用
- 智能调度减少空闲
- 显存复用降低浪费

**高可用性**
- 实例故障自动恢复
- 请求重试和降级
- 多副本容错

**易维护性**
- 清晰的模块边界
- 独立的功能迭代
- 便于测试和调试

## 六、代码导读

### 6.1 核心数据结构

```cpp
// Engine层核心接口
class Engine {
public:
    virtual void forward(Batch& batch) = 0;
    virtual void allocate_memory(size_t size) = 0;
    virtual void free_memory(void* ptr) = 0;
};

// Service层调度器
class Scheduler {
public:
    virtual Instance select_instance(const Request& request) = 0;
    virtual void update_metrics(const Instance& instance, 
                                const Metrics& metrics) = 0;
};
```

### 6.2 关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_disagg_pd` | 启用PD分离 | false |
| `enable_chunked_prefill` | 启用分块Prefill | true |
| `enable_schedule_overlap` | 启用调度重叠 | true |
| `max_num_batched_tokens` | 最大批处理token数 | 8192 |

### 6.3 启动示例

```bash
# 启动PD分离架构
/path/to/xllm \
  --model /path/to/model \
  --devices="npu:0,npu:1" \
  --enable_disagg_pd=true \
  --prefill_instances=1 \
  --decode_instances=1
```

## 七、小结

本文深入解析了xLLM的服务-引擎解耦架构：

- **架构设计**：Service层负责调度，Engine层负责计算
- **调度策略**：多种智能调度算法，适应不同场景
- **PD分离**：针对Prefill和Decode特点独立优化
- **动态调整**：自适应流量和长度变化

**下一篇文章预告**：《ACLGraph：NPU推理加速的秘密武器》，将深入介绍NPU专属优化技术。

---

> **参考资料**：
> - [xLLM技术报告](https://arxiv.org/abs/2510.14686)
> - [xLLM文档 - 架构设计](https://xllm.readthedocs.io/zh-cn/latest/)

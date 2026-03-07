# xLLM在京东：日均亿级请求的推理实践

> 本文分享xLLM在京东内部的落地实践，展示国产芯片在企业级场景中的应用成果。

## 一、业务背景

### 1.1 京东AI应用场景

京东作为领先的电商平台，AI技术已深入应用于各个业务环节：

```
┌──────────────────────────────────────────────────────────────┐
│                    京东AI应用全景                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    智能客服                              │ │
│  │  - 智能问答：日均处理5000万+次咨询                       │ │
│  │  - 意图识别：准确率98%+                                  │ │
│  │  - 情感分析：实时监控用户情绪                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    风控系统                              │ │
│  │  - 欺诈检测：实时识别异常交易                            │ │
│  │  - 内容审核：商品描述、用户评论审核                      │ │
│  │  - 风险评估：商家信用评估                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    供应链优化                            │ │
│  │  - 需求预测：销量预测准确率95%+                          │ │
│  │  - 库存优化：降低库存成本20%+                            │ │
│  │  - 路径规划：配送效率提升15%+                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    广告推荐                              │ │
│  │  - 个性化推荐：点击率提升30%+                            │ │
│  │  - 创意生成：自动生成广告文案                            │ │
│  │  - 智能出价：ROI提升25%+                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 推理需求特点

京东的大模型推理场景具有以下特点：

**高并发**
- 日均请求量：数亿次
- 峰值QPS：数万
- 要求：高吞吐、低延迟

**多场景**
- 不同场景对延迟要求不同
- 在线服务：要求<500ms
- 离线处理：可接受分钟级

**成本敏感**
- 大规模部署，成本压力大
- 需要平衡性能与成本
- 追求最优TCO

**高可用**
- 7×24小时服务
- 故障自动恢复
- 多机房容灾

### 1.3 为什么选择xLLM + NPU？

**技术选型考量**

| 维度 | GPU方案 | NPU方案 | 评价 |
|------|---------|---------|------|
| 性能 | 高 | 中高 | 满足业务需求 |
| 成本 | 高 | 低 | NPU优势明显 |
| 供应 | 紧张 | 充足 | NPU供应稳定 |
| 自主可控 | 依赖进口 | 国产自主 | NPU符合信创要求 |
| 生态 | 成熟 | 发展中 | xLLM补齐生态短板 |

**决策因素**

```
选择xLLM + NPU的核心原因：

1. 成本优势
   - 硬件成本降低40%
   - 运维成本降低30%
   - TCO降低35%

2. 供应保障
   - 不受国际形势影响
   - 采购周期短
   - 产能充足

3. 技术能力
   - xLLM性能满足业务需求
   - 支持主流模型
   - 持续优化迭代

4. 战略意义
   - 符合国产化战略
   - 技术自主可控
   - 支持国产芯片生态
```

## 二、技术方案

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                    xLLM生产架构                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    接入层                                │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐  │ │
│  │  │ API网关 │  │ 负载均衡│  │  认证鉴权              │  │ │
│  │  └─────────┘  └─────────┘  └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    调度层                                │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐  │ │
│  │  │请求队列 │  │智能路由 │  │  弹性伸缩              │  │ │
│  │  └─────────┘  └─────────┘  └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    计算层                                │ │
│  │  ┌─────────────────────────────────────────────────────┐│ │
│  │  │  xLLM集群 (NPU)                                     ││ │
│  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  ││ │
│  │  │  │实例1│ │实例2│ │实例3│ │实例4│ │实例5│ │实例N│  ││ │
│  │  │  │Qwen │ │Qwen │ │GLM  │ │Deep │ │Qwen │ │...  │  ││ │
│  │  │  │ 3-8B│ │3-32B│ │ 4.5 │ │Seek │ │3-VL │ │     │  ││ │
│  │  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  ││ │
│  │  └─────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    存储层                                │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐  │ │
│  │  │模型仓库 │  │KV Cache │  │  监控数据              │  │ │
│  │  └─────────┘  └─────────┘  └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

**API网关**

```yaml
# Kong网关配置
_format_version: "3.0"

services:
  - name: xllm-service
    url: http://xllm-backend:18000
    routes:
      - name: chat-route
        paths:
          - /v1/chat
    plugins:
      - name: rate-limiting
        config:
          minute: 1000
          policy: local
      - name: jwt
        config:
          secret: your_secret
```

**负载均衡**

```nginx
# Nginx负载均衡配置
upstream xllm_cluster {
    least_conn;
    server 10.0.1.1:18001 weight=5;
    server 10.0.1.2:18002 weight=5;
    server 10.0.1.3:18003 weight=3;
    server 10.0.1.4:18004 weight=3;
    
    keepalive 100;
    keepalive_timeout 60s;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://xllm_cluster;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

**智能路由**

```python
# 基于请求特征的路由策略
class IntelligentRouter:
    def __init__(self):
        self.instances = {
            'small': ['instance1', 'instance2'],  # 小模型实例
            'large': ['instance3', 'instance4'],  # 大模型实例
            'vlm': ['instance5'],                  # 多模态实例
        }
    
    def route(self, request):
        # 根据模型类型路由
        model_type = self.classify_model(request.model)
        
        # 根据负载选择实例
        instance = self.select_instance(model_type)
        
        return instance
    
    def select_instance(self, model_type):
        instances = self.instances[model_type]
        
        # 选择负载最低的实例
        loads = [self.get_load(i) for i in instances]
        return instances[loads.index(min(loads))]
```

**弹性伸缩**

```python
# 基于负载的自动伸缩
class AutoScaler:
    def __init__(self, min_instances=2, max_instances=20):
        self.min_instances = min_instances
        self.max_instances = max_instances
        
    def check_and_scale(self):
        metrics = self.get_metrics()
        
        # 计算当前负载
        cpu_util = metrics['cpu_utilization']
        queue_len = metrics['queue_length']
        
        # 扩容条件
        if cpu_util > 80 or queue_len > 100:
            self.scale_up()
        
        # 缩容条件
        elif cpu_util < 30 and queue_len < 10:
            self.scale_down()
    
    def scale_up(self):
        current = self.get_instance_count()
        if current < self.max_instances:
            self.launch_instance()
    
    def scale_down(self):
        current = self.get_instance_count()
        if current > self.min_instances:
            self.terminate_instance()
```

### 2.3 多模型部署

**模型矩阵**

| 业务场景 | 模型 | 部署规模 |
|----------|------|----------|
| 智能客服 | Qwen3-8B | 50实例 |
| 内容生成 | Qwen3-32B | 20实例 |
| 多模态理解 | Qwen3-VL | 10实例 |
| MoE推理 | DeepSeek-V3 | 5实例 |
| 代码生成 | Qwen2.5-Coder | 10实例 |

**模型热更新**

```bash
# 无缝更新模型
# 1. 启动新版本实例
./start_instance.sh --model Qwen3-8B-v2 --port 18099

# 2. 健康检查
curl http://localhost:18099/health

# 3. 切换流量
nginx -s reload

# 4. 下线旧版本
./stop_instance.sh --port 18001
```

## 三、性能数据

### 3.1 线上性能

**整体指标**

| 指标 | 数值 |
|------|------|
| 日均请求量 | 1.2亿次 |
| 峰值QPS | 25000 |
| 平均延迟 | 180ms |
| P99延迟 | 450ms |
| 可用性 | 99.95% |

**分场景性能**

| 场景 | 日均请求 | 平均延迟 | P99延迟 |
|------|----------|----------|---------|
| 智能客服 | 5000万 | 120ms | 300ms |
| 内容生成 | 3000万 | 250ms | 600ms |
| 多模态 | 1000万 | 350ms | 800ms |
| 代码生成 | 500万 | 200ms | 500ms |

### 3.2 资源利用

**NPU利用率**

```
NPU利用率分布（24小时）

利用率
 100%┤
     │              ┌───────┐
  80%┤         ┌────┤       ├────┐
     │    ┌────┤    │       │    └────┐
  60%┤────┤    │    │       │         │
     │    │    │    │       │         │
  40%┤    │    │    │       │         │
     │    │    │    │       │         │
  20%┤    │    │    │       │         │
     │    │    │    │       │         │
   0%┼────┴────┴────┴───────┴─────────┴────
      0    4    8   12   16   20   24  时间

平均利用率: 72%
峰值利用率: 95%
```

**显存利用率**

| 模型 | 显存利用率 | 最大batch |
|------|------------|-----------|
| Qwen3-8B | 92% | 180 |
| Qwen3-32B | 90% | 64 |
| Qwen3-VL | 88% | 32 |

### 3.3 成本收益

**成本对比**

| 项目 | GPU方案 | NPU方案 | 节省 |
|------|---------|---------|------|
| 硬件采购 | ¥5000万 | ¥3000万 | 40% |
| 年电费 | ¥800万 | ¥400万 | 50% |
| 年运维 | ¥500万 | ¥350万 | 30% |
| 3年TCO | ¥8900万 | ¥5400万 | 39% |

**ROI分析**

```
投资回报分析：

初始投资：
├── NPU硬件: ¥3000万
├── xLLM开发适配: ¥500万
├── 基础设施: ¥300万
└── 总计: ¥3800万

年度节省：
├── 硬件成本节省: ¥670万/年
├── 电费节省: ¥400万/年
├── 运维节省: ¥150万/年
└── 总计: ¥1220万/年

投资回报周期: 3800 / 1220 ≈ 3.1年
3年总收益: 1220 × 3 - 3800 = -140万
5年总收益: 1220 × 5 - 3800 = +2300万
```

## 四、踩坑与解决

### 4.1 问题一：冷启动延迟高

**问题描述**

```
场景：新实例启动后，首次请求延迟异常高

现象：
├── 正常请求延迟: 200ms
├── 首次请求延迟: 5000ms+
└── 影响: 弹性伸缩效果差
```

**解决方案**

```python
# 预热机制
class WarmupManager:
    def warmup(self, instance):
        # 发送预热请求
        warmup_prompts = [
            "Hello, this is a warmup request.",
            "请介绍一下你自己。",
            "What is the capital of France?",
        ]
        
        for prompt in warmup_prompts:
            instance.request(prompt, max_tokens=10)
        
        # 等待ACLGraph编译完成
        time.sleep(30)
        
        # 标记实例为就绪
        instance.mark_ready()
```

**效果**

```
预热前后对比：

预热前：
├── 首次请求延迟: 5000ms
├── 第二次请求延迟: 200ms

预热后：
├── 首次请求延迟: 250ms
├── 第二次请求延迟: 200ms
```

### 4.2 问题二：长尾延迟

**问题描述**

```
场景：P99延迟远高于P50延迟

现象：
├── P50延迟: 150ms
├── P90延迟: 300ms
├── P99延迟: 800ms
└── 影响: 用户体验不稳定
```

**解决方案**

```python
# 请求超时和重试
class TimeoutRetry:
    def __init__(self, timeout_ms=500, max_retries=2):
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
    
    def request_with_retry(self, prompt):
        for i in range(self.max_retries):
            try:
                result = self._request(prompt, timeout=self.timeout_ms)
                return result
            except TimeoutError:
                if i == self.max_retries - 1:
                    raise
                continue
    
    def _request(self, prompt, timeout):
        # 发送请求，设置超时
        pass

# 请求优先级队列
class PriorityQueue:
    def __init__(self):
        self.high_priority = Queue()
        self.normal_priority = Queue()
        self.low_priority = Queue()
    
    def enqueue(self, request, priority='normal'):
        if priority == 'high':
            self.high_priority.put(request)
        elif priority == 'low':
            self.low_priority.put(request)
        else:
            self.normal_priority.put(request)
    
    def dequeue(self):
        if not self.high_priority.empty():
            return self.high_priority.get()
        if not self.normal_priority.empty():
            return self.normal_priority.get()
        return self.low_priority.get()
```

**效果**

```
优化前后对比：

优化前：
├── P50: 150ms
├── P90: 300ms
├── P99: 800ms

优化后：
├── P50: 150ms
├── P90: 250ms
├── P99: 400ms
```

### 4.3 问题三：显存碎片

**问题描述**

```
场景：长时间运行后，显存利用率下降

现象：
├── 初始显存利用率: 92%
├── 运行24小时后: 85%
├── 运行72小时后: 78%
└── 影响: 最大batch size下降
```

**解决方案**

```python
# 定期显存整理
class MemoryDefragmenter:
    def __init__(self, interval_hours=24):
        self.interval = interval_hours * 3600
    
    def run_defragmentation(self):
        # 1. 暂停新请求
        self.pause_new_requests()
        
        # 2. 等待现有请求完成
        self.wait_for_completion()
        
        # 3. 整理显存
        self.defrag_memory()
        
        # 4. 恢复服务
        self.resume_service()
    
    def defrag_memory(self):
        # 使用xTensor的显存整理功能
        xtensor_manager.defragment()
```

**效果**

```
显存利用率变化：

整理前（运行72小时）：
├── 显存利用率: 78%
├── 最大batch: 140

整理后：
├── 显存利用率: 91%
├── 最大batch: 178
```

### 4.4 最佳实践总结

```
生产环境最佳实践：

1. 预热机制
   - 新实例启动后预热
   - 预编译ACLGraph
   - 预加载常用模型

2. 超时重试
   - 设置合理超时时间
   - 实现自动重试
   - 降级策略

3. 监控告警
   - 实时监控关键指标
   - 设置多级告警
   - 自动故障恢复

4. 容量规划
   - 预留20%冗余
   - 定期容量评估
   - 提前扩容

5. 定期维护
   - 显存整理
   - 日志清理
   - 版本更新
```

## 五、未来规划

### 5.1 技术演进

**短期计划（6个月）**

- 支持更多模型：GLM-5、Kimi-k2等
- 优化多模态性能
- 完善监控告警体系

**中期计划（1年）**

- 全局KV Cache管理（基于Mooncake）
- PD分离架构全面落地
- 多机房容灾

**长期愿景（2年）**

- 支持更多国产芯片
- 构建完整的国产AI生态
- 开源社区建设

### 5.2 业务拓展

```
业务拓展方向：

1. 更多业务场景
   ├── 财务报表分析
   ├── 法律合同审核
   ├── 医疗健康咨询
   └── 教育智能辅导

2. 对外服务
   ├── AI开放平台
   ├── 行业解决方案
   └── 私有化部署

3. 生态合作
   ├── 与模型厂商合作
   ├── 与ISV合作
   └── 与高校合作
```

## 六、小结

本文分享了xLLM在京东内部的落地实践：

- **业务背景**：京东AI应用场景和推理需求
- **技术方案**：整体架构、关键组件、多模型部署
- **性能数据**：日均亿级请求、高可用性、成本收益
- **踩坑解决**：冷启动、长尾延迟、显存碎片
- **未来规划**：技术演进、业务拓展

xLLM + NPU的组合在京东证明了其生产级能力，实现了高性能、低成本、高可用的推理服务，为国产AI芯片的落地应用提供了成功案例。

---

> **关于xLLM团队**：xLLM是京东零售技术团队开源的大模型推理框架，专注于国产芯片优化。欢迎在GitHub上关注我们：https://github.com/jd-opensource/xllm

> **致谢**：感谢NPU厂商的技术支持，感谢所有为xLLM贡献代码的开发者，感谢京东内部各业务团队的信任和支持。

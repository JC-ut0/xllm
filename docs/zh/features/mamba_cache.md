# Mamba Cache 优化

## 功能介绍

xLLM支持针对线性注意力层（如Qwen-Next的Gated Delta Net）的Mamba Cache优化。Qwen-Next是一种混合架构模型，结合了标准Transformer注意力和线性注意力，其Prefix Cache实现与传统LLM模型存在显著差异。

### 架构特点

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Qwen-Next 层结构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Layer 0:    Linear Attention (GDN) - 需要 conv_cache, ssm_cache       │
│   Layer 1:    Linear Attention (GDN) - 需要 conv_cache, ssm_cache       │
│   Layer 2:    Linear Attention (GDN) - 需要 conv_cache, ssm_cache       │
│   Layer 3:    Standard Attention - 需要 key_cache, value_cache          │
│   ...                                                                    │
│   (每4层一个标准注意力层，其余为线性注意力层)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mamba Cache Mode

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `none` | 不缓存Mamba状态 | Prefix Cache禁用时 |
| `all` | 缓存所有block边界位置的状态 | 不适用于Qwen-Next |
| `align` | 仅在调度结束时缓存对齐位置的状态 | **Qwen-Next推荐使用** |

## 使用方式

### Python API

```python
from xllm import LLM, RequestParams

# 创建LLM实例，启用prefix cache和mamba cache
llm = LLM(
    model="/path/to/qwen3-next-model",
    devices="npu:0",              # 或 "cuda:0"
    block_size=560,               # 建议设置为560以获得更好的对齐
    disable_prefix_cache=False,   # 启用prefix cache (默认启用)
    mamba_cache_mode="align",     # 启用mamba cache align模式
    max_tokens_per_batch=20480,
    max_seqs_per_batch=256,
)

# 生成文本
output = llm.generate("你好，请介绍一下你自己。", RequestParams())
print(output[0].text)

# 多个请求共享前缀时，prefix cache会自动复用
prompts = [
    "请解释什么是机器学习？",
    "请解释什么是深度学习？",
    "请解释什么是自然语言处理？",
]
outputs = llm.generate(prompts, RequestParams())

llm.finish()
```

### 命令行启动

```bash
# 启动服务
python -m xllm.server \
    --model /path/to/qwen3-next-model \
    --devices npu:0 \
    --block_size 560 \
    --enable_prefix_cache=true \
    --mamba_cache_mode=align
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `disable_prefix_cache` | `False` | 是否禁用prefix cache |
| `mamba_cache_mode` | `"none"` | Mamba缓存模式 |
| `block_size` | `128` | KV Cache block大小 |

### 推荐配置

对于Qwen-Next模型，推荐以下配置：

```python
llm = LLM(
    model="/path/to/qwen3-next-model",
    block_size=560,               # 与vLLM保持一致
    disable_prefix_cache=False,   # 启用prefix cache
    mamba_cache_mode="align",     # 使用align模式
)
```

## 工作原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Qwen-Next Prefix Cache 工作流程                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  请求1: "请解释什么是机器学习？请详细介绍..."                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Prefill (560 tokens)                                             │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  Standard Attention: 计算KV Cache → 保存到PrefixCache        │ │  │
│  │  │  Linear Attention (GDN): 计算状态 → 在560位置保存状态        │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  请求2: "请解释什么是机器学习？请简要说明..."                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Prefix Match (命中前560个tokens)                                 │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  Standard Attention: 从PrefixCache加载KV Cache               │ │  │
│  │  │  Linear Attention (GDN): 从MambaCache加载状态                │ │  │
│  │  │  → 跳过前560个tokens的计算，直接继续forward                   │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Align模式状态缓存

```
Token Position:  0   100  200  300  400  500  600  700  800  900
                 │    │    │    │    │    │    │    │    │    │
                 ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼

Mode: "align" (推荐用于Qwen-Next)

Prefill Step 1 (560 tokens):
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX┌───┐
                                                   │ S │
pos:0                                             560│
                                                   └───┘

Prefill Step 2 (560 tokens):
[复用缓存状态]────────────────────────────────────┐┌───┐
                                                 ││ S │
pos:560                                       1120│
                                                 └───┘

S = 在block边界保存的状态
X = 当前步骤计算的tokens
```

## 注意事项

1. **block_size对齐**: 建议设置`block_size=560`，与vLLM保持一致，确保状态在正确的边界保存
2. **自动选择**: 如果不指定`mamba_cache_mode`，系统会在启用prefix cache时自动选择`align`模式
3. **状态依赖**: 线性注意力层的状态具有递归依赖性，必须在正确的位置保存和恢复

## 性能效果

开启Mamba Cache后，在Qwen-Next模型上，对于共享前缀的请求：
- 减少重复计算，提升prefill速度
- 降低首token延迟（TTFT）
- 提高整体吞吐量

## 参考资料

- vLLM实现: [PR #30877](https://github.com/vllm-project/vllm/pull/30877)
- 设计文档: [qwen_next_prefix_cache_design.md](../design/qwen_next_prefix_cache_design.md)

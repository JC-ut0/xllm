# Qwen3-Next 模型 Prefix Cache 特性适配文档

## 1. 概述

本文档详细描述 Qwen3-Next 模型的 Prefix Cache 特性适配方案。Qwen3-Next 是一种混合架构模型，结合了标准 Attention 和 Linear Attention（Gated Delta Net）两种注意力机制。经过深入分析，我们发现 **Qwen3-Next 模型在当前实现下只能部分支持 Prefix Cache**，仅有标准 Attention 层的 Attention 部分可以复用已缓存的 KV。

---

## 2. 背景：Prefix Cache 基础原理

### 2.1 什么是 Prefix Cache？

在 LLM 推理应用中，经常面临以下场景：

1. **长 System Prompt 场景**：不同请求使用相同的 System Prompt
2. **多轮对话场景**：每轮对话需要依赖历史上下文

Prefix Cache 的核心思想是：**将公共前缀 token 的 KV Cache 保存下来，供后续请求复用**，从而避免重复计算。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefix Cache 工作原理                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  请求1: [System Prompt A] + [User Query 1]                      │
│         └──────────────────┘                                     │
│              ↓ 计算 KV Cache                                     │
│         [KV Cache A] 缓存起来                                    │
│                                                                 │
│  请求2: [System Prompt A] + [User Query 2]                      │
│         └──────────────────┘                                     │
│              ↓ 命中缓存！直接复用                                 │
│         [KV Cache A] 无需重新计算                                │
│                                                                 │
│  收益: 跳过 System Prompt 的 KV 计算，降低 TTFT                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 传统 Transformer 的 Prefix Cache 实现

对于标准 Transformer 模型（如 LLaMA、Qwen2.5），Prefix Cache 的实现相对简单：

```
┌─────────────────────────────────────────────────────────────────┐
│              标准 Transformer 的 KV Cache 结构                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  每个 Token 生成一对 KV 向量:                                     │
│                                                                 │
│  Token:    [T1]  [T2]  [T3]  [T4]  [T5]  [T6]  [T7]  ...        │
│            ↓     ↓     ↓     ↓     ↓     ↓     ↓                │
│  Key:      [K1]  [K2]  [K3]  [K4]  [K5]  [K6]  [K7]  ...        │
│  Value:    [V1]  [V2]  [V3]  [V4]  [V5]  [V6]  [V7]  ...        │
│                                                                 │
│  KV Cache 按 Block 存储 (假设 block_size = 4):                   │
│                                                                 │
│  Block 0: [K1,K2,K3,K4] [V1,V2,V3,V4]  ← 可共享                 │
│  Block 1: [K5,K6,K7,...] [V5,V6,V7,...]                         │
│                                                                 │
│  特点:                                                          │
│  ✓ 每个 token 的 KV 独立存储                                     │
│  ✓ 可以按 block 边界任意切分和共享                               │
│  ✓ Hash 匹配高效 (基于 token 序列)                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 xLLM 中 Prefix Cache 的核心机制

#### 2.3.1 计算跳过机制

xLLM 通过 `n_kv_cache_tokens` 参数控制哪些 token 需要计算：

```
┌─────────────────────────────────────────────────────────────────┐
│                    n_kv_cache_tokens 机制                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  代码位置: batch_input_builder.cpp:276-282                      │
│                                                                 │
│  const uint32_t n_kv_cache_tokens =                             │
│      sequence->kv_state().kv_cache_tokens_num();                │
│  const uint32_t q_seq_len = n_tokens - n_kv_cache_tokens;       │
│                                                                 │
│  // 只有 n_kv_cache_tokens 之后的 token 才会被处理               │
│  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {       │
│      state.flatten_tokens_vec.emplace_back(token_ids[j]);       │
│  }                                                              │
│                                                                 │
│  说明:                                                          │
│  • n_kv_cache_tokens: 已缓存的 token 数量                       │
│  • 输入模型的 tokens: 只包含未缓存的 token                       │
│  • Embedding、MLP 等只计算未缓存的 token                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 Hash 匹配机制

```
┌─────────────────────────────────────────────────────────────────┐
│                    增量 Hash 计算流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Token 序列: [T1, T2, T3, T4] [T5, T6, T7, T8] [T9, ...]        │
│              └──── Block 0 ────┘ └──── Block 1 ────┘            │
│                                                                 │
│  Block 0 Hash:                                                  │
│    hash_0 = MurmurHash3([T1, T2, T3, T4])                       │
│                                                                 │
│  Block 1 Hash (增量计算):                                        │
│    hash_1 = MurmurHash3(hash_0 + [T5, T6, T7, T8])              │
│                                                                 │
│  匹配过程:                                                       │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                    │
│  │ Block 0 │────▶│ Block 1 │────▶│ Block 2 │ ...                │
│  │ hash_0  │     │ hash_1  │     │ hash_2  │                    │
│  └────┬────┘     └────┬────┘     └────┬────┘                    │
│       │               │               │                         │
│       ▼               ▼               ▼                         │
│    [命中!]         [命中!]        [未命中] → 停止匹配            │
│                                                                 │
│  结果: 返回 Block 0, Block 1 的缓存引用                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Qwen3-Next 模型架构分析

### 3.1 混合 Attention 架构

Qwen3-Next 采用混合注意力架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                 Qwen3-Next 模型层结构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  总层数: 48 层                                                   │
│  full_attention_interval: 4 (每 4 层使用标准 Attention)          │
│                                                                 │
│  Layer 0:  Linear Attention (Gated Delta Net)                   │
│  Layer 1:  Linear Attention (Gated Delta Net)                   │
│  Layer 2:  Linear Attention (Gated Delta Net)                   │
│  Layer 3:  标准 Attention ◀── 每 4 层一次                        │
│  Layer 4:  Linear Attention (Gated Delta Net)                   │
│  Layer 5:  Linear Attention (Gated Delta Net)                   │
│  Layer 6:  Linear Attention (Gated Delta Net)                   │
│  Layer 7:  标准 Attention ◀── 每 4 层一次                        │
│  ...                                                            │
│  Layer 47: 标准 Attention ◀── 最后一层                           │
│                                                                 │
│  统计:                                                          │
│  • 标准 Attention 层数: 12 层 (25%)                              │
│  • Linear Attention 层数: 36 层 (75%)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 两种 Attention 的缓存结构对比

```
┌─────────────────────────────────────────────────────────────────┐
│              标准 Attention vs Linear Attention 缓存对比         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              标准 Attention (每 4 层)                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Token:  [T1] [T2] [T3] [T4] [T5] [T6] [T7] ...        │   │
│  │          ↓    ↓    ↓    ↓    ↓    ↓    ↓                │   │
│  │  Key:    [K1] [K2] [K3] [K4] [K5] [K6] [K7] ...        │   │
│  │  Value:  [V1] [V2] [V3] [V4] [V5] [V6] [V7] ...        │   │
│  │                                                         │   │
│  │  缓存结构:                                               │   │
│  │  • key_cache: [n_blocks, n_kv_heads, block_size, head_dim]│  │
│  │  • value_cache: [n_blocks, n_kv_heads, block_size, head_dim]│  │
│  │                                                         │   │
│  │  特点: ✅ 每个 token 独立存储，可按 block 共享            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Linear Attention (Gated Delta Net)             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  递归状态更新:                                            │   │
│  │  state_0 → state_1 → state_2 → ... → state_n            │   │
│  │    ↑          ↑          ↑              ↑                │   │
│  │  [T1]       [T2]       [T3]    ...    [Tn]              │   │
│  │                                                         │   │
│  │  缓存结构:                                               │   │
│  │  • conv_cache: [n_blocks, channels, kernel_dim-1]       │   │
│  │    - 存储 1D 卷积的滑动窗口状态                           │   │
│  │                                                         │   │
│  │  • ssm_cache: [n_blocks, n_v_heads, k_dim, k_dim]       │   │
│  │    - 存储递归状态 (压缩的序列记忆)                        │   │
│  │    - 大小固定，与序列长度无关                             │   │
│  │                                                         │   │
│  │  特点: ❌ 递归状态依赖完整历史，无法按 token 分块          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Gated Delta Net 的递归计算

```
┌─────────────────────────────────────────────────────────────────┐
│              Gated Delta Net 递归状态计算                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  核心公式:                                                       │
│  state_t = state_{t-1} * g_t + k_t ⊗ (v_t - state_{t-1} * k_t) * β_t│
│                                                                 │
│  计算流程:                                                       │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ state_0  │───▶│ state_1  │───▶│ state_2  │───▶│ state_n  │  │
│  │ (zeros)  │    │          │    │          │    │ (final)  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│    [T1 处理]      [T2 处理]      [T3 处理]      [Tn 处理]      │
│                                                                 │
│  关键特点:                                                       │
│  • state_t 依赖 state_{t-1}，形成链式依赖                        │
│  • 每个 token 的计算都需要完整的历史状态                          │
│  • 无法独立计算某个 token 的输出                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 问题分析：Prefix Cache 与 Linear Attention 的冲突

### 4.1 核心问题

```
┌─────────────────────────────────────────────────────────────────┐
│           Prefix Cache 命中时的计算流程分析                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  假设: Prefix = 1000 tokens (已缓存), 新请求 = 100 tokens        │
│                                                                 │
│  输入 tokens: 只包含 100 个新 token (不包含前缀)                  │
│              这是由 n_kv_cache_tokens 机制决定的                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              标准 Attention 层 (12层)                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  输入 hidden_states: [100 tokens]                       │   │
│  │                                                         │   │
│  │  Embedding: 计算 100 tokens (无法跳过，输入只有100个)    │   │
│  │                                                         │   │
│  │  Attention:                                             │   │
│  │    • Q: 100 tokens (从 hidden_states)                   │   │
│  │    • K, V 前1000个: 从 KV Cache 读取 ✅ 复用             │   │
│  │    • K, V 后100个: 需要新计算                            │   │
│  │    • Attention 计算: Q × K^T (使用完整的 K)              │   │
│  │                                                         │   │
│  │  MLP: 计算 100 tokens (无法跳过，输入只有100个)          │   │
│  │                                                         │   │
│  │  收益: ✅ 跳过前缀的 KV 计算 (1000 tokens)               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Linear Attention 层 (36层)                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  输入 hidden_states: [100 tokens]                       │   │
│  │                                                         │   │
│  │  Embedding: 计算 100 tokens (无法跳过)                   │   │
│  │                                                         │   │
│  │  Linear Attention:                                      │   │
│  │    • 输入: 只有 100 个新 token 的 q, k, v               │   │
│  │    • ssm_cache: 存储的是之前序列的最终状态               │   │
│  │    • 问题: 新 token 需要基于前缀的 ssm_state 计算        │   │
│  │    • 但 ssm_state 是基于完整序列计算的，无法复用         │   │
│  │                                                         │   │
│  │  MLP: 计算 100 tokens (无法跳过)                         │   │
│  │                                                         │   │
│  │  结果: ❌ 无法正确计算，ssm_state 不匹配！               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Linear Attention 层无法工作的原因

```
┌─────────────────────────────────────────────────────────────────┐
│              Linear Attention 层的核心问题                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  正常情况 (无 Prefix Cache):                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输入: [T1, T2, ..., T1000, T1001, ..., T1100]          │   │
│  │                                                         │   │
│  │  ssm_state 计算:                                         │   │
│  │    state_0 → state_1 → ... → state_1000 → ... → state_1100│  │
│  │                                                         │   │
│  │  最终: state_1100 存入 ssm_cache                        │   │
│  │                                                         │   │
│  │  ✅ 正确工作                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Prefix Cache 命中时:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输入: [T1001, ..., T1100]  (只有新 token)              │   │
│  │                                                         │   │
│  │  ssm_cache 中存储: state_1000 (之前序列的最终状态)       │   │
│  │                                                         │   │
│  │  问题:                                                   │   │
│  │    • state_1000 是基于 [T1...T1000] 计算的              │   │
│  │    • 新输入 [T1001...T1100] 的 k, v 是新计算的          │   │
│  │    • 新 k, v 与 state_1000 不匹配                       │   │
│  │    • 无法正确计算 output                                │   │
│  │                                                         │   │
│  │  ❌ 无法正确工作                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 当前实现的局限性

从代码分析：

```cpp
// qwen3_next_gated_delta_net.cpp:411-416
if (attn_metadata.is_prefill) {
    std::tie(core_attn_out, last_recurrent_state) =
        torch_chunk_gated_delta_rule(...);
    // 只存储最终状态！
    ssm_cache.index_put_({input_params.block_tables.select(1, 0)},
                         last_recurrent_state.to(ssm_cache.dtype()));
}
```

**问题确认**：
- `ssm_cache` 只存储整个序列的最终状态
- 没有存储每个 block 边界的中间状态
- 无法支持 Prefix Cache 的部分匹配

---

## 5. 收益分析

### 5.1 各模块的 Prefix Cache 支持情况

```
┌─────────────────────────────────────────────────────────────────┐
│                    各模块 Prefix Cache 支持情况                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    标准 Attention 层                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  模块          │ 是否可跳过 │ 说明                       │   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  Embedding     │ ❌ 否     │ 输入只有新 token           │   │
│  │  QKV 投影      │ ❌ 否     │ 需要计算新 token 的 Q      │   │
│  │  K, V 计算     │ ✅ 是     │ 前缀的 KV 从缓存读取       │   │
│  │  Attention     │ ❌ 否     │ 需要计算 Q × K^T           │   │
│  │  MLP           │ ❌ 否     │ 输入只有新 token           │   │
│  │                                                         │   │
│  │  收益: 跳过前缀的 K, V 计算                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Linear Attention 层                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  模块          │ 是否可跳过 │ 说明                       │   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  Embedding     │ ❌ 否     │ 输入只有新 token           │   │
│  │  QKV 投影      │ ❌ 否     │ 需要计算新 token 的 QKV    │   │
│  │  ssm_state     │ ❌ 否     │ 无法复用，状态不匹配       │   │
│  │  MLP           │ ❌ 否     │ 输入只有新 token           │   │
│  │                                                         │   │
│  │  收益: 无 (无法正确工作)                                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 收益量化估算

```
┌─────────────────────────────────────────────────────────────────┐
│                    收益量化估算                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  假设参数:                                                       │
│  • Prefix 长度: 1000 tokens                                     │
│  • 新请求长度: 100 tokens                                        │
│  • 标准 Attention KV 计算时间: 1.0 ms/100 tokens/layer          │
│  • 其他计算时间: 不变                                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    无 Prefix Cache                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  标准 Attention (12层):                                  │   │
│  │    • KV 计算: 12 × 1.0 × 11 = 132 ms                    │   │
│  │                                                         │   │
│  │  总 KV 计算时间: 132 ms                                  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              有 Prefix Cache (当前实现)                   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  标准 Attention (12层):                                  │   │
│  │    • KV 计算: 12 × 1.0 × 1 = 12 ms (只计算新 token)     │   │
│  │    • 节省: 132 - 12 = 120 ms                            │   │
│  │                                                         │   │
│  │  Linear Attention (36层):                                │   │
│  │    • 无收益 (需要完整计算)                               │   │
│  │                                                         │   │
│  │  净收益: 约 120 ms                                       │   │
│  │  TTFT 降低: 约 10-15% (取决于模型配置)                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 收益总结

| 层类型 | 可复用部分 | 收益 |
|--------|-----------|------|
| 标准 Attention (12层, 25%) | KV Cache | 跳过前缀的 KV 计算 |
| Linear Attention (36层, 75%) | 无 | 无法正确工作 |

**关键结论**：
- 只有 **25% 的层**（标准 Attention）可以受益于 Prefix Cache
- 收益仅限于 **KV 计算部分**
- 整体 TTFT 降低约 **10-15%**

---

## 6. 适配方案

### 6.1 方案一：禁用 Linear Attention 层的 Prefix Cache（推荐）

```
┌─────────────────────────────────────────────────────────────────┐
│              方案一：禁用 Linear Attention 的 Prefix Cache        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略:                                                          │
│  • 标准 Attention 层: 正常使用 Prefix Cache                     │
│  • Linear Attention 层: 禁用 Prefix Cache，完整计算              │
│                                                                 │
│  实现:                                                          │
│  1. 在 BlockManager 中识别 Qwen3-Next 模型                      │
│  2. 根据 full_attention_interval 判断层类型                     │
│  3. Linear Attention 层不参与 Prefix Cache 匹配                 │
│                                                                 │
│  优点:                                                          │
│  • 实现简单，风险低                                             │
│  • 保证正确性                                                   │
│  • 仍有部分收益 (标准 Attention 层)                             │
│                                                                 │
│  缺点:                                                          │
│  • 收益有限 (只有 25% 的层受益)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 方案二：完整支持 Linear Attention 的 Prefix Cache（复杂）

```
┌─────────────────────────────────────────────────────────────────┐
│        方案二：完整支持 Linear Attention 的 Prefix Cache          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略:                                                          │
│  • 存储每个 block 边界的 ssm_state                              │
│  • Prefix Cache 匹配时同时匹配 ssm_state                        │
│                                                                 │
│  实现步骤:                                                       │
│  1. 修改 chunk_gated_delta_rule，输出每个 chunk 的中间状态      │
│  2. 修改 ssm_cache 存储逻辑，存储中间状态                        │
│  3. 修改 Prefix Cache 匹配逻辑，同时匹配 ssm_state              │
│  4. 修改模型前向计算，正确读取和使用 ssm_state                   │
│                                                                 │
│  优点:                                                          │
│  • 完整支持 Prefix Cache                                        │
│  • 收益最大化                                                   │
│                                                                 │
│  缺点:                                                          │
│  • 实现复杂，风险高                                             │
│  • 增加显存开销 (存储中间状态)                                   │
│  • 需要大量测试验证                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 方案对比

| 方案 | 实现复杂度 | 收益 | 风险 | 推荐度 |
|------|-----------|------|------|--------|
| 方案一：禁用 Linear Attention 的 Prefix Cache | 低 | 10-15% | 低 | ✅ **推荐** |
| 方案二：完整支持 | 高 | 40-50% | 高 | ⚠️ 可选 |

---

## 7. 代码修改清单

### 7.1 方案一实现（推荐）

```cpp
// 1. 在 BlockManager 中添加模型类型识别
// 文件: block_manager_impl.h

class BlockManagerImpl : public BlockManager {
 public:
  void set_model_type(const std::string& model_type) {
    model_type_ = model_type;
  }
  
  void set_full_attention_interval(int32_t interval) {
    full_attention_interval_ = interval;
  }
  
  bool is_linear_attention_layer(int32_t layer_id) const {
    if (full_attention_interval_ <= 1) return false;
    return (layer_id + 1) % full_attention_interval_ != 0;
  }

 private:
  std::string model_type_;
  int32_t full_attention_interval_ = 1;
};

// 2. 在 Prefix Cache 匹配时跳过 Linear Attention 层
// 文件: prefix_cache.cpp

std::vector<Block> PrefixCache::match(
    const Slice<int32_t>& token_ids,
    int32_t layer_id,
    const Slice<Block>& existed_shared_blocks) {
  
  // 如果是 Linear Attention 层，不进行匹配
  if (block_manager_->is_linear_attention_layer(layer_id)) {
    return {};
  }
  
  // 标准 Attention 层，正常匹配
  // ... 原有匹配逻辑 ...
}
```

### 7.2 方案二实现（可选，复杂）

```cpp
// 1. 修改 chunk_gated_delta_rule，输出中间状态
// 文件: qwen3_next_gated_delta_net.cpp

std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor>
torch_chunk_gated_delta_rule_with_states(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    c10::optional<torch::Tensor> initial_state = c10::nullopt) {
    
    std::vector<torch::Tensor> intermediate_states;
    torch::Tensor last_recurrent_state;
    
    // ... 初始化 ...
    
    for (int64_t i = 0; i < num_chunks; ++i) {
        // ... 计算 chunk i ...
        
        // 每个 chunk 结束时保存状态
        intermediate_states.push_back(last_recurrent_state.clone());
    }
    
    return {core_attn_out, intermediate_states, last_recurrent_state};
}

// 2. 修改 forward 函数，存储中间状态
// 文件: qwen3_next_gated_delta_net.cpp

if (attn_metadata.is_prefill) {
    auto [output, intermediate_states, final_state] = 
        torch_chunk_gated_delta_rule_with_states(...);
    
    // 存储每个 chunk 边界的 ssm_state
    // 注意: 需要正确映射 chunk 到 block
    for (size_t i = 0; i < intermediate_states.size(); ++i) {
        int32_t block_id = get_block_id_for_chunk(i, input_params);
        ssm_cache.index_put_({block_id}, intermediate_states[i]);
    }
}
```

---

## 8. 测试计划

### 8.1 功能测试

```bash
# 测试 1: 验证标准 Attention 层的 Prefix Cache
./xllm --model_path=/path/to/qwen3-next \
       --enable_prefix_cache=true \
       --prompt="Hello, how are you?"

# 测试 2: 验证多轮对话
./xllm --model_path=/path/to/qwen3-next \
       --enable_prefix_cache=true \
       --multi_turn=true

# 测试 3: 验证长前缀
./xllm --model_path=/path/to/qwen3-next \
       --enable_prefix_cache=true \
       --prompt="Long system prompt..." \
       --max_seq_len=8192
```

### 8.2 正确性测试

| 测试场景 | 预期结果 |
|---------|---------|
| 无 Prefix Cache | 输出正确 |
| 有 Prefix Cache (方案一) | 输出正确，与无缓存一致 |
| 有 Prefix Cache (方案二) | 输出正确，与无缓存一致 |

### 8.3 性能测试

| 测试场景 | Prefix 长度 | 预期 TTFT 降低 (方案一) |
|---------|------------|------------------------|
| 短前缀 | 256 tokens | 5-10% |
| 中等前缀 | 1024 tokens | 10-15% |
| 长前缀 | 4096 tokens | 15-20% |

---

## 9. 总结

### 9.1 关键结论

```
┌─────────────────────────────────────────────────────────────────┐
│                         关键结论                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Qwen3-Next 模型只能部分支持 Prefix Cache                    │
│     • 标准 Attention 层 (25%): 可以复用 KV Cache                │
│     • Linear Attention 层 (75%): 无法正确工作                   │
│                                                                 │
│  2. 收益有限                                                    │
│     • 只有标准 Attention 层的 KV 计算可以跳过                   │
│     • 整体 TTFT 降低约 10-15%                                   │
│                                                                 │
│  3. 推荐方案                                                    │
│     • 方案一: 禁用 Linear Attention 层的 Prefix Cache           │
│     • 实现简单，风险低，保证正确性                               │
│                                                                 │
│  4. 可选优化                                                    │
│     • 方案二: 完整支持 Linear Attention 的 Prefix Cache         │
│     • 实现复杂，需要存储中间 ssm_state                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 实施建议

| 阶段 | 任务 | 优先级 |
|------|------|--------|
| 阶段一 | 实现方案一，禁用 Linear Attention 的 Prefix Cache | 高 |
| 阶段二 | 功能测试，验证正确性 | 高 |
| 阶段三 | 性能测试，验证收益 | 中 |
| 阶段四 | 可选：实现方案二，完整支持 | 低 |

---

## 10. 参考资料

1. [Prefix Cache 文档](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/docs/zh/features/prefix_cache.md)
2. [Qwen3-Next 模型实现](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/models/llm/qwen3_next.h)
3. [Gated Delta Net 论文](https://arxiv.org/abs/2404.03087)
4. [xLLM Prefix Cache 实现](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp)
5. [Batch Input Builder](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/batch/batch_input_builder.cpp)

# Qwen3-Next 混合架构模型 Prefix Cache 完整支持设计文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v2.0 |
| 日期 | 2026-03-07 |
| 作者 | xLLM Team |
| 状态 | 设计阶段 |

---

## 1. 概述

### 1.1 背景

Qwen3-Next 采用混合 Attention 架构（标准 Attention + Linear Attention），当前架构中所有层共享同一组 blocks，导致无法为不同类型的层分配不同的缓存，从而无法正确支持 Prefix Cache。

### 1.2 目标

设计并实现一个完整的方案，使混合架构模型能够正确支持 Prefix Cache：
- 标准 Attention 层：复用 KV Cache
- Linear Attention 层：使用独立的 blocks，不参与 Prefix Cache 共享

### 1.3 关键挑战

| 挑战 | 描述 |
|------|------|
| 架构限制 | 当前所有层共享同一组 blocks |
| 状态一致性 | Linear Attention 的 ssm_state 无法共享 |
| 内存管理 | 需要为不同层类型管理不同的 block pools |

---

## 2. 当前架构分析

### 2.1 当前架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    当前架构（所有层共享 blocks）                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Sequence                            │   │
│  │                                                         │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              KVCacheState                        │  │   │
│  │   │                                                 │  │   │
│  │   │   blocks_: [Block 0, Block 1, Block 2, ...]     │  │   │
│  │   │        ↓                                        │  │   │
│  │   │   block_table_: [0, 1, 2, 3, 4, 5, ...]        │  │   │
│  │   │                                                 │  │   │
│  │   │   问题: 所有层使用相同的 block_table             │  │   │
│  │   │                                                 │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                        │                                │   │
│  │                        ▼                                │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              Model Forward                       │  │   │
│  │   │                                                 │  │   │
│  │   │   Layer 0 (Linear Attn):                        │  │   │
│  │   │     ssm_cache[block_table] ← 读取状态           │  │   │
│  │   │     问题: 读取到其他 sequence 的状态！           │  │   │
│  │   │                                                 │  │   │
│  │   │   Layer 3 (Standard Attn):                      │  │   │
│  │   │     key_cache[block_table] ← 读取 KV            │  │   │
│  │   │     正确: 可以共享 KV Cache                      │  │   │
│  │   │                                                 │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 问题场景详解

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefix Cache 命中场景分析                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  时间线:                                                        │
│                                                                 │
│  T1: Sequence A 请求 (tokens: [0-99])                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  block_table = [0, 1, 2, 3, 4, 5]                       │   │
│  │                                                         │   │
│  │  Layer 0 (Linear Attn):                                 │   │
│  │    ssm_cache[5] = state_after_token_99                  │   │
│  │    (存储的是 Sequence A 的最终状态)                      │   │
│  │                                                         │   │
│  │  Layer 3 (Standard Attn):                               │   │
│  │    key_cache[0-5] = K[0-99]                             │   │
│  │    value_cache[0-5] = V[0-99]                           │   │
│  │    (存储的是 Sequence A 的 KV)                           │   │
│  │                                                         │   │
│  │  Prefix Cache: 缓存 block_table [0-5]                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  T2: Sequence B 请求 (tokens: [0-99] + [100-109])              │
│      前缀 [0-99] 匹配 Prefix Cache                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  block_table = [0, 1, 2, 3, 4, 5, 6, 7]  (共享 0-5)     │   │
│  │                                                         │   │
│  │  Layer 0 (Linear Attn):                                 │   │
│  │    读取 ssm_cache[5] = Sequence A 的状态 ❌ 错误！       │   │
│  │    应该是: state_after_token_99 for Sequence B          │   │
│  │    但实际: state_after_token_99 for Sequence A          │   │
│  │                                                         │   │
│  │  Layer 3 (Standard Attn):                               │   │
│  │    读取 key_cache[0-5] = K[0-99] ✅ 正确！              │   │
│  │    KV Cache 可以正确共享                                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  结论:                                                          │
│  • 标准 Attention 层: Prefix Cache 正常工作                     │
│  • Linear Attention 层: ssm_state 无法正确共享                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 新架构设计

### 3.1 核心设计思想

**分离 Block Table**：为标准 Attention 层和 Linear Attention 层使用不同的 block table。

```
┌─────────────────────────────────────────────────────────────────┐
│                    新架构（分离 Block Table）                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Sequence                            │   │
│  │                                                         │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              KVCacheState (新设计)               │  │   │
│  │   │                                                 │  │   │
│  │   │   // 标准 Attention 层使用的 blocks              │  │   │
│  │   │   standard_blocks_: [Block 0, Block 1, ...]     │  │   │
│  │   │   standard_block_table_: [0, 1, 2, 3, ...]      │  │   │
│  │   │                                                 │  │   │
│  │   │   // Linear Attention 层使用的 blocks            │  │   │
│  │   │   linear_blocks_: [Block 10, Block 11, ...]     │  │   │
│  │   │   linear_block_table_: [10, 11, 12, ...]        │  │   │
│  │   │                                                 │  │   │
│  │   │   特点:                                         │  │   │
│  │   │   • 标准 Attention 层使用独立的 blocks           │  │   │
│  │   │   • Linear Attention 层使用独立的 blocks         │  │   │
│  │   │   • 两者互不干扰                                 │  │   │
│  │   │                                                 │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                        │                                │   │
│  │                        ▼                                │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              Model Forward                       │  │   │
│  │   │                                                 │  │   │
│  │   │   Layer 0 (Linear Attn):                        │  │   │
│  │   │     ssm_cache[linear_block_table] ← 独立状态    │  │   │
│  │   │     ✅ 每个 sequence 有自己的 ssm_state          │  │   │
│  │   │                                                 │  │   │
│  │   │   Layer 3 (Standard Attn):                      │  │   │
│  │   │     key_cache[standard_block_table] ← 共享 KV   │  │   │
│  │   │     ✅ 可以正确共享 KV Cache                     │  │   │
│  │   │                                                 │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 新架构下的 Prefix Cache 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    新架构 Prefix Cache 流程                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T1: Sequence A 请求 (tokens: [0-99])                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  standard_block_table = [0, 1, 2, 3, 4, 5]              │   │
│  │  linear_block_table = [100, 101, 102, 103, 104, 105]    │   │
│  │                                                         │   │
│  │  Layer 0-2 (Linear Attn):                               │   │
│  │    ssm_cache[100-105] = states for Sequence A           │   │
│  │                                                         │   │
│  │  Layer 3 (Standard Attn):                               │   │
│  │    key_cache[0-5] = K[0-99] for Sequence A              │   │
│  │    value_cache[0-5] = V[0-99] for Sequence A            │   │
│  │                                                         │   │
│  │  Prefix Cache: 缓存 standard_block_table [0-5]          │   │
│  │              不缓存 linear_block_table                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  T2: Sequence B 请求 (tokens: [0-99] + [100-109])              │
│      前缀 [0-99] 匹配 Prefix Cache                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  standard_block_table = [0, 1, 2, 3, 4, 5, 6, 7]        │   │
│  │                         └── 共享 ──┘  └─ 新分配 ─┘       │   │
│  │                                                         │   │
│  │  linear_block_table = [200, 201, 202, 203, 204, 205,    │   │
│  │                        206, 207]                        │   │
│  │                    └── 完全独立分配 ──┘                  │   │
│  │                                                         │   │
│  │  Layer 0-2 (Linear Attn):                               │   │
│  │    ssm_cache[200-207] = states for Sequence B           │   │
│  │    ✅ 独立的状态，不受 Sequence A 影响                   │   │
│  │                                                         │   │
│  │  Layer 3 (Standard Attn):                               │   │
│  │    key_cache[0-5] = K[0-99] (共享) ✅                    │   │
│  │    key_cache[6-7] = K[100-109] (新计算)                 │   │
│  │    ✅ 正确复用前缀 KV Cache                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  收益:                                                          │
│  • 标准 Attention 层: 复用前缀 KV Cache ✅                      │
│  • Linear Attention 层: 独立状态，保证正确性 ✅                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 详细设计

### 4.1 KVCacheState 重构

```cpp
// 文件: xllm/core/framework/request/sequence_kv_state.h

class KVCacheState {
 public:
  // 获取标准 Attention 层的 block table
  Slice<Block> standard_kv_blocks() const;
  std::vector<int32_t> standard_block_table() const;
  
  // 获取 Linear Attention 层的 block table
  Slice<Block> linear_kv_blocks() const;
  std::vector<int32_t> linear_block_table() const;
  
  // 添加 blocks（区分类型）
  void add_standard_kv_blocks(const std::vector<Block>& new_blocks);
  void add_linear_kv_blocks(const std::vector<Block>& new_blocks);
  
  // 添加共享 blocks（仅用于标准 Attention）
  void add_shared_standard_kv_blocks(std::vector<Block>&& blocks,
                                      size_t current_total_num_tokens);
  
  // 获取共享 blocks 数量
  size_t shared_standard_kv_blocks_num() const;
  
  // 兼容性方法（保持向后兼容）
  Slice<Block> kv_blocks() const;  // 返回 standard_kv_blocks()
  size_t num_kv_blocks() const;    // 返回总数

 private:
  // 标准 Attention 层使用的 blocks
  std::vector<Block> standard_blocks_;
  uint32_t num_shared_standard_blocks_ = 0;
  
  // Linear Attention 层使用的 blocks
  std::vector<Block> linear_blocks_;
  
  // 其他现有字段...
  size_t kv_cache_tokens_num_ = 0;
};
```

### 4.2 BlockManager 接口扩展

```cpp
// 文件: xllm/core/framework/block/block_manager.h

class BlockManager {
 public:
  // 现有方法...
  
  // 新增: 分配标准 Attention 层的共享 blocks
  virtual std::vector<Block> allocate_standard_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) = 0;
  
  // 新增: 分配 Linear Attention 层的独立 blocks
  virtual std::vector<Block> allocate_linear_blocks(
      size_t num_blocks) = 0;
  
  // 新增: 缓存标准 Attention 层的 blocks
  virtual void cache_standard(
      const Slice<int32_t>& token_ids,
      std::vector<Block>& blocks,
      size_t existed_shared_blocks_num = 0) = 0;
};
```

### 4.3 BlockManagerImpl 实现

```cpp
// 文件: xllm/core/framework/block/block_manager_impl.h

class BlockManagerImpl : public BlockManager {
 public:
  // 标准 Attention 层的共享 blocks（使用 Prefix Cache）
  std::vector<Block> allocate_standard_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override {
    
    if (options_.enable_prefix_cache()) {
      AUTO_COUNTER(prefix_cache_latency_seconds_match);
      
      // 对于混合架构模型，只匹配标准 Attention 层
      std::vector<Block> shared_blocks =
          prefix_cache_->match(tokens_ids, existed_shared_blocks);
      
      // 更新引用计数...
      return shared_blocks;
    }
    return {};
  }
  
  // Linear Attention 层的独立 blocks（不使用 Prefix Cache）
  std::vector<Block> allocate_linear_blocks(
      size_t num_blocks) override {
    // 直接分配新 blocks，不参与 Prefix Cache
    return allocate(num_blocks);
  }
  
  // 缓存标准 Attention 层的 blocks
  void cache_standard(
      const Slice<int32_t>& token_ids,
      std::vector<Block>& blocks,
      size_t existed_shared_blocks_num = 0) override {
    
    if (options_.enable_prefix_cache()) {
      AUTO_COUNTER(prefix_cache_latency_seconds_insert);
      prefix_cache_->insert(token_ids, blocks, existed_shared_blocks_num);
    }
  }
};
```

### 4.4 BlockManagerPool 扩展

```cpp
// 文件: xllm/core/framework/block/block_manager_pool.h

class BlockManagerPool : public KVCacheManager {
 public:
  // 新增: 分配标准 Attention 层的共享 blocks
  void allocate_standard_shared(Sequence* sequence) override {
    if (options_.enable_prefix_cache()) {
      int32_t dp_rank = get_dp_rank(sequence);
      
      const auto& existed_shared_blocks = 
          sequence->kv_state().standard_kv_blocks().slice(
              0, sequence->kv_state().shared_standard_kv_blocks_num());
      
      std::vector<Block> shared_blocks =
          block_managers_[dp_rank]->allocate_standard_shared(
              sequence->tokens(), existed_shared_blocks);
      
      sequence->add_shared_standard_kv_blocks(std::move(shared_blocks));
    }
  }
  
  // 新增: 分配 Linear Attention 层的独立 blocks
  void allocate_linear_blocks(Sequence* sequence, size_t num_blocks) override {
    int32_t dp_rank = get_dp_rank(sequence);
    
    std::vector<Block> blocks =
        block_managers_[dp_rank]->allocate_linear_blocks(num_blocks);
    
    sequence->kv_state().add_linear_kv_blocks(blocks);
  }
  
  // 新增: 缓存标准 Attention 层的 blocks
  void cache_standard(Sequence* sequence) override {
    int32_t dp_rank = get_dp_rank(sequence);
    const auto token_ids = sequence->cached_tokens();
    auto* blocks = sequence->kv_state().mutable_standard_kv_blocks();
    auto existed_shared_blocks_num = 
        sequence->kv_state().shared_standard_kv_blocks_num();
    
    block_managers_[dp_rank]->cache_standard(
        token_ids, *blocks, existed_shared_blocks_num);
  }
};
```

### 4.5 BatchInputBuilder 修改

```cpp
// 文件: xllm/core/framework/batch/batch_input_builder.cpp

void BatchInputBuilder::setup_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  
  BuilderState& state = state_ptr ? *state_ptr : state_;
  
  // 获取模型参数
  const auto& model_args = args_;
  bool is_hybrid = model_args && model_args->is_hybrid_attention_model();
  
  // 构建标准 Attention 层的 block table
  auto standard_blocks = sequence->kv_state().standard_kv_blocks();
  std::vector<int32_t> standard_block_ids;
  for (const auto& block : standard_blocks) {
    standard_block_ids.push_back(block.id());
  }
  
  // 构建完整 block table（用于标准 Attention 层）
  state.standard_block_tables_vec.emplace_back(std::move(standard_block_ids));
  
  if (is_hybrid) {
    // 构建线性 Attention 层的 block table
    auto linear_blocks = sequence->kv_state().linear_kv_blocks();
    std::vector<int32_t> linear_block_ids;
    for (const auto& block : linear_blocks) {
      linear_block_ids.push_back(block.id());
    }
    state.linear_block_tables_vec.emplace_back(std::move(linear_block_ids));
  }
  
  // 保持向后兼容
  state.block_tables_vec = state.standard_block_tables_vec;
}
```

### 4.6 ModelInputParams 扩展

```cpp
// 文件: xllm/core/framework/model/model_input_params.h

struct ModelInputParams {
  // 现有字段...
  
  // 新增: 标准 Attention 层的 block tables
  std::vector<std::vector<int32_t>> standard_block_tables;
  
  // 新增: Linear Attention 层的 block tables
  std::vector<std::vector<int32_t>> linear_block_tables;
  
  // 获取指定层类型的 block table
  const std::vector<int32_t>& get_block_table(int64_t layer_id,
                                               const ModelArgs& args) const {
    if (args.is_linear_attention_layer(layer_id)) {
      return linear_block_tables[0];  // Linear Attention 层
    } else {
      return standard_block_tables[0];  // 标准 Attention 层
    }
  }
};
```

### 4.7 模型层修改

```cpp
// 文件: xllm/core/layers/common/qwen3_next_gated_delta_net.cpp

torch::Tensor Qwen3NextGatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  
  // 获取 Linear Attention 层专用的 block table
  auto linear_block_table = input_params.get_linear_block_table(layer_id_);
  
  // 使用 linear_block_table 而不是通用的 block_table
  if (attn_metadata.is_prefill) {
    // ...
    ssm_cache.index_put_({linear_block_table}, last_recurrent_state);
  } else {
    auto ssm_state = torch::index_select(ssm_cache, 0, linear_block_table);
    // ...
  }
  
  // ...
}
```

```cpp
// 文件: xllm/core/layers/common/qwen3_next_attention.cpp

torch::Tensor Qwen3NextAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  
  // 标准 Attention 层使用标准的 block table
  // 这与现有实现相同，因为 block_table 已经是共享的
  
  // ...
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  // ...
}
```

### 4.8 Scheduler 修改

```cpp
// 文件: xllm/core/scheduler/chunked_prefill_scheduler.cpp

void ChunkedPrefillScheduler::allocate_blocks_for_hybrid_model(
    Sequence* sequence) {
  
  const auto& model_args = model_context_.get_model_args();
  
  // 1. 为标准 Attention 层分配共享 blocks（使用 Prefix Cache）
  kv_cache_manager_->allocate_standard_shared(sequence);
  
  // 2. 为 Linear Attention 层分配独立 blocks
  size_t num_tokens = sequence->num_tokens();
  size_t block_size = kv_cache_manager_->block_size();
  size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  
  // 减去已分配的标准 Attention blocks
  size_t standard_blocks = sequence->kv_state().standard_kv_blocks().size();
  size_t linear_blocks_needed = num_blocks_needed - standard_blocks;
  
  if (linear_blocks_needed > 0) {
    kv_cache_manager_->allocate_linear_blocks(sequence, linear_blocks_needed);
  }
}

void ChunkedPrefillScheduler::cache_blocks_for_hybrid_model(
    Sequence* sequence) {
  
  // 只缓存标准 Attention 层的 blocks
  kv_cache_manager_->cache_standard(sequence);
  
  // Linear Attention 层的 blocks 不缓存
}
```

---

## 5. 内存布局设计

### 5.1 Block Pool 分配

```
┌─────────────────────────────────────────────────────────────────┐
│                    Block Pool 内存布局                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  总 Block Pool:                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  [Block 0-999]     标准 Attention 层使用                │   │
│  │  ├─ 可共享 (Prefix Cache)                               │   │
│  │  ├─ 引用计数管理                                        │   │
│  │  └─ LRU 淘汰策略                                        │   │
│  │                                                         │   │
│  │  [Block 1000-1999] Linear Attention 层使用              │   │
│  │  ├─ 不可共享                                            │   │
│  │  ├─ 每个 sequence 独立                                  │   │
│  │  └─ 随 sequence 释放                                    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  内存使用估算 (假设 48 层, block_size=16):                       │
│                                                                 │
│  标准 Attention 层 (12层):                                      │
│  • key_cache: [n_blocks, n_kv_heads, block_size, head_dim]     │
│  • value_cache: [n_blocks, n_kv_heads, block_size, head_dim]   │
│                                                                 │
│  Linear Attention 层 (36层):                                    │
│  • conv_cache: [n_blocks, channels, kernel_dim-1]              │
│  • ssm_cache: [n_blocks, n_v_heads, k_dim, k_dim]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Sequence 内存结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sequence 内存结构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sequence A (前缀匹配):                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  standard_blocks_: [Block 0, 1, 2, 3, 4, 5]             │   │
│  │                    └── 共享 (ref_count=2) ──┘            │   │
│  │                                                         │   │
│  │  linear_blocks_: [Block 1000, 1001, 1002, 1003, ...]    │   │
│  │                   └── 独立 (ref_count=1) ──┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Sequence B (前缀匹配):                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  standard_blocks_: [Block 0, 1, 2, 3, 4, 5, 6, 7]       │   │
│  │                    └── 共享 ──┘ └─ 新分配 ─┘             │   │
│  │                                                         │   │
│  │  linear_blocks_: [Block 2000, 2001, 2002, 2003, ...]    │   │
│  │                   └── 完全独立 ──┘                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  内存节省:                                                       │
│  • 标准 Attention 层: 共享 blocks，节省内存                      │
│  • Linear Attention 层: 独立 blocks，保证正确性                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 4+1 视图设计

### 6.1 逻辑视图

```
┌─────────────────────────────────────────────────────────────────┐
│                         逻辑视图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    KVCacheState                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - standard_blocks_: vector<Block>                        │   │
│  │ - linear_blocks_: vector<Block>                          │   │
│  │ - num_shared_standard_blocks_: uint32_t                  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + standard_kv_blocks(): Slice<Block>                     │   │
│  │ + linear_kv_blocks(): Slice<Block>                       │   │
│  │ + add_standard_kv_blocks(blocks): void                   │   │
│  │ + add_linear_kv_blocks(blocks): void                     │   │
│  │ + add_shared_standard_kv_blocks(blocks): void            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             △                                   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    BlockManager                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + allocate_standard_shared(tokens): Blocks               │   │
│  │ + allocate_linear_blocks(num_blocks): Blocks             │   │
│  │ + cache_standard(tokens, blocks): void                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             △                                   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    PrefixCache                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + match(tokens): Blocks                                  │   │
│  │ + insert(tokens, blocks): size_t                         │   │
│  │ + evict(n_blocks): size_t                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 开发视图

```
┌─────────────────────────────────────────────────────────────────┐
│                         开发视图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  xllm/                                                          │
│  ├── core/                                                      │
│  │   ├── framework/                                             │
│  │   │   ├── block/                                             │
│  │   │   │   ├── block_manager.h          【修改】              │
│  │   │   │   ├── block_manager_impl.h     【修改】              │
│  │   │   │   ├── block_manager_impl.cpp   【修改】              │
│  │   │   │   └── block_manager_pool.h     【修改】              │
│  │   │   │                                                      │
│  │   │   ├── request/                                           │
│  │   │   │   └── sequence_kv_state.h      【修改】              │
│  │   │   │                                                      │
│  │   │   ├── batch/                                             │
│  │   │   │   └── batch_input_builder.cpp  【修改】              │
│  │   │   │                                                      │
│  │   │   └── model/                                             │
│  │   │       └── model_input_params.h     【修改】              │
│  │   │                                                          │
│  │   ├── layers/                                                │
│  │   │   └── common/                                            │
│  │   │       ├── qwen3_next_attention.cpp  【修改】             │
│  │   │       └── qwen3_next_gated_delta_net.cpp【修改】         │
│  │   │                                                          │
│  │   └── scheduler/                                             │
│  │       └── chunked_prefill_scheduler.cpp【修改】              │
│  │                                                              │
│  └── models/                                                    │
│      └── llm/                                                   │
│          └── qwen3_next.h                 【修改】              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 过程视图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefill 阶段时序图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scheduler     BlockManager     PrefixCache     Sequence        │
│     │              │               │               │            │
│     │              │               │               │            │
│     │ allocate_blocks_for_hybrid() │               │            │
│     │──────────────────────────────────────────────▶            │
│     │              │               │               │            │
│     │              │               │               │            │
│     │              │ allocate_standard_shared()    │            │
│     │              │──────────────▶│               │            │
│     │              │               │               │            │
│     │              │               │ match(tokens) │            │
│     │              │               │──────────────▶│            │
│     │              │               │               │            │
│     │              │               │◀──────────────│            │
│     │              │               │ shared_blocks │            │
│     │              │               │               │            │
│     │              │◀──────────────│               │            │
│     │              │ shared_blocks │               │            │
│     │              │               │               │            │
│     │              │               │               │            │
│     │              │ allocate_linear_blocks()      │            │
│     │              │──────────────────────────────▶│            │
│     │              │               │               │            │
│     │              │◀──────────────────────────────│            │
│     │              │ linear_blocks │               │            │
│     │              │               │               │            │
│     │◀─────────────│               │               │            │
│     │              │               │               │            │
│     │              │               │               │            │
│     │ prefill 完成 │               │               │            │
│     │              │               │               │            │
│     │ cache_standard()             │               │            │
│     │──────────────▶               │               │            │
│     │              │               │               │            │
│     │              │ insert(tokens, blocks)        │            │
│     │              │──────────────▶│               │            │
│     │              │               │               │            │
│     │              │◀──────────────│               │            │
│     │              │               │               │            │
│     │◀─────────────│               │               │            │
│     │              │               │               │            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 物理视图

```
┌─────────────────────────────────────────────────────────────────┐
│                         物理视图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      GPU Device                          │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              KV Cache Memory                     │   │   │
│  │  │                                                  │   │   │
│  │  │  [标准 Attention 区域]                           │   │   │
│  │  │  key_cache: [blocks, heads, block_size, dim]    │   │   │
│  │  │  value_cache: [blocks, heads, block_size, dim]  │   │   │
│  │  │  ├─ Block 0-5: Sequence A & B 共享              │   │   │
│  │  │  └─ Block 6-7: Sequence B 独占                  │   │   │
│  │  │                                                  │   │   │
│  │  │  [Linear Attention 区域]                         │   │   │
│  │  │  conv_cache: [blocks, channels, kernel-1]       │   │   │
│  │  │  ssm_cache: [blocks, v_heads, k_dim, k_dim]     │   │   │
│  │  │  ├─ Block 1000-1005: Sequence A 独占            │   │   │
│  │  │  └─ Block 2000-2005: Sequence B 独占            │   │   │
│  │  │                                                  │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      CPU Memory                          │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │           Prefix Cache Metadata                  │   │   │
│  │  │                                                  │   │   │
│  │  │  • Hash Table: token_hash → Block*              │   │   │
│  │  │  • LRU List: Block eviction order               │   │   │
│  │  │  • 只管理标准 Attention 层的 blocks              │   │   │
│  │  │                                                  │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5 场景视图

```
┌─────────────────────────────────────────────────────────────────┐
│                    场景：多轮对话                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  第一轮对话:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tokens: [System Prompt 1000] + [Query 100]             │   │
│  │                                                         │   │
│  │  分配:                                                   │   │
│  │  • standard_blocks: [0-68] (1000/16 + 100/16 ≈ 69)      │   │
│  │  • linear_blocks: [1000-1068]                           │   │
│  │                                                         │   │
│  │  缓存:                                                   │   │
│  │  • Prefix Cache: 缓存 standard_blocks [0-62]            │   │
│  │  • Linear blocks 不缓存                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  第二轮对话:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tokens: [System Prompt 1000] + [Query1 100]            │   │
│  │         + [Response1 200] + [Query2 100]                │   │
│  │                                                         │   │
│  │  Prefix Cache 匹配:                                      │   │
│  │  • 标准 Attention: 命中 [0-62] (System Prompt)           │   │
│  │  • 节省: 1000 tokens 的 KV 计算                          │   │
│  │                                                         │   │
│  │  分配:                                                   │   │
│  │  • standard_blocks: [0-62] (共享) + [69-88] (新分配)     │   │
│  │  • linear_blocks: [2000-2088] (完全独立)                │   │
│  │                                                         │   │
│  │  收益:                                                   │   │
│  │  • 标准 Attention 层: 跳过 1000 tokens 的 KV 计算        │   │
│  │  • Linear Attention 层: 正确计算，状态独立               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 收益分析

### 7.1 性能收益

```
┌─────────────────────────────────────────────────────────────────┐
│                    性能收益分析                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  假设参数:                                                       │
│  • Prefix 长度: 1000 tokens                                     │
│  • 新请求长度: 100 tokens                                        │
│  • 标准 Attention 层数: 12 (25%)                                 │
│  • Linear Attention 层数: 36 (75%)                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    无 Prefix Cache                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  TTFT = 全部 1100 tokens 的计算                          │   │
│  │       ≈ 500-600 ms                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              有 Prefix Cache (新方案)                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  标准 Attention 层 (12层):                               │   │
│  │    • 跳过: 1000 tokens 的 KV 计算                        │   │
│  │    • 节省: ~120 ms                                       │   │
│  │                                                         │   │
│  │  Linear Attention 层 (36层):                             │   │
│  │    • 无节省 (需要完整计算 ssm_state)                     │   │
│  │                                                         │   │
│  │  总节省: ~120 ms                                          │   │
│  │  TTFT 降低: ~20-25%                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  与当前方案对比:                                                 │
│  • 当前方案 (完全禁用): 0% 收益                                  │
│  • 新方案 (分离 blocks): ~20-25% 收益                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 内存收益

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存收益分析                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  标准 Attention 层内存节省:                                      │
│  • 公共前缀的 blocks 可以共享                                    │
│  • 假设 10 个并发请求共享相同前缀                                 │
│  • 节省: 9 × 前缀 blocks 的内存                                  │
│                                                                 │
│  Linear Attention 层内存开销:                                    │
│  • 每个 sequence 独立分配 blocks                                 │
│  • 无内存节省，但保证正确性                                       │
│                                                                 │
│  总体:                                                          │
│  • 标准 Attention 层: 内存节省 ✅                                │
│  • Linear Attention 层: 内存不变                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 实施计划

### 8.1 阶段划分

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|---------|------|
| 阶段一 | KVCacheState 重构，支持分离 blocks | 2-3 天 | ✅ 完成 |
| 阶段二 | BlockManager 接口扩展和实现 | 2-3 天 | ✅ 完成 |
| 阶段三 | BatchInputBuilder 和 ModelInputParams 修改 | 2-3 天 | ✅ 完成 |
| 阶段四 | 模型层适配 | 1-2 天 | ✅ 完成 |
| 阶段五 | Scheduler 修改 | 1-2 天 | ✅ 完成 |
| 阶段六 | 测试和验证 | 2-3 天 | 待进行 |

### 8.2 已修改文件列表

| 文件 | 修改内容 |
|------|---------|
| `sequence_kv_state.h` | 添加 `standard_blocks_`、`linear_blocks_` 和相关方法 |
| `sequence_kv_state.cpp` | 实现混合架构模型支持的新方法 |
| `block_manager.h` | 添加 `allocate_standard_shared()`、`allocate_linear_blocks()`、`cache_standard()` |
| `block_manager_impl.h` | 声明新方法 |
| `block_manager_impl.cpp` | 实现混合架构模型的特殊处理 |
| `kv_cache_manager.h` | 添加混合架构模型支持的虚方法 |
| `block_manager_pool.h` | 声明新方法 |
| `block_manager_pool.cpp` | 实现混合架构模型的分配和缓存方法 |
| `model_input_params.h` | 添加 `linear_block_tables` 和 `is_hybrid_attention_model` |
| `batch_input_builder.h` | 添加 `linear_block_tables_vec` 和 `is_hybrid_attention_model` |
| `batch_input_builder.cpp` | 构建混合架构模型的 block_tables |
| `chunked_prefill_scheduler.h` | 声明 `allocate_hybrid_blocks_for()` 和 `cache_hybrid_blocks_for()` |
| `chunked_prefill_scheduler.cpp` | 实现混合架构模型的分配逻辑 |
| `qwen3_next_gated_delta_net.cpp` | 使用 `linear_block_tables` |

### 8.3 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 向后兼容性 | 高 | 保持现有接口，新增方法 |
| 内存管理复杂度 | 中 | 充分测试，添加内存监控 |
| 性能回归 | 中 | 性能基准测试 |
| 正确性验证 | 高 | 添加单元测试和集成测试 |

---

## 9. 测试计划

### 9.1 单元测试

| 测试项 | 测试内容 |
|--------|---------|
| KVCacheState | 分离 blocks 的添加和访问 |
| BlockManager | allocate_standard_shared 和 allocate_linear_blocks |
| PrefixCache | 只缓存标准 Attention blocks |

### 9.2 集成测试

| 测试场景 | 预期结果 |
|---------|---------|
| 首次请求 | 正常计算，缓存标准 Attention blocks |
| 前缀匹配 | 标准 Attention 复用，Linear Attention 独立 |
| 多轮对话 | 历史 KV 正确复用 |
| 正确性验证 | 输出与无缓存一致 |

### 9.3 性能测试

| 测试场景 | 预期收益 |
|---------|---------|
| 短前缀 (256 tokens) | 10-15% TTFT 降低 |
| 中等前缀 (1024 tokens) | 20-25% TTFT 降低 |
| 长前缀 (4096 tokens) | 25-30% TTFT 降低 |

---

## 10. 总结

本设计文档详细描述了如何为混合架构模型（如 Qwen3-Next）实现完整的 Prefix Cache 支持。核心思想是：

1. **分离 Block Table**：为标准 Attention 层和 Linear Attention 层使用不同的 blocks
2. **选择性缓存**：只缓存标准 Attention 层的 blocks
3. **独立状态**：Linear Attention 层使用独立 blocks，保证正确性

该方案可以在保证正确性的前提下，为混合架构模型提供 Prefix Cache 优化，预计可实现 20-30% 的 TTFT 降低。

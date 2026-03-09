# xLLM框架Qwen-Next Prefix Cache适配设计文档

## 1. 概述

### 1.1 文档目的

本文档描述xLLM框架为Qwen-Next模型适配Prefix Cache特性的设计方案。Qwen-Next是一种混合架构模型，结合了标准Transformer注意力和线性注意力（Gated Delta Net），其Prefix Cache实现与传统LLM模型存在显著差异。

### 1.2 参考资料

- vLLM实现: [PR #30877](https://github.com/vllm-project/vllm/pull/30877)
- Qwen-Next模型架构: Gated Delta Net线性注意力 + 标准注意力混合架构

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| Prefix Cache | 前缀缓存，复用已计算KV Cache的技术 |
| Mamba Cache Mode | 线性注意力层的状态缓存模式 |
| Align Mode | 对齐模式，仅在block边界缓存状态 |
| All Mode | 全量模式，缓存所有block边界位置的状态 |
| GDN | Gated Delta Net，Qwen-Next使用的线性注意力变体 |
| Conv Cache | 卷积状态缓存，GDN的1D卷积中间状态 |
| SSM Cache | 状态空间模型缓存，GDN的递归状态 |

---

## 2. 背景分析

### 2.1 Qwen-Next模型架构特点

Qwen-Next采用混合注意力架构：

```
┌─────────────────────────────────────────────────────────┐
│                    Qwen-Next Layer                      │
├─────────────────────────────────────────────────────────┤
│  Layer 0, 1, 2:     Linear Attention (GDN)              │
│  Layer 3:           Standard Attention                  │
│  Layer 4, 5, 6:     Linear Attention (GDN)              │
│  Layer 7:           Standard Attention                  │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

**关键差异点**：

| 特性 | 标准LLM | Qwen-Next |
|------|---------|-----------|
| 注意力类型 | 纯标准注意力 | 混合（标准+线性） |
| KV Cache | key, value | key, value, conv_state, ssm_state |
| 状态依赖 | 无状态 | 有递归状态依赖 |
| Prefix Cache | 直接复用 | 需要状态对齐 |

### 2.2 线性注意力的状态依赖

Gated Delta Net的核心递归公式：

```
h_t = h_{t-1} * g_t + k_t * (v_t - h_{t-1} * k_t) * beta_t
```

这意味着：
1. **状态连续性**：当前状态依赖于历史所有输入
2. **无法直接复用**：传统KV Cache无法直接应用于线性注意力
3. **需要对齐缓存**：必须在特定位置缓存完整状态

### 2.3 vLLM的Mamba Cache Mode设计

vLLM引入了三种缓存模式：

```python
MambaCacheMode = Literal["all", "align", "none"]
```

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `none` | 不缓存线性注意力状态 | Prefix Cache禁用 |
| `all` | 缓存所有block边界位置的状态 | 支持状态快照的模型 |
| `align` | 仅在调度结束时缓存对齐位置的状态 | Qwen-Next等模型 |

**Qwen-Next限制**：不支持`all`模式，只能使用`align`模式。

---

## 3. 架构设计（4+1视图）

### 3.1 逻辑视图（Logical View）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           xLLM Framework                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Scheduler Layer                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │ Request Scheduler│  │ Block Allocator │  │ Token Aligner   │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │   │
│  └───────────┼─────────────────────┼─────────────────────┼──────────┘   │
│              │                     │                     │              │
│              ▼                     ▼                     ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Cache Management Layer                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │  PrefixCache    │  │  BlockManager   │  │ MambaCacheMgr   │  │   │
│  │  │  (Token-based)  │  │  (Block-based)  │  │ (State-based)   │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │   │
│  └───────────┼─────────────────────┼─────────────────────┼──────────┘   │
│              │                     │                     │              │
│              ▼                     ▼                     ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       Model Layer                                │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              Qwen3NextDecoderLayer                       │    │   │
│  │  │  ┌───────────────────┐  ┌───────────────────────────┐   │    │   │
│  │  │  │ Qwen3NextAttention│  │ Qwen3NextGatedDeltaNet    │   │    │   │
│  │  │  │ (Standard Attn)   │  │ (Linear Attn)             │   │    │   │
│  │  │  │ - k,v cache       │  │ - conv_cache, ssm_cache   │   │    │   │
│  │  │  └───────────────────┘  └───────────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 进程视图（Process View）

#### 3.2.1 Prefill阶段流程

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│ Scheduler│────▶│  Model   │────▶│  Cache   │
└──────────┘     └──────────┘     │ Executor │     │ Manager  │
                      │           └──────────┘     └──────────┘
                      │                 │               │
                      ▼                 ▼               ▼
                 ┌─────────────────────────────────────────────┐
                 │  1. Token Alignment Check                   │
                 │     - 检查token数是否为block_size的倍数      │
                 │     - 如果不是，调整调度策略                  │
                 │                                              │
                 │  2. Prefix Match                             │
                 │     - 标准注意力层：匹配KV Cache blocks       │
                 │     - 线性注意力层：匹配Mamba State blocks    │
                 │                                              │
                 │  3. State Computation                        │
                 │     - 计算未命中部分的forward pass            │
                 │     - 在block边界保存状态                     │
                 │                                              │
                 │  4. Cache Insert                             │
                 │     - 插入KV Cache到PrefixCache              │
                 │     - 插入Mamba State到MambaCacheManager     │
                 └─────────────────────────────────────────────┘
```

#### 3.2.2 Decode阶段流程

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│ Scheduler│────▶│  Model   │────▶│  Cache   │
└──────────┘     └──────────┘     │ Executor │     │ Manager  │
                      │           └──────────┘     └──────────┘
                      │                 │               │
                      ▼                 ▼               ▼
                 ┌─────────────────────────────────────────────┐
                 │  1. State Load                              │
                 │     - 加载最近block边界的状态                 │
                 │     - 如果有prefix命中，加载缓存状态          │
                 │                                              │
                 │  2. Incremental Computation                 │
                 │     - 从缓存状态继续forward                  │
                 │     - 更新conv_cache和ssm_cache             │
                 │                                              │
                 │  3. State Update                            │
                 │     - 更新当前block的状态缓存                 │
                 │     - 如果到达新block边界，创建新缓存条目     │
                 └─────────────────────────────────────────────┘
```

### 3.3 开发视图（Development View）

```
xllm/
├── core/
│   ├── framework/
│   │   ├── prefix_cache/
│   │   │   ├── prefix_cache.h              # 现有：Token-based prefix cache
│   │   │   ├── prefix_cache.cpp
│   │   │   ├── mamba_cache_manager.h       # 新增：Mamba状态缓存管理
│   │   │   └── mamba_cache_manager.cpp
│   │   ├── block/
│   │   │   ├── block_manager.h             # 现有：Block管理接口
│   │   │   ├── block_manager_impl.h        # 现有：Block管理实现
│   │   │   ├── hybrid_block_manager.h      # 新增：混合架构Block管理
│   │   │   └── hybrid_block_manager.cpp
│   │   ├── kv_cache/
│   │   │   ├── kv_cache.h                  # 现有：KV Cache数据结构
│   │   │   └── mamba_state.h               # 新增：Mamba状态定义
│   │   └── model/
│   │       ├── model_args.h                # 修改：添加mamba_cache_mode参数
│   │       └── model_context.h             # 修改：添加缓存模式上下文
│   │
│   ├── layers/
│   │   ├── common/
│   │   │   ├── qwen3_next_gated_delta_net.h    # 修改：支持状态缓存
│   │   │   └── qwen3_next_gated_delta_net.cpp
│   │   └── qwen3_next_decoder_layer.cpp        # 修改：层类型判断
│   │
│   └── scheduler/
│       ├── scheduler.h                       # 修改：添加对齐调度逻辑
│       └── token_aligner.h                   # 新增：Token对齐工具
│
├── models/
│   └── llm/
│       └── qwen3_next.h                      # 修改：添加缓存模式配置
│
└── utils/
    └── mamba_state_copy.h                    # 新增：状态复制工具函数
```

### 3.4 物理视图（Physical View）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GPU Memory                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    KV Cache Memory Pool                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │  Key Cache      │  │  Value Cache    │  │  Padding Block  │  │   │
│  │  │  [num_blocks,   │  │  [num_blocks,   │  │  (Block 0)      │  │   │
│  │  │   num_layers,   │  │   num_layers,   │  │                 │  │   │
│  │  │   block_size,   │  │   block_size,   │  │                 │  │   │
│  │  │   num_heads,    │  │   num_heads,    │  │                 │  │   │
│  │  │   head_dim]     │  │   head_dim]     │  │                 │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Mamba State Memory Pool (新增)                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Conv State Cache                                        │    │   │
│  │  │  [num_blocks, num_layers, conv_dim, conv_kernel_size]   │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  SSM State Cache                                         │    │   │
│  │  │  [num_blocks, num_layers, num_heads, k_dim, v_dim]      │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                              CPU Memory                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Prefix Cache Index                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Hash Map: token_hash -> block_id                        │    │   │
│  │  │  LRU List: for eviction management                       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Mamba Cache Index (新增)                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Hash Map: (token_hash, layer_id) -> state_block_id      │    │   │
│  │  │  State Block Table: mapping request to state blocks      │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 场景视图（Scenario View）

#### 3.5.1 场景一：Prefill with Prefix Cache Hit

```
时序图:

Client          Scheduler        ModelRunner       GDN Layer        Cache Manager
   │                │                 │                │                 │
   │───Request────▶│                 │                │                 │
   │   (tokens)     │                 │                │                 │
   │                │                 │                │                 │
   │                │──Match Prefix──▶│                │                 │
   │                │                 │─────────────────────────────────▶│
   │                │                 │                │    match()      │
   │                │                 │◀─────────────────────────────────│
   │                │                 │  matched_blocks │                 │
   │                │                 │                │                 │
   │                │                 │──Load States──▶│                 │
   │                │                 │                │──Get SSM State─▶│
   │                │                 │                │◀────────────────│
   │                │                 │                │  cached_state   │
   │                │                 │                │                 │
   │                │                 │──Forward──────▶│                 │
   │                │                 │  (new tokens)  │                 │
   │                │                 │                │──Compute───────▶│
   │                │                 │                │  with cached    │
   │                │                 │                │  state          │
   │                │                 │◀───────────────│                 │
   │                │                 │  output        │                 │
   │                │                 │                │                 │
   │                │                 │──Cache States──────────────────▶│
   │                │                 │                │    insert()     │
   │                │                 │                │                 │
   │◀───Response────│                 │                │                 │
   │   (output)      │                 │                │                 │
```

#### 3.5.2 场景二：Align Mode State Copy

```
状态复制场景 (Speculative Decoding后):

Block Table:
┌─────────────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │
│ (cached)│ (cached)│ (cached)│ (temp)  │ (temp)  │ (new)   │
└─────────────────────────────────────────────────────────────┘
     │         │         │         │         │         │
     │         │         │         │         │         │
     ▼         ▼         ▼         ▼         ▼         ▼
State:    State:    State:    State:    State:    State:
cached    cached    cached    temp      temp      new
at 0      at 560    at 1120   at 1680   at 1680+  at 2240
                              +tokens   +tokens

当accepted_tokens=2时:
1. 从Block 2 (src_block_idx=2) 复制状态到 Block 3 (dest_block_idx=3)
2. 继续从Block 3的状态计算剩余tokens
```

---

## 4. 详细设计

### 4.1 MambaCacheMode配置

#### 4.1.1 配置参数定义

```cpp
// model_args.h

enum class MambaCacheMode {
  kNone,    // 不缓存Mamba状态
  kAll,     // 缓存所有block边界位置的状态
  kAlign,   // 仅在调度结束时缓存对齐位置的状态
};

struct ModelArgs {
  // ... 现有参数 ...
  
  // Mamba缓存模式
  PROPERTY(MambaCacheMode, mamba_cache_mode) = MambaCacheMode::kNone;
  
  // Mamba block size (与KV Cache block size对齐)
  PROPERTY(int32_t, mamba_block_size) = 0;
  
  // 是否支持Mamba prefix caching (all模式)
  PROPERTY(bool, supports_mamba_prefix_caching) = false;
};
```

#### 4.1.2 配置验证逻辑

```cpp
// model_config_validator.cpp

void validate_mamba_cache_config(ModelArgs& args, const CacheConfig& cache_config) {
  if (cache_config.enable_prefix_cache) {
    if (args.mamba_cache_mode() == MambaCacheMode::kNone) {
      // 根据模型能力自动选择模式
      args.set_mamba_cache_mode(
          args.supports_mamba_prefix_caching() 
              ? MambaCacheMode::kAll 
              : MambaCacheMode::kAlign);
    }
    
    // Qwen-Next不支持all模式
    if (args.mamba_cache_mode() == MambaCacheMode::kAll && 
        !args.supports_mamba_prefix_caching()) {
      args.set_mamba_cache_mode(MambaCacheMode::kAlign);
      LOG(WARNING) << "Model does not support 'all' mamba cache mode, "
                   << "falling back to 'align' mode";
    }
    
    // Align模式需要chunked prefill
    if (args.mamba_cache_mode() == MambaCacheMode::kAlign) {
      CHECK(args.enable_chunked_prefill()) 
          << "Chunked prefill is required for mamba cache 'align' mode";
    }
    
    // 设置mamba_block_size
    if (args.mamba_block_size() == 0) {
      args.set_mamba_block_size(cache_config.block_size);
    }
  }
}
```

### 4.2 MambaCacheManager设计

#### 4.2.1 类定义

```cpp
// mamba_cache_manager.h

struct MambaStateCopySpec {
  int64_t start_addr;
  int64_t num_elements;
};

using MambaStateCopyFunc = std::function<MambaStateCopySpec(
    torch::Tensor state, 
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens)>;

class MambaCacheManager {
 public:
  explicit MambaCacheManager(
      uint32_t block_size,
      MambaCacheMode cache_mode,
      const std::vector<MambaStateCopyFunc>& copy_funcs);
  
  // 匹配Mamba状态缓存
  std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      int32_t layer_id,
      const Slice<Block>& existed_blocks = {});
  
  // 插入Mamba状态缓存
  void insert(
      const Slice<int32_t>& token_ids,
      int32_t layer_id,
      std::vector<Block>& blocks,
      size_t existed_blocks_num = 0);
  
  // 状态复制 (用于align模式)
  void copy_state(
      torch::Tensor src_state,
      torch::Tensor dst_state,
      const MambaStateCopySpec& spec);
  
  // 获取状态复制函数
  const std::vector<MambaStateCopyFunc>& get_copy_funcs() const {
    return copy_funcs_;
  }
  
  // 驱逐缓存
  size_t evict(size_t n_blocks);
  
 private:
  uint32_t block_size_;
  MambaCacheMode cache_mode_;
  std::vector<MambaStateCopyFunc> copy_funcs_;
  
  // 每层独立的缓存索引
  std::unordered_map<int32_t, std::unique_ptr<PrefixCache>> layer_caches_;
};
```

#### 4.2.2 状态复制函数实现

```cpp
// mamba_state_copy.cpp

// Conv状态复制函数 (用于GDN)
MambaStateCopySpec get_conv_copy_spec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens) {
  int32_t src_block_id = block_ids[cur_block_idx];
  auto src_state = state[src_block_id].slice(0, num_accepted_tokens - 1);
  return MambaStateCopySpec{
    .start_addr = src_state.data_ptr(),
    .num_elements = src_state.numel()
  };
}

// Temporal (SSM) 状态复制函数
MambaStateCopySpec get_temporal_copy_spec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens) {
  int32_t src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1];
  auto src_state = state[src_block_id];
  return MambaStateCopySpec{
    .start_addr = src_state.data_ptr(),
    .num_elements = src_state.numel()
  };
}

// GDN状态复制函数组合
std::vector<MambaStateCopyFunc> get_gdn_copy_funcs() {
  return {get_conv_copy_spec, get_temporal_copy_spec};
}
```

### 4.3 HybridBlockManager设计

#### 4.3.1 类定义

```cpp
// hybrid_block_manager.h

class HybridBlockManager : public BlockManager {
 public:
  struct Options : public BlockManager::Options {
    PROPERTY(MambaCacheMode, mamba_cache_mode) = MambaCacheMode::kNone;
    PROPERTY(bool, has_linear_attention_layers) = false;
  };
  
  explicit HybridBlockManager(const Options& options);
  
  // 分配blocks (考虑混合架构)
  std::vector<Block> allocate(size_t num_blocks) override;
  
  // 分配共享blocks (prefix cache匹配)
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;
  
  // 缓存blocks
  void cache(
      const Slice<int32_t>& token_ids,
      std::vector<Block>& blocks,
      size_t existed_shared_blocks_num = 0) override;
  
  // 获取Mamba缓存管理器
  MambaCacheManager* get_mamba_cache_manager(int32_t layer_id);
  
  // 检查是否需要token对齐
  bool needs_token_alignment() const {
    return options_.mamba_cache_mode() == MambaCacheMode::kAlign;
  }
  
  // 计算对齐后的token数量
  int32_t compute_aligned_tokens(int32_t num_tokens) const;
  
 private:
  Options options_;
  std::unique_ptr<PrefixCache> prefix_cache_;
  std::unordered_map<int32_t, std::unique_ptr<MambaCacheManager>> mamba_cache_managers_;
};
```

### 4.4 Qwen3NextGatedDeltaNet修改

#### 4.4.1 状态缓存支持

```cpp
// qwen3_next_gated_delta_net.cpp

torch::Tensor Qwen3NextGatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  
  auto mamba_cache_mode = input_params.mamba_cache_mode;
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  
  // ... 现有的qkv投影和预处理 ...
  
  if (attn_metadata.is_prefill) {
    // Prefill模式
    if (mamba_cache_mode == MambaCacheMode::kAll) {
      // 在每个block边界保存状态
      return prefill_with_all_cache(
          processed_q, processed_k, processed_v, g, beta,
          conv_cache, ssm_cache, input_params);
    } else if (mamba_cache_mode == MambaCacheMode::kAlign) {
      // 仅在最后保存状态（如果对齐）
      return prefill_with_align_cache(
          processed_q, processed_k, processed_v, g, beta,
          conv_cache, ssm_cache, input_params);
    } else {
      // 无缓存
      return prefill_no_cache(
          processed_q, processed_k, processed_v, g, beta);
    }
  } else {
    // Decode模式
    return decode_with_cache(
        processed_q, processed_k, processed_v, g, beta,
        conv_cache, ssm_cache, input_params);
  }
}

// Align模式的prefill实现
torch::Tensor Qwen3NextGatedDeltaNetImpl::prefill_with_align_cache(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
    torch::Tensor& g, torch::Tensor& beta,
    torch::Tensor& conv_cache, torch::Tensor& ssm_cache,
    const ModelInputParams& input_params) {
  
  int64_t seq_len = q.size(1);
  int64_t block_size = input_params.block_size;
  
  // 使用chunk算法计算
  auto [core_attn_out, last_state] = torch_chunk_gated_delta_rule(
      q, k, v, g, beta);
  
  // 检查是否对齐到block边界
  if (seq_len % block_size == 0) {
    int32_t block_id = input_params.block_tables.select(1, 0).item<int32_t>();
    
    // 保存conv状态
    torch::Tensor conv_state = (seq_len < conv_kernel_size_ - 1)
        ? torch::pad(mixed_qkv, {0, conv_kernel_size_ - 1 - seq_len})
        : mixed_qkv.narrow(-1, seq_len - conv_kernel_size_ + 1, conv_kernel_size_ - 1);
    conv_cache.index_put_({block_id}, conv_state.to(conv_cache.dtype()));
    
    // 保存ssm状态
    ssm_cache.index_put_({block_id}, last_state.to(ssm_cache.dtype()));
  }
  
  return core_attn_out;
}
```

### 4.5 调度器修改

#### 4.5.1 Token对齐逻辑

```cpp
// token_aligner.h

class TokenAligner {
 public:
  explicit TokenAligner(int32_t block_size) 
      : block_size_(block_size) {}
  
  // 计算需要调度的token数量（对齐到block边界）
  int32_t compute_scheduled_tokens(
      int32_t num_tokens,
      int32_t num_computed_tokens,
      bool force_align = false) const {
    
    if (!force_align) {
      return num_tokens;
    }
    
    // 计算下一个block边界
    int32_t next_boundary = 
        ((num_computed_tokens + block_size_ - 1) / block_size_) * block_size_;
    
    // 不超过实际token数
    return std::min(next_boundary, num_tokens);
  }
  
  // 检查是否在block边界
  bool is_at_block_boundary(int32_t token_position) const {
    return token_position % block_size_ == 0;
  }
  
 private:
  int32_t block_size_;
};
```

#### 4.5.2 调度器集成

```cpp
// scheduler.cpp

SchedulerOutput Scheduler::schedule(const std::vector<Request>& requests) {
  // ... 现有调度逻辑 ...
  
  for (auto& request : requests) {
    int32_t num_tokens = request.num_tokens();
    int32_t num_computed = request.num_computed_tokens();
    
    // 对于align模式，需要调整调度token数
    if (needs_token_alignment(request)) {
      int32_t aligned_tokens = token_aligner_.compute_scheduled_tokens(
          num_tokens, num_computed, true);
      
      // 更新调度决策
      scheduled_tokens = aligned_tokens;
      
      // 标记是否需要保存状态
      if (token_aligner_.is_at_block_boundary(num_computed + aligned_tokens)) {
        request.set_needs_state_save(true);
      }
    }
  }
  
  // ... 后续处理 ...
}
```

---

## 5. 接口设计

### 5.1 公共API

#### 5.1.1 配置接口

```cpp
// 创建支持Mamba prefix cache的模型
auto model = create_llm_model(ModelContext{
  .model_args = ModelArgs{
    .model_type = "qwen3_next",
    .mamba_cache_mode = MambaCacheMode::kAlign,
    .mamba_block_size = 560,
    // ... 其他参数
  },
  .cache_config = CacheConfig{
    .enable_prefix_cache = true,
    .block_size = 560,
    // ... 其他参数
  }
});
```

#### 5.1.2 推理接口

```cpp
// 推理请求
struct InferenceRequest {
  std::vector<int32_t> token_ids;
  SamplingParams sampling_params;
  
  // Prefix cache相关
  std::optional<std::vector<Block>> cached_blocks;
  std::optional<MambaCachedStates> cached_mamba_states;
};

// 推理响应
struct InferenceResponse {
  std::vector<int32_t> output_token_ids;
  
  // 缓存信息
  std::vector<Block> new_cached_blocks;
  std::optional<MambaCachedStates> new_mamba_states;
};
```

### 5.2 内部接口

#### 5.2.1 状态复制接口

```cpp
// 执行状态复制
void copy_mamba_states(
    const std::vector<torch::Tensor>& src_states,
    const std::vector<torch::Tensor>& dst_states,
    const std::vector<MambaStateCopySpec>& specs);

// 批量状态复制
void batch_copy_mamba_states(
    const std::vector<MambaCopyRequest>& requests);
```

#### 5.2.2 缓存管理接口

```cpp
// 获取或创建层的Mamba缓存管理器
MambaCacheManager* get_or_create_mamba_cache_manager(
    int32_t layer_id,
    const LayerConfig& layer_config);

// 同步KV Cache和Mamba Cache
void sync_caches(
    const std::vector<Block>& kv_blocks,
    const std::vector<Block>& mamba_blocks);
```

---

## 6. 数据结构设计

### 6.1 MambaCachedStates

```cpp
struct MambaCachedStates {
  // 每层的状态
  std::unordered_map<int32_t, LayerMambaStates> layer_states;
  
  // 对应的token位置
  int32_t cached_token_position;
  
  // 对应的block id
  int32_t cached_block_id;
};

struct LayerMambaStates {
  // Conv状态
  torch::Tensor conv_state;
  
  // SSM状态
  torch::Tensor ssm_state;
  
  // 状态有效性
  bool is_valid;
};
```

### 6.2 MambaCopyRequest

```cpp
struct MambaCopyRequest {
  // 源block索引
  int32_t src_block_idx;
  
  // 目标block索引
  int32_t dst_block_idx;
  
  // 层ID
  int32_t layer_id;
  
  // 接受的token数
  int32_t num_accepted_tokens;
  
  // 状态类型 (conv, ssm)
  enum class StateType { kConv, kSSM };
  StateType state_type;
};
```

---

## 7. 性能考虑

### 7.1 内存开销

| 组件 | 额外内存 | 说明 |
|------|----------|------|
| Conv Cache | num_blocks × num_layers × conv_dim × kernel_size | 每个block一份 |
| SSM Cache | num_blocks × num_layers × num_heads × k_dim × v_dim | 每个block一份 |
| Index Structures | O(num_cached_blocks) | Hash map + LRU list |

### 7.2 计算开销

| 操作 | 开销 | 优化策略 |
|------|------|----------|
| 状态复制 | O(state_size) | 异步复制、批量处理 |
| Hash计算 | O(block_size) | 增量Hash |
| 状态加载 | O(state_size) | 预取、缓存 |

### 7.3 延迟分析

```
Prefill with Cache Hit:
┌─────────────────────────────────────────────────────────┐
│ 阶段                    │ 延迟占比  │ 优化空间          │
├─────────────────────────┼───────────┼───────────────────┤
│ Prefix Match            │ ~1%       │ 已优化            │
│ State Load              │ ~5%       │ 预取优化          │
│ Forward (partial)       │ ~90%      │ 取决于命中率      │
│ State Save              │ ~4%       │ 异步保存          │
└─────────────────────────┴───────────┴───────────────────┘
```

---

## 8. 测试策略

### 8.1 单元测试

```cpp
// 测试状态复制函数
TEST(MambaStateCopyTest, TestConvCopySpec) {
  auto state = torch::randn({10, 128, 4});  // [blocks, dim, kernel]
  auto spec = get_conv_copy_spec(state, {0, 1, 2}, 1, 2);
  EXPECT_EQ(spec.start_addr, state[1].select(0, 1).data_ptr());
}

// 测试Token对齐
TEST(TokenAlignerTest, TestAlignment) {
  TokenAligner aligner(560);
  EXPECT_EQ(aligner.compute_scheduled_tokens(1000, 0, true), 560);
  EXPECT_EQ(aligner.compute_scheduled_tokens(1000, 560, true), 560);
  EXPECT_TRUE(aligner.is_at_block_boundary(560));
  EXPECT_FALSE(aligner.is_at_block_boundary(100));
}
```

### 8.2 集成测试

```cpp
// 测试完整的prefix cache流程
TEST(QwenNextPrefixCacheTest, TestPrefillWithCacheHit) {
  // 1. 创建模型和缓存管理器
  auto model = create_qwen3_next_model(/*enable_prefix_cache=*/true);
  
  // 2. 第一次prefill
  auto request1 = create_request(/*tokens=*/generate_tokens(1000));
  auto response1 = model->generate(request1);
  
  // 3. 第二次prefill (应该命中缓存)
  auto request2 = create_request(/*tokens=*/generate_tokens(1200));  // 前1000个token相同
  auto response2 = model->generate(request2);
  
  // 4. 验证缓存命中
  EXPECT_GT(response2.cache_hit_length, 560);  // 至少命中一个block
}

// 测试align模式
TEST(QwenNextPrefixCacheTest, TestAlignMode) {
  auto model = create_qwen3_next_model(
      /*enable_prefix_cache=*/true,
      /*mamba_cache_mode=*/MambaCacheMode::kAlign);
  
  // 测试不对齐的情况
  auto request = create_request(/*tokens=*/generate_tokens(1000));  // 不是560的倍数
  auto response = model->generate(request);
  
  // 验证状态只在边界保存
  EXPECT_EQ(response.saved_state_position, 560);  // 只保存560位置的状态
}
```

### 8.3 端到端测试

```python
# test_qwen_next_prefix_cache_e2e.py

def test_prefix_cache_accuracy():
    """验证prefix cache不影响输出准确性"""
    model = LLM(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        block_size=560
    )
    
    prompt = "这是一个长文本..." * 100  # 足够长的prompt
    
    # 不使用缓存的输出
    output_no_cache = model.generate(prompt, sampling_params)
    
    # 使用缓存的输出
    model.reset_prefix_cache()
    _ = model.generate(prompt, sampling_params)  # 预热缓存
    output_with_cache = model.generate(prompt, sampling_params)
    
    # 验证输出一致
    assert outputs_equal(output_no_cache, output_with_cache)

def test_prefix_cache_performance():
    """验证prefix cache带来性能提升"""
    model = LLM(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        block_size=560
    )
    
    common_prefix = "共享前缀..." * 50
    
    # 测试多个请求共享前缀
    requests = [
        TokensPrompt(prompt_token_ids=tokenize(common_prefix + f"请求{i}"))
        for i in range(10)
    ]
    
    # 第一次批量请求（填充缓存）
    start_time = time.time()
    outputs1 = model.generate(requests, sampling_params)
    time_no_cache = time.time() - start_time
    
    # 第二次批量请求（应该命中缓存）
    model.reset_prefix_cache()
    start_time = time.time()
    outputs2 = model.generate(requests, sampling_params)
    time_with_cache = time.time() - start_time
    
    # 验证性能提升
    assert time_with_cache < time_no_cache * 0.8  # 至少20%提升
```

---

## 9. 实施计划

### 9.1 阶段一：基础设施（预计2周）

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 添加MambaCacheMode配置 | P0 | 无 |
| 实现MambaCacheManager基础框架 | P0 | 配置 |
| 实现状态复制函数 | P0 | 无 |
| 单元测试 | P0 | 全部 |

### 9.2 阶段二：核心功能（预计3周）

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 修改Qwen3NextGatedDeltaNet | P0 | 阶段一 |
| 实现HybridBlockManager | P0 | 阶段一 |
| 调度器Token对齐逻辑 | P0 | 无 |
| 集成测试 | P0 | 全部 |

### 9.3 阶段三：优化与验证（预计2周）

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 性能优化 | P1 | 阶段二 |
| 端到端测试 | P0 | 阶段二 |
| 文档完善 | P1 | 全部 |
| 代码审查 | P0 | 全部 |

---

## 10. 风险与缓解

### 10.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 状态复制开销过大 | 性能下降 | 中 | 异步复制、批量处理 |
| 内存占用过高 | OOM风险 | 中 | 动态调整block数量 |
| 对齐逻辑复杂 | Bug风险 | 高 | 充分测试、渐进式发布 |

### 10.2 兼容性风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 与现有prefix cache冲突 | 功能异常 | 低 | 独立管理Mamba缓存 |
| 不同平台行为不一致 | 测试困难 | 中 | 平台特定测试用例 |

---

## 11. 附录

### 11.1 vLLM关键代码参考

```python
# vllm/config/cache.py
class CacheConfig:
    mamba_cache_mode: MambaCacheMode = "none"
    """The cache strategy for Mamba layers.
    - "none": set when prefix caching is disabled.
    - "all": cache the mamba state of all tokens at position i * block_size.
    - "align": only cache the mamba state of the last token of each scheduler step
           and when the token is at position i * block_size.
    """

# vllm/model_executor/layers/mamba/mamba_utils.py
@dataclass
class MambaCopySpec:
    start_addr: int
    num_elements: int

def get_conv_copy_spec(state, block_ids, cur_block_idx, num_accepted_tokens):
    src_block_id = block_ids[cur_block_idx]
    src_state = state[src_block_id, num_accepted_tokens - 1:]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), 
        num_elements=src_state.numel()
    )

def get_temporal_copy_spec(state, block_ids, cur_block_idx, num_accepted_tokens):
    src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]
    src_state = state[src_block_id]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), 
        num_elements=src_state.numel()
    )
```

### 11.2 Qwen-Next GDN状态维度

```cpp
// Conv State维度
// [batch_size, num_heads, conv_kernel_size - 1, head_dim]
// 对于Qwen-Next: conv_kernel_size = 4

// SSM State维度  
// [batch_size, num_heads, head_k_dim, head_v_dim]
// 对于Qwen-Next: head_k_dim = 128, head_v_dim = 128
```

### 11.3 Block Table示例

```
Align模式下的Block Table:

Request 1 (tokens: 0-1120, 已计算: 560):
┌─────────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ ...  │
│ (cached)│ (cached)│ (active)│ (empty) │ (empty) │      │
│ pos:0   │ pos:560 │ pos:1120│         │         │      │
└─────────────────────────────────────────────────────────┘

状态分布:
- Block 0: 完整的conv_state和ssm_state (在位置560保存)
- Block 1: 完整的conv_state和ssm_state (在位置1120保存)
- Block 2: 当前活跃block，状态在计算中
```

---

## 12. 总结

本设计文档详细描述了xLLM框架为Qwen-Next模型适配Prefix Cache特性的完整方案。核心要点包括：

1. **Mamba Cache Mode**: 引入`none`/`all`/`align`三种缓存模式，Qwen-Next使用`align`模式
2. **状态管理**: 新增MambaCacheManager管理线性注意力层的conv_state和ssm_state
3. **Token对齐**: 调度器需要支持token对齐逻辑，确保状态在block边界保存
4. **状态复制**: 实现高效的状态复制机制，支持speculative decoding场景

该方案参考了vLLM的实现，并针对xLLM框架的架构特点进行了适配设计。

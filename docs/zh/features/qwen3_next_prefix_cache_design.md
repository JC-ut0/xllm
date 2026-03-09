# Qwen3-Next Prefix Cache 适配实现设计文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 日期 | 2026-03-07 |
| 作者 | xLLM Team |
| 状态 | 设计阶段 |

---

## 1. 概述

### 1.1 背景

Qwen3-Next 模型采用混合 Attention 架构（标准 Attention + Linear Attention），当前 Prefix Cache 实现无法正确支持 Linear Attention 层。本设计文档描述如何实现**方案一：禁用 Linear Attention 层的 Prefix Cache**。

### 1.2 设计目标

1. **正确性**：保证 Qwen3-Next 模型在启用 Prefix Cache 时输出正确
2. **兼容性**：不影响其他模型的 Prefix Cache 功能
3. **最小改动**：尽量减少代码修改范围
4. **可维护性**：代码结构清晰，易于理解和维护

### 1.3 关键约束

| 约束 | 说明 |
|------|------|
| Linear Attention 层 | 不参与 Prefix Cache 匹配和缓存 |
| 标准 Attention 层 | 正常使用 Prefix Cache |
| 模型识别 | 通过 `full_attention_interval` 参数识别层类型 |

---

## 2. 4+1 视图架构设计

### 2.1 视图概述

```
┌─────────────────────────────────────────────────────────────────┐
│                        4+1 视图模型                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │  场景视图    │                            │
│                      │ (Scenarios) │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  逻辑视图    │     │  开发视图    │     │  过程视图    │       │
│  │ (Logical)   │     │(Development)│     │ (Process)   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │  物理视图    │                            │
│                      │ (Physical)  │                            │
│                      └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 逻辑视图 (Logical View)

### 3.1 核心类图

```
┌─────────────────────────────────────────────────────────────────┐
│                         逻辑视图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    BlockManager                          │   │
│  │  <<interface>>                                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + allocate_shared(tokens_ids, existed_blocks): Blocks   │   │
│  │ + cache(token_ids, blocks, existed_num): void           │   │
│  │ + deallocate(blocks): void                              │   │
│  │ + num_blocks_in_prefix_cache(): size_t                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             △                                   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  BlockManagerImpl                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - options_: Options                                      │   │
│  │ - prefix_cache_: std::unique_ptr<PrefixCache>           │   │
│  │ - model_args_: const ModelArgs*  ◀── 【新增】            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + allocate_shared(tokens_ids, existed_blocks): Blocks   │   │
│  │ + cache(token_ids, blocks, existed_num): void           │   │
│  │ + is_linear_attention_layer(layer_id): bool  ◀── 【新增】│   │
│  │ + set_model_args(args): void  ◀── 【新增】              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     PrefixCache                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - block_size_: uint32_t                                  │   │
│  │ - cached_blocks_: unordered_map<Hash, Node*>            │   │
│  │ - lru_lst_: DNodeList                                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + match(token_ids, existed_blocks): Blocks              │   │
│  │ + insert(token_ids, blocks, existed_num): size_t        │   │
│  │ + evict(n_blocks): size_t                                │   │
│  │ + match_for_layer(token_ids, layer_id): Blocks ◀──【新增】│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模型参数扩展

```
┌─────────────────────────────────────────────────────────────────┐
│                    ModelArgs 扩展                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     ModelArgs                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ // 现有字段                                              │   │
│  │ - model_type_: std::string                               │   │
│  │ - n_layers_: int64_t                                     │   │
│  │ - n_heads_: int64_t                                      │   │
│  │ ...                                                      │   │
│  │                                                          │   │
│  │ // Qwen3-Next 特有字段 (已存在)                          │   │
│  │ - full_attention_interval_: int64_t  // 每 N 层使用标准Attn│   │
│  │                                                          │   │
│  │ // 新增方法                                              │   │
│  │ + is_hybrid_attention_model(): bool                      │   │
│  │   // 返回 full_attention_interval_ > 1                   │   │
│  │                                                          │   │
│  │ + is_linear_attention_layer(layer_id): bool              │   │
│  │   // 返回 (layer_id + 1) % full_attention_interval_ != 0 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 层类型判断逻辑

```
┌─────────────────────────────────────────────────────────────────┐
│                    层类型判断逻辑                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: layer_id (0-based 索引)                                  │
│        full_attention_interval (模型参数)                       │
│                                                                 │
│  判断逻辑:                                                       │
│                                                                 │
│  if (full_attention_interval <= 1) {                            │
│      // 传统模型，所有层都是标准 Attention                       │
│      return STANDARD_ATTENTION;                                 │
│  }                                                              │
│                                                                 │
│  // Qwen3-Next 混合架构                                         │
│  // Layer 3, 7, 11, ... (每 4 层) 是标准 Attention              │
│  if ((layer_id + 1) % full_attention_interval == 0) {           │
│      return STANDARD_ATTENTION;   // 支持 Prefix Cache          │
│  } else {                                                       │
│      return LINEAR_ATTENTION;     // 不支持 Prefix Cache        │
│  }                                                              │
│                                                                 │
│  示例 (full_attention_interval = 4):                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer ID │ (ID+1) % 4 │ 类型        │ Prefix Cache     │   │
│  │ ─────────────────────────────────────────────────────── │   │
│  │    0     │     1      │ Linear      │ ❌ 禁用          │   │
│  │    1     │     2      │ Linear      │ ❌ 禁用          │   │
│  │    2     │     3      │ Linear      │ ❌ 禁用          │   │
│  │    3     │     0      │ Standard    │ ✅ 启用          │   │
│  │    4     │     1      │ Linear      │ ❌ 禁用          │   │
│  │   ...    │    ...     │ ...         │ ...              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 开发视图 (Development View)

### 4.1 模块组织

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
│  │   │   │   └── block_manager_impl.cpp   【修改】              │
│  │   │   │                                                      │
│  │   │   ├── prefix_cache/                                      │
│  │   │   │   ├── prefix_cache.h           【修改】              │
│  │   │   │   └── prefix_cache.cpp         【修改】              │
│  │   │   │                                                      │
│  │   │   └── model/                                             │
│  │   │       └── model_args.h             【修改】              │
│  │   │                                                          │
│  │   ├── scheduler/                                             │
│  │   │   ├── chunked_prefill_scheduler.h  【修改】              │
│  │   │   ├── chunked_prefill_scheduler.cpp【修改】              │
│  │   │   ├── mix_scheduler.h              【修改】              │
│  │   │   └── mix_scheduler.cpp            【修改】              │
│  │   │                                                          │
│  │   └── runtime/                                               │
│  │       └── worker_impl.cpp              【修改】              │
│  │                                                              │
│  └── models/                                                    │
│      └── llm/                                                   │
│          └── qwen3_next.h                 【无需修改】          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 文件修改清单

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| `model_args.h` | 扩展 | 添加 `is_linear_attention_layer()` 方法 |
| `block_manager.h` | 扩展 | 添加 `set_model_args()` 和 `is_linear_attention_layer()` 接口 |
| `block_manager_impl.h` | 扩展 | 实现新接口，存储 model_args 引用 |
| `block_manager_impl.cpp` | 修改 | 修改 `allocate_shared()` 和 `cache()` 方法 |
| `prefix_cache.h` | 扩展 | 添加 `match_for_layer()` 方法 |
| `prefix_cache.cpp` | 扩展 | 实现 `match_for_layer()` 方法 |
| `chunked_prefill_scheduler.cpp` | 修改 | 传递 layer_id 到 BlockManager |
| `mix_scheduler.cpp` | 修改 | 传递 layer_id 到 BlockManager |
| `worker_impl.cpp` | 修改 | 初始化时设置 model_args |

### 4.3 接口定义

```cpp
// ==================== model_args.h ====================
struct ModelArgs {
    // ... 现有字段 ...
    
    // 新增方法
    bool is_hybrid_attention_model() const {
        return full_attention_interval() > 1;
    }
    
    bool is_linear_attention_layer(int64_t layer_id) const {
        if (full_attention_interval() <= 1) return false;
        return (layer_id + 1) % full_attention_interval() != 0;
    }
};

// ==================== block_manager.h ====================
class BlockManager {
 public:
    // ... 现有方法 ...
    
    // 新增方法
    virtual void set_model_args(const ModelArgs* args) = 0;
    virtual bool is_linear_attention_layer(int64_t layer_id) const = 0;
};

// ==================== prefix_cache.h ====================
class PrefixCache {
 public:
    // ... 现有方法 ...
    
    // 新增方法：带层 ID 的匹配
    std::vector<Block> match_for_layer(
        const Slice<int32_t>& token_ids,
        int64_t layer_id,
        const Slice<Block>& existed_shared_blocks = {});
};
```

---

## 5. 过程视图 (Process View)

### 5.1 Prefill 阶段时序图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefill 阶段时序图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scheduler     BlockManager     PrefixCache     ModelArgs       │
│     │              │               │              │             │
│     │              │               │              │             │
│     │ handle_prefill_request()     │              │             │
│     │──────────────────────────────▶              │             │
│     │              │               │              │             │
│     │              │ allocate_shared(tokens, layer_id)          │
│     │              │──────────────▶│              │             │
│     │              │               │              │             │
│     │              │               │ is_linear_attention_layer()?│
│     │              │               │─────────────▶│             │
│     │              │               │              │             │
│     │              │               │◀─────────────│             │
│     │              │               │  true/false  │             │
│     │              │               │              │             │
│     │              │               │              │             │
│     │              │    ┌──────────┴──────────┐   │             │
│     │              │    │                     │   │             │
│     │              │    │ if (Linear Attn)    │   │             │
│     │              │    │   return {}         │   │             │
│     │              │    │ else                │   │             │
│     │              │    │   match(tokens)     │   │             │
│     │              │    │   return blocks     │   │             │
│     │              │    │                     │   │             │
│     │              │    └──────────┬──────────┘   │             │
│     │              │               │              │             │
│     │              │◀──────────────│              │             │
│     │              │  blocks/[]    │              │             │
│     │              │               │              │             │
│     │◀─────────────│               │              │             │
│     │              │               │              │             │
│     │              │               │              │             │
│     │ prefill 完成 │               │              │             │
│     │              │               │              │             │
│     │ cache(tokens, blocks, layer_id)             │             │
│     │──────────────▶               │              │             │
│     │              │               │              │             │
│     │              │ is_linear_attention_layer()? │             │
│     │              │─────────────────────────────▶│             │
│     │              │               │              │             │
│     │              │◀────────────────────────────│             │
│     │              │  true/false   │              │             │
│     │              │               │              │             │
│     │              │    ┌──────────┴──────────┐   │             │
│     │              │    │ if (Linear Attn)    │   │             │
│     │              │    │   return (不缓存)   │   │             │
│     │              │    │ else                │   │             │
│     │              │    │   insert(tokens, blocks)│            │
│     │              │    │                     │   │             │
│     │              │    └──────────┬──────────┘   │             │
│     │              │               │              │             │
│     │◀─────────────│               │              │             │
│     │              │               │              │             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 状态转换图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefix Cache 状态转换                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Sequence 状态                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │   ┌──────────┐         ┌──────────┐         ┌──────────┐│   │
│  │   │  WAITING │────────▶│ RUNNING  │────────▶│ FINISHED ││   │
│  │   └──────────┘         └──────────┘         └──────────┘│   │
│  │        │                    │                    │      │   │
│  │        │                    │                    │      │   │
│  │        ▼                    ▼                    ▼      │   │
│  │   ┌──────────────────────────────────────────────────┐  │   │
│  │   │              Prefix Cache 操作                    │  │   │
│  │   ├──────────────────────────────────────────────────┤  │   │
│  │   │                                                  │  │   │
│  │   │  WAITING:                                        │  │   │
│  │   │    • 无操作                                       │  │   │
│  │   │                                                  │  │   │
│  │   │  RUNNING (Prefill):                              │  │   │
│  │   │    • 标准 Attention 层:                          │  │   │
│  │   │        - allocate_shared() → 匹配缓存            │  │   │
│  │   │        - 命中: 复用 blocks, ref_count++          │  │   │
│  │   │        - 未命中: 分配新 blocks                   │  │   │
│  │   │    • Linear Attention 层:                        │  │   │
│  │   │        - allocate_shared() → 返回空              │  │   │
│  │   │        - 分配新 blocks                           │  │   │
│  │   │                                                  │  │   │
│  │   │  FINISHED:                                      │  │   │
│  │   │    • 标准 Attention 层:                          │  │   │
│  │   │        - cache() → 插入 Prefix Cache            │  │   │
│  │   │    • Linear Attention 层:                        │  │   │
│  │   │        - 不缓存                                  │  │   │
│  │   │                                                  │  │   │
│  │   └──────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 并发处理

```
┌─────────────────────────────────────────────────────────────────┐
│                    并发处理流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  多序列并发 Prefill 场景:                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Scheduler                            │   │
│  │                                                         │   │
│  │  Sequence 1 (标准 Attn 层):                              │   │
│  │    ┌─────────────────────────────────────────────┐      │   │
│  │    │ allocate_shared() → match() → 命中 Block 0,1│      │   │
│  │    │ ref_count(Block 0) = 2                      │      │   │
│  │    │ ref_count(Block 1) = 2                      │      │   │
│  │    └─────────────────────────────────────────────┘      │   │
│  │                                                         │   │
│  │  Sequence 1 (Linear Attn 层):                            │   │
│  │    ┌─────────────────────────────────────────────┐      │   │
│  │    │ allocate_shared() → 跳过匹配                 │      │   │
│  │    │ 分配新 Block 5,6,7                          │      │   │
│  │    └─────────────────────────────────────────────┘      │   │
│  │                                                         │   │
│  │  Sequence 2 (标准 Attn 层):                              │   │
│  │    ┌─────────────────────────────────────────────┐      │   │
│  │    │ allocate_shared() → match() → 命中 Block 0,1│      │   │
│  │    │ ref_count(Block 0) = 3                      │      │   │
│  │    │ ref_count(Block 1) = 3                      │      │   │
│  │    └─────────────────────────────────────────────┘      │   │
│  │                                                         │   │
│  │  Sequence 2 (Linear Attn 层):                            │   │
│  │    ┌─────────────────────────────────────────────┐      │   │
│  │    │ allocate_shared() → 跳过匹配                 │      │   │
│  │    │ 分配新 Block 8,9,10                         │      │   │
│  │    └─────────────────────────────────────────────┘      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  结果:                                                          │
│  • 标准 Attention 层: Block 0,1 被多个序列共享                  │
│  • Linear Attention 层: 每个序列独立分配 blocks                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 物理视图 (Physical View)

### 6.1 部署架构

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
│  │  │  ┌────────────────────────────────────────────┐ │   │   │
│  │  │  │          Block Pool                        │ │   │   │
│  │  │  │                                            │ │   │   │
│  │  │  │  Block 0 ── Block 1 ── Block 2 ── ...      │ │   │   │
│  │  │  │    │          │                            │ │   │   │
│  │  │  │    │          │                            │ │   │   │
│  │  │  │    ▼          ▼                            │ │   │   │
│  │  │  │  ┌────────────────────────────────────┐   │ │   │   │
│  │  │  │  │      Prefix Cache (共享区域)        │   │ │   │   │
│  │  │  │  │                                    │   │ │   │   │
│  │  │  │  │  • 标准 Attention 层可共享         │   │ │   │   │
│  │  │  │  │  • Linear Attention 层独占         │   │ │   │   │
│  │  │  │  │                                    │   │ │   │   │
│  │  │  │  └────────────────────────────────────┘   │ │   │   │
│  │  │  │                                            │ │   │   │
│  │  │  └────────────────────────────────────────────┘ │   │   │
│  │  │                                                  │   │   │
│  │  │  ┌────────────────────────────────────────────┐ │   │   │
│  │  │  │      ssm_cache / conv_cache               │ │   │   │
│  │  │  │      (Linear Attention 专用)               │ │   │   │
│  │  │  │                                            │ │   │   │
│  │  │  │  • 每个 sequence 独立                      │ │   │   │
│  │  │  │  • 不参与 Prefix Cache 共享               │ │   │   │
│  │  │  │                                            │ │   │   │
│  │  │  └────────────────────────────────────────────┘ │   │   │
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
│  │  │  • Layer Type Info: 标准/Linear Attention       │   │   │
│  │  │                                                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 内存布局

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存布局示意                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  标准 Attention 层 KV Cache:                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  key_cache: [n_blocks, n_kv_heads, block_size, head_dim]│   │
│  │  value_cache: [n_blocks, n_kv_heads, block_size, head_dim]│  │
│  │                                                         │   │
│  │  Block 0: Token 0-15   ← 可共享 (Prefix Cache)         │   │
│  │  Block 1: Token 16-31  ← 可共享 (Prefix Cache)         │   │
│  │  Block 2: Token 32-47  ← 独占                          │   │
│  │  ...                                                    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Linear Attention 层 Cache:                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  conv_cache: [n_blocks, channels, kernel_dim-1]         │   │
│  │  ssm_cache: [n_blocks, n_v_heads, k_dim, k_dim]         │   │
│  │                                                         │   │
│  │  注意:                                                   │   │
│  │  • 每个 sequence 独立占用 blocks                        │   │
│  │  • 不参与 Prefix Cache 共享                             │   │
│  │  • Block 分配但内容独立                                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 场景视图 (Scenario View)

### 7.1 场景一：首次请求（无缓存）

```
┌─────────────────────────────────────────────────────────────────┐
│                    场景一：首次请求                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前提条件: Prefix Cache 为空                                    │
│                                                                 │
│  步骤:                                                          │
│                                                                 │
│  1. 用户发送请求: [System Prompt 1000 tokens] + [Query 100 tokens]│
│                                                                 │
│  2. Scheduler 处理:                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ for each layer:                                      │    │
│     │   if (is_standard_attention_layer(layer_id)):        │    │
│     │     blocks = allocate_shared(tokens)                 │    │
│     │     # 返回空，因为缓存为空                            │    │
│     │     # 分配新 blocks                                   │    │
│     │   else:  # Linear Attention                          │    │
│     │     blocks = allocate_shared(tokens)                 │    │
│     │     # 跳过匹配，直接分配新 blocks                     │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  3. 执行 Prefill 计算:                                          │
│     • 所有层正常计算                                            │
│     • 标准 Attention 层: 计算 KV 并存储到 blocks               │
│     • Linear Attention 层: 计算 ssm_state 并存储               │
│                                                                 │
│  4. Prefill 完成后缓存:                                         │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ for each layer:                                      │    │
│     │   if (is_standard_attention_layer(layer_id)):        │    │
│     │     cache(tokens, blocks)  # 插入 Prefix Cache      │    │
│     │   else:  # Linear Attention                          │    │
│     │     # 不缓存                                          │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  结果:                                                          │
│  • 标准 Attention 层的 Block 0-62 (1000 tokens) 被缓存         │
│  • Linear Attention 层不缓存                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 场景二：缓存命中

```
┌─────────────────────────────────────────────────────────────────┐
│                    场景二：缓存命中                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前提条件: Prefix Cache 中已有 1000 tokens 的缓存               │
│                                                                 │
│  步骤:                                                          │
│                                                                 │
│  1. 用户发送请求: [相同 System Prompt] + [新 Query 100 tokens]  │
│                                                                 │
│  2. Scheduler 处理:                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ for each layer:                                      │    │
│     │   if (is_standard_attention_layer(layer_id)):        │    │
│     │     blocks = allocate_shared(tokens)                 │    │
│     │     # 命中！返回 Block 0-62                          │    │
│     │     # ref_count++                                    │    │
│     │     # 只需分配 Block 63+ 用于新 tokens              │    │
│     │   else:  # Linear Attention                          │    │
│     │     blocks = allocate_shared(tokens)                 │    │
│     │     # 跳过匹配，分配所有 blocks                      │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  3. 执行 Prefill 计算:                                          │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ 标准 Attention 层:                                   │    │
│     │   • 输入: 100 新 tokens                              │    │
│     │   • KV: 前 1000 从缓存读取，后 100 新计算            │    │
│     │   • 节省: 1000 tokens 的 KV 计算                     │    │
│     │                                                      │    │
│     │ Linear Attention 层:                                 │    │
│     │   • 输入: 100 新 tokens                              │    │
│     │   • 需要完整计算 ssm_state                           │    │
│     │   • 无节省                                           │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. Prefill 完成后缓存:                                         │
│     • 标准 Attention 层: 更新缓存（新 tokens 部分）             │
│     • Linear Attention 层: 不缓存                              │
│                                                                 │
│  结果:                                                          │
│  • TTFT 降低约 10-15%                                           │
│  • 显存节省（共享 blocks）                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 场景三：多轮对话

```
┌─────────────────────────────────────────────────────────────────┐
│                    场景三：多轮对话                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前提条件: 已完成第一轮对话，生成了 200 tokens                   │
│                                                                 │
│  步骤:                                                          │
│                                                                 │
│  1. 第二轮对话请求:                                             │
│     [System Prompt 1000] + [Query1 100] + [Response1 200]       │
│     + [Query2 100]                                              │
│                                                                 │
│  2. Prefix Cache 匹配:                                          │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ 标准 Attention 层:                                   │    │
│     │   • 命中: Block 0-62 (System Prompt)                │    │
│     │   • 命中: Block 63-81 (Query1 + Response1 部分)     │    │
│     │   • 新分配: Block 82+ (Query2)                      │    │
│     │                                                      │    │
│     │ Linear Attention 层:                                 │    │
│     │   • 跳过匹配，完整计算                               │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  3. 收益分析:                                                   │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ 标准 Attention 层:                                   │    │
│     │   • 节省: 1300 tokens 的 KV 计算                     │    │
│     │                                                      │    │
│     │ Linear Attention 层:                                 │    │
│     │   • 无节省                                           │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  结果:                                                          │
│  • 多轮对话场景收益更明显                                       │
│  • 历史对话越长，标准 Attention 层收益越大                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 详细设计

### 8.1 实现方案说明

经过深入分析，发现当前架构中 KV Cache 是以 Sequence 为单位管理的，所有层共享同一组 blocks。因此无法实现"仅对标准 Attention 层启用 Prefix Cache"的方案。

**最终实现方案**：对于混合架构模型（如 Qwen3-Next），完全禁用 Prefix Cache。

### 8.2 ModelArgs 扩展

```cpp
// 文件: xllm/core/framework/model/model_args.h

struct ModelArgs {
    // ... 现有字段 ...
    
    // 判断是否为混合 Attention 模型
    bool is_hybrid_attention_model() const {
        return full_attention_interval() > 1;
    }
    
    // 判断指定层是否为 Linear Attention
    bool is_linear_attention_layer(int64_t layer_id) const {
        if (full_attention_interval() <= 1) {
            return false;  // 传统模型，所有层都是标准 Attention
        }
        return (layer_id + 1) % full_attention_interval() != 0;
    }
    
    // 判断指定层是否为标准 Attention
    bool is_standard_attention_layer(int64_t layer_id) const {
        return !is_linear_attention_layer(layer_id);
    }
};
```

### 8.3 BlockManager 接口扩展

```cpp
// 文件: xllm/core/framework/block/block_manager.h

class BlockManager {
 public:
    // ... 现有方法 ...
    
    // 设置模型参数（用于混合架构模型检测）
    virtual void set_model_args(const ModelArgs* args) {
        model_args_ = args;
    }
    
    // 检查是否为混合架构模型
    // 对于混合架构模型，禁用 Prefix Cache
    virtual bool is_hybrid_attention_model() const {
        if (!model_args_) {
            return false;
        }
        return model_args_->is_hybrid_attention_model();
    }

 protected:
    Options options_;
    const ModelArgs* model_args_ = nullptr;
};
```

### 8.4 BlockManagerImpl 实现

```cpp
// 文件: xllm/core/framework/block/block_manager_impl.cpp

std::vector<Block> BlockManagerImpl::allocate_shared(...) {
    if (options_.enable_prefix_cache()) {
        // 对于混合架构模型，禁用 Prefix Cache
        if (is_hybrid_attention_model()) {
            return {};
        }
        // ... 原有匹配逻辑 ...
    }
    return {};
}

void BlockManagerImpl::cache(...) {
    if (options_.enable_prefix_cache()) {
        // 对于混合架构模型，禁用 Prefix Cache
        if (is_hybrid_attention_model()) {
            return;
        }
        // ... 原有缓存逻辑 ...
    }
}
```

### 8.5 BlockManagerPool 适配

```cpp
// 文件: xllm/core/framework/block/block_manager_pool.h

class BlockManagerPool : public KVCacheManager {
 public:
    // 设置模型参数，传递给所有 BlockManager
    virtual void set_model_args(const ModelArgs* args) {
        model_args_ = args;
        for (auto& manager : block_managers_) {
            manager->set_model_args(args);
        }
    }

 protected:
    const ModelArgs* model_args_ = nullptr;
};
```

### 8.6 引擎初始化

```cpp
// 文件: xllm/core/distributed_runtime/llm_engine.cpp

// 创建 BlockManagerPool 后设置 model_args
kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, dp_size_);
kv_cache_manager_->set_model_args(&args_);
```

---

## 9. 测试计划

### 9.1 单元测试

| 测试项 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `is_linear_attention_layer()` | layer_id = 0, 1, 2, 3 | true, true, true, false |
| `is_standard_attention_layer()` | layer_id = 0, 1, 2, 3 | false, false, false, true |
| `allocate_shared()` Linear 层 | Linear Attention 层调用 | 返回空，分配新 blocks |
| `allocate_shared()` 标准层 | 标准 Attention 层调用 | 正常匹配缓存 |
| `cache()` Linear 层 | Linear Attention 层调用 | 不缓存 |
| `cache()` 标准层 | 标准 Attention 层调用 | 正常缓存 |

### 9.2 集成测试

| 测试场景 | 测试内容 | 预期结果 |
|---------|---------|---------|
| 首次请求 | 无缓存时的 Prefill | 正常计算，缓存标准层 |
| 缓存命中 | 相同前缀的请求 | 标准 Attn 层复用缓存 |
| 多轮对话 | 历史对话复用 | 历史 KV 正确复用 |
| 正确性验证 | 对比有无缓存输出 | 输出一致 |

### 9.3 性能测试

| 测试场景 | Prefix 长度 | 预期 TTFT 降低 |
|---------|------------|---------------|
| 短前缀 | 256 tokens | 5-10% |
| 中等前缀 | 1024 tokens | 10-15% |
| 长前缀 | 4096 tokens | 15-20% |

---

## 10. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 层 ID 传递错误 | 中 | 添加断言和日志验证 |
| 缓存不一致 | 低 | 充分测试，添加一致性检查 |
| 性能回归 | 低 | 性能基准测试 |
| 兼容性问题 | 低 | 保持向后兼容，默认行为不变 |

---

## 11. 参考资料

1. [Qwen3-Next Prefix Cache 适配文档](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/docs/zh/features/qwen3_next_prefix_cache_adaptation.md)
2. [Prefix Cache 文档](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/docs/zh/features/prefix_cache.md)
3. [xLLM Block Manager 实现](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/block/block_manager_impl.h)
4. [xLLM Prefix Cache 实现](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp)

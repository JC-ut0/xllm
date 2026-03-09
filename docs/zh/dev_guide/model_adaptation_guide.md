# xLLM 模型适配设计文档

## 概述

本文档描述 xLLM 框架的模型适配设计方案，以 Qwen3-Next 模型适配为例，详细说明如何在 xLLM 中添加新模型支持。文档采用 4+1 视图方法进行组织，从多个维度全面展示模型适配的设计思路。

---

## 一、4+1 视图架构

```
                                    ┌─────────────────────────────────┐
                                    │         场景视图 (Scenarios)     │
                                    │    开发者适配新模型的完整流程     │
                                    └─────────────────┬───────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
    │     逻辑视图 (Logical)     │   │     开发视图 (Development)  │   │    过程视图 (Process)      │
    │   模型类层次与接口设计     │   │   代码组织与模块结构       │   │   模型加载与推理流程       │
    └───────────────────────────┘   └───────────────────────────┘   └───────────────────────────┘
                    │                                 │                                 │
                    └─────────────────────────────────┼─────────────────────────────────┘
                                                      │
                                                      ▼
                                    ┌─────────────────────────────────┐
                                    │     物理视图 (Physical)         │
                                    │   文件布局与部署结构            │
                                    └─────────────────────────────────┘
```

---

## 二、场景视图 (Scenario View)

### 2.1 核心场景：适配新模型

**场景描述**：开发者需要在 xLLM 框架中添加一个新的 LLM 模型支持。

**参与者**：
- 开发者
- xLLM 框架
- 模型权重文件

**前置条件**：
- 模型权重为 HuggingFace 格式
- 了解模型架构（层数、注意力类型、MLP 结构等）

**主成功场景**：

```
1. 开发者创建模型定义文件 (models/llm/your_model.h)
2. 开发者实现核心层组件 (core/layers/)
3. 开发者使用注册宏注册模型工厂和参数加载器
4. 开发者实现权重加载逻辑 (load_state_dict)
5. 开发者编译并测试模型
6. 模型成功加载并推理
```

**场景流程图**：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   创建模型    │     │   实现层组件  │     │   注册模型    │     │   测试验证    │
│   定义文件    │────>│   (Attention │────>│   工厂和参数  │────>│   编译运行    │
│              │     │   MLP等)     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.2 扩展场景：异构模型适配

**场景描述**：适配像 Qwen3-Next 这样的异构架构模型（不同层使用不同组件）。

**特殊处理**：
- 无法继承基类模板，需要独立实现
- 需要在 Decoder Layer 中根据 `layer_id` 动态选择组件
- 需要处理多种注意力类型的 KV Cache

---

## 三、逻辑视图 (Logical View)

### 3.1 核心类层次结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ModelRegistry (单例)                            │
│  - 管理所有模型的工厂函数                                                      │
│  - 管理参数加载器                                                            │
│  - 提供 get_causallm_factory() 等接口                                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 注册
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CausalLMImpl<ModelClass>                            │
│  - 包装具体模型类                                                            │
│  - 提供 forward(), logits() 接口                                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 组合
                                      ▼
          ┌───────────────────────────┴───────────────────────────┐
          │                                                       │
          ▼                                                       ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│  LlmModelImplBase<Layer>│                         │   独立模型实现 (如       │
│  (基类模板)              │                         │   Qwen3NextModelImpl)   │
│  - embed_tokens_        │                         │   - 自定义层结构         │
│  - layers_: vector<Layer>│                        │   - 异构组件支持         │
│  - norm_                │                         │                         │
└───────────┬─────────────┘                         └───────────┬─────────────┘
            │                                                   │
            │ 继承                               独立实现        │
            ▼                                                   ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│   QWen3ModelImpl        │                         │  Qwen3NextModelImpl     │
│   (同构模型示例)         │                         │  (异构模型示例)          │
│   - 所有层类型相同       │                         │  - 层结构动态变化        │
└─────────────────────────┘                         └─────────────────────────┘
```

### 3.2 Decoder Layer 接口设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DecoderLayer 接口                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  + forward(hidden_states, positions, attn_metadata, kv_cache, input_params) │
│  + load_state_dict(state_dict)                                              │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 实现
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ Qwen3DecoderLayer   │   │ Qwen3NextDecoderLayer│   │  其他 DecoderLayer  │
│ (同构)              │   │ (异构)               │   │                     │
│ - attention_        │   │ - attention_ (可选)  │   │                     │
│ - mlp_              │   │ - linear_attention_  │   │                     │
│ - norm_             │   │ - mlp_ / moe_mlp_    │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

### 3.3 关键设计模式

#### 工厂模式 + 宏注册

```cpp
// 注册宏定义
#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass)      \
  const bool ModelType##_registered = []() {              \
    ModelRegistry::register_causallm_factory(             \
        #ModelType, [](const ModelContext& context) {     \
          ModelClass model(context);                      \
          model->eval();                                  \
          return std::make_unique<CausalLMImpl<ModelClass>>( \
              std::move(model), context.get_tensor_options()); \
        });                                               \
    return true;                                          \
  }()

// 使用示例
REGISTER_CAUSAL_MODEL(qwen3_next, Qwen3NextForCausalLM);
```

#### CRTP 模式 (基类模板)

```cpp
template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 protected:
  std::vector<DecoderLayerType> layers_;  // 同构层容器
};
```

---

## 四、开发视图 (Development View)

### 4.1 目录结构

```
xllm/
├── models/                              # 模型定义层
│   ├── model_registry.h                 # 模型注册中心
│   │
│   ├── llm/                             # 大语言模型
│   │   ├── llm_model_base.h             # LLM 基类模板
│   │   ├── qwen2.h                      # Qwen2 模型
│   │   ├── qwen3.h                      # Qwen3 模型
│   │   ├── qwen3_moe.h                  # Qwen3 MoE 模型
│   │   └── qwen3_next.h                 # Qwen3-Next 模型 ⭐
│   │
│   └── vlm/                             # 视觉语言模型
│       ├── qwen2_vl.h
│       ├── qwen2_5_vl.h
│       └── qwen3_vl.h
│
├── core/
│   ├── layers/                          # 网络层实现
│   │   ├── common/                      # 通用层组件
│   │   │   ├── attention.h              # 标准注意力
│   │   │   ├── linear.h                 # 线性层
│   │   │   ├── rms_norm.h               # RMS 归一化
│   │   │   ├── qwen3_next_attention.h       ⭐
│   │   │   ├── qwen3_next_gated_delta_net.h ⭐
│   │   │   └── qwen3_next_rms_norm.h        ⭐
│   │   │
│   │   ├── qwen3_decoder_layer.h
│   │   ├── qwen3_next_decoder_layer.h       ⭐
│   │   └── npu/                         # NPU 特定实现
│   │
│   └── framework/                       # 核心框架
│       ├── model_loader.h               # 模型加载器基类
│       ├── hf_model_loader.h            # HuggingFace 格式加载器
│       ├── model_context.h              # 模型上下文
│       ├── model_args.h                 # 模型参数定义
│       └── state_dict/                  # 权重字典
│
└── api_service/                         # API 服务层
    ├── chat_service_impl.h
    └── completion_service_impl.h
```

### 4.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              api_service                                     │
│  (HTTP/gRPC 接口层)                                                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 依赖
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               models                                         │
│  (模型定义: Qwen3NextForCausalLM, QWen3ForCausalLM, ...)                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 依赖
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              core/layers                                     │
│  (网络层: Qwen3NextDecoderLayer, Qwen3NextAttention, ...)                    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 依赖
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             core/framework                                   │
│  (框架核心: ModelLoader, StateDict, ModelContext, KVCache, ...)              │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ 依赖
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             core/kernels                                     │
│  (算子实现: CUDA, NPU, MLU)                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 新模型适配文件清单

适配新模型需要创建/修改的文件：

| 文件类型 | 路径模板 | 说明 |
|---------|---------|------|
| 模型定义 | `models/llm/{model_name}.h` | 模型类定义和注册 |
| Decoder Layer | `core/layers/{model_name}_decoder_layer.h` | 解码层实现 |
| Attention | `core/layers/common/{model_name}_attention.h` | 注意力层（如有特殊实现） |
| MLP | `core/layers/common/{model_name}_mlp.h` | MLP 层（如有特殊实现） |
| Norm | `core/layers/common/{model_name}_norm.h` | 归一化层（如有特殊实现） |

---

## 五、过程视图 (Process View)

### 5.1 模型加载流程

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────────────┐
│  Client  │     │ ModelLoader  │     │ ModelRegistry │     │ Qwen3NextForCausalLM│
└────┬─────┘     └──────┬───────┘     └───────┬───────┘     └──────────┬──────────┘
     │                  │                     │                        │
     │ 1. create(path)  │                     │                        │
     │─────────────────>│                     │                        │
     │                  │                     │                        │
     │                  │ 2. load_model_args()│                        │
     │                  │   (解析 config.json)│                        │
     │                  │────────────────────>│                        │
     │                  │                     │                        │
     │                  │ 3. get_state_dicts()│                        │
     │                  │   (加载权重文件)     │                        │
     │                  │────────────────────>│                        │
     │                  │                     │                        │
     │ 4. get_causallm_factory("qwen3_next")  │                        │
     │─────────────────────────────────────────>                        │
     │                  │                     │                        │
     │                  │                     │ 5. create(context)     │
     │                  │                     │───────────────────────>│
     │                  │                     │                        │
     │                  │                     │ 6. load_state_dict()   │
     │                  │                     │<───────────────────────│
     │                  │                     │                        │
     │ 7. 返回模型实例  │                     │                        │
     │<─────────────────│─────────────────────│────────────────────────│
```

### 5.2 推理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              推理请求入口                                    │
│                           (ChatService/CompletionService)                   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 LlmMaster                                    │
│  - 调度请求                                                                  │
│  - 管理批处理                                                                │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 LlmEngine                                    │
│  - 执行推理步骤 (step)                                                       │
│  - 管理 KV Cache                                                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Qwen3NextForCausalLM                              │
│  - forward(tokens, positions, kv_caches, input_params)                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Qwen3NextModel                                  │
│  1. embed_tokens_(tokens) → hidden_states                                   │
│  2. for each layer: layer_(hidden_states, ...)                             │
│  3. norm_(hidden_states)                                                    │
│  4. return ModelOutput(hidden_states)                                       │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Qwen3NextDecoderLayer                               │
│  1. input_norm_(x)                                                          │
│  2. attention_ / linear_attention_ (根据 layer_id 选择)                      │
│  3. residual connection                                                     │
│  4. post_norm_(x)                                                           │
│  5. mlp_ / moe_mlp_ (根据配置选择)                                           │
│  6. residual connection                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 权重加载流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          load_model(loader)                                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    遍历 loader->get_state_dicts()                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          │                                                       │
          ▼                                                       ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│  model_->load_state_dict│                         │  lm_head_->load_state_dict│
│  (prefix: "model.")     │                         │  (prefix: "lm_head.")    │
└───────────┬─────────────┘                         └─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Qwen3NextModel::load_state_dict                           │
│  1. embed_tokens_->load_state_dict("embed_tokens.")                         │
│  2. for each layer: layers_[i]->load_state_dict("layers.{i}.")              │
│  3. norm_->load_state_dict("norm.")                                          │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Qwen3NextDecoderLayer::load_state_dict                       │
│  1. attention_->load_state_dict("self_attn.")      [如果存在]                │
│     或 linear_attention_->load_state_dict("linear_attn.")                    │
│  2. input_norm_->load_state_dict("input_layernorm.")                         │
│  3. post_norm_->load_state_dict("post_attention_layernorm.")                 │
│  4. mlp_->load_state_dict("mlp.") 或 moe_mlp_->load_state_dict("mlp.")       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、物理视图 (Physical View)

### 6.1 文件布局

```
xllm/
│
├── models/                              # 模型定义模块
│   ├── model_registry.h                 # 注册中心 (单例)
│   │
│   ├── llm/                             # LLM 模型
│   │   ├── llm_model_base.h             # 基类模板 (~220 行)
│   │   ├── qwen3.h                      # Qwen3 (~260 行)
│   │   └── qwen3_next.h                 # Qwen3-Next (~350 行)
│   │
│   └── vlm/                             # VLM 模型
│
├── core/
│   ├── layers/                          # 网络层模块
│   │   ├── common/                      # 通用组件
│   │   │   ├── attention.h              # 标准注意力
│   │   │   ├── linear.h                 # 线性层
│   │   │   ├── rms_norm.h               # RMS Norm
│   │   │   ├── qwen3_next_attention.h   # Qwen3-Next 注意力
│   │   │   └── qwen3_next_gated_delta_net.h  # 线性注意力
│   │   │
│   │   ├── qwen3_decoder_layer.h        # Qwen3 解码层
│   │   ├── qwen3_decoder_layer.cpp
│   │   ├── qwen3_next_decoder_layer.h   # Qwen3-Next 解码层
│   │   └── qwen3_next_decoder_layer.cpp
│   │
│   └── framework/                       # 框架核心
│       ├── model_loader.h               # 加载器基类
│       ├── hf_model_loader.h            # HF 加载器
│       ├── hf_model_loader.cpp
│       ├── model_args.h                 # 参数定义
│       └── model_context.h              # 模型上下文
│
└── api_service/                         # API 服务
```

### 6.2 编译产物

```
build/
├── libxllm.so / xllm.dll               # 核心库
├── xllm_server                          # 服务可执行文件
└── tests/
    └── model_test                       # 模型测试
```

### 6.3 部署结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              xLLM Server                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Master    │  │   Engine    │  │   Worker    │  │   Model     │        │
│  │  (调度)     │  │  (推理)     │  │  (计算)     │  │  (权重)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        模型权重文件            │
                    │  (model.safetensors,          │
                    │   config.json,                │
                    │   tokenizer.json)             │
                    └───────────────────────────────┘
```

---

## 七、Qwen3-Next 适配详解

### 7.1 架构特点

Qwen3-Next 是一种**异构混合架构**模型，具有以下特点：

| 特性 | 描述 |
|------|------|
| 混合注意力 | 标准注意力 + 线性注意力（Gated Delta Net）交替使用 |
| MoE 架构 | 支持混合专家模型，512 专家，每 token 激活 10 专家 |
| Q-K 归一化 | 注意力层使用 Q/K 归一化提升稳定性 |
| 部分旋转编码 | 仅对 25% 的维度应用 RoPE |
| 异构层结构 | 不同层使用不同的组件组合 |

### 7.2 为什么需要独立实现

**基类模板的限制**：

```cpp
// llm_model_base.h - 基类模板假设所有层同构
template <typename DecoderLayerType>  // 只有一个类型参数！
class LlmModelImplBase : public torch::nn::Module {
 protected:
  std::vector<DecoderLayerType> layers_;  // 所有层必须是同一类型！
};
```

**Qwen3-Next 的异构结构**：

```
Layer 0: LinearAttention + DenseMLP
Layer 1: LinearAttention + DenseMLP
Layer 2: LinearAttention + DenseMLP
Layer 3: StandardAttention + MoE        ← 每 4 层一次标准注意力
Layer 4: LinearAttention + DenseMLP
Layer 5: LinearAttention + DenseMLP
Layer 6: LinearAttention + DenseMLP
Layer 7: StandardAttention + MoE        ← 每 4 层一次标准注意力
...
```

**对比图**：

```
┌─────────────────────────────────────────────────────────────────┐
│                         Qwen3 (同构)                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0: Qwen3DecoderLayer (Attention + MLP)                   │
│  Layer 1: Qwen3DecoderLayer (Attention + MLP)                   │
│  ...                                                            │
│  Layer N: Qwen3DecoderLayer (Attention + MLP)                   │
│                         ↓ 可继承基类模板 ↓                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Qwen3-Next (异构)                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0: LinearAttention + DenseMLP                            │
│  Layer 1: LinearAttention + DenseMLP                            │
│  Layer 2: LinearAttention + DenseMLP                            │
│  Layer 3: StandardAttention + MoE        ← 每4层一次             │
│  Layer 4: LinearAttention + DenseMLP                            │
│  ...                                                            │
│                      ↓ 必须独立实现 ↓                            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 关键实现代码

#### Decoder Layer 动态组件选择

```cpp
// qwen3_next_decoder_layer.cpp
Qwen3NextDecoderLayerImpl::Qwen3NextDecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& model_args = context.get_model_args();
  
  // 根据层 ID 和配置参数动态选择注意力类型
  int32_t full_attn_interval = model_args.full_attention_interval();
  if (full_attn_interval > 0 && (layer_id + 1) % full_attn_interval == 0) {
    // 标准注意力
    attention_ = register_module("self_attn", Qwen3NextAttention(...));
  } else {
    // 线性注意力
    linear_attention_ = register_module("linear_attn", Qwen3NextGatedDeltaNet(...));
  }
  
  // 根据配置动态选择 MLP 类型
  if (... && model_args.n_routed_experts() > 0 && ...) {
    moe_mlp_ = register_module("mlp", FusedMoE(...));
  } else {
    mlp_ = register_module("mlp", DenseMLP(...));
  }
}
```

#### 模型注册

```cpp
// qwen3_next.h

// 注册模型工厂
REGISTER_CAUSAL_MODEL(qwen3_next, Qwen3NextForCausalLM);

// 注册模型参数
REGISTER_MODEL_ARGS(qwen3_next, [&] {
  // 基础参数
  LOAD_ARG_OR(model_type, "model_type", "qwen3_next");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 2);
  
  // MoE 参数
  LOAD_ARG_OR(num_experts, "num_experts", 512);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 10);
  
  // Qwen3-Next 特有参数
  LOAD_ARG_OR(full_attention_interval, "full_attention_interval", 4);
  LOAD_ARG_OR(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 0.25f);
});
```

### 7.4 关键参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `hidden_size` | 2048 | 隐藏层维度 |
| `n_layers` | 48 | 层数 |
| `n_heads` | 16 | 注意力头数 |
| `n_kv_heads` | 2 | KV 头数（GQA） |
| `num_experts` | 512 | MoE 专家数 |
| `num_experts_per_tok` | 10 | 每 token 激活专家数 |
| `full_attention_interval` | 4 | 标准注意力间隔 |
| `partial_rotary_factor` | 0.25 | RoPE 应用比例 |

---

## 八、新模型适配步骤

### 8.1 步骤概览

```
Step 1: 创建模型定义文件
    └── models/llm/your_model.h

Step 2: 实现核心层组件
    └── core/layers/common/your_attention.h
    └── core/layers/your_decoder_layer.h

Step 3: 注册模型
    └── 在模型头文件中使用注册宏

Step 4: 添加模型参数
    └── 使用 LOAD_ARG_OR 宏定义参数映射

Step 5: 实现权重加载
    └── load_state_dict() 方法

Step 6: 测试验证
    └── 单元测试 + 集成测试
```

### 8.2 详细步骤

#### Step 1: 创建模型头文件

```cpp
// models/llm/your_model.h
#pragma once

#include "core/framework/model_context.h"
#include "core/layers/your_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

// 方式一：继承基类模板（适用于同构模型）
class YourModelImpl : public LlmModelImplBase<layer::YourDecoderLayer> {
 public:
  YourModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::YourDecoderLayer>("your_model",
                                                   context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ = register_module("embed_tokens", layer::WordEmbedding(context));
    
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      layers_.push_back(layer::YourDecoderLayer(context));
    }
  }
};
TORCH_MODULE(YourModel);

class YourForCausalLMImpl : public LlmForCausalLMImplBase<YourModel> {
 public:
  YourForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<YourModel>(context) {}
};
TORCH_MODULE(YourForCausalLM);

// 注册模型
REGISTER_CAUSAL_MODEL(your_model, YourForCausalLM);

// 注册参数
REGISTER_MODEL_ARGS(your_model, [&] {
  LOAD_ARG_OR(model_type, "model_type", "your_model");
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  // ... 更多参数
});

}  // namespace xllm
```

#### Step 2: 实现 Decoder Layer

```cpp
// core/layers/your_decoder_layer.h
namespace xllm {
namespace layer {

class YourDecoderLayerImpl : public torch::nn::Module {
 public:
  explicit YourDecoderLayerImpl(const ModelContext& context);
  
  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);
  
  void load_state_dict(const StateDict& state_dict);

 private:
  YourAttention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
};
TORCH_MODULE(YourDecoderLayer);

}  // namespace layer
}  // namespace xllm
```

#### Step 3: 实现权重加载

```cpp
void YourDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}
```

---

## 九、最佳实践

### 9.1 代码规范

1. **使用参数而非硬编码**：所有配置项应通过 `ModelArgs` 传递
2. **遵循命名约定**：权重名称与 HuggingFace 格式保持一致
3. **添加必要注释**：解释特殊处理的逻辑
4. **复用现有组件**：优先使用 `core/layers/common/` 中的通用组件

### 9.2 性能优化

1. **使用 `torch::NoGradGuard`**：推理时禁用梯度计算
2. **合理使用并行线性层**：`QKVParallelLinear`、`RowParallelLinear`
3. **优化 KV Cache**：使用 PagedAttention 或 ContinuousBatching

### 9.3 调试技巧

1. **检查权重加载**：确保所有权重都被正确加载
2. **验证输出形状**：每层的输出形状应与预期一致
3. **对比参考实现**：与 HuggingFace 实现对比输出

---

## 十、参考文件

| 文件 | 说明 |
|------|------|
| [model_registry.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/models/model_registry.h) | 模型注册中心 |
| [llm_model_base.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/models/llm/llm_model_base.h) | LLM 基类模板 |
| [qwen3.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/models/llm/qwen3.h) | Qwen3 模型（同构示例） |
| [qwen3_next.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/models/llm/qwen3_next.h) | Qwen3-Next 模型（异构示例） |
| [qwen3_next_decoder_layer.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/layers/qwen3_next_decoder_layer.h) | Decoder 层实现 |
| [qwen3_next_attention.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/layers/common/qwen3_next_attention.h) | 标准注意力实现 |
| [qwen3_next_gated_delta_net.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/layers/common/qwen3_next_gated_delta_net.h) | 线性注意力实现 |
| [hf_model_loader.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/hf_model_loader.h) | HF 格式模型加载器 |
| [model_args.h](file:///d:/ext.xuyexiong1/Desktop/工作/xllm/xllm/core/framework/model/model_args.h) | 模型参数定义 |

---

## 附录：常见问题

### Q1: 什么时候应该继承基类模板，什么时候应该独立实现？

**继承基类模板**：
- 所有 Decoder Layer 结构相同（同构模型）
- 标准 Transformer 架构
- 如：Qwen2、Qwen3、Llama 等

**独立实现**：
- 不同层使用不同组件（异构模型）
- 混合架构（如混合注意力、混合 MLP）
- 如：Qwen3-Next（混合 Standard + Linear Attention）

### Q2: 如何添加新的模型参数？

1. 在 `model_args.h` 中添加 `PROPERTY` 定义
2. 在模型注册时使用 `LOAD_ARG_OR` 加载参数
3. 在代码中通过 `model_args.xxx()` 访问参数

### Q3: 权重名称不匹配怎么办？

在 `load_state_dict` 中使用 `get_dict_with_prefix` 进行前缀映射，或在加载时手动重命名权重。

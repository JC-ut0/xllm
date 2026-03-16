# Qwen3-Next Ascend C causal_conv1d_fn 适配设计文档

## 一、背景

### 1.1 PR #6661 核心变更

vllm-ascend PR #6661 新增了 **prefill 阶段** 的 Ascend C `causal_conv1d_fn` 算子，用于替代 PyTorch 的 `torch.conv1d` 实现，提升性能。

| 特性 | 说明 |
|------|------|
| **算子名称** | `aclnnCausalConv1d` |
| **适用场景** | Prefill/Extend 阶段（非 decode） |
| **数据类型** | BF16, FP16 |
| **激活函数** | 支持 silu |
| **输入布局** | `[cu_seqlen, dim]` 或 `[batch, seq_len, dim]` |

### 1.2 当前 xLLM 实现现状

当前 `qwen3_next_gated_delta_net.cpp` 中：

- **Prefill 阶段**：使用 `torch::conv1d`，性能较差
- **Decode 阶段**：已有 `causal_conv1d_update` Triton kernel

### 1.3 代码库目录结构

```
xllm/
├── core/
│   └── kernels/
│       ├── param.h                    # 参数结构体定义
│       ├── ops_api.h                  # 算子接口声明
│       ├── ops_api.cpp                # 算子接口实现
│       └── npu/
│           ├── npu_ops_api.h          # NPU 算子接口
│           ├── utils.cpp              # ACLNN 辅助函数
│           └── xllm_ops/              # ACLNN 自定义算子调用层
│               ├── xllm_ops_api.h     # 接口声明
│               ├── causal_conv1d_fn.cpp  # ACLNN 调用实现
│               └── CMakeLists.txt     # 构建配置

third_party/
├── xllm_ops/                          # 自定义算子源码仓库 (git submodule)
│   └── xllm_ops/
│       └── causal_conv1d/             # causal_conv1d 算子
│           ├── CMakeLists.txt
│           ├── causal_conv1d_tiling.h
│           ├── causal_conv1d.h
│           ├── causal_conv1d.cpp
│           └── causal_conv1d_cpu.cpp
└── torch_npu_ops/                     # ATB/Triton 算子
```

---

## 二、设计方案

### 2.1 算子放置位置

**选择 `third_party/xllm_ops/xllm_ops/causal_conv1d/` 目录**，原因：

1. `aclnnCausalConv1d` 是 ACLNN 自定义算子，与 `xllm_ops` 仓库风格一致
2. 编译时会随 `xllm_ops` 一起自动编译，无需额外配置
3. 与 `torch_npu_ops/` 中的 ATB/Triton 算子区分开

### 2.2 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    qwen3_next_gated_delta_net.cpp               │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐│
│  │   Prefill Phase     │    │       Decode Phase              ││
│  │ causal_conv1d_fn()  │    │  causal_conv1d_update()         ││
│  │   (新增)            │    │  (已有)                         ││
│  └─────────────────────┘    └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ops_api.cpp                                │
│  causal_conv1d_fn() ─────► npu::causal_conv1d_fn()             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          xllm/core/kernels/npu/xllm_ops/causal_conv1d_fn.cpp    │
│  causal_conv1d_fn() ──► aclnnCausalConv1d (标准 ACLNN 调用)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│       third_party/xllm_ops/xllm_ops/causal_conv1d/              │
│  Ascend C Kernel 实现 (编译后生成 aclnnCausalConv1d 算子)       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 需要修改/新增的文件

| 文件路径 | 操作类型 | 说明 |
|----------|----------|------|
| `xllm/core/kernels/param.h` | ✅ 已完成 | 新增 `CausalConv1dFnParams` 参数结构体 |
| `xllm/core/kernels/ops_api.h` | ✅ 已完成 | 新增 `causal_conv1d_fn` 函数声明 |
| `xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h` | ✅ 已完成 | 新增 `causal_conv1d_fn` 函数声明 |
| `xllm/core/kernels/npu/xllm_ops/causal_conv1d_fn.cpp` | ✅ 已完成 | 新增 ACLNN 调用实现 |
| `xllm/core/kernels/npu/xllm_ops/CMakeLists.txt` | ✅ 已完成 | 添加新源文件 |
| `xllm/core/kernels/ops_api.cpp` | ✅ 已完成 | 新增 `causal_conv1d_fn` 函数实现 |
| `third_party/xllm_ops/xllm_ops/causal_conv1d/` | ✅ 已完成 | 自定义算子源码目录 |

---

## 三、详细设计

### 3.1 参数结构体 (`param.h`)

**文件位置**: `xllm/core/kernels/param.h`

**作用**: 定义算子调用时需要的所有参数，方便统一管理。

```cpp
// NPU Causal Conv1d Fn parameters (for prefill phase)
// Based on vllm-ascend PR #6661: aclnnCausalConv1d
struct CausalConv1dFnParams {
  // 输入张量，形状为 [cu_seqlen, dim] 或 [batch, seq_len, dim]
  // cu_seqlen 是所有序列 token 数的总和
  torch::Tensor x;
  
  // 卷积权重，形状为 [width, dim]
  // width=4 表示卷积核大小为 4
  torch::Tensor weight;
  
  // 可选偏置，形状为 [dim]
  std::optional<torch::Tensor> bias = std::nullopt;
  
  // 卷积状态缓存，形状为 [num_cache_lines, state_len, dim]
  // 用于存储每个序列的历史状态
  torch::Tensor conv_state;
  
  // 每个序列是否有初始状态，形状为 [batch]
  // 布尔值，true 表示有历史状态
  torch::Tensor has_initial_state;
  
  // 每个序列的缓存索引，形状为 [batch]
  // 指向 conv_state 中的哪一行
  torch::Tensor cache_indices;
  
  // 查询起始位置，形状为 [batch+1]
  // 累积序列长度，类似 CSR 格式的行指针
  torch::Tensor query_start_loc;
  
  // 激活模式: 0=无激活, 1=silu
  int64_t activation_mode = 1;
  
  // 无效槽位的填充 ID
  int64_t pad_slot_id = -1;
};
```

### 3.2 算子接口声明 (`ops_api.h`)

**文件位置**: `xllm/core/kernels/ops_api.h`

**作用**: 声明算子接口，供上层代码调用。

```cpp
torch::Tensor causal_conv1d_fn(CausalConv1dFnParams& params);
```

### 3.3 NPU 层接口声明 (`xllm_ops_api.h`)

**文件位置**: `xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h`

**作用**: 声明 NPU 层的具体实现接口。

```cpp
torch::Tensor causal_conv1d_fn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation_mode,
    torch::Tensor& conv_state,
    const torch::Tensor& has_initial_state,
    const torch::Tensor& cache_indices,
    const torch::Tensor& query_start_loc,
    int64_t pad_slot_id);
```

### 3.4 ACLNN 调用实现 (`causal_conv1d_fn.cpp`)

**文件位置**: `xllm/core/kernels/npu/xllm_ops/causal_conv1d_fn.cpp`

**作用**: 调用 ACLNN 算子执行计算。

**ACLNN 调用流程**:

```
1. 检查输入张量
2. 创建输出张量
3. 获取 NPU stream
4. 创建 ACL tensor (将 torch::Tensor 转换为 aclTensor*)
5. 获取 workspace 大小 (aclnnXxxGetWorkspaceSize)
6. 分配 workspace 内存
7. 执行算子 (aclnnXxx)
8. 同步 stream
9. 释放资源
10. 返回结果
```

**核心代码**:

```cpp
torch::Tensor causal_conv1d_fn(...) {
  // 1. 检查输入
  check_tensor(x, "x", "causal_conv1d_fn");
  
  // 2. 创建输出张量
  torch::Tensor output = at::empty(x.sizes(), x.options());
  
  // 3. 获取 NPU stream
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  
  // 4. 创建 ACL tensor
  aclTensor* x_tensor = nullptr;
  create_acltensor(&x_tensor, x);
  // ... 其他张量
  
  // 5. 获取 workspace 大小
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnCausalConv1dGetWorkspaceSize(..., &workspace_size, &executor);
  
  // 6. 分配 workspace
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspace_addr, workspace_size, ...);
  }
  
  // 7. 执行算子
  aclnnCausalConv1d(workspace_addr, workspace_size, executor, stream);
  
  // 8. 同步
  aclrtSynchronizeStream(stream);
  
  // 9. 释放资源
  aclDestroyTensor(x_tensor);
  // ...
  
  return output;
}
```

### 3.5 算子接口实现 (`ops_api.cpp`)

**文件位置**: `xllm/core/kernels/ops_api.cpp`

**作用**: 将参数结构体拆分，调用 NPU 层实现。

```cpp
torch::Tensor causal_conv1d_fn(CausalConv1dFnParams& params) {
#if defined(USE_NPU)
  return npu::causal_conv1d_fn(params.x,
                               params.weight,
                               params.bias,
                               params.activation_mode,
                               params.conv_state,
                               params.has_initial_state,
                               params.cache_indices,
                               params.query_start_loc,
                               params.pad_slot_id);
#else
  NOT_IMPLEMENTED();
#endif
}
```

### 3.6 自定义算子源码 (`third_party/xllm_ops/`)

**目录结构**:

```
third_party/xllm_ops/xllm_ops/causal_conv1d/
├── CMakeLists.txt           # 构建配置
├── causal_conv1d_tiling.h   # Tiling 数据结构定义
├── causal_conv1d.h          # Kernel 实现头文件
├── causal_conv1d.cpp        # Kernel 入口函数
└── causal_conv1d_cpu.cpp    # Tiling 函数和算子定义
```

**各文件作用**:

| 文件 | 作用 |
|------|------|
| `causal_conv1d_tiling.h` | 定义 Tiling 数据结构，包含 dim、batch、seqLen 等参数 |
| `causal_conv1d.h` | 定义 Kernel 类，实现 Init 和 Process 方法 |
| `causal_conv1d.cpp` | Kernel 入口函数，根据数据类型调用模板实例 |
| `causal_conv1d_cpu.cpp` | Tiling 函数（计算分块参数）和算子注册定义 |
| `CMakeLists.txt` | 构建配置，指定编译选项和源文件 |

---

## 四、数据布局对比

### 4.1 Prefill vs Decode 数据布局

| 阶段 | 输入布局 | 权重布局 | 输出布局 |
|------|----------|----------|----------|
| **Prefill (新)** | `[cu_seqlen, dim]` | `[width, dim]` | `[cu_seqlen, dim]` |
| **Decode (已有)** | `[batch, dim, 1]` | `[dim, width]` | `[batch, dim]` |

### 4.2 关键概念解释

- **cu_seqlen**: 累积序列长度，所有序列的 token 拼接在一起的总长度
- **dim**: 隐藏层维度
- **width**: 卷积核大小，qwen3-next 中为 4
- **batch**: 批次大小

---

## 五、构建流程

### 5.1 编译 xllm 时的自动流程

```
xllm 构建开始
    │
    ▼
CMakeLists.txt 检测到 USE_NPU=ON
    │
    ▼
执行 third_party/xllm_ops/build.sh
    │
    ▼
编译所有算子（包括 causal_conv1d）
    │
    ▼
生成 xllm_ops.run 安装包
    │
    ▼
安装到 $NPU_HOME_PATH/opp/vendors/xllm/
    │
    ▼
xllm 链接 cust_opapi 库
    │
    ▼
构建完成
```

### 5.2 环境变量要求

| 环境变量 | 说明 |
|----------|------|
| `ASCEND_HOME_PATH` 或 `ASCEND_CANN_PACKAGE_PATH` | CANN 安装路径 |
| `NPU_HOME_PATH` | NPU 工具路径 |
| `ATB_HOME_PATH` | ATB 库路径 |

---

## 六、实施步骤

### 第一阶段：参数结构和接口定义 ✅ 已完成
- [x] `param.h` 新增 `CausalConv1dFnParams` 结构体
- [x] `ops_api.h` 新增 `causal_conv1d_fn` 函数声明

### 第二阶段：ACLNN 层实现 ✅ 已完成
- [x] `xllm_ops/xllm_ops_api.h` 新增函数声明
- [x] `xllm_ops/causal_conv1d_fn.cpp` 新增 ACLNN 调用实现
- [x] `xllm_ops/CMakeLists.txt` 添加新源文件
- [x] `ops_api.cpp` 实现 `causal_conv1d_fn`

### 第二.五阶段：自定义算子源码 ✅ 已完成
- [x] 将算子源码添加到 `third_party/xllm_ops/xllm_ops/causal_conv1d/`
- [x] `causal_conv1d_tiling.h` - Tiling 数据结构
- [x] `causal_conv1d.h` - Kernel 实现头文件
- [x] `causal_conv1d.cpp` - Kernel 入口
- [x] `causal_conv1d_cpu.cpp` - Tiling 和算子定义
- [x] `CMakeLists.txt` - 构建配置
- [x] 编译时会随 `xllm_ops` 一起自动编译

### 第三阶段：集成调用 ✅ 已完成
- [x] 修改 `qwen3_next_gated_delta_net.cpp` 的 prefill 逻辑
- [x] 处理数据布局转换

#### 集成代码说明

**文件位置**: `xllm/core/layers/common/qwen3_next_gated_delta_net.cpp`

**修改内容**:

```cpp
if (attn_metadata.is_prefill) {
#if defined(USE_NPU)
    xllm::kernel::CausalConv1dFnParams conv_params;
    
    // 数据布局转换: [batch, dim, seq_len] -> [cu_seqlen, dim]
    torch::Tensor mixed_qkv_transposed = mixed_qkv.transpose(1, 2).contiguous();
    int64_t batch = mixed_qkv_transposed.size(0);
    int64_t dim = mixed_qkv_transposed.size(2);
    
    // 设置参数
    conv_params.x = mixed_qkv_transposed.view({-1, dim});
    conv_params.weight = conv_weight.transpose(0, 1).contiguous();  // [dim, width] -> [width, dim]
    conv_params.bias = std::nullopt;
    conv_params.conv_state = conv_cache;
    conv_params.has_initial_state = torch::zeros({batch}, torch::kBool).to(device);
    conv_params.cache_indices = input_params.block_tables.select(1, 0);
    conv_params.query_start_loc = attn_metadata.q_cu_seq_lens;
    conv_params.activation_mode = 1;  // silu
    conv_params.pad_slot_id = -1;
    
    // 调用算子
    auto conv_output = xllm::kernel::causal_conv1d_fn(conv_params);
    
    // 数据布局转换: [cu_seqlen, dim] -> [batch, dim, seq_len]
    mixed_qkv = conv_output.view({batch, seq_len, dim}).transpose(1, 2).contiguous();
#else
    // 原始 torch::conv1d 实现 (非 NPU 平台)
    ...
#endif
}
```

**数据布局转换说明**:

| 步骤 | 张量 | 形状 |
|------|------|------|
| 输入 | `mixed_qkv` | `[batch, dim, seq_len]` |
| 转换后 | `mixed_qkv_transposed` | `[batch, seq_len, dim]` |
| 展平后 | `conv_params.x` | `[cu_seqlen, dim]` |
| 权重转换 | `conv_params.weight` | `[width, dim]` |
| 输出 | `conv_output` | `[cu_seqlen, dim]` |
| 最终输出 | `mixed_qkv` | `[batch, dim, seq_len]` |

### 第四阶段：测试验证 ✅ 测试用例已完成
- [x] 创建测试用例 `third_party/xllm_ops/test/cpp_test/causal_conv1d_test.cpp`
- [x] 更新 CMakeLists.txt 添加测试目标
- [ ] 编译测试
- [ ] 功能正确性测试
- [ ] 性能对比测试

#### 测试用例说明

**文件位置**: `third_party/xllm_ops/test/cpp_test/causal_conv1d_test.cpp`

**测试场景**:

| 测试用例 | 说明 |
|----------|------|
| `BasicCorrectness` | 基本正确性测试，验证有 bias + silu 激活 |
| `WithoutBias` | 无 bias 测试 |
| `WithInitialState` | 有初始状态测试 |
| `NoActivation` | 无激活函数测试 |

---

## 七、编译和测试指南

### 7.1 环境要求

| 环境变量 | 说明 |
|----------|------|
| `ASCEND_CANN_PACKAGE_PATH` 或 `ASCEND_HOME_PATH` | CANN 安装路径 |
| `NPU_HOME_PATH` | NPU 工具路径 |
| `ATB_HOME_PATH` | ATB 库路径 |

### 7.2 编译步骤

#### 步骤 1：编译自定义算子

```bash
# 进入 xllm_ops 目录
cd third_party/xllm_ops

# 编译所有算子（包括 causal_conv1d）
bash build.sh -c ascend910b

# 或者只编译 causal_conv1d 算子
bash build.sh -n causal_conv1d -c ascend910b
```

编译完成后，算子会安装到 `$HOME/Ascend/ascend-toolkit/latest/opp/vendors/xllm/` 目录。

#### 步骤 2：编译 xllm

```bash
# 回到 xllm 根目录
cd ../..

# 创建 build 目录
mkdir -p build && cd build

# 配置 CMake
cmake .. -DUSE_NPU=ON -DDEVICE_TYPE=USE_A2

# 编译
make -j$(nproc)
```

### 7.3 运行测试

#### 运行算子单元测试

```bash
cd third_party/xllm_ops/test/cpp_test
mkdir -p build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make causal_conv1d_test
./causal_conv1d_test
```

#### 运行端到端测试

```bash
# 在 xllm build 目录下
./xllm_test --model qwen3-next --test prefill
```

### 7.4 验证方法

1. **功能正确性验证**：
   - 对比新算子输出与原始 `torch::conv1d` 输出
   - 误差应小于 1e-2

2. **性能验证**：
   - 使用 `nsys` 或 `msprof` 工具分析性能
   - 对比 prefill 阶段延迟

---

## 八、风险与注意事项

1. **数据布局差异**：PR 中使用 `[cu_seqlen, dim]` 布局，当前代码可能需要调整
2. **状态管理**：需要正确处理 `conv_state` 的读写
3. **边界条件**：处理 `has_initial_state` 为 false 的情况
4. **兼容性**：确保 decode 阶段不受影响
5. **CANN 依赖**：确保 CANN 环境正确配置

---

## 九、参考资料

- vllm-ascend PR #6661: https://github.com/vllm-project/vllm-ascend/pull/6661
- CANN 自定义算子开发指南
- xllm_ops 仓库: https://gitcode.com/xLLM-AI/xllm_ops.git

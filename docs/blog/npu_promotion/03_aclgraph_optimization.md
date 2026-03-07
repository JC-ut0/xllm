# ACLGraph：NPU推理加速的秘密武器

> 本文深入解析xLLM中的ACLGraph优化技术，揭示如何在NPU上实现8%-10%的性能提升。

## 一、NPU推理的性能瓶颈

### 1.1 传统执行模式

在传统的推理执行模式中，CPU和NPU之间的交互方式如下：

```
┌─────────┐                    ┌─────────┐
│   CPU   │                    │   NPU   │
│ (Host)  │                    │(Device) │
└────┬────┘                    └────┬────┘
     │                              │
     │  1. 提交小任务（Kernel 1）    │
     │─────────────────────────────▶│
     │                              │ 执行
     │  2. 等待完成                 │◀────┐
     │◀─────────────────────────────│     │
     │                              │     │
     │  3. 提交小任务（Kernel 2）    │     │ 空泡
     │─────────────────────────────▶│     │
     │                              │ 执行 │
     │  4. 等待完成                 │◀────┘
     │◀─────────────────────────────│
     │                              │
     │  ... 重复多次 ...             │
     │                              │
```

**问题分析**：

1. **CPU密集小任务提交**：每个Kernel都需要CPU单独提交
2. **NPU频繁启动小Kernel**：启动开销累积
3. **大量计算空泡**：CPU和NPU之间的等待时间

### 1.2 性能分析

以典型的Decode阶段为例，每个token生成都需要：

```
单次Decode迭代：
├── Kernel 1: RMSNorm        ~0.1ms
├── Kernel 2: Attention      ~0.5ms
├── Kernel 3: MLP            ~0.8ms
├── Kernel 4: Sampling       ~0.05ms
└── Kernel 5: Logits处理     ~0.1ms

传统模式下：
- Kernel启动开销：~0.05ms × 5 = 0.25ms
- 实际计算时间：~1.55ms
- 开销占比：~14%
```

当batch size增大时，Kernel执行时间增加，但启动开销基本不变，占比下降。但在小batch场景下，启动开销成为显著瓶颈。

## 二、ACLGraph技术原理

### 2.1 什么是ACLGraph？

ACLGraph是NPU推出的类似CUDA Graph的图模式方案，其核心思想是：

**将多个Kernel操作录制为一个计算图，一次性提交给NPU执行**

```
┌─────────┐                    ┌─────────┐
│   CPU   │                    │   NPU   │
│ (Host)  │                    │(Device) │
└────┬────┘                    └────┬────┘
     │                              │
     │  1. 一次性提交整个计算图       │
     │─────────────────────────────▶│
     │                              │ 执行 Kernel 1
     │                              │ 执行 Kernel 2
     │  2. NPU内部流式执行          │ 执行 Kernel 3
     │                              │ 执行 Kernel 4
     │                              │ 执行 Kernel 5
     │◀─────────────────────────────│
     │                              │
```

### 2.2 核心优势

**降低Host启动开销**
- CPU只需一次提交
- 减少系统调用次数
- 降低CPU负载

**减少NPU气泡**
- NPU内部连续执行
- Kernel之间无缝衔接
- 提升计算密度

**提升整体吞吐**
- Decode阶段吞吐提升8%-10%
- 小batch场景收益更明显
- 与其他优化技术正交

### 2.3 与CUDA Graph对比

| 特性 | CUDA Graph | ACLGraph |
|------|------------|----------|
| 平台 | NVIDIA GPU | NPU |
| 录制方式 | 显式录制 | 显式录制 |
| 动态参数支持 | 有限支持 | 参数化支持 |
| 内存管理 | 固定地址 | 可扩张内存池 |
| 适用阶段 | Decode | Decode |

## 三、xLLM中的ACLGraph实现

### 3.1 动态维度参数化

大模型推理的一个核心挑战是**动态shape**：batch size和序列长度在运行时才能确定。

ACLGraph要求计算图结构固定，如何支持动态维度？

**xLLM的解决方案：参数化**

将关键动态维度作为整图输入参数：

```cpp
// 动态参数定义
struct GraphParams {
    int32_t batch_size;      // 批大小
    int32_t max_seq_len;     // 最大序列长度
    int32_t* block_table;    // Block表指针
    // ... 其他参数
};

// 图编译时：使用参数计算shape
int32_t block_table_size = batch_size * (max_seq_len / block_size);

// 图执行时：传入实际参数
GraphParams params;
params.batch_size = actual_batch_size;
params.max_seq_len = actual_seq_len;
graph.execute(params);
```

**关键实现**：

1. **Block Table动态计算**
   ```
   block_table_size = batch_size × (max_seq_len / block_size)
   ```

2. **Attention Mask动态生成**
   ```
   mask_shape = [batch_size, num_heads, max_seq_len, max_seq_len]
   ```

3. **输出Tensor动态shape**
   ```
   output_shape = [batch_size, seq_len, hidden_size]
   ```

### 3.2 多shape复用显存池

不同batch size和序列长度需要不同的显存空间，如何避免浪费？

**xLLM的解决方案：可扩张显存池**

```
┌──────────────────────────────────────────────────────────────┐
│                     可扩张显存池                              │
│                                                              │
│  基地址: 0x7F0000000000                                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    显存池                               │  │
│  │                                                        │  │
│  │  Shape 1 (bs=1):  Offset=0,     Size=100MB            │  │
│  │  ├────────────────────────────────────┤               │  │
│  │                                                        │  │
│  │  Shape 2 (bs=8):  Offset=0,     Size=300MB            │  │
│  │  ├──────────────────────────────────────────────────┤  │  │
│  │                                                        │  │
│  │  Shape 3 (bs=32): Offset=0,     Size=800MB            │  │
│  │  ├────────────────────────────────────────────────────────┤
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘

特点：
- 多shape复用基地址
- 不同shape使用不同偏移量
- 按需扩张，避免预分配浪费
```

**实现细节**：

```cpp
class ExpandableMemoryPool {
public:
    // 获取指定shape的内存
    void* get_memory(int64_t batch_size, int64_t seq_len) {
        int64_t required_size = calculate_size(batch_size, seq_len);
        
        // 如果需要，扩张池
        if (required_size > current_size_) {
            expand_pool(required_size);
        }
        
        return base_address_;  // 始终返回基地址
    }
    
private:
    void* base_address_;
    int64_t current_size_;
};
```

### 3.3 关键代码解析

**图录制阶段**：

```cpp
// 录制ACLGraph
void record_acl_graph(GraphExecutor& executor) {
    // 开始录制
    aclGraph graph;
    aclGraphExec graph_exec;
    
    aclCreateGraph(&graph);
    
    // 录制所有Kernel
    for (auto& kernel : decode_kernels_) {
        kernel.launch(stream);  // 录制而非执行
    }
    
    // 实例化图
    aclInstantiateGraph(&graph_exec, graph);
    
    // 保存图执行器
    graph_exec_ = graph_exec;
}
```

**图执行阶段**：

```cpp
// 执行ACLGraph
void execute_acl_graph(const GraphParams& params) {
    // 设置动态参数
    aclSetGraphExecParam(graph_exec_, params);
    
    // 一次性执行整个图
    aclLaunchGraph(graph_exec_, stream_);
}
```

**动态参数传递**：

```cpp
// 设置动态参数
void aclSetGraphExecParam(aclGraphExec graph_exec, 
                          const GraphParams& params) {
    // 设置batch size
    aclSetGraphExecAttrValue(graph_exec, "batch_size", 
                             params.batch_size);
    
    // 设置max_seq_len
    aclSetGraphExecAttrValue(graph_exec, "max_seq_len", 
                             params.max_seq_len);
    
    // 设置block_table指针
    aclSetGraphExecInput(graph_exec, "block_table", 
                         params.block_table);
}
```

## 四、性能测试

### 4.1 测试环境

| 配置项 | 规格 |
|--------|------|
| NPU型号 | A2 / A3 |
| 驱动版本 | HDK Driver 25.2.0 |
| 测试模型 | Qwen3-0.6B, Qwen3-1.7B |
| 测试场景 | Decode阶段 |

### 4.2 测试结果

**Qwen3-0.6B性能对比**：

| Batch Size | 关闭ACLGraph | 开启ACLGraph | 提升幅度 |
|------------|-------------|-------------|---------|
| 1 | 120 tokens/s | 132 tokens/s | +10.0% |
| 4 | 380 tokens/s | 415 tokens/s | +9.2% |
| 8 | 650 tokens/s | 705 tokens/s | +8.5% |
| 16 | 1100 tokens/s | 1190 tokens/s | +8.2% |

**Qwen3-1.7B性能对比**：

| Batch Size | 关闭ACLGraph | 开启ACLGraph | 提升幅度 |
|------------|-------------|-------------|---------|
| 1 | 85 tokens/s | 93 tokens/s | +9.4% |
| 4 | 280 tokens/s | 306 tokens/s | +9.3% |
| 8 | 480 tokens/s | 522 tokens/s | +8.8% |
| 16 | 820 tokens/s | 890 tokens/s | +8.5% |

### 4.3 性能分析

**收益来源分析**：

```
单次Decode迭代时间分解：

传统模式：
├── Kernel启动开销: 0.25ms (14%)
├── 实际计算时间:   1.55ms (86%)
└── 总计:          1.80ms

ACLGraph模式：
├── 图启动开销:     0.05ms (3%)
├── 实际计算时间:   1.55ms (89%)
├── 参数设置开销:   0.05ms (3%)
├── 其他开销:       0.10ms (5%)
└── 总计:          1.75ms

提升: (1.80 - 1.75) / 1.80 = 2.8% (单次迭代)

累计效果：
- 小batch: 启动开销占比高，收益明显
- 大batch: 启动开销占比低，收益降低
```

**性能曲线**：

```
吞吐提升幅度
    │
12% ┤
    │    ●
10% ┤         ●
    │              ●
 8% ┤                   ●
    │                        ●
 6% ┤
    │
 4% ┤
    │
 2% ┤
    └───┬────┬────┬────┬────┬──
        1    4    8   16   32   Batch Size
```

## 五、使用指南

### 5.1 如何开启

在xLLM启动参数中添加：

```bash
/path/to/xllm \
  --model /path/to/model \
  --devices="npu:0" \
  --enable_graph=true    # 开启ACLGraph
```

### 5.2 注意事项

**适用阶段**
- ACLGraph仅用于Decode阶段
- Prefill阶段仍使用传统模式
- 原因：Prefill阶段shape变化大，图模式收益有限

**Kernel支持检查**
- 并非所有Kernel都支持动态维度参数化
- 为新模型添加支持时需要检查Kernel实现
- 不支持的Kernel会回退到传统模式

**支持的模型列表**
- Qwen2/2.5/3系列
- Llama2/3系列
- GLM系列
- DeepSeek系列
- 更多模型持续适配中

### 5.3 调试技巧

**查看图执行状态**：

```bash
# 启用详细日志（glog的VLOG级别）
export GLOG_v=2  # 输出 VLOG(2) 及以下的详细日志

# 查看图编译信息
grep "ACLGraph" /path/to/log
```

**性能分析**：

xLLM提供两种性能分析方式：

**方式一：内置Profile功能**

```bash
# 启动时启用profile参数
./xllm --model /path/to/model --devices="npu:0" \
  --enable_profile_step_time=true \
  --enable_profile_kv_blocks=true

# Profile数据输出到文件：
# - profile_prefill_step_time_xxx.txt
# - profile_decode_step_time_xxx.txt
```

**方式二：NPU级别分析（MSPTI）**

MSPTI (NPU Performance Tools Interface) 是NPU的性能分析工具，类似NVIDIA的NVTX。

```bash
# 1. 编译时启用MSPTI（默认关闭）
# USE_MSPTI选项定义在项目根目录 CMakeLists.txt
cmake -DUSE_MSPTI=ON ..
make -j$(nproc)

# 2. 运行程序，日志输出MSPTI数据
./xllm --model /path/to/model --devices="npu:0" 2>&1 | tee mspti.log

# 3. 转换为Chrome Trace格式
python tools/npu_timeline.py -i mspti.log -o timeline.json

# 4. 在Chrome浏览器中打开 chrome://tracing/ 加载timeline.json
```

**MSPTI相关文件**：

| 文件 | 作用 |
|------|------|
| `CMakeLists.txt` | 定义 `USE_MSPTI` 编译选项 |
| `xllm/core/common/mspti_helper.cpp` | MSPTI实现代码 |
| `xllm/core/common/mspti_helper.h` | MSPTI头文件 |
| `tools/npu_timeline.py` | 日志转换工具 |

## 六、未来规划

### 6.1 近期计划

**支持更多模型**
- 扩展Kernel动态参数支持
- 优化图编译流程
- 减少适配工作量

**性能持续优化**
- 减少参数设置开销
- 优化图执行调度
- 提升大batch收益

### 6.2 中期计划

**MoE模型适配**
- 支持Attention DP和FFN EP
- 适配不同shape的通信操作
- 实现MoE专用图模式

**与投机推理结合**
- 图模式下的投机推理
- 多流并行优化
- 进一步提升吞吐

### 6.3 长期愿景

**全图模式**
- Prefill阶段图模式支持
- 动态shape通用解决方案
- 与编译器深度集成

## 七、小结

本文深入介绍了xLLM中的ACLGraph优化技术：

- **问题背景**：传统模式下Kernel启动开销成为瓶颈
- **技术原理**：通过图模式一次性提交，减少启动开销
- **实现细节**：动态维度参数化、可扩张显存池
- **性能收益**：Decode阶段吞吐提升8%-10%

ACLGraph是xLLM针对NPU的重要优化之一，与后续文章将介绍的xTensor显存管理、全局KV Cache管理等技术协同工作，共同实现高性能推理。

**下一篇文章预告**：《xTensor：让NPU显存利用率提升30%的秘密》，将深入介绍xLLM的动态显存管理技术。

---

> **参考资料**：
> - [ACLGraph文档](https://xllm.readthedocs.io/zh-cn/latest/features/acl_graph.html)
> - [xLLM技术报告](https://arxiv.org/abs/2510.14686)
> - [NPU开发文档](https://www.hiascend.com/)

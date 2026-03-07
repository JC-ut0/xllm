# xTensor：让NPU显存利用率提升30%的秘密

> 本文深入解析xLLM的xTensor显存管理框架，揭示如何通过动态内存池实现高效显存利用。

## 一、LLM推理的显存挑战

### 1.1 显存占用分析

大模型推理过程中，显存主要被以下几个部分占用：

```
┌──────────────────────────────────────────────────────────────┐
│                    显存占用分布                               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    模型权重                             │  │
│  │  Qwen3-8B: ~16GB (FP16)                               │  │
│  │  DeepSeek-V3: ~1.3TB (FP8, 需要分布式)                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    KV Cache                            │  │
│  │  与batch size和序列长度成正比                           │  │
│  │  通常是显存占用的主要部分                                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    激活值                               │  │
│  │  前向计算中间结果                                       │  │
│  │  与batch size和序列长度相关                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    临时缓冲区                           │  │
│  │  采样、解码等临时数据                                   │  │
│  │  生命周期短，频繁分配释放                                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**显存占用公式**：

```
总显存 = 模型权重 + KV Cache + 激活值 + 临时缓冲区

KV Cache = 2 × num_layers × batch_size × seq_len × num_heads × head_dim × dtype_size

激活值 ≈ batch_size × seq_len × hidden_size × num_layers × intermediate_size
```

### 1.2 传统显存管理的问题

**静态分配浪费**

```python
# 传统方式：为最大shape预分配
max_batch_size = 256
max_seq_len = 8192

# 即使实际只用了batch_size=1，也占用最大空间
kv_cache = allocate_kv_cache(max_batch_size, max_seq_len)
# 浪费率：(256-1)/256 = 99.6%
```

**内存碎片严重**

```
多次分配释放后的内存布局：

┌────┬────┬────┬────┬────┬────┬────┬────┐
│已用│空闲│已用│空闲│已用│空闲│已用│空闲│
│8GB │2GB │4GB │1GB │6GB │3GB │2GB │4GB │
└────┴────┴────┴────┴────┴────┴────┴────┘

总空闲: 10GB
最大连续空闲: 4GB
无法分配8GB的连续空间！
```

**动态shape适配困难**

```
场景：处理不同长度的请求

请求1: seq_len = 100
请求2: seq_len = 1000
请求3: seq_len = 8000

传统方式需要：
1. 为每个请求单独分配
2. 或者预分配最大长度
3. 都存在效率问题
```

## 二、xTensor设计理念

### 2.1 核心思想

xTensor采用**物理内存页池预分配 + 虚拟地址连续性映射**的方法：

```
┌──────────────────────────────────────────────────────────────┐
│                     xTensor架构                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              虚拟地址空间（连续）                        │  │
│  │  0x7F0000000000 ──────────────────────▶ 0x7F0080000000 │  │
│  │  │         │         │         │         │             │  │
│  │  │  Page 0 │  Page 1 │  Page 2 │  Page 3 │  ...        │  │
│  │  │         │         │         │         │             │  │
│  └──┼─────────┼─────────┼─────────┼─────────┼─────────────┘  │
│     │         │         │         │         │                │
│     │ 映射    │ 映射    │ 映射    │ 映射    │                │
│     ▼         ▼         ▼         ▼         ▼                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              物理内存页池（离散）                        │  │
│  │  Page A   Page B   Page C   Page D   Page E   ...      │  │
│  │  (4MB)    (4MB)    (4MB)    (4MB)    (4MB)             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**关键优势**：

1. **虚拟地址连续**：对上层呈现连续地址空间
2. **物理页离散**：底层物理内存可以不连续
3. **按需映射**：只映射实际使用的物理页

### 2.2 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                    xTensor组件架构                           │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   xTensorManager                        │  │
│  │  - 管理全局显存池                                       │  │
│  │  - 协调内存分配与回收                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                  │
│           ┌───────────────┼───────────────┐                  │
│           ▼               ▼               ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ PhyPagePool │  │   xTensor   │  │   Mapper    │          │
│  │  物理页池    │  │  张量抽象   │  │  地址映射   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │               │                   │
│         ▼                ▼               ▼                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  PhyPage    │  │  TensorView │  │ VirtualMem  │          │
│  │  物理页管理  │  │  张量视图   │  │ 虚拟内存管理│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 三、关键技术实现

### 3.1 物理页池管理

**预分配策略**

```cpp
class PhyPagePool {
public:
    PhyPagePool(size_t page_size, size_t num_pages) {
        // 预分配所有物理页
        for (size_t i = 0; i < num_pages; ++i) {
            void* page = device_malloc(page_size);
            free_pages_.push(page);
        }
    }
    
    PhyPage* allocate_page() {
        if (free_pages_.empty()) {
            return nullptr;  // 或者触发扩容
        }
        PhyPage* page = free_pages_.top();
        free_pages_.pop();
        used_pages_.insert(page);
        return page;
    }
    
    void free_page(PhyPage* page) {
        used_pages_.erase(page);
        free_pages_.push(page);
    }
    
private:
    std::stack<PhyPage*> free_pages_;
    std::set<PhyPage*> used_pages_;
};
```

**页复用机制**

```cpp
// 页状态管理
enum class PageState {
    FREE,        // 空闲，可分配
    ALLOCATED,   // 已分配，使用中
    REUSABLE,    // 可复用（如KV Cache中的历史页）
    PINNED       // 固定，不可移动
};

// 智能页选择
PhyPage* select_best_page(PageState preferred_state) {
    // 优先选择可复用页（避免重新分配）
    if (preferred_state == PageState::REUSABLE) {
        for (auto& page : reusable_pages_) {
            if (page.can_reuse()) {
                return page;
            }
        }
    }
    
    // 其次选择空闲页
    return allocate_from_free_pool();
}
```

**异步预映射**

```cpp
// 预测即将需要的页，提前映射
class AsyncPreMapper {
public:
    void predict_and_premap(const Batch& batch) {
        // 根据当前batch预测下一batch需要的页数
        size_t predicted_pages = predict_page_count(batch);
        
        // 异步预映射
        for (size_t i = 0; i < predicted_pages; ++i) {
            PhyPage* page = pool_->allocate_page();
            premapped_queue_.push(page);
        }
    }
    
    PhyPage* get_premapped_page() {
        if (premapped_queue_.empty()) {
            return pool_->allocate_page();  // 同步分配
        }
        PhyPage* page = premapped_queue_.front();
        premapped_queue_.pop();
        return page;
    }
};
```

### 3.2 虚拟地址映射

**连续虚拟地址空间**

```cpp
class VirtualMemoryManager {
public:
    VirtualMemoryManager(size_t total_size) {
        // 申请连续虚拟地址空间（不占用物理内存）
        base_addr_ = virtual_alloc(total_size);
        total_size_ = total_size;
    }
    
    void* map_pages(size_t offset, const std::vector<PhyPage*>& pages) {
        // 将物理页映射到虚拟地址
        void* virt_addr = (char*)base_addr_ + offset;
        
        for (size_t i = 0; i < pages.size(); ++i) {
            void* page_addr = (char*)virt_addr + i * page_size_;
            map_physical_to_virtual(pages[i], page_addr);
        }
        
        return virt_addr;
    }
    
private:
    void* base_addr_;
    size_t total_size_;
};
```

**地址安全复用**

```cpp
// 确保地址复用的安全性
class SafeAddressReuse {
public:
    void* allocate(size_t size) {
        // 查找可复用的地址范围
        AddressRange range = find_reusable_range(size);
        
        if (range.valid()) {
            // 验证该范围已完全释放
            assert(all_pages_freed(range));
            return range.addr;
        }
        
        // 分配新地址
        return allocate_new_range(size);
    }
    
private:
    bool all_pages_freed(const AddressRange& range) {
        for (auto& page : get_pages_in_range(range)) {
            if (page.state() != PageState::FREE) {
                return false;
            }
        }
        return true;
    }
};
```

### 3.3 NPU算子适配

**虚拟地址化FlashMLA**

```cpp
// NPU上的FlashMLA算子适配
class NpuFlashMLA {
public:
    void forward(
        const xTensor& query,      // 虚拟地址连续
        const xTensor& key_cache,  // 虚拟地址连续
        const xTensor& value_cache,
        xTensor& output
    ) {
        // 算子内部处理虚拟地址到物理页的映射
        // 上层无需关心物理内存布局
        
        // 获取物理页信息
        auto& key_pages = key_cache.physical_pages();
        auto& value_pages = value_cache.physical_pages();
        
        // 调用NPU内核
        npu_flash_mla_kernel(
            query.data(),           // 虚拟地址
            key_cache.data(),       // 虚拟地址
            value_cache.data(),     // 虚拟地址
            key_pages.page_table(), // 物理页表
            value_pages.page_table(),
            output.data()
        );
    }
};
```

**自定义算子适配示例**

```cpp
// 为新算子添加xTensor支持
class CustomAttention {
public:
    void forward(const xTensor& input, xTensor& output) {
        // 1. 检查输入是否为xTensor
        if (input.is_xtensor()) {
            // 2. 获取虚拟地址（连续）
            void* virt_addr = input.data();
            
            // 3. 获取物理页表（如果需要）
            auto& page_table = input.page_table();
            
            // 4. 调用支持虚拟地址的内核
            custom_attention_kernel_virt(
                virt_addr,
                page_table.data(),
                page_table.size(),
                output.data()
            );
        } else {
            // 回退到传统方式
            custom_attention_kernel_trad(input.data(), output.data());
        }
    }
};
```

## 四、性能收益

### 4.1 内存利用率提升

**对比测试**：

```
测试场景：Qwen3-8B，NPU A2，64GB显存

传统方式：
├── 模型权重: 16GB
├── KV Cache预分配: 40GB (为最大batch预分配)
├── 激活值缓冲: 4GB
├── 实际可用: 4GB
└── 最大batch size: 128

xTensor方式：
├── 模型权重: 16GB
├── KV Cache动态分配: 按需
├── 物理页池: 42GB
├── 激活值缓冲: 4GB
├── 实际可用: 2GB (页池管理开销)
└── 最大batch size: 180

提升: (180-128)/128 = 40.6%
```

**内存利用率曲线**：

```
内存利用率
    │
100%┤                    ──────── xTensor
    │               ─────
    │          ─────
 80%┤     ─────
    │─────
    │     ╲
 60%┤      ╲ ───────────────────── 传统方式
    │       ╲
    │        ╲
 40%┤         ╲
    │          ╲
    │           ╲
 20%┤            ╲
    └───┬────┬────┬────┬────┬──
       10   20   40   80  160  Batch Size
```

### 4.2 延迟降低

**分配延迟对比**：

| 操作 | 传统方式 | xTensor | 降低幅度 |
|------|----------|---------|---------|
| 首次分配1GB | 50ms | 5ms | 90% |
| 重复分配1GB | 50ms | 0.1ms | 99.8% |
| 释放1GB | 10ms | 0.5ms | 95% |

**原因分析**：

```
传统方式：
├── 搜索空闲块: O(n) 或 O(log n)
├── 可能需要整理碎片
├── 系统调用开销
└── 总延迟: 高

xTensor：
├── 从页池取页: O(1)
├── 虚拟地址映射: O(1)
├── 无需整理碎片
└── 总延迟: 极低
```

### 4.3 实际案例

**Qwen3-8B显存占用对比**：

```
场景：batch_size=64, seq_len=4096

传统方式显存占用：
├── 模型权重: 16GB
├── KV Cache: 28GB
├── 激活值: 8GB
├── 碎片浪费: 4GB
└── 总计: 56GB

xTensor显存占用：
├── 模型权重: 16GB
├── KV Cache: 28GB
├── 激活值: 8GB
├── 管理开销: 0.5GB
└── 总计: 52.5GB

节省: 3.5GB (6.25%)
```

**支持的最大batch size提升**：

```
NPU A2 (64GB显存) + Qwen3-8B

传统方式最大batch size: 128
xTensor最大batch size: 180
提升: 40.6%
```

## 五、最佳实践

### 5.1 参数调优

**显存利用率上限**

```bash
# 设置显存利用率上限
--max_memory_utilization=0.86

# 建议值：
# - NPU: 0.85-0.90
# - GPU: 0.80-0.85
# - 留出缓冲空间避免OOM
```

**Block Size选择**

```bash
# KV Cache块大小
--block_size=128

# 选择建议：
# - 小block (16-32): 更细粒度管理，适合变长请求
# - 中block (64-128): 平衡管理开销和利用率
# - 大block (256-512): 减少管理开销，适合固定长度
```

**页池大小配置**

```bash
# 物理页池大小（可选，默认自动）
--phy_page_pool_size=4096  # 单位：页数

# 计算方式：
# 页数 = (总显存 - 模型权重) / page_size
```

### 5.2 监控与调试

**显存使用监控**

```python
# 通过Prometheus监控
# 指标：xllm_memory_used_bytes, xllm_memory_total_bytes

import prometheus_client

# 获取显存使用情况
memory_used = prometheus_client.get_metric('xllm_memory_used_bytes')
memory_total = prometheus_client.get_metric('xllm_memory_total_bytes')
utilization = memory_used / memory_total

print(f"显存利用率: {utilization:.2%}")
```

**内存泄漏排查**

```bash
# 启用内存跟踪
export XLLM_ENABLE_MEMORY_TRACKING=1

# 查看内存分配日志
grep "xTensor" /path/to/log | grep "allocate\|free"

# 分析内存泄漏
python tools/analyze_memory_leak.py --log /path/to/log
```

**性能分析**

```python
# 分析xTensor性能
from xllm import Profiler

profiler = Profiler()
profiler.start()

# ... 运行推理 ...

stats = profiler.stop()
print(f"页分配次数: {stats.page_allocations}")
print(f"页复用次数: {stats.page_reuses}")
print(f"平均分配延迟: {stats.avg_alloc_latency}ms")
```

## 六、与其他组件协同

### 6.1 与ACLGraph协同

```cpp
// xTensor为ACLGraph提供可扩张显存池
class ACLGraphMemoryPool {
public:
    void* get_memory_for_shape(int batch_size, int seq_len) {
        // 使用xTensor的虚拟地址空间
        size_t size = calculate_size(batch_size, seq_len);
        
        // xTensor保证基地址不变，只改变偏移
        return xtensor_manager_->allocate(size);
    }
};
```

### 6.2 与KV Cache管理协同

```cpp
// KV Cache使用xTensor管理显存
class KVCacheManager {
public:
    Block allocate_block() {
        // 从xTensor获取内存
        void* memory = xtensor_->allocate(block_size_);
        
        // 创建Block对象
        return Block(memory, block_size_);
    }
    
    void free_block(Block& block) {
        // 释放到xTensor
        xtensor_->free(block.memory());
    }
};
```

## 七、小结

本文深入介绍了xLLM的xTensor显存管理框架：

- **问题背景**：传统显存管理存在浪费、碎片、动态适配困难
- **核心设计**：物理页池预分配 + 虚拟地址连续映射
- **关键技术**：页复用机制、异步预映射、算子适配
- **性能收益**：内存利用率提升30%+，分配延迟降低90%+

xTensor是xLLM高性能推理的基础设施，与ACLGraph、全局KV Cache管理等技术协同工作，共同实现高效推理。

**下一篇文章预告**：《NPU能跑赢GPU吗？xLLM性能基准测试报告》，将全面展示NPU与GPU的性能对比数据。

---

> **参考资料**：
> - [xTensor文档](https://xllm.readthedocs.io/zh-cn/latest/features/xtensor_memory.html)
> - [xLLM技术报告](https://arxiv.org/abs/2510.14686)
> - [NPU内存管理最佳实践](https://www.hiascend.com/)

## 实现计划

### 阶段一：ModelArgs 扩展
1. **修改 `model_args.h`**
   - 添加 `is_hybrid_attention_model()` 方法
   - 添加 `is_linear_attention_layer(int64_t layer_id)` 方法
   - 添加 `is_standard_attention_layer(int64_t layer_id)` 方法

### 阶段二：BlockManager 接口扩展
2. **修改 `block_manager.h`**
   - 添加 `set_model_args(const ModelArgs* args)` 虚方法
   - 添加 `is_linear_attention_layer(int64_t layer_id)` 虚方法

3. **修改 `block_manager_impl.h`**
   - 添加 `model_args_` 成员变量
   - 实现新接口方法
   - 修改 `allocate_shared()` 和 `cache()` 方法签名

4. **修改 `block_manager_impl.cpp`**
   - 实现 `set_model_args()`
   - 实现 `is_linear_attention_layer()`
   - 修改 `allocate_shared()` 逻辑：Linear Attention 层跳过匹配
   - 修改 `cache()` 逻辑：Linear Attention 层不缓存

### 阶段三：BlockManagerPool 适配
5. **修改 `block_manager_pool.h/cpp`**
   - 添加 `set_model_args()` 方法传递到各 BlockManager

### 阶段四：Scheduler 调用修改
6. **修改 `chunked_prefill_scheduler.cpp`**
   - 在 `allocate_shared_blocks_for()` 中传递 layer_id
   - 在缓存时传递 layer_id

7. **修改 `mix_scheduler.cpp`**
   - 同样传递 layer_id 到 BlockManager

### 阶段五：初始化设置
8. **修改 `worker_impl.cpp` 或 `llm_engine.cpp`**
   - 在创建 BlockManager 后设置 model_args

### 阶段六：编译验证
9. **编译测试**
   - 运行编译确保无语法错误
   - 检查类型匹配

---

**预计修改文件数量**: 8 个文件
**预计代码行数**: 约 100-150 行
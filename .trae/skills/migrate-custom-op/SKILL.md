---
name: "migrate-custom-op"
description: "Migrate CANN custom operators from vllm-ascend to xLLM. Invoke when adding new Ascend C kernels or porting operators from other projects."
---

# Migrate Custom Operator to xLLM

This skill provides a step-by-step guide for migrating CANN custom operators (ACLNN) from vllm-ascend or other projects to xLLM.

## Overview

The migration process involves:
1. Adding parameter structures
2. Creating interface declarations
3. Implementing ACLNN call layer
4. Adding custom operator source code
5. Integrating into the calling code
6. Writing test cases

## Directory Structure

```
xllm/
├── core/
│   └── kernels/
│       ├── param.h                    # Parameter structures
│       ├── ops_api.h                  # Interface declarations
│       ├── ops_api.cpp                # Interface implementations
│       └── npu/
│           └── xllm_ops/              # ACLNN call layer
│               ├── xllm_ops_api.h
│               ├── <op_name>_fn.cpp
│               └── CMakeLists.txt

third_party/
└── xllm_ops/                          # Custom operator source (git submodule)
    └── xllm_ops/
        └── <op_name>/                 # Operator source directory
            ├── CMakeLists.txt
            ├── <op_name>_tiling.h
            ├── <op_name>.h
            ├── <op_name>.cpp
            └── <op_name>_cpu.cpp
```

## Step-by-Step Migration Guide

### Step 1: Add Parameter Structure

**File**: `xllm/core/kernels/param.h`

```cpp
// NPU <OpName> parameters
struct <OpName>Params {
  // Input tensors with shape comments
  torch::Tensor input1;  // [shape description]
  torch::Tensor input2;  // [shape description]
  
  // Optional inputs
  std::optional<torch::Tensor> optional_input = std::nullopt;
  
  // Configuration parameters
  int64_t param1 = default_value;
  bool param2 = false;
};
```

### Step 2: Add Interface Declaration

**File**: `xllm/core/kernels/ops_api.h`

```cpp
torch::Tensor <op_name>(<OpName>Params& params);
```

### Step 3: Add NPU Layer Declaration

**File**: `xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h`

```cpp
torch::Tensor <op_name>(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const c10::optional<torch::Tensor>& optional_input,
    int64_t param1,
    bool param2);
```

### Step 4: Implement ACLNN Call

**File**: `xllm/core/kernels/npu/xllm_ops/<op_name>_fn.cpp`

```cpp
#include <acl/acl.h>
#include "aclnnop/aclnn_<op_name>.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor <op_name>(...) {
  // 1. Check input tensors
  check_tensor(input1, "input1", "<op_name>");
  
  // 2. Create output tensor
  torch::Tensor output = at::empty(...);
  
  // 3. Get NPU stream
  int32_t device_id = input1.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  
  // 4. Create ACL tensors
  aclTensor* input1_tensor = nullptr;
  create_acltensor(&input1_tensor, input1);
  // ... create other tensors
  
  // 5. Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(
      aclnn<OpName>GetWorkspaceSize(..., &workspace_size, &executor),
      "<op_name>: failed to get workspace size");
  
  // 6. Allocate workspace
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  // 7. Execute operator
  CHECK_ACL_SUCCESS(
      aclnn<OpName>(workspace_addr, workspace_size, executor, stream),
      "<op_name>: failed to execute");
  
  // 8. Synchronize
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "<op_name>: failed to synchronize stream");
  
  // 9. Cleanup
  aclDestroyTensor(input1_tensor);
  // ... destroy other tensors
  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }
  
  return output;
}

}  // namespace xllm::kernel::npu
```

### Step 5: Update CMakeLists.txt

**File**: `xllm/core/kernels/npu/xllm_ops/CMakeLists.txt`

```cmake
SRCS
  ...
  <op_name>_fn.cpp
```

### Step 6: Implement Interface

**File**: `xllm/core/kernels/ops_api.cpp`

```cpp
torch::Tensor <op_name>(<OpName>Params& params) {
#if defined(USE_NPU)
  return npu::<op_name>(params.input1, params.input2, ...);
#else
  NOT_IMPLEMENTED();
#endif
}
```

### Step 7: Add Operator Source Code

**Directory**: `third_party/xllm_ops/xllm_ops/<op_name>/`

Required files:
- `<op_name>_tiling.h` - Tiling data structure
- `<op_name>.h` - Kernel header
- `<op_name>.cpp` - Kernel implementation
- `<op_name>_cpu.cpp` - Tiling function and operator registration
- `CMakeLists.txt` - Build configuration

### Step 8: Integrate into Calling Code

```cpp
#if defined(USE_NPU)
    xllm::kernel::<OpName>Params params;
    params.input1 = ...;
    params.input2 = ...;
    auto result = xllm::kernel::<op_name>(params);
#else
    // Original implementation
#endif
```

### Step 9: Write Test Cases

**File**: `third_party/xllm_ops/test/cpp_test/<op_name>_test.cpp`

```cpp
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_<op_name>.h"

class <OpName>Test : public ::testing::Test {
protected:
    void SetUp() override { /* Initialize ACL */ }
    void TearDown() override { /* Cleanup ACL */ }
};

TEST_F(<OpName>Test, BasicCorrectness) {
    // Create test inputs
    // Call operator
    // Compare with reference
    EXPECT_LT(diff, tolerance);
}
```

## Build and Test

### Build Custom Operator

```bash
cd third_party/xllm_ops
bash build.sh -n <op_name> -c ascend910b
```

### Build xLLM

```bash
mkdir -p build && cd build
cmake .. -DUSE_NPU=ON -DDEVICE_TYPE=USE_A2
make -j$(nproc)
```

### Run Tests

```bash
cd third_party/xllm_ops/test/cpp_test
mkdir -p build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make <op_name>_test
./<op_name>_test
```

## Common Issues

1. **Data Layout Mismatch**: Check tensor shapes and transpose if needed
2. **Missing ACLNN Header**: Ensure CANN version supports the operator
3. **Build Errors**: Check CMakeLists.txt configuration
4. **Runtime Errors**: Verify ACL tensor creation and parameter passing

## Reference

- vllm-ascend: https://github.com/vllm-project/vllm-ascend
- CANN Custom Operator Guide
- xllm_ops Repository: https://gitcode.com/xLLM-AI/xllm_ops.git

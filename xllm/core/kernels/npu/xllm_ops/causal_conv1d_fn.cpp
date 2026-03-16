/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor causal_conv1d_fn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation_mode,
    torch::Tensor& conv_state,
    const torch::Tensor& has_initial_state,
    const torch::Tensor& cache_indices,
    const torch::Tensor& query_start_loc,
    int64_t pad_slot_id) {
  check_tensor(x, "x", "causal_conv1d_fn");
  check_tensor(weight, "weight", "causal_conv1d_fn");
  check_tensor(conv_state, "conv_state", "causal_conv1d_fn");
  check_tensor(has_initial_state, "has_initial_state", "causal_conv1d_fn");
  check_tensor(cache_indices, "cache_indices", "causal_conv1d_fn");
  check_tensor(query_start_loc, "query_start_loc", "causal_conv1d_fn");

  CHECK(x.dim() >= 2) << "x must have at least 2 dimensions, got " << x.dim();
  CHECK(weight.dim() >= 2) << "weight must have at least 2 dimensions, got "
                           << weight.dim();
  CHECK(x.size(-1) == weight.size(-1))
      << "x and weight must have same last dimension, got " << x.size(-1)
      << " and " << weight.size(-1);

  torch::Tensor output = at::empty(x.sizes(), x.options());

  int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  aclTensor* x_tensor = nullptr;
  aclTensor* weight_tensor = nullptr;
  aclTensor* bias_tensor = nullptr;
  aclTensor* conv_state_tensor = nullptr;
  aclTensor* has_initial_state_tensor = nullptr;
  aclTensor* cache_indices_tensor = nullptr;
  aclTensor* query_start_loc_tensor = nullptr;
  aclTensor* output_tensor = nullptr;

  auto cleanup_acl_tensors = [&]() {
    if (x_tensor) aclDestroyTensor(x_tensor);
    if (weight_tensor) aclDestroyTensor(weight_tensor);
    if (bias_tensor) aclDestroyTensor(bias_tensor);
    if (conv_state_tensor) aclDestroyTensor(conv_state_tensor);
    if (has_initial_state_tensor) aclDestroyTensor(has_initial_state_tensor);
    if (cache_indices_tensor) aclDestroyTensor(cache_indices_tensor);
    if (query_start_loc_tensor) aclDestroyTensor(query_start_loc_tensor);
    if (output_tensor) aclDestroyTensor(output_tensor);
  };

  auto cleanup_all = [&](void* workspace_addr, uint64_t workspace_size) {
    cleanup_acl_tensors();
    if (workspace_size > 0 && workspace_addr) {
      aclrtFree(workspace_addr);
    }
  };

  create_acltensor(&x_tensor, x);
  create_acltensor(&weight_tensor, weight);
  if (bias.has_value() && bias.value().defined()) {
    create_acltensor(&bias_tensor, bias.value());
  }
  create_acltensor(&conv_state_tensor, conv_state);
  create_acltensor(&has_initial_state_tensor, has_initial_state);
  create_acltensor(&cache_indices_tensor, cache_indices);
  create_acltensor(&query_start_loc_tensor, query_start_loc);
  create_acltensor(&output_tensor, output);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclStatus status = aclnnCausalConv1dGetWorkspaceSize(
      x_tensor,
      weight_tensor,
      bias_tensor,
      conv_state_tensor,
      query_start_loc_tensor,
      cache_indices_tensor,
      has_initial_state_tensor,
      activation_mode,
      pad_slot_id,
      output_tensor,
      &workspace_size,
      &executor);
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "causal_conv1d_fn: failed to get workspace size, status: "
               << status;
    cleanup_all(nullptr, 0);
    TORCH_CHECK(false, "Failed to get workspace size for causal_conv1d_fn");
  }

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    aclStatus malloc_status =
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (malloc_status != ACL_SUCCESS) {
      LOG(ERROR) << "causal_conv1d_fn: failed to allocate workspace, status: "
                 << malloc_status;
      cleanup_all(nullptr, 0);
      TORCH_CHECK(false, "Failed to allocate workspace for causal_conv1d_fn");
    }
  }

  status = aclnnCausalConv1d(workspace_addr, workspace_size, executor, stream);
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "causal_conv1d_fn: failed to execute, status: " << status;
    cleanup_all(workspace_addr, workspace_size);
    TORCH_CHECK(false, "Failed to execute causal_conv1d_fn");
  }

  status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "causal_conv1d_fn: failed to synchronize stream, status: "
               << status;
    cleanup_all(workspace_addr, workspace_size);
    TORCH_CHECK(false, "Failed to synchronize stream for causal_conv1d_fn");
  }

  cleanup_all(workspace_addr, workspace_size);

  return output;
}

}  // namespace xllm::kernel::npu

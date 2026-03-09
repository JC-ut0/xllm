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

#include "mamba_cache_manager.h"

#include <glog/logging.h>
#include <torch/torch.h>

#if defined(USE_NPU)
#include <acl/acl.h>
#include <torch_npu/torch_npu.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace xllm {

MambaCacheMode ParseMambaCacheMode(const std::string& mode) {
  if (mode == "all") {
    return MambaCacheMode::kAll;
  } else if (mode == "align") {
    return MambaCacheMode::kAlign;
  }
  return MambaCacheMode::kNone;
}

std::string MambaCacheModeToString(MambaCacheMode mode) {
  switch (mode) {
    case MambaCacheMode::kAll:
      return "all";
    case MambaCacheMode::kAlign:
      return "align";
    default:
      return "none";
  }
}

MambaStateCopySpec GetConvCopySpec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens) {
  CHECK_GE(cur_block_idx, 0);
  CHECK_LT(cur_block_idx, static_cast<int32_t>(block_ids.size()));
  CHECK_GE(num_accepted_tokens, 1);

  int32_t src_block_id = block_ids[cur_block_idx];
  auto src_state = state[src_block_id].slice(0, num_accepted_tokens - 1);

  return MambaStateCopySpec{
      .start_addr = reinterpret_cast<int64_t>(src_state.data_ptr()),
      .num_elements = src_state.numel()};
}

MambaStateCopySpec GetTemporalCopySpec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens) {
  CHECK_GE(cur_block_idx, 0);
  CHECK_GE(num_accepted_tokens, 1);

  int32_t src_block_idx = cur_block_idx + num_accepted_tokens - 1;
  CHECK_LT(src_block_idx, static_cast<int32_t>(block_ids.size()));

  int32_t src_block_id = block_ids[src_block_idx];
  auto src_state = state[src_block_id];

  return MambaStateCopySpec{
      .start_addr = reinterpret_cast<int64_t>(src_state.data_ptr()),
      .num_elements = src_state.numel()};
}

std::vector<MambaStateCopyFunc> GetGdnCopyFuncs() {
  return {GetConvCopySpec, GetTemporalCopySpec};
}

MambaCacheManager::MambaCacheManager(
    uint32_t block_size,
    MambaCacheMode cache_mode,
    const std::vector<MambaStateCopyFunc>& copy_funcs)
    : block_size_(block_size),
      cache_mode_(cache_mode),
      copy_funcs_(copy_funcs) {
  CHECK_GT(block_size, 0) << "Block size must be positive";
  CHECK(cache_mode != MambaCacheMode::kNone)
      << "MambaCacheManager should not be created with kNone mode";
}

std::vector<Block> MambaCacheManager::match(
    const Slice<int32_t>& token_ids,
    int32_t layer_id,
    const Slice<Block>& existed_blocks) {
  if (cache_mode_ == MambaCacheMode::kNone) {
    return {};
  }

  auto it = layer_caches_.find(layer_id);
  if (it == layer_caches_.end()) {
    it = layer_caches_.emplace(layer_id, std::make_unique<PrefixCache>(block_size_)).first;
  }

  return it->second->match(token_ids, existed_blocks);
}

void MambaCacheManager::insert(
    const Slice<int32_t>& token_ids,
    int32_t layer_id,
    std::vector<Block>& blocks,
    size_t existed_blocks_num) {
  if (cache_mode_ == MambaCacheMode::kNone) {
    return;
  }

  auto it = layer_caches_.find(layer_id);
  if (it == layer_caches_.end()) {
    it = layer_caches_.emplace(layer_id, std::make_unique<PrefixCache>(block_size_)).first;
  }

  it->second->insert(token_ids, blocks, existed_blocks_num);
}

void MambaCacheManager::copy_state(
    torch::Tensor src_state,
    torch::Tensor dst_state,
    const MambaStateCopySpec& spec) {
  CHECK(src_state.defined()) << "Source state is not defined";
  CHECK(dst_state.defined()) << "Destination state is not defined";

  auto src_ptr = reinterpret_cast<uint8_t*>(spec.start_addr);
  auto dst_ptr = reinterpret_cast<uint8_t*>(dst_state.data_ptr());

  size_t num_bytes = spec.num_elements * src_state.element_size();

#if defined(USE_NPU)
  aclError err = aclrtMemcpy(
      dst_ptr, num_bytes, src_ptr, num_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (err != ACL_SUCCESS) {
    LOG(ERROR) << "ACL memcpy failed: " << err;
  }
#elif defined(USE_CUDA)
  cudaMemcpyKind kind;
  if (src_state.device().is_cuda() && dst_state.device().is_cuda()) {
    kind = cudaMemcpyDeviceToDevice;
  } else if (src_state.device().is_cpu() && dst_state.device().is_cuda()) {
    kind = cudaMemcpyHostToDevice;
  } else if (src_state.device().is_cuda() && dst_state.device().is_cpu()) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyHostToHost;
  }

  cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, num_bytes, kind);
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA memcpy failed: " << cudaGetErrorString(err);
  }
#else
  memcpy(dst_ptr, src_ptr, num_bytes);
#endif
}

size_t MambaCacheManager::evict(size_t n_blocks) {
  size_t total_evicted = 0;
  for (auto& [layer_id, cache] : layer_caches_) {
    total_evicted += cache->evict(n_blocks);
  }
  return total_evicted;
}

size_t MambaCacheManager::num_blocks() const {
  size_t total = 0;
  for (const auto& [layer_id, cache] : layer_caches_) {
    total += cache->num_blocks();
  }
  return total;
}

}  // namespace xllm

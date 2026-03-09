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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "framework/block/block.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "util/slice.h"

namespace xllm {

enum class MambaCacheMode {
  kNone = 0,
  kAll = 1,
  kAlign = 2,
};

MambaCacheMode ParseMambaCacheMode(const std::string& mode);

std::string MambaCacheModeToString(MambaCacheMode mode);

struct MambaStateCopySpec {
  int64_t start_addr;
  int64_t num_elements;
};

using MambaStateCopyFunc = std::function<MambaStateCopySpec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens)>;

MambaStateCopySpec GetConvCopySpec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens);

MambaStateCopySpec GetTemporalCopySpec(
    torch::Tensor state,
    std::vector<int32_t> block_ids,
    int32_t cur_block_idx,
    int32_t num_accepted_tokens);

std::vector<MambaStateCopyFunc> GetGdnCopyFuncs();

class MambaCacheManager {
 public:
  MambaCacheManager(
      uint32_t block_size,
      MambaCacheMode cache_mode,
      const std::vector<MambaStateCopyFunc>& copy_funcs);

  ~MambaCacheManager() = default;

  MambaCacheManager(const MambaCacheManager&) = delete;
  MambaCacheManager& operator=(const MambaCacheManager&) = delete;

  std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      int32_t layer_id,
      const Slice<Block>& existed_blocks = {});

  void insert(
      const Slice<int32_t>& token_ids,
      int32_t layer_id,
      std::vector<Block>& blocks,
      size_t existed_blocks_num = 0);

  void copy_state(
      torch::Tensor src_state,
      torch::Tensor dst_state,
      const MambaStateCopySpec& spec);

  const std::vector<MambaStateCopyFunc>& get_copy_funcs() const {
    return copy_funcs_;
  }

  size_t evict(size_t n_blocks);

  size_t num_blocks() const;

  MambaCacheMode cache_mode() const { return cache_mode_; }

  uint32_t block_size() const { return block_size_; }

 private:
  uint32_t block_size_;
  MambaCacheMode cache_mode_;
  std::vector<MambaStateCopyFunc> copy_funcs_;

  std::unordered_map<int32_t, std::unique_ptr<PrefixCache>> layer_caches_;
};

}  // namespace xllm

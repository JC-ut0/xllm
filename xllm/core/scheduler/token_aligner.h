

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

#include <algorithm>

namespace xllm {

class TokenAligner {
 public:
  explicit TokenAligner(int32_t block_size)
      : block_size_(block_size) {}

  int32_t compute_scheduled_tokens(
      int32_t num_tokens,
      int32_t num_computed_tokens,
      bool force_align = false) const {
    if (!force_align || block_size_ <= 0) {
      return num_tokens;
    }

    int32_t remaining_tokens = num_tokens - num_computed_tokens;
    if (remaining_tokens <= 0) {
      return 0;
    }

    int32_t next_boundary =
        ((num_computed_tokens + block_size_ - 1) / block_size_) * block_size_;

    if (next_boundary <= num_computed_tokens) {
      next_boundary = num_computed_tokens + block_size_;
    }

    return std::min(next_boundary - num_computed_tokens, remaining_tokens);
  }

  int32_t compute_aligned_tokens(int32_t num_tokens) const {
    if (block_size_ <= 0) {
      return num_tokens;
    }
    return (num_tokens / block_size_) * block_size_;
  }

  bool is_at_block_boundary(int32_t token_position) const {
    if (block_size_ <= 0) {
      return false;
    }
    return token_position > 0 && token_position % block_size_ == 0;
  }

  int32_t get_block_size() const { return block_size_; }

 private:
  int32_t block_size_;
};

}  // namespace xllm

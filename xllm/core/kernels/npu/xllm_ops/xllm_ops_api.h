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

#include <torch/torch.h>

namespace xllm::kernel::npu {

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_token_ids);

void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);

void replace_token(torch::Tensor& dst, torch::Tensor& src);

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

}  // namespace xllm::kernel::npu

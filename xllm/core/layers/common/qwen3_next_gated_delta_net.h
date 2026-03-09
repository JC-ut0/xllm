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

#include "attention.h"
#include "linear.h"
#include "rms_norm_gated.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/prefix_cache/mamba_cache_manager.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class Qwen3NextGatedDeltaNetImpl : public torch::nn::Module {
 public:
  Qwen3NextGatedDeltaNetImpl() = default;
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

  void set_mamba_cache_mode(MambaCacheMode mode) { mamba_cache_mode_ = mode; }
  void set_mamba_block_size(int32_t block_size) { mamba_block_size_ = block_size; }

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
  process_qkvz_tensor(const torch::Tensor& qkvz);
  std::tuple<torch::Tensor, torch::Tensor> process_ba_tensor(const torch::Tensor& ba);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_mixed_qkv(torch::Tensor& mixed_qkv);

  torch::Tensor reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata, const torch::Tensor& qkvz);
  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata, const torch::Tensor& padded_qkvz);

  int64_t num_k_heads_;
  int64_t num_v_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_k_dim_;
  int64_t head_v_dim_;
  int64_t k_size_;
  int64_t v_size_;
  int64_t tp_size_; 
  int64_t rank_;
  int32_t conv_kernel_size_;

  MambaCacheMode mamba_cache_mode_ = MambaCacheMode::kNone;
  int32_t mamba_block_size_ = 0;

  ColumnParallelLinear qkvz_proj_{nullptr};
  ColumnParallelLinear ba_proj_{nullptr};
  ColumnParallelLinear conv1d_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  RmsNormGated norm_{nullptr};
  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};
TORCH_MODULE(Qwen3NextGatedDeltaNet);

}  // namespace layer
}  // namespace xllm

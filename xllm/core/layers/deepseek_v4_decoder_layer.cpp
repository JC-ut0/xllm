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

#include "deepseek_v4_decoder_layer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {
namespace {

bool debug_trace_all() {
  const char* value = std::getenv("XLLM_DEBUG_TRACE_ALL");
  return value != nullptr &&
         (std::string(value) == "1" || std::string(value) == "true" ||
          std::string(value) == "TRUE" || std::string(value) == "True");
}

bool debug_moe() {
  const char* value = std::getenv("XLLM_DEBUG_MOE");
  return debug_trace_all() ||
         (value != nullptr &&
          (std::string(value) == "1" || std::string(value) == "true" ||
           std::string(value) == "TRUE" || std::string(value) == "True"));
}

int64_t debug_rows() {
  const char* value = std::getenv("XLLM_DEBUG_MOE_ROWS");
  if (value == nullptr) {
    return 4;
  }
  char* end = nullptr;
  const int64_t rows = std::strtoll(value, &end, 10);
  return end == value ? 4 : std::max<int64_t>(1, rows);
}

int64_t debug_top_rows() {
  const char* value = std::getenv("XLLM_DEBUG_MOE_TOP_ROWS");
  if (value == nullptr) {
    return 3;
  }
  char* end = nullptr;
  const int64_t rows = std::strtoll(value, &end, 10);
  return end == value ? 3 : std::max<int64_t>(0, rows);
}

double debug_abs_threshold() {
  const char* value = std::getenv("XLLM_DEBUG_MOE_ABS_THRESHOLD");
  if (value == nullptr) {
    return 1000.0;
  }
  char* end = nullptr;
  const double threshold = std::strtod(value, &end);
  return end == value ? 1000.0 : threshold;
}

void log_tensor_summary(const char* tag,
                        int32_t layer_id,
                        const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    LOG(WARNING) << "[XLLM_DEBUG][dsv4_moe] layer=" << layer_id << " " << tag
                 << "=<undefined_or_empty>";
    return;
  }
  auto flat = tensor.detach().to(torch::kFloat32).reshape({-1});
  const auto finite = torch::isfinite(flat);
  const int64_t finite_count = finite.sum().item<int64_t>();
  LOG(WARNING) << "[XLLM_DEBUG][dsv4_moe] layer=" << layer_id << " " << tag
               << " shape=" << tensor.sizes() << " finite=" << finite_count
               << "/" << flat.numel() << " min=" << flat.min().item<float>()
               << " max=" << flat.max().item<float>()
               << " mean=" << flat.mean().item<float>();
}

void log_moe_row(int32_t layer_id,
                 int64_t row,
                 const torch::Tensor& input_ids_cpu,
                 const torch::Tensor& ids_cpu,
                 const torch::Tensor& weights_cpu,
                 const std::string& extra) {
  if (!ids_cpu.defined() || ids_cpu.dim() != 2 || row >= ids_cpu.size(0)) {
    LOG(WARNING) << "[XLLM_DEBUG][dsv4_moe] layer=" << layer_id
                 << " row=" << row << " routing=<out_of_range>" << extra;
    return;
  }
  const int64_t topk = ids_cpu.size(1);
  std::string experts;
  std::string weights;
  for (int64_t k = 0; k < topk; ++k) {
    if (!experts.empty()) {
      experts += ",";
      weights += ",";
    }
    experts += std::to_string(ids_cpu[row][k].item<int32_t>());
    if (weights_cpu.defined() && weights_cpu.dim() == 2 &&
        row < weights_cpu.size(0) && k < weights_cpu.size(1)) {
      weights += std::to_string(weights_cpu[row][k].item<float>());
    } else {
      weights += "<missing>";
    }
  }
  std::string token = "<none>";
  if (input_ids_cpu.defined() && input_ids_cpu.numel() > row) {
    token = std::to_string(input_ids_cpu[row].item<int64_t>());
  }
  LOG(WARNING) << "[XLLM_DEBUG][dsv4_moe] layer=" << layer_id << " row=" << row
               << " token_id=" << token << " experts=" << experts
               << " weights=" << weights << extra;
}

void log_moe_extreme_rows(int32_t layer_id,
                          const torch::Tensor& input_ids_cpu,
                          const torch::Tensor& ids_cpu,
                          const torch::Tensor& weights_cpu,
                          const torch::Tensor& ffn_output) {
  const int64_t rows = debug_top_rows();
  if (rows <= 0 || !ffn_output.defined() || ffn_output.dim() != 2 ||
      ffn_output.size(0) <= 1) {
    return;
  }

  auto output = ffn_output.detach().to(torch::kFloat32);
  auto row_abs_max = std::get<0>(torch::abs(output).max(/*dim=*/1));
  const int64_t top_rows = std::min<int64_t>(rows, row_abs_max.size(0));
  auto [top_values, top_indices] = row_abs_max.topk(top_rows);
  top_values = top_values.to(torch::kCPU);
  top_indices = top_indices.to(torch::kCPU);

  const double threshold = debug_abs_threshold();
  for (int64_t i = 0; i < top_rows; ++i) {
    const double max_abs = top_values[i].item<float>();
    if (max_abs < threshold) {
      continue;
    }
    const int64_t row = top_indices[i].item<int64_t>();
    auto row_tensor = output[row];
    const float row_min = row_tensor.min().item<float>();
    const float row_max = row_tensor.max().item<float>();
    const float row_mean = row_tensor.mean().item<float>();
    std::string extra = " max_abs=" + std::to_string(max_abs) +
                        " min=" + std::to_string(row_min) +
                        " max=" + std::to_string(row_max) +
                        " mean=" + std::to_string(row_mean);
    log_moe_row(layer_id, row, input_ids_cpu, ids_cpu, weights_cpu, extra);
  }
}

void log_moe_routing_debug(int32_t layer_id,
                           const torch::Tensor& input_ids,
                           const torch::Tensor& topk_weights,
                           const torch::Tensor& topk_ids,
                           const torch::Tensor& ffn_input,
                           const torch::Tensor& ffn_output) {
  if (!debug_moe()) {
    return;
  }
  log_tensor_summary("ffn_input", layer_id, ffn_input);
  log_tensor_summary("ffn_output", layer_id, ffn_output);
  if (!topk_ids.defined() || topk_ids.numel() == 0) {
    return;
  }

  auto ids_cpu = topk_ids.detach().to(torch::kCPU);
  auto weights_cpu = topk_weights.detach().to(torch::kFloat32).to(torch::kCPU);
  torch::Tensor input_ids_cpu =
      input_ids.defined() ? input_ids.detach().to(torch::kCPU).reshape({-1})
                          : torch::Tensor();
  const int64_t rows = std::min<int64_t>(debug_rows(), ids_cpu.size(0));
  for (int64_t row = 0; row < rows; ++row) {
    log_moe_row(layer_id, row, input_ids_cpu, ids_cpu, weights_cpu, "");
  }
  log_moe_extreme_rows(
      layer_id, input_ids_cpu, ids_cpu, weights_cpu, ffn_output);
}

}  // namespace

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  layer_id_ = layer_id;
  int64_t hidden_size = args.hidden_size();

  hc_mult_ = args.hc_mult();
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DSAttention(context, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  FusedMoEArgs moe_args;
  moe_args.is_gated = true;
  // DeepseekV4 drives expert routing through its own DeepseekV4Gate and only
  // calls forward_with_selected_experts().  The FusedMoE internal gate_ is
  // therefore never used; skip loading its weights to avoid redundant memory
  // allocation and a duplicate copy of the router weight matrix.
  moe_args.skip_gate_load = true;
  moe_mlp_ = register_module(
      "ffn", FusedMoE(args, moe_args, quant_args, parallel_args, options));
  // Register as "gate" to match Python's mlp.gate module path.
  gate_ = register_module("gate", DeepseekV4Gate(context, layer_id));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() == 0) {
    ffn_state = state_dict.get_dict_with_prefix("mlp.");
  }
  if (ffn_state.size() > 0) {
    auto gate_state = ffn_state.get_dict_with_prefix("gate.");
    if (gate_state.size() == 0) {
      gate_state = state_dict.get_dict_with_prefix("gate.");
    }
    if (gate_state.size() > 0) {
      gate_->load_state_dict(gate_state);
    }
    moe_mlp_->load_state_dict(ffn_state);
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

void DeepseekV4DecoderLayerImpl::verify_loaded_weights() const {}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids) {
  (void)positions;

  residual = std::nullopt;

  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4DecoderLayer requires DSA metadata for DSAttention path.";

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));

  auto& dsa = *(attn_metadata.dsa_metadata);
  const auto compress_metadata = std::make_tuple(
      dsa.c1_metadata, dsa.c4_metadata, dsa.c128_metadata, dsa.qli_metadata);
  KVState kv_state{kv_cache.get_swa_cache(),
                   kv_cache.get_compress_kv_state(),
                   kv_cache.get_compress_score_state(),
                   kv_cache.get_compress_index_kv_state(),
                   kv_cache.get_compress_index_score_state()};
  auto [attn_output, attn_lse] = attention_->forward(
      dsa,
      attn_input,
      kv_cache,
      kv_state,
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill,
      std::to_string(dsa.layer_id),
      compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));

  auto ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
  std::optional<torch::Tensor> gate_input_ids = std::nullopt;
  if (input_ids.has_value() && input_ids.value().defined()) {
    auto flat_input_ids =
        input_ids.value().reshape({-1}).to(ffn_input.device());
    const int64_t token_count = flat_input_ids.size(0);
    const int64_t hidden_rows = ffn_input_2d.size(0);
    if (token_count == hidden_rows) {
      gate_input_ids = flat_input_ids;
    } else if (token_count > 0 && hidden_rows % token_count == 0) {
      const int64_t repeat_factor = hidden_rows / token_count;
      gate_input_ids = flat_input_ids.unsqueeze(1)
                           .repeat({1, repeat_factor})
                           .reshape({hidden_rows});
    }
  }
  if (gate_->is_hash_layer()) {
    CHECK(gate_input_ids.has_value())
        << "DeepseekV4 hash gate requires input_ids for routing";
  }
  auto [topk_weights, topk_ids] = gate_->forward(ffn_input_2d, gate_input_ids);
  ffn_input = moe_mlp_->forward_with_selected_experts(
      ffn_input, topk_weights, topk_ids, input_params, gate_input_ids);
  log_moe_routing_debug(
      layer_id_,
      gate_input_ids.has_value() ? gate_input_ids.value() : torch::Tensor(),
      topk_weights,
      topk_ids,
      ffn_input_2d,
      ffn_input.reshape({-1, ffn_input.size(-1)}));
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  kernel::HcPreParams params;
  params.x = x;
  params.hc_fn = hc_fn;
  params.hc_scale = hc_scale;
  params.hc_base = hc_base;
  params.hc_mult = hc_mult_;
  params.hc_sinkhorn_iters = hc_sinkhorn_iters_;
  params.norm_eps = norm_eps_;
  params.hc_eps = hc_eps_;
  return kernel::hc_pre(params);
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  kernel::HcPostParams params;
  if (x.dim() == 2 && residual.dim() == 3 && post.dim() == 2 &&
      comb.dim() == 3) {
    params.x = x.unsqueeze(0);
    params.residual = residual.unsqueeze(0);
    params.post = post.unsqueeze(0);
    params.comb = comb.unsqueeze(0);
    return kernel::hc_post(params).squeeze(0);
  }

  params.x = x;
  params.residual = residual;
  params.post = post;
  params.comb = comb;
  return kernel::hc_post(params);
}

}  // namespace layer
}  // namespace xllm

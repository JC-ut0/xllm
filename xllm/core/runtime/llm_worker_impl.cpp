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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {
namespace {

std::string tensor_debug_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << "sizes=" << tensor.sizes() << ", dtype=" << tensor.dtype()
     << ", device=" << tensor.device() << ", numel=" << tensor.numel();
  return os.str();
}

std::string tensor_head_values(const torch::Tensor& tensor,
                               int64_t max_values = 8) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return "[]";
  }
  auto flat = tensor.reshape({-1});
  const int64_t n = std::min<int64_t>(flat.numel(), max_values);
  auto head = flat.slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                  .to(torch::kCPU, /*non_blocking=*/false);
  std::ostringstream os;
  os << head;
  return os.str();
}

}  // namespace

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA) || defined(USE_MUSA)
  threadpool_.schedule([this]() mutable {
    // initialize flashinfer workspace
    ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
        device_);
  });
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
  if (FLAGS_enable_manual_loader) {
#if defined(USE_NPU)
    SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
#endif
    return step_internal(input);
  }
  return step_internal(input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input) {
  MULTI_MODEL_STEP_LOCK(FLAGS_enable_xtensor);

  Timer timer;
  auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(input.eplb_info);
  }

  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!model_output.hidden_states.defined()) {
    LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] model hidden_states undefined, "
              << "batch_forward_type="
              << input.input_params.batch_forward_type.to_string();
    return std::nullopt;
  }

  LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] after model forward: "
            << "batch_forward_type="
            << input.input_params.batch_forward_type.to_string()
            << ", hidden_states="
            << tensor_debug_string(model_output.hidden_states)
            << ", token_ids=" << tensor_debug_string(input.token_ids)
            << ", positions=" << tensor_debug_string(input.positions)
            << ", selected_token_idxes="
            << tensor_debug_string(sampling_params.selected_token_idxes)
            << ", selected_token_idxes_head="
            << tensor_head_values(sampling_params.selected_token_idxes)
            << ", sample_idxes="
            << tensor_debug_string(sampling_params.sample_idxes)
            << ", sample_idxes_head="
            << tensor_head_values(sampling_params.sample_idxes)
            << ", do_sample=" << tensor_debug_string(sampling_params.do_sample)
            << ", skip_sampling_for_logits_only="
            << input.skip_sampling_for_logits_only
            << ", enable_schedule_overlap=" << enable_schedule_overlap()
            << ", driver=" << static_cast<bool>(driver_)
            << ", dp_driver=" << static_cast<bool>(dp_driver_)
            << ", spec_decode=" << options_.enable_speculative_decode();

  torch::Tensor logits;
  torch::Tensor selected_hidden_from_lm_head;
  if (sampling_params.selected_token_idxes.defined()) {
    if (options_.cp_size() > 1) {
      logits = model_->logits(model_output.hidden_states,
                              sampling_params.selected_token_idxes,
                              selected_hidden_from_lm_head);
    } else {
      logits = model_->logits(model_output.hidden_states,
                              sampling_params.selected_token_idxes);
    }
    LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] after lm_head: logits="
              << tensor_debug_string(logits)
              << ", logits_head=" << tensor_head_values(logits)
              << ", selected_hidden_from_lm_head="
              << tensor_debug_string(selected_hidden_from_lm_head);
  } else {
    LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] skip lm_head because "
              << "selected_token_idxes is undefined";
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    MULTI_MODEL_STEP_UNLOCK();
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        // TODO: Add error handling
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          break;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] returning nullopt before sampling "
              << "because overlap/driver/spec path is disabled";
    return std::nullopt;
  }

  // driver prepare model output
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    if (!input.skip_sampling_for_logits_only) {
      auto sample_output = sampler_->forward(logits, sampling_params);
      LOG(INFO) << "[PREFILL_OUTPUT_DEBUG] after sampler: next_tokens="
                << tensor_debug_string(sample_output.next_tokens)
                << ", next_tokens_head="
                << tensor_head_values(sample_output.next_tokens)
                << ", probs=" << tensor_debug_string(sample_output.probs)
                << ", logprobs=" << tensor_debug_string(sample_output.logprobs)
                << ", top_tokens="
                << tensor_debug_string(sample_output.top_tokens)
                << ", top_logprobs="
                << tensor_debug_string(sample_output.top_logprobs);

      // beam search kernel
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
          input.acc_logprob.numel() > 0) {
        beam_search_output =
            beam_searcher_->forward(input.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
      }

      // set sample output to output
      output.sample_output = sample_output;
      // set beam search output to output
      output.beam_search_output = beam_search_output;
    }
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.batch_forward_type.is_decode() && !is_spec_draft_) {
      output.sample_output.embeddings = embeddings;
    } else if (sampling_params.selected_token_idxes.defined()) {
      if (options_.cp_size() > 1) {
        CHECK(selected_hidden_from_lm_head.defined())
            << "selected_hidden_from_lm_head must be defined when "
               "selected_token_idxes is defined.";
        output.sample_output.embeddings = selected_hidden_from_lm_head;
      } else {
        output.sample_output.embeddings = embeddings.index_select(
            /*dim=*/0, sampling_params.selected_token_idxes);
      }
    }
  }

  MULTI_MODEL_STEP_UNLOCK();
  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      // TODO: Add error handling
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        break;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

}  // namespace xllm

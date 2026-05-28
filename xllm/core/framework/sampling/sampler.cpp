/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampling_params.h"

namespace xllm {
namespace {

bool debug_trace_all() {
  const char* value = std::getenv("XLLM_DEBUG_TRACE_ALL");
  return value != nullptr &&
         (std::string(value) == "1" || std::string(value) == "true" ||
          std::string(value) == "TRUE" || std::string(value) == "True");
}

int64_t debug_token_id() {
  const char* value = std::getenv("XLLM_DEBUG_TOKEN_ID");
  if (value == nullptr) {
    return -1;
  }
  char* end = nullptr;
  const int64_t id = std::strtoll(value, &end, 10);
  return end == value ? -1 : id;
}

std::vector<int64_t> debug_token_ids() {
  std::vector<int64_t> ids;
  const char* values = std::getenv("XLLM_DEBUG_TOKEN_IDS");
  if (values != nullptr) {
    std::string text(values);
    size_t start = 0;
    while (start < text.size()) {
      const size_t end = text.find(',', start);
      const std::string item = text.substr(
          start, end == std::string::npos ? std::string::npos : end - start);
      char* parse_end = nullptr;
      const int64_t id = std::strtoll(item.c_str(), &parse_end, 10);
      if (parse_end != item.c_str()) {
        ids.push_back(id);
      }
      if (end == std::string::npos) {
        break;
      }
      start = end + 1;
    }
  }
  const int64_t id = debug_token_id();
  if (id >= 0 && std::find(ids.begin(), ids.end(), id) == ids.end()) {
    ids.push_back(id);
  }
  return ids;
}

bool contains_token(const std::vector<int64_t>& ids, int64_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

int64_t debug_topk() {
  const char* value = std::getenv("XLLM_DEBUG_TOPK");
  if (value == nullptr) {
    return 10;
  }
  char* end = nullptr;
  const int64_t k = std::strtoll(value, &end, 10);
  return end == value ? 10 : std::max<int64_t>(1, k);
}

void log_sampler_debug(const torch::Tensor& sample_logits,
                       const torch::Tensor& samples,
                       const SamplingParameters& params) {
  const bool trace_all = debug_trace_all();
  const auto watch_ids = debug_token_ids();
  if (!trace_all && watch_ids.empty()) {
    return;
  }
  if (!sample_logits.defined() || !samples.defined() ||
      sample_logits.numel() == 0 || samples.numel() == 0) {
    return;
  }

  const int64_t rows = sample_logits.size(0);
  const int64_t vocab_size = sample_logits.size(1);
  const int64_t k = std::min<int64_t>(debug_topk(), vocab_size);
  auto logits_fp32 = sample_logits.to(torch::kFloat32);
  auto topk = logits_fp32.topk(k, /*dim=*/-1);
  auto top_values = std::get<0>(topk).detach().to(torch::kCPU);
  auto top_ids = std::get<1>(topk).detach().to(torch::kCPU);
  auto samples_cpu = samples.detach().to(torch::kCPU).reshape({-1});
  auto sample_idxes_cpu = params.sample_idxes.defined()
                              ? params.sample_idxes.detach().to(torch::kCPU)
                              : torch::Tensor();

  for (int64_t row = 0; row < rows; ++row) {
    const int64_t chosen = samples_cpu[row].item<int64_t>();
    bool top_contains_watch = false;
    if (!trace_all && !watch_ids.empty()) {
      for (int64_t i = 0; i < k && !top_contains_watch; ++i) {
        top_contains_watch =
            contains_token(watch_ids, top_ids[row][i].item<int64_t>());
      }
    }
    if (!trace_all && !contains_token(watch_ids, chosen) &&
        !top_contains_watch) {
      continue;
    }
    std::string top_desc;
    for (int64_t i = 0; i < k; ++i) {
      if (!top_desc.empty()) {
        top_desc += " ";
      }
      top_desc += std::to_string(top_ids[row][i].item<int64_t>());
      top_desc += ":";
      top_desc += std::to_string(top_values[row][i].item<float>());
    }
    std::string watch_desc;
    for (const int64_t watch_id : watch_ids) {
      if (watch_id < 0 || watch_id >= vocab_size) {
        continue;
      }
      if (!watch_desc.empty()) {
        watch_desc += " ";
      }
      watch_desc += std::to_string(watch_id);
      watch_desc += ":";
      watch_desc += std::to_string(logits_fp32[row][watch_id].item<float>());
    }
    int64_t source_row = row;
    if (sample_idxes_cpu.defined() && sample_idxes_cpu.numel() > row) {
      source_row = sample_idxes_cpu[row].item<int32_t>();
    }
    LOG(WARNING) << "[XLLM_DEBUG][sampler] row=" << row
                 << " source_row=" << source_row << " chosen=" << chosen
                 << " all_greedy=" << params.all_greedy_sample
                 << " all_random=" << params.all_random_sample << " watch_hit="
                 << (contains_token(watch_ids, chosen) ? "chosen" : "topk")
                 << " watch_logits=" << watch_desc << " top" << k << "="
                 << top_desc;
  }
}

}  // namespace

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) const {
  SampleOutput output;
  // apply frequency and presence penalties
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  // apply repetition penalties
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  torch::Tensor sample_logits = logits;
  torch::Tensor sample_temperatures = params.temperatures;
  torch::Tensor sample_top_k = params.top_k;
  torch::Tensor sample_top_p = params.top_p;
  const bool use_sample_indices =
      params.selected_token_idxes.numel() != params.sample_idxes.numel();
  if (use_sample_indices) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
    if (params.temperatures.defined()) {
      sample_temperatures =
          params.temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_k.defined()) {
      sample_top_k = params.top_k.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_p.defined()) {
      sample_top_p = params.top_p.index_select(/*dim=*/0, params.sample_idxes);
    }
  }

  if (filter_mask.defined()) {
    CHECK_EQ(filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << filter_mask.dim();
    CHECK_EQ(filter_mask.size(0), sample_logits.size(0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << filter_mask.size(0)
        << ", sample_logits.size(0)=" << sample_logits.size(0);
    CHECK_EQ(filter_mask.size(1), sample_logits.size(1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << filter_mask.size(1)
        << ", sample_logits.size(1)=" << sample_logits.size(1);
    sample_logits = sample_logits + filter_mask;
  }

  // apply temperatures, top-k and top-p
  apply_top_k_top_p(
      sample_logits, sample_temperatures, sample_top_k, sample_top_p);
  if (use_sample_indices) {
    logits.index_copy_(/*dim=*/0, params.sample_idxes, sample_logits);
  }

  CHECK(params.do_sample.defined()) << "params.do_sample must be defined";
  CHECK_EQ(params.do_sample.dim(), 1)
      << "params.do_sample must be 1D [num_seqs], got "
      << params.do_sample.sizes();
  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto probs =
      torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  torch::Tensor samples;
  if (params.all_random_sample) {
    samples = random_sample(probs);
  } else if (params.all_greedy_sample) {
    samples = greedy_sample(probs);
  } else {
    // mixed sample, sample both then choose based on do_sample
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(params.do_sample, random, greedy);
  }
  auto sample_indices = samples.to(torch::kLong);
  output.probs = probs.to(logits.dtype());
  output.next_tokens = sample_indices;
  log_sampler_debug(sample_logits, samples, params);

  if (params.logprobs) {
    if (FLAGS_enable_qwen3_reranker) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      sample_logits = sample_logits.index_select(/*dim=*/1, indices);
      auto logprobs = torch::log_softmax(
          sample_logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs =
        logprobs.gather(/*dim=*/-1, sample_indices.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
      auto [values, indices] =
          logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
#if defined(USE_MLU) || defined(USE_CUDA)
  xllm::kernel::RandomSampleParams params;
  params.logits = probs;
  return xllm::kernel::random_sample(params);
#endif
  if (probs.dim() == 3) {
    auto batch_size = probs.size(0);
    auto seq_len = probs.size(1);
    auto vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});
    auto sampled =
        flat_probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
    return sampled.reshape({batch_size, seq_len});
  } else {
    return probs.multinomial(/*num_samples=*/1, /*replacement=*/false)
        .flatten();
  }
}

}  // namespace xllm

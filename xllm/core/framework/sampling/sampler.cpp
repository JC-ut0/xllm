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
#include <sstream>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampling_params.h"

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

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params) const {
  SampleOutput output;
  LOG(INFO) << "[PREFILL_OUTPUT_DEBUG][Sampler] input logits="
            << tensor_debug_string(logits)
            << ", logits_head=" << tensor_head_values(logits)
            << ", selected_token_idxes="
            << tensor_debug_string(params.selected_token_idxes)
            << ", selected_token_idxes_head="
            << tensor_head_values(params.selected_token_idxes)
            << ", sample_idxes=" << tensor_debug_string(params.sample_idxes)
            << ", sample_idxes_head=" << tensor_head_values(params.sample_idxes)
            << ", do_sample=" << tensor_debug_string(params.do_sample)
            << ", do_sample_head=" << tensor_head_values(params.do_sample)
            << ", all_greedy=" << params.all_greedy_sample
            << ", all_random=" << params.all_random_sample
            << ", logprobs=" << params.logprobs
            << ", max_top_logprobs=" << params.max_top_logprobs;
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

  // apply temperatures, top-k and top-p
  apply_top_k_top_p(logits, params.temperatures, params.top_k, params.top_p);

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  LOG(INFO) << "[PREFILL_OUTPUT_DEBUG][Sampler] sample_logits="
            << tensor_debug_string(sample_logits)
            << ", sample_logits_head=" << tensor_head_values(sample_logits);

  CHECK(params.do_sample.defined()) << "params.do_sample must be defined";
  CHECK_EQ(params.do_sample.dim(), 1)
      << "params.do_sample must be 1D [num_seqs], got "
      << params.do_sample.sizes();
  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto probs = sample_logits;
  torch::Tensor samples;
  if (params.all_random_sample) {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    samples = random_sample(probs);
  } else if (params.all_greedy_sample) {
    samples = greedy_sample(probs);
  } else {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // mixed sample, sample both then choose based on do_sample
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(params.do_sample, random, greedy);
  }
  output.probs = probs.to(logits.dtype());
  output.next_tokens = samples;
  LOG(INFO) << "[PREFILL_OUTPUT_DEBUG][Sampler] output next_tokens="
            << tensor_debug_string(output.next_tokens)
            << ", next_tokens_head=" << tensor_head_values(output.next_tokens)
            << ", probs=" << tensor_debug_string(output.probs);

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
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
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
#if defined(USE_MLU)
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

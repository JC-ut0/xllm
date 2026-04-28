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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/tensor_helper.h"

namespace xllm {
namespace tensor_dump {

struct DumpContext {
  int64_t step = -1;
  int32_t rank = -1;
  int64_t layer = -1;
};

inline thread_local DumpContext g_context;

struct ModuleSequenceState {
  int64_t step = -1;
  int32_t rank = -1;
  int64_t layer = -1;
  int64_t next_seq = 1;
  std::string last_raw_module;
  std::string last_prefixed_module;
};

inline thread_local ModuleSequenceState g_module_sequence_state;

inline void set_step(int64_t step) { g_context.step = step; }

inline void clear_step() { g_context.step = -1; }

inline std::string module_with_sequence(const std::string& module) {
  if (g_context.step < 0 || g_context.rank < 0 || g_context.layer < 0) {
    return module;
  }

  if (g_module_sequence_state.step != g_context.step ||
      g_module_sequence_state.rank != g_context.rank ||
      g_module_sequence_state.layer != g_context.layer) {
    g_module_sequence_state.step = g_context.step;
    g_module_sequence_state.rank = g_context.rank;
    g_module_sequence_state.layer = g_context.layer;
    g_module_sequence_state.next_seq = 1;
    g_module_sequence_state.last_raw_module.clear();
    g_module_sequence_state.last_prefixed_module.clear();
  }

  if (g_module_sequence_state.last_raw_module == module &&
      !g_module_sequence_state.last_prefixed_module.empty()) {
    return g_module_sequence_state.last_prefixed_module;
  }

  const int64_t seq = g_module_sequence_state.next_seq++;
  std::ostringstream os;
  os << std::setw(2) << std::setfill('0') << seq << "_" << module;
  g_module_sequence_state.last_raw_module = module;
  g_module_sequence_state.last_prefixed_module = os.str();
  return g_module_sequence_state.last_prefixed_module;
}

class ScopedStep {
 public:
  explicit ScopedStep(int64_t step) : previous_step_(g_context.step) {
    set_step(step);
  }

  ~ScopedStep() { set_step(previous_step_); }

 private:
  int64_t previous_step_ = -1;
};

class ScopedRankLayer {
 public:
  ScopedRankLayer(int32_t rank, int64_t layer)
      : previous_rank_(g_context.rank), previous_layer_(g_context.layer) {
    g_context.rank = rank;
    g_context.layer = layer;
  }

  ~ScopedRankLayer() {
    g_context.rank = previous_rank_;
    g_context.layer = previous_layer_;
  }

 private:
  int32_t previous_rank_ = -1;
  int64_t previous_layer_ = -1;
};

inline bool env_flag_enabled(const char* value) {
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

inline bool env_flag_disabled(const char* value) {
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized == "0" || normalized == "false" || normalized == "off" ||
         normalized == "no";
}

inline bool enabled() {
  return !env_flag_disabled(std::getenv("XLLM_DUMP_TENSOR"));
}

inline std::string target_layers_raw() {
  const char* value = std::getenv("XLLM_DUMP_LAYER");
  if (value == nullptr || value[0] == '\0') {
    return "0";
  }
  return std::string(value);
}

inline std::string target_steps_raw() {
  const char* value = std::getenv("XLLM_DUMP_STEP");
  if (value == nullptr || value[0] == '\0') {
    return "0";
  }
  return std::string(value);
}

inline std::vector<int64_t> parse_non_negative_int64_list(
    const std::string& raw,
    const char* env_name,
    int64_t fallback_value,
    bool* warned) {
  std::vector<int64_t> values;
  std::stringstream ss(raw);
  std::string token;

  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(),
                               token.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                token.end());
    if (token.empty()) {
      continue;
    }

    try {
      size_t parsed_chars = 0;
      const int64_t value = std::stoll(token, &parsed_chars);
      if (parsed_chars != token.size() || value < 0) {
        throw std::invalid_argument("invalid value");
      }
      values.push_back(value);
    } catch (const std::exception&) {
      if (warned == nullptr || !*warned) {
        LOG(WARNING) << "Invalid " << env_name << "=" << raw << "; fallback to "
                     << fallback_value << ".";
        if (warned != nullptr) {
          *warned = true;
        }
      }
      return {fallback_value};
    }
  }

  if (values.empty()) {
    return {fallback_value};
  }
  return values;
}

inline std::vector<int64_t> target_layers() {
  static bool warned = false;
  return parse_non_negative_int64_list(
      target_layers_raw(), "XLLM_DUMP_LAYER", 0, &warned);
}

inline std::vector<int64_t> target_steps() {
  static bool warned = false;
  return parse_non_negative_int64_list(
      target_steps_raw(), "XLLM_DUMP_STEP", 0, &warned);
}

inline bool is_target_layer(int64_t layer) {
  const auto layers = target_layers();
  return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

inline bool is_target_step(int64_t step) {
  const auto steps = target_steps();
  return std::find(steps.begin(), steps.end(), step) != steps.end();
}

inline std::optional<std::filesystem::path> dump_root() {
  const char* root = std::getenv("DUMP_DIR");
  if (root == nullptr || root[0] == '\0') {
    static bool warned = false;
    if (enabled() && !warned) {
      LOG(WARNING) << "XLLM_DUMP_TENSOR is enabled but DUMP_DIR is not set; "
                      "tensor dump is disabled.";
      warned = true;
    }
    return std::nullopt;
  }
  return std::filesystem::path(root);
}

inline bool should_dump(int64_t layer) {
  return enabled() && is_target_step(g_context.step) &&
         is_target_layer(layer) && dump_root().has_value();
}

inline bool should_dump_current() {
  return g_context.rank >= 0 && should_dump(g_context.layer);
}

inline std::string tensor_info(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << "shape=" << tensor.sizes()
     << ", dtype=" << c10::toString(tensor.scalar_type())
     << ", device=" << tensor.device();
  return os.str();
}

inline std::string sanitize_path_component(std::string value) {
  for (char& c : value) {
    const auto uc = static_cast<unsigned char>(c);
    if (!(std::isalnum(uc) || c == '_' || c == '-' || c == '.')) {
      c = '_';
    }
  }
  return value;
}

inline std::filesystem::path tensor_path(int64_t step,
                                         int32_t rank,
                                         int64_t layer,
                                         const std::string& module,
                                         const std::string& name) {
  auto root = dump_root().value();
  return root / ("step" + std::to_string(step)) /
         ("rank" + std::to_string(rank)) / ("layer" + std::to_string(layer)) /
         sanitize_path_component(module) /
         (sanitize_path_component(name) + ".pt");
}

inline std::filesystem::path tensor_path(const std::filesystem::path& root,
                                         int64_t step,
                                         int32_t rank,
                                         int64_t layer,
                                         const std::string& module,
                                         const std::string& name) {
  return root / ("step" + std::to_string(step)) /
         ("rank" + std::to_string(rank)) / ("layer" + std::to_string(layer)) /
         sanitize_path_component(module) /
         (sanitize_path_component(name) + ".pt");
}

inline std::optional<std::filesystem::path> dump_root_or_log_skip(
    int32_t rank,
    int64_t layer,
    const std::string& module,
    const std::string& name,
    const torch::Tensor& tensor) {
  if (!enabled()) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": XLLM_DUMP_TENSOR is disabled, rank=" << rank
               << ", step=" << g_context.step << ", layer=" << layer << ", "
               << tensor_info(tensor);
    return std::nullopt;
  }
  if (!is_target_step(g_context.step)) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": current target steps are [" << target_steps_raw()
               << "], rank=" << rank << ", step=" << g_context.step
               << ", layer=" << layer << ", " << tensor_info(tensor);
    return std::nullopt;
  }
  if (!is_target_layer(layer)) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": current target layers are [" << target_layers_raw()
               << "], rank=" << rank << ", step=" << g_context.step
               << ", layer=" << layer << ", " << tensor_info(tensor);
    return std::nullopt;
  }
  auto root = dump_root();
  if (!root.has_value()) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": DUMP_DIR is not set, rank=" << rank
               << ", step=" << g_context.step << ", layer=" << layer << ", "
               << tensor_info(tensor);
    return std::nullopt;
  }
  return root;
}

inline void save_tensor(int32_t rank,
                        int64_t layer,
                        const std::string& module,
                        const std::string& name,
                        const torch::Tensor& tensor) {
  const std::string prefixed_module = module_with_sequence(module);
  if (!tensor.defined()) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << prefixed_module << "/" << name
               << ": tensor is undefined, rank=" << rank
               << ", step=" << g_context.step << ", layer=" << layer;
    return;
  }

  auto root = dump_root_or_log_skip(rank, layer, prefixed_module, name, tensor);
  if (!root.has_value()) {
    return;
  }

  try {
    const auto path = tensor_path(
        root.value(), g_context.step, rank, layer, prefixed_module, name);
    std::filesystem::create_directories(path.parent_path());
    auto saved = tensor.detach().to(torch::kCPU).contiguous();
    save_tensor_as_pickle(saved, path.string());
    LOG(INFO) << "[TENSOR_DUMP] saved " << prefixed_module << "/" << name
              << " to " << path.string() << ", rank=" << rank
              << ", step=" << g_context.step << ", layer=" << layer << ", "
              << tensor_info(tensor);
  } catch (const c10::Error& e) {
    LOG(ERROR) << "Failed to dump tensor " << prefixed_module << "/" << name
               << ": " << e.what_without_backtrace();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to dump tensor " << prefixed_module << "/" << name
               << ": " << e.what();
  }
}

inline void save_optional_tensor(int32_t rank,
                                 int64_t layer,
                                 const std::string& module,
                                 const std::string& name,
                                 const std::optional<torch::Tensor>& tensor) {
  if (tensor.has_value()) {
    save_tensor(rank, layer, module, name, tensor.value());
  } else {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": optional tensor has no value, rank=" << rank
               << ", step=" << g_context.step << ", layer=" << layer;
  }
}

inline void save_tensor(const std::string& module,
                        const std::string& name,
                        const torch::Tensor& tensor) {
  if (g_context.rank < 0 || g_context.layer < 0) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": dump rank/layer context is not set, rank="
               << g_context.rank << ", step=" << g_context.step
               << ", layer=" << g_context.layer << ", " << tensor_info(tensor);
    return;
  }
  save_tensor(g_context.rank, g_context.layer, module, name, tensor);
}

inline void save_optional_tensor(const std::string& module,
                                 const std::string& name,
                                 const std::optional<torch::Tensor>& tensor) {
  if (g_context.rank < 0 || g_context.layer < 0) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": dump rank/layer context is not set, rank="
               << g_context.rank << ", step=" << g_context.step
               << ", layer=" << g_context.layer;
    return;
  }
  if (!tensor.has_value()) {
    DLOG(INFO) << "[TENSOR_DUMP] skip " << module << "/" << name
               << ": optional tensor has no value, rank=" << g_context.rank
               << ", step=" << g_context.step << ", layer=" << g_context.layer;
    return;
  }
  save_tensor(module, name, tensor.value());
}

}  // namespace tensor_dump
}  // namespace xllm

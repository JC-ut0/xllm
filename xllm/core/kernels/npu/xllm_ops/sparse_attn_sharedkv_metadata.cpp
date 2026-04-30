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

#include <glog/logging.h>
#include <torch/library.h>

#include <algorithm>
#include <sstream>

#include "core/kernels/npu/pytorch_npu_helper.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

std::string tensor_debug_string(const at::Tensor& tensor) {
  std::ostringstream oss;
  oss << "defined=" << tensor.defined();
  if (!tensor.defined()) {
    return oss.str();
  }
  oss << ", device=" << tensor.device() << ", dtype=" << tensor.scalar_type()
      << ", sizes=" << tensor.sizes() << ", numel=" << tensor.numel();
  if (tensor.scalar_type() == at::kInt && tensor.numel() > 0) {
    constexpr int64_t kMaxPrintedValues = 32;
    const auto value_count =
        std::min<int64_t>(tensor.numel(), kMaxPrintedValues);
    auto cpu_tensor = tensor.detach().to(at::kCPU).contiguous().view({-1});
    const auto* values = cpu_tensor.data_ptr<int32_t>();
    oss << ", values=[";
    for (int64_t i = 0; i < value_count; ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << values[i];
    }
    if (tensor.numel() > kMaxPrintedValues) {
      oss << ",...";
    }
    oss << "]";
  }
  return oss.str();
}

std::string optional_tensor_debug_string(
    const c10::optional<at::Tensor>& tensor_opt) {
  std::ostringstream oss;
  oss << "has_value=" << tensor_opt.has_value();
  if (tensor_opt.has_value()) {
    oss << ", " << tensor_debug_string(tensor_opt.value());
  }
  return oss.str();
}

}  // namespace

at::Tensor sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cu_seqlens_q: "
            << optional_tensor_debug_string(cu_seqlens_q);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cu_seqlens_ori_kv: "
            << optional_tensor_debug_string(cu_seqlens_ori_kv);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cu_seqlens_cmp_kv: "
            << optional_tensor_debug_string(cu_seqlens_cmp_kv);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] seqused_q: "
            << optional_tensor_debug_string(seqused_q);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] seqused_kv: "
            << optional_tensor_debug_string(seqused_kv);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] num_heads_q: " << num_heads_q;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] num_heads_kv: "
            << num_heads_kv;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] head_dim: " << head_dim;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] batch_size: " << batch_size;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] max_seqlen_q: "
            << max_seqlen_q;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] max_seqlen_kv: "
            << max_seqlen_kv;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] ori_topk: " << ori_topk;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cmp_topk: " << cmp_topk;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cmp_ratio: " << cmp_ratio;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] ori_mask_mode: "
            << ori_mask_mode;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] cmp_mask_mode: "
            << cmp_mask_mode;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] ori_win_left: "
            << ori_win_left;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] ori_win_right: "
            << ori_win_right;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] layout_q: "
            << std::string(layout_q);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] layout_kv: "
            << std::string(layout_kv);
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] has_ori_kv: " << has_ori_kv;
  LOG(INFO) << "[SparseAttnSharedkvMetadata][arg] has_cmp_kv: " << has_cmp_kv;

  at::Device output_device = at::Device("npu");
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_ori_kv.has_value()) {
    output_device = cu_seqlens_ori_kv.value().device();
  } else if (cu_seqlens_cmp_kv.has_value()) {
    output_device = cu_seqlens_cmp_kv.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_kv.has_value()) {
    output_device = seqused_kv.value().device();
  }
  at::Tensor output =
      torch::empty({1024}, torch::dtype(torch::kInt32).device(output_device));

  auto valid_tensor = [output_device](
                          const c10::optional<at::Tensor>& tensor_opt) {
    return tensor_opt.has_value()
               ? tensor_opt.value()
               : torch::empty(
                     {0}, torch::dtype(torch::kInt32).device(output_device));
  };
  auto cu_seqlens_q_val = valid_tensor(cu_seqlens_q);
  auto cu_seqlens_ori_kv_val = valid_tensor(cu_seqlens_ori_kv);
  auto cu_seqlens_cmp_kv_val = valid_tensor(cu_seqlens_cmp_kv);
  auto seqused_q_val = valid_tensor(seqused_q);
  auto seqused_kv_val = valid_tensor(seqused_kv);

  // convert str
  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());

  EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata,
               cu_seqlens_q_val,
               cu_seqlens_ori_kv_val,
               cu_seqlens_cmp_kv_val,
               seqused_q_val,
               seqused_kv_val,
               num_heads_q,
               num_heads_kv,
               head_dim,
               batch_size,
               max_seqlen_q,
               max_seqlen_kv,
               ori_topk,
               cmp_topk,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_win_left,
               ori_win_right,
               layout_q_ptr,
               layout_kv_ptr,
               has_ori_kv,
               has_cmp_kv,
               output);

  return output;
}

}  // namespace xllm::kernel::npu

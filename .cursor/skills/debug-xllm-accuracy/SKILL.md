---
name: debug-xllm-accuracy
description: 在本项目中指导基于 xLLM 进行精度比对与首个异常 tensor 定位。参考 Ascend MSIT 文档《大模型精度问题定位全流程》，给出从端到端精度验证到逐层/逐 tensor 对比的完整流程与示例。当用户在本项目中提到“精度不对齐”“和参考实现结果不一致”或“想找出第一个有问题的 tensor”时使用。
---

# 基于 xLLM 的精度比对与首个异常 tensor 定位

> 参考流程来源：`大模型精度问题定位全流程`  
> 文档地址：`https://gitcode.com/Ascend/msit/blob/master/msit/docs/llm/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E5%85%A8%E6%B5%81%E7%A8%8B.md`

---

## 一、使用场景与前置条件

**在本仓库中，当用户遇到以下问题时使用本 Skill：**

- **端到端输出不对齐**：同一模型、同一输入，在 xLLM 上与参考实现（如 PyTorch、GPU 版本、官方脚本等）输出差异较大。
- **怀疑 NPU/算子/量化导致精度问题**：推理能跑通，但 loss/perplexity/评分明显劣化。
- **需要定位“第一个异常的 tensor”**：希望从端到端差异收窄到具体层/算子/tensor。

**默认前置条件：**

- 已有 **参考实现**（如：PyTorch + GPU）可得到“黄金结果”（golden result）。
- 已有 **xLLM 推理路径**（如：NPU 或 GPU 上的 xLLM 服务/离线脚本）。
- 可以在参考实现和 xLLM 中：
  - 使用 **相同的模型权重**（或已确认转换/加载逻辑一致）。
  - 使用 **完全相同的输入与随机种子**。
  - 在需要时，对中间 tensor 进行 dump（保存为 `.npy` 或其他可比较格式）。

---

## 二、整体流程概览（对应 MSIT 全流程）

当用户说“想做精度比对、找第一个错误 tensor”时，应引导其按照以下顺序进行：

1. **确认环境与模型一致性**
   - 确认：
     - 模型结构、参数版本、权重来源一致。
     - tokenizer / vocab / pad_id / eos_id 等配置一致。
     - 前处理（prompt 构造）、后处理（解码策略）一致。
   - 若不一致，优先修正配置而不是直接做 tensor 对比。

2. **端到端精度对比**
   - 选择若干典型输入（prompt 或样本），固定随机种子：
     - 同一 batch 输入到 **参考实现** 与 **xLLM**。
   - 对比：
     - 文本输出、logits、loss、perplexity 或 BLEU/ROUGE 等指标。
   - 目标：判断是否“确实存在显著差异”，并记录差异样例，后续用于 tensor 对比。

3. **确定对比入口（如 Encoder/Decoder 输入）**
   - 明确本次要对齐的 **子图/阶段**，例如：
     - Embedding + Encoder 层
     - Decoder block（自注意力 + FFN）
     - 特定算子子图（如 LayerNorm、MatMul）
   - 根据 MSIT 文档中的建议，将模型拆分为若干可独立运行的子模块，便于单独对比。

4. **开启中间 tensor dump（参考实现 & xLLM）**
   - 在参考实现和 xLLM 中，以 **统一命名规则** dump 中间 tensor：
     - 建议路径格式：
       - 参考实现：`golden/<stage>/<layer_name>_<tensor_name>.npy`
       - xLLM：`xllm/<stage>/<layer_name>_<tensor_name>.npy`
   - Dump 内容示例：
     - `input_ids`
     - `embedding_output`
     - 每一层的：
       - `attention_q`, `attention_k`, `attention_v`, `attention_output`
       - `ffn_input`, `ffn_output`
       - `residual`, `layernorm_output`
   - 尽量只 dump 必要 tensor，避免 I/O 过大。

5. **使用脚本自动对比 tensor 并定位首个异常点**
   - 编写/使用对比脚本：
     - 遍历同名 tensor 文件。
     - 计算 `max_abs_diff`, `max_rel_diff`, `cosine_similarity` 等指标。
     - 按模型前向顺序找到 **第一个超出阈值的 tensor**。
   - 输出报告包括：
     - 首个异常 tensor 的文件名/层名。
     - 该 tensor 的形状、最大绝对/相对误差。
     - （可选）打印若干位置的详细值对比。

6. **缩小到算子级别**
   - 根据第一个异常 tensor 所在层：
     - 进一步拆分该层内部子算子（例如：Q/K/V projection、Softmax、MatMul、Bias、LayerNorm）。
     - 按子算子再次 dump & 对比，直到找到首个不对齐的算子输出。
   - 结合 MSIT 文档的指导，进一步分析：
     - 是否为算子实现差异（如精度/约束）。
     - 是否为参数转换/shape/broadcast 逻辑错误。
     - 是否由量化/低精度格式引入的误差。

---

## 三、端到端比对与样例脚本（参考）

### 1. 端到端调用约定

在回答用户时，建议他们：

- **统一随机种子**：
  - 参考实现与 xLLM 都设置相同的 `seed`。
- **关闭/统一随机性操作**：
  - 如 dropout、随机 mask、采样策略（使用 greedy 或固定温度+top_k/top_p）。
- **保存输出**：
  - 将参考实现和 xLLM 的输出以 JSON 或文本形式存档，便于复查。

可建议的最小对比内容：

- **文本输出**：逐 token 比较是否在某个位置开始分叉。
- **logits 或 `logits_before_softmax`**：在首个分叉位置，对该步 logits 做数值对比。

---

## 四、中间 tensor dump 规范建议

当用户准备开始做 tensor 对比时，应提示他们：

1. **统一命名规范**
   - 建议采用以下格式：
     - `layer_{idx}_{block_name}_{tensor_name}.npy`
   - 示例：
     - `layer_00_attn_q.npz`
     - `layer_00_attn_k.npz`
     - `layer_00_attn_v.npz`
     - `layer_00_attn_out.npz`
     - `layer_00_ffn_in.npz`
     - `layer_00_ffn_out.npz`
   - 参考实现与 xLLM 必须使用 **完全相同的命名**，便于脚本自动遍历。

2. **统一数据格式**
   - 建议使用 `.npy` 或 `.npz`：
     - 参考实现中使用 `numpy.save`。
     - xLLM 中在相应位置将 tensor 转为 CPU numpy 后保存。
   - 注意：要保证 dtype 一致（例如都为 `float32`，避免混用 `float16/bfloat16` 直接比对）。

3. **统一 shape 与维度顺序**
   - 在回答时明确提醒：比对前要确认 shape 与维度顺序一致，如：
     - `[batch, seq_len, hidden_size]`
     - `[seq_len, batch, hidden_size]`
   - 如有差异，应在 dump 前对齐（如 transpose）再保存。

---

## 五、首个异常 tensor 自动对比脚本示例

> 以下为一个**概念性 Python 脚本示例**，用来对比 `golden/` 与 `xllm/` 两个目录下的同名 tensor 文件，并找出第一个超过阈值的 tensor。实际路径与阈值可根据项目需要调整。
>
> 如果你的 xLLM/NPU 环境 dump 出来主要是 **`.bin`**：
>
>- **`.bin` 通常是“纯二进制原始数据”**，必须配合 **shape/dtype 元信息**才能还原成数组进行比对。
>- 推荐做法是：先把 `.bin` **转换成 `.npy`**（或在脚本里直接加载 `.bin`），再用同一套对比逻辑跑“首个异常 tensor”定位。
>- 元信息来源常见两种：
>  - **同目录下的 sidecar `.json`**（文件名相同，仅后缀不同）。
>  - 或者你自己整理一个 `tensor_meta.json`（把每个 `.bin` 的 `dtype/shape` 写进去）。

```python
import os
import json
import numpy as np

GOLDEN_DIR = "golden"  # 参考实现 dump 的根目录
XLLM_DIR = "xllm"      # xLLM dump 的根目录

# 允许的误差阈值，可根据 MSIT 文档建议或项目经验调整
MAX_ABS_TOL = 1e-3
MAX_REL_TOL = 1e-3

# 可选：给 .bin 提供统一的元信息映射（当 sidecar json 不可用时）
# 文件格式示例（key 为相对路径）：
# {
#   "layer_00_attn_out.bin": {"dtype": "float16", "shape": [1, 2048, 4096]},
#   "layer_00_ffn_out.bin":  {"dtype": "float16", "shape": [1, 2048, 4096]}
# }
META_JSON = "tensor_meta.json"


def _load_meta_map(meta_path: str):
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _np_dtype(dtype_str: str):
    s = str(dtype_str).lower()
    if s in {"fp16", "float16"}:
        return np.float16
    if s in {"fp32", "float32"}:
        return np.float32
    if s in {"fp64", "float64"}:
        return np.float64
    if s in {"int32"}:
        return np.int32
    if s in {"int64"}:
        return np.int64
    if s in {"uint8"}:
        return np.uint8
    if s in {"bf16", "bfloat16"}:
        # bfloat16 的解析依赖具体 dump 格式；这里用 uint16 读入后再转 float32
        return np.uint16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def _bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    # bfloat16: 高 16 位为 float32 的高位
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def list_tensors(root):
    tensor_files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npy") or name.endswith(".npz") or name.endswith(".bin"):
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                tensor_files.append(rel)
    tensor_files.sort()
    return tensor_files


def load_tensor(path, rel_path: str, meta_map: dict):
    if path.endswith(".npz"):
        data = np.load(path)
        # 约定单 tensor 存在 key 为 "data" 或仅有一个数组
        if "data" in data:
            return data["data"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"多数组 npz 未约定 key: {path}")
    if path.endswith(".npy"):
        return np.load(path)

    # .bin：需要 shape/dtype
    meta = meta_map.get(rel_path)
    if meta is None:
        # 尝试读取同名 sidecar json（仅做兜底示例：真实字段名可能随环境不同）
        sidecar = path + ".json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            sidecar2 = os.path.splitext(path)[0] + ".json"
            if os.path.exists(sidecar2):
                with open(sidecar2, "r", encoding="utf-8") as f:
                    meta = json.load(f)

    if meta is None:
        raise ValueError(
            f"无法解析 .bin 元信息：{rel_path}。请提供同名 .json 或在 {META_JSON} 中配置 dtype/shape。"
        )

    # 约定 meta 至少包含 dtype/shape；如果 sidecar json 字段不同，需要你按实际改这里
    dtype = meta.get("dtype")
    shape = meta.get("shape")
    if dtype is None or shape is None:
        raise ValueError(f"meta 缺少 dtype/shape：{rel_path} meta={meta}")

    dt = _np_dtype(dtype)
    arr = np.fromfile(path, dtype=dt)
    arr = arr.reshape(shape)
    if str(dtype).lower() in {"bf16", "bfloat16"}:
        arr = _bf16_u16_to_f32(arr)
    return arr


def compare_tensor(name, golden_arr, xllm_arr):
    if golden_arr.shape != xllm_arr.shape:
        return {
            "ok": False,
            "reason": f"shape mismatch: golden {golden_arr.shape}, xllm {xllm_arr.shape}",
        }

    # 避免整型等类型的无意义相对误差
    if golden_arr.dtype != xllm_arr.dtype:
        return {
            "ok": False,
            "reason": f"dtype mismatch: golden {golden_arr.dtype}, xllm {xllm_arr.dtype}",
        }

    diff = xllm_arr.astype(np.float64) - golden_arr.astype(np.float64)
    abs_diff = np.abs(diff)
    max_abs = abs_diff.max()

    # 相对误差：避免除零
    denom = np.maximum(np.abs(golden_arr), 1e-8)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max()

    ok = (max_abs <= MAX_ABS_TOL) and (max_rel <= MAX_REL_TOL)

    result = {
        "ok": ok,
        "max_abs_diff": float(max_abs),
        "max_rel_diff": float(max_rel),
    }

    if not ok:
        # 可选：记录最大误差位置
        idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        result["max_diff_index"] = tuple(int(i) for i in idx)
        result["golden_value"] = float(golden_arr[idx])
        result["xllm_value"] = float(xllm_arr[idx])

    return result


def main():
    meta_map = _load_meta_map(META_JSON)
    golden_files = list_tensors(GOLDEN_DIR)
    xllm_files = list_tensors(XLLM_DIR)

    # 只对两边都存在的文件进行对比
    common_files = sorted(set(golden_files) & set(xllm_files))

    if not common_files:
        print("未找到可比对的公共 tensor 文件，请检查 dump 路径与命名。")
        return

    print(f"发现 {len(common_files)} 个公共 tensor，将按文件名顺序依次对比。")

    for rel_path in common_files:
        g_path = os.path.join(GOLDEN_DIR, rel_path)
        x_path = os.path.join(XLLM_DIR, rel_path)

        golden = load_tensor(g_path, rel_path, meta_map)
        xllm = load_tensor(x_path, rel_path, meta_map)

        res = compare_tensor(rel_path, golden, xllm)

        if not res["ok"]:
            print("=== 首个异常 tensor 发现 ===")
            print(f"tensor: {rel_path}")
            print(f"max_abs_diff = {res['max_abs_diff']:.6e}")
            print(f"max_rel_diff = {res['max_rel_diff']:.6e}")
            if "reason" in res:
                print(f"reason: {res['reason']}")
            if "max_diff_index" in res:
                print(f"max_diff_index: {res['max_diff_index']}")
                print(f"golden_value = {res['golden_value']}")
                print(f"xllm_value   = {res['xllm_value']}")
            print("请重点检查该 tensor 所在层/算子。")
            return

    print("所有公共 tensor 均在给定阈值内对齐，建议：")
    print("- 检查端到端后处理/解码逻辑是否一致；")
    print("- 适当收紧阈值或扩大 dump 范围再次对比。")


if __name__ == "__main__":
    main()
```

在回答用户时，应提醒其：

- 将 `GOLDEN_DIR` 与 `XLLM_DIR` 修改为实际 dump 路径。
- 根据模型规模与精度要求，调整 `MAX_ABS_TOL`、`MAX_REL_TOL`。
- 如果首个异常点出现在某一层（如 `layer_10_ffn_out.npy`），下一步要在该层内部进一步细分 dump，以便缩小到具体算子。

---

## 六、答复用户时的推荐话术流程

当用户说「xLLM 精度有问题」「如何找第一个错误的 tensor」时，建议按以下顺序回答：

1. **确认前置条件**
   - 提问并确认：
     - 是否有参考实现（如官方 PyTorch 脚本）及其版本。
     - 是否确保模型权重、tokenizer、前后处理配置一致。
   - 如有明显不一致，优先指导统一配置。

2. **建议先做端到端精度比对**
   - 告诉用户：
     - 固定随机种子。
     - 给出同一个输入，在参考实现与 xLLM 中跑一遍。
     - 保存输出，并大致比较差异位置（如第几个 token 开始不一致）。

3. **引导做中间 tensor dump**
   - 告诉用户在哪些关键层/阶段 dump tensor（embedding、每层 attention/FFN 输入输出等）。
   - 提醒统一命名/shape/dtype，并说明可用 `.npy/.npz` 格式。

4. **提供对比脚本与使用方法**
   - 给出上面的 Python 对比脚本或其简化版，并解释：
     - 如何设置目录与阈值。
     - 运行后如何解读“首个异常 tensor”的输出。

5. **异常定位后的下一步建议**
   - 根据异常 tensor 所在层：
     - 建议进一步拆分为算子级别对比。
     - 结合 MSIT 文档分析可能的根因（算子实现、量化、广播逻辑等）。
   - 如有需要，引导用户打开对应源码目录进一步排查。

通过上述流程，可较系统地从“端到端精度异常”定位到“首个异常 tensor/算子”，与《大模型精度问题定位全流程》中推荐的方法保持一致。


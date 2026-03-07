---
name: use-xllm-dit
description: 在本项目中指导基于 xLLM 的 DiT/FLUX 图像生成能力进行推理调用，主要围绕 image_generation.proto 中定义的请求与参数结构。当用户在本项目中提到用 xLLM 做文生图、图生图或基于 FLUX/DiT 的图片生成时使用。
---

# 使用 xLLM 的 DiT/FLUX 做图像生成

## 使用场景

在本仓库中，当用户希望基于 xLLM 内置的 DiT / FLUX 模型进行：

- 文生图（text-to-image）
- 图生图（image-to-image、inpaint 等）

时，使用本 Skill。

> 前置前提：已经按部署相关 Skill（如 `deploy-xllm-npu`）或项目文档，将带有 DiT/FLUX 能力的 xLLM 服务部署并接入到内部 RPC/网关中。

核心协议定义见：

- `xllm/proto/image_generation.proto`

---

## 一、核心协议结构概览（image_generation.proto）

### 1. ImageGenerationRequest

```protobuf
message ImageGenerationRequest {
  string model = 1;        // 使用的 FLUX 文生图模型，如 "flux-schnell"、"flux-dev"
  Input input = 2;         // 输入内容（prompt / 图像等）
  Parameters parameters = 3; // 生成参数（size/steps/seed 等）
  optional string user = 4;
  optional string request_id = 5;
}
```

### 2. Input（输入内容）

- `prompt`：主描述文本
- `prompt_2`：补充描述
- `negative_prompt` / `negative_prompt_2`：反向提示词，过滤不希望出现的元素
- `image` / `mask_image` / `control_image`：基于 base64 的图像输入，用于图生图/控制图等场景

常用最小配置一般只用到：

- `prompt`
- （可选）`negative_prompt`
- （可选）`image` / `mask_image` / `control_image`

### 3. Parameters（生成参数）

常用字段：

- `size`：生成图片尺寸（字符串形式，例如 `"1024*1024"`）
- `num_inference_steps`：推理步数
- `true_cfg_scale` / `guidance_scale`：控制提示词约束程度
- `num_images_per_prompt`：一次生成图片数量
- `seed`：随机种子（用于结果可复现）
- `strength`：图生图时原图保留程度（0~1）

### 4. ImageGenerationResponse

```protobuf
message ImageGenerationResponse {
  string id = 1;
  string object = 2;
  int64 created = 3;
  string model = 4;
  ImageGenerationOutput output = 5;
}

message ImageGenerationOutput {
  repeated ImageGenData results = 1;
}

message ImageGenData {
  optional string image = 1;  // 通常为 base64 编码图像
  int32 width = 3;
  int32 height = 4;
  int64 seed = 5;
}
```

> 在回答用户时，应明确说明：`results[i].image` 一般是 base64，需要在客户端解码后保存为图片文件。

---

## 二、典型调用模式（伪代码示例）

> 由于具体 RPC 服务名和网关路径依赖你们内部基础设施，本 Skill 只给出「构造请求 + 解析响应」的结构性示例，调用端可选择 gRPC、HTTP+JSON 或内部网关封装。

### 1. 文生图（text-to-image）最小请求示例

无论底层是 gRPC 还是 HTTP，只要遵守 `ImageGenerationRequest` 结构，典型字段如下：

- `model`: `"flux-schnell"` 或 `"flux-dev"`（具体可用模型以部署配置为准）
- `input.prompt`: 自然语言描述
- `parameters.size`: 如 `"1024*1024"`
- `parameters.num_inference_steps`: 如 `20`
- `parameters.seed`: 可选，固定后结果更易复现

在回答用户时，可以给出类似 JSON 形态的结构示例：

```json
{
  "model": "flux-schnell",
  "input": {
    "prompt": "a cat sitting on a futuristic neon-lit rooftop at night",
    "negative_prompt": "low quality, blurry"
  },
  "parameters": {
    "size": "1024*1024",
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0,
    "guidance_scale": 7.5,
    "seed": 123456789
  },
  "user": "demo-user",
  "request_id": "req-123"
}
```

### 2. 图生图/编辑场景要点

在图生图或带蒙版编辑的场景下，常见字段组合为：

- `input.image`: 原图的 base64 编码
- `input.mask_image`: 掩码图的 base64（可选）
- `parameters.strength`: 控制结果对原图的偏离程度（接近 1 意味着更接近 prompt，远离原图）

Skill 回答中，可以提示用户：

- 如何在客户端将本地图片文件读取成 base64
- 哪些字段必须同时提供（例如 `prompt` + `image`）

---

## 三、如何在项目中使用本 Skill 回答问题

当用户在本项目中说「用 xLLM 做图像生成 / FLUX / DiT」时：

1. **先确认部署情况**
   - 是否已经有带 DiT/FLUX 能力的 xLLM 服务在跑
   - 调用方式是 gRPC、HTTP 还是内部 SDK
2. **解释任务映射关系**
   - 文生图 → `ImageGenerationRequest.input.prompt + parameters`
   - 图生图 → `prompt + image (+ mask_image) + strength`
3. **给出结构化请求示例**
   - 使用上文 JSON 结构作为参考
   - 指出关键可调参数（尺寸、步数、seed 等）
4. **说明结果解析**
   - 告诉用户需要从 `response.output.results[i].image` 中取出 base64，解码后写入图片文件
5. **必要时引导查看源码**
   - 若用户需要自定义更底层行为，可引导其查看：
     - `xllm/proto/image_generation.proto`
     - `xllm/models/dit/` 下的相关模型实现


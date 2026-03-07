---
name: use-xllm-vlm
description: 在本项目中指导基于已部署的 xLLM 服务进行 VLM 推理与调用，包含在线 HTTP/OpenAI API 调用示例以及离线推理脚本入口。当用户在本项目中提到使用 xLLM 做图文理解或多模态推理时使用。
---

# 使用 xLLM 进行 VLM 推理

## 使用场景

在本仓库中，当用户已经 **部署并启动好 xLLM 服务**，希望：

- 通过 HTTP 或 OpenAI 兼容接口调用 **多模态 VLM 模型**（图文对话）
- 在离线脚本中使用 VLM 进行图片理解推理

时，使用本 Skill。

> 前置条件：请先按 `deploy-xllm-npu` 或其他部署 Skill / 文档启动好 xLLM 服务。

主要参考文档：

- 在线服务示例：`docs/zh/getting_started/online_service.md`
- 离线推理示例：`docs/zh/getting_started/offline_service.md`

---

## 一、在线 VLM 服务调用

### 1. HTTP API 示例（Python）

默认访问地址形如 `http://localhost:<port>/v1/chat/completions`，需要根据实际部署时的端口修改。  
下例以 `Qwen2.5-VL-7B-Instruct` 为例：

```python
import base64
import requests

api_url = "http://localhost:12345/v1/chat/completions"  # 按实际端口修改
image_url = ""  # 待分析图片的 URL


def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")
    return result


image_base64 = encode_image(image_url)
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    "model": "Qwen2.5-VL-7B-Instruct",
    "max_completion_tokens": 128,
}

response = requests.post(
    api_url,
    json=payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())
```

使用本 Skill 回答时，应指导用户：

- 将 `api_url` 中的端口改为实际暴露的端口
- 将 `image_url` 换成实际图片地址，或说明如何改为本地文件读取 + base64 编码

### 2. OpenAI 兼容接口示例

如果用户习惯用 OpenAI SDK，可以通过配置 `base_url` 指向 xLLM 服务：

```python
from openai import OpenAI
import base64
import requests

openai_api_key = "EMPTY"  # xLLM 默认可设为占位值
openai_api_base = "http://localhost:12345/v1"  # 按实际端口修改
image_url = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")
    return result


image_base64 = encode_image(image_url)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    model="Qwen2.5-VL-7B-Instruct",
    max_completion_tokens=128,
)

result = chat_completion.choices[0].message.content
print("Chat completion output:", result)
```

在回答用户时，说明：

- `openai_api_base` 需指向部署好的 xLLM 网关地址
- `model` 字段应与服务端加载的多模态模型名称一致

---

## 二、离线 VLM 推理脚本入口

项目为 VLM 离线推理提供了示例脚本，路径见：

- `examples/generate_vlm.py`

当用户希望在 **不走 HTTP 服务** 的情况下直接用 Python 脚本对图像进行推理时，可以引导其参考该脚本。  
在使用本 Skill 回答时，建议：

1. 先确认用户期望的运行方式：
   - 是否依赖现有在线服务接口
   - 是否更偏好在训练/批处理环境中直接跑 Python
2. 如果选择离线脚本：
   - 提示用户检查当前 Python 环境与依赖安装情况
   - 引导其打开并根据需要修改 `examples/generate_vlm.py`（如模型路径、设备类型等）

---

## 三、答复流程建议

当用户在本项目中说「用 xLLM 做图文理解 / 多模态推理」时：

1. **先确认服务是否已部署**  
   - 如果还没部署，引导其先使用部署相关 Skill（如 `deploy-xllm-npu`）。
2. **确认期望调用方式**  
   - 在线 HTTP / OpenAI 兼容接口  
   - 还是离线 Python 脚本
3. **给出对应的最小可运行示例**  
   - 替换 `api_url` / `openai_api_base`、`image_url` 等关键字段  
   - 根据用户环境说明如何从本地文件读取图片并编码为 base64
4. **提示调试手段**  
   - 请求失败时打印状态码与返回体  
   - 如有需要提醒检查端口、防火墙及模型名称是否正确匹配


# Mamba Cache Optimization

## Overview

xLLM supports Mamba Cache optimization for linear attention layers (e.g., Qwen-Next's Gated Delta Net). Qwen-Next is a hybrid architecture model that combines standard Transformer attention with linear attention, and its Prefix Cache implementation differs significantly from traditional LLM models.

### Architecture Characteristics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Qwen-Next Layer Structure                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Layer 0:    Linear Attention (GDN) - requires conv_cache, ssm_cache   │
│   Layer 1:    Linear Attention (GDN) - requires conv_cache, ssm_cache   │
│   Layer 2:    Linear Attention (GDN) - requires conv_cache, ssm_cache   │
│   Layer 3:    Standard Attention - requires key_cache, value_cache      │
│   ...                                                                    │
│   (Every 4th layer is standard attention, others are linear attention)  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mamba Cache Mode

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No caching for Mamba states | When Prefix Cache is disabled |
| `all` | Cache states at all block boundaries | Not applicable to Qwen-Next |
| `align` | Cache states only at aligned positions | **Recommended for Qwen-Next** |

## Usage

### Python API

```python
from xllm import LLM, RequestParams

# Create LLM instance with prefix cache and mamba cache enabled
llm = LLM(
    model="/path/to/qwen3-next-model",
    devices="npu:0",              # or "cuda:0"
    block_size=560,               # Recommended for better alignment
    disable_prefix_cache=False,   # Enable prefix cache (default)
    mamba_cache_mode="align",     # Enable mamba cache align mode
    max_tokens_per_batch=20480,
    max_seqs_per_batch=256,
)

# Generate text
output = llm.generate("Hello, please introduce yourself.", RequestParams())
print(output[0].text)

# Multiple requests with shared prefix will automatically reuse cache
prompts = [
    "Explain what machine learning is.",
    "Explain what deep learning is.",
    "Explain what natural language processing is.",
]
outputs = llm.generate(prompts, RequestParams())

llm.finish()
```

### Command Line

```bash
# Start server
python -m xllm.server \
    --model /path/to/qwen3-next-model \
    --devices npu:0 \
    --block_size 560 \
    --enable_prefix_cache=true \
    --mamba_cache_mode=align
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_prefix_cache` | `False` | Whether to disable prefix cache |
| `mamba_cache_mode` | `"none"` | Mamba cache mode |
| `block_size` | `128` | KV Cache block size |

### Recommended Configuration

For Qwen-Next models, the following configuration is recommended:

```python
llm = LLM(
    model="/path/to/qwen3-next-model",
    block_size=560,               # Consistent with vLLM
    disable_prefix_cache=False,   # Enable prefix cache
    mamba_cache_mode="align",     # Use align mode
)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Qwen-Next Prefix Cache Workflow                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Request 1: "Explain what machine learning is? Please provide details..."│
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Prefill (560 tokens)                                             │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  Standard Attention: Compute KV Cache → Save to PrefixCache  │ │  │
│  │  │  Linear Attention (GDN): Compute state → Save at pos 560     │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Request 2: "Explain what machine learning is? Briefly explain..."      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Prefix Match (hits first 560 tokens)                            │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  Standard Attention: Load KV Cache from PrefixCache          │ │  │
│  │  │  Linear Attention (GDN): Load state from MambaCache          │ │  │
│  │  │  → Skip computation for first 560 tokens, continue forward   │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Align Mode State Caching

```
Token Position:  0   100  200  300  400  500  600  700  800  900
                 │    │    │    │    │    │    │    │    │    │
                 ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼

Mode: "align" (Recommended for Qwen-Next)

Prefill Step 1 (560 tokens):
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX┌───┐
                                                   │ S │
pos:0                                             560│
                                                   └───┘

Prefill Step 2 (560 tokens):
[Reuse cached state]─────────────────────────────┐┌───┐
                                                 ││ S │
pos:560                                       1120│
                                                 └───┘

S = State saved at block boundary
X = Tokens computed in current step
```

## Important Notes

1. **block_size Alignment**: It is recommended to set `block_size=560`, consistent with vLLM, to ensure states are saved at the correct boundaries
2. **Auto Selection**: If `mamba_cache_mode` is not specified, the system will automatically select `align` mode when prefix cache is enabled
3. **State Dependency**: Linear attention layer states have recursive dependencies and must be saved and restored at correct positions

## Performance Impact

With Mamba Cache enabled on Qwen-Next models, for requests with shared prefixes:
- Reduce redundant computation, improve prefill speed
- Lower Time To First Token (TTFT)
- Increase overall throughput

## References

- vLLM Implementation: [PR #30877](https://github.com/vllm-project/vllm/pull/30877)
- Design Document: [qwen_next_prefix_cache_design.md](../design/qwen_next_prefix_cache_design.md)

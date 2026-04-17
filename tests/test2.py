import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Check what's actually being loaded
config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
print(f"Config dtype: {config.torch_dtype}")
print(f"Attention impl: {getattr(config, '_attn_implementation', 'default')}")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
print(f"Model dtype: {model.dtype}")
print(f"First param dtype: {next(model.parameters()).dtype}")
print(f"Device: {next(model.parameters()).device}")

# Check a specific weight value as a fingerprint
print(f"Weight fingerprint: {model.model.embed_tokens.weight[0, :5]}")

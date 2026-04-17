import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

print(f"Param dtype: {next(model.parameters()).dtype}")
print(f"Device: {next(model.parameters()).device}")

inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits[0, -1, :]
    print(f"Logit sum: {logits.sum().item()}")
    print(f"Logit mean: {logits.mean().item()}")
    print(f"Top token: {logits.argmax().item()}")

output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0]))

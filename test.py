import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

inputs = tokenizer("Your test prompt here", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token position
    
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    print(f"Entropy: {entropy.item()}")
    print(f"Top 10 logits: {torch.topk(logits, 10)}")
    print(f"Logit at token 4000: {logits[4000].item()}")
    print(f"Logit at token 20000: {logits[20000].item()}")

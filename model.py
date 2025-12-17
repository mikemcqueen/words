# model.py

MODEL_NAME = "Qwen/Qwen3-1.7B"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from info import info

def load_model():
    # Detect and set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        info("Using CPU")

    # Load model
    info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16
    )

    #torch.set_float32_matmul_precision('high')
    #model = torch.compile(model, mode="max-autotune")

    model = model.to(device)
    model.eval()

    info(f"Model loaded successfully on {device}")
        
    return device, model, tokenizer

def clear_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

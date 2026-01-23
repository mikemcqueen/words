# model.py

Q3 =       "Qwen/Qwen3-1.7B"
L2 =       "meta-llama/Llama-2-7b-hf"
G2_2 =     "google/gemma-2-2b"
G2_2b_it = "google/gemma-2-2b-it"
G3_4b_it = "google/gemma-3-4b-it"
#G2_9 = "google/gemma-2-9b"

from info import info
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

"""
# Example usage
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

def get_model_name(abbrev_name: str) -> str:
    name = abbrev_name.lower()
    if name == 'l2':
        return L2
    if name == 'q3':
        return Q3
    if name == 'g2':
        return G2_2
    if name == 'g2it':
        return G2_2b_it
    if name == 'g3it':
        return G3_4b_it
    print(f"get_model_name(): unknown model name: '{name}'")
    exit()

def is_gemma_2(name: str):
    return name.startswith("google/gemma-2")

def is_gemma_3(name: str):
    return name.startswith("google/gemma-3")

def is_gemma(name: str):
    return name.startswith("google/gemma")

def is_gemma_model(model):
    return is_gemma(model.name_or_path.lower())

def is_instruct(name: str):
    name = name.lower()
    return name.endswith("it") or name.endswith("instruct")

def is_instruct_model(model):
    return is_instruct(model.name_or_path.lower())

def gemmify_prompt(prompt: str) -> str:
    p = ""
    p += "<start_of_turn>user\n"
    p += prompt
    p += "<end_of_turn>\n"
    p += "<start_of_turn>model\n"
    return p

def specialize_prompt(model, prompt: str) -> str:
    gi = is_gemma_model(model) and is_instruct_model(model)
    return gemmify_prompt(prompt) if gi else prompt


def get_yesno_answer(model, response: str) -> str:
    gi = is_gemma_model(model)
    if gi:
        # hacky for gemma,
        lines = response.split('\n')
        return lines[3].strip().upper()
        if answer.startswith("YES"):
            return "YES"
        if answer.startswith("NO"):
            return "NO"
    return "OTHER"

def _load_model(name: str, device):
    if name == Q3 or is_gemma_2(name):
        model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )
        return model, False

    if is_gemma_3(name):
        model = Gemma3ForCausalLM.from_pretrained(
            name,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )
        return model, False

    if name == L2:
        # Configure INT8 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            #llm_int8_threshold=6.0,  # default threshold for outlier detection
            #llm_int8_has_fp16_weight=False,
        )

        # Load model with INT8 quantization
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=quantization_config,
            device_map={"": "cuda:0"}
        )
        return model, False
        
    print(f"_load_model(): unknown model name: '{name}'")
    exit()

def load_model(name):
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

    model_name = get_model_name(name)
    info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model, to = _load_model(model_name, device)

    if to:
        model, tokenizer = model.to(device)

    #torch.set_float32_matmul_precision('high')
    #model = torch.compile(model, mode="max-autotune")

    model.eval()

    info(f"Model loaded successfully on {device}")
        
    return device, model, tokenizer

def clear_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

# model.py

Q3 =        "Qwen/Qwen3-1.7B"
"""
G2_2 =      "google/gemma-2-2b"
G2_2b_it =  "google/gemma-2-2b-it"
G3_4b_it =  "google/gemma-3-4b-it"
G3_12b_it = "google/gemma-3-12b-it"
"""
G4_2b_it =  "google/gemma-4-E2B-it"
G4_4b_it =  "google/gemma-4-E4B-it"
GLM47 =     "mlx-community/GLM-4.7-Flash-4bit"

from info import info
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, AutoProcessor
from types import SimpleNamespace
import sys
import torch

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

"""
# Example usage
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

def get_model_name(abbrev_name: str) -> str:
    name = abbrev_name.lower()
    if name == 'q3':
        return Q3
    """
    if name == 'g2':
        return G2_2
    if name == 'g2it':
        return G2_2b_it
    if name == 'g3it':
        return G3_4b_it
    if name == 'g312it':
        return G3_12b_it
    """
    if name == 'g4it':
        return G4_2b_it
    if name == 'glm47':
        return GLM47
    print(f"get_model_name(): unknown model name: '{name}'")
    exit()

def is_gemma_2(name: str):
    return "gemma-2" in name

def is_gemma_3(name: str):
    return "gemma-3" in name

def is_gemma_4(name: str):
    return "gemma-4" in name

def is_gemma(name: str):
    return "gemma" in name.lower()

def is_mlx_model(name: str) -> bool:
    return name.startswith("mlx-community/")

def is_gemma_model(model):
    if getattr(model, 'is_mlx', False):
        return False
    return is_gemma(model.name_or_path.lower())

def is_glm_model(model):
    if getattr(model, 'is_mlx', False):
        name = getattr(model, 'model_name', '')
        return 'glm' in name.lower()
    return 'glm' in model.name_or_path.lower()

def is_instruct(name: str):
    name = name.lower()
    return "-it" in name or "-instruct" in name

def is_instruct_model(model):
    # hack for now
    return True
    """
    if getattr(model, 'is_mlx', False):
        return False
    return is_instruct(model.name_or_path.lower())
    """

def is_quantized(name: str):
    name = name.lower()
    return "-qat" in name

def is_quantized_model(model):
    if getattr(model, 'is_mlx', False):
        return False
    return is_quantized(model.name_or_path)

def needs_quantizing(name: str):
    name = name.lower()
    return "-12b" in name

def gemmify_prompt(prompt: str) -> str:
    p = ""
    p += "<start_of_turn>user\n"
    p += prompt
    p += "<end_of_turn>\n"
    p += "<start_of_turn>model\n"
    return p

def specialize_prompt(model, tokenizer, prompt: str) -> str:
    if getattr(model, 'is_mlx', False) and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gi = is_gemma_model(model) and is_instruct_model(model)
    return gemmify_prompt(prompt) if gi else prompt

def get_yesno_answer(model, response: str) -> str:
    lines = response.split('\n')

    if is_glm_model(model):
        # Walk lines from end, find '</think>' and return text after it
        for line in reversed(lines):
            if '</think>' in line:
                idx = line.rfind('</think>')
                return line[idx + len('</think>'):].strip().upper()
        return "OTHER"

    if is_gemma_model(model):
        # hacky for gemma
        return lines[3].strip().upper()

    return "OTHER"

def _load_model_mlx(name: str):
    if not HAS_MLX:
        print("MLX not available. Install with: pip install mlx-lm")
        exit()
    model, tokenizer = mlx_load(name)
    model.is_mlx = True
    model.model_name = name
    return model, tokenizer

def _load_model(name: str):
    if name == Q3 or is_gemma_2(name) or is_gemma_4(name):
        model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )
        return model, False

    if is_gemma_3(name):
        if needs_quantizing(name):
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Critical for Gemma 3
                bnb_4bit_use_double_quant=True,
            )
            model = Gemma3ForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"}
            )
        else:
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
    model_name = get_model_name(name)
    info(f"Loading {model_name}...")

    # MLX models use their own loading path
    if is_mlx_model(model_name):
        model, tokenizer = _load_model_mlx(model_name)
        info(f"Model loaded successfully via MLX")
        return None, model, tokenizer

    # Detect and set up device for torch models
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        info("Using CPU")

    if not is_gemma_4(model_name):
        print(f"not sure AutoProcessor works for anything but gemma 4, check model card: {model_name}")
        sys.exit()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #processor = AutoProcessor.from_pretrained(model_name)
    assert tokenizer, "no tokenizer"
    model, to = _load_model(model_name)

    if to:
        assert False, ".to() not supported atm"
        model, tokenizer = model.to(device)

    #torch.set_float32_matmul_precision('high')
    #model = torch.compile(model, mode="max-autotune")

    model.eval()

    info(f"Model loaded successfully on {device}")

    return SimpleNamespace(model=model, tokenizer=tokenizer)

def clear_cache(device):
    if device is None:
        return
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

def supports_thinking(model_id: str) -> bool:
    """Return True if the remote model supports thinking output."""
    return model_id.upper().startswith("GLM") or model_id.upper().startswith("QWEN")


def adjust_thinking(payload: dict, model_id: str, thinking: bool) -> None:
    """Modify thinking-related request fields for the given model."""
    if model_id.upper().startswith("GLM") or model_id.upper().startswith("QWEN"):
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}


def generate_text(model, tokenizer, prompt: str, max_tokens: int, sampler) -> str:
    """Unified generation for torch and MLX models."""
    if getattr(model, 'is_mlx', False):
        return mlx_generate(model, tokenizer, prompt=prompt,
                           sampler=sampler, max_tokens=max_tokens)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True)

# server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm.sample_utils import make_sampler
from src.model import load_model, specialize_prompt, get_yesno_answer, generate_text

# Read env var at import time (cheap + safe)
MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME="g3it"

DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.01
DEFAULT_MAX_TOKENS = 1000

app = FastAPI()

# Globals initialized during startup only
tokenizer = None
model = None

class Prompt(BaseModel):
    text: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temp: float = DEFAULT_TEMP
    top_p: float = DEFAULT_TOP_P
    min_p: float = DEFAULT_MIN_P


@app.on_event("startup")
def startup():
    global tokenizer, model

    print(f"Loading model '{MODEL_NAME}'")
    _, model, tokenizer = load_model(MODEL_NAME)


@app.post("/generate")
def generate(p: Prompt):
    print(F"DEBUG: generate({p})")
    prompt = specialize_prompt(model, tokenizer, p.text)
    sampler = make_sampler(temp=p.temp, top_p=p.top_p, min_p=p.min_p)
    response = generate_text(model, tokenizer, prompt, p.max_tokens, sampler=sampler)
    return {"response": response}

@app.post("/yesno")
def yesno(p: Prompt):
    print(F"DEBUG: generate({p})")
    prompt = specialize_prompt(model, tokenizer, p.text + " Answer with a YES or NO only.")
    sampler = make_sampler(temp=p.temp, top_p=p.top_p, min_p=p.min_p)
    response = generate_text(model, tokenizer, prompt, p.max_tokens, sampler=sampler)
    yesno = get_yesno_answer(model, response)
    print(f"yesno prompt: {prompt}")
    print(f"yesno: {yesno} response: {response}")
    return {"response": yesno}

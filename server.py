# server.py
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
#from transformers import AutoModelForCausalLM, AutoTokenizer
from model import load_model, specialize_prompt, get_yesno_answer

# Read env var at import time (cheap + safe)
MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME="g3it"

app = FastAPI()

# Globals initialized during startup only
tokenizer = None
model = None

class Prompt(BaseModel):
    text: str
    max_tokens: int = 1000


@app.on_event("startup")
def startup():
    global tokenizer, model

    print(f"Loading model '{MODEL_NAME}'")
    _, model, tokenizer = load_model(MODEL_NAME)


@app.post("/generate")
def generate(p: Prompt):
    print(F"DEBUG: generate({p})")
    prompt = specialize_prompt(model, p.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=p.max_tokens)

    return {
        "response": tokenizer.decode(out[0], skip_special_tokens=True)
    }

@app.post("/yesno")
def yesno(p: Prompt):
    print(F"DEBUG: generate({p})")
    prompt = specialize_prompt(model, p.text + " Answer with a YES or NO only.")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=p.max_tokens)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    yesno = get_yesno_answer(model, response)
    print(f"yesno prompt: {prompt}")
    print(f"yesno: {yesno} response: {response}")
    return {
        "response": yesno
    }

import readline
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/GLM-4.7-Flash-4bit")

# Creates a sampler with both Temperature and Top-P
sampler = make_sampler(
    temp=1.0, 
    top_p=0.95
)

verbose = True
print(f"verbose: {'on' if verbose else 'off'}")

while True:
    prompt = input("\n>>> ")
    if prompt.lower() in ("quit", "exit"):
        break
    if prompt.lower() == "verbose on":
        verbose = True
        print(f"verbose: on")
        continue
    if prompt.lower() == "verbose off":
        verbose = False
        print(f"verbose: off")
        continue

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=verbose,
            sampler=sampler,
            max_tokens=10000
        )
    print(f"\n{response}")

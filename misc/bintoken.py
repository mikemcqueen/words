# binary classify next token

import argparse
import asyncio
import httpx
import json
import math
import sys
import torch

from common import file_pair_generator, single_pair_generator, parse_yesno_response
from info import info
from model import load_model, clear_cache
from pathlib import Path

# Write tuples to file (already sorted by probability)
def dump_probs(filename, word_probs, args):
    path = args.data / filename
    prob_sum = 0.0
    with open(path, "w") as f:
        for word, prob in word_probs:
            prob_sum += prob
            f.write(f"{word},{prob}\n")
    info(f"prob_sum: {prob_sum}")

# Sort alphabetically and write just the words
def dump_words(filename, word_probs, args):
    words = sorted(word_probs, key=lambda x: x[0])
    path = args.data / filename
    with open(path, "w") as f:
        for word, _ in words:
            f.write(f"{word}\n")

def do(explorer, word: str, args):
    info(f"Word: {word}")
    clear_cache(explorer.device)

    word_log_probs, t = explorer.find_word_log_probs(args.context + word, args)
    top_words = explorer.to_word_probs(word_log_probs)

    if not args.dry_run:
        dump_probs(f"{word}.probs", top_words, args)
        dump_words(f"{word}.all", top_words, args)

    """
    threshold = 0.95
    prob_sum = 0.0
    for i, (word, prob) in enumerate(top_words):
        prob_sum += prob
        #if threshold and prob_sum >= threshold:
        #    print("--------threshold---------")
        #    threshold = None
        print(f"{word},{prob:.8f}")
    info(f"prob_sum: {prob_sum}")
    """
    info(f"Time: total: {t['total']:.3f}s  next: {t['next']:.3f}s  forward: {t['forward']:.3f}s  iters: {t['iters']}s")

def word_generator(filepath: str):
    """Generator that yields words from a text file one at a time."""
    with open(filepath, 'r') as f:
        for line in f:
            for word in line.split():
                yield word


def llm_prefill(mpd, prompt: str):
    messages = [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": prompt }
    ]

    # Process input
    text = mpd.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    #inputs = mpd.processor(text=text, return_tensors="pt").to(mpd.model.device)
    inputs = mpd.tokenizer(text, return_tensors="pt").to(mpd.model.device)
    return inputs

async def _remote_logprobs(args, prompt: str):
    from client import send_openai_request
    async with httpx.AsyncClient() as client:
        _, _, payload = await send_openai_request(
            client, args, prompt,
            model=args.model_id
        )
    return payload

def determine_yesno_tokens_remote(prompt: str, args):
    payload = asyncio.run(_remote_logprobs(args, prompt))
    content = payload["logprobs"]["content"]
    top_token = content[0]["token"]
    top_logprob = content[0]["logprob"]
    print(f"response: {repr(top_token)}  logprob={top_logprob:.7f}  prob={math.exp(top_logprob):.7f}", file=sys.stderr)
    top = min(args.top_k, args.top_logprobs)
    top_prob_sum = 0.0
    prob_sum = 0.0
    for i, entry in enumerate(content[0]["top_logprobs"]):
        token = entry["token"]
        lp = entry["logprob"]
        prob = math.exp(lp)
        prob_sum += prob
        if i < top:
            top_prob_sum += prob
        print(f"  {repr(token):<12} logprob={lp:>10.4f}  prob={prob:.7f}", file=sys.stderr)
    print(f"cumulative prob={prob_sum:.7f} top{top}={top_prob_sum:.7f}")

    return parse_yesno_response(top_token)

def determine_yesno_tokens(mpd, prompt: str, args):
    inputs = llm_prefill(mpd, prompt)

    # Generate and print response
    outputs = mpd.model.generate(**inputs, max_new_tokens=50)
    input_len = inputs["input_ids"].shape[-1]
    response = mpd.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(f"response:\n{response}")

    # Get next-token logits via forward pass
    with torch.no_grad():
        logits = mpd.model(**inputs).logits[0, -1, :]

    logits = logits.float()  # avoid float16 overflow in softmax (exp saturates at ~65504)
    topk_logits, topk_ids = torch.topk(logits, args.top_k)
    all_probs = torch.softmax(logits, dim=-1)
    topk_probs = torch.softmax(topk_logits, dim=-1)

    for i, token_id in enumerate(topk_ids.tolist()):
        decoded = mpd.tokenizer.decode([token_id])
        logit_val = topk_logits[i].item()
        all_prob = all_probs[token_id].item()
        topk_prob = topk_probs[i].item()
        print(f"{token_id:>6}: {decoded:<10} logit={logit_val:>10.4f}  vocab={all_prob:.9f}  top{args.top_k}={topk_prob:.9f}")


def main():
    DEFAULT_MODEL = "g4it"
    DEFAULT_TOP_K = 50
    DEFAULT_PORT = 8001

    parser = argparse.ArgumentParser(description='Determine next-word probability for a word or all unprocessed words in a file')
    parser.add_argument('-k', '--top-k', type=int, default=DEFAULT_TOP_K, help=f"select top-k tokens, default: {DEFAULT_TOP_K}")

    pair_group = parser.add_mutually_exclusive_group(required=True)
    pair_group.add_argument("-p", "--pair", type=str, help='single pair to evaluate (e.g., "foo,bar")')
    pair_group.add_argument("--pair-file", type=str, help='file containing pairs (one per line)')

    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"server port (default: {DEFAULT_PORT})")

    hostmodel = parser.add_mutually_exclusive_group()
    hostmodel.add_argument("--host", type=str, help="remote server host; mutually exclusive with --model")
    hostmodel.add_argument("-m", "--model", metavar='q3|g4it', type=str, default=DEFAULT_MODEL, help=f"local model, default: {DEFAULT_MODEL}")
    parser.add_argument("--key", type=str, help="API key")
    parser.add_argument("-s", "--system-prompt", type=str, help="system prompt file (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--thinking", type=str, choices=("on", "off"), default="off",
                        help="control model thinking/reasoning (default: off)")

    yngroup = parser.add_mutually_exclusive_group(required=True)
    yngroup.add_argument('-y', '--yes', action="store_true", help='Determine yes tokens')
    yngroup.add_argument('-n', '--no', action="store_true", help='Determine no tokens')

    parser.add_argument("--tlp", "--top-logprobs", dest="top_logprobs", type=int, default=20,
                        metavar='N', help="number of top log probs to request (default: 20)")

    args = parser.parse_args()
    args.thinking = args.thinking == "on"
    args.logprobs = True

    if args.system_prompt:
        p = Path(args.system_prompt)
        if not p.is_file():
            print(f"Error: system prompt file not found: {args.system_prompt}", file=sys.stderr)
            sys.exit(1)
        args.system_prompt = p.read_text().strip()


    def yes_no_none_str(fst: str, snd: str):
        if fst == "YES" or snd == "YES":
            return "YES"
        if fst == "NO" or snd == "NO":
            return "NO"
        return "None"
        

    def make_pair_prompt(pair):
        third_p4 = f"Is this a valid crossword clue? If an obvious, well-known answer immediately comes to mind, say YES. If you find yourself reasoning through associations to get there, say NO. Do not fabricate. Answer YES or NO only.\nClue: {pair}"

        third_p3 = f"Quick clue test. Answer YES if either these words already read like a familiar clue fragment, label, title, or stock description, or they directly point to one obvious, well-known answer on first read. Answer NO if they feel like two separate prompt words that only become useful after you combine associations or force a reading. Do not fabricate. Answer YES or NO only.\nClue: {pair}"

        third_p4_p2 = f"Is this a valid crossword clue? If an obvious, well-known answer immediately comes to mind, say YES. If you find yourself reasoning through associations to get there, say NO. Do not fabricate. Answer YES or NO only.\nClue: '{pair}'"

        fourth_p66 = f"Judge by recall, not invention. Answer YES when the pair already sounds like standard crossword shorthand for a familiar answer: an exact name or compound, a direct answer fragment such as a simple definition or negation, or a plain headword paired with its exact well-known identifier in ordinary clue or listing style. Answer NO when the wording is merely thematic, just piled-up description, or depends on paraphrase, synonym swap, branding, technical jargon, or obscurity. Answer YES or NO only.\nClue: '{pair}'\nAnswer: "

        return third_p3

    pairs = single_pair_generator(args.pair) if args.pair else file_pair_generator(args.pair_file)

    if args.host:
        from client import resolve_host, query_model_id
        args.host = resolve_host(args.host)
        args.model_id = query_model_id(args.host, args.port, args.key)
        for csv_pair in pairs:
            pair = csv_pair.replace(',', ' ')
            print(f"Testing {pair}", file=sys.stderr)
            fst = determine_yesno_tokens_remote(make_pair_prompt(pair), args)
            pair = ' '.join(reversed(csv_pair.split(',')))
            print(f"\nTesting {pair}", file=sys.stderr)
            snd = determine_yesno_tokens_remote(make_pair_prompt(pair), args)
            print(f"{csv_pair:<20} {yes_no_none_str(fst, snd)}")
            print("-----", file=sys.stderr)
    else:
        mt = load_model(args.model)
        for pair in pairs:
            print(f"Testing {pair}")
            determine_yesno_tokens(mt, make_pair_prompt(pair), args)


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f"ImportError: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

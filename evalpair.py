# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import asyncio
import itertools
import signal
import sys
from pathlib import Path

import httpx

signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit("SIGTERM"))

from client import SERVER_URL, run_concurrent, get_inference_params, send_yesno_request, send_openai_request, query_model_id, resolve_host, parse_nginx_upstream
from info import info
from model import load_model, is_gemma_model, specialize_prompt, generate_text, supports_thinking

def generate_response(model, tokenizer, prompt: str, skip_special = True, max_new_tokens: int = 1000 ) -> str:
    return generate_text(model, tokenizer, prompt, max_new_tokens)

def generate_response_fast(args, prompt: str) -> str:
    """Send prompt to server for generation using shared request functions (sync wrapper)."""
    async def _do():
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            if args.client == "yesno":
                actual, _, _ = await send_yesno_request(client, args, prompt)
            else:
                actual, _, _ = await send_openai_request(client, args, prompt)
            return actual
    try:
        return asyncio.run(_do())
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {args.host}:{args.port}", file=sys.stderr)
        print("Start the server with: ./server", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling server: {e}", file=sys.stderr)
        sys.exit(1)


def is_yes_response(model, response: str):
    """Return True for YES, False for NO, None for garbage/unexpected."""
    if not model:
        answer = response
    elif is_gemma_model(model):
        lines = response.split('\n')
        answer = lines[-2].strip() # NOTE: Gemma specific
    else:
        assert False, "is_yes_response not implemented for this model"
    upper = answer.upper()
    if upper.startswith("YES"):
        return True
    if upper.startswith("NO"):
        return False
    return None


def format_result(orig_pair, yes, as_txt, upstream=None):
    """Format a result line for display."""
    label = 'YES' if yes else ('NO' if yes is False else 'None')
    prefix = f"[{upstream.upper()}] " if upstream else ""
    return f"{prefix}{orig_pair} {label} {as_txt}"


async def async_request_and_classify(client, args, prompt):
    """Send request and return (YES/NO/None, truncated, upstream) classification."""
    if args.client == "yesno":
        response, _, meta = await send_yesno_request(client, args, prompt)
        return is_yes_response(None, response), False, meta.get('upstream')

    for attempt in range(3):
        response, _, payload = await send_openai_request(client, args, prompt)
        if payload.get("finish_reason") != "stop":
            continue
        result = is_yes_response(None, response)
        if result is not None:
            return result, False, payload.get('upstream')
    return None, True, None


async def process_pair_async(client: httpx.AsyncClient, args, ctx: str, orig_pair: str) -> tuple:
    """Process a single pair with retry on flipped pair if NO."""
    pair = orig_pair.replace(',', ' ')
    prompt = ctx.replace("{PAIR}", pair)
    yes, bad_response, upstream = await async_request_and_classify(client, args, prompt)
    as_txt = ""

    if not yes:
        pair = flip(orig_pair).replace(',', ' ')
        prompt = ctx.replace("{PAIR}", pair)
        yes, bad_flip_response, upstream = await async_request_and_classify(client, args, prompt)
        if yes:
            as_txt = f"as {pair}"

    return orig_pair, yes, as_txt, upstream


async def process_pairs_async(ctx: str, pairs, args):
    """Process all pairs with max_concurrent in flight at once."""
    yes_pairs = []
    no_pairs = []
    bad_pairs = []

    async def process(client, orig_pair):
        return await process_pair_async(client, args, ctx, orig_pair)

    last_processed = None
    try:
        async for result in run_concurrent(pairs, process, args.max_concurrent, args.timeout):
            print(format_result(*result))
            pair, yes, _, _ = result
            last_processed = pair
            if yes is True:
                yes_pairs.append(pair)
            elif yes is False:
                no_pairs.append(pair)
            else:
                bad_pairs.append(pair)
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
            print(f"\n{type(e).__name__}: {e}")
        msg = f"Interrupted. {len(yes_pairs)} YES, {len(no_pairs)} NO, {len(bad_pairs)} BAD."
        if last_processed:
            msg += f" Resume with: --last-pair {last_processed}"
        print(msg)

    return yes_pairs, no_pairs, bad_pairs, last_processed


def get_first_line(text: str) -> str:
    """Extract the first line from the text."""
    lines = text.split('\n')
    return lines[0] if lines else text

def parse_response(model, text: str) -> list[str]:
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if line == "model":
            return lines[idx+1:]

    return None

def custom_context(model, tokenizer, args, ctx: str, pair: str):
    if pair:
        ctx = ctx.replace("{PAIR}", pair)
    if model is None:
        prompt = ctx  # Server handles specialization
        response = generate_response_fast(args, prompt)
    else:
        prompt = specialize_prompt(model, tokenizer, ctx)
        response = generate_response(model, tokenizer, prompt, False)
    return prompt, response

def evaluate_pair_sync(model, tokenizer, args, ctx, orig_pair):
    """Evaluate a single pair synchronously, retrying flipped if NO."""
    pair = orig_pair.replace(',', ' ')
    _, response = custom_context(model, tokenizer, args, ctx, pair)
    yes = is_yes_response(model, response)
    as_txt = ""
    if yes is False:
        pair = flip(orig_pair).replace(',', ' ')
        _, response = custom_context(model, tokenizer, args, ctx, pair)
        yes = is_yes_response(model, response)
        if yes:
            as_txt = f"as {pair}"
    return orig_pair, yes, as_txt

def process_pairs_sync(ctx, pairs, model, tokenizer, args):
    """Process all pairs synchronously (local model path)."""
    yes_pairs, no_pairs, last_processed = [], [], None
    for orig_pair in pairs:
        orig_pair, yes, as_txt = evaluate_pair_sync(model, tokenizer, args, ctx, orig_pair)
        last_processed = orig_pair
        print(format_result(orig_pair, yes, as_txt))
        if yes:
            yes_pairs.append(orig_pair)
        else:
            no_pairs.append(orig_pair)
    return yes_pairs, no_pairs, last_processed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate word pairs with a language model")

    # Pair input: either --pair or --pair-file (mutually exclusive)
    pair_group = parser.add_mutually_exclusive_group(required=True)
    pair_group.add_argument('--pair', type=str, help='Single pair to evaluate (e.g., "foo,bar")')
    pair_group.add_argument('--pair-file', type=str, help='File containing pairs (one per line)')

    # Prompt input: either --prompt or (--prompt-file + --pid)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt', type=str, help='Prompt context (use {PAIR} as placeholder)')
    prompt_group.add_argument('--prompt-file', type=str, help='JSON file containing prompts')

    parser.add_argument('--pid', '--prompt-id', type=str, dest='prompt_id',
                        help='Prompt ID to use from prompt file')

    # Model: omit for server (default), provide for local model
    parser.add_argument('-m', '--model', metavar='MODEL', type=str, default=None,
                        help='Local model to load (omit to use server)')

    # Server/client options
    parser.add_argument('-s', '--system-prompt', type=str,
                        help='System prompt file (optional)')
    parser.add_argument('-c', '--client', choices=["yesno", "openai"], default="openai",
                        help="Client type: 'yesno' or 'openai' (default)")
    parser.add_argument('--host', default="http://localhost",
                        help="Server host (default: http://localhost)")
    parser.add_argument('--port', type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument('--key', type=str, help="API key")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument('--top_p', type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument('--min_p', type=float, default=0.01,
                        help="Min-p sampling (default: 0.01)")
    parser.add_argument('--repeat-penalty', '--rp', type=float, default=1.0,
                        help="Repeat penalty (default: 1.0)")
    parser.add_argument('--repeat-last-n', '--rln', type=int, default=32,
                        help="Repeat last n tokens (default: 32)")
    parser.add_argument('--think', action="store_true",
                        help="Enable thinking output (for models that support it)")
    parser.add_argument('--timeout', type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument('--save', nargs='?', const='', default=None, metavar='TAG',
                        help="Save YES/NO pairs to {pair-file}[.TAG].yes and .no")
    parser.add_argument('--last-pair', type=str, metavar='TYPE,PAIR',
                        help="Skip pairs in --pair-file up to and including this pair, then resume processing")
    parser.add_argument('--max-concurrent', '--mc', type=int, default=1,
                        help="Max concurrent requests (default: 1)")
    parser.add_argument('--nginx-config', type=str,
                        help='Path to nginx upstream config (auto-detected if omitted)')
    parser.add_argument('-n', '--num-pairs', type=int, metavar='N',
                        help='Process only N pairs from pair-file')

    args = parser.parse_args()

    if args.system_prompt:
        p = Path(args.system_prompt)
        if not p.is_file():
            print(f"Error: system prompt file not found: {args.system_prompt}", file=sys.stderr)
            sys.exit(1)
        args.system_prompt = p.read_text().strip()

    # Validate --prompt-file requires --pid
    if args.prompt_file and not args.prompt_id:
        parser.error("--prompt-file requires --pid/--prompt-id")

    # --save requires --pair-file
    if args.save is not None and not args.pair_file:
        parser.error("--save requires --pair-file")

    # --last-pair requires --pair-file
    if args.last_pair and not args.pair_file:
        parser.error("--last-pair requires --pair-file")

    # --num-pairs requires --pair-file
    if args.num_pairs and not args.pair_file:
        parser.error("--num-pairs requires --pair-file")

    return args

def flip(pair: str) -> str:
    flipped = pair.split(',')
    assert len(flipped) == 2, f"{pair} len: {len(flipped)}"
    flipped = [flipped[1], flipped[0]]
    return ",".join(flipped)

def single_pair_generator(pair_string):
    """Generator that yields a single pair string, then ends."""
    yield pair_string

def file_pair_generator(filename, last_pair=None):
    """Generator that yields pairs from a file. If last_pair is set, skip up to and including it, then yield the rest.
    Returns (generator, start_index) where start_index is the 0-based index of the first pair yielded."""
    def _gen(f, start_index):
        for line in f:
            pair = line.strip()
            if pair:
                yield pair

    f = open(filename, 'r')
    start_index = 0
    if last_pair:
        found = False
        for line in f:
            pair = line.strip()
            if pair:
                start_index += 1
            if pair == last_pair:
                found = True
                break
        if not found:
            f.close()
            raise ValueError(f"--last-pair '{last_pair}' not found in {filename}")

    return _gen(f, start_index), start_index

def load_prompt_from_file(prompt_file: str, prompt_id: str) -> str:
    """Load a prompt from a JSON prompt file by ID."""
    import json
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    for p in prompts:
        if p.get('id') == prompt_id:
            return p.get('text')
    raise ValueError(f"Prompt ID '{prompt_id}' not found in {prompt_file}")


def save_results(pair_file, tag, yes_pairs, no_pairs, bad_pairs, start_pair=0):
    """Write YES/NO/BAD pairs to results/{basename}.{tag}.{start_pair}.{yes,no,bad} files."""
    import os
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    basename = os.path.basename(pair_file)
    tag_part = f".{tag}" if tag else ""
    prefix = f"{results_dir}/{basename}{tag_part}.{start_pair}"
    for pairs, ext in [(yes_pairs, "yes"), (no_pairs, "no"), (bad_pairs, "bad")]:
        filename = f"{prefix}.{ext}"
        with open(filename, 'w') as f:
            for p in pairs:
                f.write(p + '\n')
        print(f"Saved {len(pairs)} {ext.upper()} pairs to {filename}")


def get_pairs(args):
    """Return (pairs_generator, start_pair_index) from args."""
    start_pair = 0
    if args.pair:
        return single_pair_generator(args.pair), start_pair
    pairs, start_pair = file_pair_generator(args.pair_file, last_pair=args.last_pair)
    if args.num_pairs:
        pairs = itertools.islice(pairs, args.num_pairs)
    return pairs, start_pair


def handle_results(args, yes_pairs, no_pairs, bad_pairs, last_processed, start_pair):
    """Print resume info and save results if requested."""
    total = len(yes_pairs) + len(no_pairs) + len(bad_pairs)
    if args.num_pairs and last_processed and total == args.num_pairs:
        print(f"Processed {args.num_pairs} pairs. Resume with: --last-pair {last_processed}")
    if args.save is not None and args.pair_file:
        save_results(args.pair_file, args.save, yes_pairs, no_pairs, bad_pairs, start_pair)


def main():
    args = parse_args()

    if args.model is None:
        # Server path (default)
        model, tokenizer = None, None
        args.host = resolve_host(args.host)

        # Auto-detect max-concurrent from nginx config when going through nginx
        if args.host in ("http://localhost", "http://127.0.0.1") and args.port == 80:
            upstream = parse_nginx_upstream(args.nginx_config)
            if upstream:
                user_set_mc = '--max-concurrent' in sys.argv or '--mc' in sys.argv
                if not user_set_mc:
                    args.max_concurrent = upstream['total_capacity']
                    server_desc = ", ".join(
                        f"{s['name'] or s['ip']}:{s['max_conns']}"
                        for s in upstream['servers']
                    )
                    print(f"nginx: auto-set --max-concurrent={args.max_concurrent} "
                          f"({len(upstream['servers'])} servers: {server_desc}, "
                          f"queue={upstream['queue_size']})")
                elif args.max_concurrent > upstream['total_capacity']:
                    print(f"Warning: --max-concurrent={args.max_concurrent} exceeds "
                          f"nginx capacity ({upstream['total_capacity']})")

        args.model_id = query_model_id(args.host, args.port, args.key)
        if args.think and not supports_thinking(args.model_id):
            print(f"Error: --think not supported for model '{args.model_id}'", file=sys.stderr)
            sys.exit(1)
    else:
        # Local model path
        _, model, tokenizer = load_model(args.model)

    ctx = args.prompt
    if args.prompt_file:
        ctx = load_prompt_from_file(args.prompt_file, args.prompt_id)

    pairs, start_pair = get_pairs(args)

    if args.model is None and args.pair_file:
        yes, no, bad, last = asyncio.run(process_pairs_async(ctx, pairs, args))
    else:
        yes, no, last = process_pairs_sync(ctx, pairs, model, tokenizer, args)
        bad = []

    handle_results(args, yes, no, bad, last, start_pair)

if __name__ == "__main__":
    main()

# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import asyncio
import itertools
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx

signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit("SIGTERM"))

from client import run_concurrent, get_inference_params, send_yesno_request, send_openai_request, query_model_id, resolve_host, get_server_name, auto_detect_max_concurrent
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


def fmt_duration(s):
    """Format seconds as compact duration: 5s, 3m30s, 2h03m04s, 1 day, 2h03m04s."""
    td = str(timedelta(seconds=round(s)))
    if ", " in td:
        days, hms = td.split(", ")
    else:
        days, hms = None, td
    h, m, sec = hms.split(":")
    parts = []
    if h != "0":
        parts.append(f"{h}h")
    if parts or m != "00":
        parts.append(f"{int(m)}m")
    parts.append(f"{sec}s")
    result = "".join(parts)
    if days:
        result = f"{days}, {result}"
    return result


def get_retry_stats(retry_map):
    """Return (total_retries, total_duration) across all pairs/hosts/reasons."""
    total_retries = 0
    total_duration = 0.0
    for hosts in retry_map.values():
        for reasons in hosts.values():
            for durs in reasons.values():
                total_retries += len(durs)
                total_duration += sum(durs)
    return total_retries, total_duration


def print_retries(retry_map, file=sys.stdout, successful_requests=None):
    """Print retry map as: pair  {host reason: N @ duration, ...},{host2 ...}"""
    max_len = max(len(pair) for pair in retry_map)
    print("Retries:", file=file)
    for pair, hosts in retry_map.items():
        parts = []
        for host, reasons in hosts.items():
            counts = ", ".join(f"{r}: {len(durs)} @ {fmt_duration(sum(durs))}" for r, durs in reasons.items())
            parts.append(f"{{{host} {counts}}}")
        print(f"  {pair:<{max_len}}  {','.join(parts)}", file=file)
    total_retries, retry_duration = get_retry_stats(retry_map)
    header = f"{total_retries} retries"
    if successful_requests is not None:
        pct = round(total_retries / (successful_requests + total_retries) * 100)
        header += f" ({pct}% of total, avg {fmt_duration(retry_duration / total_retries)})"
    print(header, file=file)


def merge_retries(a, b):
    """Merge {host: {reason: [durations]}} dicts, extending lists."""
    for host, reasons in b.items():
        if host not in a:
            a[host] = {}
        for reason, durations in reasons.items():
            a[host].setdefault(reason, []).extend(durations)
    return a


def format_result(orig_pair, yes, flipped, upstream=None):
    """Format a result line for display."""
    label = 'YES' if yes else ('NO' if yes is False else 'None')
    prefix = f"[{upstream.upper()}] " if upstream else ""
    as_txt = f" as {flip(orig_pair).replace(',', ' ')}" if flipped else ""
    return f"{prefix}{orig_pair} {label}{as_txt}"


async def async_request_and_classify(client, args, prompt, label=None):
    """Send request and return (YES/NO/None, upstream, retries) classification."""
    if args.client == "yesno":
        response, _, meta = await send_yesno_request(client, args, prompt, label=label)
        return is_yes_response(None, response), meta.get('upstream'), {}

    retries = {}
    def record_retry(payload, duration):
        host = payload.get('upstream') or get_server_name(args.host) or args.host
        reason = payload.get('finish_reason')
        retries.setdefault(host, {})
        retries[host].setdefault(reason, []).append(duration)

    for attempt in range(3):
        t0 = time.monotonic()
        response, _, payload = await send_openai_request(client, args, prompt, label=label)
        duration = time.monotonic() - t0
        if payload.get("finish_reason") != "stop":
            record_retry(payload, duration)
            continue
        result = is_yes_response(None, response)
        if result is not None:
            return result, payload.get('upstream'), retries
        record_retry(payload, duration)
    return None, None, retries


async def process_pair_async(client: httpx.AsyncClient, args, ctx: str, orig_pair: str) -> tuple:
    """Process a single pair with retry on flipped pair if NO."""
    pair = orig_pair.replace(',', ' ')
    prompt = ctx.replace("{PAIR}", pair)
    yes, upstream, orig_retries = await async_request_and_classify(client, args, prompt, label=orig_pair)
    flipped = False
    retry_map = {}

    if orig_retries:
        retry_map[orig_pair] = orig_retries

    if not yes:
        orig_yes = yes
        flipped_pair = flip(orig_pair)
        pair = flipped_pair.replace(',', ' ')
        prompt = ctx.replace("{PAIR}", pair)
        flip_yes, upstream, flip_retries = await async_request_and_classify(client, args, prompt, label=flipped_pair)
        if flip_yes:
            yes = flipped = True
        elif orig_yes is None:
            yes = None  # orig was bad, flip wasn't YES — can't trust it
        if flip_retries:
            retry_map[flipped_pair] = flip_retries

    return orig_pair, yes, flipped, upstream, retry_map


async def process_pairs_async(ctx: str, pairs, args, writer=None):
    """Process all pairs with max_concurrent in flight at once."""
    yes_pairs = []
    no_pairs = []
    bad_pairs = []
    retry_map = {}
    flips = 0

    async def process(client, item):
        seq, orig_pair = item
        result = await process_pair_async(client, args, ctx, orig_pair)
        return (seq, *result)

    numbered_pairs = enumerate(pairs, start=writer.next_seq if writer else 0)
    try:
        async for result in run_concurrent(numbered_pairs, process, args.max_concurrent, args.timeout):
            seq, pair, yes, flipped, upstream, pair_retries = result
            print(format_result(pair, yes, flipped, upstream))
            retry_map.update(pair_retries)
            if flipped:
                flips += 1
            if yes is True:
                classification = "yes"
                yes_pairs.append(pair)
            elif yes is False:
                classification = "no"
                no_pairs.append(pair)
            else:
                classification = "bad"
                bad_pairs.append(pair)
            if writer:
                writer.submit(seq, pair, classification)
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
            print(f"\n{type(e).__name__}: {e}")
        msg = f"Interrupted. {len(yes_pairs)} YES, {len(no_pairs)} NO, {len(bad_pairs)} BAD."
        if args.save:
            msg += " Re-run to continue."
        print(msg)

    successful = len(yes_pairs) + flips + 2 * (len(no_pairs) + len(bad_pairs))
    return yes_pairs, no_pairs, bad_pairs, retry_map, successful


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
    flipped = False
    if yes is False:
        pair = flip(orig_pair).replace(',', ' ')
        _, response = custom_context(model, tokenizer, args, ctx, pair)
        yes = is_yes_response(model, response)
        if yes:
            flipped = True
    return orig_pair, yes, flipped

def process_pairs_sync(ctx, pairs, model, tokenizer, args):
    """Process all pairs synchronously (local model path)."""
    yes_pairs, no_pairs, bad_pairs, last_processed = [], [], [], None
    for orig_pair in pairs:
        orig_pair, yes, flipped = evaluate_pair_sync(model, tokenizer, args, ctx, orig_pair)
        last_processed = orig_pair
        print(format_result(orig_pair, yes, flipped))
        if yes is True:
            yes_pairs.append(orig_pair)
        elif yes is False:
            no_pairs.append(orig_pair)
        else:
            bad_pairs.append(orig_pair)
    return yes_pairs, no_pairs, bad_pairs, last_processed

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
    parser.add_argument('--host', default="localhost",
                        help="Server host (default: localhost)")
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
    parser.add_argument('--save', action='store_true',
                        help="Save YES/NO/BAD pairs to results/ directory (auto-resumes from existing files)")
    parser.add_argument('--max-concurrent', '--mc', type=int, default=1,
                        help="Max concurrent requests (default: 1)")
    parser.add_argument('--nginx-config', type=str,
                        help='Path to nginx upstream config (auto-detected if omitted)')
    parser.add_argument('-n', '--num-pairs', type=int, metavar='N',
                        help='Process only N pairs from pair-file')
    parser.add_argument('--tag', type=str,
                        help='Tag to append to result filename')

    args = parser.parse_args()

    if args.system_prompt:
        p = Path(args.system_prompt)
        if not p.is_file():
            print(f"Error: system prompt file not found: {args.system_prompt}", file=sys.stderr)
            sys.exit(1)
        args.system_prompt_filename = p.name
        args.system_prompt = p.read_text().strip()
    else:
        args.system_prompt_filename = None

    # Validate --prompt-file requires --pid
    if args.prompt_file and not args.prompt_id:
        parser.error("--prompt-file requires --pid/--prompt-id")

    # --save requires --pair-file
    if args.save and not args.pair_file:
        parser.error("--save requires --pair-file")

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

def file_pair_generator(filename):
    """Generator that yields pairs from a file."""
    with open(filename, 'r') as f:
        for line in f:
            pair = line.strip()
            if pair:
                yield pair

def load_prompt_from_file(prompt_file: str, prompt_id: str) -> str:
    """Load a prompt from a JSON prompt file by ID."""
    import json
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    for p in prompts:
        if p.get('id') == prompt_id:
            return p.get('text')
    raise ValueError(f"Prompt ID '{prompt_id}' not found in {prompt_file}")


class IncrementalWriter:
    def __init__(self, prefix, next_seq=0):
        self.prefix = prefix
        self.files = {
            "yes": open(f"{prefix}.yes", "a"),
            "no":  open(f"{prefix}.no", "a"),
            "bad": open(f"{prefix}.bad", "a"),
        }
        self.next_seq = next_seq
        self.buffer = {}         # {seq: (pair, classification)}

    def submit(self, seq, pair, classification):
        """Buffer a result; flush all consecutive completed results."""
        self.buffer[seq] = (pair, classification)
        while self.next_seq in self.buffer:
            p, c = self.buffer.pop(self.next_seq)
            self.files[c].write(p + "\n")
            self.files[c].flush()
            self.next_seq += 1

    def close(self):
        for f in self.files.values():
            f.close()


def make_result_prefix(args):
    """Build the result file prefix from args."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    basename = os.path.basename(args.pair_file)
    if args.prompt_file:
        prompt_source = Path(args.prompt_file).stem
        pid = args.prompt_id
    else:
        prompt_source = "manual"
        pid = "manual"
    base_name = f"{basename}_{prompt_source}_{pid}"
    server_name = get_server_name(args.host)
    if server_name:
        base_name += f"_{server_name}"
    if args.system_prompt_filename:
        base_name += f"_{Path(args.system_prompt_filename).stem}"
    base_name += f"_mc{args.max_concurrent}"
    if args.tag:
        base_name += f".{args.tag}"
    return f"{results_dir}/{base_name}"


def count_completed_pairs(prefix):
    """Count completed pairs across .yes/.no/.bad files."""
    count = 0
    for ext in ("yes", "no", "bad"):
        path = f"{prefix}.{ext}"
        if os.path.exists(path):
            with open(path) as f:
                count += sum(1 for line in f if line.strip())
    return count


def save_results(args, yes_pairs, no_pairs, bad_pairs, retry_map=None, successful_requests=None, writer=None):
    """Write results. If writer was used, only write .retries file."""
    prefix = make_result_prefix(args)
    if not writer:
        for pairs, ext in [(yes_pairs, "yes"), (no_pairs, "no"), (bad_pairs, "bad")]:
            filename = f"{prefix}.{ext}"
            with open(filename, 'w') as f:
                for p in pairs:
                    f.write(p + '\n')
            print(f"Saved {len(pairs)} {ext.upper()} pairs to {filename}")
    if retry_map:
        filename = f"{prefix}.retries"
        with open(filename, 'w') as f:
            print_retries(retry_map, file=f, successful_requests=successful_requests)
        print(f"Saved retries to {filename}")


def get_pairs(args, skip=0):
    """Return pairs generator from args, optionally skipping the first `skip` pairs."""
    if args.pair:
        return single_pair_generator(args.pair)
    pairs = file_pair_generator(args.pair_file)
    if skip:
        pairs = itertools.islice(pairs, skip, None)
    if args.num_pairs:
        pairs = itertools.islice(pairs, args.num_pairs)
    return pairs


def handle_results(args, yes_pairs, no_pairs, bad_pairs, retry_map=None, successful_requests=None, writer=None):
    """Print batch info and save results if requested."""
    total = len(yes_pairs) + len(no_pairs) + len(bad_pairs)
    if args.num_pairs and total == args.num_pairs:
        print(f"Processed {args.num_pairs} pairs. Re-run to continue.")
    if args.save and args.pair_file:
        save_results(args, yes_pairs, no_pairs, bad_pairs, retry_map, successful_requests, writer)


def main():
    args = parse_args()

    if args.model is None:
        # Server path (default)
        model, tokenizer = None, None
        args.host = resolve_host(args.host)

        auto_detect_max_concurrent(args)

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

    # When --save, count existing results and resume from where we left off
    skip = 0
    writer = None
    if args.save and args.pair_file:
        prefix = make_result_prefix(args)
        skip = count_completed_pairs(prefix)
        if skip:
            print(f"Resuming: skipping {skip} completed pairs")
        writer = IncrementalWriter(prefix, next_seq=skip)

    pairs = get_pairs(args, skip=skip)

    retry_map = None
    successful = None
    if args.model is None and args.pair_file:
        t0 = time.monotonic()
        try:
            yes, no, bad, retry_map, successful = asyncio.run(process_pairs_async(ctx, pairs, args, writer))
        finally:
            if writer:
                writer.close()
        duration = time.monotonic() - t0
        num_pairs = len(yes) + len(no) + len(bad)
        total_retries, _ = get_retry_stats(retry_map) if retry_map else (0, 0)
        total_requests = successful + total_retries
        print(f"Total {num_pairs} pairs ({total_requests} requests) in {fmt_duration(duration)}")
        if retry_map:
            print_retries(retry_map, successful_requests=successful)
    else:
        yes, no, bad, last = process_pairs_sync(ctx, pairs, model, tokenizer, args)

    handle_results(args, yes, no, bad, retry_map, successful, writer)

if __name__ == "__main__":
    main()

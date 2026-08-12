# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import asyncio
import itertools
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import httpx

signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit("SIGTERM"))

from src.classify import classify_pairs_async
from src.client import run_concurrent, send_openai_request, resolve_model_id, get_compact_server_name, is_llamacpp_backend, build_url, ssl_verify
from src.common import add_inference_args, parse_on_off, load_prompts_from_file, single_pair_generator, file_pair_generator, flip_pair, eval_with_flipped_retry, parse_yesno_response
from src.info import info
from src.model import is_gemma

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


def get_system_prompt_templated(args):
    if "phi" in args.model_id:
        return "<|im_start|>system<|im_sep|>" + args.system_prompt + "<|im_end|>"

    if not "Qwen" in args.model_id:
        print(f"{args.model_id} has not yet been verified to work with a system prompt")
        sys.exit()

    return "<|im_start|>system\n" + args.system_prompt + "<|im_end|>\n"


def get_completion_prefix(args):
    if is_gemma(args.model_id):
        return "<|turn>user\n"
    
    if "phi" in args.model_id:
        return "<|im_start|>user<|im_sep|>"

    """
    # NOTE HARDCODED DATE:
    if "gpt-oss" in args.model_id:
        return "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-04-17\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>"
    """

    # Qwen family, others
    return "<|im_start|>user\n"

def send_anchor_prefix(args, ctx: str) -> None:
    """Send prompt prefix up to 'Clue' to v1/completions for KV cache warming."""
    m = re.search(r'^(.*Clue)', ctx, re.DOTALL)
    if not m:
        return
        #raise ValueError("send_anchor_prefix: no 'Clue' found in prompt context")
    prefix = ""
    if args.system_prompt:
        prefix = get_system_prompt_templated(args)
    prefix += get_completion_prefix(args) + m.group(1)
    url = build_url(args.host, args.port, "/v1/completions")
    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    payload = {"prompt": prefix, "cache_prefix": True}
    if not args.thinking:
        payload["reasoning_budget_tokens"] = 0
        payload["reasoning_budget_start_tag"] = "<think>"
        payload["reasoning_budget_end_tag"] = "</think>"
    response = httpx.post(url, headers=headers, json=payload, timeout=args.timeout, verify=ssl_verify(args.host))
    response.raise_for_status()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate word pairs with a language model")

    # Pair input: either --pair or --pair-file (mutually exclusive)
    pair_group = parser.add_mutually_exclusive_group(required=True)
    pair_group.add_argument('--pair', type=str, help='Single pair to evaluate (e.g., "foo,bar")')
    pair_group.add_argument('--pair-file', type=str, help='File containing pairs (one per line)')

    # Prompt input: either --prompt or (--prompt-file + --pid)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt', type=str, help='Prompt context (use {PAIR} as placeholder)')
    prompt_group.add_argument('--prompt-file', type=str, help='JSONL file containing prompts')

    pid_group = parser.add_mutually_exclusive_group()
    pid_group.add_argument('--pid', '--prompt-id', type=str, dest='prompt_id',
                           help='Prompt ID to use from prompt file')
    pid_group.add_argument('--all', action='store_true',
                           help='Run all prompts in prompt file; with --save, creates one result file per prompt')

    # Server/client options
    parser.add_argument('-s', '--system-prompt', type=str,
                        help='System prompt file (optional)')
    parser.add_argument('--host', default="localhost",
                        help="Server host (default: localhost)")
    parser.add_argument('--port', type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument('--key', type=str, help="API key")
    add_inference_args(parser)
    parser.add_argument('--timeout', type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument('--save', action='store_true',
                        help="Save YES/NO/BAD pairs to results/ directory (auto-resumes from existing files)")
    parser.add_argument('-r', '--results-dir', type=Path, default=Path("results"),
                        help="Directory for result files (default: results/)")
    parser.add_argument('--max-concurrent', '--mc', type=int, default=1,
                        help="Max concurrent requests (default: 1)")
    #parser.add_argument('--nginx-config', type=str, help='Path to nginx upstream config (auto-detected if omitted)')
    parser.add_argument('-n', '--num-pairs', type=int, metavar='N',
                        help='Process at most N pairs from pair-file')
    parser.add_argument('--tag', type=str,
                        help='Tag to append to result filename')
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show actual response and message")
    parser.add_argument("-q", "--quiet", action="store_true",
                      help="Suppress REQUEST START/END messages")
    parser.add_argument('--lp', '--logprobs', dest='logprobs', type=parse_on_off,
                        choices=(True, False), default=True, metavar="on|off",
                        help='Evaluate using single-token logprobs instead of full generation (server only; forces thinking off) (default: on)')
    parser.add_argument('--tlp', '--top-logprobs', dest='top_logprobs', type=int, default=2,
                        help='Number of top logprobs to request in --lp mode (default: 2)')

    args = parser.parse_args()

    # TODO: share between here and ask_openai
    if args.logprobs:
        if args.thinking:
            print("logprobs specified - forcing thinking off")
        args.thinking = False

    if args.system_prompt:
        p = Path(args.system_prompt)
        if not p.is_file():
            print(f"Error: system prompt file not found: {args.system_prompt}", file=sys.stderr)
            sys.exit(1)
        args.system_prompt_filename = p.name
        args.system_prompt = p.read_text().strip()
    else:
        args.system_prompt_filename = None

    # Validate --prompt-file requires --pid or --all
    if args.prompt_file and not args.prompt_id and not args.all:
        parser.error("--prompt-file requires --pid/--prompt-id or --all")
    if args.all and not args.prompt_file:
        parser.error("--all requires --prompt-file")

    # --save requires --pair-file
    if args.save and not args.pair_file:
        parser.error("--save requires --pair-file")

    if args.save and not args.logprobs:
        parser.error("--save requires --logprobs")

    # --num-pairs requires --pair-file
    if args.num_pairs and not args.pair_file:
        parser.error("--num-pairs requires --pair-file")

    return args


class JsonlWriter:
    def __init__(self, path, next_seq=0):
        self.path = path
        self.file = open(path, "a")
        self.next_seq = next_seq
        self.buffer = {}  # {seq: pair}

    def submit_logprob(self, seq, pair, logprobs):
        """Buffer a result; flush all consecutive completed results."""
        self.buffer[seq] = (pair, logprobs)
        while self.next_seq in self.buffer:
            p, r = self.buffer.pop(self.next_seq)
            self.file.write(json.dumps({"pair": p, "seq": self.next_seq, "logprobs": r}) + "\n")
            self.file.flush()
            print(f"{self.next_seq}: {p}")
            self.next_seq += 1

    def close(self):
        self.file.close()


def apply_pair_order(orig_pair: str, order: str) -> str:
    words = orig_pair.split(",")
    if order == "rvs":
        words = list(reversed(words))
    elif not order == "fwd":
        raise ValueError(f"Unsupported order: {order}")
    return " ".join(words)


def parse_top_logprobs(top_logprobs):
    return [{entry["token"]: math.exp(entry["logprob"])} for entry in top_logprobs]


async def logprob_request_async(client: httpx.AsyncClient, prompt: str, order: str, args) -> tuple:
    response, message, payload = await send_openai_request(client, args, prompt)
    if args.verbose:
        print(f"response:\n{response}\nmessage:\n{message}\npayload:\n{payload}")
    return order, parse_top_logprobs(payload["logprobs"]["content"][0]["top_logprobs"])
    # TODO: for alibaba's qwen (at least) (NOTE: also in ask_openai.py)
    #return order, parse_top_logprobs(message["logprobs"]["content"][0]["top_logprobs"])


async def process_pair_async(client: httpx.AsyncClient, ctx: str, orig_pair: str, orders, args) -> tuple:
    """Process a single pair in all specified orders."""
    results = {}
    for order in orders:
        new_pair = apply_pair_order(orig_pair, order)
        prompt = ctx.replace("{PAIR}", new_pair)
        order_key, logprobs = await logprob_request_async(client, prompt, order, args)
        results[order_key] = logprobs
    return orig_pair, results


_PARSE_ERRORS = (
    TypeError,      # None["x"], list["x"], etc.
    KeyError,       # dict missing expected key
    IndexError,     # list missing expected index
    AttributeError, # None.foo, list.foo, etc.
)


async def process_pairs_async(ctx: str, pairs, orders, args, writer=None):
    """Process all pairs with max_concurrent in flight at once."""

    async def process(client, item):
        seq, orig_pair = item
        result = await process_pair_async(client, ctx, orig_pair, orders, args)
        return (seq, *result)

    numbered_pairs = enumerate(pairs, start=writer.next_seq if writer else 0)
    try:
        async for result in run_concurrent(numbered_pairs, process, args):
            seq, pair, results  = result
            if writer:
                writer.submit_logprob(seq, pair, results)
            else:
                print(f"{pair} {results}")
                
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
            print(f"\n{type(e).__name__}: {e}")
        if isinstance(e, _PARSE_ERRORS):
            traceback.print_exc()
        msg = f"Interrupted."
        if args.save:
            msg += " Re-run to continue."
        print(msg)

    return True


def make_result_prefix(args, prompt_id=None):
    """Build the result file prefix from args."""
    results_dir = args.results_dir
    results_dir.mkdir(exist_ok=True)
    basename = os.path.basename(args.pair_file)
    if args.prompt_file:
        prompt_source = Path(args.prompt_file).stem
        pid = prompt_id if prompt_id is not None else args.prompt_id
    else:
        prompt_source = "manual"
        pid = ""
    base_name = f"{basename}_{prompt_source}_{pid}"
    server_name = get_compact_server_name(args.host)
    if server_name:
        base_name += f"_{server_name}"
    if args.system_prompt_filename:
        base_name += f"_{Path(args.system_prompt_filename).stem}"
    #base_name += f"_mc{args.max_concurrent}"
    if args.tag:
        base_name += f".{args.tag}"
    return str(results_dir / base_name)


def count_lines(filename):
    count = 0
    if os.path.exists(filename):
        with open(filename) as f:
            count += sum(1 for line in f if line.strip())
    return count


def single_prompt_generator(prompts, prompt_id):
    prompt_obj = next((p for p in prompts if p.get('id') == prompt_id), None)
    if not prompt_obj:
        raise ValueError(f"Prompt ID '{prompt_id}' not found")
    yield prompt_obj


def all_prompts_generator(prompts):
    yield from prompts


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


def save_results(args, retry_map=None, writer=None):
    """Write results. If writer was used, only write .retries file."""
    """
    prefix = make_result_prefix(args)
    if retry_map:
        filename = f"{prefix}.retries"
        with open(filename, 'w') as f:
            print_retries(retry_map, file=f, successful_requests=successful_requests)
        print(f"Saved retries to {filename}")
    """


def handle_results(args, retry_map=None, writer=None):
    """Print batch info and save results if requested."""
    #total = len(yes_pairs) + len(no_pairs) + len(bad_pairs)
    total = 0
    if args.num_pairs and total == args.num_pairs:
        print(f"Processed {args.num_pairs} pairs. Re-run to continue.")
    """
    if args.save and args.pair_file:
        save_results(args, retry_map, writer)
    """


def main():
    args = parse_args()
    #auto_detect_max_concurrent(args)
    args.model_id = resolve_model_id(args)
    args.llamacpp = is_llamacpp_backend(args.host, args.port, args.key)
    print(f"Model: {args.model_id} Gemma={is_gemma(args.model_id)} llamacpp={args.llamacpp}")

    if args.prompt_file:
        all_prompts = load_prompts_from_file(args.prompt_file)
        if args.all:
            prompt_gen = all_prompts_generator(all_prompts)
        else:
            prompt_gen = single_prompt_generator(all_prompts, args.prompt_id)
    else:
        prompt_gen = iter([{'id': None, 'text': args.prompt}])

    for prompt_obj in prompt_gen:
        ctx = prompt_obj['text']
        current_pid = prompt_obj.get('id')
        if args.all:
            print(f"\n=== Prompt: {current_pid} ===")

        if args.llamacpp:
            send_anchor_prefix(args, ctx)

        # When --save, count existing results and resume from where we left off
        skip = 0
        writer = None
        if args.save:  # args.logprobs guaranteed by parse_args
            filename = f"{make_result_prefix(args, prompt_id=current_pid)}.jsonl"
            skip = count_lines(filename)
            if skip:
                print(f"Skipping {skip} pairs")
            writer = JsonlWriter(filename, next_seq=skip)

        pairs = get_pairs(args, skip=skip)
        retry_map = None
        orders = []

        t0 = time.monotonic()
        try:
            if not args.logprobs:
                print(f"Thinking: {args.thinking}")
                retry_map = asyncio.run(classify_pairs_async(ctx, pairs, writer))
            else:
                orders = ["fwd", "rvs"]
                asyncio.run(process_pairs_async(ctx, pairs, orders, args, writer))
        finally:
            if writer:
                writer.close()
        duration = time.monotonic() - t0

        num_pairs = (writer.next_seq - skip) if writer else 0
        total_retries, _ = get_retry_stats(retry_map) if retry_map else (0, 0)
        total_requests = (num_pairs * len(orders) + total_retries) if args.logprobs else total_retries
        print(f"Total {num_pairs} pairs ({total_requests} requests) in {fmt_duration(duration)}")
        if retry_map:
            print_retries(retry_map)
        if args.save and num_pairs > 0:
            assert writer is not None
            print(f"{num_pairs} results saved to: {writer.path}")

        handle_results(args, writer)


if __name__ == "__main__":
    main()

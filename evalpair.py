# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import asyncio
import itertools
import os
import re
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx

signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit("SIGTERM"))

from client import add_inference_args, run_concurrent, get_inference_params, send_yesno_request, send_openai_request, query_model_id, resolve_host, get_server_name, auto_detect_max_concurrent
from common import load_prompts_from_file, single_pair_generator, file_pair_generator, flip_pair, eval_with_flipped_retry, parse_yesno_response
from info import info
from model import load_model, is_gemma_model, specialize_prompt, generate_text

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
    if args.logprobs:
        _, _, payload = await send_openai_request(
            client, args, prompt, label=label,
            extra_payload={"max_tokens": 1, "logprobs": True, "top_logprobs": args.top_logprobs},
        )
        top_token = payload["logprobs"]["content"][0]["token"]
        result_str = parse_yesno_response(top_token)
        yes = True if result_str == "YES" else (False if result_str == "NO" else None)
        return yes, payload.get('upstream'), {}

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
        response, _, payload = await send_openai_request(client, args, prompt, label=label, model="1454cffb1a21737e162f508e5bc70be9def89276")
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
    """Process a single pair with retry on flipped pair unless the first result is YES."""
    retry_map = {}

    async def attempt_fn(candidate_pair: str, prompt: str, flipped: bool) -> dict:
        yes, upstream, retries = await async_request_and_classify(client, args, prompt, label=candidate_pair)
        normalized = "YES" if yes is True else "NO" if yes is False else None
        if retries:
            retry_map[candidate_pair] = retries
        return {
            "pair": candidate_pair,
            "flipped": flipped,
            "raw": normalized,
            "normalized": normalized,
            "upstream": upstream,
        }

    evaluation = await eval_with_flipped_retry(ctx, orig_pair, attempt_fn)
    yes = evaluation["normalized"] == "YES" if evaluation["normalized"] is not None else None
    attempts = evaluation["attempts"]
    upstream = None
    if evaluation["decision_source"] == "flipped" and len(attempts) > 1:
        upstream = attempts[1].get("upstream")
    else:
        upstream = attempts[0].get("upstream")

    return orig_pair, yes, evaluation["flipped"], upstream, retry_map


async def process_pairs_async(ctx: str, pairs, args, writer=None, on_result=None):
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
        async for result in run_concurrent(numbered_pairs, process, args):
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
            if on_result:
                on_result(pair, classification)
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


def send_anchor_prefix(args, ctx: str) -> None:
    """Send prompt prefix up to 'Clue' to v1/completions for KV cache warming."""
    m = re.search(r'^(.*?Clue)', ctx, re.DOTALL)
    if not m:
        raise ValueError("send_anchor_prefix: no 'Clue' found in prompt context")
    prefix = "<|im_start|>user\n" + m.group(1)
    url = f"{args.host}:{args.port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    payload = {"prompt": prefix, "cache_prefix": True}
    if not args.thinking:
        payload["reasoning_budget_tokens"] = 0
        payload["reasoning_budget_start_tag"] = "<think>"
        payload["reasoning_budget_end_tag"] = "</think>"
    response = httpx.post(url, headers=headers, json=payload, timeout=args.timeout)
    response.raise_for_status()


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
        pair = flip_pair(orig_pair).replace(',', ' ')
        _, response = custom_context(model, tokenizer, args, ctx, pair)
        yes = is_yes_response(model, response)
        if yes:
            flipped = True
    return orig_pair, yes, flipped

def process_pairs_sync(ctx, pairs, model, tokenizer, args, writer=None, on_result=None):
    """Process all pairs synchronously (local model path)."""
    yes_pairs, no_pairs, bad_pairs, last_processed = [], [], [], None
    for seq, orig_pair in enumerate(pairs, start=writer.next_seq if writer else 0):
        orig_pair, yes, flipped = evaluate_pair_sync(model, tokenizer, args, ctx, orig_pair)
        last_processed = orig_pair
        print(format_result(orig_pair, yes, flipped))
        if yes is True:
            classification = "yes"
            yes_pairs.append(orig_pair)
        elif yes is False:
            classification = "no"
            no_pairs.append(orig_pair)
        else:
            classification = "bad"
            bad_pairs.append(orig_pair)
        if on_result:
            on_result(orig_pair, classification)
        if writer:
            writer.submit(seq, orig_pair, classification)
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
    add_inference_args(parser)
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
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show actual response and message")
    parser.add_argument("-q", "--quiet", action="store_true",
                      help="Suppress REQUEST START/END messages")
    parser.add_argument('--lp', '--logprobs', dest='logprobs', action='store_true',
                        help='Evaluate using single-token logprobs instead of full generation (server only; forces thinking off)')
    parser.add_argument('--tlp', '--top-logprobs', dest='top_logprobs', type=int, default=20,
                        help='Number of top logprobs to request in --lp mode (default: 20)')

    args = parser.parse_args()

    if args.logprobs:
        args.thinking = False
        if args.model is not None:
            parser.error("--lp/--logprobs requires server mode (omit -m/--model)")

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
    return flip_pair(pair)


class IncrementalWriter:
    def __init__(self, prefix, next_seq=0, enabled_exts=("yes", "no", "bad")):
        self.prefix = prefix
        self.files = {ext: open(f"{prefix}.{ext}", "a") for ext in enabled_exts}
        self.next_seq = next_seq
        self.buffer = {}         # {seq: (pair, classification)}

    def submit(self, seq, pair, classification):
        """Buffer a result; flush all consecutive completed results."""
        self.buffer[seq] = (pair, classification)
        while self.next_seq in self.buffer:
            p, c = self.buffer.pop(self.next_seq)
            if c in self.files:
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
        pid = ""
    base_name = f"{basename}_{prompt_source}_{pid}"
    server_name = get_server_name(args.host)
    if server_name:
        base_name += f"_{server_name}"
    if args.system_prompt_filename:
        base_name += f"_{Path(args.system_prompt_filename).stem}"
    #base_name += f"_mc{args.max_concurrent}"
    if args.tag:
        base_name += f".{args.tag}"
    return f"{results_dir}/{base_name}"


def get_result_paths(prefix):
    return {ext: f"{prefix}.{ext}" for ext in ("yes", "no", "bad")}


def get_expected_bad_metadata_suffix(args):
    if args.prompt_file:
        prompt_source = Path(args.prompt_file).stem
        pid = args.prompt_id
    else:
        prompt_source = "manual"
        pid = ""

    suffix = f"_{prompt_source}_{pid}"
    server_name = get_server_name(args.host)
    if server_name:
        suffix += f"_{server_name}"
    if args.system_prompt_filename:
        suffix += f"_{Path(args.system_prompt_filename).stem}"
    return suffix


def prepare_bad_file_mode(args):
    bad_path = Path(args.pair_file)
    if bad_path.suffix != ".bad":
        return None

    errors = []
    if args.num_pairs:
        errors.append("--num-pairs cannot be used with a .bad result file")

    base_name = bad_path.name[:-len(".bad")]
    metadata_base = re.sub(r"_mc\d+(?:\..+)?$", "", base_name)
    expected_suffix = get_expected_bad_metadata_suffix(args)
    if not metadata_base.endswith(expected_suffix):
        errors.append(
            "bad result filename does not match command-line prompt/server/system-prompt options:\n"
            f"  file: {bad_path.name}\n"
            f"  expected suffix: *{expected_suffix}[optional _mcN[.tag]].bad"
        )

    prefix = str(bad_path)[:-len(".bad")]
    result_paths = get_result_paths(prefix)
    missing = [path for ext, path in result_paths.items() if ext in ("yes", "no") and not os.path.exists(path)]
    if missing:
        errors.append("corresponding result files are missing:")
        errors.extend(f"  {path}" for path in missing)

    if errors:
        print("Error: cannot process .bad pair file:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)

    with open(result_paths["bad"], "r") as f:
        bad_pairs = {line.strip() for line in f if line.strip()}

    removed = 0
    for ext in ("yes", "no"):
        for pair in file_pair_generator(result_paths[ext]):
            if pair in bad_pairs:
                bad_pairs.remove(pair)
                removed += 1
    print(f"Filtered {removed} pairs already present in {result_paths['yes']} and {result_paths['no']}")

    return {
        "prefix": prefix,
        "bad_pairs": bad_pairs,
        "paths": result_paths,
    }


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


def get_bad_pairs(args, state):
    pairs = iter(sorted(state["bad_pairs"]))
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
    args.host = resolve_host(args.host)
    bad_mode = prepare_bad_file_mode(args) if args.pair_file else None

    if args.model is None:
        # Server path (default)
        model, tokenizer = None, None

        auto_detect_max_concurrent(args)

        args.model_id = query_model_id(args.host, args.port, args.key)
    else:
        # Local model path
        _, model, tokenizer = load_model(args.model)

    ctx = args.prompt
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        prompt_obj = next((p for p in prompts if p.get('id') == args.prompt_id), None)
        if not prompt_obj:
            raise ValueError(f"Prompt ID '{args.prompt_id}' not found in {args.prompt_file}")
        ctx = prompt_obj['text']

    if args.model is None:
        send_anchor_prefix(args, ctx)

    # When --save, count existing results and resume from where we left off
    skip = 0
    writer = None
    if bad_mode:
        writer = IncrementalWriter(bad_mode["prefix"], enabled_exts=("yes", "no"))
    elif args.save and args.pair_file:
        prefix = make_result_prefix(args)
        skip = count_completed_pairs(prefix)
        if skip:
            print(f"Resuming: skipping {skip} completed pairs")
        writer = IncrementalWriter(prefix, next_seq=skip)

    pairs = get_bad_pairs(args, bad_mode) if bad_mode else get_pairs(args, skip=skip)

    retry_map = None
    successful = None
    on_result = None
    if bad_mode:
        bad_pairs = bad_mode["bad_pairs"]

        def on_result(pair, classification):
            if classification == "bad":
                bad_pairs.add(pair)
            else:
                bad_pairs.discard(pair)

    if args.model is None and (args.pair_file or args.pair):
        t0 = time.monotonic()
        try:
            yes, no, bad, retry_map, successful = asyncio.run(process_pairs_async(ctx, pairs, args, writer, on_result=on_result))
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
        yes, no, bad, last = process_pairs_sync(ctx, pairs, model, tokenizer, args, writer, on_result=on_result)

    if bad_mode:
        bad_filename = bad_mode["paths"]["bad"]
        with open(bad_filename, "w") as f:
            for pair in sorted(bad_mode["bad_pairs"]):
                f.write(pair + "\n")
        print(f"Saved {len(bad_mode['bad_pairs'])} BAD pairs to {bad_filename}")
        if retry_map:
            retries_filename = f"{bad_mode['prefix']}.retries"
            with open(retries_filename, "w") as f:
                print_retries(retry_map, file=f, successful_requests=successful)
            print(f"Saved retries to {retries_filename}")
        return

    handle_results(args, yes, no, bad, retry_map, successful, writer)

if __name__ == "__main__":
    main()

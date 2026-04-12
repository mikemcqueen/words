import asyncio
import time

import httpx

from client import get_server_name, run_concurrent, send_openai_request
from common import eval_with_flipped_retry, flip_pair


def format_result(orig_pair, yes, flipped, upstream=None):
    """Format a result line for display."""
    label = 'YES' if yes else ('NO' if yes is False else 'None')
    prefix = f"[{upstream.upper()}] " if upstream else ""
    as_txt = f" as {flip_pair(orig_pair).replace(',', ' ')}" if flipped else ""
    return f"{prefix}{orig_pair} {label}{as_txt}"


def is_yes_response(model, response: str):
    """Return True for YES, False for NO, None for garbage/unexpected."""
    if not model:
        answer = response
    else:
        assert False, "is_yes_response not implemented for this model"
    upper = answer.upper()
    if upper.startswith("YES"):
        return True
    if upper.startswith("NO"):
        return False
    return None


async def async_request_and_classify(client, args, prompt, label=None):
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


async def classify_pair_async(client: httpx.AsyncClient, args, ctx: str, orig_pair: str) -> tuple:
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


async def classify_pairs_async(ctx: str, pairs, args, writer=None):
    """Process all pairs with max_concurrent in flight at once."""
    retry_map = {}

    async def classify(client, item):
        seq, orig_pair = item
        result = await classify_pair_async(client, args, ctx, orig_pair)
        return (seq, *result)

    numbered_pairs = enumerate(pairs, start=writer.next_seq if writer else 0)
    try:
        async for result in run_concurrent(numbered_pairs, classify, args):
            seq, pair, yes, flipped, upstream, pair_retries = result
            print(format_result(pair, yes, flipped, upstream))
            retry_map.update(pair_retries)
            if writer:
                writer.submit_logprob(seq, pair)
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
            print(f"\n{type(e).__name__}: {e}")
        msg = f"Interrupted."
        if args.save:
            msg += " Re-run to continue."
        print(msg)

    return retry_map



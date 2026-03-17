# client.py
#
# Shared async HTTP client utilities for concurrent request processing.

import asyncio
import sys
import time

import httpx

from model import add_thinking

SERVER_URL = "http://localhost"

SERVERS = {
    "juniper": "192.168.0.111",
    "mini": "192.168.0.114",
}
SERVER_IPS = {ip: name for name, ip in SERVERS.items()}


def resolve_host(host: str) -> str:
    """If host is a known server name, return http://{ip}. Otherwise return as-is.
    Raises SystemExit if it looks like a bare name but isn't recognized."""
    if host in SERVERS:
        return f"http://{SERVERS[host]}"
    if "." not in host and "://" not in host and host != "localhost":
        print(f"Error: unknown server name '{host}'")
        sys.exit(1)
    return host


def get_server_name(host: str) -> str | None:
    """Return a friendly server name for a host URL, or None if unknown."""
    bare = host.split("://", 1)[-1]
    return SERVER_IPS.get(bare)



async def run_concurrent(items, process_fn, max_concurrent, timeout):
    """
    Process items with bounded concurrency, yielding results as they complete.
    At most max_concurrent requests in flight at any time.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        pending = set()
        items_iter = iter(items)
        in_flight = 0

        async def tracked(item):
            nonlocal in_flight
            in_flight += 1
            ts = time.strftime("%H:%M:%S")
            label = item['pair'] if isinstance(item, dict) else item
            print(f"[{ts}] >>> REQUEST START {label} in_flight={in_flight} pending={len(pending)}")
            if in_flight > max_concurrent:
                print(f"[{ts}] !!! BUG: in_flight ({in_flight}) > max_concurrent ({max_concurrent})")
            try:
                return await process_fn(client, item)
            finally:
                in_flight -= 1
                ts_end = time.strftime("%H:%M:%S")
                print(f"[{ts_end}] <<< REQUEST END {label} in_flight={in_flight}")

        # Start initial batch
        for _ in range(max_concurrent):
            item = next(items_iter, None)
            if item is None:
                break
            task = asyncio.create_task(tracked(item))
            pending.add(task)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                yield task.result()

            # Refill to max_concurrent
            while len(pending) < max_concurrent:
                item = next(items_iter, None)
                if item is None:
                    break
                task = asyncio.create_task(tracked(item))
                pending.add(task)


async def _post_with_retry(client, *args, max_retries=3, **kwargs):
    """Wrap client.post() with retry logic for transient errors."""
    delays = [5, 20, 80]
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(*args, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt >= max_retries:
                raise
            delay = delays[attempt]
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] RETRY {attempt+1}/{max_retries} after {type(e).__name__}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt >= max_retries:
                raise
            delay = delays[attempt]
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] RETRY {attempt+1}/{max_retries} after HTTP {e.response.status_code}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)


def get_inference_params(args) -> dict:
    """Extract inference parameters from args into a canonical dict."""
    params = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "repeat_penalty": args.repeat_penalty,
        "repeat_last_n": args.repeat_last_n,
    }
    if args.think:
        add_thinking(params, args.model_id)
    return params


async def send_yesno_request(client: httpx.AsyncClient, args, prompt: str) -> tuple[str, dict]:
    """POST {base_url}/yesno with {"text": prompt}"""
    url = f"{args.host}:{args.port}/yesno"
    response = await _post_with_retry(client, url, json={"text": prompt})
    js = response.json()
    return js["response"], js, js


def query_model_id(host: str, port: int, key: str = None) -> str:
    """GET /v1/models and return the first model's ID."""
    url = f"{host}:{port}/v1/models"
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = httpx.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models returned from {url}")
    return data[0]["id"]


async def send_openai_request(client: httpx.AsyncClient, args, prompt: str, model: str = "haiku") -> tuple[str, dict]:
    """POST {base_url}/v1/chat/completions with OpenAI chat format"""
    url = f"{args.host}:{args.port}/v1/chat/completions"

    headers = { "Content-Type": "application/json" }
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    messages = []
    if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        **get_inference_params(args),
    }
    response = await _post_with_retry(client, url, headers=headers, json=payload)
    payload = response.json()["choices"][0]
    message = payload["message"]
    del payload["message"]
    actual = message["content"]
    del message["content"]
    return actual, message, payload

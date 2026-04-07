# client.py
#
# Shared async HTTP client utilities for concurrent request processing.

import asyncio
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

from model import adjust_thinking

SERVERS = {
    "localhost": "127.0.0.1",
    "juniper": "192.168.0.111",
    "mini": "192.168.0.114",
}
SERVER_IPS = {ip: name for name, ip in SERVERS.items()}


def parse_on_off(value: str) -> bool:
    """Parse an on/off CLI value into a boolean."""
    normalized = value.lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise argparse.ArgumentTypeError("expected 'on' or 'off'")


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add shared inference-related CLI arguments to a parser."""
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=20,
                        help="Top-k sampling (default: 20)")
    parser.add_argument("--min-p", "--min_p", dest="min_p", type=float, default=0.01,
                        help="Min-p sampling (default: 0.01)")
    parser.add_argument("--repeat-penalty", "--rp", dest="repeat_penalty", type=float, default=1.0,
                        help="Repeat penalty (default: 1.0)")
    parser.add_argument("--repeat-last-n", "--rln", dest="repeat_last_n", type=int, default=32,
                        help="Repeat last n tokens (default: 32)")
    parser.add_argument("--presence-penalty", "--pp", dest="presence_penalty",
                        type=float, default=0.0,
                        help="Presence penalty (default: 0.0)")
    parser.add_argument("--thinking", type=parse_on_off, choices=(True, False), default=True,
                        metavar="on|off",
                        help="Control API thinking mode (default: on)")


def parse_nginx_upstream(config_path=None):
    """Parse nginx upstream block to extract server topology.

    Returns dict with:
      servers: list of {ip, port, max_conns, name}
      queue_size: int (0 if no queue)
      total_capacity: sum(max_conns) + queue_size
    Or None if config not found/unparseable.
    """
    if config_path:
        candidates = [Path(config_path)]
    else:
        candidates = [
            Path("/etc/nginx/sites-available/gpu_cluster"),
            Path("/etc/nginx/sites-enabled/gpu_cluster"),
        ]

    text = None
    for p in candidates:
        try:
            text = p.read_text()
            break
        except (OSError, PermissionError):
            continue
    if text is None:
        return None

    # Extract upstream block
    m = re.search(r'upstream\s+gpu_cluster\s*\{([^}]+)\}', text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)

    servers = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or not stripped.startswith('server '):
            continue
        sm = re.match(r'server\s+([\d.]+):(\d+)\s*(.*)', stripped)
        if not sm:
            continue
        ip, port = sm.group(1), int(sm.group(2))
        rest = sm.group(3)
        max_conns = 0
        mc = re.search(r'max_conns=(\d+)', rest)
        if mc:
            max_conns = int(mc.group(1))
        # Extract comment name (e.g. "# JUNIPER")
        name = SERVER_IPS.get(ip)
        if not name:
            cm = re.search(r'#\s*(\S+)', rest)
            if cm:
                name = cm.group(1).lower()
        servers.append({'ip': ip, 'port': port, 'max_conns': max_conns, 'name': name})

    if not servers:
        return None

    queue_size = 0
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        qm = re.match(r'queue\s+(\d+)', stripped)
        if qm:
            queue_size = int(qm.group(1))

    total = sum(s['max_conns'] for s in servers) + queue_size
    return {
        'servers': servers,
        'queue_size': queue_size,
        'total_capacity': total,
    }


def resolve_host(host: str) -> str:
    """If host is a known server name, return http://{ip}. Otherwise return as-is.
    Raises SystemExit if it looks like a bare name but isn't recognized."""
    if host in SERVERS:
        return f"http://{SERVERS[host]}"
    if "." not in host and "://" not in host:
        print(f"Error: unknown server name '{host}'")
        sys.exit(1)
    return host


def get_server_name(host: str) -> str | None:
    """Return a friendly server name for a host URL, or None if unknown."""
    bare = host.split("://", 1)[-1]
    return SERVER_IPS.get(bare)


def get_max_concurrent(host: str, port: int, nginx_config=None):
    """Auto-detect max-concurrent from nginx config.

    If host is localhost:80 (nginx), returns total_capacity.
    If host is a specific backend server, returns that server's max_conns.
    Returns None if nginx config not found or host not in config.
    """
    upstream = parse_nginx_upstream(nginx_config)
    if not upstream:
        return None

    # Going through nginx — use total capacity
    bare = host.split("://", 1)[-1]
    if bare == "127.0.0.1" and port == 80:
        return upstream['total_capacity'], upstream

    # Targeting a specific server — find its max_conns
    ip = bare
    for s in upstream['servers']:
        if s['ip'] == ip or s['name'] == bare:
            return s['max_conns'], upstream

    return None



def auto_detect_max_concurrent(args):
    """Auto-detect max-concurrent from nginx config, mutating args in place."""
    nginx_config = getattr(args, 'nginx_config', None)
    result = get_max_concurrent(args.host, args.port, nginx_config)
    if not result:
        return
    mc, upstream = result
    user_set_mc = '--max-concurrent' in sys.argv or '--mc' in sys.argv
    if not user_set_mc:
        args.max_concurrent = mc
        server_name = get_server_name(args.host)
        if server_name == "localhost":
            server_desc = ", ".join(
                f"{s['name'] or s['ip']}:{s['max_conns']}"
                for s in upstream['servers']
            )
            print(f"nginx: auto-set --max-concurrent={mc} "
                  f"({len(upstream['servers'])} servers: {server_desc}, "
                  f"queue={upstream['queue_size']})")
        else:
            print(f"nginx: auto-set --max-concurrent={mc} "
                  f"(from {server_name} max_conns)")
    elif args.max_concurrent > upstream['total_capacity']:
        print(f"Warning: --max-concurrent={args.max_concurrent} exceeds "
              f"nginx capacity ({upstream['total_capacity']})")


async def run_concurrent(items, process_fn, args):
    """
    Process items with bounded concurrency, yielding results as they complete.
    At most max_concurrent requests in flight at any time.
    """
    max_concurrent = args.max_concurrent
    quiet = getattr(args, 'quiet', False)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        pending = set()
        items_iter = iter(items)
        in_flight = 0

        async def tracked(item):
            nonlocal in_flight
            in_flight += 1
            ts = time.strftime("%H:%M:%S")
            label = item['pair'] if isinstance(item, dict) else item[1] if isinstance(item, tuple) else item
            if not quiet:
                print(f"[{ts}] >>> REQUEST START {label} in_flight={in_flight} pending={len(pending)}")
            if in_flight > max_concurrent:
                print(f"[{ts}] !!! BUG: in_flight ({in_flight}) > max_concurrent ({max_concurrent})")
            try:
                return await process_fn(client, item)
            finally:
                in_flight -= 1
                ts_end = time.strftime("%H:%M:%S")
                if not quiet:
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


async def _post_with_retry(client, *args, label=None, **kwargs):
    """Wrap client.post() with retry logic for transient errors."""
    delays = [5, 20, 40, 60, 80]
    max_retries = len(delays)
    tag = f" [{label}]" if label else ""
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(*args, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            if attempt >= max_retries:
                raise
            delay = delays[attempt]
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] RETRY {attempt+1}/{max_retries}{tag} after {type(e).__name__}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt >= max_retries:
                raise
            delay = delays[attempt]
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] RETRY {attempt+1}/{max_retries}{tag} after HTTP {e.response.status_code}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)


def add_inference_options(payload: dict, args) -> dict:
    """Populate a request payload with shared inference options (skips any not present in args)."""
    for attr, key in [
        ("temp",            "temperature"),
        ("top_p",           "top_p"),
        ("top_k",           "top_k"),
        ("min_p",           "min_p"),
        ("repeat_penalty",  "repeat_penalty"),
        ("repeat_last_n",   "repeat_last_n"),
        ("presence_penalty","presence_penalty"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            payload[key] = val
    thinking = getattr(args, "thinking", None)
    model_id = getattr(args, "model_id", None)
    if thinking is not None and model_id is not None:
        adjust_thinking(payload, model_id, thinking)
    return payload


def get_inference_params(args) -> dict:
    """Extract inference parameters from args into a canonical dict."""
    return add_inference_options({}, args)


def _extract_upstream(response) -> str | None:
    """Extract friendly server name from X-Upstream-Addr header."""
    addr = response.headers.get("x-upstream-addr")
    if not addr:
        return None
    ip = addr.split(":")[0]
    return SERVER_IPS.get(ip, addr)


async def send_yesno_request(client: httpx.AsyncClient, args, prompt: str, label=None) -> tuple[str, dict]:
    """POST {base_url}/yesno with {"text": prompt}"""
    url = f"{args.host}:{args.port}/yesno"
    response = await _post_with_retry(client, url, json={"text": prompt}, label=label)
    js = response.json()
    upstream = _extract_upstream(response)
    if upstream:
        js['upstream'] = upstream
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


async def send_openai_request(client: httpx.AsyncClient, args, prompt: str, model: str = "haiku", label=None, extra_payload: dict = None) -> tuple[str, dict]:
    """POST {base_url}/v1/chat/completions with OpenAI chat format"""
    url = f"{args.host}:{args.port}/v1/chat/completions"

    headers = { "Content-Type": "application/json" }
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    messages = []
    system_prompt = getattr(args, "system_prompt", None)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = add_inference_options({
        "model": model,
        "messages": messages,
    }, args)
    if extra_payload:
        payload.update(extra_payload)
    if getattr(args, "verbose", False):
        print(json.dumps(payload, indent=2))
    response = await _post_with_retry(client, url, headers=headers, json=payload, label=label)
    upstream = _extract_upstream(response)
    payload = response.json()["choices"][0]
    message = payload["message"]
    del payload["message"]
    response = message["content"]
    del message["content"]
    if upstream:
        payload['upstream'] = upstream
    return response, message, payload

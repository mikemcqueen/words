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

from src.common import add_inference_args, parse_on_off
from src.model import is_gpt

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


#https://dashscope-us.aliyuncs.com/compatible-mode/v1
def host_path_prefix(host: str) -> str:
    if "dashscope-us" in host.lower() or "dashscope-intl" in host.lower():
        return "/compatible-mode"
    return ""


def build_url(host: str, port: int, path: str) -> str:
    path = host_path_prefix(host) + path
    if host.startswith("http://") or host.startswith("https://"):
        # append port only if host doesn't already specify one
        bare = host.split("://", 1)[1]
        if ":" not in bare:
            return f"{host}:{port}{path}"
        return f"{host}{path}"
    return f"{host}:{port}{path}"


def ssl_verify(host: str) -> bool:
    bare = host.split("://", 1)[-1].split(":")[0]
    return bare.split(".")[-1] != "local"


_URL_PREFIXES = {"api", "www", "dashscope-us"}


def get_compact_server_name(host: str) -> str:
    bare = host.split("://", 1)[-1].split(":")[0]
    labels = bare.split(".")
    if labels[0] in _URL_PREFIXES:
        assert len(labels) > 1
        return labels[1]
    return labels[0]


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
    async with httpx.AsyncClient(timeout=args.timeout, verify=ssl_verify(args.host)) as client:
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


async def _post_with_retry(client, *args, label=None, quiet=False, **kwargs) -> httpx.Response:
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
            if not quiet:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] RETRY {attempt+1}/{max_retries}{tag} after {type(e).__name__}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt >= max_retries:
                raise
            delay = delays[attempt]
            if not quiet:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] RETRY {attempt+1}/{max_retries}{tag} after HTTP {e.response.status_code}, waiting {delay}s", file=sys.stderr)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without return or raise")


def adjust_thinking(payload: dict, args) -> None:
    thinking = getattr(args, "thinking", None)
    if thinking in (None, True):
        return

    #print(f"thinking: {thinking}")
    
    name = getattr(args, "model_id", None).lower()
    if args.llamacpp:
        if name.startswith("glm") or name.startswith("qwen"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        if name.startswith("deepseek"):
            payload["thinking"] = {"type": "disabled"}
        elif is_gpt(name):
            payload["reasoning"] = { "effort": "none" }
        elif name.startswith("qwen"):
            payload["enable_thinking"] = False


def add_inference_options(payload: dict, args) -> dict:
    """Populate a request payload with shared inference options (skips any not present in args)."""
    # TODO: not all of these are restricted to llamacpp, e.g. temp, top_p are widely supported
    if args.llamacpp:
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

    adjust_thinking(payload, args)
    return payload


"""
# for nginix upstream
def _extract_upstream(response) -> str | None:
    addr = response.headers.get("x-upstream-addr")
    if not addr:
        return None
    ip = addr.split(":")[0]
    return SERVER_IPS.get(ip, addr)
"""


def query_models(host: str, port: int, key: str | None) -> list[dict]:
    url = build_url(host, port, "/v1/models")
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = httpx.get(url, headers=headers, timeout=10, verify=ssl_verify(host))
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models returned from {url}")
    return data


def is_llamacpp_backend(host: str, port: int, key: str | None) -> bool:
    """Return True if the backend is llamacpp (any model has owner='llamacpp')."""
    data = query_models(host, port, key)
    return any(m.get("owned_by") == "llamacpp" for m in data)


def resolve_model_id(args) -> str:
    """Query server for available models and resolve to a single model ID.

    If --model is specified, it must match one of the available models.
    If not specified, the server must offer exactly one model.
    """
    data = query_models(args.host, args.port, args.key)
    models = [m["id"] for m in data]
    if args.model:
        if args.model in models:
            return args.model
        print(f"Error: model '{args.model}' not available.")
    elif len(models) == 1:
        print(f"Using model: {models[0]}")
        return models[0]

    print("Available models:")
    for m in models:
        print(f"  {m}")
    sys.exit(1)


def get_max_logprobs_tokens(args) -> int:
    if not args.llamacpp:
        if is_gpt(args.model_id):
            return 16
    return 1


def use_responses(model_id: str):
    return model_id.lower().startswith("gpt")


async def send_openai_request(client: httpx.AsyncClient, args, prompt: str, label=None) -> tuple[str, dict, dict]:
    v1resp = use_responses(args.model_id)
    shape = "/v1/responses" if v1resp else "/v1/chat/completions"

    url = build_url(args.host, args.port, shape)

    headers = { "Content-Type": "application/json" }
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    if v1resp:
        payload = add_inference_options({
            "model": args.model_id,
            "input": prompt
        }, args)
    else:
        messages = []
        system_prompt = getattr(args, "system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = add_inference_options({
            "model": args.model_id,
            "messages": messages,
        }, args)

    if getattr(args, "logprobs", False):
        max_tokens = "max_output_tokens" if v1resp else "max_tokens"
        payload.update({max_tokens: get_max_logprobs_tokens(args), "top_logprobs": args.top_logprobs})
        if is_gpt(args.model_id):
            payload["include"] = ["message.output_text.logprobs"]
        else:
            payload["logprobs"] = True

    if getattr(args, "verbose", False):
        print(json.dumps(payload, indent=2))

    response = await _post_with_retry(client, url, headers=headers, json=payload, label=label, quiet=getattr(args, "quiet", False))

    #upstream = _extract_upstream(response)
    if args.verbose:
        print(json.dumps(response.json(), indent=2))

    if v1resp:
        # super over simplified
        payload = response.json()["output"]
        message = payload[0]["content"][0]
        response = message["text"]
    else:
        payload = response.json()["choices"][0]
        message = payload["message"]
        del payload["message"]
        response = message["content"]
        del message["content"]
        #if upstream:
        #payload['upstream'] = upstream
    return response, message, payload

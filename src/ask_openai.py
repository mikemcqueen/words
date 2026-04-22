#!/usr/bin/env python3

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from src.client import query_model_id, resolve_host, send_openai_request
def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a single OpenAI-style prompt to the configured server"
    )

    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to send")
    parser.add_argument("-s", "--system-prompt", type=str,
                        help="System prompt file (optional)")
    parser.add_argument("--host", default="juniper",
                        help="Server host (default: juniper)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--api-key", "--key", dest="key", type=str,
                        help="API key")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--min-p", "--min_p", dest="min_p", type=float, default=0.01,
                        help="Min-p sampling (default: 0.01)")
    parser.add_argument("--repeat-penalty", "--rp", type=float, default=1.0,
                        help="Repeat penalty (default: 1.0)")
    parser.add_argument("--repeat-last-n", "--rln", type=int, default=32,
                        help="Repeat last n tokens (default: 32)")
    parser.add_argument("--thinking", type=str, choices=("on", "off"), default="on",
                        help="Control API thinking mode (default: on)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")

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

    args.thinking = args.thinking == "on"

    return args


async def run(args):
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        response, message, payload = await send_openai_request(
            client,
            args,
            args.prompt,
            args.model_id,
        )
    return response, message, payload


def extract_reasoning(message: dict) -> str | None:
    """Return reasoning text from the OpenAI message payload, if present."""
    return message.get("reasoning_content")


def main():
    args = parse_args()
    args.host = resolve_host(args.host)

    try:
        args.model_id = query_model_id(args.host, args.port, args.key)
        print(f"temp={args.temp} min-p={args.min_p} top-p={args.top_p}")

        response, message, _ = asyncio.run(run(args))
        reasoning = extract_reasoning(message)
        if reasoning:
            print("reasoning:")
            print("===============")
            print(reasoning)
            print("===============")
        print(response)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: HTTP {e.response.status_code}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

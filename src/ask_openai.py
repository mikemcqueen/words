#!/usr/bin/env python3

import argparse
import asyncio
import math
import sys
from pathlib import Path

import httpx

from src.client import resolve_model_id, send_openai_request, is_llamacpp_backend, ssl_verify
from src.common import add_inference_args, parse_on_off

def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a single OpenAI-style prompt to the configured server"
    )

    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to send")
    parser.add_argument("-s", "--system-prompt", type=str,
                        help="System prompt file (optional)")
    parser.add_argument("--host", default="juniper.local",
                        help="Server host (default: juniper.local)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--api-key", "--key", dest="key", type=str,
                        help="API key")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")
    add_inference_args(parser)
    parser.add_argument('--lp', '--logprobs', dest='logprobs', type=parse_on_off,
                        choices=(True, False), default=False, metavar="on|off",
                        help='Request single-token logprobs instead of full generation (default: off)')
    parser.add_argument('--tlp', '--top-logprobs', dest='top_logprobs', type=int, default=2,
                        help='Number of top logprobs to return in --lp mode (default: 2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print request details (model, host, temp, etc.)')

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

    return args


async def run(args):
    async with httpx.AsyncClient(timeout=args.timeout, verify=ssl_verify(args.host)) as client:
        response, message, payload = await send_openai_request(
            client,
            args,
            args.prompt,
        )
    return response, message, payload


def extract_reasoning(message: dict) -> str | None:
    """Return reasoning text from the OpenAI message payload, if present."""
    return message.get("reasoning_content")


def main():
    args = parse_args()

    try:
        args.model_id = resolve_model_id(args)
        args.llamacpp = is_llamacpp_backend(args.host, args.port, args.key)
        print(f"model: {args.model_id} host={args.host} llamacpp={args.llamacpp}")

        response, message, payload = asyncio.run(run(args))
        reasoning = None
        if isinstance(payload, dict):
            reasoning = extract_reasoning(message)
        if reasoning:
            print("reasoning:")
            print("===============")
            print(reasoning)
            print("===============")

        import json
        top_logprobs = []
        if isinstance(payload, dict):
            print(f"payload: {json.dumps(payload, indent=2)}")
            lp_data = payload.get("logprobs")
            # TODO: for alibaba's qwen (at least) (NOTE: also in evalpair.py)
            #lp_data = message.get("logprobs")
            top_logprobs = lp_data.get("content", [{}])[0].get("top_logprobs", [])
        elif isinstance(message, dict):
            lp_data = message.get("logprobs")
            #print(f"lpdata[0]: {json.dumps(lp_data[0], indent=2)}")
            top_logprobs = lp_data[0].get("top_logprobs", [])
            #print(f"top_logprobs: {json.dumps(top_logprobs, indent=2)}")

        if top_logprobs:
            for entry in top_logprobs:
                prob = math.exp(entry["logprob"]) * 100
                print(f"  {entry['token']:>6}: {prob:5.1f}%")
        else:
            print(response)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: HTTP {e.response.status_code}: {e}\n{e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

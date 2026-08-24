# Plan: Convert nextwords.py to async server client + refactor shared client args

## Context
nextwords.py currently loads a model locally and runs inference via `WordProbabilityExplorer`. The goal is to convert it to send requests to a running inference server using `/v1/completions` with `logprobs`, replicating the multi-token beam search via sequential API calls. The synchronous model-loading path should be removed entirely. Additionally, the common client CLI arguments (host, port, max-concurrent, etc.) duplicated across evalpair.py and eval_prompt.py should be refactored into a shared `add_client_args()` function in client.py.

## Changes

### 1. Add `add_client_args()` to client.py

Add a new function that registers the shared networking/client CLI args:

```python
def add_client_args(parser):
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--max-concurrent", "--mc", type=int, default=1,
                        help="Max concurrent requests (default: 1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument("--key", type=str, help="API key")
    parser.add_argument("--nginx-config", type=str,
                        help="Path to nginx upstream config (auto-detected if omitted)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show verbose output")
```

Also add a `resolve_client_args(args)` helper that does the common post-parse steps both scripts do:
```python
def resolve_client_args(args):
    args.host = resolve_host(args.host)
    auto_detect_max_concurrent(args)
    args.model_id = query_model_id(args.host, args.port, args.key)
```

**Files:** `client.py`

### 2. Refactor evalpair.py and eval_prompt.py to use `add_client_args()`

Replace the duplicated host/port/max-concurrent/timeout/key/nginx-config/verbose argument definitions with a call to `add_client_args(parser)`. Replace the post-parse resolve_host/auto_detect/query_model_id calls with `resolve_client_args(args)`.

Keep task-specific args (--client, --system-prompt, --tag, etc.) in each script.

**Files:** `evalpair.py` (~lines 317-336), `eval_prompt.py` (~lines 318-344, 363-366)

### 3. Add `send_completions_request()` to client.py

New async function for the `/v1/completions` endpoint that returns token logprobs:

```python
async def send_completions_request(client, args, prompt, max_tokens=1, logprobs=1000, label=None):
    """POST /v1/completions and return top token logprobs for the next position."""
    url = f"{args.host}:{args.port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    payload = {
        "model": args.model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "temperature": 1.0,  # We want raw logprobs, not sampled
    }
    response = await _post_with_retry(client, url, headers=headers, json=payload, label=label)
    return response.json()
```

**Files:** `client.py`

### 4. Rewrite nextwords.py as async server client

Remove all local model loading (`load_model`, `WordProbabilityExplorer`, `clear_cache` imports). Replace with async beam search using API calls.

**Key design:**
- Use `asyncio.run()` as the entry point
- For each word, perform the iterative BFS by:
  1. Call `/v1/completions` with `logprobs=top_k` on the base prompt to get first-token logprobs
  2. Filter for valid first tokens (space + alpha, or alpha for instruct models)
  3. At each depth, fan out requests for all active paths using `run_concurrent()` or bounded `asyncio.Semaphore`
  4. Filter continuation tokens, track word boundaries, maintain top-k threshold
  5. Continue until max depth (5) or no more paths
- Multiple words from `--file` can also be processed concurrently via `--max-concurrent`

**Token filtering:** Since we no longer have the tokenizer's vocabulary pre-decoded, the logprobs response from /v1/completions returns token strings directly. We filter based on the same rules:
- First tokens: starts with space followed by alpha chars (base models) or starts with alpha (instruct)
- Continuation tokens: all alpha, no space
- Word boundary: starts with space (indicates new word, completing previous)

**CLI args:** Keep `--all`, `--context`, `--data`, `--top-k`, `--show-probs`, `--sigma`, `--dry-run`, `--word`/`--file`. Remove `--model` (local model) and `--x-factor`. Add client args via `add_client_args()`. Add `--max-depth` (default 5, was hardcoded `MAX_TOKENS_PER_WORD`).

**Files:** `nextwords.py`

### 5. Remove nextwords_sync.py

It's identical to nextwords.py and no longer needed.

## File summary

| File | Action |
|------|--------|
| `client.py` | Add `add_client_args()`, `resolve_client_args()`, `send_completions_request()` |
| `evalpair.py` | Replace duplicated args with `add_client_args()` + `resolve_client_args()` |
| `eval_prompt.py` | Replace duplicated args with `add_client_args()` + `resolve_client_args()` |
| `nextwords.py` | Full rewrite: async beam search via API |
| `nextwords_sync.py` | Delete |

## Verification

1. Run `python evalpair.py --help` and `python eval_prompt.py --help` — confirm args unchanged
2. Run `python nextwords.py --help` — confirm new args present
3. Test nextwords.py with a running server: `python nextwords.py -w "test" --host localhost --port 8000`
4. Test with file input: `python nextwords.py -f wordlist.txt --host localhost --mc 4`

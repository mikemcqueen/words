import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Awaitable, Callable, Dict, List


def single_pair_generator(pair_string):
    """Generator that yields a single pair string, then ends."""
    yield pair_string

def file_pair_generator(filename):
    """Generator that yields pairs from a file, one per line."""
    with open(filename, 'r') as f:
        for line in f:
            pair = line.strip()
            if pair:
                yield pair


def parse_yesno_response(yesno: str) -> str | None:
    """Return normalized YES/NO for exact matches, else None."""
    if yesno:
        yesno = yesno.strip().upper()
        if yesno.startswith("YES"):
            return "YES"
        elif yesno.startswith("NO"):
            return "NO"
    return None


def flip_pair(pair: str) -> str:
    """Flip a comma-separated pair."""
    flipped = pair.split(',')
    assert len(flipped) == 2, f"{pair} len: {len(flipped)}"
    return ",".join((flipped[1], flipped[0]))


async def eval_with_flipped_retry(
    prompt_text: str,
    pair: str,
    attempt_fn: Callable[[str, str, bool], Awaitable[Dict]],
) -> Dict:
    """Evaluate a pair, retrying once on the flipped pair unless the first result is YES.

    `attempt_fn` receives `(candidate_pair, prompt, flipped)` and returns a dict
    containing at least `raw` and `normalized` keys.
    """
    attempts = []

    orig_prompt = prompt_text.replace("{PAIR}", pair.replace(",", " "))
    orig_attempt = await attempt_fn(pair, orig_prompt, False)
    attempts.append(orig_attempt)
    orig_normalized = orig_attempt["normalized"]

    if orig_normalized == "YES":
        return {
            "pair": pair,
            "normalized": "YES",
            "flipped": False,
            "decision_source": "original",
            "attempts": attempts,
        }

    flipped_pair = flip_pair(pair)
    flipped_prompt = prompt_text.replace("{PAIR}", flipped_pair.replace(",", " "))
    flipped_attempt = await attempt_fn(flipped_pair, flipped_prompt, True)
    attempts.append(flipped_attempt)
    flipped_normalized = flipped_attempt["normalized"]

    if flipped_normalized == "YES":
        normalized = "YES"
        flipped = True
        decision_source = "flipped"
    elif orig_normalized is None:
        normalized = None
        flipped = False
        decision_source = "original"
    else:
        normalized = orig_normalized
        flipped = False
        decision_source = "original"

    return {
        "pair": pair,
        "normalized": normalized,
        "flipped": flipped,
        "decision_source": decision_source,
        "attempts": attempts,
    }


def parse_on_off(value: str) -> bool:
    normalized = value.lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise argparse.ArgumentTypeError("expected 'on' or 'off'")


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", metavar="MODEL", type=str, default=None,
                        help="Model ID to use (required if server has multiple models)")
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
    parser.add_argument("--thinking", type=parse_on_off, choices=(True, False), default=False,
                        metavar="on|off",
                        help="Control API thinking mode (default: off)")


def add_print_keys_arg(parser, *, help_text: str) -> None:
    """Add a shared --pk/--print-keys CLI switch."""
    parser.add_argument("--pk", "--print-keys", dest="print_keys", action="store_true",
                        help=help_text)


def print_displayed_keys(rows) -> None:
    """Print unique individual keys from displayed row labels/keys.

    Each displayed row key may itself be a comma-separated combination.
    Split each displayed key on commas, add trimmed parts to a set, then
    print the unique set as a comma-separated list.
    """
    keys = set()
    for key, _ in rows:
        for part in str(key).split(','):
            part = part.strip()
            if part:
                keys.add(part)
    if keys:
        print(f"{len(keys)} keys:\n")
        print(','.join(keys))


def load_expected_pairs(filepath: str) -> dict:
    """Load a pairs JSON file and return a {pair: expected} dict (space-separated keys)."""
    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON: {filepath}")
            raise
    return {entry["pair"].replace(",", " "): entry["expected"].upper() for entry in data}


def load_eval_results(path) -> Dict:
    """Load evalpair JSONL, return {pair: {"logprobs": ...}} sorted by seq."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("seq", 0))
    return {
        r["pair"]: {"logprobs": r["logprobs"]}
        for r in records
    }


def parse_result_filename(name):
    """Parse {pair_file}_{prompt_file}_{prompt_id}_{host}.json.
    Returns (pair_file, prompt_file, prompt_id, host) or None if no match."""
    stem = Path(name).stem
    #m = re.match(r'^(.+)_([^_]+)_(p\d+)_(.+?)(?:\.(.+))?$', stem)
    m = re.match(r'^(.+)_([^_]+)_(p\d+)_(.+?)(?:_([^._]+))?(?:\.(.+))?$', stem)
    return (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)) if m else None


def key_from_path(path) -> str:
    """Display key for a result file: {prompt_file}.{prompt_id}[.{sys}][.{tag}].
    Raises ValueError if the filename doesn't match the expected pattern."""
    name = Path(path).name
    parsed = parse_result_filename(name)
    if parsed is None:
        raise ValueError(f"could not parse key from filename: {name}")
    _, prompt_file, prompt_id, _, sys_prompt, tag = parsed
    key = f"{prompt_file}.{prompt_id}"
    if sys_prompt:
        key += f".{sys_prompt}"
    if tag:
        key += f".{tag}"
    return key


def discover_files_all(seed_path):
    """Return {prompt_file.prompt_id[.tag]: eval_results} for all .jsonl files in seed's directory.
    seed_path may be a file (uses its parent) or a directory."""
    p = Path(seed_path)
    directory = p if p.is_dir() else p.parent
    candidates = []
    for jsonl_file in sorted(directory.glob('*.jsonl')):
        try:
            key = key_from_path(jsonl_file)
        except ValueError:
            continue
        candidates.append((key, jsonl_file))

    def _load(item):
        key, jsonl_file = item
        try:
            return key, load_eval_results(jsonl_file)
        except Exception:
            return key, None

    found = {}
    with ThreadPoolExecutor() as executor:
        for key, result in executor.map(_load, candidates):
            if result is not None:
                found[key] = result
    return found


def compute_stats(eval_results):
    correct = fp = fn = 0
    for data in eval_results.values():
        label = data.get('label')
        if label == 'correct': correct += 1
        elif label == 'fp':    fp += 1
        elif label == 'fn':    fn += 1
    total = correct + fp + fn
    pct   = 100 * correct / total if total else 0.0
    return dict(correct=correct, total=total, pct=pct, fp=fp, fn=fn)


def pair_has_yes(data) -> bool:
    """Return whether any direction predicted YES for a scored pair entry."""
    result = data.get('result')
    if result is None:
        return False
    return any(result[dir_].token == 'YES' for dir_ in result['logprobs'])


def expected_yesno_from_labeled_result(data) -> str | None:
    """Derive the expected YES/NO value from a labeled result entry."""
    result = data.get('result')
    if result is None:
        return None
    first_dir = next(iter(result['logprobs']))
    tl = result[first_dir]
    if tl.label == 'fp':
        return 'NO'
    if tl.label == 'fn':
        return 'YES'
    return tl.token


def print_stats(label, stats, w=24):
    print(f"{label:<{w}s}  {stats['correct']:3d}/{stats['total']:3d} ({stats['pct']:5.1f}%)  "
          f"{stats['fp']:3d}  {stats['fn']:3d}")


def _fmt_logprobs(lp) -> str:
    """Format a logprobs dict compactly, e.g. fwd=[{NO: 0.689}, {YES: 0.310}] rvs=[...]"""
    def fmt_val(v):
        return f"{v*100.0:4.1f}%" if isinstance(v, float) else str(v)
    def fmt_dict(d):
        return ", ".join(f"{k:>3}: {fmt_val(v)}" for k, v in d.items())
    def fmt_list(lst):
        return "[" + ", ".join(fmt_dict(x) if isinstance(x, dict) else str(x) for x in lst) + "]"
    parts = []
    for k, v in lp.items():
        parts.append(f"{k}={fmt_list(v)}" if isinstance(v, list) else f"{k}={fmt_val(v)}")
    return " ".join(parts)


def print_bad_pairs(labeled_results: dict, header: str = "", sources=None) -> None:
    """Print FP and FN pair lists from a labeled results dict.

    sources: optional list of (label, results_dict) — when provided, prints the
    logprobs dict for each contributing file below each pair, indented 4 spaces.
    """
    fp_pairs = sorted(p for p, d in labeled_results.items() if d.get('label') == 'fp')
    fn_pairs = sorted(p for p, d in labeled_results.items() if d.get('label') == 'fn')
    w = max((len(lbl) for lbl, _ in sources), default=0) if sources else 0
    if header:
        print(header)
    if fp_pairs:
        print("FP:\n---")
        for pair in fp_pairs:
            print(f"{pair}")
            if sources:
                for lbl, results in sources:
                    lp = results.get(pair, {}).get('logprobs')
                    print(f"  {lbl:<{w}} {_fmt_logprobs(lp) if lp is not None else None}")
    if fn_pairs:
        print("FN:\n---")
        for pair in fn_pairs:
            print(f"{pair}")
            if sources:
                for lbl, results in sources:
                    lp = results.get(pair, {}).get('logprobs')
                    if lp is not None:
                        print(f"  {lbl:<{w}} {_fmt_logprobs(lp) if lp is not None else None}")
    if fp_pairs or fn_pairs:
        print()


def print_discovery_ranked(files_dict, sort='score'):
    """Print a ranked table of pre-labeled eval results: key | score% | correct/total | FP | FN."""
    rows = [(key, compute_stats(records)) for key, records in files_dict.items()]

    sort_lc = sort.lower()
    if sort_lc == 'fp':
        rows.sort(key=lambda x: (x[1]['fp'], -x[1]['pct']))
    elif sort_lc == 'fn':
        rows.sort(key=lambda x: (x[1]['fn'], -x[1]['pct']))
    else:
        rows.sort(key=lambda x: x[1]['pct'], reverse=True)

    w = max((len(k) for k, _ in rows), default=5)
    w = max(w, len('key'))
    print(f"{'key':<{w}s}  {'score':>7s}  {'corr':>9s}  {'FP':>4s}  {'FN':>4s}")
    print(f"{'─'*(w + 32)}")
    for key, s in rows:
        print(f"{key:<{w}s}  {s['pct']:6.1f}%  {s['correct']:4d}/{s['total']:<4d}  {s['fp']:4d}  {s['fn']:4d}")
    print()


def resolve_key(file_or_dir, key, *, enforce_unique=False):
    """Given a directory and a discovery key like 'crosswd2.p81.qwen35', find the matching .jsonl file.
    Splits key into (prompt_file, pid, tag) and returns the first matching .jsonl file.
    When enforce_unique is True, exits if multiple files match."""
    parts = key.split('.', 2)
    if len(parts) == 3:
        prompt_file, pid, tag = parts
    elif len(parts) == 2:
        prompt_file, pid, tag = parts[0], parts[1], ''
    else:
        print(f"Error: invalid key {key!r} (expected prompt_file.pid[.tag])", file=sys.stderr)
        sys.exit(1)

    p = Path(file_or_dir)
    d = p if p.is_dir() else p.parent
    pattern = f'*_{prompt_file}_{pid}_*.{tag}.jsonl' if tag else f'*_{prompt_file}_{pid}_*.jsonl'
    candidates = []
    for f in sorted(d.glob(pattern)):
        parsed = parse_result_filename(f.name)
        if parsed is None:
            continue
        _, pf, ppid, _, psys, ptag = parsed
        if pf == prompt_file and ppid == pid and (ptag or '') == tag:
            candidates.append(f)

    if len(candidates) == 0:
        print(f"Error: no file found for key {key!r} in {d}", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Error: multiple files found for key {key!r}:", file=sys.stderr)
        for c in candidates:
            print(f"  {c.name}", file=sys.stderr)
        if enforce_unique:
            sys.exit(1)
    return candidates[0]


def load_prompts_from_file(filepath) -> List[Dict]:
    """Load prompts from a file. Detects format by extension:
      .jsonl  -> one JSON object per line
      other   -> JSON array of objects
    Adds _source_file (stem) to each prompt object.
    """
    filepath = Path(filepath)
    prompts = []
    if filepath.suffix == '.jsonl':
        for line in filepath.read_text().splitlines():
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    else:
        prompts = json.loads(filepath.read_text())
    for p in prompts:
        p['_source_file'] = filepath.stem
    return prompts

import argparse
import json
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
    parser.add_argument("--thinking", type=parse_on_off, choices=(True, False), default=False,
                        metavar="on|off",
                        help="Control API thinking mode (default: off)")


def load_expected_pairs(filepath: str) -> dict:
    """Load a pairs JSON file and return a {pair: expected} dict (space-separated keys)."""
    with open(filepath) as f:
        data = json.load(f)
    return {entry["pair"].replace(",", " "): entry["expected"].upper() for entry in data}


def load_result_records(path) -> List[Dict]:
    """Load evalpair JSONL, return records sorted by seq."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("seq", 0))
    return records


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

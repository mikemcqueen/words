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

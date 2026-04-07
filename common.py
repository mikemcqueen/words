import json
from pathlib import Path
from typing import Dict, List


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

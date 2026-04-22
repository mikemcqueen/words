import argparse

from src.info import info
from src.model import load_model, clear_cache

from pathlib import Path
#from wpe import WordProbabilityExplorer

# Write tuples to file (already sorted by probability)
def dump_probs(filename, word_probs, args):
    path = args.data / filename
    prob_sum = 0.0
    with open(path, "w") as f:
        for word, prob in word_probs:
            prob_sum += prob
            f.write(f"{word},{prob}\n")
    info(f"prob_sum: {prob_sum}")

# Sort alphabetically and write just the words
def dump_words(filename, word_probs, args):
    words = sorted(word_probs, key=lambda x: x[0])
    path = args.data / filename
    with open(path, "w") as f:
        for word, _ in words:
            f.write(f"{word}\n")

def do(explorer, word: str, args):
    info(f"Word: {word}")
    clear_cache(explorer.device)

    word_log_probs, t = explorer.find_word_log_probs(args.context + word, args)
    top_words = explorer.to_word_probs(word_log_probs)
        
    if not args.dry_run:
        dump_probs(f"{word}.probs", top_words, args)
        dump_words(f"{word}.all", top_words, args)

    """
    threshold = 0.95
    prob_sum = 0.0
    for i, (word, prob) in enumerate(top_words):
        prob_sum += prob
        #if threshold and prob_sum >= threshold:
        #    print("--------threshold---------")
        #    threshold = None
        print(f"{word},{prob:.8f}")
    info(f"prob_sum: {prob_sum}")
    """
    info(f"Time: total: {t['total']:.3f}s  next: {t['next']:.3f}s  forward: {t['forward']:.3f}s  iters: {t['iters']}s")

def word_generator(filepath: str):
    """Generator that yields words from a text file one at a time."""
    with open(filepath, 'r') as f:
        for line in f:
            for word in line.split():
                yield word

def main():
    DEFAULT_CONTEXT = "<|en-us|>"
    DEFAULT_MODEL = "g4it"
    DEFAULT_FIRST_K = 1000
    DEFAULT_SIGMA = 1.0
    DEFAULT_DATA_DIR = "./data"

    parser = argparse.ArgumentParser(description='Determine next-word probability for a word or all unprocessed words in a file')
    parser.add_argument(      "--all", action="store_true", help=f"process all words in file; used with -f FILE")
    parser.add_argument("-c", "--context", type=str, default=DEFAULT_CONTEXT, help=f"context prefix, default: {DEFAULT_CONTEXT}")
    parser.add_argument("-d", "--data", type=str, default=DEFAULT_DATA_DIR, help=f"data directory, default: {DEFAULT_DATA_DIR}")    
    parser.add_argument('-k', '--top-k', type=int, default=DEFAULT_FIRST_K, help=f"select top-k first tokens, default: {DEFAULT_FIRST_K}")
    parser.add_argument("-m", "--model", metavar='q3|g4', type=str, default=DEFAULT_MODEL, help=f"select model, default: {DEFAULT_MODEL}")
    parser.add_argument("-p", "--show-probs", metavar='N', type=int, default=0, help='show N top probabilities')
    parser.add_argument("-s", "--sigma", type=float, default=DEFAULT_SIGMA, help=f"typicality sigma, default: {DEFAULT_SIGMA}")
    parser.add_argument("-x", "--x-factor", action="store_true", help=f"alternate approach")
    parser.add_argument("-y", "--dry-run", action="store_true", help=f"dry run (no data written to file)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w', '--word', type=str, help='A single word')
    group.add_argument('-f', '--file', type=str, help='Path to a text file')

    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"Data dir '${args.data}' doesn't exist.")
        exit()
    args.data = path

    device, model, tokenizer = load_model(args.model)

    explorer = {} #WordProbabilityExplorer(model, tokenizer, device, typicality_sigma=args.sigma)
        
    if args.word:
        do(explorer, args.word, args)
    else:
        info(f"File: {args.file}")
        for word in word_generator(args.file):
            probs_path = args.data / f"{word}.probs"
            if args.all or not probs_path.exists():
                do(explorer, word, args)


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print("ImportError. Could be missing 'transformers' and/or 'torch' libraries.")
        print("Try installing them using: pip install transformers torch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

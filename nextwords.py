import argparse

from model import load_model, clear_cache
from pathlib import Path
from info import info
from words import WordProbabilityExplorer

TYPICALITY_SIGMA = 2.0
DATA_DIR = './data'

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
    top_words, t = explorer.find_top_words(word)
        
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
    parser = argparse.ArgumentParser(description='Process a word or file of words')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w', '--word', type=str, help='A single word')
    group.add_argument('-f', '--file', type=str, help='Path to a text file')

    args = parser.parse_args()

    args.data = DATA_DIR

    path = Path(args.data)
    if not path.exists():
        print(f"Data dir '${args.data}' doesn't exist.")
        exit()

    args.data = path

    device, model, tokenizer = load_model()

    # Create explorer with asymmetric typical sampling
    info(f"sigma: {TYPICALITY_SIGMA}")
    explorer = WordProbabilityExplorer(model, tokenizer, device, typicality_sigma=TYPICALITY_SIGMA)
        
    if args.word:
        do(explorer, args.word, args)
    else:
        info(f"File: {args.file} (not implemented)")
        for word in word_generator(args.file):
            do(explorer, word, args)


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print("Error: The 'transformers' and 'torch' libraries are required.")
        print("Please install them using: pip install transformers torch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

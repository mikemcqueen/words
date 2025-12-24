import argparse
from collections import defaultdict


def load_wordlist(filepath: str) -> tuple[list[str], set[str]]:
    """Load a wordlist file (one word per line) and return as both list and set."""
    with open(filepath, 'r') as f:
        word_list = [line.strip() for line in f if line.strip()]
    word_set = set(word_list)
    return word_list, word_set


def load_pair_list(filepath: str) -> dict[str, set[str]]:
    """Load a pair list file (comma-delimited pairs) and create mapping."""
    pair_map: dict[str, set[str]] = defaultdict(set)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                word1, word2 = parts[0].strip(), parts[1].strip()
                pair_map[word1].add(word2)
                #pair_map[word2].add(word1)
    
    return dict(pair_map)


def find_pairs(
    word_list: list[str],
    word_set: set[str],
    pair_map: dict[str, set[str]],
    args
) -> None:
    """
    For each word in wordlist, load its word-prob file, extract potential pair words,
    and print valid pairs (with checkmark if not already in pair map).
    """
    found_pairs: dict[str, int] = defaultdict(int)

    for word in word_list:
        # Construct path to word-prob file
        word_prob_path = f"{args.data_dir}/{word}.probs"
        
        try:
            with open(word_prob_path, 'r') as f:
                count = 0
                for line in f:
                    if args.k and count >= args.k:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 2:
                        potential_pair_word = parts[0].strip()
                        if not potential_pair_word or potential_pair_word == word:
                            continue
                        
                        count += 1
                        
                        # Check if potential pair word is in the original wordlist
                        if potential_pair_word in word_set:
                            # Check if it's an existing pair
                            existing_pair = None
                            if word in pair_map and potential_pair_word in pair_map[word]:
                                existing_pair = f"{word},{potential_pair_word}"
                            elif potential_pair_word in pair_map and word in pair_map[potential_pair_word]:
                                existing_pair = f"{potential_pair_word},{word}"
                            
                            # Print based on flags
                            if existing_pair and not args.no_existing:
                                found_pairs[existing_pair] += 1 # print(f"{word},{potential_pair_word}")
                            elif not existing_pair and not args.no_new:
                                reverse_pair = f"{potential_pair_word},{word}"
                                if reverse_pair in found_pairs:
                                    found_pairs[reverse_pair] += 1
                                else:
                                    forward_pair = f"{word},{potential_pair_word}"
                                    found_pairs[forward_pair] += 1
        except FileNotFoundError:
            # Skip if word-prob file doesn't exist for this word
            continue

    return dict(found_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Find and validate word pairs from word-probability files."
    )
    parser.add_argument('-w', '--words', required=True, help="Path to wordlist file (one word per line)")
    parser.add_argument('-p', '--pairs', required=True, help="Path to pair list file (comma-delimited pairs)")
    parser.add_argument('-d', '--data-dir', required=True, help="Directory containing word-probability files")
    parser.add_argument('-k', type=int, default=100, help="Number of words to read from each word-prob file (default: 100)")
    parser.add_argument('-c', '--show-match-count', action='store_true', help="Display match count (1 or 2)" )
    parser.add_argument('--no-existing', action='store_true', help="Don't print existing pairs" )
    parser.add_argument('--no-new', action='store_true', help="Don't print new pairs")
    
    args = parser.parse_args()
    
    # Load data
    word_list, word_set = load_wordlist(args.words)
    pair_map = load_pair_list(args.pairs)
    
    # Find and print pairs
    found_pairs = find_pairs(word_list, word_set, pair_map, args)

    for pair, count in found_pairs.items():
        #check = "✓✓" if count == 2 else "✓"
        suffix = f" {count}" if args.show_match_count else ""
        print(f"{pair}{suffix}")


if __name__ == "__main__":
    main()

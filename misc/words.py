# mypy: disable-error-code="misc"

# words.py
#
# This version implements Asymmetric Typical Sampling based on standard deviation 
# around entropy. Unlike symmetric approaches, this only filters surprisingly 
# LOW-probability tokens while always keeping high-probability tokens.
#
# Asymmetric typical sampling algorithm:
# 1. Compute entropy H = E[-log(p)] = Σ(p_i * -log(p_i))
# 2. Compute standard deviation σ = sqrt(Σ(p_i * (-log(p_i) - H)²))
# 3. Apply ONE-SIDED threshold: keep tokens where -log(p) <= H + k*σ
# 4. High probability tokens (low -log p) always pass (no lower bound)
# 5. Low probability tokens filtered only if too surprising (high -log p)
#
# Key advantages:
# - Never excludes high-probability tokens just because they're "too predictable"
# - Uses familiar statistical concept (standard deviation)
# - Easy to tune: k=2.0 means "within 2 standard deviations above entropy"
# - Adapts to distribution shape (peaked vs flat)
# - No cumsum complexity, just hard threshold

import argparse
import torch
import math
import os
import time
import heapq

from src.info import info
from src.model import load_model, clear_cache, is_gemma_model, is_instruct_model, specialize_prompt

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configuration ---
TOP_K_WORDS = 1000
BATCH_SIZE = 64
BATCH_UPDATE_INTERVAL = 1             # Update top-k threshold every N batches
MAX_TOKENS_PER_WORD = 5               # Maximum word length in tokens
MIN_LOG_PROBABILITY = math.log(1e-8)  # Prune paths with log probability below this

def has_trailing_period(word: str) -> bool:
    return word[-1] == '.'

def is_alpha_with_trailing_period(word: str) -> bool:
    word = word.rstrip()
    if word and has_trailing_period(word):
        word = word[:-1]
        if word and word.isalpha():
            return True
    return False

def is_all_ascii_alpha(word: str) -> bool:
    return bool(word) and word.isascii() and word.isalpha()

def has_leading_ascii_alpha(word: str) -> bool:
    return bool(word) and word[0].isascii() and word[0].isalpha()

def has_trailing_ascii_alpha(word: str) -> bool:
    return bool(word) and word[-1].isascii() and word[-1].isalpha()

def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

class WordProbabilityExplorer:
    def __init__(self, model, tokenizer, device, typicality_sigma: float):
        info(f"sigma: {typicality_sigma}")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.typicality_sigma = typicality_sigma  # Sigma threshold for asymmetric filtering

        # Track best log probability for each unique word
        self.word_log_probs: dict[str, float] = {}

        # Cache for decoded token strings - decode each token ID only once
        self.token_text_cache: dict[int, str] = {}

        # Statistics
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0

        # Pre-compute special token IDs for faster checking
        self.special_token_ids = set(self.tokenizer.all_special_ids)

        # Get actual vocab size from model (not tokenizer, they can differ!)
        self.vocab_size = model.config.vocab_size

        # Pre-decode and categorize all tokens for faster filtering
        info("Pre-analyzing vocabulary...")
        self.token_info = self._precompute_token_info()

        info(f"  Vocab size: {self.vocab_size}")
        info(f"  Valid first tokens: {len(self.token_info['valid_first'])}")
        info(f"  Valid continuation tokens: {len(self.token_info['valid_continuation'])}")
        info(f"  Word boundary tokens: {len(self.token_info['word_boundary'])}")

        # Move masks to device
        self.token_info['first_token_mask'] = self.token_info['first_token_mask'].to(device)
        self.token_info['continuation_mask'] = self.token_info['continuation_mask'].to(device)

        self.word_boundary_mask = self.token_info['first_token_mask'].cpu().numpy()

    def _precompute_token_info(self) -> dict:
        """
        Pre-decode and categorize all tokens in vocabulary for faster filtering.
        Also precompute static masks for efficient filtering.
        Returns dict with sets of token IDs for different categories and mask tensors.
        """
        valid_first = set()  # Tokens that can start a word (space + alpha)
        valid_continuation = set()  # Tokens that can continue a word (alpha, no space)
        valid_word_boundary = set() # Tokens that indicate a word boundary

        # Use the actual model vocab size, not len(tokenizer)
        vocab_size = self.vocab_size

        eot_token_id = None
        for token_id in range(vocab_size):
            # Skip special tokens
            if token_id in self.special_token_ids:
                continue

            # Decode and cache
            token_text = self.tokenizer.decode([token_id])
            self.token_text_cache[token_id] = token_text

            if not token_text or not token_text.isascii():
                continue

            if is_instruct_model(self.model):
                # Check if starts with alpha character
                if token_text[0].isalpha():
                    valid_first.add(token_id)
                    valid_continuation.add(token_id)
                elif token_text == '<end_of_turn>': # TODO gemma specific
                    print("found eot_token")
                    eot_token_id = token_id
            else:
                # Check if starts with space
                if token_text[0] == ' ':
                    word = token_text[1:]
                    if word and word.is_alpha(): # is_all_ascii_alpha(word)
                        valid_first.add(token_id)
                        valid_word_boundary.add(token_id)
                        valid_continuation.add(token_id)
                # No space, contains all ascii alpha - can continue a word
                elif token_text.is_alpha(): #is_all_ascii_alpha(token_text):
                    valid_continuation.add(token_id)

        if is_instruct_model(self.model):
            if not eot_token_id:
                print("eot_token_id not found")
            eos = self.model.generation_config.eos_token_id
            eos_list = [] if eos is None else [eos] if isinstance(eos, int) else eos
            for token_id in eos_list:
                if token_id == eot_token_id:
                    print("found eot_token_id in eos_tokens")
                valid_word_boundary.add(token_id)
                valid_continuation.add(token_id)

        # Create static additive masks with correct vocab size
        first_token_mask = torch.full((vocab_size,), float('-inf'))
        for token_id in valid_first:
            first_token_mask[token_id] = 0.0
        first_token_mask = (first_token_mask == 0.0)

        continuation_mask = torch.full((vocab_size,), float('-inf'))
        for token_id in valid_continuation:
            continuation_mask[token_id] = 0.0
        continuation_mask = (continuation_mask == 0.0)

        return {
            'valid_first': valid_first,
            'valid_continuation': valid_continuation,
            'word_boundary': valid_word_boundary,
            'first_token_mask': first_token_mask,
            'continuation_mask': continuation_mask
        }        
    
    def show_probs(self, probs: torch.Tensor, k: int):
        top_probs, top_indices = torch.topk(probs, k)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            idx_val = idx.item()
            valid = "✓" if self.token_info['first_token_mask'][idx_val] else "✗"
            print(f"{i+1:3} {valid} Index {idx_val}: {prob.item()}")

    def decode_token(self, token_id: int) -> str:
        """
        Get decoded token text from cache (already pre-computed).
        """
        return self.token_text_cache.get(token_id, '')
    
    def get_memory_usage(self) -> str:
        """
        Get current GPU memory usage as a formatted string.
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
            return f"{allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
        elif self.device.type == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024**2  # MB
            return f"{allocated:.1f}MB allocated"
        else:
            return "N/A (CPU)"

    def pne(self, log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pne = Probs, Neg_log_probs, Entropy
        """
        # Convert to probabilities
        probs = torch.exp(log_probs)
        neg_log_probs = -log_probs

        # Compute entropy: H = E[-log(p)] = Σ(p_i * -log(p_i))
        entropy = torch.sum(probs * neg_log_probs, dim=-1, keepdim=True)  # [batch_size, 1]

        if log_probs.shape[0] == 1: # first_tokens batch (probably)
            info(f"entropy: {entropy[0][0]}")

        return probs, neg_log_probs, entropy

    def typical_sampling_batch(self, log_probs: torch.Tensor,
                               valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply asymmetric typical (entropy-based) sampling to select tokens for a batch.
        
        Uses standard deviation around entropy to filter tokens:
        - Computes entropy H (expected surprisal) and std dev σ
        - Keeps tokens where -log(p) <= H + k*σ (one-sided upper bound)
        - High probability tokens are ALWAYS kept (no lower bound)
        - Low probability tokens are filtered if too surprising
        
        This is asymmetric: we don't exclude tokens for being too probable,
        only for being too improbable relative to the distribution's entropy.
        
        Args:
            log_probs: Log probabilities of shape [batch_size, num_candidates]
            valid_mask: Additive mask tensor of shape [vocab_size] with 0.0 for valid tokens, -inf for invalid
            
        Returns:
            Tuple of (selected_log_probs, selected_indices, selected_mask)
            - selected_log_probs: [batch_size, max_selected]
            - selected_indices: [batch_size, max_selected]
            - selected_mask: [batch_size, max_selected] - True for valid tokens
        """
        batch_size, num_candidates = log_probs.shape

        if self.typicality_sigma > 0.0:
            probs, neg_log_probs, entropy = self.pne(log_probs)

            # Compute standard deviation of surprisal around entropy
            # σ = sqrt(E[(X - μ)²]) where X = -log(p), μ = H
            variance = torch.sum(probs * (neg_log_probs - entropy)**2, dim=-1, keepdim=True)
            std_dev = torch.sqrt(variance + 1e-10)  # Small epsilon for numerical stability

            # Asymmetric threshold: only filter tokens that are too surprising (high -log p)
            # No lower bound: high probability tokens (low -log p) always pass
            upper_bound = entropy + self.typicality_sigma * std_dev  # [batch_size, 1]

            typical_mask = neg_log_probs <= upper_bound  # [batch_size, vocab_size]

            # Combine with validity mask (valid_mask is True for allowed tokens)
            keep_mask = typical_mask & valid_mask  # [batch_size, vocab_size]
        else:
            # Expand valid_mask from [vocab_size] to [batch_size, vocab_size]
            keep_mask = valid_mask.unsqueeze(0).expand_as(log_probs)

        # Sort by probability (descending) - highest probability tokens first
        prob_sorted_indices = torch.argsort(log_probs, dim=-1, descending=True)
        
        # Apply sorting
        sorted_mask = torch.gather(keep_mask, -1, prob_sorted_indices)
        sorted_log_probs = torch.gather(log_probs, -1, prob_sorted_indices)

        sorted_token_indices = prob_sorted_indices
        
        # Find max number selected across batch for uniform tensor size
        num_selected_per_item = sorted_mask.sum(dim=-1)
        max_selected = num_selected_per_item.max().item()
        max_selected = max(1, max_selected)  # Ensure at least one
        
        # Truncate to max_selected for uniform tensor shape
        selected_log_probs = sorted_log_probs[:, :max_selected]
        selected_indices = sorted_token_indices[:, :max_selected]
        selected_mask = sorted_mask[:, :max_selected]
        
        return selected_log_probs, selected_indices, selected_mask

    def get_batched_log_probs(self, input_ids_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Get log probability distributions for next token for a batch of sequences.
        Only considers valid tokens (filtered by mask) when computing softmax.
        
        Args:
            input_ids_list: List of input_id tensors of shape [1, seq_len]
        
        Returns:
            log_probs: Tensor of shape [batch_size, vocab_size] with log probabilities
        """
        if not input_ids_list:
            raise Exception("what")

        # Find max length for padding
        max_len = max(ids.shape[1] for ids in input_ids_list)

        # Pad sequences to same length (pad on the left to preserve causality)
        padded_inputs = []
        attention_masks = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for input_ids in input_ids_list:
            seq_len = input_ids.shape[1]
            if seq_len < max_len:
                # Pad on the left
                padding = torch.full(
                    (1, max_len - seq_len), 
                    pad_token_id,
                    dtype=torch.long, 
                    device=self.device
                )
                padded = torch.cat([padding, input_ids], dim=1)
                # Attention mask: 0 for padding, 1 for real tokens
                mask_tensor = torch.cat([
                    torch.zeros(1, max_len - seq_len, dtype=torch.long, device=self.device),
                    torch.ones(1, seq_len, dtype=torch.long, device=self.device)
                ], dim=1)
            else:
                padded = input_ids
                mask_tensor = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

            padded_inputs.append(padded)
            attention_masks.append(mask_tensor)

        # Stack into batch
        batch_input_ids = torch.cat(padded_inputs, dim=0)  # [batch_size, max_len]
        batch_attention_mask = torch.cat(attention_masks, dim=0)  # [batch_size, max_len]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]

        self.forward_passes += 1

        # Clean up intermediate tensors
        del batch_input_ids, batch_attention_mask, outputs, logits

        return log_probs
    
    def find_word_log_probs(self, prompt: str, args) -> tuple[list[tuple[str, float]], dict]:
        """
        Find the top K most probable next words using iterative BFS with batching.
        
        Args:
            prompt: The input text prompt
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        info(f"Prompt: '{prompt}'")

        t0_total = time.perf_counter()

        # Reset state (but keep pre-computed token info and cache)
        self.word_log_probs = {}
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0

        # Dynamic threshold for top-k pruning (k-th best log prob seen so far)
        top_k_threshold = float('-inf')

        # Encode prompt
        base_input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True).to(self.device)

        #"""
        outputs = self.model.generate(input_ids=base_input_ids, max_new_tokens=1000)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"response: {response}")
        #"""

        # Get log probabilities for first token
        info("Computing first token probabilities...")
        d_first_log_probs = self.get_batched_log_probs([base_input_ids])

        # Force display of entropy
        self.pne(d_first_log_probs)

        if args.show_probs:
            self.show_probs(d_first_log_probs[0], args.show_probs)
        
        if not args.first_k:
            sampling_method = "Asymmetric typical"
            # Apply typical sampling to dynamically select first tokens
            d_first_values, d_first_indices, d_first_mask = self.typical_sampling_batch(
                d_first_log_probs, self.token_info['first_token_mask']
            )
        
            # Move to CPU
            first_values = d_first_values[0].cpu()
            first_indices = d_first_indices[0].cpu()
            first_mask = d_first_mask[0].cpu()

            # Use boolean indexing to get only valid tokens
            valid_first_values = first_values[first_mask]
            valid_first_indices = first_indices[first_mask]
        else:
            sampling_method = "TopK"

            # Get log probs for valid tokens only
            valid_first_log_probs = d_first_log_probs[0][self.token_info['first_token_mask']]
    
            # Get top k among valid tokens
            top_first = torch.topk(
                valid_first_log_probs, min(args.first_k, len(self.token_info['valid_first']))
            )
            
            valid_first_values = top_first.values.cpu()

            # Get the actual token IDs that are valid first tokens
            valid_token_ids = self.token_info['first_token_mask'].nonzero(as_tuple=True)[0]

            # Map filtered indices back to actual token IDs
            valid_first_indices = valid_token_ids[top_first.indices].cpu()

        num_first_tokens = len(valid_first_values)
        info(f"{sampling_method} sampling selected {num_first_tokens} initial tokens (out of {len(self.token_info['valid_first'])} valid)")
        if sampling_method == "Asymmetric typical":
            info(f"  Keeping tokens where -log(p) <= H + {self.typicality_sigma}σ")

        # Initialize paths for first tokens
        # Each path: {'tokens': [token_ids], 'text': decoded_text, 'log_prob': float}
        # Store text incrementally to avoid repeated decoding
        current_paths = []

        for log_prob, token_id in zip(valid_first_values, valid_first_indices):
            token_id = token_id.item()
            log_prob = log_prob.item()

            # Should always be valid due to masking
            if token_id not in self.token_info['valid_first']:
                continue

            # Get pre-decoded text and strip the leading space
            token_text = self.decode_token(int(token_id)).strip()

            current_paths.append({
                'tokens': [token_id],
                'text': token_text,  # Store decoded text to avoid re-decoding
                'log_prob': log_prob
            })

        depth = 1
        inner_loops = 0
        t_forward = 0.0
        t_mask = 0.0
        t_paths = 0.0

        t0_next = time.perf_counter()

        # Iterative BFS: process all paths at each depth level
        while current_paths and depth < MAX_TOKENS_PER_WORD:
            info(f"\n--- Depth {depth}: Processing {len(current_paths)} paths ---")
            info(f"    GPU Memory: {self.get_memory_usage()}")

            next_paths = []

            # Process paths in batches, pulling valid paths until BATCH_SIZE
            path_idx = 0
            batch_num = 0
            paths_processed = 0
            while path_idx < len(current_paths):
                # Build a full batch by pulling valid paths (skip below-threshold)
                batch_paths: list[dict] = []
                while len(batch_paths) < BATCH_SIZE and path_idx < len(current_paths):
                    path = current_paths[path_idx]
                    path_idx += 1
                    if path['log_prob'] >= top_k_threshold:
                        batch_paths.append(path)

                if not batch_paths:
                    break  # No more valid paths

                # Prepare batch of extended inputs
                batch_input_ids = []
                for path in batch_paths:
                    tokens_tensor = torch.tensor(path['tokens'], device=self.device).unsqueeze(0)
                    extended = torch.cat([base_input_ids, tokens_tensor], dim=1)
                    batch_input_ids.append(extended)

                t0_forward = time.perf_counter()
                d_next_log_probs = self.get_batched_log_probs(batch_input_ids)
                sync(self.device)
                t_forward += time.perf_counter() - t0_forward

                # Apply typical sampling to dynamically select continuation tokens
                d_next_values, d_next_indices, d_next_mask = self.typical_sampling_batch(
                    d_next_log_probs, self.token_info['continuation_mask']
                )

                # Move to CPU
                next_log_probs = d_next_values.float().cpu().numpy()
                next_indices = d_next_indices.cpu().numpy()
                next_mask = d_next_mask.cpu().numpy()

                # Free GPU memory
                del d_next_log_probs
                del d_next_values, d_next_indices, d_next_mask

                # Process each path in the batch
                t0_paths = time.perf_counter()
                for path_batch_idx, path in enumerate(batch_paths):
                    self.paths_explored += 1

                    # Use boolean indexing to get only valid tokens for this path
                    t0_mask = time.perf_counter()
                    path_mask = next_mask[path_batch_idx]
                    path_values = next_log_probs[path_batch_idx][path_mask]
                    path_indices = next_indices[path_batch_idx][path_mask]
                    t_mask += time.perf_counter() - t0_mask

                    for token_log_prob, token_id in zip(path_values, path_indices):
                        inner_loops += 1
                        new_log_prob = path['log_prob'] + token_log_prob

                        # Prune if log probability too low
                        if new_log_prob < MIN_LOG_PROBABILITY:
                            continue

                        # Check if this is a word boundary token
                        if token_id in self.token_info['word_boundary']:
                            # This completes a word - record it
                            # Word's log_prob = path['log_prob'], already >= threshold from pre-filter
                            word = path['text'].lower()

                            # Update best log probability for this word
                            if word not in self.word_log_probs:
                                self.num_unique_words += 1
                                self.word_log_probs[word] = path['log_prob']
                            elif path['log_prob'] > self.word_log_probs[word]:
                                self.word_log_probs[word] = path['log_prob']

                            continue

                        drop = new_log_prob < top_k_threshold
                        # Continuation - prune if can't make top-k
                        if drop:
                            continue

                        # Continue this path - append decoded token to text
                        next_token_text = self.decode_token(token_id)

                        next_paths.append({
                            'tokens': path['tokens'] + [token_id],
                            'text': path['text'] + next_token_text,
                            'log_prob': new_log_prob
                        })

                t_paths += time.perf_counter() - t0_paths
                paths_processed += len(batch_paths)

                # Update threshold every BATCH_UPDATE_INTERVAL batches
                if batch_num % BATCH_UPDATE_INTERVAL == 0 and len(self.word_log_probs) >= TOP_K_WORDS:
                    top_k = heapq.nlargest(TOP_K_WORDS, self.word_log_probs.values())
                    top_k_threshold = top_k[-1]

                if batch_num % 10 == 0:
                    info(f"  Batch {batch_num}: processed {paths_processed}/{len(current_paths)} paths, "
                          f"{self.num_unique_words} unique words found")

                batch_num += 1

            # Move to next depth
            current_paths = next_paths
            depth += 1

            # Explicitly free GPU memory between iterations
            clear_cache(self.device)

            info(f"  -> {len(current_paths)} paths continue to depth {depth}")
            info(f"     GPU Memory after cleanup: {self.get_memory_usage()}")

        t1 = time.perf_counter()
        t_next = t1 - t0_next
        t_total = t1 - t0_total

        info(f"\n--- Exploration Complete ---")
        info(f"Forward passes: {self.forward_passes}")
        info(f"Paths explored: {self.paths_explored}")
        info(f"Unique words found: {self.num_unique_words}")

        sorted_words = sorted(
            self.word_log_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_words, {
            'total': t_total, 'next': t_next, 'forward': t_forward, 'iters': inner_loops,
            'paths': t_paths, 'mask': t_mask
        }

    def to_word_probs(self, word_log_probs):
        # Convert log probabilities to probabilities
        word_probs = [
            (word, math.exp(log_prob))
            for word, log_prob in word_log_probs
        ]
        return word_probs

def normalize_probs(word_log_probs: list[tuple[str, float]]) -> list[float]:
    """
    Takes list of (word, log_prob) and returns a list of normalized_prob.
    """
    if not word_log_probs:
        return []
    
    log_probs = [lp for _, lp in word_log_probs]
    max_log_prob = max(log_probs)
    
    exp_probs = [math.exp(lp - max_log_prob) for lp in log_probs]
    total = sum(exp_probs)
    
    return [
        ep / total for ep in exp_probs
    ]

def parse_args():
    DEFAULT_SIGMA = 1.0        # typicality sigma
    DEFAULT_FIRST_K = 1000
    DEFAULT_MODEL = "g2"

    parser = argparse.ArgumentParser()
    parser.add_argument('ctx', nargs='?', help='Optional context')
    parser.add_argument('-k', '--first-k', type=int, default=DEFAULT_FIRST_K, help=f"select topk first tokens (default: {DEFAULT_FIRST_K})")
    parser.add_argument("-m", "--model", metavar='q3|l2|g2', type=str, default=DEFAULT_MODEL, help=f"select model (default: {DEFAULT_MODEL}")
    parser.add_argument('-s', '--sigma', type=float, default=DEFAULT_SIGMA,
                        help=f"typicality sigma (default: {DEFAULT_SIGMA}; use 0.0 to select all)")
    return parser.parse_args()

def main():
    DEF_WORD = "aquatic"
    DEF_BASE_CONTEXT = "<|en-us|>"
    DEF_INST_CONTEXT = "Given an input word, respond with a single word only," \
        " which represents the most meaningful word that follows it.\n"

    DEF_INST_CONTEXT = "You are an expert at English phrase construction. You will be given an input word, and must choose a single " \
        "word, which, when paired with the input word, results in a meaningful English phrase. Your response should consist of " \
        "the single word you chose, without any type of 'response: ' prefix or trailing punctuation.\n"

    INPUT_WORD = "Input word: "

    """
        " such that the .."

    """

    args = parse_args()

    args.show_probs = False

    device, model, tokenizer = load_model(args.model)

    if not args.ctx:
        if is_instruct_model(model):
            prompt = DEF_INST_CONTEXT + INPUT_WORD
        else:
            prompt = DEF_BASE_CONTEXT
        prompt += DEF_WORD
    else:
        prompt = args.ctx

    prompt = specialize_prompt(model, tokenizer, prompt)

    # Create explorer with asymmetric typical sampling
    info(f"Using Asymmetric Typical Sampling with sigma threshold: {args.sigma}")
    explorer = WordProbabilityExplorer(model, tokenizer, device, typicality_sigma=args.sigma)

    clear_cache(device)

    # Find top words
    word_log_probs, t = explorer.find_word_log_probs(prompt, args)

    #from pipeline import find_word_log_probs_pipelined
    #word_log_probs, t = find_word_log_probs_pipelined(explorer, prompt, args)

    top_words = explorer.to_word_probs(word_log_probs)
    top_words = top_words[:TOP_K_WORDS]

    normal_probs = normalize_probs(word_log_probs)

    # Print results
    print("\n" + "="*60)
    print(f"Top {TOP_K_WORDS} Most Probable Next Words")
    print("="*60)
    print("{:<5} {:<25} {:<15}".format("Rank", "Word", "Probability"))
    print("-" * 60)

    threshold = 0.95
    prob_sum = 0.0
    for i, ((word, prob), norm_prob) in enumerate(zip(top_words, normal_probs)):
        prob_sum += norm_prob
        if threshold and prob_sum >= threshold:
            print("--------threshold---------")
            threshold = None
        print("{:<5} {:<25} {:<15.8f}".format(i + 1, f'"{word}"', prob))
    print(f"prob_sum: {prob_sum}")

    print(f"Time: total: {t['total']:.3f}s  " \
          #f"bfs: {t['bfs']:.3f}s  " \
          f"next: {t['next']:.3f}s  forward: {t['forward']:.3f}s  " \
          f"iters: {t['iters']}s  paths: {t['paths']}s  mask: {t['mask']}s")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print((f"ImportError: {e}\n"
              "Possibly due to missing 'transformers' and 'torch' libraries."))
        print("To install them: pip install transformers torch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os
import time
from typing import Dict, List, Tuple

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-1.7B"
TOP_K_WORDS = 1000
BATCH_SIZE = 256
TYPICALITY_SIGMA = 2.0                # Asymmetric filter: keep tokens within k-sigma above entropy
MAX_TOKENS_PER_WORD = 5               # Maximum word length in tokens
MIN_LOG_PROBABILITY = math.log(1e-8)  # Prune paths with log probability below this

def is_all_ascii_alpha(word: str) -> bool:
    return word and word.isascii() and word.isalpha()

def has_leading_ascii_alpha(word: str) -> bool:
    return word and word[0].isascii() and word[0].isalpha()

def has_trailing_ascii_alpha(word: str) -> bool:
    return word and word[-1].isascii() and word[-1].isalpha()

def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

class WordProbabilityExplorer:
    def __init__(self, model, tokenizer, device, typicality_sigma: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.typicality_sigma = typicality_sigma  # Sigma threshold for asymmetric filtering

        # Track best log probability for each unique word
        self.word_log_probs: Dict[str, float] = {}

        # Cache for decoded token strings - decode each token ID only once
        self.token_text_cache: Dict[int, str] = {}

        # Statistics
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0

        # Pre-compute special token IDs for faster checking
        self.special_token_ids = set(self.tokenizer.all_special_ids)

        # Get actual vocab size from model (not tokenizer, they can differ!)
        self.vocab_size = model.config.vocab_size

        # Pre-decode and categorize all tokens for faster filtering
        print("Pre-analyzing vocabulary...")
        self.token_info = self._precompute_token_info()
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Valid first tokens: {len(self.token_info['valid_first'])}")
        print(f"  Valid continuation tokens: {len(self.token_info['valid_continuation'])}")
        print(f"  Word boundary tokens: {len(self.token_info['word_boundary'])}")

        # Move masks to device
        self.token_info['first_token_mask'] = self.token_info['first_token_mask'].to(device)
        self.token_info['continuation_mask'] = self.token_info['continuation_mask'].to(device)

    def _precompute_token_info(self) -> Dict:
        """
        Pre-decode and categorize all tokens in vocabulary for faster filtering.
        Also precompute static masks for efficient filtering.
        Returns dict with sets of token IDs for different categories and mask tensors.
        """
        valid_first = set()  # Tokens that can start a word (space + alpha)
        valid_continuation = set()  # Tokens that can continue a word (alpha, no space)

        # Use the actual model vocab size, not len(tokenizer)
        vocab_size = self.vocab_size

        for token_id in range(vocab_size):
            # Skip special tokens
            if token_id in self.special_token_ids:
                continue

            # Decode and cache
            token_text = self.tokenizer.decode([token_id])
            self.token_text_cache[token_id] = token_text

            if not token_text:
                continue

            # Check if starts with space
            if token_text[0] == ' ':
                word = token_text[1:]
                if is_all_ascii_alpha(word):
                    valid_first.add(token_id)
                    valid_continuation.add(token_id)
            # No space, contains all ascii alpha - can continue a word
            elif is_all_ascii_alpha(token_text):
                valid_continuation.add(token_id)

        # Create static additive masks with correct vocab size
        first_token_mask = torch.full((vocab_size,), float('-inf'))
        for token_id in valid_first:
            first_token_mask[token_id] = 0.0

        continuation_mask = torch.full((vocab_size,), float('-inf'))
        for token_id in valid_continuation:
            continuation_mask[token_id] = 0.0

        return {
            'valid_first': valid_first,
            'valid_continuation': valid_continuation,
            'word_boundary': valid_first, # same as first token; starts with space, all ascii/alpha
            'first_token_mask': first_token_mask,
            'continuation_mask': continuation_mask
        }        
    
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

    def typical_sampling_batch(self, log_probs: torch.Tensor, 
                              indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            indices: Corresponding token IDs of shape [batch_size, num_candidates]
            
        Returns:
            Tuple of (selected_log_probs, selected_indices, selected_mask)
            - selected_log_probs: [batch_size, max_selected]
            - selected_indices: [batch_size, max_selected]
            - selected_mask: [batch_size, max_selected] - True for valid tokens
        """
        batch_size, num_candidates = log_probs.shape
        
        # Convert to probabilities
        probs = torch.exp(log_probs)
        neg_log_probs = -log_probs
        
        # Compute entropy: H = E[-log(p)] = Σ(p_i * -log(p_i))
        entropy = torch.sum(probs * neg_log_probs, dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Compute standard deviation of surprisal around entropy
        # σ = sqrt(E[(X - μ)²]) where X = -log(p), μ = H
        variance = torch.sum(probs * (neg_log_probs - entropy)**2, dim=-1, keepdim=True)
        std_dev = torch.sqrt(variance + 1e-10)  # Small epsilon for numerical stability
        
        # Asymmetric threshold: only filter tokens that are too surprising (high -log p)
        # No lower bound: high probability tokens (low -log p) always pass
        upper_bound = entropy + self.typicality_sigma * std_dev
        typical_mask = neg_log_probs <= upper_bound  # [batch_size, num_candidates]
        
        # Sort by probability (descending) to keep highest probability tokens first
        # This ensures we always get the most probable tokens
        sorted_indices_by_prob = torch.argsort(log_probs, dim=-1, descending=True)
        
        # Apply sorting
        sorted_mask = torch.gather(typical_mask, -1, sorted_indices_by_prob)
        sorted_log_probs = torch.gather(log_probs, -1, sorted_indices_by_prob)
        sorted_token_indices = torch.gather(indices, -1, sorted_indices_by_prob)
        
        # Find max number selected across batch for uniform tensor size
        num_selected_per_item = sorted_mask.sum(dim=-1)
        max_selected = num_selected_per_item.max().item()
        max_selected = max(1, max_selected)  # Ensure at least one
        
        # Truncate to max_selected for uniform tensor shape
        selected_log_probs = sorted_log_probs[:, :max_selected]
        selected_indices = sorted_token_indices[:, :max_selected]
        selected_mask = sorted_mask[:, :max_selected]
        
        return selected_log_probs, selected_indices, selected_mask

    def get_batched_log_probs(self, input_ids_list: List[torch.Tensor], 
                              mask: torch.Tensor, num_valid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability distributions for next token for a batch of sequences.
        Only considers valid tokens (filtered by mask) when computing softmax.
        
        Args:
            input_ids_list: List of input_id tensors of shape [1, seq_len]
            mask: Additive mask tensor of shape [vocab_size] with 0.0 for valid tokens, -inf for invalid
            num_valid: Number of valid tokens (for topk)
        
        Returns:
            Tuple of (log_probs, indices):
            - log_probs: Tensor of shape [batch_size, num_valid] with log probabilities
            - indices: Tensor of shape [batch_size, num_valid] with token IDs
        """
        if not input_ids_list:
            return (torch.empty(0, num_valid, device=self.device), 
                    torch.empty(0, num_valid, dtype=torch.long, device=self.device))

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

            # Apply mask to filter out invalid tokens
            masked_logits = logits + mask  # [batch_size, vocab_size], invalid tokens are now -inf

            # Get top num_valid tokens
            top_k = torch.topk(masked_logits, k=num_valid, dim=-1)

            # Apply log_softmax ONLY to the top-k valid tokens
            top_k_log_probs = torch.log_softmax(top_k.values, dim=-1)  # [batch_size, num_valid]


        self.forward_passes += 1

        # Clean up intermediate tensors
        del batch_input_ids, batch_attention_mask, outputs, logits, masked_logits

        return top_k_log_probs, top_k.indices
    
    def find_top_words(self, context: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Find the top K most probable next words using iterative BFS with batching.
        
        Args:
            context: The input text context
            top_k: Number of top words to return
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        print(f"Context: '{context}'")

        t0_total = time.perf_counter()

        # Reset state (but keep pre-computed token info and cache)
        self.word_log_probs = {}
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0

        # Encode context
        base_input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)

        # Get log probabilities for first token
        print("Computing first token probabilities...")
        d_first_log_probs, d_first_indices = self.get_batched_log_probs(
            [base_input_ids],
            self.token_info['first_token_mask'],
            len(self.token_info['valid_first'])
        )
        
        # Apply typical sampling to dynamically select first tokens
        d_first_log_probs_selected, d_first_indices_selected, d_first_mask_selected = self.typical_sampling_batch(
            d_first_log_probs, d_first_indices
        )

        # Move to CPU
        first_values = d_first_log_probs_selected[0].cpu()
        first_indices = d_first_indices_selected[0].cpu()
        first_mask = d_first_mask_selected[0].cpu()

        # Use boolean indexing to get only valid tokens
        valid_first_values = first_values[first_mask]
        valid_first_indices = first_indices[first_mask]
        
        num_first_tokens = len(valid_first_values)
        print(f"Asymmetric typical sampling selected {num_first_tokens} initial tokens (out of {len(self.token_info['valid_first'])} valid)")
        print(f"  Keeping tokens where -log(p) <= H + {self.typicality_sigma}σ")

        # Initialize paths for first tokens
        # Each path: {'tokens': [token_ids], 'text': decoded_text, 'log_prob': float}
        # Store text incrementally to avoid repeated decoding
        current_paths = []

        for log_prob, token_id in zip(valid_first_values, valid_first_indices):
            token_id = token_id.item()
            token_log_prob = log_prob.item()

            # Should always be valid due to masking
            if token_id not in self.token_info['valid_first']:
                continue

            # Get pre-decoded text and strip the leading space
            token_text = self.decode_token(token_id).strip()

            current_paths.append({
                'tokens': [token_id],
                'text': token_text,  # Store decoded text to avoid re-decoding
                'log_prob': token_log_prob
            })

        depth = 1
        inner_loops = 0
        t0_next = time.perf_counter()
        t_forward = 0.0

        # Iterative BFS: process all paths at each depth level
        while current_paths and depth < MAX_TOKENS_PER_WORD:
            print(f"\n--- Depth {depth}: Processing {len(current_paths)} paths ---")
            print(f"    GPU Memory: {self.get_memory_usage()}")

            next_paths = []

            # Process paths in batches to avoid memory issues
            for batch_start in range(0, len(current_paths), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(current_paths))
                batch_paths = current_paths[batch_start:batch_end]

                # Prepare batch of extended inputs
                batch_input_ids = []
                for path in batch_paths:
                    tokens_tensor = torch.tensor(path['tokens'], device=self.device).unsqueeze(0)
                    extended = torch.cat([base_input_ids, tokens_tensor], dim=1)
                    batch_input_ids.append(extended)

                t0_forward = time.perf_counter()
                d_next_log_probs, d_next_indices = self.get_batched_log_probs(
                    batch_input_ids,
                    self.token_info['continuation_mask'],
                    len(self.token_info['valid_continuation'])
                )
                sync()
                t_forward += time.perf_counter() - t0_forward

                # Apply typical sampling to dynamically select continuation tokens
                d_next_log_probs_selected, d_next_indices_selected, d_next_mask_selected = self.typical_sampling_batch(
                    d_next_log_probs, d_next_indices
                )

                # Move to CPU
                next_log_probs = d_next_log_probs_selected.cpu().numpy()
                next_indices = d_next_indices_selected.cpu().numpy()
                next_mask = d_next_mask_selected.cpu().numpy()

                # Free GPU memory
                del d_next_log_probs, d_next_indices
                del d_next_log_probs_selected, d_next_indices_selected, d_next_mask_selected

                # Process each path in the batch
                for path_idx, path in enumerate(batch_paths):
                    self.paths_explored += 1

                    # Use boolean indexing to get only valid tokens for this path
                    path_mask = next_mask[path_idx]
                    path_values = next_log_probs[path_idx][path_mask]
                    path_indices = next_indices[path_idx][path_mask]
                    
                    for token_log_prob, token_id in zip(path_values, path_indices):
                        inner_loops += 1 
                        new_log_prob = path['log_prob'] + token_log_prob

                        # Prune if log probability too low
                        if new_log_prob < MIN_LOG_PROBABILITY:
                            continue

                        # Check if this is a word boundary token
                        if token_id in self.token_info['word_boundary']:
                            # This completes a word - record it
                            word = path['text'].lower()

                            # Update best log probability for this word
                            if word not in self.word_log_probs:
                                self.num_unique_words += 1
                                self.word_log_probs[word] = path['log_prob']
                            elif path['log_prob'] > self.word_log_probs[word]:
                                self.word_log_probs[word] = path['log_prob']

                            continue

                        # Continue this path - append decoded token to text
                        next_token_text = self.decode_token(token_id)

                        next_paths.append({
                            'tokens': path['tokens'] + [token_id],
                            'text': path['text'] + next_token_text,
                            'log_prob': new_log_prob
                        })

                if (batch_start // BATCH_SIZE) % 10 == 0:
                    print(f"  Processed {batch_end}/{len(current_paths)} paths, "
                          f"{self.num_unique_words} unique words found")

            # Move to next depth
            current_paths = next_paths
            depth += 1

            # Explicitly free GPU memory between iterations
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  -> {len(next_paths)} paths continue to depth {depth}")
            print(f"     GPU Memory after cleanup: {self.get_memory_usage()}")

        t1 = time.perf_counter()
        t_next = t1 - t0_next
        t_total = t1 - t0_total

        print(f"\n--- Exploration Complete ---")
        print(f"Forward passes: {self.forward_passes}")
        print(f"Paths explored: {self.paths_explored}")
        print(f"Unique words found: {self.num_unique_words}")

        # Convert log probabilities back to probabilities and sort
        word_probabilities = {
            word: math.exp(log_prob) 
            for word, log_prob in self.word_log_probs.items()
        }

        sorted_words = sorted(
            word_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_words[:top_k], {
            "total": t_total, "next": t_next, "forward": t_forward, "iters": inner_loops
        }

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Detect and set up device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # Load model
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16
        )
        
        #torch.set_float32_matmul_precision('high')
        #model = torch.compile(model, mode="max-autotune")

        model = model.to(device)
        model.eval()
        
        # Clear cache
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"Model loaded successfully on {device}")
        
        # Create explorer with asymmetric typical sampling
        print(f"Using Asymmetric Typical Sampling with sigma threshold: {TYPICALITY_SIGMA}")
        explorer = WordProbabilityExplorer(model, tokenizer, device, typicality_sigma=TYPICALITY_SIGMA)
        
        # Example context
        #CONTEXT = "volleyball"
        CONTEXT = "aquatic"
        """
            "Given an input word, respond with the most meaningful word that follows it.\n" \
            "IMPORTANT: You are to respond with only a single word.\n" \
            "## Input word:\n" \
            "aquatic"
        """
        
        # Find top words
        top_words, t = explorer.find_top_words(CONTEXT, TOP_K_WORDS)
        
        # Print results
        print("\n" + "="*60)
        print(f"Top {TOP_K_WORDS} Most Probable Next Words")
        print("="*60)
        print("{:<5} {:<25} {:<15}".format("Rank", "Word", "Probability"))
        print("-" * 60)
        
        threshold = 0.95
        prob_sum = 0.0
        for i, (word, prob) in enumerate(top_words):
            prob_sum += prob
            if threshold and prob_sum >= threshold:
                print("--------threshold---------")
                threshold = None
            print("{:<5} {:<25} {:<15.8f}".format(i + 1, f'"{word}"', prob))
        print(f"prob_sum: {prob_sum}")
        
        print(f"Time: total: {t['total']:.3f}s  next: {t['next']:.3f}s  forward: {t['forward']:.3f}s  iters: {t['iters']}s")

    except ImportError:
        print("Error: The 'transformers' and 'torch' libraries are required.")
        print("Please install them using: pip install transformers torch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# claude.12 or so: memory management

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os
from typing import Dict, List, Tuple

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TOP_K_WORDS = 1000

# Exploration limits to prevent exponential explosion
BATCH_SIZE = 128
MAX_INITIAL_TOKENS = 100        # How many first tokens to explore
MAX_CONTINUATIONS_PER_TOKEN = 100  # How many next tokens to try from each position
MAX_TOKENS_PER_WORD = 6         # Maximum word length in tokens
MIN_LOG_PROBABILITY = math.log(1e-8)  # Prune paths with log probability below this

def has_leading_ascii_alpha(word: str) -> bool:
    return word and word[0].isascii() and word[0].isalpha()

class WordProbabilityExplorer:
    """
    Explores all possible word completions using iterative breadth-first search
    with batched log probability calculations for efficiency.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
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
        
        # Pre-decode and categorize all tokens for faster filtering
        print("Pre-analyzing vocabulary...")
        self.token_info = self._precompute_token_info()
        print(f"  Valid first tokens: {len(self.token_info['valid_first'])}")
        print(f"  Valid continuation tokens: {len(self.token_info['valid_continuation'])}")
        print(f"  Word boundary tokens: {len(self.token_info['word_boundary'])}")
    
    def _precompute_token_info(self) -> Dict:
        """
        Pre-decode and categorize all tokens in vocabulary for faster filtering.
        Returns dict with sets of token IDs for different categories.
        """
        valid_first = set()  # Tokens that can start a word (space + alpha)
        valid_continuation = set()  # Tokens that can continue a word (alpha, no space)
        word_boundary = set()  # Tokens that indicate word boundary (space + alpha)
        
        vocab_size = len(self.tokenizer)
        
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
                stripped = token_text.strip()
                if has_leading_ascii_alpha(stripped):
                    valid_first.add(token_id)
                    word_boundary.add(token_id)
            # No space, starts with alpha - can continue a word
            elif has_leading_ascii_alpha(token_text):
                valid_continuation.add(token_id)
        
        return {
            'valid_first': valid_first,
            'valid_continuation': valid_continuation,
            'word_boundary': word_boundary
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
    
    def get_batched_log_probs(self, input_ids_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Get log probability distributions for next token for a batch of sequences.
        
        Args:
            input_ids_list: List of input_id tensors of shape [1, seq_len]
            
        Returns:
            Tensor of shape [batch_size, vocab_size] with log probabilities
        """
        if not input_ids_list:
            return torch.empty(0, len(self.tokenizer), device=self.device)
        
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
                mask = torch.cat([
                    torch.zeros(1, max_len - seq_len, dtype=torch.long, device=self.device),
                    torch.ones(1, seq_len, dtype=torch.long, device=self.device)
                ], dim=1)
            else:
                padded = input_ids
                mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)
            
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
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
            log_probs = torch.log_softmax(logits, dim=-1)
        
        self.forward_passes += 1
        
        # Clean up intermediate tensors
        del batch_input_ids, batch_attention_mask, outputs, logits
        
        return log_probs
    
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
        
        # Reset state (but keep pre-computed token info and cache)
        self.word_log_probs = {}
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0
        
        # Encode context
        base_input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Get log probabilities for first token
        print("Computing first token probabilities...")
        first_token_log_probs = self.get_batched_log_probs([base_input_ids])[0]
        
        # Get top N initial tokens, but only from valid first tokens
        # Create a masked version that only includes valid first tokens
        first_token_mask = torch.full_like(first_token_log_probs, float('-inf'))
        for token_id in self.token_info['valid_first']:
            first_token_mask[token_id] = first_token_log_probs[token_id]
        
        top_initial = torch.topk(
            first_token_mask, 
            min(MAX_INITIAL_TOKENS, len(self.token_info['valid_first']))
        )
        
        print(f"Starting exploration with {len(top_initial.values)} initial tokens...")
        
        # Initialize paths for first tokens
        # Each path: {'tokens': [token_ids], 'text': decoded_text, 'log_prob': float}
        # Store text incrementally to avoid repeated decoding
        current_paths = []
        
        for log_prob, token_id in zip(top_initial.values, top_initial.indices):
            token_id = token_id.item()
            token_log_prob = log_prob.item()
            
            # Should always be valid due to masking, but double-check
            if token_id not in self.token_info['valid_first']:
                continue
            
            # Get pre-decoded text and strip the leading space
            token_text = self.decode_token(token_id).strip()
            
            current_paths.append({
                'tokens': [token_id],
                'text': token_text,  # Store decoded text to avoid re-decoding
                'log_prob': token_log_prob
            })
        
        # Iterative BFS: process all paths at each depth level
        depth = 1
        
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
                
                # Get log probs for entire batch in one forward pass
                batch_log_probs = self.get_batched_log_probs(batch_input_ids)
                
                # Move to CPU immediately to free GPU memory
                # We only need this for comparison/sorting, not for further computation
                batch_log_probs_cpu = batch_log_probs.cpu()
                del batch_log_probs  # Explicitly delete GPU tensor
                
                # Process each path in the batch
                for path_idx, path in enumerate(batch_paths):
                    self.paths_explored += 1
                    
                    log_probs = batch_log_probs_cpu[path_idx]
                    
                    # Get top continuations for this path
                    top_k_next = torch.topk(
                        log_probs, 
                        min(MAX_CONTINUATIONS_PER_TOKEN, len(log_probs))
                    )
                    
                    for token_log_prob, token_id in zip(top_k_next.values, top_k_next.indices):
                        token_id = token_id.item()
                        token_log_prob = token_log_prob.item()
                        
                        new_log_prob = path['log_prob'] + token_log_prob
                        
                        # Prune if log probability too low
                        if new_log_prob < MIN_LOG_PROBABILITY:
                            continue
                        
                        # Check if this is a word boundary token
                        if token_id in self.token_info['word_boundary']:
                            # This completes a word - record it
                            word = path['text']  # Already stripped, no decoding needed!
                            
                            # Update best log probability for this word
                            if word not in self.word_log_probs:
                                self.num_unique_words += 1
                                self.word_log_probs[word] = path['log_prob']
                            elif path['log_prob'] > self.word_log_probs[word]:
                                self.word_log_probs[word] = path['log_prob']
                            
                            continue
                        
                        # Check if this is a valid continuation token
                        if token_id not in self.token_info['valid_continuation']:
                            continue
                        
                        # Continue this path - append decoded token to text
                        next_token_text = self.decode_token(token_id)
                        
                        next_paths.append({
                            'tokens': path['tokens'] + [token_id],
                            'text': path['text'] + next_token_text,  # Incremental build!
                            'log_prob': new_log_prob
                        })
                
                if (batch_start // BATCH_SIZE) % 10 == 0:
                    print(f"  Processed {batch_end}/{len(current_paths)} paths, "
                          f"{self.num_unique_words} unique words found")
                
                # Clean up batch tensors explicitly
                del batch_input_ids, batch_log_probs_cpu
            
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
        
        return sorted_words[:top_k]


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
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()
        
        # Clear cache
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"Model loaded successfully on {device}")
        
        # Create explorer
        explorer = WordProbabilityExplorer(model, tokenizer, device)
        
        # Example context
        CONTEXT = "" \
            "Given an input word, respond with the most meaningful word that follows it.\n" \
            "IMPORTANT: You are to respond with only a single word.\n" \
            "## Input word:\n" \
            "aquatic"
        
        # Find top words
        top_words = explorer.find_top_words(CONTEXT, TOP_K_WORDS)
        
        # Print results
        print("\n" + "="*60)
        print(f"Top {TOP_K_WORDS} Most Probable Next Words")
        print("="*60)
        print("{:<5} {:<25} {:<15}".format("Rank", "Word", "Probability"))
        print("-" * 60)
        
        for i, (word, prob) in enumerate(top_words):
            print("{:<5} {:<25} {:<15.8f}".format(i + 1, f'"{word}"', prob))
        
    except ImportError:
        print("Error: The 'transformers' and 'torch' libraries are required.")
        print("Please install them using: pip install transformers torch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

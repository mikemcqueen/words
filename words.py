# optimized/bugfixed 7

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os
from typing import Dict, List, Tuple
from collections import defaultdict

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TOP_K_WORDS = 50

# Exploration limits to prevent exponential explosion
MAX_INITIAL_TOKENS = 50        # How many first tokens to explore
MAX_CONTINUATIONS_PER_TOKEN = 50  # How many next tokens to try from each position
MAX_TOKENS_PER_WORD = 5         # Maximum word length in tokens
MIN_LOG_PROBABILITY = math.log(1e-8)  # Prune paths with log probability below this
MAX_CACHE_SIZE = 10000          # Maximum cache entries before clearing

def has_leading_ascii_alpha(word: str) -> bool:
    """Check if first character is alphabetic and word is non-empty."""
    return word and word[0].isascii() and word[0].isalpha()


class WordProbabilityExplorer:
    """
    Explores all possible word completions by recursively traversing
    the token vocabulary tree using log probabilities for numerical stability.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Track best log probability for each unique word
        self.word_log_probs: Dict[str, float] = {}
        
        # Cache for token log probabilities to avoid redundant forward passes
        self.log_prob_cache: Dict[tuple, torch.Tensor] = {}
        
        # Cache for decoded token strings
        self.token_text_cache: Dict[int, str] = {}
        
        # Statistics
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0
        
        # Pre-compute special token IDs for faster checking
        self.special_token_ids = set(self.tokenizer.all_special_ids)
    
    def get_token_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get log probability distribution for next token.
        Uses caching to avoid redundant forward passes.
        Returns log probabilities for numerical stability.
        """
        # More efficient cache key creation
        cache_key = tuple(input_ids[0].cpu().numpy())
        
        if cache_key in self.log_prob_cache:
            return self.log_prob_cache[cache_key]
        
        # Cache size management to prevent OOM
        if len(self.log_prob_cache) > MAX_CACHE_SIZE:
            print(f"  Cache limit reached ({MAX_CACHE_SIZE}), clearing cache...")
            self.log_prob_cache.clear()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            # Use log_softmax directly - no need to exp then log
            log_probs = torch.log_softmax(logits, dim=-1)
        
        self.forward_passes += 1
        self.log_prob_cache[cache_key] = log_probs
        
        return log_probs
    
    def decode_token(self, token_id: int) -> str:
        """
        Decode a single token to string with caching.
        """
        if token_id not in self.token_text_cache:
            self.token_text_cache[token_id] = self.tokenizer.decode([token_id])
        return self.token_text_cache[token_id]
    
    def is_word_boundary(self, token_text: str) -> bool:
        """
        More robust word boundary detection.
        Checks for space, newline, or other common delimiters.
        """
        if not token_text:
            return False
        return token_text[0] == ' ' #in (' ', '\n', '\t', '\r')
    
    def explore_continuations(
        self, 
        input_ids: torch.Tensor,
        current_tokens: List[int],
        current_log_prob: float,
        depth: int
    ):
        """
        Recursively explore token continuations to find complete words.
        Uses log probabilities to avoid numerical underflow.
        
        Args:
            input_ids: Context (original prompt)
            current_tokens: Tokens accumulated so far for this word
            current_log_prob: Sum of log probabilities so far
            depth: Current token depth (to enforce MAX_TOKENS_PER_WORD)
        """
        self.paths_explored += 1
        
        # Prune if log probability too low
        if current_log_prob < MIN_LOG_PROBABILITY:
            return
        
        # Prune if word is too long
        if depth >= MAX_TOKENS_PER_WORD:
            return
        
        # Decode current tokens once
        current_text = self.tokenizer.decode(current_tokens, skip_special_tokens=True)
        
        # More efficient tensor creation
        current_tokens_tensor = torch.tensor(current_tokens, device=self.device).unsqueeze(0)
        extended_input = torch.cat([input_ids, current_tokens_tensor], dim=1)
        
        # Get log probabilities for next token
        next_log_probs = self.get_token_log_probs(extended_input)
        
        # Get top K next tokens to explore
        top_k = torch.topk(next_log_probs, min(MAX_CONTINUATIONS_PER_TOKEN, len(next_log_probs)))
        
        for log_prob, token_id in zip(top_k.values, top_k.indices):
            token_id = token_id.item()
            
            # Skip special tokens (EOS, BOS, PAD, etc.)
            if token_id in self.special_token_ids:
                continue
            
            token_log_prob = log_prob.item()
            new_log_prob = current_log_prob + token_log_prob
            
            # Skip if log probability too low
            if new_log_prob < MIN_LOG_PROBABILITY:
                continue
            
            # Decode this token using cache
            next_token_text = self.decode_token(token_id)
            
            if not has_leading_ascii_alpha(next_token_text):
                # TODO: word = strip_non_ascii_alph(current_text)
                word = current_text.strip()
                # Update best log probability for current word
                if word not in self.word_log_probs:
                    self.num_unique_words += 1
                    self.word_log_probs[word] = current_log_prob
                elif current_log_prob > self.word_log_probs[word]:
                    self.word_log_probs[word] = current_log_prob
                continue
            
            # Word is not complete - continue exploring
            new_tokens = current_tokens + [token_id]
            
            self.explore_continuations(
                input_ids,
                new_tokens,
                new_log_prob,
                depth + 1
            )
    
    def find_top_words(self, context: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Find the top K most probable next words.
        
        Args:
            context: The input text context
            top_k: Number of top words to return
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        print(f"Context: '{context}'")
        
        # Reset state
        self.word_log_probs = {}
        self.log_prob_cache = {}
        self.token_text_cache = {}
        self.forward_passes = 0
        self.paths_explored = 0
        self.num_unique_words = 0
        
        # Encode context
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Get log probabilities for first token
        first_token_log_probs = self.get_token_log_probs(input_ids)
        
        # Get top N initial tokens to explore
        top_initial = torch.topk(
            first_token_log_probs, 
            min(1000, len(first_token_log_probs))
        )
        
        print(f"Exploring {len(top_initial.values)} initial tokens...")
        
        num_valid_initial_tokens = 0

        # Explore each initial token
        for i, (log_prob, token_id) in enumerate(zip(top_initial.values, top_initial.indices)):
            token_id = token_id.item()
            
            # Skip special tokens
            if token_id in self.special_token_ids:
                continue
            
            # Decode this token using cache
            token_text = self.decode_token(token_id)
            token_text = token_text.strip()
                
            if not has_leading_ascii_alpha(token_text):
                print(f"  Skipping '{token_text}'")
                continue

            num_valid_initial_tokens += 1

            print(f"  Token {num_valid_initial_tokens}/{MAX_INITIAL_TOKENS} '{token_text}' - "
                  f"{self.num_unique_words} words found")
            
            token_log_prob = log_prob.item()

            # Check if this single token is already a complete word
            if 0:
                if word not in self.word_log_probs:
                    self.num_unique_words += 1
                    self.word_log_probs[word] = token_log_prob
                elif token_log_prob > self.word_log_probs[word]:
                    self.word_log_probs[word] = token_log_prob
            else:
                # Explore continuations of this token
                self.explore_continuations(
                    input_ids,
                    [token_id],
                    token_log_prob,
                    depth=1
                )

            if num_valid_initial_tokens == MAX_INITIAL_TOKENS:
                break
        
        print(f"\n--- Exploration Complete ---")
        print(f"Forward passes: {self.forward_passes}")
        print(f"Paths explored: {self.paths_explored}")
        print(f"Unique words found: {self.num_unique_words}")
        print(f"Cache entries: {len(self.log_prob_cache)}")
        
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
        CONTEXT = "Given a word, respond with the most meaningful word that follows it.\n" \
            "## Here's the word:\n" \
            "aquatic"

        """
            "## Example 1\n"  \
            "**Input**: banana\n" \
            "**Output**: peel\n" \
            "Input: roman\n" \
            "Output: legion\n" \
            "Input: coffee\n" \
            "Output: cup\n" \
        """
            
        
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

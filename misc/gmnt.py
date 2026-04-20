import numpy as np

def normalize_gnmt(log_prob: float, length: int, alpha: float = 0.7) -> float:
    """
    Applies the GNMT length normalization to the sequence score.

    The score is calculated as:
    Score_GNMT(Y) = Sum(log P(y_i | y_<i, Context)) / ((5 + N)^alpha / (5 + 1)^alpha)

    Args:
        log_prob: The raw cumulative log probability (a negative number).
        length: The number of tokens (N) in the sequence Y.
        alpha: The length normalization exponent (hyperparameter, typically 0.6 to 1.0).

    Returns:
        The length-normalized score (a less negative number, higher is better).
    """
    
    # Check for zero length to prevent division by zero, though unlikely in beam search
    if length == 0:
        return -np.inf 

    # The denominator is the 'Length Penalty' term from the GNMT paper
    # (5 + N)^alpha / (5 + 1)^alpha
    # We use (5 + 1)^alpha (which is 6^alpha) as the base since N >= 1
    
    denom = ((5.0 + length) ** alpha) / (6.0 ** alpha)
    
    # Divide the raw score (log_prob) by the denom
    normalized_score = log_prob / denom
    
    return normalized_score

# --- Example Usage ---
# Context: "cat"
# Goal: Find the best next word (sequence of tokens ending in a space)
alpha = 0.7 

# 1. Short, high probability word ("in")
score_in = -1.5     # Raw log probability
length_in = 2       # Example: ' in' + ' '
normalized_in = normalize_gnmt(score_in, length_in, alpha)

# 2. Longer, lower probability word ("collaborate")
score_collab = -4.0 # Raw log probability
length_collab = 5   # Example: ' col' + 'lab' + 'or' + 'ate' + ' '
normalized_collab = normalize_gnmt(score_collab, length_collab, alpha)


print(f"--- Alpha: {alpha} ---")
print(f"Raw Score 'in' (Length 2): {score_in}")
print(f"Normalized Score 'in': {normalized_in:.4f}")
print("-" * 20)
print(f"Raw Score 'collaborate' (Length 5): {score_collab}")
print(f"Normalized Score 'collaborate': {normalized_collab:.4f}")

# Now, compare the normalized scores:
if normalized_collab > normalized_in:
    print(f"\nResult: 'collaborate' is the preferred word (Normalized Score: {normalized_collab:.4f} > {normalized_in:.4f})")
else:
    print(f"\nResult: 'in' is the preferred word (Normalized Score: {normalized_in:.4f} > {normalized_collab:.4f})")

from heapq import nlargest
from typing import Callable, Sequence


def select_valid_token(
    logits: Sequence[float],
    is_valid: Callable[[int], bool],
    top_k: int = 128
        ) -> int:
    """Find the highest-scoring valid token with fallback."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    token_ids = range(len(logits))

    top_ids = nlargest(
        top_k,
        token_ids,
        key=lambda token_id: logits[token_id]
    )

    for token_id in top_ids:
        if is_valid(token_id):
            return token_id

    ranked_ids = sorted(
        token_ids,
        key=lambda token_id: logits[token_id],
        reverse=True
    )

    for token_id in ranked_ids:
        if is_valid(token_id):
            return token_id

    raise ValueError("No valid next token found")

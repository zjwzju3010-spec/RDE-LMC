import os
import sys
import pickle
from typing import List, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rank_bm25 import BM25Okapi
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_bm25_index(corpus: List[dict], cache_path: str) -> BM25Okapi:
    """
    Build BM25 index from article corpus and save to disk.

    Args:
        corpus: List of article dicts with 'content' field
        cache_path: Path to save pickled BM25 index

    Returns:
        BM25Okapi instance
    """
    import jieba

    logger.info(f"Building BM25 index for {len(corpus)} articles...")

    tokenized_corpus = []
    for art in corpus:
        tokens = list(jieba.cut(art["content"]))
        tokens = [t for t in tokens if t.strip()]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(bm25, f)

    logger.info(f"BM25 index saved to {cache_path}")
    return bm25


def load_bm25_index(cache_path: str) -> BM25Okapi:
    """Load BM25 index from disk."""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def bm25_search(
    bm25: BM25Okapi,
    article_ids: List[str],
    query_tokens: List[str],
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Search using BM25.

    Args:
        bm25: BM25Okapi instance
        article_ids: Ordered list of article IDs (same order as BM25 corpus)
        query_tokens: Tokenized query
        top_k: Number of top results to return

    Returns:
        List of (article_id, score) sorted by score descending
    """
    if not query_tokens:
        return []

    import numpy as np
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if idx < len(article_ids) and scores[idx] > 0:
            results.append((article_ids[idx], float(scores[idx])))

    return results

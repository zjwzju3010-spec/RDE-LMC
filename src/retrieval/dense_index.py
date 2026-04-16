import os
import sys
import numpy as np
from typing import List, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# BGE-M3 query instruction for retrieval tasks
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def build_dense_index(embeddings: np.ndarray):
    """
    Build FAISS index from embeddings.
    Uses IndexFlatIP (inner product = cosine when vectors are normalized).

    Args:
        embeddings: float32 numpy array [N, dim], already normalized

    Returns:
        faiss.Index
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Ensure float32 and normalized
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def dense_search(
    index,
    article_ids: List[str],
    query_embedding: np.ndarray,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Search using FAISS dense index.

    Args:
        index: faiss.Index
        article_ids: Ordered list of article IDs
        query_embedding: float32 array [dim]
        top_k: Number of top results to return

    Returns:
        List of (article_id, score) sorted by score descending
    """
    import faiss

    # Normalize and reshape query
    query = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(article_ids):
            results.append((article_ids[idx], float(score)))

    return results


def encode_query(model, query: str) -> np.ndarray:
    """
    Encode a query using the embedding model.

    Qwen3-Embedding is an asymmetric model:
    - Documents in KB were encoded WITHOUT instruction (plain text)
    - Queries should be encoded WITH prompt_name="query" to use Qwen3's
      built-in task instruction, which improves retrieval quality.

    This follows the Qwen3-Embedding README recommendation:
      query_embeddings = model.encode(queries, prompt_name="query")
      document_embeddings = model.encode(documents)  # no instruction

    Args:
        model: SentenceTransformer model (Qwen3-Embedding)
        query: Query string

    Returns:
        Normalized float32 array [dim]
    """
    embedding = model.encode(
        query,
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding.astype(np.float32)

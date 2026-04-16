import os
import sys
from typing import List, Dict, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.retrieval.keyword_extractor import tokenize
from src.retrieval.bm25_index import bm25_search
from src.retrieval.dense_index import dense_search, encode_query
from src.retrieval.law_name_retriever import retrieve_by_law_name

logger = get_logger(__name__)


def rrf_fusion(
    ranked_list_1: List[Tuple[str, float]],
    ranked_list_2: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    RRF score(d) = sum over lists: 1 / (k + rank(d))
    where rank is 1-indexed.

    Args:
        ranked_list_1, ranked_list_2: Lists of (article_id, score), sorted descending
        k: RRF constant (default 60)

    Returns:
        Merged list of (article_id, rrf_score) sorted by rrf_score descending
    """
    scores = {}

    for rank, (article_id, _) in enumerate(ranked_list_1, start=1):
        scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + rank)

    for rank, (article_id, _) in enumerate(ranked_list_2, start=1):
        scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, kb: dict, config: dict) -> List[Dict]:
    """
    Main EntityBR retrieval function.
    Combines BM25 and dense retrieval with RRF fusion.

    Args:
        query: User query string
        kb: Knowledge base dict (from build_kb.load_kb)
        config: Configuration dict

    Returns:
        List of article dicts (top-k), sorted by RRF score
    """
    retrieval_cfg = config.get("retrieval", {})
    bm25_top_k = retrieval_cfg.get("bm25_top_k", 20)
    dense_top_k = retrieval_cfg.get("dense_top_k", 20)
    hybrid_top_k = retrieval_cfg.get("hybrid_top_k", 10)
    rrf_k = retrieval_cfg.get("rrf_k", 60)

    # Tokenize query for BM25 using the same tokenizer as the corpus
    # (plain jieba.cut with stopword filtering matches the build_kb tokenization)
    query_tokens = tokenize(query)
    logger.debug(f"Query tokens: {query_tokens[:10]}")

    # BM25 retrieval
    bm25_results = []
    if kb.get("bm25") is not None and kb.get("article_ids"):
        try:
            bm25_results = bm25_search(
                kb["bm25"],
                kb["article_ids"],
                query_tokens,
                top_k=bm25_top_k
            )
            logger.debug(f"BM25 retrieved {len(bm25_results)} candidates")
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")

    # Dense retrieval
    dense_results = []
    if kb.get("embedding_model") is not None and kb.get("dense_index") is not None:
        try:
            query_emb = encode_query(kb["embedding_model"], query)
            dense_results = dense_search(
                kb["dense_index"],
                kb["article_ids"],
                query_emb,
                top_k=dense_top_k
            )
            logger.debug(f"Dense retrieved {len(dense_results)} candidates")
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")

    # RRF fusion
    if bm25_results and dense_results:
        fused = rrf_fusion(bm25_results, dense_results, k=rrf_k)
    elif bm25_results:
        fused = bm25_results
    elif dense_results:
        fused = dense_results
    else:
        logger.warning("No retrieval results!")
        return []

    # Take top-k and return full article dicts
    top_ids = [aid for aid, _ in fused[:hybrid_top_k]]

    results = []
    article_index = kb.get("article_index", {})
    for article_id in top_ids:
        if article_id in article_index:
            results.append(article_index[article_id])

    # Layer 2: Law-name keyword direct lookup.
    # Matches extracted keywords against article_id strings, bypassing BM25
    # tokenization issues for specialized laws (烟叶税法, 耕地占用税法, etc.).
    seen_ids = {a["article_id"] for a in results}
    law_name_articles = retrieve_by_law_name(query, kb, top_k=20)
    for art in law_name_articles:
        if art["article_id"] not in seen_ids:
            results.append(art)
            seen_ids.add(art["article_id"])

    logger.info(f"Retrieved {len(results)} articles for query: {query[:50]}...")
    return results

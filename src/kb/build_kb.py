import os
import sys
import json
import pickle
import numpy as np
import argparse

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.kb.law_parser import parse_all_laws
from src.kb.schema import LegalArticle, LegalElement

logger = get_logger(__name__)


def build_kb(config: dict) -> None:
    """
    Build knowledge base from raw law files.
    Steps:
    1. Parse all law files -> corpus.json (all articles, no LLM extraction)
    2. Build FAISS embeddings -> embeddings.npy + article_ids.json
    3. Build BM25 index -> bm25_cache.pkl
    LLM extraction is done lazily during pipeline execution.
    """
    paths = config["paths"]
    project_root = _project_root

    # Resolve paths relative to project root
    raw_laws_dir = os.path.join(project_root, paths["raw_laws_dir"])
    processed_kb_dir = os.path.join(project_root, paths["processed_kb_dir"])

    os.makedirs(processed_kb_dir, exist_ok=True)

    # Step 1: Parse all law files
    corpus_path = os.path.join(project_root, paths["corpus_json"])
    article_ids_path = os.path.join(project_root, paths["article_ids_json"])

    logger.info("Step 1: Parsing law files...")
    articles = parse_all_laws(raw_laws_dir)

    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved corpus with {len(articles)} articles to {corpus_path}")

    article_ids = [a["article_id"] for a in articles]
    with open(article_ids_path, 'w', encoding='utf-8') as f:
        json.dump(article_ids, f, ensure_ascii=False, indent=2)

    # Step 2: Build BM25 index
    bm25_path = os.path.join(project_root, paths["bm25_cache"])
    _build_bm25(articles, bm25_path)

    # Step 3: Build embeddings
    embeddings_path = os.path.join(project_root, paths["embeddings_npy"])
    model_name = config["model"]["embedding"]
    _build_embeddings(articles, model_name, embeddings_path)

    logger.info("KB build complete!")


def _build_bm25(articles: list, cache_path: str) -> None:
    """Build and save BM25 index."""
    logger.info("Building BM25 index...")
    try:
        from rank_bm25 import BM25Okapi
        import jieba

        # Tokenize all article contents
        tokenized_corpus = []
        for art in articles:
            tokens = list(jieba.cut(art["content"]))
            tokens = [t for t in tokens if t.strip() and len(t) > 0]
            tokenized_corpus.append(tokens)

        bm25 = BM25Okapi(tokenized_corpus)

        with open(cache_path, 'wb') as f:
            pickle.dump(bm25, f)

        logger.info(f"BM25 index saved to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        raise


def _build_embeddings(articles: list, model_name: str, embeddings_path: str) -> None:
    """Build and save dense embeddings using SentenceTransformer."""
    logger.info(f"Building embeddings with model {model_name}...")
    try:
        from sentence_transformers import SentenceTransformer

        # Use CPU to avoid GPU OOM issues (the GPU may be occupied by other processes)
        model = SentenceTransformer(model_name, device='cpu')

        contents = [art["content"] for art in articles]

        # Encode in batches
        logger.info(f"Encoding {len(contents)} articles...")
        embeddings = model.encode(
            contents,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(np.float32)
        np.save(embeddings_path, embeddings)
        logger.info(f"Embeddings saved: shape {embeddings.shape} to {embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to build embeddings: {e}")
        raise


def load_kb(config: dict) -> dict:
    """
    Load all KB artifacts from disk into memory.
    Returns dict with: corpus, article_index, bm25, dense_index, article_ids, embedding_model
    """
    import pickle
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    project_root = _project_root
    paths = config["paths"]

    logger.info("Loading KB from disk...")

    # Load corpus
    corpus_path = os.path.join(project_root, paths["corpus_json"])
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    logger.info(f"Loaded {len(corpus)} articles")

    # Build article index
    article_index = {art["article_id"]: art for art in corpus}

    # Derive article_ids from corpus (always aligned with BM25 and embeddings,
    # which are both built from corpus.json in the same positional order).
    # NOTE: We do NOT load from article_ids.json because that file may have been
    # manually edited and could be misaligned (different number of entries).
    article_ids = [art["article_id"] for art in corpus]
    logger.info(f"Derived {len(article_ids)} article_ids from corpus")

    # Load BM25
    bm25_path = os.path.join(project_root, paths["bm25_cache"])
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    logger.info("BM25 index loaded")

    # Load embeddings and build FAISS index
    embeddings_path = os.path.join(project_root, paths["embeddings_npy"])
    embeddings = np.load(embeddings_path)

    dim = embeddings.shape[1]
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(embeddings)
    logger.info(f"FAISS index built: {dense_index.ntotal} vectors, dim={dim}")

    # Load embedding model
    logger.info(f"Loading embedding model {config['model']['embedding']}...")
    embedding_model = SentenceTransformer(config['model']['embedding'])

    logger.info("KB loaded successfully!")

    return {
        "corpus": corpus,
        "article_index": article_index,
        "bm25": bm25,
        "dense_index": dense_index,
        "article_ids": article_ids,
        "embedding_model": embedding_model,
    }


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Build RDE-LMC Knowledge Base")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config_path = os.path.join(_project_root, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    build_kb(config)

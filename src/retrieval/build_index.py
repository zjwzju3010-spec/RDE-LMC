"""
Build retrieval indices (BM25 + Dense/FAISS).

This is a convenience script mentioned in the Readme Step 2.
The actual index building is integrated into build_kb.py, but
this script provides a standalone way to rebuild indices without
re-parsing the law corpus.

Usage:
    python src/retrieval/build_index.py [--config config/config.yaml]
"""
import os
import sys
import argparse
import json
import pickle
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_bm25_index(config: dict) -> None:
    """Build BM25 index from existing corpus.json."""
    import jieba
    from rank_bm25 import BM25Okapi

    project_root = _project_root
    paths = config["paths"]

    corpus_path = os.path.join(project_root, paths["corpus_json"])
    bm25_path = os.path.join(project_root, paths["bm25_cache"])

    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    logger.info(f"Tokenizing {len(corpus)} articles...")
    tokenized_corpus = []
    for art in corpus:
        tokens = list(jieba.cut(art["content"]))
        tokens = [t for t in tokens if t.strip()]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)

    logger.info(f"BM25 index saved to {bm25_path}")


def build_dense_index(config: dict) -> None:
    """Build dense embedding index from existing corpus.json."""
    from sentence_transformers import SentenceTransformer

    project_root = _project_root
    paths = config["paths"]
    model_name = config["model"]["embedding"]

    corpus_path = os.path.join(project_root, paths["corpus_json"])
    embeddings_path = os.path.join(project_root, paths["embeddings_npy"])
    article_ids_path = os.path.join(project_root, paths["article_ids_json"])

    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    logger.info(f"Loading embedding model {model_name} on CPU...")
    model = SentenceTransformer(model_name, device='cpu')

    contents = [art["content"] for art in corpus]
    article_ids = [art["article_id"] for art in corpus]

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
    logger.info(f"Embeddings saved: shape {embeddings.shape}")

    with open(article_ids_path, 'w', encoding='utf-8') as f:
        json.dump(article_ids, f, ensure_ascii=False, indent=2)
    logger.info(f"Article IDs saved to {article_ids_path}")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices (BM25 + Dense)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--bm25-only", action="store_true", help="Only rebuild BM25 index")
    parser.add_argument("--dense-only", action="store_true", help="Only rebuild dense index")
    args = parser.parse_args()

    import yaml
    config_path = os.path.join(_project_root, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.dense_only:
        build_dense_index(config)
    elif args.bm25_only:
        build_bm25_index(config)
    else:
        logger.info("Building BM25 index...")
        build_bm25_index(config)
        logger.info("Building dense index...")
        build_dense_index(config)

    logger.info("All indices built!")


if __name__ == "__main__":
    main()

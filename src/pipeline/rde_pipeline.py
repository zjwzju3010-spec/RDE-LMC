"""
RDE-LMC Pipeline (rde_pipeline.py)
Strictly follows the structure described in the Readme:
  retrieval → discrimination → execution → dynamic loop
  run_case(query) -> (y, A)
"""
import os
import sys
import yaml
from typing import List, Tuple, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RDEPipeline:
    """
    RDE-LMC Pipeline:
    Retrieval (EntityBR) → Discrimination (LogicAD) → Execution (SymbolCE) → Dynamic loop (DynamicCM)
    """

    def __init__(self, config: dict, kb: dict):
        self.config = config
        self.kb = kb
        logger.info("RDE-LMC Pipeline initialized")

    def run_case(self, query: str) -> Tuple[List[float], List[str]]:
        """
        Run the complete pipeline for a single query.

        Args:
            query: Legal question in Chinese

        Returns:
            Tuple (y, A):
                y: List of numerical answers (float)
                A: List of applicable article IDs
        """
        from src.dynamic.controller import run_dynamic_loop

        logger.info(f"Processing: {query[:80]}...")

        result = run_dynamic_loop(
            initial_query=query,
            kb=self.kb,
            config=self.config
        )

        y = result.get("numerical_answer", [])
        A = result.get("article_answer", [])

        return y, A

    def run_batch(self, queries: List[str]) -> List[Dict]:
        """
        Run pipeline on a list of queries.

        Returns:
            List of {"numerical_answer": y, "article_answer": A} dicts
        """
        results = []
        try:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Processing queries")
        except ImportError:
            iterator = queries

        for query in iterator:
            y, A = self.run_case(query)
            results.append({"numerical_answer": y, "article_answer": A})

        return results


def load_kb():
    """Load KB using default config."""
    config_path = os.path.join(_project_root, "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from src.kb.build_kb import load_kb as _load_kb
    return _load_kb(config), config


def run_pipeline(query: str, config: dict = None, kb: dict = None) -> Dict:
    """
    Convenience function matching Readme main.py pattern:
        load_kb() → run_pipeline() → evaluate()

    Returns:
        {"numerical_answer": [...], "article_answer": [...]}
    """
    if config is None or kb is None:
        kb, config = load_kb()

    pipeline = RDEPipeline(config, kb)
    y, A = pipeline.run_case(query)
    return {"numerical_answer": y, "article_answer": A}


def load_pipeline(config_path: str = None) -> RDEPipeline:
    """
    Load the full pipeline: config + KB.

    Args:
        config_path: Path to config.yaml

    Returns:
        Initialized RDEPipeline
    """
    from src.kb.build_kb import load_kb as _load_kb

    if config_path is None:
        config_path = os.path.join(_project_root, "config", "config.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.join(_project_root, config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    kb = _load_kb(config)
    return RDEPipeline(config, kb)

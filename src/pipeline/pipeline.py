"""
RDE-LMC Pipeline - main entry point for the complete system.
"""
import os
import sys
import yaml
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.dynamic.controller import run_dynamic_loop

logger = get_logger(__name__)


class RDEPipeline:
    """
    RDE-LMC Pipeline: Retrieval, Discrimination, Execution for Legal Mathematical Calculation.
    """

    def __init__(self, config: dict, kb: dict):
        self.config = config
        self.kb = kb
        logger.info("RDE-LMC Pipeline initialized")

    def run_case(self, query: str) -> Dict:
        """
        Run the complete pipeline for a single query.

        Args:
            query: Legal question in Chinese

        Returns:
            Dict with: numerical_answer (List[float]), article_answer (List[str])
        """
        logger.info(f"Processing query: {query[:80]}...")

        result = run_dynamic_loop(
            initial_query=query,
            kb=self.kb,
            config=self.config
        )

        return {
            "numerical_answer": result.get("numerical_answer", []),
            "article_answer": result.get("article_answer", []),
        }

    def run_batch(self, queries: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Run pipeline on a list of queries.

        Args:
            queries: List of query strings
            show_progress: Show progress bar

        Returns:
            List of result dicts
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="Processing queries")
            except ImportError:
                iterator = queries
        else:
            iterator = queries

        for query in iterator:
            result = self.run_case(query)
            results.append(result)

        return results


def load_pipeline(config_path: str = None) -> RDEPipeline:
    """
    Load the full pipeline: config + KB + model.

    Args:
        config_path: Path to config.yaml (absolute or relative to project root)

    Returns:
        Initialized RDEPipeline
    """
    from src.kb.build_kb import load_kb

    if config_path is None:
        config_path = os.path.join(_project_root, "config", "config.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.join(_project_root, config_path)

    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    kb = load_kb(config)

    return RDEPipeline(config, kb)

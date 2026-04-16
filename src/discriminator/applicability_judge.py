import os
import sys
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.utils.llm_api import call_llm
from src.discriminator.logic_prompt import build_applicability_prompt, build_relaxed_applicability_prompt

logger = get_logger(__name__)


def judge(article: Dict, case_facts: str, config: dict) -> Dict:
    """
    Judge whether a single article is applicable to the case facts using LLM.

    Args:
        article: Article dict with article_id, content, etc.
        case_facts: The case description/query
        config: Configuration dict

    Returns:
        Dict with: article_id, is_applicable, confidence, judgement, reasoning, missing_info
    """
    article_id = article.get("article_id", "")

    if not article.get("content"):
        return {
            "article_id": article_id,
            "is_applicable": False,
            "confidence": 0.0,
            "judgement": "NOT_APPLICABLE",
            "reasoning": [],
            "missing_info": []
        }

    prompt = build_applicability_prompt(article, case_facts)

    try:
        result = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=True
        )

        # Ensure required fields
        result.setdefault("article_id", article_id)
        result.setdefault("is_applicable", False)
        result.setdefault("confidence", 0.5 if result.get("is_applicable") else 0.0)
        result.setdefault("judgement", "APPLICABLE" if result.get("is_applicable") else "NOT_APPLICABLE")
        result.setdefault("reasoning", [])
        result.setdefault("missing_info", [])
        result["article_id"] = article_id  # Always set from our side

        logger.debug(f"Article {article_id}: applicable={result['is_applicable']}, confidence={result['confidence']}")
        return result

    except Exception as e:
        logger.warning(f"Failed to judge article {article_id}: {e}")
        return {
            "article_id": article_id,
            "is_applicable": False,
            "confidence": 0.0,
            "judgement": "NOT_APPLICABLE",
            "reasoning": [],
            "missing_info": [str(e)]
        }


def judge_all(articles: List[Dict], case_facts: str, config: dict) -> List[Dict]:
    """
    Judge all retrieved articles in parallel.
    Returns only applicable articles, sorted by confidence descending.

    Args:
        articles: List of article dicts
        case_facts: The case description/query
        config: Configuration dict

    Returns:
        List of applicable article judgment dicts, sorted by confidence
    """
    if not articles:
        return []

    logger.info(f"Judging {len(articles)} articles for applicability...")

    results = []

    # Use ThreadPoolExecutor for parallel LLM calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article = {
            executor.submit(judge, article, case_facts, config): article
            for article in articles
        }

        for future in as_completed(future_to_article):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                article = future_to_article[future]
                logger.error(f"Unexpected error judging {article.get('article_id', '?')}: {e}")

    # Filter applicable ones and sort by confidence
    applicable = [r for r in results if r.get("is_applicable", False)]
    applicable.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    logger.info(f"Found {len(applicable)} applicable articles out of {len(articles)}")
    return applicable


def judge_relaxed(article: Dict, case_facts: str, config: dict) -> Dict:
    """Judge with relaxed criteria (2 conditions instead of 3), for emergency fallback."""
    article_id = article.get("article_id", "")
    if not article.get("content"):
        return {"article_id": article_id, "is_applicable": False, "confidence": 0.0,
                "judgement": "NOT_APPLICABLE", "reasoning": [], "missing_info": []}

    prompt = build_relaxed_applicability_prompt(article, case_facts)
    try:
        result = call_llm(prompt, model=config.get("model", {}).get("llm", "gpt-4o"), expect_json=True)
        result.setdefault("article_id", article_id)
        result.setdefault("is_applicable", False)
        result.setdefault("confidence", 0.5 if result.get("is_applicable") else 0.0)
        result.setdefault("judgement", "APPLICABLE" if result.get("is_applicable") else "NOT_APPLICABLE")
        result.setdefault("reasoning", [])
        result.setdefault("missing_info", [])
        result["article_id"] = article_id
        return result
    except Exception as e:
        logger.warning(f"Relaxed judge failed for {article_id}: {e}")
        return {"article_id": article_id, "is_applicable": False, "confidence": 0.0,
                "judgement": "NOT_APPLICABLE", "reasoning": [], "missing_info": [str(e)]}


def judge_all_relaxed(articles: List[Dict], case_facts: str, config: dict) -> List[Dict]:
    """
    Judge articles with relaxed 2-criteria standard (emergency fallback).
    Used when all regular iterations fail to find applicable articles.
    """
    if not articles:
        return []

    logger.info(f"Relaxed judging {len(articles)} articles (emergency fallback)...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article = {
            executor.submit(judge_relaxed, article, case_facts, config): article
            for article in articles
        }
        for future in as_completed(future_to_article):
            try:
                results.append(future.result())
            except Exception as e:
                article = future_to_article[future]
                logger.error(f"Relaxed judge error {article.get('article_id', '?')}: {e}")

    applicable = [r for r in results if r.get("is_applicable", False)]
    applicable.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    logger.info(f"Relaxed judge: {len(applicable)} applicable out of {len(articles)}")
    return applicable

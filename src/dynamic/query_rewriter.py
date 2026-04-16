"""
Query rewriter for the DynamicCM module.
Generates targeted retrieval queries for missing information.
"""
import os
import sys
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.utils.llm_api import call_llm

logger = get_logger(__name__)

REWRITE_PROMPT = """在解答以下法律数值计算问题时，我们发现缺少一些关键信息。

## 原始问题
{original_query}

## 已找到的适用法条
{article_ids}

## 缺失的关键信息
{missing_info}

## 任务
请生成一个新的检索查询，用于查找能补充上述缺失信息的法律条文。
新查询应该：
1. 聚焦于缺失的概念或计算参数
2. 使用法律专业术语
3. 简短精准（10-30个字）
4. 不重复原始问题的内容

只输出新的查询文本，不要有任何其他内容："""


def rewrite(
    original_query: str,
    missing_info: List[Dict],
    applied_article_ids: List[str],
    config: dict
) -> str:
    """
    Generate a new retrieval query targeting missing information.

    Args:
        original_query: The original user query
        missing_info: List of missing variable dicts from deficiency_detector
        applied_article_ids: Already applied article IDs
        config: Configuration dict

    Returns:
        New query string
    """
    if not missing_info:
        return original_query

    # Format missing info
    missing_desc = "\n".join([f"- {m.get('description', str(m))}" for m in missing_info])
    article_ids_str = "\n".join([f"- {aid}" for aid in applied_article_ids])

    prompt = REWRITE_PROMPT.format(
        original_query=original_query,
        article_ids=article_ids_str if article_ids_str else "（无）",
        missing_info=missing_desc
    )

    try:
        new_query = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=False
        )
        new_query = new_query.strip()

        logger.info(f"Rewrote query: {new_query}")
        return new_query

    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}, using original")
        return original_query

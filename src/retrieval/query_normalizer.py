"""
Pre-retrieval query normalization.

Converts colloquial Chinese legal queries to formal legal vocabulary
so that BM25/dense retrieval can find the correct legal articles.

Problem: Users write "忘记按时交税，拖欠20天" but legal texts say
"纳税人未按照规定期限缴纳税款". This vocabulary gap causes BM25 and
dense retrieval to fail (BM25 rank 7693, dense rank 264).

Solution: LLM rewrites the colloquial query using formal legal terms
before retrieval. The original query is kept for the LLM judge stage.
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.utils.llm_api import call_llm

logger = get_logger(__name__)

NORMALIZE_PROMPT = """将以下法律问题改写为使用正式法律术语的检索查询，以便在法律条文数据库中检索到相关法条。

原始问题：{query}

改写要求：
1. 将口语化表达替换为法律条文中常见的正式术语
   例：交税→缴纳税款，拖欠→未按期缴纳，开除→解除劳动合同，赔钱→赔偿损失
2. 识别法律领域关键词（如：滞纳金、违约金、赔偿金、罚款、处罚等）
3. 保留原始问题中的数值、时间、比例等关键信息
4. 改写后的查询应类似于法律条文中的表述方式
5. 只输出改写后的查询文本，不超过60字，不要任何其他说明

改写后的查询："""


def normalize_query(query: str, config: dict) -> str:
    """
    Normalize a colloquial legal query to formal legal vocabulary.

    Args:
        query: Original user query (may contain colloquial language)
        config: Configuration dict

    Returns:
        Normalized query using formal legal terminology.
        Falls back to original query if LLM call fails.
    """
    try:
        prompt = NORMALIZE_PROMPT.format(query=query)
        normalized = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=False
        )
        normalized = normalized.strip()

        # Sanity check: if the result is too short or too long, keep original
        if len(normalized) < 5 or len(normalized) > 200:
            logger.warning(f"Normalization result suspicious (len={len(normalized)}), using original")
            return query

        logger.info(f"Query normalized: [{query[:50]}] → [{normalized}]")
        return normalized

    except Exception as e:
        logger.warning(f"Query normalization failed: {e}, using original query")
        return query

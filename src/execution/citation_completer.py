"""
Citation Completer — post-execution legal citation completion.
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

_CITATION_COMPLETION_PROMPT = """法律引用完整性检查：本案已通过以下法条计算出数值答案。请极度保守地判断是否需要补充其他法条。

## 案件事实
{case_facts}

## 已引用法条（直接用于计算的）
{primary_articles_text}

## 候选补充法条
{candidate_articles_text}

## 判断标准（必须同时满足才补充）
**仅在以下情形补充**，且补充总数不超过2条：
1. **强依赖定义**：已引用法条明确引用了该候选条款（如"依照本法第X条规定的纳税人"），缺少该条款则已引用法条无法理解
2. **计算要素分离**：计算公式的关键变量（税率、基数、倍数）分散在多个条款，已引用法条中缺失的要素恰好在候选条款中明确规定，且本案事实中用到了该要素

**绝对不补充**：
- 仅是背景介绍、立法目的、适用范围的条款
- 程序性、处罚性、征管性条款
- 与本案计算结果无直接关联的条款
- 候选条款内容与已引用法条高度重叠的
- 不确定是否需要时：**不补充**

## 默认倾向
大多数案件（约70%）只需1条法条即可完整引用。补充前请反问：去掉该候选条款，计算结果是否会改变或法律依据是否不完整？若不影响，则不补充。

请输出JSON：
{{
  "additional_article_ids": ["需要补充的法条ID，通常为空列表或1条，极少超过2条"],
  "reasoning": "一句话说明是否补充及原因"
}}"""


def complete_citations(
    primary_article_ids: List[str],
    case_facts: str,
    candidate_articles: List[Dict],
    article_index: Dict,
    config: dict
) -> List[str]:
    """
    Given primary articles (used for computation) and candidate articles
    (retrieved but not selected), ask the LLM which candidates should be
    added as co-citations for a complete legal basis.

    Args:
        primary_article_ids: Article IDs already in the answer
        case_facts: Original case query
        candidate_articles: Articles to consider adding (dicts with article_id, content)
        article_index: KB article index for content lookup
        config: Configuration dict

    Returns:
        List of additional article IDs to include (may be empty)
    """
    if not candidate_articles:
        return []

    # Build primary articles text
    primary_texts = []
    for aid in primary_article_ids:
        art = article_index.get(aid, {})
        content = art.get("content", "")[:300]
        primary_texts.append(f"- {aid}：{content}")
    primary_articles_text = "\n".join(primary_texts)

    # Build candidate articles text (limit to 15 candidates to keep prompt manageable)
    candidate_texts = []
    for art in candidate_articles[:15]:
        aid = art.get("article_id", "")
        content = art.get("content", "")[:200]
        candidate_texts.append(f"- {aid}：{content}")
    candidate_articles_text = "\n".join(candidate_texts)

    prompt = _CITATION_COMPLETION_PROMPT.format(
        case_facts=case_facts,
        primary_articles_text=primary_articles_text,
        candidate_articles_text=candidate_articles_text,
    )

    try:
        result = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=True
        )
        additional = result.get("additional_article_ids", [])
        if isinstance(additional, str):
            additional = [additional]

        # Validate: only return IDs that actually exist in candidates; hard cap at 2
        candidate_ids = {a.get("article_id", "") for a in candidate_articles}
        valid = [aid for aid in additional if aid in candidate_ids][:2]

        logger.debug(f"Citation completion: +{len(valid)} articles ({valid})")
        return valid

    except Exception as e:
        logger.warning(f"Citation completion failed: {e}")
        return []

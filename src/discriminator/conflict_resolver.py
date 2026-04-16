import os
import sys
import re
from typing import List, Dict, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.utils.llm_api import call_llm
from src.discriminator.logic_prompt import build_conflict_resolution_prompt

logger = get_logger(__name__)

# ── Rule 1: 特别法优先 (special law > general law) ─────────────────────────
# Indicators that a law is a special/specific implementation
SPECIAL_LAW_INDICATORS = [
    '实施条例', '实施细则', '解释', '若干规定', '若干问题', '若干意见',
    '补充规定', '特别规定', '暂行规定', '暂行办法', '管理办法',
    '规定', '办法', '条例', '规则', '细则', '补充', '专项', '暂行'
]

# Indicators of a general/framework law (lower specificity)
GENERAL_LAW_INDICATORS = ['基本法', '通则', '总则', '准则']

# ── Rule 2: 上位法优先 (higher-level law > lower-level law) ──────────────────
# Law hierarchy levels (higher value = higher level = higher priority)
HIERARCHY_LEVELS = {
    '宪法': 10,
    '基本法': 9,
    '法典': 8,     # 民法典 etc.
    '法': 5,        # 中华人民共和国XXX法
    '条例': 3,      # 行政法规
    '规定': 2,      # 部门规章
    '办法': 2,
    '规则': 1,
    '细则': 1,
    '解释': 1,      # judicial interpretation
}

# ── Rule 3: 新法优先 (newer law > older law) ─────────────────────────────────
# Parse year from law name or content
_YEAR_PATTERN = re.compile(r'(19|20)\d{2}')


def _get_year(article: Dict) -> int:
    """Extract year from law name or content. Returns 0 if not found."""
    text = article.get("law_name", "") + " " + article.get("content", "")
    matches = _YEAR_PATTERN.findall(text)
    if matches:
        return max(int(y) for y in matches)
    return 0


def _get_hierarchy_level(law_name: str) -> int:
    """
    Estimate the legal hierarchy level (Rule 3: 上位法优先).
    Higher = more authoritative.
    """
    # Check for 宪法 / 基本法
    if '宪法' in law_name:
        return HIERARCHY_LEVELS['宪法']
    if '基本法' in law_name:
        return HIERARCHY_LEVELS['基本法']

    # 民法典 / 刑法典
    if '法典' in law_name:
        return HIERARCHY_LEVELS['法典']

    # Check for implementing regulations (lower level despite containing '法')
    for indicator in ['实施条例', '实施细则', '暂行条例', '若干规定', '管理办法', '办法', '规则', '细则']:
        if indicator in law_name:
            return 2

    # 最高人民法院/最高人民检察院 interpretations
    if '最高人民法院' in law_name or '最高人民检察院' in law_name:
        return 3

    # National laws: 中华人民共和国XXX法
    if '中华人民共和国' in law_name and law_name.rstrip().endswith('法'):
        return HIERARCHY_LEVELS['法']

    # Other laws ending in 法
    if law_name.rstrip().endswith('法'):
        return 4

    # Administrative regulations (条例)
    if '条例' in law_name:
        return HIERARCHY_LEVELS['条例']

    # Other regulations
    for key in ['规定', '办法']:
        if key in law_name:
            return HIERARCHY_LEVELS[key]

    return 1


def _get_specificity_score(law_name: str) -> int:
    """
    Estimate how specific a law is (Rule 1: 特别法优先).
    Higher = more specific = should take priority over general laws.
    """
    score = 0
    for indicator in SPECIAL_LAW_INDICATORS:
        if indicator in law_name:
            score += 1
    for indicator in GENERAL_LAW_INDICATORS:
        if indicator in law_name:
            score -= 1
    return score


def _rule_based_priority(articles: List[Dict]) -> Optional[List[Dict]]:
    """
    Apply the three conflict resolution rules in order:
      1. 特别法优先 (special law > general law)
      2. 新法优先 (new law > old law)
      3. 上位法优先 (higher-level law > lower-level)

    Returns sorted list if deterministic (clear winner), else None (ambiguous → LLM).
    """
    if len(articles) <= 1:
        return articles

    def sort_key(art):
        name = art.get("law_name", "")
        specificity = _get_specificity_score(name)
        year = _get_year(art)
        hierarchy = _get_hierarchy_level(name)
        return (specificity, year, hierarchy)

    scored = [(art, sort_key(art)) for art in articles]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_score = scored[0][1]
    runner_score = scored[1][1]

    # Clear winner only if the top score differs on any dimension from second
    if best_score != runner_score:
        return [art for art, _ in scored]

    # Tied → return None so LLM resolves
    return None


def resolve(
    applicable_articles: List[Dict],
    case_facts: str,
    config: dict,
    article_index: Dict = None
) -> List[Dict]:
    """
    Resolve conflicts between multiple applicable articles.

    Rules applied in order:
      1. 特别法优先 – special/specific law overrides general law
      2. 新法优先   – newer law overrides older law
      3. 上位法优先 – higher-level law (宪法>法律>条例>规章) has authority
    Falls back to LLM when rules are ambiguous.

    Args:
        applicable_articles: List of applicable article judgment dicts
        case_facts: The case description/query
        config: Configuration dict
        article_index: Optional dict mapping article_id -> full article dict

    Returns:
        Ordered list of full article dicts: [primary, ...secondary]
    """
    if not applicable_articles:
        return []

    # Fetch full article dicts
    article_dicts = []
    for judgment in applicable_articles:
        article_id = judgment.get("article_id", "")
        if article_index and article_id in article_index:
            art = dict(article_index[article_id])
        else:
            art = {
                "article_id": article_id,
                "law_name": judgment.get("law_name", ""),
                "content": judgment.get("content", ""),
            }
        article_dicts.append(art)

    if len(article_dicts) == 1:
        return article_dicts

    logger.info(f"Resolving conflicts among {len(article_dicts)} applicable articles")

    # Always use LLM for conflict resolution.
    # Rule-based specificity scoring (特别法优先) is unreliable when judged solely from
    # law names — e.g., a "条例" about a different topic should NOT rank higher than a
    # directly-applicable "法". LLM understands content relevance.
    try:
        prompt = build_conflict_resolution_prompt(article_dicts, case_facts)
        result = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=True
        )

        primary_id = result.get("primary_article_id", "")
        secondary_ids = result.get("secondary_article_ids", [])
        all_ids = result.get("all_applicable", [primary_id] + secondary_ids)

        id_to_art = {art["article_id"]: art for art in article_dicts}
        ordered = []
        for aid in all_ids:
            if aid in id_to_art:
                ordered.append(id_to_art[aid])
        # Append any not mentioned
        for art in article_dicts:
            if art["article_id"] not in {a["article_id"] for a in ordered}:
                ordered.append(art)

        logger.info(f"LLM-based resolution: primary={primary_id}")
        return ordered

    except Exception as e:
        logger.warning(f"Conflict resolution LLM call failed: {e}, keeping original order")
        return article_dicts

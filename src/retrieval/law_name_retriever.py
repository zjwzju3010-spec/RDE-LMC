"""
Law-Name Direct Lookup Retriever.

Bypasses BM25 tokenization mismatches by matching extracted keywords directly
against article_id strings (which contain the law name).
Effective for specialized laws like 烟叶税法, 耕地占用税法, 城市维护建设税法, etc.
"""
import os
import sys
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Generic legal terms that appear in many laws — not useful for discrimination
_STOPWORDS = {
    '应当', '规定', '法律', '条款', '处罚', '违法', '行为', '机关', '本案', '情形',
    '依据', '人民', '中华', '共和国', '法院', '人民法院', '最高', '管理', '办法',
    '条例', '实施', '暂行', '关于', '若干', '问题', '解释', '规则', '细则',
    '适用', '处理', '认定', '计算', '方法', '标准', '金额', '数额', '赔偿',
    '违反', '依照', '按照', '给予', '处以', '不得', '必须', '可以',
    '企业', '单位', '个人', '当事人', '申请', '申报', '缴纳', '税款', '税率',
}


def retrieve_by_law_name(query: str, kb: dict, top_k: int = 20) -> List[Dict]:
    """
    Extract domain-specific keywords from the query and match them against
    article_id strings to directly retrieve articles from the relevant law.

    This is especially effective when:
    - The query explicitly mentions a law domain (e.g., "烟叶税", "耕地占用税")
    - BM25 tokenization mismatches cause the right articles to fall outside top-k

    Args:
        query: The user query (or normalized query)
        kb: Knowledge base dict with 'corpus' key
        top_k: Max articles to return

    Returns:
        List of article dicts matching the extracted keywords
    """
    try:
        import jieba.analyse
    except ImportError:
        logger.warning("jieba not available, skipping law-name retrieval")
        return []

    keywords = jieba.analyse.extract_tags(query, topK=10, withWeight=False)
    domain_keywords = [kw for kw in keywords if len(kw) >= 2 and kw not in _STOPWORDS]

    if not domain_keywords:
        return []

    corpus = kb.get('corpus', [])
    seen_ids = set()
    matched = []

    for kw in domain_keywords:
        for art in corpus:
            aid = art['article_id']
            if kw in aid and aid not in seen_ids:
                matched.append(art)
                seen_ids.add(aid)

    result = matched[:top_k]
    if result:
        logger.debug(f"Law-name retrieval: {len(result)} articles for keywords={domain_keywords[:5]}")
    return result

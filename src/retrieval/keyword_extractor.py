import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jieba
import jieba.analyse
from typing import List, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Legal stopwords - common Chinese words to exclude from retrieval
LEGAL_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '她', '他', '它', '们', '什么', '如果', '因为', '所以',
    '但是', '虽然', '然而', '应该', '可以', '可能', '已经', '正在', '关于',
    '对于', '根据', '按照', '依据', '通过', '由于', '因此', '从而', '此外',
    '其中', '其他', '以及', '还有', '同时', '另外', '并且', '而且', '或者',
    '以上', '以下', '以内', '之内', '之外', '之间', '请问', '我是', '我有',
    '需要', '想知道', '想了解', '帮我', '帮助', '告诉', '解释', '说明',
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize Chinese text using jieba, remove stopwords.
    Used for BM25 indexing and query expansion.
    """
    tokens = jieba.cut(text, cut_all=False)
    result = []
    for t in tokens:
        t = t.strip()
        if t and t not in LEGAL_STOPWORDS and len(t) >= 1:
            result.append(t)
    return result


def extract_keywords(query: str) -> Dict[str, List[str]]:
    """
    Extract search keywords from a Chinese legal query.

    Returns:
    {
        "concepts": [...],   # legal concept terms
        "facts": [...],      # factual keywords (numbers, entities)
        "all": [...]         # combined deduped list for BM25 query
    }
    """
    # TF-IDF based keyword extraction
    tfidf_keywords = jieba.analyse.extract_tags(query, topK=20, withWeight=False)

    # Full segmentation
    all_tokens = tokenize(query)

    # Legal concept patterns - terms that indicate legal concepts
    concept_patterns = [
        '违法', '违规', '处罚', '罚款', '赔偿', '补偿', '赔偿金', '补助',
        '税款', '税率', '税额', '缴纳', '滞纳金', '利息', '费用', '金额',
        '行为', '责任', '义务', '权利', '许可', '禁止', '限制', '要求',
        '合同', '劳动', '工伤', '保险', '社会保险', '养老', '医疗', '失业',
        '经济补偿', '赔偿金', '津贴', '补贴', '奖励', '惩罚', '制裁',
        '罚则', '法律责任', '行政处罚', '刑事责任', '民事责任',
    ]

    # Numeric/factual patterns
    import re
    numeric_pattern = re.compile(r'\d+[\.\d]*[万千百元%‰倍年月日天周]?')

    concepts = []
    facts = []

    # Classify TF-IDF keywords
    for kw in tfidf_keywords:
        if any(cp in kw or kw in cp for cp in concept_patterns):
            concepts.append(kw)
        elif numeric_pattern.search(kw):
            facts.append(kw)
        else:
            # Default: add to concepts if looks like a legal term
            if len(kw) >= 2:
                concepts.append(kw)

    # Extract numbers and entities as facts
    numbers = numeric_pattern.findall(query)
    facts.extend(numbers)

    # Deduplicate
    concepts = list(dict.fromkeys(concepts))
    facts = list(dict.fromkeys(facts))

    # Combined list: merge tfidf keywords + all tokens (deduped)
    all_kw = list(dict.fromkeys(tfidf_keywords + all_tokens))

    return {
        "concepts": concepts,
        "facts": facts,
        "all": all_kw
    }

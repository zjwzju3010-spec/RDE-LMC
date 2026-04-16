import os
import re
from typing import List, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_article_id(law_name: str, article_num: str) -> str:
    """Returns canonical article ID: 《{law_name}》第{article_num}条"""
    # Strip brackets if already present
    law_name = law_name.strip('《》')
    return f"《{law_name}》第{article_num}条"


def _detect_format(first_line: str) -> str:
    """Detect law file format: 'A' or 'B'"""
    stripped = first_line.strip()
    if stripped.startswith('《'):
        return 'A'
    return 'B'


def _parse_format_a(lines: List[str], filepath: str) -> List[Dict]:
    """
    Format A: 《法律名》第X条规定，content
    Pattern: ^《(.+?)》第(.+?)条[规定，,]?(.*)$
    """
    results = []
    # Match: 《law_name》第article_num条 possibly followed by 规定，or 规定, or just ，
    pattern = re.compile(r'^《(.+?)》第(.+?)条(?:规定[，,]?)?(.*)$')

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            law_name = m.group(1).strip()
            article_num = m.group(2).strip()
            content = line  # Keep full line as content
            article_id = build_article_id(law_name, article_num)
            results.append({
                "article_id": article_id,
                "law_name": law_name,
                "article_num": article_num,
                "content": content,
                "metadata": {"source_file": os.path.basename(filepath), "format": "A", "line": line_no}
            })
        else:
            logger.debug(f"Format A: Skipped line {line_no} in {os.path.basename(filepath)}: {line[:50]}")

    return results


def _parse_format_b(lines: List[str], filepath: str) -> List[Dict]:
    """
    Format B: law_name 第X条　content (全角空格 \\u3000 after article number)
    Also handles: law_name 第X条 content (regular space)
    """
    results = []
    # Match: law_name<whitespace>第article_num条<fullwidth_space_or_regular_space>content
    pattern = re.compile(r'^(.+?)\s+第(.+?)条[\u3000\s](.*)$')
    # Fallback: law_name 第X条 (no content on same line, or content follows directly)
    pattern2 = re.compile(r'^(.+?)\s+第(.+?)条(.*)$')

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line) or pattern2.match(line)
        if m:
            law_name = m.group(1).strip()
            article_num = m.group(2).strip()
            content = line
            article_id = build_article_id(law_name, article_num)
            results.append({
                "article_id": article_id,
                "law_name": law_name,
                "article_num": article_num,
                "content": content,
                "metadata": {"source_file": os.path.basename(filepath), "format": "B", "line": line_no}
            })
        else:
            logger.debug(f"Format B: Skipped line {line_no} in {os.path.basename(filepath)}: {line[:50]}")

    return results


def parse_law_file(filepath: str) -> List[Dict]:
    """
    Parse a single law .txt file.
    Returns list of article dicts: {article_id, law_name, article_num, content, metadata}
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return []

    if not lines:
        return []

    # Find first non-empty line for format detection
    first_line = next((l.strip() for l in lines if l.strip()), "")
    fmt = _detect_format(first_line)

    if fmt == 'A':
        return _parse_format_a(lines, filepath)
    else:
        return _parse_format_b(lines, filepath)


def parse_all_laws(raw_laws_dir: str) -> List[Dict]:
    """
    Parse all .txt law files in the directory.
    Returns deduplicated list of article dicts ordered by file/line.
    """
    all_articles = []
    seen_ids = set()

    txt_files = [f for f in os.listdir(raw_laws_dir) if f.endswith('.txt')]
    txt_files.sort()

    logger.info(f"Parsing {len(txt_files)} law files from {raw_laws_dir}")

    for fname in txt_files:
        fpath = os.path.join(raw_laws_dir, fname)
        articles = parse_law_file(fpath)

        added = 0
        for art in articles:
            if art["article_id"] not in seen_ids:
                seen_ids.add(art["article_id"])
                all_articles.append(art)
                added += 1

        logger.debug(f"  {fname}: {added} articles")

    logger.info(f"Total articles parsed: {len(all_articles)}")
    return all_articles

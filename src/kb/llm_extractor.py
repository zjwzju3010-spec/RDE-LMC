import os
import json
import hashlib
from typing import Optional
from src.utils.logger import get_logger
from src.utils.llm_api import call_llm

logger = get_logger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """你是一个专业的法律条文结构化分析专家。请分析以下法律条文，提取其结构化信息。

法条文本：
{article_text}

请严格按照以下JSON格式输出（不要添加任何其他内容）：
{{
  "concept": "本条文规范的核心法律概念（简短描述，如：营利性治沙违规罚款）",
  "elements": [
    {{
      "id": "e1",
      "cond": "适用条件描述（从法条中提取）",
      "desc": "简短说明"
    }}
  ],
  "parameters": {{
    "参数名（中文）": {{
      "value": null,
      "min": null,
      "max": null,
      "unit": "单位",
      "description": "参数说明"
    }}
  }},
  "formula": "Python可执行的计算表达式，用变量名代替数值（如 fine_per_hectare_max * area_hectares）。若无数值计算则为空字符串",
  "formula_variables": {{
    "变量名（英文下划线）": "变量含义（中文）"
  }},
  "references": []
}}

重要说明：
1. formula必须是合法Python算术表达式，可包含 +,-,*,/,//,**,min(),max(),条件表达式(x if cond else y)
2. 所有变量名使用英文小写加下划线
3. 如涉及范围（如"不低于X不高于Y"），以最大值作为默认计算
4. 若法条不涉及数值计算，formula为空字符串""
5. 只输出JSON，不要任何其他文字"""


def _article_id_to_cache_path(article_id: str, cache_dir: str) -> str:
    """Convert article_id to a safe cache file path using MD5 hash."""
    md5 = hashlib.md5(article_id.encode('utf-8')).hexdigest()
    prefix = md5[:2]
    return os.path.join(cache_dir, prefix, md5 + ".json")


def _load_from_cache(article_id: str, cache_dir: str) -> Optional[dict]:
    path = _article_id_to_cache_path(article_id, cache_dir)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_to_cache(article_id: str, result: dict, cache_dir: str) -> None:
    path = _article_id_to_cache_path(article_id, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def extract_structure(article_text: str, article_id: str, cache_dir: str) -> dict:
    """
    Extract structured information from a law article using LLM.
    Results are cached by article_id MD5 hash.

    Returns dict with: concept, elements, parameters, formula, formula_variables, references
    """
    # Check cache first
    cached = _load_from_cache(article_id, cache_dir)
    if cached is not None:
        logger.debug(f"Cache hit for {article_id}")
        return cached

    logger.info(f"Extracting structure for {article_id}")

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(article_text=article_text)

    try:
        result = call_llm(prompt, expect_json=True)

        # Validate and fill defaults
        result.setdefault("concept", "")
        result.setdefault("elements", [])
        result.setdefault("parameters", {})
        result.setdefault("formula", "")
        result.setdefault("formula_variables", {})
        result.setdefault("references", [])

        # Cache the result
        _save_to_cache(article_id, result, cache_dir)
        return result

    except Exception as e:
        logger.warning(f"Failed to extract structure for {article_id}: {e}")
        # Return minimal structure on failure
        fallback = {
            "concept": "",
            "elements": [],
            "parameters": {},
            "formula": "",
            "formula_variables": {},
            "references": []
        }
        return fallback

"""
Intermediate Representation (IR) generator for the SymbolCE module.
Uses LLM to extract variable bindings and computation expressions from law articles.
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

IR_GENERATION_PROMPT = """你是一个法律数值计算专家。请根据法律条文和案件事实，提取计算变量并生成计算表达式。

## 案件事实
{case_facts}

## 适用法条
法条编号：{article_id}
法条内容：{article_content}

## 任务
1. 从案件事实中提取所有计算所需的变量值
2. 生成**唯一一个**与本案问题最相关的Python计算表达式（使用实际数值）

请严格按照以下JSON格式输出：
{{
  "variable_bindings": {{
    "变量名（英文下划线）": {{
      "value": 数值（已知时填写，未知时为null）,
      "source": "从案件中提取的原文依据",
      "unit": "单位（元/天/公顷等）"
    }}
  }},
  "expression": "直接使用数值的Python计算表达式（如 50000 * 1 或 10000 * 20 * 0.0005）",
  "missing_variables": ["无法从案件中确定的信息描述"],
  "intermediate_steps": [
    "步骤1：...",
    "步骤2：..."
  ],
  "expected_result_unit": "结果单位（元）",
  "result_description": "结果含义说明"
}}

重要规则：
1. expression中只能使用 +,-,*,/,//,**,min(),max(),abs(),round() 和数字
2. expression使用实际数值，不使用变量名
3. **只生成一个expression**——本案问题的核心数值答案。法条若有多个计算规则（如上限/下限、不同情形），只计算与本案最直接相关的那一条规则
4. 若无法确定某个值，在missing_variables中说明，expression可留空""
5. 遇到"每日万分之五"，转换为 * 0.0005（万分之五 = 5/10000）
6. 遇到百分比如5%，转换为 * 0.05
7. 严格输出JSON，不要任何其他说明
8. **年限/月数取整规则（重要）**：
   法条规定"六个月以上不满一年的，按一年计算；不满六个月的，按半年（0.5年）计算"时，严格按此规则先取整再乘以月薪：
   - 整年数 + 6个月 → 整年数 + 1年（例：7年6个月 → 8年，expression: 8 * 月薪）
   - 整年数 + 7/8/9/10/11个月 → 整年数 + 1年（例：7年7个月 → 8年）
   - 整年数 + 1/2/3/4/5个月 → 整年数 + 0.5年（例：7年3个月 → 7.5年）
   - 恰好整年 → 直接用整年数（例：7年0个月 → 7年）
   **绝对不能**直接将 7.5年 代入公式，必须先按上述规则取整！
   - "不足X个月的，按X个月计算" → 向上取整到X个月
   - "满X年不足X+1年，按X年" → 向下取整
   - 计算工作年限等时，必须先应用上述规则再代入公式
9. **计算上下限**：法条若规定"不超过X"或"最高为X"，expression中用 min(计算值, 上限值)；规定"不低于X"则用 max(计算值, 下限值)"""


MULTI_ANSWER_PROMPT = """你是一个法律数值计算专家。该法条可能涉及多个数值计算结果。

## 案件事实
{case_facts}

## 适用法条
法条编号：{article_id}
法条内容：{article_content}

## 任务
请识别所有需要计算的数值项目，并分别给出计算表达式。

请严格按照以下JSON格式输出：
{{
  "calculations": [
    {{
      "name": "计算项目名称（如一次性伤残补助金��",
      "expression": "Python计算表达式（直接用数值）",
      "result_description": "结果说明",
      "missing_variables": []
    }}
  ],
  "all_missing": ["所有无法确定的信息"]
}}

规则同上，expression只使用数值和 +,-,*,/,min,max,abs,round"""


def generate_ir(article: Dict, case_facts: str, config: dict) -> Dict:
    """
    Generate Intermediate Representation from an applicable article and case facts.

    Args:
        article: Article dict with article_id, content, etc.
        case_facts: The case description/query
        config: Configuration dict

    Returns:
        IR dict with: variable_bindings, expression, missing_variables, intermediate_steps
    """
    article_id = article.get("article_id", "")
    article_content = article.get("content", "")

    logger.info(f"Generating IR for {article_id}")

    prompt = IR_GENERATION_PROMPT.format(
        case_facts=case_facts,
        article_id=article_id,
        article_content=article_content
    )

    try:
        result = call_llm(
            prompt,
            model=config.get("model", {}).get("llm", "gpt-4o"),
            expect_json=True
        )

        result.setdefault("article_id", article_id)
        result.setdefault("variable_bindings", {})
        result.setdefault("expression", "")
        result.setdefault("missing_variables", [])
        result.setdefault("intermediate_steps", [])
        result.setdefault("expected_result_unit", "元")
        result.setdefault("result_description", "")
        result["article_id"] = article_id

        logger.debug(f"IR generated: expression={result['expression']}, missing={result['missing_variables']}")
        return result

    except Exception as e:
        logger.warning(f"Failed to generate IR for {article_id}: {e}")
        return {
            "article_id": article_id,
            "variable_bindings": {},
            "expression": "",
            "missing_variables": [f"IR generation failed: {str(e)}"],
            "intermediate_steps": [],
            "expected_result_unit": "元",
            "result_description": ""
        }


def generate_ir_multi(articles: List[Dict], case_facts: str, config: dict) -> List[Dict]:
    """
    Generate IR for multiple applicable articles.
    For cases that require multiple articles (e.g., base rate + multiplier).

    Returns list of IR dicts, one per article.
    """
    results = []
    for article in articles:
        ir = generate_ir(article, case_facts, config)
        results.append(ir)
    return results

"""
Prompt templates for the LogicAD (Logic Applicability Discriminator) module.
"""
from typing import List, Dict

APPLICABILITY_PROMPT_TEMPLATE = """你是一位严谨的法律专家，需要判断某法律条文是否适用于给定案件，并能用于计算本案所求的数值。

## 案件事实
{case_facts}

## 待判断法条
法条编号：{article_id}
法条内容：{article_content}

## 判断标准（需全部满足才能标记为APPLICABLE）
1. **主体匹配**：法条规定的行为主体类型与案件中的主体类型一致（如纳税人、劳动者、生产经营单位等）
2. **情形匹配**：法条描述的具体违法行为或法律情形与案件描述的情形相符，而非仅泛泛相关
3. **计算可行**：法条包含数值计算公式、比例标准或金额范围，且结合案件信息可以进行计算

## 判断任务
逐条验证上述3项标准。**若有任一标准不满足，则is_applicable=false。**

请严格按照以下JSON格式输出（不要输出任何其他内容）：
{{
  "is_applicable": true或false,
  "confidence": 0.0到1.0之间的数值,
  "judgement": "APPLICABLE"或"NOT_APPLICABLE",
  "reasoning": [
    {{
      "criterion": "主体匹配/情形匹配/计算可行",
      "satisfied": true或false,
      "evidence": "案件事实对应证据，或不满足的原因"
    }}
  ],
  "missing_info": ["若APPLICABLE但缺少计算参数，列出缺失信息；否则为空列表"]
}}

注意：
1. 法条必须包含可计算的公式或标准，纯粹的行为规范（无数值计算规定）不适用
2. 严格输出JSON，不添加其他说明"""


CONFLICT_RESOLUTION_PROMPT_TEMPLATE = """以下多个法律条文均被初步判断为适用，请从中**精确选出计算本案数值答案所必须用到的全部法条**。

## 案件事实
{case_facts}

## 候选法条
{articles_text}

## 分析步骤
1. **识别计算结构**：本案数值答案需要哪些计算要素？（如税基定义、税率、处罚倍数、工资基数、年限取整规则等）
2. **判断法条关系**：
   - **互为补充**：各法条分别提供不同的计算要素，或同一部法律的多个条款共同构成完整计算依据——此时应**全部选入**
   - **互相冲突**：多条法条对同一问题有不同规定，只能选其一——此时按冲突解决原则选最优
3. **补充关系举例**：
   - 同一部法律的第1条（定义应税对象）+ 第3条（税率）+ 第4条（计税方法）→ 全部选入
   - 最高院解释（提供计算公式）+ 基本法律（提供利率参数或法律授权）→ 全部选入
   - 主体罚款公式条款 + 加重情节倍数条款 → 全部选入

## 冲突时的选择原则（仅在法条互相冲突时适用）
1. **优先特别法**：有专门针对本案情形的具体规定时，优先采用
2. **优先新法**：新法与旧法同等专业性时，优先适用较新的
3. **优先上位法**：层级高的法条优先（宪法>法律>条例>规章>解释）

## 输出要求
请分析后输出JSON：
{{
  "primary_article_id": "最主要、最直接适用的法条ID（必填）",
  "secondary_article_ids": ["当计算必须借助这些法条时填写，否则为空列表"],
  "all_applicable": ["所有必须引用的法条ID列表（可以是1个、2个或3个）"],
  "is_complementary": true或false（true=互为补充均需引用，false=存在冲突只选最优）,
  "reasoning": "说明各法条的分工或冲突关系，以及选择理由"
}}

注意：准确判断需要几个法条就填几个，不要人为减少。"""


RELAXED_APPLICABILITY_PROMPT_TEMPLATE = """你是一位法律专家，需要判断某法律条文是否与给定案件相关，并能用于数值计算。

## 案件事实
{case_facts}

## 待判断法条
法条编号：{article_id}
法条内容：{article_content}

## 判断标准（满足以下2条即可标记为APPLICABLE）
1. **情形相关**：法条描述的法律情形或领域与案件描述的情形相关
2. **计算可行**：法条包含数值计算公式、比例标准或金额范围，结合案件信息可以尝试计算

请严格按照以下JSON格式输出：
{{
  "is_applicable": true或false,
  "confidence": 0.0到1.0之间的数值,
  "judgement": "APPLICABLE"或"NOT_APPLICABLE",
  "reasoning": [
    {{
      "criterion": "情形相关/计算可行",
      "satisfied": true或false,
      "evidence": "简要说明"
    }}
  ],
  "missing_info": []
}}

注意：这是宽松判断，只要法条与案件领域相关且有计算内容即可，严格输出JSON。"""


def build_applicability_prompt(article: Dict, case_facts: str) -> str:
    """Build prompt for applicability judgment."""
    return APPLICABILITY_PROMPT_TEMPLATE.format(
        case_facts=case_facts,
        article_id=article.get("article_id", ""),
        article_content=article.get("content", ""),
    )


def build_relaxed_applicability_prompt(article: Dict, case_facts: str) -> str:
    """Build relaxed prompt for applicability judgment (emergency fallback)."""
    return RELAXED_APPLICABILITY_PROMPT_TEMPLATE.format(
        case_facts=case_facts,
        article_id=article.get("article_id", ""),
        article_content=article.get("content", ""),
    )


def build_conflict_resolution_prompt(articles: List[Dict], case_facts: str) -> str:
    """Build prompt for conflict resolution."""
    articles_text = ""
    for i, art in enumerate(articles, 1):
        articles_text += f"\n{i}. 法条ID：{art.get('article_id', '')}\n"
        articles_text += f"   内容：{art.get('content', '')}\n"

    return CONFLICT_RESOLUTION_PROMPT_TEMPLATE.format(
        case_facts=case_facts,
        articles_text=articles_text
    )

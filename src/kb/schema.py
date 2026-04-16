from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LegalElement:
    id: str          # e.g. "e1"
    cond: str        # condition description in Chinese
    desc: str        # human-readable short description


@dataclass
class LegalArticle:
    article_id: str              # e.g. "《中华人民共和国防沙治沙法》第四十条"
    law_name: str                # e.g. "中华人民共和国防沙治沙法"
    article_num: str             # e.g. "四十"
    content: str                 # raw article text
    concept: str = ""            # what legal situation this covers
    elements: List[LegalElement] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    formula: str = ""            # Python-evaluable expression
    formula_variables: Dict[str, str] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "law_name": self.law_name,
            "article_num": self.article_num,
            "content": self.content,
            "concept": self.concept,
            "elements": [{"id": e.id, "cond": e.cond, "desc": e.desc} for e in self.elements],
            "parameters": self.parameters,
            "formula": self.formula,
            "formula_variables": self.formula_variables,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LegalArticle':
        elements = [LegalElement(e["id"], e["cond"], e["desc"])
                    for e in d.get("elements", [])]
        return cls(
            article_id=d["article_id"],
            law_name=d["law_name"],
            article_num=d["article_num"],
            content=d["content"],
            concept=d.get("concept", ""),
            elements=elements,
            parameters=d.get("parameters", {}),
            formula=d.get("formula", ""),
            formula_variables=d.get("formula_variables", {}),
            references=d.get("references", []),
            metadata=d.get("metadata", {}),
        )

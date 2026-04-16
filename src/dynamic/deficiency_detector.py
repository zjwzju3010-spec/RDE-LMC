"""
Deficiency detector for the DynamicCM module.
Identifies missing information that prevents calculation completion.
"""
import os
import sys
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFICIENCY_NONE = "NONE"
DEFICIENCY_MISSING_VARS = "MISSING_VARS"
DEFICIENCY_AMBIGUOUS_ARTICLE = "AMBIGUOUS_ARTICLE"
DEFICIENCY_EXECUTION_ERROR = "EXECUTION_ERROR"


def detect(ir: Dict, execution_result: Dict = None) -> Dict:
    """
    Detect deficiencies in the IR or execution result.

    Args:
        ir: IR dict from ir_generator
        execution_result: Optional execution result dict from executor

    Returns:
        Dict with: has_deficiency, missing_variables, deficiency_type
    """
    missing = []
    deficiency_type = DEFICIENCY_NONE

    # Check IR for missing variables
    ir_missing = ir.get("missing_variables", [])
    if ir_missing:
        for m in ir_missing:
            if m and str(m).strip():
                missing.append({"name": "unknown", "description": str(m)})
        if missing:
            deficiency_type = DEFICIENCY_MISSING_VARS

    # Check if expression is empty
    if not ir.get("expression"):
        if not missing:
            missing.append({"name": "expression", "description": "无法生成计算表达式"})
        deficiency_type = DEFICIENCY_MISSING_VARS

    # Check execution result for errors
    if execution_result is not None and not execution_result.get("success", True):
        error = execution_result.get("error", "")
        if error:
            # Check if error is about undefined variable
            if "Undefined variable" in error or "undefined" in error.lower():
                # Extract variable name from error
                import re
                m = re.search(r"Undefined variable: '([^']+)'", error)
                var_name = m.group(1) if m else "unknown"
                missing.append({"name": var_name, "description": f"计算中未定义的变量: {var_name}"})
                deficiency_type = DEFICIENCY_MISSING_VARS
            else:
                missing.append({"name": "execution_error", "description": error})
                deficiency_type = DEFICIENCY_EXECUTION_ERROR

    has_deficiency = len(missing) > 0

    if has_deficiency:
        logger.debug(f"Detected deficiency: {deficiency_type}, missing: {missing}")

    return {
        "has_deficiency": has_deficiency,
        "missing_variables": missing,
        "deficiency_type": deficiency_type,
    }


def detect_all(ir_list: List[Dict], execution_results: List[Dict] = None) -> Dict:
    """
    Detect deficiencies across all IRs and execution results.

    Returns aggregated deficiency info.
    """
    all_missing = []
    has_any_deficiency = False

    for i, ir in enumerate(ir_list):
        exec_result = execution_results[i] if (execution_results and i < len(execution_results)) else None
        deficiency = detect(ir, exec_result)

        if deficiency["has_deficiency"]:
            has_any_deficiency = True
            all_missing.extend(deficiency["missing_variables"])

    # Deduplicate missing variables by description
    seen = set()
    unique_missing = []
    for m in all_missing:
        key = m.get("description", "")
        if key not in seen:
            seen.add(key)
            unique_missing.append(m)

    return {
        "has_deficiency": has_any_deficiency,
        "missing_variables": unique_missing,
        "deficiency_type": DEFICIENCY_MISSING_VARS if has_any_deficiency else DEFICIENCY_NONE,
    }

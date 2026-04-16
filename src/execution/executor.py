"""
Executor for the SymbolCE module.
Builds expression tree from IR and safely evaluates it.
"""
import os
import sys
from typing import List, Dict, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.execution.expression_tree import parse_expression, evaluate_node

logger = get_logger(__name__)


def _safe_execute(expression: str, variable_bindings: dict = None) -> float:
    """
    Safely parse and execute an expression string.

    Args:
        expression: Python-syntax arithmetic expression with numeric values
        variable_bindings: Dict of {var_name: {value: float, ...}} from IR

    Returns:
        float result

    Raises:
        ValueError, SyntaxError, ZeroDivisionError
    """
    if not expression or not expression.strip():
        raise ValueError("Empty expression")

    # Build variables dict from bindings
    variables = {}
    if variable_bindings:
        for var_name, binding in variable_bindings.items():
            if isinstance(binding, dict):
                val = binding.get("value")
            else:
                val = binding
            if val is not None:
                try:
                    variables[var_name] = float(val)
                except (TypeError, ValueError):
                    pass

    # Parse and evaluate
    tree = parse_expression(expression)
    return evaluate_node(tree, variables)


def execute(ir: Dict, config: dict = None) -> Dict:
    """
    Execute a single IR to compute a numerical result.

    Args:
        ir: IR dict from ir_generator.generate_ir()
        config: Configuration dict (optional)

    Returns:
        Dict with: result, success, error, expression_used, intermediate_steps
    """
    article_id = ir.get("article_id", "")
    expression = ir.get("expression", "")
    variable_bindings = ir.get("variable_bindings", {})
    intermediate_steps = ir.get("intermediate_steps", [])

    if not expression:
        missing = ir.get("missing_variables", [])
        return {
            "article_id": article_id,
            "result": None,
            "success": False,
            "error": f"No expression generated. Missing: {missing}",
            "expression_used": "",
            "intermediate_steps": intermediate_steps,
        }

    try:
        result = _safe_execute(expression, variable_bindings)

        logger.info(f"Executed {article_id}: {expression} = {result}")

        return {
            "article_id": article_id,
            "result": result,
            "success": True,
            "error": None,
            "expression_used": expression,
            "intermediate_steps": intermediate_steps,
        }

    except Exception as e:
        logger.warning(f"Execution failed for {article_id}: {expression!r}. Error: {e}")
        return {
            "article_id": article_id,
            "result": None,
            "success": False,
            "error": str(e),
            "expression_used": expression,
            "intermediate_steps": intermediate_steps,
        }


def execute_multi(ir_list: List[Dict], config: dict = None) -> List[Dict]:
    """
    Execute multiple IRs, one per article.

    Returns list of execution result dicts.
    """
    results = []
    for ir in ir_list:
        result = execute(ir, config)
        results.append(result)
    return results

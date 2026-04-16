"""
Safe expression tree for evaluating legal calculation formulas.
Uses Python's ast module for parsing (NOT eval) with strict whitelist validation.
"""
import ast
import operator
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union

# Node type constants
NODE_CONST = "CONST"      # Numeric literal
NODE_VAR = "VAR"          # Variable (must be resolved before evaluation)
NODE_BINOP = "BINOP"      # Binary operation
NODE_UNOP = "UNOP"        # Unary operation
NODE_IF = "IF"            # Ternary: body if test else orelse (children: [test, body, orelse])
NODE_CALL = "CALL"        # Allowed function call (children: args)
NODE_COMPARE = "COMPARE"  # Comparison
NODE_BOOLOP = "BOOLOP"    # Boolean operation (and/or)

# Allowed binary operators
ALLOWED_BINOPS = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
    ast.FloorDiv: ("//", operator.floordiv),
    ast.Pow: ("**", operator.pow),
    ast.Mod: ("%", operator.mod),
}

# Allowed comparison operators
ALLOWED_COMPOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# Allowed function calls
ALLOWED_FUNCTIONS = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
}


@dataclass
class Node:
    """Expression tree node."""
    type: str
    value: Any = None
    children: List['Node'] = field(default_factory=list)

    def __repr__(self):
        if self.type == NODE_CONST:
            return f"CONST({self.value})"
        elif self.type == NODE_VAR:
            return f"VAR({self.value})"
        elif self.type in (NODE_BINOP, NODE_UNOP):
            return f"{self.type}({self.value}, {self.children})"
        elif self.type == NODE_CALL:
            return f"CALL({self.value}, {self.children})"
        else:
            return f"{self.type}({self.value})"


def _ast_to_node(node: ast.AST) -> Node:
    """
    Recursively convert Python AST node to our safe Node tree.
    Raises ValueError for any disallowed construct.
    """
    if isinstance(node, (ast.Constant, ast.Num)):
        # Numeric constant
        val = node.n if isinstance(node, ast.Num) else node.value
        if not isinstance(val, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got: {type(val)}")
        return Node(type=NODE_CONST, value=float(val))

    elif isinstance(node, ast.Name):
        # Variable reference
        name = node.id
        # Safety: disallow access to builtins
        if name.startswith('__') or name in ('exec', 'eval', 'import', 'open',
                                               'compile', 'globals', 'locals',
                                               'getattr', 'setattr', 'delattr',
                                               'input', 'print', 'vars', 'dir'):
            raise ValueError(f"Disallowed name: {name}")
        return Node(type=NODE_VAR, value=name)

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINOPS:
            raise ValueError(f"Disallowed binary operator: {op_type}")
        op_symbol, _ = ALLOWED_BINOPS[op_type]
        left = _ast_to_node(node.left)
        right = _ast_to_node(node.right)
        return Node(type=NODE_BINOP, value=op_symbol, children=[left, right])

    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            operand = _ast_to_node(node.operand)
            return Node(type=NODE_UNOP, value="-", children=[operand])
        elif isinstance(node.op, ast.UAdd):
            return _ast_to_node(node.operand)  # +x is just x
        else:
            raise ValueError(f"Disallowed unary operator: {type(node.op)}")

    elif isinstance(node, ast.IfExp):
        # Ternary: body if test else orelse
        test = _ast_to_node(node.test)
        body = _ast_to_node(node.body)
        orelse = _ast_to_node(node.orelse)
        return Node(type=NODE_IF, children=[test, body, orelse])

    elif isinstance(node, ast.Call):
        # Only allow specific functions
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed (no method calls)")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Disallowed function: {func_name}")
        args = [_ast_to_node(arg) for arg in node.args]
        return Node(type=NODE_CALL, value=func_name, children=args)

    elif isinstance(node, ast.Compare):
        # Comparison: left op1 comparators[0]
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple comparisons (a op b) allowed")
        op_type = type(node.ops[0])
        if op_type not in ALLOWED_COMPOPS:
            raise ValueError(f"Disallowed comparison operator: {op_type}")
        left = _ast_to_node(node.left)
        right = _ast_to_node(node.comparators[0])
        return Node(type=NODE_COMPARE, value=op_type.__name__, children=[left, right])

    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            op_str = "and"
        elif isinstance(node.op, ast.Or):
            op_str = "or"
        else:
            raise ValueError(f"Disallowed bool op: {type(node.op)}")
        children = [_ast_to_node(v) for v in node.values]
        return Node(type=NODE_BOOLOP, value=op_str, children=children)

    elif isinstance(node, ast.Expression):
        return _ast_to_node(node.body)

    else:
        raise ValueError(f"Disallowed AST node type: {type(node).__name__}")


def parse_expression(expr_str: str) -> Node:
    """
    Parse a Python expression string into a safe Node tree.
    Uses Python's ast module for parsing but validates all node types.

    Args:
        expr_str: Python-syntax arithmetic expression

    Returns:
        Node tree

    Raises:
        ValueError: If expression contains disallowed constructs
        SyntaxError: If expression is not valid Python syntax
    """
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")

    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {expr_str!r}. Error: {e}")

    return _ast_to_node(tree)


def evaluate_node(node: Node, variables: Dict[str, float] = None) -> float:
    """
    Recursively evaluate the node tree.

    Args:
        node: Root node of expression tree
        variables: Dict mapping variable names to numeric values

    Returns:
        float result

    Raises:
        ValueError: If a VAR node has no binding in variables
        ZeroDivisionError: If division by zero
    """
    if variables is None:
        variables = {}

    if node.type == NODE_CONST:
        return float(node.value)

    elif node.type == NODE_VAR:
        name = node.value
        if name not in variables:
            raise ValueError(f"Undefined variable: {name!r}. Available: {list(variables.keys())}")
        val = variables[name]
        return float(val)

    elif node.type == NODE_BINOP:
        left = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)

        # Find the operator function
        for op_type, (symbol, func) in ALLOWED_BINOPS.items():
            if symbol == node.value:
                if symbol in ('/', '//') and right == 0:
                    raise ZeroDivisionError(f"Division by zero in expression")
                return float(func(left, right))

        raise ValueError(f"Unknown binary operator: {node.value}")

    elif node.type == NODE_UNOP:
        operand = evaluate_node(node.children[0], variables)
        if node.value == "-":
            return -operand
        return operand

    elif node.type == NODE_IF:
        # children: [test, body, orelse]
        test_val = evaluate_node(node.children[0], variables)
        if test_val:  # Truthy test
            return evaluate_node(node.children[1], variables)
        else:
            return evaluate_node(node.children[2], variables)

    elif node.type == NODE_CALL:
        func_name = node.value
        args = [evaluate_node(child, variables) for child in node.children]
        func = ALLOWED_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")
        # min/max with a single scalar arg: just return the value
        if func_name in ("min", "max") and len(args) == 1:
            return float(args[0])
        return float(func(*args))

    elif node.type == NODE_COMPARE:
        left = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)

        # Map op name to operator
        op_name_to_op = {
            t.__name__: func for t, func in ALLOWED_COMPOPS.items()
        }
        op_func = op_name_to_op.get(node.value)
        if op_func is None:
            raise ValueError(f"Unknown comparison operator: {node.value}")
        return float(op_func(left, right))

    elif node.type == NODE_BOOLOP:
        if node.value == "and":
            result = True
            for child in node.children:
                val = evaluate_node(child, variables)
                result = result and bool(val)
            return float(result)
        elif node.value == "or":
            result = False
            for child in node.children:
                val = evaluate_node(child, variables)
                result = result or bool(val)
            return float(result)

    raise ValueError(f"Unknown node type: {node.type}")


def safe_eval_expression(expr_str: str, variables: Dict[str, float] = None) -> float:
    """
    Convenience function: parse and evaluate an expression string.

    Args:
        expr_str: Python-syntax arithmetic expression
        variables: Variable bindings

    Returns:
        float result
    """
    tree = parse_expression(expr_str)
    return evaluate_node(tree, variables or {})

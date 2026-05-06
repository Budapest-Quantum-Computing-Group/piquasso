#
# Copyright 2021-2026 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import operator as op
from typing import Optional, Tuple, Any

from piquasso.api.exceptions import InvalidExpression

BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
}

UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
    ast.Not: op.not_,
}

BOOLOPS = {
    ast.And: all,
    ast.Or: any,
}

CMPOPS = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

# derive ALLOWED from mappings + structural nodes
ALLOWED = (
    {
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Subscript,
        ast.Slice,
        ast.Constant,
        ast.List,
        ast.Tuple,
        # optional: for Python < 3.9 subscripts
        getattr(ast, "Index", object()),
    }
    | set(BINOPS.keys())
    | set(UNARYOPS.keys())
    | set(BOOLOPS.keys())
    | set(CMPOPS.keys())
)
# remove the placeholder object if present
ALLOWED.discard(object())


class Expression:
    """Safe expression over a single variable `x` (tuple-like)."""

    def __init__(self, src: str):
        self._src = src.strip()
        try:
            self._tree = ast.parse(self._src, mode="eval")
        except SyntaxError as e:
            raise InvalidExpression(f"Invalid syntax: {e}") from e
        self._validate(self._tree)

    def __call__(self, x: Optional[Tuple] = None) -> Any:
        x = x if x is not None else tuple()
        return self._eval(self._tree.body, x)

    def __str__(self) -> str:
        return self._src

    __repr__ = __str__

    @staticmethod
    def _validate(tree: ast.AST) -> None:
        allowed_tuple = tuple(ALLOWED)
        for n in ast.walk(tree):
            if not isinstance(n, allowed_tuple):  # type: ignore
                raise InvalidExpression(
                    f"Invalid construct in the expression: {type(n).__name__}"
                )
            if isinstance(n, ast.Name) and n.id != "x":
                raise InvalidExpression("Only the variable name 'x' is allowed")
            if isinstance(n, ast.Constant) and not isinstance(
                n.value, (int, float, bool)
            ):
                raise InvalidExpression(
                    "Only numeric and boolean constants are allowed"
                )

    def _eval(self, node: ast.AST, x: Tuple) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return x

        if isinstance(node, ast.Tuple):
            return tuple(self._eval(elt, x) for elt in node.elts)

        if isinstance(node, ast.List):
            return [self._eval(elt, x) for elt in node.elts]

        if isinstance(node, ast.UnaryOp):
            fn = UNARYOPS.get(type(node.op))
            if fn is None:
                raise InvalidExpression(
                    f"Unsupported unary op: {type(node.op).__name__}"
                )
            return fn(self._eval(node.operand, x))  # type: ignore

        if isinstance(node, ast.BinOp):
            fn = BINOPS.get(type(node.op))
            if fn is None:
                raise InvalidExpression(
                    f"Unsupported binary op: {type(node.op).__name__}"
                )
            return fn(self._eval(node.left, x), self._eval(node.right, x))

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = self._eval(v, x)
                    if not result:  # falsy → return immediately
                        return result
                return result

            if isinstance(node.op, ast.Or):
                result = False
                for v in node.values:
                    result = self._eval(v, x)
                    if result:  # truthy → return immediately
                        return result
                return result

            raise InvalidExpression("Unsupported boolean operator")

        if isinstance(node, ast.Compare):
            left = self._eval(node.left, x)
            for op_node, right_expr in zip(node.ops, node.comparators):
                right = self._eval(right_expr, x)
                fn = CMPOPS.get(type(op_node))
                if fn is None:
                    raise InvalidExpression(
                        f"Unsupported cmp op: {type(op_node).__name__}"
                    )
                if not fn(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.Subscript):
            seq = self._eval(node.value, x)
            sl = node.slice
            # Python < 3.9 wraps index in ast.Index
            if isinstance(sl, getattr(ast, "Index", ())):
                sl = sl.value  # type: ignore
            if isinstance(sl, ast.Slice):
                start = self._eval(sl.lower, x) if sl.lower else None
                stop = self._eval(sl.upper, x) if sl.upper else None
                step = self._eval(sl.step, x) if sl.step else None
                return seq[slice(start, stop, step)]
            return seq[self._eval(sl, x)]

        raise InvalidExpression(f"Unsupported node: {type(node).__name__}")

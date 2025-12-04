#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

import pytest

from piquasso.core._expressions import Expression
from piquasso.api.exceptions import InvalidExpression


def test_Expression_evaluation_boolean():
    cond = Expression("x[0] == 2 and x[1] >= 0")

    assert cond((2, 3)) is True
    assert str(cond) == repr(cond) == "x[0] == 2 and x[1] >= 0"


def test_Expression_evaluation_numeric():
    expr = Expression("x[0] ** 2 + x[1] ** 2")

    assert expr((3, 4)) == 3**2 + 4**2
    assert str(expr) == repr(expr) == "x[0] ** 2 + x[1] ** 2"


def test_invalid_expression_raises_exception():
    invalid_expressions = [
        "import os",
        "__import__('os').system('rm -rf /')",
        "x[0] > 0 and os.path.exists('file.txt')",
        "sum(x)",
        "x[0] > 0 or __import__('math').sqrt(4)",
        "x[0] > 0; print('Hello World')",
    ]

    for expr in invalid_expressions:
        try:
            Expression(expr)
        except InvalidExpression:
            pass
        else:
            assert False, f"Expression '{expr}' should have raised ValueError"


@pytest.mark.parametrize(
    "src,x,expected",
    [
        ("1 + 2 * 3", (), 7),
        ("(1 + 2) * 3", (), 9),
        ("2 ** 3 ** 2", (), 512),  # right-associative
        ("10 % 4", (), 2),
        ("10 / 4", (), 2.5),
        ("+5 + -2", (), 3),
        ("True and False", (), False),
        ("True or False", (), True),
        ("not False", (), True),
        ("1 < 2 < 3", (), True),  # chained comparison
        ("1 < 2 > 3", (), False),
        ("x[0] + x[1]", (3, 4), 7),
        ("x[-1]", (1, 2, 3), 3),
        ("x[1:4]", (0, 1, 2, 3, 4), (1, 2, 3)),  # slicing returns same type as seq
        ("x[::-1][0]", (1, 2, 3), 3),
        ("(x[0] ** 2 + x[1] ** 2) > 10 and x[0] < x[1]", (3, 4), True),
        ("[x[0], x[1], 5][2]", (9, 8), 5),  # list literal + index
        ("(x[0], x[1], 5)[1]", (9, 8), 8),  # tuple literal + index
        ("x[0:5:2]", (0, 1, 2, 3, 4, 5), (0, 2, 4)),
    ],
)
def test_basic_expressions(src, x, expected):
    expr = Expression(src)
    assert expr(x) == expected


# ---------- short-circuit semantics ----------


@pytest.mark.parametrize(
    "src",
    [
        "False and (1 / 0)",  # left False => do not eval right
        "True or (1 / 0)",  # left True  => do not eval right
    ],
)
def test_short_circuit_no_eval_right(src):
    expr = Expression(src)
    # Should not raise ZeroDivisionError due to short-circuit
    assert isinstance(expr(()), bool)


# ---------- input coercion: list or tuple works ----------


@pytest.mark.parametrize(
    "x",
    [
        [3, 4],
        (3, 4),
    ],
)
def test_x_coercion_list_tuple(x):
    expr = Expression("x[0] ** 2 + x[1] ** 2")
    assert expr(x) == 25


# ---------- string representation ----------


def test_str_and_repr():
    src = "x[0] + 1"
    expr = Expression(src)
    assert str(expr) == src
    assert repr(expr) == src


# ---------- error cases: validation ----------


@pytest.mark.parametrize(
    "src",
    [
        "y + 1",  # forbidden name
        "x.__class__",  # attribute access is not in ALLOWED
        "len(x)",  # function call not allowed
        "{1: 2}",  # dict constant not allowed
        "{1, 2}",  # set literal not allowed
        "(i for i in x)",  # generator not allowed
        "[i for i in x]",  # list comp not allowed
        "lambda z: z",  # lambda not allowed
        "x if True else 0",  # conditional expression not allowed
    ],
)
def test_invalid_constructs_raise(src):
    with pytest.raises(InvalidExpression):
        Expression(src)


def test_invalid_literal_type_raises():
    # string literal not permitted
    with pytest.raises(InvalidExpression):
        Expression("'abc'")


def test_invalid_syntax_raises():
    with pytest.raises(InvalidExpression):
        Expression("1 +")  # SyntaxError -> InvalidExpression


def test_runtime_type_error_index():
    with pytest.raises(InvalidExpression):
        Expression("x['a']")


# ---------- runtime errors are surfaced when reached ----------


def test_runtime_zero_division_when_reached():
    expr = Expression("True and (1/0)")  # right is evaluated
    with pytest.raises(ZeroDivisionError):
        expr(())


def test_runtime_index_error_when_reached():
    expr = Expression("x[1]")  # valid, but out-of-range at runtime
    with pytest.raises(IndexError):
        expr((0,))


# ---------- slicing variants ----------


@pytest.mark.parametrize(
    "src,x,expected",
    [
        ("x[:3]", (0, 1, 2, 3), (0, 1, 2)),
        ("x[1:]", (0, 1, 2, 3), (1, 2, 3)),
        ("x[:-1]", (0, 1, 2, 3), (0, 1, 2)),
        ("x[::2]", (0, 1, 2, 3, 4), (0, 2, 4)),
        ("x[-3:-1]", (0, 1, 2, 3, 4), (2, 3)),
    ],
)
def test_slicing_variants(src, x, expected):
    expr = Expression(src)
    assert expr(x) == expected


# ---------- comparison coverage ----------


@pytest.mark.parametrize(
    "src,x,expected",
    [
        ("x[0] == 3", (3,), True),
        ("x[0] != 3", (3,), False),
        ("x[0] <  4", (3,), True),
        ("x[0] <= 3", (3,), True),
        ("x[0] >  2", (3,), True),
        ("x[0] >= 4", (3,), False),
        ("0 < x[0] < 10", (3,), True),
        ("0 < x[0] < 3", (3,), False),
    ],
)
def test_comparisons(src, x, expected):
    expr = Expression(src)
    assert expr(x) == expected


# ---------- unary operators coverage ----------


@pytest.mark.parametrize(
    "src,expected",
    [
        ("+5", 5),
        ("-5", -5),
        ("not True", False),
    ],
)
def test_unary_ops(src, expected):
    expr = Expression(src)
    assert expr(()) == expected


@pytest.mark.parametrize(
    "src",
    [
        # plain booleans
        "True and False",
        "True or False",
        "False and True",
        "False or True",
        # numeric + boolean mixing
        "2 or False",
        "False or 2",
        "0 or 5",
        "5 or 0",
        "0 and 5",
        "5 and 0",
        "2 and 3",
        "0 and 3",
        # chains
        "0 or 1 or 2",
        "0 and 1 and 2",
        "1 and 2 or 0",
        "0 or 1 and 2",
        "(0 or 1) and 2",
        "0 or (1 and 2)",
        # with comparisons
        "(1 < 2) and 3",
        "3 and (1 < 2)",
        "(1 < 2) or 3",
        "3 or (1 < 2)",
    ],
)
def test_bool_op_value_semantics_match_python(src):
    """Ensure 'and'/'or' return the same value as Python's eval."""
    expr = Expression(src)
    expected = eval(src)
    assert expr(()) == expected


def test_Expression_evaluation_with_no_input():
    expr = Expression("42")
    assert expr() == 42


def test_bitxor_python_semantics():
    cases = [
        ("True ^ False", ()),
        ("False ^ False", ()),
        ("True ^ True", ()),
        ("3 ^ 5", ()),
        ("x[0] ^ x[1]", (3, 5)),
        ("(1 < 2) ^ (2 < 1)", ()),
    ]
    for src, x in cases:
        expr = Expression(src)
        assert expr(x) == eval(src)

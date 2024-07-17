import numpy as np

import numba
from numba.types import void, voidptr, CPointer

import jax
import jax.numpy as jnp
from jax import core, jit, grad, jvp
from jax.interpreters import mlir, ad
import jaxlib.mlir.ir as ir
from jax.lib import xla_client

from piquasso._math.permanent import permanent as _numba_permanent

import ctypes


_custom_call_cache = []


def _permanent_cpu_lowering(ctx, matrix, rows, cols):
    matrix_dtype = ctx.avals_in[0].dtype
    occ_num_dtype = ctx.avals_in[1].dtype

    custom_call_name = f"cpu_permanent_{matrix_dtype}_{matrix_dtype}"

    if custom_call_name not in _custom_call_cache:

        @numba.cfunc(void(voidptr, CPointer(voidptr)))
        def xla_cpu_custom_call_target(output_ptr, input_ptrs):
            size = numba.carray(input_ptrs[3], 1, dtype=numba.types.int32)[0]
            output = numba.carray(output_ptr, (1,), dtype=matrix_dtype)

            matrix_carr = numba.carray(input_ptrs[0], (size, size), dtype=matrix_dtype)
            rows_carr = numba.carray(input_ptrs[1], size, dtype=occ_num_dtype)
            cols_carr = numba.carray(input_ptrs[2], size, dtype=occ_num_dtype)

            output[0] = _numba_permanent(matrix_carr, rows_carr, cols_carr)

        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

        capsule = ctypes.pythonapi.PyCapsule_New(
            xla_cpu_custom_call_target.address, b"xla._CUSTOM_CALL_TARGET", None
        )

        xla_client.register_custom_call_target(custom_call_name, capsule, "cpu")

        _custom_call_cache.append(custom_call_name)

    custom_call_op = mlir.custom_call(
        custom_call_name,
        result_types=[ir.RankedTensorType.get((1,), matrix.type.element_type)],
        operands=[matrix, rows, cols, mlir.ir_constant(matrix.type.shape[0])],
        operand_layouts=[(1, 0), (0,), (0,), ()],
        result_layouts=[(0,)],
    )

    return custom_call_op.results


def _abstract_eval(matrix, rows, cols):
    return core.ShapedArray((1,), matrix.dtype)


def falling_factorial(n, k):
    res = 1.0
    for i in range(k):
        res *= n - i

    return res


def _permanent_jvp(args, tangents):
    # TODO: THIS IS WRONG
    matrix, rows, cols = args
    tangent = tangents[0]
    primal_out = _permanent_primitive.bind(matrix, rows, cols)

    tangent_out = 0.0

    # NOTE: anything you do here you'd normally do with NumPy, you should do it with
    # JAX, otherwise, the result will be incorrect. If you want to avoid this, maybe you
    # can implement another JAX primitive for the gradient. Or, you can try to implement
    # this with fori_loops and whatnot.
    rows_copy = jnp.copy(rows)
    cols_copy = jnp.copy(cols)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            m = cols[col]
            max_index = min(rows[row], cols[col])
            permanent_partial_derivative = 0.0

            for k in range(max_index):
                rows_copy = rows_copy.at[row].set(rows_copy[row] - 1)
                cols_copy = cols_copy.at[col].set(cols_copy[col] - 1)
                perm_term = _permanent_primitive.bind(matrix, rows_copy, cols_copy)

                permanent_partial_derivative += (
                    falling_factorial(m, k + 1) * matrix[row, col] ** k * perm_term
                )
                breakpoint()

            rows_copy = rows_copy.at[row].set(rows[row])
            cols_copy = cols_copy.at[col].set(cols[col])

            tangent_out += permanent_partial_derivative * tangent[row, col]

    return primal_out, tangent_out


_permanent_primitive = jax.core.Primitive("permanent")

_permanent_primitive.def_impl(
    lambda matrix, rows, cols: np.array(
        [_numba_permanent(np.array(matrix), np.array(rows), np.array(cols))]
    )
)

_permanent_primitive.def_abstract_eval(_abstract_eval)

mlir.register_lowering(_permanent_primitive, _permanent_cpu_lowering, platform="cpu")

ad.primitive_jvps[_permanent_primitive] = _permanent_jvp


def permanent_with_reduction(matrix, rows, cols):
    """JAX implementation of the permanent, using the Glynn formula with Gray codes."""

    rows_array = np.array(rows)
    cols_array = np.array(cols)

    permanent_in_array = _permanent_primitive.bind(matrix, rows_array, cols_array)
    just_the_permanent = permanent_in_array[0]

    return just_the_permanent


def test_():
    A = np.random.rand(2, 2) + np.random.rand(2, 2) * 1j

    expected = np.array(
        [
            [
                4 * A[0, 0] * A[1, 1] + 2 * A[0, 1] * A[1, 0] + 2 * A[0, 1] * A[1, 0],
                4 * A[0, 0] * A[1, 0],
            ],
            [4 * A[0, 0] * A[0, 1], 2 * A[0, 0] ** 2],
        ]
    )

    assert np.isclose(
        _permanent_jvp(
            (A, np.array([2, 1]), np.array([2, 1])),
            (np.array([[1, 0], [0, 0]]), None, None),
        ),
        expected[0, 0],
    )


if __name__ == "__main__":
    z = np.ones((3, 3)) * 5 + 1j * np.ones((3, 3))
    z4 = np.ones((4, 4)) * 5 + 1j * np.ones((4, 4))

    rows = (1, 2, 1)
    cols = (3, 0, 1)

    print(permanent_with_reduction(z, rows, cols))

    jitted_jax_permanent = jit(permanent_with_reduction, static_argnums=(1, 2))

    import time

    start_time = time.time()

    print(jitted_jax_permanent(z, rows, cols))
    print(time.time() - start_time)
    start_time = time.time()

    print(jitted_jax_permanent(z, rows, (2, 1, 1)))
    print(time.time() - start_time)
    start_time = time.time()

    print("---------------")
    print(permanent_with_reduction(z4, (0, 2, 1, 3), (3, 1, 2, 0)))
    print(jitted_jax_permanent(z4, (0, 2, 1, 3), (3, 1, 2, 0)))
    print(time.time() - start_time)
    start_time = time.time()

    def func2(matrix):
        return jitted_jax_permanent(matrix, (1, 1), (1, 1))

    E00 = np.zeros((2, 2), dtype=complex)
    E00[0, 0] = 1.0

    z2 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)

    assert np.isclose(
        _permanent_jvp((z2, np.array([1, 1]), np.array([1, 1])), (E00,))[1], z2[1, 1]
    )
    jvp_result_2by2 = jvp(func2, (z2,), (E00,))

    assert np.isclose(jvp_result_2by2[0], permanent_with_reduction(z2, (1, 1), (1, 1)))
    assert np.isclose(jvp_result_2by2[1], permanent_with_reduction(z2, (0, 1), (0, 1)))

    grad_jax_permanent = jit(
        grad(jitted_jax_permanent, holomorphic=True), static_argnums=(1, 2)
    )

    A = np.random.rand(2, 2) + np.random.rand(2, 2) * 1j

    assert np.allclose(
        jitted_jax_permanent(A, (1, 1), (1, 1)), A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1]
    )

    assert np.allclose(
        grad_jax_permanent(A, (1, 1), (1, 1)),
        np.array(
            [
                [
                    A[1, 1],
                    A[1, 0],
                ],
                [A[0, 1], A[0, 0]],
            ]
        ),
    )

    assert np.allclose(
        grad_jax_permanent(A, (2, 0), (2, 0)), np.array([[4 * A[0, 0], 0], [0, 0]])
    )

    assert np.allclose(
        jitted_jax_permanent(A, (2, 1), (2, 1)),
        2 * A[0, 0] * (A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0])
        + 2 * A[0, 1] * A[0, 0] * A[1, 0],
    )

    expected = np.array(
        [
            [
                4 * A[0, 0] * A[1, 1] + 2 * A[0, 1] * A[1, 0] + 2 * A[0, 1] * A[1, 0],
                4 * A[0, 0] * A[1, 0],
            ],
            [4 * A[0, 0] * A[0, 1], 2 * A[0, 0] ** 2],
        ]
    )

    assert np.isclose(
        _permanent_jvp(
            (A, np.array([2, 1]), np.array([2, 1])),
            (np.array([[1, 0], [0, 0]]), None, None),
        ),
        expected[0, 0],
    )
    assert np.allclose(
        grad_jax_permanent(A, (2, 1), (2, 1)),
        expected,
    )

    def func(matrix):
        return jitted_jax_permanent(matrix, rows, cols)

    print(jvp(func, (z,), (z,)))

    print(jit(grad(func, holomorphic=True))(z))

    print(grad_jax_permanent(z4, (0, 2, 1, 3), (3, 1, 2, 0)))

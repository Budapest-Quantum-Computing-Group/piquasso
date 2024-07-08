import numpy as np

import numba
from numba.types import void, voidptr, CPointer

from functools import partial


import jax
import jax.numpy as jnp
from jax import core, jit, grad, jvp
from jax.interpreters import mlir, ad
import jaxlib.mlir.ir as ir
from jax.lib import xla_client

from piquasso._math.permanent import permanent as _numba_permanent

import ctypes


_method_cache = []


def _permanent_lowering(ctx, matrix, rows, cols):
    if "cpu_permanent" not in _method_cache:

        matrix_dtype = ctx.avals_in[0].dtype
        occ_num_dtype = ctx.avals_in[1].dtype

        @numba.cfunc(void(voidptr, CPointer(voidptr)))
        def xla_cpu_custom_call_target(output_ptr, input_ptrs):
            size = numba.carray(input_ptrs[3], 1, dtype=numba.types.int32)[0]
            output = numba.carray(output_ptr, (1,), dtype=matrix_dtype)

            matrix_carr = numba.carray(input_ptrs[0], (size,size), dtype=matrix_dtype)
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

        xla_client.register_custom_call_target("cpu_permanent", capsule, "cpu")

        _method_cache.append("cpu_permanent")

    custom_call_op = mlir.custom_call(
        "cpu_permanent",
        result_types=[ir.RankedTensorType.get((1,), matrix.type.element_type)],
        operands=[matrix, rows, cols, mlir.ir_constant(matrix.type.shape[0])],
        operand_layouts=[(1, 0), (0,), (0,), ()],
        result_layouts=[(0,)],
    )

    return custom_call_op.results


def _abstract_eval(matrix, rows, cols):
    return core.ShapedArray((1,), matrix.dtype)


def _permanent_jvp(args, tangents):
    matrix, rows, cols = args
    tangent = tangents[0]
    primal_out = _permanent_primitive.bind(matrix, rows, cols)

    return primal_out, tangent  # TODO: Finish it!


_permanent_primitive = jax.core.Primitive("permanent")

_permanent_primitive.def_impl(
    lambda matrix, rows, cols: np.array([
        _numba_permanent(matrix, np.array(rows), np.array(cols))
    ])
)

_permanent_primitive.def_abstract_eval(_abstract_eval)

mlir.register_lowering(_permanent_primitive, _permanent_lowering, platform="cpu")

ad.primitive_jvps[_permanent_primitive] = _permanent_jvp



def jax_permanent(matrix, rows, cols):
    rows_array = np.array(rows)
    cols_array = np.array(cols)

    return _permanent_primitive.bind(
        matrix, rows_array, cols_array
    )


if __name__ == "__main__":
    z = np.ones((3, 3)) * 5 + 1j * np.ones((3, 3))
    z4 = np.ones((4, 4)) * 5 + 1j * np.ones((4, 4))

    rows = (1, 2, 1)
    cols = (3, 0, 1)

    print(jax_permanent(z, rows, cols)[0])

    jitted_jax_permanent = jit(jax_permanent, static_argnums=(1, 2))

    import time

    start_time = time.time()

    print(jitted_jax_permanent(z, rows, cols)[0])
    print(time.time() - start_time); start_time = time.time()

    print(jitted_jax_permanent(z, rows, (2, 1, 1))[0])
    print(time.time() - start_time); start_time = time.time()


    print("---------------")
    print(jax_permanent(z4, (0, 2, 1, 3), (3, 1, 2, 0))[0])
    print(jitted_jax_permanent(z4, (0, 2, 1, 3), (3, 1, 2, 0))[0])
    print(time.time() - start_time); start_time = time.time()

    def func(matrix):
        return jitted_jax_permanent(matrix, rows, cols)

    print(jvp(func, (z,), (z,)))

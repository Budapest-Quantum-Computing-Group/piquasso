import numpy as np

import numba
from numba.types import void, voidptr, CPointer

import jax
from jax import core, jit
from jax.interpreters import mlir
import jaxlib.mlir.ir as ir
from jax.lib import xla_client

from piquasso._math.permanent import permanent as _numba_permanent

import ctypes


def _permanent_lowering(ctx, matrix, rows, cols):
    matrix_shape = ctx.avals_in[0].shape
    matrix_dtype = ctx.avals_in[0].dtype

    occ_num_shape = ctx.avals_in[1].shape
    occ_num_dtype = ctx.avals_in[1].dtype

    @numba.cfunc(void(voidptr, CPointer(voidptr)))
    def xla_cpu_custom_call_target(output_ptr, input_ptrs):
        output = numba.carray(output_ptr, (1,), dtype=matrix_dtype)

        matrix_carr = numba.carray(input_ptrs[0], matrix_shape, dtype=matrix_dtype)
        rows_carr = numba.carray(input_ptrs[1], occ_num_shape, dtype=occ_num_dtype)
        cols_carr = numba.carray(input_ptrs[2], occ_num_shape, dtype=occ_num_dtype)

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

    custom_call_op = mlir.custom_call(
        "cpu_permanent",
        result_types=[ir.RankedTensorType.get((1,), matrix.type.element_type)],
        operands=[matrix, rows, cols],
        operand_layouts=[(1, 0), (0,), (0,)],
        result_layouts=[(0,)],
    )

    return custom_call_op.results


def _abstract_eval(matrix, rows, cols):
    return core.ShapedArray((1,), matrix.dtype)


_permanent_primitive = jax.core.Primitive("permanent")

_permanent_primitive.def_impl(lambda *args: [_numba_permanent(*args)])
_permanent_primitive.def_abstract_eval(_abstract_eval)

mlir.register_lowering(_permanent_primitive, _permanent_lowering, platform="cpu")

jax_permanent = lambda *args: _permanent_primitive.bind(*args)[0]


if __name__ == "__main__":
    z = np.ones((3, 3)) * 5 + 1j * np.ones((3, 3))

    rows = np.array([1, 2, 1])
    cols = np.array([3, 0, 1])

    print(jax_permanent(z, rows, cols))

    print(jit(jax_permanent)(z, rows, cols))

    #print(grad(jax_permanent)(z, rows, cols))

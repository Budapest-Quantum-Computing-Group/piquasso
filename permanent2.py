#
# Copyright 2021-2024 Budapest Quantum Computing Group
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


import ctypes


from jax import core
from jaxlib.hlo_helpers import custom_call
from jax.lib import xla_client
from jax.interpreters import mlir

import numba as nb


#@nb.njit
def numba_permanent(ret, matrix, rows, cols):
    ret[0] = np.sum(matrix)


_permanent_prim = core.Primitive("permanent")
_permanent_prim.multiple_results = True


def encapsulate(address):
    PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
    capsule = PyCapsule_New(
        address,
        b"xla._CUSTOM_CALL_TARGET",
        PyCapsule_Destructor(0),
    )

    return capsule


xla_client.register_custom_call_target(
    "cpu_permanent", encapsulate(hex(id(numba_permanent))), platform="cpu"
)


def permanent(ret, matrix, rows, cols):
    return _permanent_prim.bind(ret, matrix, rows, cols)


def _permanent_impl(ret, matrix, rows, cols):
    numba_permanent(ret, matrix, rows, cols)


def _permanent_abstract_eval(ret_s, matrix_s, rows_s, cols_s):
    assert rows_s.shape == cols_s.shape

    return [core.ShapedArray((1,), matrix_s.dtype)]


def _permanent_lowering(ctx, ret, matrix, rows, cols, *, platform="cpu"):
    assert rows.type == cols.type

    return custom_call(
        "cpu_permanent",
        result_types=[mlir.ir.RankedTensorType.get((1,), matrix.type.element_type)],
        operands=[ret, matrix, rows, cols],
    ).results


_permanent_prim.def_impl(_permanent_impl)
_permanent_prim.def_abstract_eval(_permanent_abstract_eval)

from functools import partial


mlir.register_lowering(_permanent_prim, partial(_permanent_lowering, platform="cpu"), platform="cpu")


if __name__ == "__main__":
    import numpy as np

    matrix = np.array([[1, 2], [3, 4]], dtype=complex)

    rows = np.array([1, 2])
    cols = np.array([2, 1])

    ret = np.array([0.0])

    print(permanent(ret, matrix, rows, cols))

    from jax import jit

    jitted_permanent = jit(permanent)

    print(jitted_permanent(ret, matrix, rows, cols))

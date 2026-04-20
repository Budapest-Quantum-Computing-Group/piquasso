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
#
# Portions of this file are based on work by Bence Soóki-Tóth, used with
# permission and originally made available under the MIT License.
#
# Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
# in photonic quantum computing simulations", Eötvös Loránd University, 2025.

import warnings
from functools import partial

import jax
import jax.numpy as jnp

from ._perm_boost_core import registrations as _registrations

jax.config.update("jax_enable_x64", True)

try:
    _jax_ffi = jax.ffi
    _ffi_call_new_api = True   # jax.ffi era: ffi_call(name, out)(args)
except AttributeError:
    import jax.extend.ffi as _jax_ffi
    _ffi_call_new_api = False  # jax.extend.ffi era: ffi_call(name, out, args)


def _ffi_call(target_name, out_type, *args):
    """Compatibility shim for the ffi_call API change between JAX versions.

    jax.extend.ffi (< 0.6): ffi_call(name, out_type, *args) → result
    jax.ffi        (>= 0.6): ffi_call(name, out_type)(*args) → result

    vmap_method="sequential" is required on JAX >= 0.10 when the function is
    traced under vmap (e.g. via jax.jacobian); without it JAX raises
    NotImplementedError at trace time.
    """
    if _ffi_call_new_api:
        return _jax_ffi.ffi_call(target_name, out_type, vmap_method="sequential")(*args)
    return _jax_ffi.ffi_call(target_name, out_type, *args)

for _name, _target in _registrations().items():
    _jax_ffi.register_ffi_target(_name, _target)

_gpu = False

try:
    from . import _perm_boost_gpu_ops as _gpu_ops
    _gpu_targets = _gpu_ops.registrations()
    for _name, _target in _gpu_targets.items():
        _jax_ffi.register_ffi_target(_name, _target, platform="CUDA")
        _gpu = True
except (ImportError, AttributeError) as _exc:
    warnings.warn(f"GPU support initialization failed: {_exc}")
    _gpu = False


@partial(jax.custom_vjp)
def perm(A, rows, cols):
    if rows.ndim != 1 or cols.ndim != 1:
        raise ValueError("perm: rows and cols must be 1D arrays")
    if rows.dtype not in (jnp.uint32, jnp.uint64) or cols.dtype not in (
        jnp.uint32,
        jnp.uint64,
    ):
        raise ValueError("perm: rows and cols must be uint32 or uint64")

    total_in = int(rows.sum())
    total_out = int(cols.sum())

    if total_in != total_out:
        raise ValueError(
            f"perm: sum(rows)={total_in} must equal sum(cols)={total_out}"
        )
    if A.dtype != jnp.complex128:
        raise ValueError("perm: A.dtype must be complex128")
    if total_in == 0 and total_out == 0:
        return 1.0

    out_type = jax.ShapeDtypeStruct((), A.dtype)

    def impl(target_name):
        return lambda: _ffi_call(target_name, out_type, A, rows, cols)

    return jax.lax.platform_dependent(cpu=impl("perm"), cuda=impl("dperm"))


def _perm_fwd(A, rows, cols):
    out_type = (jax.ShapeDtypeStruct((), A.dtype), jax.ShapeDtypeStruct((), A.dtype))

    def impl(target_name):
        return lambda: _ffi_call(target_name, out_type, A, rows, cols)

    y, res = jax.lax.platform_dependent(
        cpu=impl("perm_fwd"), cuda=impl("dperm_fwd")
    )
    return y, (res, A, rows, cols)


def _perm_bwd(res, ct):
    res, A, rows, cols = res

    def impl(target_name):
        return lambda: (
            _ffi_call(
                target_name,
                jax.ShapeDtypeStruct(A.shape, A.dtype),
                res, A, rows, cols, ct,
            ),
            None,
            None,
        )

    return jax.lax.platform_dependent(
        cpu=impl("perm_bwd"), cuda=impl("dperm_bwd")
    )


perm.defvjp(_perm_fwd, _perm_bwd)

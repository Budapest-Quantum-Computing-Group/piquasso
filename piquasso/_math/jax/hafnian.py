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

import jax

import jax.numpy as jnp

from piquasso._math.linalg import reduce_

from .utils import factorial, eig


def loop_hafnian_with_reduction(matrix, diagonal, reduce_on):
    """JAX implementation of the loop hafnian.

    This functions implement the loop hafnian in JAX with Glynn-type iterations, because
    this way the size of the matrix stays constant, which is required by JAX. Also, the
    speedup using repetitions cannot be utilized, since that would require dynamic for
    loops.

    This is just a placeholder implementation. To properly implement this, one would
    need to use JAX FFI.
    """
    reduced_matrix = _reduce_matrix_with_diagonal(matrix, diagonal, reduce_on)

    return _loop_hafnian(reduced_matrix)


def _reduce_matrix_with_diagonal(matrix, diagonal, reduce_on):
    reduced_diagonal = reduce_(diagonal, jnp.array(reduce_on))
    reduced_matrix = reduce_(matrix, jnp.array(reduce_on))

    for i in range(len(reduced_diagonal)):
        reduced_matrix = reduced_matrix.at[i, i].set(reduced_diagonal[i])

    return reduced_matrix


def _calculate_f(polynom_coefficients):
    dim_over_2 = len(polynom_coefficients)
    dim = 2 * dim_over_2

    data = jnp.zeros(shape=(2, dim_over_2 + 1), dtype=polynom_coefficients.dtype)

    data = data.at[0, 0].set(1.0)
    indices = jnp.arange(dim_over_2 + 1)

    def outer_body_fun(idx, args):
        data, p_aux1 = args
        factor = polynom_coefficients[idx - 1]

        p_aux0 = jax.lax.cond(idx % 2 == 1, lambda: 0, lambda: 1)
        p_aux1 = jax.lax.cond(idx % 2 == 1, lambda: 1, lambda: 0)

        data = data.at[p_aux1].set(data[p_aux0])

        size = dim // (2 * idx) + 1

        powfactors = jnp.power(factor, jnp.arange(dim_over_2 + 1)) / factorial(
            jnp.arange(dim_over_2 + 1, dtype=float)
        )

        def inner_body_fun(jdx, data):
            mask = jnp.where(idx * jdx <= indices, 1, 0)
            aux0_copy = jnp.copy(data[p_aux0])
            aux0_rolled = jnp.roll(aux0_copy, idx * jdx)

            mask2 = jnp.where(jdx < size, 1, 0)
            return data * (1 - mask2) + mask2 * data.at[p_aux1].set(
                data[p_aux1] + aux0_rolled * mask * powfactors[jdx]
            )

        data = jax.lax.fori_loop(1, dim_over_2 + 1, inner_body_fun, data)

        return (data, p_aux1)

    data, p_aux1 = jax.lax.fori_loop(1, dim_over_2 + 1, outer_body_fun, (data, 1))

    return data[p_aux1]


@jax.jit
def _loop_hafnian(A):
    if len(A) % 2 == 1:
        # NOTE: If the ijnput matrix `A` has an odd dimension, e.g. 7x7, then the matrix
        # should be padded to an even dimension, to e.g. 8x8.
        A = jnp.pad(A, pad_width=((1, 0), (1, 0)))
        A[0, 0] = 1.0

    degree = A.shape[0] // 2

    if degree == 0:
        return 1.0

    ret = 0.0

    size = 2**degree

    def body_fun(permutation_idx, ret):
        prefact = 1

        delta = jnp.empty(degree, dtype=int)

        def func_delta(i, args):
            j, delta = args
            delta = delta.at[i].set(2 * (j % 2) - 1)

            return j // 2, delta

        delta = jax.lax.fori_loop(0, degree, func_delta, (permutation_idx, delta))[1]

        prefact = jnp.prod(delta)

        O = jnp.zeros(shape=(degree, degree), dtype=A.dtype)
        diag_delta = jnp.diag(delta)
        X_delta = jnp.block([[O, diag_delta], [diag_delta, O]])

        polynom_coefficients = _get_loop_polynom_coefficients(A, X_delta, degree)

        ret += prefact * _calculate_f(polynom_coefficients)[degree]

        return ret

    ret = jax.lax.fori_loop(0, size, body_fun, ret)

    return ret / size


def _get_loop_polynom_coefficients(A, X_delta, degree):
    AX_delta = A @ X_delta

    eigenvalues, O = eig(AX_delta)

    Oinv = jnp.linalg.inv(O)

    v = jnp.diag(A)

    left = v @ X_delta @ O
    right = Oinv @ v.T

    def func(power):
        diags = jnp.diag(jnp.power(eigenvalues, power - 1))
        powertrace = jnp.sum(jnp.power(eigenvalues, power))
        return (powertrace / power + (left @ diags @ right)) / 2.0

    return jax.lax.map(func, jnp.arange(1, degree + 1))

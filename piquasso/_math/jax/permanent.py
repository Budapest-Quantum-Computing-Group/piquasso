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

from jax import jit, lax

import jax.numpy as jnp

from piquasso._math.linalg import assym_reduce


def permanent_with_reduction(matrix, rows, cols):
    """JAX implementation of the permanent, using the Glynn formula with Gray codes.

    Note:
        Unfortunately, JAX (or rather XLA) could not JIT-compile creating arrays
        whose shapes depend on the values of the input, so the reduction of the
        interferometer is needed to be made in advance.
    """

    reduced_matrix = assym_reduce(matrix, rows, cols)

    return permanent(reduced_matrix)


@jit
def permanent(M):
    n = M.shape[0]

    binary_powers = 2 ** jnp.arange(n)

    N = 2 ** (n - 1)

    def body_fun(index, val):
        sum_, sign, old_gray_code, row_sum = val

        sum_ += sign * jnp.prod(row_sum)

        new_gray_code = index ^ (index // 2)
        grey_diff = old_gray_code ^ new_gray_code
        matrix_row_index = jnp.where(binary_powers == grey_diff, size=1)[0][0]

        diff = lax.cond(
            old_gray_code > new_gray_code,
            lambda: 1,
            lambda: -1,
        )

        row_sum += M[matrix_row_index] * 2 * diff

        sign = -sign
        old_gray_code = new_gray_code

        return sum_, sign, old_gray_code, row_sum

    sum_ = 0.0
    sign = +1
    old_gray_code = 0
    row_sum = jnp.sum(M, axis=0)

    sum_ = lax.fori_loop(
        lower=1,
        upper=N + 1,
        body_fun=body_fun,
        init_val=(sum_, sign, old_gray_code, row_sum),
    )[0]

    return sum_ / N

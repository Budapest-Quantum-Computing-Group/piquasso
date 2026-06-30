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

import numpy as np


def multiply_by_linear_truncated(
    polynomial,
    constant,
    linear_coefficients,
    out,
):
    """Multiply a truncated multivariate polynomial by a linear polynomial.

    Computes

        (constant + sum_j linear_coefficients[j] * x_j) * polynomial

    while keeping only coefficients within the existing polynomial shape.
    """
    out[...] = constant * polynomial

    for axis, coefficient in enumerate(linear_coefficients):
        np.moveaxis(out, axis, 0)[1:] += (
            coefficient * np.moveaxis(polynomial, axis, 0)[:-1]
        )

    return out

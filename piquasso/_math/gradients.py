#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from typing import Callable

from piquasso.api.calculator import BaseCalculator


def create_single_mode_displacement_gradient(
    r: float,
    phi: float,
    cutoff: int,
    transformation: np.ndarray,
    calculator: BaseCalculator,
) -> Callable:
    def grad(upstream):
        np = calculator.fallback_np
        tf = calculator._tf

        r_ = r.numpy()
        phi_ = phi.numpy()

        r_grad = np.zeros((cutoff,) * 2, dtype=complex)
        phi_grad = np.zeros((cutoff,) * 2, dtype=complex)
        # NOTE: This algorithm deliberately overindexes the gate matrix.
        for row in range(cutoff):
            for col in range(cutoff):
                r_grad[row, col] = (
                    -r_ * transformation[row, col]
                    + np.exp(1j * phi_) * np.sqrt(row) * transformation[row - 1, col]
                    - np.exp(-1j * phi_) * np.sqrt(col) * transformation[row, col - 1]
                )
                phi_grad[row, col] = (
                    r_
                    * 1j
                    * (
                        np.sqrt(row) * np.exp(1j * phi_) * transformation[row - 1, col]
                        + np.sqrt(col)
                        * np.exp(-1j * phi_)
                        * transformation[row, col - 1]
                    )
                )
        r_grad_sum = tf.math.real(tf.reduce_sum(upstream * r_grad))
        phi_grad_sum = tf.math.real(tf.reduce_sum(upstream * phi_grad))
        return (r_grad_sum, phi_grad_sum)

    return grad
